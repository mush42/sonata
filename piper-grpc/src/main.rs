use once_cell::sync::OnceCell;
use pgrpc::piper_grpc_server::{PiperGrpc, PiperGrpcServer};
use piper_core::{PiperError, PiperModel, PiperResult};
use piper_synth::{AudioOutputConfig, PiperSpeechStreamBatched, PiperSpeechSynthesizer};
use piper_vits::VitsModel;
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use xxhash_rust::xxh3::xxh3_64;

type PiperGrpcResult<T> = Result<T, PiperGrpcError>;

const DEFAULT_PIPER_GRPC_SERVER_PORT: u16 = 49314;
const VOICE_ID_REDUCTION_FACTOR: u64 = 10000000000000;
static ORT_ENVIRONMENT: OnceCell<Arc<ort::Environment>> = OnceCell::new();

pub mod pgrpc {
    tonic::include_proto!("piper_grpc");
}

#[derive(Debug)]
enum PiperGrpcError {
    PiperError(PiperError),
    VoiceNotFound(String),
}

impl std::error::Error for PiperGrpcError {}

impl std::fmt::Display for PiperGrpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PiperGrpcError::PiperError(e) => e.fmt(f),
            PiperGrpcError::VoiceNotFound(msg) => write!(f, "{}", msg),
        }
    }
}

impl From<PiperError> for PiperGrpcError {
    fn from(other: PiperError) -> Self {
        Self::PiperError(other)
    }
}

impl From<PiperGrpcError> for Status {
    fn from(other: PiperGrpcError) -> Self {
        match other {
            PiperGrpcError::PiperError(piper_error) => match piper_error {
                PiperError::FailedToLoadResource(msg) | PiperError::PhonemizationError(msg) => {
                    Status::aborted(msg)
                }
                PiperError::OperationError(msg) => Status::unknown(msg),
            },
            PiperGrpcError::VoiceNotFound(msg) => Status::not_found(msg),
        }
    }
}

struct Voice {
    model: Arc<VitsModel>,
    synth: PiperSpeechSynthesizer,
}

impl Voice {
    fn new(vits: VitsModel) -> PiperResult<Self> {
        let model = Arc::new(vits);
        let model_clone = Arc::clone(&model);
        let synth = PiperSpeechSynthesizer::new(model_clone)?;
        Ok(Self { model, synth })
    }
}

struct PiperGrpcService(RwLock<HashMap<String, Voice>>);

impl PiperGrpcService {
    fn new() -> Self {
        Self(Default::default())
    }
    fn get_ort_environment() -> &'static Arc<ort::Environment> {
        ORT_ENVIRONMENT.get_or_init(|| {
            Arc::new(
                ort::Environment::builder()
                    .with_name("piper")
                    .with_execution_providers([ort::ExecutionProvider::CPU(Default::default())])
                    .build()
                    .unwrap(),
            )
        })
    }
    fn _load_vits_voice(
        &self,
        onnx_path: PathBuf,
        config_path: PathBuf,
    ) -> PiperGrpcResult<pgrpc::VoiceInfo> {
        let voice_id = if onnx_path.is_file() {
            let voice_path = onnx_path
                .canonicalize()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            (xxh3_64(voice_path.as_bytes()) / VOICE_ID_REDUCTION_FACTOR).to_string()
        } else {
            return Err(PiperGrpcError::VoiceNotFound(format!(
                "ONNX file does not exists: `{}`",
                onnx_path.display()
            )));
        };
        if let Some(voice) = (self.0.read().unwrap()).get(&voice_id) {
            return self._get_voice_info(voice_id, &voice.model);
        }
        let vits_model =
            VitsModel::new(config_path, onnx_path.clone(), Self::get_ort_environment())?;
        log::info!(
            "Loaded voice from: `{}`. Voice ID: {}",
            onnx_path.display(),
            voice_id
        );
        let voice_info = self._get_voice_info(voice_id.clone(), &vits_model)?;
        let voice = Voice::new(vits_model)?;
        (self.0.write().unwrap()).insert(voice_id, voice);
        Ok(voice_info)
    }
    fn _get_voice_info(
        &self,
        voice_id: String,
        vits_model: &VitsModel,
    ) -> PiperGrpcResult<pgrpc::VoiceInfo> {
        let wav_info = vits_model.wave_info()?;
        let speakers = vits_model.speakers()?;
        let language = vits_model.language();
        let quality = vits_model.quality().map(|q| match q.as_str() {
            "x_low" => pgrpc::Quality::XLow,
            "low" => pgrpc::Quality::Low,
            "medium" => pgrpc::Quality::Medium,
            "high" => pgrpc::Quality::High,
            _ => pgrpc::Quality::Unspecified,
        });
        let audio_info = pgrpc::AudioInfo {
            sample_rate: wav_info.sample_rate as u32,
            num_channels: wav_info.num_channels as u32,
            sample_width: wav_info.sample_width as u32,
        };
        let synth_options = {
            let default_synth_config = vits_model.default_synthesis_config();
            let speaker = default_synth_config.speaker.map(|(name, _idx)| name);
            pgrpc::SynthesisOptions {
                speaker,
                length_scale: Some(default_synth_config.length_scale),
                noise_scale: Some(default_synth_config.noise_scale),
                noise_w: Some(default_synth_config.noise_w),
            }
        };
        Ok(pgrpc::VoiceInfo {
            voice_id,
            synth_options: Some(synth_options),
            language,
            speakers,
            audio: Some(audio_info),
            quality: quality.map(|q| q.into()),
        })
    }
    fn _create_speech_synthesis_stream(
        &self,
        voice_id: &str,
        text: String,
        output_config: Option<AudioOutputConfig>,
        batch_size: Option<usize>,
    ) -> PiperGrpcResult<PiperSpeechStreamBatched> {
        match (self.0.read().unwrap()).get(voice_id) {
            Some(voice) => Ok(voice
                .synth
                .synthesize_batched(text, output_config, batch_size)?),
            None => Err(PiperGrpcError::VoiceNotFound(format!(
                "A voice with the key `{}` has not been loaded",
                voice_id
            ))),
        }
    }
    fn _get_synth_options_from_model(
        &self,
        model: &VitsModel,
    ) -> PiperGrpcResult<pgrpc::SynthesisOptions> {
        let speaker = match model.get_speaker() {
            Ok(speaker) => speaker,
            Err(_) => None,
        };
        Ok(pgrpc::SynthesisOptions {
            speaker,
            length_scale: Some(model.get_length_scale()?),
            noise_scale: Some(model.get_noise_scale()?),
            noise_w: Some(model.get_noise_w()?),
        })
    }
    fn _get_synth_options(&self, voice_id: &str) -> PiperGrpcResult<pgrpc::SynthesisOptions> {
        let voices = self.0.read().unwrap();
        let voice = match voices.get(voice_id) {
            Some(voice) => voice,
            None => {
                return Err(PiperGrpcError::VoiceNotFound(format!(
                    "A voice with the key `{}` has not been loaded",
                    voice_id
                )))
            }
        };
        self._get_synth_options_from_model(&voice.model)
    }
    fn _set_synth_options(
        &self,
        voice_id: &str,
        synth_opts: pgrpc::SynthesisOptions,
    ) -> PiperGrpcResult<pgrpc::SynthesisOptions> {
        let voices = self.0.read().unwrap();
        let voice = match voices.get(voice_id) {
            Some(voice) => voice,
            None => {
                return Err(PiperGrpcError::VoiceNotFound(format!(
                    "A voice with the key `{}` has not been loaded",
                    voice_id
                )))
            }
        };
        if let Some(speaker) = synth_opts.speaker {
            voice.model.set_speaker(speaker)?;
        }
        if let Some(length_scale) = synth_opts.length_scale {
            voice.model.set_length_scale(length_scale)?;
        }
        if let Some(noise_scale) = synth_opts.noise_scale {
            voice.model.set_noise_scale(noise_scale)?;
        }
        if let Some(noise_w) = synth_opts.noise_w {
            voice.model.set_noise_w(noise_w)?;
        }
        self._get_synth_options_from_model(&voice.model)
    }
}

#[tonic::async_trait]
impl PiperGrpc for PiperGrpcService {
    async fn get_piper_version(
        &self,
        _request: Request<pgrpc::Empty>,
    ) -> Result<Response<pgrpc::Version>, Status> {
        let version = pgrpc::Version {
            version: env!("CARGO_PKG_VERSION").into(),
        };
        return Ok(Response::new(version));
    }
    async fn load_voice(
        &self,
        _request: Request<pgrpc::VoicePath>,
    ) -> Result<Response<pgrpc::VoiceInfo>, Status> {
        let voice_path = _request.into_inner();
        let onnx_path = PathBuf::from(voice_path.onnx_path);
        let config_path = match voice_path.config_path {
            Some(pth) => PathBuf::from(pth),
            None => onnx_path.with_extension("onnx.json"),
        };
        let voice_info = self._load_vits_voice(onnx_path, config_path)?;
        Ok(Response::new(voice_info))
    }
    async fn get_voice_info(
        &self,
        _request: Request<pgrpc::VoiceIdentifier>,
    ) -> Result<Response<pgrpc::VoiceInfo>, Status> {
        let voice_id = _request.into_inner().voice_id;
        let voices = self.0.read().unwrap();
        let voice = match voices.get(&voice_id) {
            Some(voice) => voice,
            None => {
                return Err(PiperGrpcError::VoiceNotFound(format!(
                    "A voice with the key `{}` has not been loaded",
                    voice_id
                )))?
            }
        };
        let voice_info = self._get_voice_info(voice_id, &voice.model)?;
        Ok(Response::new(voice_info))
    }
    async fn get_synthesis_options(
        &self,
        _request: Request<pgrpc::VoiceIdentifier>,
    ) -> Result<Response<pgrpc::SynthesisOptions>, Status> {
        let voice_id = _request.into_inner().voice_id;
        let synth_opts = self._get_synth_options(&voice_id)?;
        Ok(Response::new(synth_opts))
    }
    async fn set_synthesis_options(
        &self,
        _request: Request<pgrpc::VoiceSynthesisOptions>,
    ) -> Result<Response<pgrpc::SynthesisOptions>, Status> {
        let req = _request.into_inner();
        let synth_opts = match req.synthesis_options {
            Some(opts) => opts,
            None => {
                let status = Status::invalid_argument("No synthesis options provided");
                return Err(status);
            }
        };
        let new_synth_opts = self._set_synth_options(&req.voice_id, synth_opts)?;
        let response = Response::new(new_synth_opts);
        Ok(response)
    }
    type SynthesizeUtteranceStream = ReceiverStream<Result<pgrpc::SynthesisResult, Status>>;
    async fn synthesize_utterance(
        &self,
        _request: Request<pgrpc::Utterance>,
    ) -> Result<Response<Self::SynthesizeUtteranceStream>, Status> {
        let req = _request.into_inner();
        let output_config = req.speech_args.map(|args| AudioOutputConfig {
            rate: args.rate.map(|i| i as u8),
            volume: args.volume.map(|i| i as u8),
            pitch: args.pitch.map(|i| i as u8),
            appended_silence_ms: args.appended_silence_ms,
        });
        let piper_stream =
            self._create_speech_synthesis_stream(&req.voice_id, req.text, output_config, None)?;
        let (tx, rx) = mpsc::channel(4);
        tokio::spawn(async move {
            for wav_result in piper_stream {
                let wav = match wav_result {
                    Ok(wav) => wav,
                    Err(e) => {
                        let err = Err(PiperGrpcError::from(e).into());
                        tx.send(err).await.ok();
                        // Stop this stream here. No retrys
                        return;
                    }
                };
                let synth_result = pgrpc::SynthesisResult {
                    wav_samples: wav.as_wave_bytes(),
                    rtf: wav.real_time_factor().unwrap_or_default(),
                };
                // We can do nothing about this error
                tx.send(Ok(synth_result)).await.ok();
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

fn setup_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().filter_or("PIPER_GRPC", "info"))
        .init();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    setup_logging();

    let port = std::env::var("PIPER_GRPC_SERVER_PORT")
        .map(|val| val.parse().unwrap_or(DEFAULT_PIPER_GRPC_SERVER_PORT))
        .unwrap_or(DEFAULT_PIPER_GRPC_SERVER_PORT);
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port);

    let service = PiperGrpcService::new();
    let server = PiperGrpcServer::new(service);

    log::info!("Starting Piper GRPC server at address: {}", addr);

    Server::builder().add_service(server).serve(addr).await?;

    Ok(())
}
