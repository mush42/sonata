use once_cell::sync::OnceCell;
use pgrpc::piper_grpc_server::{PiperGrpc, PiperGrpcServer};
use piper_core::{PiperError, PiperModel, PiperResult};
use piper_synth::{
    AudioOutputConfig, PiperSpeechStreamLazy, PiperSpeechSynthesizer
};
use piper_vits::VitsSynthesisConfig;
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

struct Voice(Arc<PiperSpeechSynthesizer>);

impl Voice {
    fn new(model: Arc<dyn PiperModel + Send + Sync>) -> PiperResult<Self> {
        let synth = Arc::new(PiperSpeechSynthesizer::new(model)?);
        Ok(Self(synth))
    }
    fn model_ref(&self) -> &dyn PiperModel {
        self.synth_ref()
    }
    fn synth_ref(&self) -> &PiperSpeechSynthesizer {
        self.0.as_ref()
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
    fn _load_vits_voice(&self, config_path: PathBuf) -> PiperGrpcResult<pgrpc::VoiceInfo> {
        let voice_id = if config_path.is_file() {
            let voice_path = config_path
                .canonicalize()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            (xxh3_64(voice_path.as_bytes()) / VOICE_ID_REDUCTION_FACTOR).to_string()
        } else {
            return Err(PiperGrpcError::VoiceNotFound(format!(
                "Config file does not exists: `{}`",
                config_path.display()
            )));
        };
        if let Some(voice) = (self.0.read().unwrap()).get(&voice_id) {
            return self._get_voice_info(voice_id, voice.model_ref());
        }
        let vits_model = piper_vits::from_config_path(&config_path, Self::get_ort_environment())?;
        log::info!(
            "Loaded Vits voice from: `{}`. Voice ID: {}",
            config_path.display(),
            voice_id
        );
        let voice = Voice::new(vits_model)?;
        let voice_info = self._get_voice_info(voice_id.clone(), voice.model_ref())?;
        (self.0.write().unwrap()).insert(voice_id, voice);
        Ok(voice_info)
    }
    fn _create_speech_synthesis_stream(
        &self,
        voice_id: &str,
        text: String,
        output_config: Option<AudioOutputConfig>,
    ) -> PiperGrpcResult<PiperSpeechStreamLazy> {
        match (self.0.read().unwrap()).get(voice_id) {
            Some(voice) => Ok(voice.synth_ref().synthesize_lazy(text, output_config)?),
            None => Err(PiperGrpcError::VoiceNotFound(format!(
                "A voice with the key `{}` has not been loaded",
                voice_id
            ))),
        }
    }
    fn _get_voice_info(
        &self,
        voice_id: String,
        model: &(impl PiperModel + ?Sized),
    ) -> PiperGrpcResult<pgrpc::VoiceInfo> {
        let wav_info = model.wave_info()?;
        let speakers = model.get_speakers()?;
        let language = model.get_language()?;
        let audio_info = pgrpc::AudioInfo {
            sample_rate: wav_info.sample_rate as u32,
            num_channels: wav_info.num_channels as u32,
            sample_width: wav_info.sample_width as u32,
        };
        let synth_options = {
            let config_cast = model
                .get_default_synthesis_config()?
                .downcast::<VitsSynthesisConfig>();
            let default_synth_config = match config_cast {
                Ok(synth_config) => synth_config,
                Err(_) => {
                    return Err(PiperError::OperationError(
                        "Invalid synthesis config for Vits model".to_string(),
                    )
                    .into())
                }
            };
            let speaker = match default_synth_config.speaker {
                Some(ref sid) => model.speaker_id_to_name(sid)?,
                None => Some("Default".to_string()),
            };
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
            speakers: speakers.cloned().unwrap_or_default(),
            audio: Some(audio_info),
            supports_streaming_output: Some(model.supports_streaming_output()),
            quality: None,
        })
    }
    fn _get_synth_options_from_model(
        &self,
        model: &(impl PiperModel + ?Sized),
    ) -> PiperGrpcResult<pgrpc::SynthesisOptions> {
        let synth_config = match model
            .get_fallback_synthesis_config()?
            .downcast::<VitsSynthesisConfig>()
        {
            Ok(synth_config) => synth_config,
            Err(_) => {
                return Err(PiperError::OperationError(
                    "Invalid synthesis config for Vits model".to_string(),
                )
                .into())
            }
        };
        let speaker = match synth_config.speaker {
            Some(ref sid) => model.speaker_id_to_name(sid)?,
            None => model.speaker_id_to_name(&0)?,
        };
        Ok(pgrpc::SynthesisOptions {
            speaker,
            length_scale: Some(synth_config.length_scale),
            noise_scale: Some(synth_config.noise_scale),
            noise_w: Some(synth_config.noise_w),
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
        self._get_synth_options_from_model(voice.model_ref())
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
        let model = voice.model_ref();
        let mut synth_config = match model
            .get_fallback_synthesis_config()?
            .downcast::<VitsSynthesisConfig>()
        {
            Ok(synth_config) => synth_config,
            Err(_) => {
                return Err(PiperError::OperationError(
                    "Could not set synthesis parameters ".to_string(),
                )
                .into())
            }
        };
        if let Some(sname) = synth_opts.speaker {
            if let Some(sid) = model.speaker_name_to_id(&sname)? {
                synth_config.speaker = Some(sid)
            }
        }
        if let Some(length_scale) = synth_opts.length_scale {
            synth_config.length_scale = length_scale;
        }
        if let Some(noise_scale) = synth_opts.noise_scale {
            synth_config.noise_scale = noise_scale;
        }
        if let Some(noise_w) = synth_opts.noise_w {
            synth_config.noise_w = noise_w;
        }
        model.set_fallback_synthesis_config(synth_config.as_ref())?;
        self._get_synth_options_from_model(model)
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
        let config_path = PathBuf::from(voice_path.config_path);
        let voice_info = self._load_vits_voice(config_path)?;
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
        let voice_info = self._get_voice_info(voice_id, voice.model_ref())?;
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
            self._create_speech_synthesis_stream(&req.voice_id, req.text, output_config)?;
        let (tx, rx) = mpsc::channel(512);
        tokio::task::spawn_blocking(move || {
            for wav_result in piper_stream {
                let wav = match wav_result {
                    Ok(wav) => wav,
                    Err(e) => {
                        let err = Err(PiperGrpcError::from(e).into());
                        tx.blocking_send(err).ok();
                        return;
                    }
                };
                let synth_result = pgrpc::SynthesisResult {
                    wav_samples: wav.as_wave_bytes(),
                    rtf: wav.real_time_factor().unwrap_or_default(),
                };
                if tx.blocking_send(Ok(synth_result)).is_err() {
                    return;
                }
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
    type SynthesizeUtteranceRealtimeStream = ReceiverStream<Result<pgrpc::WaveSamples, Status>>;
    async fn synthesize_utterance_realtime(
        &self,
        _request: Request<pgrpc::Utterance>,
    ) -> Result<Response<Self::SynthesizeUtteranceRealtimeStream>, Status> {
        let req = _request.into_inner();
        let output_config = req.speech_args.map(|args| AudioOutputConfig {
            rate: args.rate.map(|i| i as u8),
            volume: args.volume.map(|i| i as u8),
            pitch: args.pitch.map(|i| i as u8),
            appended_silence_ms: args.appended_silence_ms,
        });
        let voice_id = &req.voice_id;
        let voices = self.0.read().unwrap();
        let voice = match voices.get(voice_id) {
            Some(voice) => voice,
            None => {
                return Err(PiperGrpcError::VoiceNotFound(format!(
                    "A voice with the key `{}` has not been loaded",
                    voice_id
                )).into())
            }
        };
        let synth = Arc::clone(&voice.0);
        let (tx, rx) = mpsc::channel(512);
        tokio::task::spawn_blocking(move || {
            let stream_result = synth.synthesize_streamed(req.text, output_config, 100, 2);
            let realtime_speech_stream = match stream_result {
                Ok(stream) => stream,
                Err(e) => {
                    let err = Err(PiperGrpcError::from(e).into());
                    tx.blocking_send(err).ok();
                    return;
                }
            };
            for wav_result in realtime_speech_stream {
                let wav = match wav_result {
                    Ok(wav) => wav,
                    Err(e) => {
                        let err = Err(PiperGrpcError::from(e).into());
                        tx.blocking_send(err).ok();
                        return;
                    }
                };
                let synth_result = pgrpc::WaveSamples {
                    wav_samples: wav.as_wave_bytes(),
                };
                if tx.blocking_send(Ok(synth_result)).is_err() {
                    return;
                }
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
