use espeak_phonemizer::text_to_phonemes;
use libtashkeel_base::do_tashkeel;
use ndarray::Axis;
use ndarray::{Array, Array1, Array2, ArrayView, CowArray, Dim, IxDynImpl};
use ort::{tensor::OrtOwnedTensor, Environment, GraphOptimizationLevel, SessionBuilder, Value};
use sonata_core::{
    Phonemes, SonataError, SonataModel, SonataResult, AudioInfo, SonataAudioResult,
    Audio, AudioSamples,
};
use serde::Deserialize;
use std::any::Any;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

const MIN_CHUNK_SIZE: usize = 100;
#[allow(dead_code)]
const FADE_SECS: f64 = 0.002;
const BOS: char = '^';
const EOS: char = '$';
const PAD: char = '_';

#[inline(always)]
fn reversed_mapping<K, V>(input: &HashMap<K, V>) -> HashMap<V, K>
where
    K: ToOwned<Owned = K>,
    V: ToOwned<Owned = V> + std::hash::Hash + std::cmp::Eq,
{
    HashMap::from_iter(input.iter().map(|(k, v)| (v.to_owned(), k.to_owned())))
}

fn load_model_config(config_path: &Path) -> SonataResult<(ModelConfig, PiperSynthesisConfig)> {
    let file = match File::open(config_path) {
        Ok(file) => file,
        Err(why) => {
            return Err(SonataError::FailedToLoadResource(format!(
                "Faild to load model config: `{}`. Caused by: `{}`",
                config_path.display(),
                why
            )))
        }
    };
    let model_config: ModelConfig = match serde_json::from_reader(file) {
        Ok(config) => config,
        Err(why) => {
            return Err(SonataError::FailedToLoadResource(format!(
                "Faild to parse model config from file: `{}`. Caused by: `{}`",
                config_path.display(),
                why
            )))
        }
    };
    let synth_config = PiperSynthesisConfig {
        speaker: None,
        noise_scale: model_config.inference.noise_scale,
        length_scale: model_config.inference.length_scale,
        noise_w: model_config.inference.noise_w,
    };
    Ok((model_config, synth_config))
}

fn create_tashkeel_engine(
    config: &ModelConfig,
) -> SonataResult<Option<libtashkeel_base::DynamicInferenceEngine>> {
    if config.espeak.voice == "ar" {
        match libtashkeel_base::create_inference_engine(None) {
            Ok(engine) => Ok(Some(engine)),
            Err(msg) => Err(SonataError::OperationError(format!(
                "Failed to create inference engine for libtashkeel. {}",
                msg
            ))),
        }
    } else {
        Ok(None)
    }
}

fn create_inference_session(
    model_path: &Path,
    ort_env: &'static Arc<Environment>,
) -> Result<ort::Session, ort::OrtError> {
    SessionBuilder::new(ort_env)?
        .with_optimization_level(GraphOptimizationLevel::Disable)?
        .with_memory_pattern(false)?
        .with_parallel_execution(false)?
        .with_model_from_file(model_path)
}

pub fn from_config_path(
    config_path: &Path,
    ort_env: &'static Arc<Environment>,
) -> SonataResult<Arc<dyn SonataModel + Send + Sync>> {
    let (config, synth_config) = load_model_config(config_path)?;
    if config.streaming.unwrap_or_default() {
        Ok(Arc::new(VitsStreamingModel::from_config(
            config,
            synth_config,
            &config_path.with_file_name("encoder.onnx"),
            &config_path.with_file_name("decoder.onnx"),
            ort_env,
        )?))
    } else {
        let Some(onnx_filename) = config_path.file_stem() else {
            return Err(SonataError::OperationError(format!(
                "Invalid config filename format `{}`",
                config_path.display()
            )));
        };
        Ok(Arc::new(VitsModel::from_config(
            config,
            synth_config,
            &config_path.with_file_name(onnx_filename),
            ort_env,
        )?))
    }
}

#[derive(Deserialize, Default)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub quality: Option<String>,
}

#[derive(Deserialize, Default)]
pub struct ESpeakConfig {
    voice: String,
}

#[derive(Deserialize, Default, Clone)]
pub struct InferenceConfig {
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
}

#[derive(Clone, Deserialize, Default)]
pub struct Language {
    code: String,
    #[allow(dead_code)]
    family: Option<String>,
    #[allow(dead_code)]
    region: Option<String>,
    #[allow(dead_code)]
    name_native: Option<String>,
    #[allow(dead_code)]
    name_english: Option<String>,
}

#[derive(Deserialize, Default)]
pub struct ModelConfig {
    pub key: Option<String>,
    pub language: Option<Language>,
    pub audio: AudioConfig,
    pub num_speakers: u32,
    pub speaker_id_map: HashMap<String, i64>,
    streaming: Option<bool>,
    espeak: ESpeakConfig,
    inference: InferenceConfig,
    #[allow(dead_code)]
    num_symbols: u32,
    #[allow(dead_code)]
    phoneme_map: HashMap<i64, char>,
    phoneme_id_map: HashMap<char, Vec<i64>>,
}

#[derive(Debug, Clone, Default)]
pub struct PiperSynthesisConfig {
    pub speaker: Option<i64>,
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_w: f32,
}

trait VitsModelCommons {
    fn get_synth_config(&self) -> &RwLock<PiperSynthesisConfig>;
    fn get_config(&self) -> &ModelConfig;
    fn get_speaker_map(&self) -> &HashMap<i64, String>;
    fn get_tashkeel_engine(&self) -> Option<&libtashkeel_base::DynamicInferenceEngine>;
    fn get_meta_ids(&self) -> (i64, i64, i64) {
        let config = self.get_config();
        let pad_id = *config.phoneme_id_map.get(&PAD).unwrap().first().unwrap();
        let bos_id = *config.phoneme_id_map.get(&BOS).unwrap().first().unwrap();
        let eos_id = *config.phoneme_id_map.get(&EOS).unwrap().first().unwrap();
        (pad_id, bos_id, eos_id)
    }
    fn language(&self) -> Option<String> {
        self.get_config()
            .language
            .as_ref()
            .map(|lang| lang.code.clone())
    }
    fn get_properties(&self) -> HashMap<String, String> {
        HashMap::from([(
            "quality".to_string(),
            self.get_config()
                .audio
                .quality
                .clone()
                .unwrap_or("unknown".to_string()),
        )])
    }
    fn factory_synthesis_config(&self) -> PiperSynthesisConfig {
        let config = self.get_config();

        let speaker = if config.num_speakers > 0 {
            Some(0)
        } else {
            None
        };
        PiperSynthesisConfig {
            speaker,
            length_scale: config.inference.length_scale,
            noise_scale: config.inference.noise_scale,
            noise_w: config.inference.noise_w,
        }
    }
    fn speakers(&self) -> SonataResult<HashMap<i64, String>> {
        Ok(self.get_speaker_map().clone())
    }
    fn _do_set_default_synth_config(&self, new_config: &PiperSynthesisConfig) -> SonataResult<()> {
        let mut synth_config = self.get_synth_config().write().unwrap();
        synth_config.length_scale = new_config.length_scale;
        synth_config.noise_scale = new_config.noise_scale;
        synth_config.noise_w = new_config.noise_w;
        if let Some(sid) = new_config.speaker {
            if self.get_speaker_map().contains_key(&sid) {
                synth_config.speaker = Some(sid);
            } else {
                return Err(SonataError::OperationError(format!(
                    "No speaker was found with the given id `{}`",
                    sid
                )));
            }
        }
        Ok(())
    }
    fn phonemes_to_input_ids(
        &self,
        phonemes: &str,
        pad_id: i64,
        bos_id: i64,
        eos_id: i64,
    ) -> Vec<i64> {
        let config = self.get_config();
        let mut phoneme_ids: Vec<i64> = Vec::with_capacity((phonemes.len() + 1) * 2);
        phoneme_ids.push(bos_id);
        for phoneme in phonemes.chars() {
            if let Some(id) = config.phoneme_id_map.get(&phoneme) {
                phoneme_ids.push(*id.first().unwrap());
                phoneme_ids.push(pad_id);
            }
        }
        phoneme_ids.push(eos_id);
        phoneme_ids
    }
    fn do_phonemize_text(&self, text: &str) -> SonataResult<Phonemes> {
        let config = self.get_config();
        let text = if config.espeak.voice == "ar" {
            let diacritized = self.diacritize_text(text)?;
            Cow::from(diacritized)
        } else {
            Cow::from(text)
        };
        let phonemes = match text_to_phonemes(&text, &config.espeak.voice, None, true, false) {
            Ok(ph) => ph,
            Err(e) => {
                return Err(SonataError::PhonemizationError(format!(
                    "Failed to phonemize given text using espeak-ng. Error: {}",
                    e
                )))
            }
        };
        Ok(phonemes.into())
    }
    fn diacritize_text(&self, text: &str) -> SonataResult<String> {
        let diacritized_text = match do_tashkeel(self.get_tashkeel_engine().unwrap(), text, None) {
            Ok(d_text) => d_text,
            Err(msg) => {
                return Err(SonataError::OperationError(format!(
                    "Failed to diacritize text using  libtashkeel. {}",
                    msg
                )))
            }
        };
        Ok(diacritized_text)
    }
    fn get_audio_output_info(&self) -> SonataResult<AudioInfo> {
        Ok(AudioInfo {
            sample_rate: self.get_config().audio.sample_rate as usize,
            num_channels: 1usize,
            sample_width: 2usize,
        })
    }
}

pub struct VitsModel {
    synth_config: RwLock<PiperSynthesisConfig>,
    config: ModelConfig,
    speaker_map: HashMap<i64, String>,
    session: ort::Session,
    tashkeel_engine: Option<libtashkeel_base::DynamicInferenceEngine>,
}

impl VitsModel {
    pub fn new(
        config_path: PathBuf,
        onnx_path: &Path,
        ort_env: &'static Arc<Environment>,
    ) -> SonataResult<Self> {
        match load_model_config(&config_path) {
            Ok((config, synth_config)) => {
                Self::from_config(config, synth_config, onnx_path, ort_env)
            }
            Err(error) => Err(error),
        }
    }
    fn from_config(
        config: ModelConfig,
        synth_config: PiperSynthesisConfig,
        onnx_path: &Path,
        ort_env: &'static Arc<Environment>,
    ) -> SonataResult<Self> {
        let session = match create_inference_session(onnx_path, ort_env) {
            Ok(session) => session,
            Err(err) => {
                return Err(SonataError::OperationError(format!(
                    "Failed to initialize onnxruntime inference session: `{}`",
                    err
                )))
            }
        };
        let speaker_map = reversed_mapping(&config.speaker_id_map);
        let tashkeel_engine = if config.espeak.voice == "ar" {
            match libtashkeel_base::create_inference_engine(None) {
                Ok(engine) => Some(engine),
                Err(msg) => {
                    return Err(SonataError::OperationError(format!(
                        "Failed to create inference engine for libtashkeel. {}",
                        msg
                    )))
                }
            }
        } else {
            None
        };
        Ok(Self {
            synth_config: RwLock::new(synth_config),
            config,
            speaker_map,
            session,
            tashkeel_engine,
        })
    }
    fn infer_with_values(&self, input_phonemes: Vec<i64>) -> SonataAudioResult {
        let synth_config = self.synth_config.read().unwrap();

        let input_len = input_phonemes.len();
        let phoneme_inputs =
            CowArray::from(Array2::<i64>::from_shape_vec((1, input_len), input_phonemes).unwrap())
                .into_dyn();

        let input_lengths = CowArray::from(Array1::<i64>::from_iter([input_len as i64])).into_dyn();

        let scales = Array1::<f32>::from_iter([
            synth_config.noise_scale,
            synth_config.length_scale,
            synth_config.noise_w,
        ]);
        let scales = CowArray::from(scales).into_dyn();

        let speaker_id = if self.config.num_speakers > 1 {
            let sid = synth_config.speaker.unwrap_or(0);

            Some(CowArray::from(Array1::<i64>::from_iter([sid])).into_dyn())
        } else {
            None
        };

        let session = &self.session;
        let timer = std::time::Instant::now();
        let outputs: Vec<Value> = {
            let mut inputs = vec![
                Value::from_array(session.allocator(), &phoneme_inputs).unwrap(),
                Value::from_array(session.allocator(), &input_lengths).unwrap(),
                Value::from_array(session.allocator(), &scales).unwrap(),
            ];
            if let Some(ref sid_tensor) = speaker_id {
                inputs.push(Value::from_array(session.allocator(), sid_tensor).unwrap());
            }
            match session.run(inputs) {
                Ok(out) => out,
                Err(e) => {
                    return Err(SonataError::OperationError(format!(
                        "Failed to run model inference. Error: {}",
                        e
                    )))
                }
            }
        };
        let inference_ms = timer.elapsed().as_millis() as f32;

        let outputs: OrtOwnedTensor<f32, _> = match outputs[0].try_extract() {
            Ok(out) => out,
            Err(e) => {
                return Err(SonataError::OperationError(format!(
                    "Failed to run model inference. Error: {}",
                    e
                )))
            }
        };

        let audio = Vec::from(outputs.view().as_slice().unwrap());

        Ok(Audio::new(
            audio.into(),
            self.config.audio.sample_rate as usize,
            Some(inference_ms),
        ))
    }
    pub fn get_input_output_info(&self) -> SonataResult<Vec<String>> {
        Ok(self
            .session
            .inputs
            .iter()
            .map(|i| {
                let name = i.name.clone();
                let dim: Vec<String> = i
                    .dimensions
                    .iter()
                    .map(|o| o.unwrap_or(42).to_string())
                    .collect();
                let dt = i.input_type;
                format!("#name: {}#dims: {}#type: {:?}", name, dim.join(", "), dt)
            })
            .collect())
    }
}

impl VitsModelCommons for VitsModel {
    fn get_synth_config(&self) -> &RwLock<PiperSynthesisConfig> {
        &self.synth_config
    }
    fn get_config(&self) -> &ModelConfig {
        &self.config
    }
    fn get_speaker_map(&self) -> &HashMap<i64, String> {
        &self.speaker_map
    }
    fn get_tashkeel_engine(&self) -> Option<&libtashkeel_base::DynamicInferenceEngine> {
        self.tashkeel_engine.as_ref()
    }
}

impl SonataModel for VitsModel {
    fn phonemize_text(&self, text: &str) -> SonataResult<Phonemes> {
        self.do_phonemize_text(text)
    }

    fn speak_batch(&self, phoneme_batches: Vec<String>) -> SonataResult<Vec<Audio>> {
        let (pad_id, bos_id, eos_id) = self.get_meta_ids();
        let phoneme_batches = Vec::from_iter(
            phoneme_batches
                .into_iter()
                .map(|phonemes| self.phonemes_to_input_ids(&phonemes, pad_id, bos_id, eos_id)),
        );
        let mut retval = Vec::new();
        for phonemes in phoneme_batches.into_iter() {
            retval.push(self.infer_with_values(phonemes)?);
        }
        Ok(retval)
    }

    fn speak_one_sentence(&self, phonemes: String) -> SonataAudioResult {
        let (pad_id, bos_id, eos_id) = self.get_meta_ids();
        let phonemes = self.phonemes_to_input_ids(&phonemes, pad_id, bos_id, eos_id);
        self.infer_with_values(phonemes)
    }
    fn get_default_synthesis_config(&self) -> SonataResult<Box<dyn Any>> {
        Ok(Box::new(PiperSynthesisConfig {
            speaker: Some(0),
            noise_scale: self.config.inference.noise_scale,
            noise_w: self.config.inference.noise_w,
            length_scale: self.config.inference.length_scale,
        }))
    }
    fn get_fallback_synthesis_config(&self) -> SonataResult<Box<dyn Any>> {
        Ok(Box::new(self.synth_config.read().unwrap().clone()))
    }
    fn set_fallback_synthesis_config(&self, synthesis_config: &dyn Any) -> SonataResult<()> {
        match synthesis_config.downcast_ref::<PiperSynthesisConfig>() {
            Some(new_config) => self._do_set_default_synth_config(new_config),
            None => Err(SonataError::OperationError(
                "Invalid configuration for Vits Model".to_string(),
            )),
        }
    }
    fn get_language(&self) -> SonataResult<Option<String>> {
        Ok(self.language())
    }
    fn get_speakers(&self) -> SonataResult<Option<&HashMap<i64, String>>> {
        Ok(Some(self.get_speaker_map()))
    }
    fn speaker_name_to_id(&self, name: &str) -> SonataResult<Option<i64>> {
        Ok(self.config.speaker_id_map.get(name).copied())
    }
    fn properties(&self) -> SonataResult<HashMap<String, String>> {
        Ok(self.get_properties())
    }
    fn audio_output_info(&self) -> SonataResult<AudioInfo> {
        self.get_audio_output_info()
    }
}

pub struct VitsStreamingModel {
    synth_config: RwLock<PiperSynthesisConfig>,
    config: ModelConfig,
    speaker_map: HashMap<i64, String>,
    encoder_model: ort::Session,
    decoder_model: Arc<ort::Session>,
    tashkeel_engine: Option<libtashkeel_base::DynamicInferenceEngine>,
}

impl VitsStreamingModel {
    fn from_config(
        config: ModelConfig,
        synth_config: PiperSynthesisConfig,
        encoder_path: &Path,
        decoder_path: &Path,
        ort_env: &'static Arc<Environment>,
    ) -> SonataResult<Self> {
        let encoder_model = match create_inference_session(encoder_path, ort_env) {
            Ok(model) => model,
            Err(err) => {
                return Err(SonataError::OperationError(format!(
                    "Failed to initialize onnxruntime inference session: `{}`",
                    err
                )))
            }
        };
        let decoder_model = match create_inference_session(decoder_path, ort_env) {
            Ok(model) => Arc::new(model),
            Err(err) => {
                return Err(SonataError::OperationError(format!(
                    "Failed to initialize onnxruntime inference session: `{}`",
                    err
                )))
            }
        };
        let speaker_map = reversed_mapping(&config.speaker_id_map);
        let tashkeel_engine = create_tashkeel_engine(&config)?;
        Ok(Self {
            synth_config: RwLock::new(synth_config),
            config,
            speaker_map,
            encoder_model,
            decoder_model,
            tashkeel_engine,
        })
    }

    fn infer_with_values(&self, input_phonemes: Vec<i64>) -> SonataAudioResult {
        let timer = std::time::Instant::now();
        let encoder_output = self.infer_encoder(input_phonemes)?;
        let audio = encoder_output.infer_decoder(self.decoder_model.as_ref())?;
        let inference_ms = timer.elapsed().as_millis() as f32;
        Ok(Audio::new(
            audio,
            self.config.audio.sample_rate as usize,
            Some(inference_ms),
        ))
    }
    fn infer_encoder(&self, input_phonemes: Vec<i64>) -> SonataResult<EncoderOutputs> {
        let synth_config = self.synth_config.read().unwrap();

        let input_len = input_phonemes.len();
        let phoneme_inputs =
            CowArray::from(Array2::<i64>::from_shape_vec((1, input_len), input_phonemes).unwrap())
                .into_dyn();

        let input_lengths = CowArray::from(Array1::<i64>::from_iter([input_len as i64])).into_dyn();

        let scales = Array1::<f32>::from_iter([
            synth_config.noise_scale,
            synth_config.length_scale,
            synth_config.noise_w,
        ]);
        let scales = CowArray::from(scales).into_dyn();

        let speaker_id = if self.config.num_speakers > 1 {
            let sid = synth_config.speaker.unwrap_or(0);

            Some(CowArray::from(Array1::<i64>::from_iter([sid])).into_dyn())
        } else {
            None
        };

        let session = &self.encoder_model;
        {
            let mut inputs = vec![
                Value::from_array(session.allocator(), &phoneme_inputs).unwrap(),
                Value::from_array(session.allocator(), &input_lengths).unwrap(),
                Value::from_array(session.allocator(), &scales).unwrap(),
            ];
            if let Some(ref sid_tensor) = speaker_id {
                inputs.push(Value::from_array(session.allocator(), sid_tensor).unwrap());
            }
            match session.run(inputs) {
                Ok(ort_values) => EncoderOutputs::from_values(ort_values),
                Err(e) => Err(SonataError::OperationError(format!(
                    "Failed to run model inference. Error: {}",
                    e
                ))),
            }
        }
    }
}

impl VitsModelCommons for VitsStreamingModel {
    fn get_synth_config(&self) -> &RwLock<PiperSynthesisConfig> {
        &self.synth_config
    }
    fn get_config(&self) -> &ModelConfig {
        &self.config
    }
    fn get_speaker_map(&self) -> &HashMap<i64, String> {
        &self.speaker_map
    }
    fn get_tashkeel_engine(&self) -> Option<&libtashkeel_base::DynamicInferenceEngine> {
        self.tashkeel_engine.as_ref()
    }
}

impl SonataModel for VitsStreamingModel {
    fn phonemize_text(&self, text: &str) -> SonataResult<Phonemes> {
        self.do_phonemize_text(text)
    }

    fn speak_batch(&self, phoneme_batches: Vec<String>) -> SonataResult<Vec<Audio>> {
        let (pad_id, bos_id, eos_id) = self.get_meta_ids();
        let phoneme_batches = Vec::from_iter(
            phoneme_batches
                .into_iter()
                .map(|phonemes| self.phonemes_to_input_ids(&phonemes, pad_id, bos_id, eos_id)),
        );
        let mut retval = Vec::new();
        for phonemes in phoneme_batches.into_iter() {
            retval.push(self.infer_with_values(phonemes)?);
        }
        Ok(retval)
    }
    fn speak_one_sentence(&self, phonemes: String) -> SonataAudioResult {
        let (pad_id, bos_id, eos_id) = self.get_meta_ids();
        let phonemes = self.phonemes_to_input_ids(&phonemes, pad_id, bos_id, eos_id);
        self.infer_with_values(phonemes)
    }
    fn get_default_synthesis_config(&self) -> SonataResult<Box<dyn Any>> {
        Ok(Box::new(PiperSynthesisConfig {
            speaker: Some(0),
            noise_scale: self.config.inference.noise_scale,
            noise_w: self.config.inference.noise_w,
            length_scale: self.config.inference.length_scale,
        }))
    }
    fn get_fallback_synthesis_config(&self) -> SonataResult<Box<dyn Any>> {
        Ok(Box::new(self.synth_config.read().unwrap().clone()))
    }
    fn set_fallback_synthesis_config(&self, synthesis_config: &dyn Any) -> SonataResult<()> {
        match synthesis_config.downcast_ref::<PiperSynthesisConfig>() {
            Some(new_config) => self._do_set_default_synth_config(new_config),
            None => Err(SonataError::OperationError(
                "Invalid configuration for Vits Model".to_string(),
            )),
        }
    }
    fn get_language(&self) -> SonataResult<Option<String>> {
        Ok(self.language())
    }
    fn get_speakers(&self) -> SonataResult<Option<&HashMap<i64, String>>> {
        Ok(Some(self.get_speaker_map()))
    }
    fn speaker_name_to_id(&self, name: &str) -> SonataResult<Option<i64>> {
        Ok(self.config.speaker_id_map.get(name).copied())
    }
    fn properties(&self) -> SonataResult<HashMap<String, String>> {
        Ok(self.get_properties())
    }
    fn audio_output_info(&self) -> SonataResult<AudioInfo> {
        self.get_audio_output_info()
    }
    fn supports_streaming_output(&self) -> bool {
        true
    }
    fn stream_synthesis<'a>(
        &'a self,
        phonemes: String,
        chunk_size: usize,
        chunk_padding: usize,
    ) -> SonataResult<Box<dyn Iterator<Item = SonataResult<AudioSamples>> + Send + Sync + 'a>> {
        let (pad_id, bos_id, eos_id) = self.get_meta_ids();
        let phonemes = self.phonemes_to_input_ids(&phonemes, pad_id, bos_id, eos_id);
        let encoder_outputs = self.infer_encoder(phonemes)?;
        let streamer = Box::new(SpeechStreamer::new(
            Arc::clone(&self.decoder_model),
            encoder_outputs,
            self.config.audio.sample_rate as usize,
            chunk_size,
            chunk_padding,
        ));
        Ok(streamer)
    }
}

struct EncoderOutputs {
    z: Array<f32, Dim<IxDynImpl>>,
    y_mask: Array<f32, Dim<IxDynImpl>>,
    g: Array<f32, Dim<IxDynImpl>>,
}

impl EncoderOutputs {
    #[inline(always)]
    fn from_values(values: Vec<Value>) -> SonataResult<Self> {
        let z = {
            let z_t: OrtOwnedTensor<f32, _> = match values[0].try_extract() {
                Ok(out) => out,
                Err(e) => {
                    return Err(SonataError::OperationError(format!(
                        "Failed to run model inference. Error: {}",
                        e
                    )))
                }
            };
            z_t.view().clone().into_owned()
        };
        let y_mask = {
            let y_mask_t: OrtOwnedTensor<f32, _> = match values[1].try_extract() {
                Ok(out) => out,
                Err(e) => {
                    return Err(SonataError::OperationError(format!(
                        "Failed to run model inference. Error: {}",
                        e
                    )))
                }
            };
            y_mask_t.view().clone().into_owned()
        };
        let g = if values.len() == 3 {
            let g_t: OrtOwnedTensor<f32, _> = match values[2].try_extract() {
                Ok(out) => out,
                Err(e) => {
                    return Err(SonataError::OperationError(format!(
                        "Failed to run model inference. Error: {}",
                        e
                    )))
                }
            };
            g_t.view().clone().into_owned()
        } else {
            Array1::<f32>::from_iter([]).into_dyn()
        };
        Ok(Self { z, y_mask, g })
    }
    fn infer_decoder(&self, session: &ort::Session) -> SonataResult<AudioSamples> {
        let outputs: Vec<Value> = {
            let z_input = CowArray::from(self.z.view());
            let y_mask_input = CowArray::from(self.y_mask.view());
            let g_input = CowArray::from(self.g.view());
            let mut inputs = vec![
                Value::from_array(session.allocator(), &z_input).unwrap(),
                Value::from_array(session.allocator(), &y_mask_input).unwrap(),
            ];
            if !g_input.is_empty() {
                inputs.push(Value::from_array(session.allocator(), &g_input).unwrap())
            }
            match session.run(inputs) {
                Ok(out) => out,
                Err(e) => {
                    return Err(SonataError::OperationError(format!(
                        "Failed to run model inference. Error: {}",
                        e
                    )))
                }
            }
        };
        match outputs[0].try_extract() {
            Ok(out) => Ok(Vec::from(out.view().as_slice().unwrap()).into()),
            Err(e) => Err(SonataError::OperationError(format!(
                "Failed to run model inference. Error: {}",
                e
            ))),
        }
    }
}

struct SpeechStreamer {
    decoder_model: Arc<ort::Session>,
    encoder_outputs: EncoderOutputs,
    #[allow(dead_code)]
    sample_rate: f64,
    chunk_size: isize,
    chunk_padding: isize,
    num_frames: isize,
    chunk_enumerater: std::vec::IntoIter<usize>,
    one_shot: bool,
    last_chunk_suffix: AudioSamples,
}

impl SpeechStreamer {
    fn new(
        decoder_model: Arc<ort::Session>,
        encoder_outputs: EncoderOutputs,
        sample_rate: usize,
        chunk_size: usize,
        chunk_padding: usize,
    ) -> Self {
        let chunk_size = chunk_size.max(MIN_CHUNK_SIZE);
        let chunk_padding = chunk_padding.min(chunk_size / 2).min(1);
        let num_frames = encoder_outputs.z.shape()[2];
        let num_chunks = (num_frames as f32 / chunk_size as f32).ceil() as usize;
        let one_shot = num_frames <= (chunk_size + (chunk_padding * 3));
        Self {
            decoder_model,
            encoder_outputs,
            sample_rate: sample_rate as f64,
            chunk_size: chunk_size as isize,
            chunk_padding: chunk_padding as isize,
            num_frames: num_frames as isize,
            chunk_enumerater: Vec::from_iter(0..num_chunks).into_iter(),
            one_shot,
            last_chunk_suffix: Default::default(),
        }
    }
    fn synthesize_chunk(&mut self, chunk_idx: isize) -> SonataResult<AudioSamples> {
        let (start_index, start_padding) = if chunk_idx == 0 {
            (0, 0)
        } else {
            let start = (chunk_idx * self.chunk_size) - self.chunk_padding;
            (start, self.chunk_padding)
        };
        let chunk_end = ((chunk_idx + 1) * self.chunk_size) + self.chunk_padding;
        let (end_index, end_padding) = if chunk_end > self.num_frames {
            (None, None)
        } else if (self.num_frames - chunk_end) <= MIN_CHUNK_SIZE as isize {
            self.consume();
            (None, None)
        } else {
            (Some(chunk_end), Some(-self.chunk_padding))
        };
        let index = ndarray::Slice::new(start_index, end_index, 1);
        let session = &self.decoder_model;
        {
            let z_view = self.encoder_outputs.z.view();
            let y_mask_view = self.encoder_outputs.y_mask.view();
            let z_chunk = z_view.slice_axis(Axis(2), index).into_dyn();
            let y_mask_chunk = y_mask_view.slice_axis(Axis(2), index).into_dyn();
            let z_input = CowArray::from(z_chunk);
            let y_mask_input = CowArray::from(y_mask_chunk);
            let g_input = CowArray::from(self.encoder_outputs.g.view());
            let mut inputs = vec![
                Value::from_array(session.allocator(), &z_input).unwrap(),
                Value::from_array(session.allocator(), &y_mask_input).unwrap(),
            ];
            if !g_input.is_empty() {
                inputs.push(Value::from_array(session.allocator(), &g_input).unwrap())
            }
            match session.run(inputs) {
                Ok(outputs) => match outputs[0].try_extract() {
                    Ok(audio_t) => {
                        self.process_chunk_audio(audio_t.view().view(), start_padding, end_padding)
                    }
                    Err(e) => Err(SonataError::OperationError(format!(
                        "Failed to run model inference. Error: {}",
                        e
                    ))),
                },
                Err(e) => Err(SonataError::OperationError(format!(
                    "Failed to run model inference. Error: {}",
                    e
                ))),
            }
        }
    }
    #[inline(always)]
    fn process_chunk_audio(
        &mut self,
        audio_view: ArrayView<f32, Dim<IxDynImpl>>,
        start_padding: isize,
        end_padding: Option<isize>,
    ) -> SonataResult<AudioSamples> {
        let audio_idx = ndarray::Slice::new(start_padding, end_padding, 1);
        let mut audio_data = Vec::from(
            audio_view
                .slice_axis(Axis(2), audio_idx)
                .as_slice()
                .unwrap(),
        );
        const N_SFX_SAMPLES: usize = 16;
        let this_chunk_suffix = if audio_data.len() >= N_SFX_SAMPLES {
            let idx = audio_data.len() - N_SFX_SAMPLES..audio_data.len();
            Vec::from_iter(audio_data.drain(idx))
        } else {
            Default::default()
        };
        let mut audio =
            std::mem::replace(&mut self.last_chunk_suffix, this_chunk_suffix.into());
        audio.merge(AudioSamples::from(audio_data));
        Ok(audio)
    }
    fn consume(&mut self) {
        self.chunk_enumerater.find(|_| false);
    }
}

impl Iterator for SpeechStreamer {
    type Item = SonataResult<AudioSamples>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk_idx = self.chunk_enumerater.next()?;
        if self.one_shot {
            self.consume();
            Some(
                self.encoder_outputs
                    .infer_decoder(self.decoder_model.as_ref()),
            )
        } else {
            Some(self.synthesize_chunk(chunk_idx as isize))
        }
    }
}
