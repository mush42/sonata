use espeak_phonemizer::{text_to_phonemes, ESpeakError, Phonemes};
use lazy_static::lazy_static;
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use once_cell::sync::OnceCell;
use ort::{
    tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtError, SessionBuilder,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

const MAX_WAV_VALUE: f32 = 32767.0;
const BOS: char = '^';
const EOS: char = '$';
const PAD: char = '_';

pub type PiperResult<T> = Result<T, PiperError>;
use PiperError::{
    FailedToLoadModel, FailedToLoadModelConfig, InferenceError, OperationError, PhonemizationError,
};

lazy_static! {
    static ref _ENVIRONMENT: Arc<ort::Environment> = Arc::new(
        Environment::builder()
            .with_name("piper")
            .with_execution_providers([ExecutionProvider::cpu()])
            .build()
            .unwrap()
    );
}

#[derive(Debug)]
pub enum PiperError {
    FailedToLoadModelConfig(String),
    FailedToLoadModel(String),
    PhonemizationError(String),
    InferenceError(String),
    OperationError(String),
}

impl Error for PiperError {}

impl fmt::Display for PiperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let err_message = match self {
            FailedToLoadModelConfig(filename) => {
                format!("Failed to load config from file: `{}`", filename)
            }
            FailedToLoadModel(filename) => {
                format!("Failed to load onnx model from file: `{}`", filename)
            }
            InferenceError(msg) => format!("Inference Error: `{}`", msg),
            PhonemizationError(msg) => msg.to_string(),
            OperationError(msg) => msg.to_string(),
        };
        write!(f, "{}", err_message)
    }
}

impl From<ESpeakError> for PiperError {
    fn from(other: ESpeakError) -> Self {
        PhonemizationError(other.0.clone())
    }
}

impl From<OrtError> for PiperError {
    fn from(error: OrtError) -> Self {
        InferenceError(format!(
            "Failed to run onnxruntime inference: `{}`",
            error.to_string()
        ))
    }
}

#[derive(Deserialize, Default)]
pub struct AudioConfig {
    pub sample_rate: u32,
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

#[derive(Deserialize, Default)]
pub struct ModelConfig {
    pub audio: AudioConfig,
    pub num_speakers: u32,
    pub speaker_id_map: HashMap<String, i64>,
    espeak: ESpeakConfig,
    inference: InferenceConfig,
    #[allow(dead_code)]
    num_symbols: u32,
    #[allow(dead_code)]
    phoneme_map: HashMap<i64, char>,
    phoneme_id_map: HashMap<char, Vec<i64>>,
}

#[derive(Debug, Clone)]
pub struct SynthesisConfig {
    pub noise_scale: Option<f32>,
    pub length_scale: Option<f32>,
    pub noise_w: Option<f32>,
    pub speaker: Option<String>,
}

impl SynthesisConfig {
    pub fn new(
        noise_scale: Option<f32>,
        length_scale: Option<f32>,
        noise_w: Option<f32>,
        speaker: Option<String>,
    ) -> Self {
        Self {
            noise_scale,
            length_scale,
            noise_w,
            speaker,
        }
    }
}

pub struct PiperWaveSamples(Vec<i16>);

impl PiperWaveSamples {
    pub fn to_vec(self) -> Vec<i16> {
        self.0
    }

    pub fn as_wave_bytes(&self) -> Vec<u8> {
        self.0.iter().map(|i| i.to_le_bytes()).flatten().collect()
    }
}

impl IntoIterator for PiperWaveSamples {
    type Item = i16;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl From<Vec<i16>> for PiperWaveSamples {
    fn from(samples: Vec<i16>) -> Self {
        Self(samples)
    }
}

impl From<PiperWaveSamples> for Vec<i16> {
    fn from(piper_wave: PiperWaveSamples) -> Self {
        piper_wave.0
    }
}

pub struct PiperModel {
    pub config: ModelConfig,
    onnx_path: PathBuf,
    session: OnceCell<PiperResult<ort::Session>>,
}

impl PiperModel {
    pub fn new(config_path: PathBuf, onnx_path: PathBuf) -> PiperResult<Self> {
        match Self::load_model_config(&config_path) {
            Ok(config) => Ok(Self {
                config: config,
                onnx_path: onnx_path,
                session: OnceCell::new(),
            }),
            Err(error) => Err(error),
        }
    }

    pub fn speak_text(
        &self,
        text: &str,
        synth_config: &Option<SynthesisConfig>,
    ) -> PiperResult<PiperWaveSamples> {
        self.speak_phonemes(&self.phonemize_text(&text)?.to_string(), &synth_config)
    }

    pub fn speak_phonemes(
        &self,
        phonemes: &str,
        synth_config: &Option<SynthesisConfig>,
    ) -> PiperResult<PiperWaveSamples> {
        let mut phoneme_ids: Vec<i64> = Vec::with_capacity((phonemes.len() + 1) * 2);
        let pad = self
            .config
            .phoneme_id_map
            .get(&PAD)
            .unwrap()
            .first()
            .unwrap();
        phoneme_ids.push(
            *self
                .config
                .phoneme_id_map
                .get(&BOS)
                .unwrap()
                .first()
                .unwrap(),
        );
        for phoneme in phonemes.chars() {
            if let Some(id) = self.config.phoneme_id_map.get(&phoneme) {
                phoneme_ids.push(*id.first().unwrap());
                phoneme_ids.push(*pad);
            }
        }
        phoneme_ids.push(
            *self
                .config
                .phoneme_id_map
                .get(&EOS)
                .unwrap()
                .first()
                .unwrap(),
        );
        self.infer_with_values(phoneme_ids, synth_config)
    }

    pub fn infer_with_values(
        &self,
        phoneme_ids: Vec<i64>,
        synth_config: &Option<SynthesisConfig>,
    ) -> PiperResult<PiperWaveSamples> {
        let session = match self.get_or_create_inference_session() {
            Ok(ref session) => session,
            Err(err) => {
                return Err(InferenceError(format!(
                    "Failed to initialize onnxruntime inference session: `{}`",
                    err.to_string()
                )))
            }
        };

        let (noise_scale, length_scale, noise_w, speaker) = if let Some(ref conf) = synth_config {
            (
                conf.noise_scale
                    .unwrap_or(self.config.inference.noise_scale),
                conf.length_scale
                    .unwrap_or(self.config.inference.length_scale),
                conf.noise_w.unwrap_or(self.config.inference.noise_w),
                conf.speaker.clone().unwrap_or(Default::default()),
            )
        } else {
            (
                self.config.inference.noise_scale,
                self.config.inference.length_scale,
                self.config.inference.noise_w,
                Default::default(),
            )
        };

        let mut input_tensors: Vec<InputTensor> = Vec::with_capacity(4);
        let ph_len = phoneme_ids.len();

        let inputs = Array2::<i64>::from_shape_vec((1, ph_len), phoneme_ids).unwrap();
        input_tensors.push(InputTensor::from_array(inputs.into_dyn()));

        let input_lengths = Array1::<i64>::from_iter([ph_len.try_into().unwrap()]);
        input_tensors.push(InputTensor::from_array(input_lengths.into_dyn()));

        let scales = Array1::<f32>::from_iter([noise_scale, length_scale, noise_w]);
        input_tensors.push(InputTensor::from_array(scales.into_dyn()));

        if self.config.num_speakers > 1 {
            let sid = self.config.speaker_id_map.get(&speaker).unwrap_or(&0);
            let sid_tensor = Array1::<i64>::from_iter([sid.clone()]);
            input_tensors.push(InputTensor::from_array(sid_tensor.into_dyn()));
        }

        let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> =
            session.run(input_tensors)?;
        let outputs: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
        let audio_output = outputs.view();

        let Ok(min_audio_value) = audio_output.min() else {
            return Err(InferenceError("Invalid output from model inference.".to_string()))
        };
        let Ok(max_audio_value) = audio_output.max() else {
            return Err(InferenceError("Invalid output from model inference. ".to_string()))
        };
        let abs_max = max_audio_value.max(min_audio_value.abs());
        let audio_scale = MAX_WAV_VALUE / abs_max.max(0.01f32);
        let samples: Vec<i16> = audio_output
            .iter()
            .map(|i| (i * audio_scale).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
            .collect::<Vec<i16>>();
        Ok(samples.into())
    }

    pub fn phonemize_text(&self, text: &str) -> PiperResult<Phonemes> {
        Ok(text_to_phonemes(text, &self.config.espeak.voice, None)?)
    }

    fn get_or_create_inference_session(&self) -> &PiperResult<ort::Session> {
        self.session.get_or_init(|| {
            Ok(SessionBuilder::new(&_ENVIRONMENT)?
                .with_optimization_level(GraphOptimizationLevel::Disable)?
                .with_model_from_file(&self.onnx_path)?)
            // .with_parallel_execution(true)
            // .unwrap()
            // .with_intra_threads(2)
            // .unwrap()
            // .with_memory_pattern(false)
            // .unwrap()
        })
    }

    pub fn info(&self) -> PiperResult<Vec<String>> {
        let session = match self.get_or_create_inference_session() {
            Ok(ref session) => session,
            Err(err) => {
                return Err(InferenceError(format!(
                    "Failed to initialize onnxruntime inference session: `{}`",
                    err.to_string()
                )))
            }
        };
        Ok(session
            .inputs
            .iter()
            .map(|i| {
                let name = i.name.clone();
                let dim: Vec<String> = i
                    .dimensions
                    .iter()
                    .map(|o| o.unwrap_or(42).to_string())
                    .collect();
                let dt = i.input_type.clone();
                format!("#name: {}#dims: {}#type: {:?}", name, dim.join(", "), dt)
            })
            .collect())
    }

    fn load_model_config(config_path: &PathBuf) -> PiperResult<ModelConfig> {
        let file = match File::open(&config_path) {
            Ok(file) => file,
            Err(why) => {
                return Err(FailedToLoadModelConfig(format!(
                    "Faild to load model config: `{}`. Caused by: `{}`",
                    config_path.display(),
                    why
                )))
            }
        };
        let model_config: ModelConfig = match serde_json::from_reader(file) {
            Ok(config) => config,
            Err(why) => {
                return Err(FailedToLoadModelConfig(format!(
                    "Faild to parse model config from file: `{}`. Caused by: `{}`",
                    config_path.display(),
                    why
                )))
            }
        };
        Ok(model_config)
    }
}
