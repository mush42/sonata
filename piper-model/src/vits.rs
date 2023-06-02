use crate::{PiperError, FailedToLoadResource, OperationError, PiperResult, PiperWaveResult, PiperWaveSamples, PiperModel};
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use once_cell::sync::{OnceCell, Lazy};
use ort::{
    tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtError, SessionBuilder,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;


const MAX_WAV_VALUE: f32 = 32767.0;
const BOS: char = '^';
const EOS: char = '$';
const PAD: char = '_';

static ENVIRONMENT: Lazy<Arc<ort::Environment>> = Lazy::new(|| {
    Arc::new(
        Environment::builder()
            .with_name("piper")
            .with_execution_providers([ExecutionProvider::cpu()])
            .build()
            .unwrap()
    )
});


impl From<OrtError> for PiperError {
    fn from(error: OrtError) -> Self {
        OperationError(format!("Failed to run onnxruntime inference: `{}`", error))
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

#[derive(Debug, Clone, Default)]
pub struct SynthesisConfig {
    pub noise_scale: Option<f32>,
    pub length_scale: Option<f32>,
    pub noise_w: Option<f32>,
    pub speaker: Option<String>,
}


pub struct VitsModel {
    pub synth_config: SynthesisConfig,
    config: ModelConfig,
    onnx_path: PathBuf,
    session: OnceCell<PiperResult<ort::Session>>,
}

impl VitsModel {
    pub fn new(config_path: PathBuf, onnx_path: PathBuf) -> PiperResult<Self> {
        match Self::load_model_config(&config_path) {
            Ok(config) => Ok(Self {
                synth_config: Default::default(),
                config,
                onnx_path,
                session: OnceCell::new(),
            }),
            Err(error) => Err(error),
        }
    }
    pub fn speakers(&self) -> PiperResult<Vec<String>> {
        let speaker_ids = Vec::from_iter(self.config.speaker_id_map.keys().map(|k| k.clone()));
        Ok(speaker_ids)
    }
    pub fn infer_with_values(
        &self,
        phoneme_ids: Vec<i64>,
        synth_config: Option<&SynthesisConfig>,
    ) -> PiperWaveResult {
        let session = match self.get_or_create_inference_session() {
            Ok(ref session) => session,
            Err(err) => {
                return Err(OperationError(format!(
                    "Failed to initialize onnxruntime inference session: `{}`",
                    err
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
            let sid_tensor = Array1::<i64>::from_iter([*sid]);
            input_tensors.push(InputTensor::from_array(sid_tensor.into_dyn()));
        }

        let timer = std::time::Instant::now();
        let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> =
            session.run(input_tensors)?;
        let inference_ms = timer.elapsed().as_millis() as f32;

        let outputs: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
        let audio_output = outputs.view();

        let Ok(min_audio_value) = audio_output.min() else {
            return Err(OperationError("Invalid output from model inference.".to_string()))
        };
        let Ok(max_audio_value) = audio_output.max() else {
            return Err(OperationError("Invalid output from model inference. ".to_string()))
        };
        let abs_max = max_audio_value.max(min_audio_value.abs());
        let audio_scale = MAX_WAV_VALUE / abs_max.max(0.01f32);
        let samples: Vec<i16> = audio_output
            .iter()
            .map(|i| (i * audio_scale).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
            .collect::<Vec<i16>>();
        Ok(PiperWaveSamples::new(
            samples,
            self.config.audio.sample_rate as usize,
            Some(inference_ms),
        ))
    }
    fn get_or_create_inference_session(&self) -> &PiperResult<ort::Session> {
        self.session.get_or_init(|| {
            Ok(SessionBuilder::new(&ENVIRONMENT)?
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
    fn get_input_output_info(&self) -> PiperResult<Vec<String>> {
        let session = match self.get_or_create_inference_session() {
            Ok(ref session) => session,
            Err(err) => {
                return Err(OperationError(format!(
                    "Failed to initialize onnxruntime inference session: `{}`",
                    err
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
                let dt = i.input_type;
                format!("#name: {}#dims: {}#type: {:?}", name, dim.join(", "), dt)
            })
            .collect())
    }
    fn load_model_config(config_path: &PathBuf) -> PiperResult<ModelConfig> {
        let file = match File::open(config_path) {
            Ok(file) => file,
            Err(why) => {
                return Err(FailedToLoadResource(format!(
                    "Faild to load model config: `{}`. Caused by: `{}`",
                    config_path.display(),
                    why
                )))
            }
        };
        let model_config: ModelConfig = match serde_json::from_reader(file) {
            Ok(config) => config,
            Err(why) => {
                return Err(FailedToLoadResource(format!(
                    "Faild to parse model config from file: `{}`. Caused by: `{}`",
                    config_path.display(),
                    why
                )))
            }
        };
        Ok(model_config)
    }
}


impl PiperModel for VitsModel {

    fn espeak_voice(&self) -> PiperResult<String> {
        Ok(self.config.espeak.voice.clone())
    }
    fn speak_phonemes(&self, phonemes: String) -> PiperWaveResult {
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
        self.infer_with_values(phoneme_ids, Some(&self.synth_config))
    }
    fn info(&self) -> PiperResult<String> {
        Ok(self.get_input_output_info()?.join("\n"))
    }
}
