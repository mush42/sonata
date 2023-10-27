use espeak_phonemizer::text_to_phonemes;
use libtashkeel_base::do_tashkeel;
use ndarray::{Array1, Array2, CowArray};
use ndarray_stats::QuantileExt;
use once_cell::sync::{Lazy, OnceCell};
use ort::{tensor::OrtOwnedTensor, Environment, GraphOptimizationLevel, SessionBuilder, Value};
use piper_core::{
    Phonemes, PiperError, PiperModel, PiperResult, PiperWaveInfo, PiperWaveResult, PiperWaveSamples,
};
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

const MAX_WAV_VALUE: f32 = 32767.0;
const BOS: char = '^';
const EOS: char = '$';
const PAD: char = '_';

#[allow(dead_code)]
static CPU_COUNT: Lazy<i16> = Lazy::new(|| num_cpus::get().try_into().unwrap_or(4));

fn reversed_mapping<K, V>(input: &HashMap<K, V>) -> HashMap<V, K>
where
    K: ToOwned<Owned = K>,
    V: ToOwned<Owned = V> + std::hash::Hash + std::cmp::Eq,
{
    HashMap::from_iter(input.iter().map(|(k, v)| (v.to_owned(), k.to_owned())))
}


fn load_model_config(config_path: &PathBuf) -> PiperResult<(ModelConfig, SynthesisConfig)> {
    let file = match File::open(config_path) {
        Ok(file) => file,
        Err(why) => {
            return Err(PiperError::FailedToLoadResource(format!(
                "Faild to load model config: `{}`. Caused by: `{}`",
                config_path.display(),
                why
            )))
        }
    };
    let model_config: ModelConfig = match serde_json::from_reader(file) {
        Ok(config) => config,
        Err(why) => {
            return Err(PiperError::FailedToLoadResource(format!(
                "Faild to parse model config from file: `{}`. Caused by: `{}`",
                config_path.display(),
                why
            )))
        }
    };
    let synth_config = SynthesisConfig {
        speaker: None,
        noise_scale: model_config.inference.noise_scale,
        length_scale: model_config.inference.length_scale,
        noise_w: model_config.inference.noise_w,
    };
    Ok((model_config, synth_config))
}

#[derive(Deserialize, Default)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub quality: Option<String>
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
    pub speaker: Option<(String, i64)>,
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_w: f32,
}

pub trait VitsModelCommons {
    fn get_synth_config(&self) -> &RwLock<SynthesisConfig>;
    fn get_config(&self) -> &ModelConfig;
    fn get_speaker_map(&self) -> &HashMap<i64, String>;
    fn get_tashkeel_engine(&self) -> Option<&libtashkeel_base::DynamicInferenceEngine>;

    fn get_meta_ids(&self) -> (i64, i64, i64) {
        let config = self.get_config();
        let pad_id = *config
            .phoneme_id_map
            .get(&PAD)
            .unwrap()
            .first()
            .unwrap();
        let bos_id = *config
            .phoneme_id_map
            .get(&BOS)
            .unwrap()
            .first()
            .unwrap();
        let eos_id = *config
            .phoneme_id_map
            .get(&EOS)
            .unwrap()
            .first()
            .unwrap();
        (pad_id, bos_id, eos_id)
    }
    fn language(&self) -> Option<String> {
        self.get_config().language.as_ref().map(|lang| lang.code.clone())
    }
    fn quality(&self) -> Option<String> {
        self.get_config().audio.quality.as_ref().cloned()
    }
    fn default_synthesis_config(&self) -> SynthesisConfig {
        let config = self.get_config();
        
        let speaker = if config.num_speakers > 0 {
            self.get_speaker_map().get(&0).as_ref().cloned().map(|name| (name.clone(), 0))
        } else {
            None
        };
        SynthesisConfig {
            speaker,
            length_scale: config.inference.length_scale,
            noise_scale: config.inference.noise_scale,
            noise_w: config.inference.noise_w,
        }
    }
    fn speakers(&self) -> PiperResult<HashMap<i64, String>> {
        Ok(self.get_speaker_map().clone())
    }
    fn get_speaker(&self) -> PiperResult<Option<String>> {
        if self.get_config().num_speakers == 0 {
            return Err(PiperError::OperationError(
                "This model is a single speaker model.".to_string(),
            ));
        }
        if let Some(ref speaker) = self.get_synth_config().read().unwrap().speaker {
            Ok(Some(speaker.0.clone()))
        } else {
            let default_speaker = match self.get_speaker_map().get(&0) {
                Some(name) => name.clone(),
                None => "Default".to_string(),
            };
            Ok(Some(default_speaker))
        }
    }
    fn set_speaker(&self, name: String) -> PiperResult<()> {
        let config = self.get_config();
        if config.num_speakers == 0 {
            return Err(PiperError::OperationError(
                "This model is a single speaker model.".to_string(),
            ));
        }
        if let Some(sid) = config.speaker_id_map.get(&name) {
            let mut synth_config = self.get_synth_config().write().unwrap();
            synth_config.speaker = Some((name, *sid));
            Ok(())
        } else {
            Err(PiperError::OperationError(format!(
                "Invalid speaker name: `{}`",
                name
            )))
        }
    }
    fn get_noise_scale(&self) -> PiperResult<f32> {
        Ok(self.get_synth_config().read().unwrap().noise_scale)
    }
    fn set_noise_scale(&self, value: f32) -> PiperResult<()> {
        self.get_synth_config().write().unwrap().noise_scale = value;
        Ok(())
    }
    fn get_length_scale(&self) -> PiperResult<f32> {
        Ok(self.get_synth_config().read().unwrap().length_scale)
    }
    fn set_length_scale(&self, value: f32) -> PiperResult<()> {
        self.get_synth_config().write().unwrap().length_scale = value;
        Ok(())
    }
    fn get_noise_w(&self) -> PiperResult<f32> {
        Ok(self.get_synth_config().read().unwrap().noise_w)
    }
    fn set_noise_w(&self, value: f32) -> PiperResult<()> {
        self.get_synth_config().write().unwrap().noise_w = value;
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
    fn phonemize_text(&self, text: &str) -> PiperResult<Phonemes> {
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
                return Err(PiperError::PhonemizationError(format!(
                    "Failed to phonemize given text using espeak-ng. Error: {}",
                    e
                )))
            }
        };
        Ok(phonemes.into())
    }
    fn diacritize_text(&self, text: &str) -> PiperResult<String> {
        let diacritized_text = match do_tashkeel(self.get_tashkeel_engine().unwrap(), text, None)
        {
            Ok(d_text) => d_text,
            Err(msg) => {
                return Err(PiperError::OperationError(format!(
                    "Failed to diacritize text using  libtashkeel. {}",
                    msg
                )))
            }
        };
        Ok(diacritized_text)
    }
    fn wave_info(&self) -> PiperResult<PiperWaveInfo> {
        Ok(PiperWaveInfo {
            sample_rate: self.get_config().audio.sample_rate as usize,
            num_channels: 1usize,
            sample_width: 2usize,
        })
    }
}

pub struct VitsModel {
    synth_config: RwLock<SynthesisConfig>,
    config: ModelConfig,
    speaker_map: HashMap<i64, String>,
    onnx_path: PathBuf,
    ort_env: &'static Arc<Environment>,
    session: OnceCell<Result<ort::Session, ort::OrtError>>,
    tashkeel_engine: Option<libtashkeel_base::DynamicInferenceEngine>,
}

impl VitsModel {
    pub fn new(
        config_path: PathBuf,
        onnx_path: PathBuf,
        ort_env: &'static Arc<ort::Environment>,
    ) -> PiperResult<Self> {
        match load_model_config(&config_path) {
            Ok((config, synth_config)) => {
                let speaker_map = reversed_mapping(&config.speaker_id_map);
                let tashkeel_engine = if config.espeak.voice == "ar" {
                    match libtashkeel_base::create_inference_engine(None) {
                        Ok(engine) => Some(engine),
                        Err(msg) => {
                            return Err(PiperError::OperationError(format!(
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
                    onnx_path,
                    ort_env,
                    session: OnceCell::new(),
                    tashkeel_engine,
                })
            }
            Err(error) => Err(error),
        }
    }
    fn infer_with_values_batched(
        &self,
        mut input_batches: Vec<Vec<i64>>,
    ) -> PiperResult<Vec<PiperWaveSamples>> {
        let session = match self.get_or_create_inference_session() {
            Ok(ref session) => session,
            Err(err) => {
                return Err(PiperError::OperationError(format!(
                    "Failed to initialize onnxruntime inference session: `{}`",
                    err
                )))
            }
        };

        let synth_config = self.synth_config.read().unwrap();

        let pad_input_id = self
            .config
            .phoneme_id_map
            .get(&PAD)
            .unwrap()
            .first()
            .unwrap();
        let num_batches = input_batches.len();
        let max_len = match input_batches.iter().map(|v| v.len()).max() {
            Some(length) => length,
            None => {
                return Err(PiperError::OperationError(
                    "Empty phoneme input".to_string(),
                ))
            }
        };
        for input in input_batches.iter_mut() {
            input.resize(max_len, *pad_input_id);
        }
        let input_batches = Vec::from_iter(input_batches.into_iter().flatten());
        let phoneme_inputs = CowArray::from(
            Array2::<i64>::from_shape_vec((num_batches, max_len), input_batches).unwrap(),
        )
        .into_dyn();

        let input_lengths = CowArray::from(Array1::<i64>::from_iter(
            (0..num_batches).map(|_| max_len as i64),
        ))
        .into_dyn();

        let scales = Array1::<f32>::from_iter([
            synth_config.noise_scale,
            synth_config.length_scale,
            synth_config.noise_w,
        ]);
        let scales = CowArray::from(scales).into_dyn();

        let speaker_id = if self.config.num_speakers > 1 {
            let sid = match synth_config.speaker {
                Some((_, sid)) => sid,
                None => 0,
            };
            Some(CowArray::from(Array1::<i64>::from_iter([sid])).into_dyn())
        } else {
            None
        };

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
                    return Err(PiperError::OperationError(format!(
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
                return Err(PiperError::OperationError(format!(
                    "Failed to run model inference. Error: {}",
                    e
                )))
            }
        };
        let outputs = outputs.view();
        let audio_outputs = outputs
            .view()
            .into_shape((num_batches, *outputs.shape().last().unwrap()))
            .unwrap();

        let mut samples: Vec<Vec<i16>> = Vec::with_capacity(num_batches);
        for audio in audio_outputs.rows().into_iter() {
            let Ok(min_audio_value) = audio.min() else {
                return Err(PiperError::OperationError(
                    "Invalid output from model inference.".to_string(),
                ));
            };
            let Ok(max_audio_value) = audio.max() else {
                return Err(PiperError::OperationError(
                    "Invalid output from model inference. ".to_string(),
                ));
            };
            let abs_max = max_audio_value.max(min_audio_value.abs());
            let audio_scale = MAX_WAV_VALUE / abs_max.max(0.01f32);
            let clampped = Vec::from_iter(
                audio
                    .into_iter()
                    .map(|i| (i * audio_scale).clamp(i16::MIN as f32, i16::MAX as f32) as i16),
            );
            samples.push(clampped);
        }
        Ok(Vec::from_iter(samples.into_iter().map(|audio| {
            PiperWaveSamples::new(
                audio.into(),
                self.config.audio.sample_rate as usize,
                Some(inference_ms),
            )
        })))
    }
    fn infer_with_values(&self, input_phonemes: Vec<i64>) -> PiperWaveResult {
        let session = match self.get_or_create_inference_session() {
            Ok(ref session) => session,
            Err(err) => {
                return Err(PiperError::OperationError(format!(
                    "Failed to initialize onnxruntime inference session: `{}`",
                    err
                )))
            }
        };

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
            let sid = match synth_config.speaker {
                Some((_, sid)) => sid,
                None => 0,
            };
            Some(CowArray::from(Array1::<i64>::from_iter([sid])).into_dyn())
        } else {
            None
        };

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
                    return Err(PiperError::OperationError(format!(
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
                return Err(PiperError::OperationError(format!(
                    "Failed to run model inference. Error: {}",
                    e
                )))
            }
        };

        let audio_output = outputs.view();

        let Ok(min_audio_value) = audio_output.min() else {
            return Err(PiperError::OperationError(
                "Invalid output from model inference.".to_string(),
            ));
        };
        let Ok(max_audio_value) = audio_output.max() else {
            return Err(PiperError::OperationError(
                "Invalid output from model inference. ".to_string(),
            ));
        };
        let abs_max = max_audio_value.max(min_audio_value.abs());
        let audio_scale = MAX_WAV_VALUE / abs_max.max(0.01f32);
        let samples: Vec<i16> = audio_output
            .iter()
            .map(|i| (i * audio_scale).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
            .collect::<Vec<i16>>();
        Ok(PiperWaveSamples::new(
            samples.into(),
            self.config.audio.sample_rate as usize,
            Some(inference_ms),
        ))
    }
    fn get_or_create_inference_session(&self) -> &Result<ort::Session, ort::OrtError> {
        self.session.get_or_init(|| {
            SessionBuilder::new(self.ort_env)?
                .with_optimization_level(GraphOptimizationLevel::Disable)?
                .with_memory_pattern(false)?
                .with_parallel_execution(false)?
                .with_model_from_file(&self.onnx_path)
        })
    }
    pub fn get_input_output_info(&self) -> PiperResult<Vec<String>> {
        let session = match self.get_or_create_inference_session() {
            Ok(ref session) => session,
            Err(err) => {
                return Err(PiperError::OperationError(format!(
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
}


impl VitsModelCommons for &VitsModel {
    fn get_synth_config(&self) -> &RwLock<SynthesisConfig> {
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

impl PiperModel for VitsModel {
    fn phonemize_text(&self, text: &str) -> PiperResult<Phonemes> {
        VitsModelCommons::phonemize_text(&self, text)
    }

    fn speak_batch(&self, phoneme_batches: Vec<String>) -> PiperResult<Vec<PiperWaveSamples>> {
        let (pad_id, bos_id, eos_id) = self.get_meta_ids();
        let phoneme_batches = Vec::from_iter(
            phoneme_batches
                .into_iter()
                .map(|batch| self.phonemes_to_input_ids(&batch, pad_id, bos_id, eos_id)),
        );
        self.infer_with_values_batched(phoneme_batches)
    }

    fn speak_one_sentence(&self, phonemes: String) -> PiperWaveResult {
        let (pad_id, bos_id, eos_id) = self.get_meta_ids();
        let phonemes = self.phonemes_to_input_ids(&phonemes, pad_id, bos_id, eos_id);
        self.infer_with_values(phonemes)
    }
    fn wave_info(&self) -> PiperResult<PiperWaveInfo> {
        VitsModelCommons::wave_info(&self)
    }

}
