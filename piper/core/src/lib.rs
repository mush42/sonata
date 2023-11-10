use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

const I16MIN_F32: f32 = i16::MIN as f32;
const I16MAX_F32: f32 = i16::MAX as f32;
const MAX_WAV_VALUE: f32 = 32767.0;

pub type PiperResult<T> = Result<T, PiperError>;
pub type PiperWaveResult = PiperResult<PiperWaveSamples>;

#[derive(Debug)]
pub enum PiperError {
    FailedToLoadResource(String),
    PhonemizationError(String),
    OperationError(String),
}

impl Error for PiperError {}

impl fmt::Display for PiperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let err_message = match self {
            PiperError::FailedToLoadResource(msg) => {
                format!("Failed to load resource from. Error `{}`", msg)
            }
            PiperError::PhonemizationError(msg) => msg.to_string(),
            PiperError::OperationError(msg) => msg.to_string(),
        };
        write!(f, "{}", err_message)
    }
}

impl From<wave_writer::WaveWriterError> for PiperError {
    fn from(error: wave_writer::WaveWriterError) -> Self {
        PiperError::OperationError(error.to_string())
    }
}

/// A wrapper type that holds sentence phonemes
pub struct Phonemes(Vec<String>);

impl Phonemes {
    pub fn sentences(&self) -> &Vec<String> {
        &self.0
    }

    pub fn to_vec(self) -> Vec<String> {
        self.0
    }

    pub fn num_sentences(&self) -> usize {
        self.0.len()
    }
}

impl From<Vec<String>> for Phonemes {
    fn from(other: Vec<String>) -> Self {
        Self(other)
    }
}

impl std::string::ToString for Phonemes {
    fn to_string(&self) -> String {
        self.0.join(" ")
    }
}

#[derive(Debug, Clone)]
pub struct PiperWaveInfo {
    pub sample_rate: usize,
    pub num_channels: usize,
    pub sample_width: usize,
}

#[derive(Clone, Debug, Default)]
#[must_use]
pub struct RawWaveSamples(Vec<f32>);

impl RawWaveSamples {
    pub fn new(samples: Vec<f32>) -> Self {
        Self(samples)
    }
    pub fn as_slice(&self) -> &[f32] {
        self.0.as_slice()
    }
    pub fn as_vec(&self) -> &Vec<f32> {
        &self.0
    }
    pub fn as_mut_vec(&mut self) -> &mut Vec<f32> {
        &mut self.0
    }
    pub fn into_vec(self) -> Vec<f32> {
        self.0
    }
    pub fn take(&mut self) -> Vec<f32> {
        std::mem::take(self.0.as_mut())
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn to_i16_vec(&self) -> Vec<i16> {
        if self.is_empty() {
            return Default::default();
        }
        let min_audio_value = self.0.iter().copied().reduce(|x, y| x.min(y)).unwrap();
        let max_audio_value = self.0.iter().copied().reduce(|x, y| x.max(y)).unwrap();
        let abs_max = max_audio_value.max(min_audio_value.abs());
        let audio_scale = MAX_WAV_VALUE / abs_max.max(f32::EPSILON);
        Vec::from_iter(
            self.0
                .iter()
                .map(|i| (i * audio_scale).clamp(I16MIN_F32, I16MAX_F32) as i16),
        )
    }
    pub fn as_wave_bytes(&self) -> Vec<u8> {
        Vec::from_iter(self.to_i16_vec().iter().flat_map(|i| i.to_le_bytes()))
    }
}

impl From<RawWaveSamples> for Vec<f32> {
    fn from(other: RawWaveSamples) -> Self {
        other.into_vec()
    }
}

impl From<Vec<f32>> for RawWaveSamples {
    fn from(other: Vec<f32>) -> Self {
        Self::new(other)
    }
}

impl IntoIterator for RawWaveSamples {
    type Item = f32;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct PiperWaveSamples {
    pub samples: RawWaveSamples,
    pub info: PiperWaveInfo,
    pub inference_ms: Option<f32>,
}

impl PiperWaveSamples {
    pub fn new(samples: RawWaveSamples, sample_rate: usize, inference_ms: Option<f32>) -> Self {
        Self {
            samples,
            inference_ms,
            info: PiperWaveInfo {
                sample_rate,
                num_channels: 1,
                sample_width: 2,
            },
        }
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.samples.into_vec()
    }

    pub fn as_wave_bytes(&self) -> Vec<u8> {
        self.samples.as_wave_bytes()
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn duration_ms(&self) -> f32 {
        (self.len() as f32 / self.info.sample_rate as f32) * 1000.0f32
    }

    pub fn inference_ms(&self) -> Option<f32> {
        self.inference_ms
    }

    pub fn real_time_factor(&self) -> Option<f32> {
        let infer_ms = self.inference_ms?;
        let audio_duration = self.duration_ms();
        if audio_duration == 0.0 {
            return Some(0.0f32);
        }
        Some(infer_ms / audio_duration)
    }

    pub fn save_to_file(&self, filename: &str) -> PiperResult<()> {
        Ok(wave_writer::write_wave_samples_to_file(
            filename.into(),
            self.samples.to_i16_vec().iter(),
            self.info.sample_rate as u32,
            self.info.num_channels as u32,
            self.info.sample_width as u32,
        )?)
    }
}

impl IntoIterator for PiperWaveSamples {
    type Item = f32;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.samples.into_iter()
    }
}

pub trait PiperModel {
    fn wave_info(&self) -> PiperResult<PiperWaveInfo>;
    fn phonemize_text(&self, text: &str) -> PiperResult<Phonemes>;
    fn speak_batch(&self, phoneme_batches: Vec<String>) -> PiperResult<Vec<PiperWaveSamples>>;
    fn speak_one_sentence(&self, phonemes: String) -> PiperWaveResult;

    fn get_default_synthesis_config(&self) -> PiperResult<Box<dyn Any>>;
    fn get_fallback_synthesis_config(&self) -> PiperResult<Box<dyn Any>>;
    fn set_fallback_synthesis_config(&self, synthesis_config: &dyn Any) -> PiperResult<()>;

    fn get_language(&self) -> PiperResult<Option<String>> {
        Ok(None)
    }
    fn get_speakers(&self) -> PiperResult<Option<&HashMap<i64, String>>> {
        Ok(None)
    }
    fn speaker_id_to_name(&self, sid: &i64) -> PiperResult<Option<String>> {
        Ok(self
            .get_speakers()?
            .and_then(|speakers| speakers.get(sid))
            .cloned())
    }
    fn speaker_name_to_id(&self, name: &str) -> PiperResult<Option<i64>> {
        Ok(self.get_speakers()?.and_then(|speakers| {
            for (sid, sname) in speakers {
                if sname == name {
                    return Some(*sid);
                }
            }
            None
        }))
    }
    fn properties(&self) -> PiperResult<HashMap<String, String>> {
        Ok(HashMap::with_capacity(0))
    }

    fn supports_streaming_output(&self) -> bool {
        false
    }
    fn stream_synthesis<'a>(
        &'a self,
        #[allow(unused_variables)] phonemes: String,
        #[allow(unused_variables)] chunk_size: usize,
        #[allow(unused_variables)] chunk_padding: usize,
    ) -> PiperResult<Box<dyn Iterator<Item = PiperResult<RawWaveSamples>> + Send + Sync + 'a>> {
        Ok(Box::new(
            [Err(PiperError::OperationError(
                "Streaming synthesis is not supported for this model".to_string(),
            ))]
            .into_iter(),
        ))
    }
}
