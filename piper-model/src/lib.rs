pub mod vits;

use espeak_phonemizer::{text_to_phonemes, ESpeakError, Phonemes};
use std::error::Error;
use std::fmt;

pub type PiperResult<T> = Result<T, PiperError>;
pub type PiperWaveResult = PiperResult<PiperWaveSamples>;
use PiperError::{
    FailedToLoadResource, PhonemizationError, OperationError,
};

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
            FailedToLoadResource(msg) => {
                format!("Failed to load resource from. Error `{}`", msg)
            }
            PhonemizationError(msg) => msg.to_string(),
            OperationError(msg) => msg.to_string(),
        };
        write!(f, "{}", err_message)
    }
}

impl From<ESpeakError> for PiperError {
    fn from(other: ESpeakError) -> Self {
        PhonemizationError(other.0)
    }
}


impl From<wave_writer::WaveWriterError> for PiperError {
    fn from(error: wave_writer::WaveWriterError) -> Self {
        OperationError(error.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct PiperWaveInfo {
    pub sample_rate: usize,
    pub num_channels: usize,
    pub sample_width: usize,
    pub inference_ms: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct PiperWaveSamples {
    samples: Vec<i16>,
    info: PiperWaveInfo,
}

impl PiperWaveSamples {
    pub fn new(samples: Vec<i16>, sample_rate: usize, inference_ms: Option<f32>) -> Self {
        Self {
            samples,
            info: PiperWaveInfo {
                sample_rate,
                inference_ms,
                num_channels: 1,
                sample_width: 2,
            },
        }
    }

    pub fn from_raw(samples: Vec<i16>, info: PiperWaveInfo) -> Self {
        Self { samples, info }
    }

    pub fn into_raw(self) -> (Vec<i16>, PiperWaveInfo) {
        (self.samples, self.info)
    }

    pub fn as_wave_bytes(&self) -> Vec<u8> {
        self.samples.iter().flat_map(|i| i.to_le_bytes()).collect()
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn sample_rate(&self) -> usize {
        self.info.sample_rate
    }

    pub fn num_channels(&self) -> usize {
        self.info.num_channels
    }

    pub fn sample_width(&self) -> usize {
        self.info.sample_width
    }

    pub fn duration_ms(&self) -> f32 {
        (self.len() as f32 / self.sample_rate() as f32) * 1000.0f32
    }

    pub fn inference_ms(&self) -> Option<f32> {
        self.info.inference_ms
    }

    pub fn real_time_factor(&self) -> Option<f32> {
        let Some(infer_ms) = self.info.inference_ms else {
             return None
         };
        let audio_duration = self.duration_ms();
        if audio_duration == 0.0 {
            return Some(0.0f32);
        }
        Some(infer_ms / audio_duration)
    }

    pub fn save_to_file(&self, filename: &str) -> PiperResult<()> {
        Ok(wave_writer::write_wave_samples_to_file(
            filename.into(),
            self.samples.iter(),
            self.sample_rate() as u32,
            self.num_channels() as u32,
            self.sample_width() as u32
        )?)
    }
}


impl IntoIterator for PiperWaveSamples {
    type Item = i16;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.samples.into_iter()
    }
}


pub trait PiperModel<T> {

    fn speak_phonemes(&self, phonemes: String, synth_params: &T) -> PiperWaveResult;
    fn espeak_voice(&self) -> PiperResult<String>;
    fn info(&self) -> PiperResult<String>;

    fn phonemize_text(&self, text: &str) -> PiperResult<Phonemes> {
        Ok(text_to_phonemes(text, &self.espeak_voice()?, None)?)
    }

    fn speak_text(&self, text: &str, synth_params: &T) -> PiperWaveResult {
        self.speak_phonemes(self.phonemize_text(text)?.to_string(), synth_params)
    }
}
