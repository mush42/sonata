use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;


pub use audio_ops::{
    AudioInfo as PiperWaveInfo,
    SynthesisAudioSamples as PiperWaveSamples,
    RawAudioSamples as RawWaveSamples,
    WaveWriterError
};
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

impl From<WaveWriterError> for PiperError {
    fn from(error: WaveWriterError) -> Self {
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

