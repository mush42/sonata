use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;


pub use audio_ops::{
    Audio,
    AudioInfo,
    AudioSamples,
    WaveWriterError
};


pub type SonataResult<T> = Result<T, SonataError>;
pub type SonataAudioResult = SonataResult<Audio>;


#[derive(Debug)]
pub enum SonataError {
    FailedToLoadResource(String),
    PhonemizationError(String),
    OperationError(String),
}

impl Error for SonataError {}

impl fmt::Display for SonataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let err_message = match self {
            SonataError::FailedToLoadResource(msg) => {
                format!("Failed to load resource from. Error `{}`", msg)
            }
            SonataError::PhonemizationError(msg) => msg.to_string(),
            SonataError::OperationError(msg) => msg.to_string(),
        };
        write!(f, "{}", err_message)
    }
}

impl From<WaveWriterError> for SonataError {
    fn from(error: WaveWriterError) -> Self {
        SonataError::OperationError(error.to_string())
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


pub trait SonataModel {
    fn audio_output_info(&self) -> SonataResult<AudioInfo>;
    fn phonemize_text(&self, text: &str) -> SonataResult<Phonemes>;
    fn speak_batch(&self, phoneme_batches: Vec<String>) -> SonataResult<Vec<Audio>>;
    fn speak_one_sentence(&self, phonemes: String) -> SonataAudioResult;

    fn get_default_synthesis_config(&self) -> SonataResult<Box<dyn Any>>;
    fn get_fallback_synthesis_config(&self) -> SonataResult<Box<dyn Any>>;
    fn set_fallback_synthesis_config(&self, synthesis_config: &dyn Any) -> SonataResult<()>;

    fn get_language(&self) -> SonataResult<Option<String>> {
        Ok(None)
    }
    fn get_speakers(&self) -> SonataResult<Option<&HashMap<i64, String>>> {
        Ok(None)
    }
    fn speaker_id_to_name(&self, sid: &i64) -> SonataResult<Option<String>> {
        Ok(self
            .get_speakers()?
            .and_then(|speakers| speakers.get(sid))
            .cloned())
    }
    fn speaker_name_to_id(&self, name: &str) -> SonataResult<Option<i64>> {
        Ok(self.get_speakers()?.and_then(|speakers| {
            for (sid, sname) in speakers {
                if sname == name {
                    return Some(*sid);
                }
            }
            None
        }))
    }
    fn properties(&self) -> SonataResult<HashMap<String, String>> {
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
    ) -> SonataResult<Box<dyn Iterator<Item = SonataResult<AudioSamples>> + Send + Sync + 'a>> {
        Ok(Box::new(
            [Err(SonataError::OperationError(
                "Streaming synthesis is not supported for this model".to_string(),
            ))]
            .into_iter(),
        ))
    }
}

