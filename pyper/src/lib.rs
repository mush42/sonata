use piper_model::{PiperError, SynthesisConfig};
use piper_synth::{AudioOutputConfig, PiperSpeechSynthesizer};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::PathBuf;

type PyPiperResult<T> = Result<T, PiperException>;

struct PiperException(PiperError);

impl From<PiperException> for PyErr {
    fn from(err: PiperException) -> Self {
        PyRuntimeError::new_err(err.0.to_string())
    }
}

impl From<PiperError> for PiperException {
    fn from(other: PiperError) -> Self {
        Self(other)
    }
}

#[pyclass(weakref, module = "piper")]
struct Piper(PiperSpeechSynthesizer);

#[pymethods]
impl Piper {
    #[new]
    fn new(config_path: &str, model_path: &str) -> PyPiperResult<Self> {
        Ok(Self(PiperSpeechSynthesizer::new(
            PathBuf::from(config_path),
            PathBuf::from(model_path),
        )?))
    }

    pub fn info(&self) -> PyPiperResult<Vec<String>> {
        Ok(self.0.info()?)
    }

    fn synthesize(
        &self,
        py: Python,
        text: &str,
        speaker: Option<String>,
        rate: Option<u8>,
        volume: Option<u8>,
        pitch: Option<u8>,
    ) -> PyPiperResult<PyObject> {
        let audio_data = py.allow_threads(move || {
            self.0.synthesize(
                text,
                Some(SynthesisConfig::new(None, None, None, speaker)),
                Some(AudioOutputConfig::new(volume, rate, pitch)),
            )
        })?;
        let audio_bytes: Vec<u8> = audio_data
            .iter()
            .map(|i| i.to_ne_bytes())
            .flatten()
            .collect();
        Ok(PyBytes::new(py, &audio_bytes).into())
    }
}

/// A fast, local neural text-to-speech system
#[pymodule]
fn pyper(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Piper>()?;
    Ok(())
}
