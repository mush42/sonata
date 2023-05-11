use piper_model::{PiperError, PiperWaveSamples, SynthesisConfig};
use piper_synth::{AudioOutputConfig, PiperSpeechGenerator, PiperSpeechSynthesizer};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::PathBuf;

type PyPiperResult<T> = Result<T, PiperException>;

struct PiperException(PiperError);

impl From<PiperException> for PyErr {
    fn from(other: PiperException) -> Self {
        PyRuntimeError::new_err(other.0.to_string())
    }
}

impl From<PiperError> for PiperException {
    fn from(other: PiperError) -> Self {
        Self(other)
    }
}

#[pyclass(weakref, module = "piper", frozen)]
struct WaveSamples(PiperWaveSamples, u32);

#[pymethods]
impl WaveSamples {
    fn get_wave_bytes(&self, py: Python) -> PyObject {
        let bytes_vec = py.allow_threads(move || self.0.as_wave_bytes());
        PyBytes::new(py, &bytes_vec).into()
    }
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.1
    }
    #[getter]
    fn num_channels(&self) -> u8 {
        1
    }
}

#[pyclass(weakref, module = "piper")]
struct SpeechGenerator(PiperSpeechGenerator);

impl From<PiperSpeechGenerator> for SpeechGenerator {
    fn from(other: PiperSpeechGenerator) -> Self {
        Self(other)
    }
}

#[pymethods]
impl SpeechGenerator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> Option<WaveSamples> {
        let next_item = py.allow_threads(|| self.0.next());
        let audio_result = match next_item {
            Some(result) => result,
            None => return None,
        };
        match audio_result {
            Ok(audio_data) => Some(WaveSamples(audio_data, self.0.get_sample_rate())),
            Err(e) => {
                PyErr::from(PiperException::from(e)).restore(py);
                None
            }
        }
    }
}

#[pyclass(weakref, module = "piper", frozen)]
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
        text: String,
        speaker: Option<String>,
        rate: Option<u8>,
        volume: Option<u8>,
        pitch: Option<u8>,
    ) -> SpeechGenerator {
        self.0
            .synthesize(
                text,
                Some(SynthesisConfig::new(None, None, None, speaker)),
                Some(AudioOutputConfig::new(rate, volume, pitch)),
            )
            .into()
    }
}

/// A fast, local neural text-to-speech system
#[pymodule]
fn pyper(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WaveSamples>()?;
    m.add_class::<SpeechGenerator>()?;
    m.add_class::<Piper>()?;
    Ok(())
}
