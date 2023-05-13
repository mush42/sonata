use piper_model::{PiperError, PiperWaveSamples, SynthesisConfig};
use piper_synth::{
    AudioOutputConfig, Batched, Lazy, Parallel, PiperSpeechStream, PiperSpeechSynthesizer,
};
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
struct WaveSamples(PiperWaveSamples);

#[pymethods]
impl WaveSamples {
    fn get_wave_bytes(&self, py: Python) -> PyObject {
        let bytes_vec = py.allow_threads(move || self.0.as_wave_bytes());
        PyBytes::new(py, &bytes_vec).into()
    }
    #[getter]
    fn sample_rate(&self) -> usize {
        self.0.sample_rate()
    }
    #[getter]
    fn num_channels(&self) -> usize {
        self.0.num_channels()
    }
    #[getter]
    fn sample_width(&self) -> usize {
        self.0.sample_width()
    }
}

#[pyclass(weakref, module = "piper")]
struct LazySpeechStream(PiperSpeechStream<Lazy>);

impl From<PiperSpeechStream<Lazy>> for LazySpeechStream {
    fn from(other: PiperSpeechStream<Lazy>) -> Self {
        Self(other)
    }
}

#[pymethods]
impl LazySpeechStream {
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
            Ok(audio_data) => Some(WaveSamples(audio_data)),
            Err(e) => {
                PyErr::from(PiperException::from(e)).restore(py);
                None
            }
        }
    }
}

#[pyclass(weakref, module = "piper")]
struct ParallelSpeechStream(PiperSpeechStream<Parallel>);

impl From<PiperSpeechStream<Parallel>> for ParallelSpeechStream {
    fn from(other: PiperSpeechStream<Parallel>) -> Self {
        Self(other)
    }
}

#[pymethods]
impl ParallelSpeechStream {
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
            Ok(audio_data) => Some(WaveSamples(audio_data)),
            Err(e) => {
                PyErr::from(PiperException::from(e)).restore(py);
                None
            }
        }
    }
}

#[pyclass(weakref, module = "piper")]
struct BatchedSpeechStream(PiperSpeechStream<Batched>);

impl From<PiperSpeechStream<Batched>> for BatchedSpeechStream {
    fn from(other: PiperSpeechStream<Batched>) -> Self {
        Self(other)
    }
}

#[pymethods]
impl BatchedSpeechStream {
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
            Ok(audio_data) => Some(WaveSamples(audio_data)),
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

    fn info(&self) -> PyPiperResult<Vec<String>> {
        Ok(self.0.info()?)
    }

    fn synthesize(
        &self,
        text: String,
        speaker: Option<String>,
        rate: Option<u8>,
        volume: Option<u8>,
        pitch: Option<u8>,
    ) -> PyPiperResult<LazySpeechStream> {
        self.synthesize_lazy(text, speaker, rate, volume, pitch)
    }

    fn synthesize_lazy(
        &self,
        text: String,
        speaker: Option<String>,
        rate: Option<u8>,
        volume: Option<u8>,
        pitch: Option<u8>,
    ) -> PyPiperResult<LazySpeechStream> {
        Ok(self
            .0
            .synthesize_lazy(
                text,
                Some(SynthesisConfig::new(None, None, None, speaker)),
                Some(AudioOutputConfig::new(rate, volume, pitch)),
            )?
            .into())
    }

    fn synthesize_parallel(
        &self,
        text: String,
        speaker: Option<String>,
        rate: Option<u8>,
        volume: Option<u8>,
        pitch: Option<u8>,
    ) -> PyPiperResult<ParallelSpeechStream> {
        Ok(self
            .0
            .synthesize_parallel(
                text,
                Some(SynthesisConfig::new(None, None, None, speaker)),
                Some(AudioOutputConfig::new(rate, volume, pitch)),
            )?
            .into())
    }

    fn synthesize_batched(
        &self,
        text: String,
        speaker: Option<String>,
        rate: Option<u8>,
        volume: Option<u8>,
        pitch: Option<u8>,
        batch_size: Option<usize>,
    ) -> PyPiperResult<BatchedSpeechStream> {
        Ok(self
            .0
            .synthesize_batched(
                text,
                Some(SynthesisConfig::new(None, None, None, speaker)),
                Some(AudioOutputConfig::new(rate, volume, pitch)),
                batch_size,
            )?
            .into())
    }
}

/// A fast, local neural text-to-speech system
#[pymodule]
fn pyper(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WaveSamples>()?;
    m.add_class::<LazySpeechStream>()?;
    m.add_class::<ParallelSpeechStream>()?;
    m.add_class::<BatchedSpeechStream>()?;
    m.add_class::<Piper>()?;
    Ok(())
}
