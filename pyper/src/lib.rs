use piper_model::{PiperError, PiperWaveSamples, SynthesisConfig};
use piper_synth::{
    AudioOutputConfig, PiperSpeechStreamBatched, PiperSpeechStreamLazy, PiperSpeechStreamParallel,
    PiperSpeechSynthesizer,
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
#[pyo3(name = "SynthConfig")]
#[derive(Clone)]
struct PySynthConfig(SynthesisConfig);

#[pymethods]
impl PySynthConfig {
    #[new]
    fn new(
        speaker: Option<String>,
        noise_scale: Option<f32>,
        length_scale: Option<f32>,
        noise_w: Option<f32>,
    ) -> Self {
        Self(SynthesisConfig {
            speaker,
            noise_scale,
            length_scale,
            noise_w,
        })
    }
}

impl From<PySynthConfig> for SynthesisConfig {
    fn from(other: PySynthConfig) -> Self {
        other.0
    }
}

#[pyclass(weakref, module = "piper", frozen)]
#[pyo3(name = "AudioOutputConfig")]
#[derive(Clone)]
struct PyAudioOutputConfig(AudioOutputConfig);

#[pymethods]
impl PyAudioOutputConfig {
    #[new]
    fn new(
        rate: Option<u8>,
        volume: Option<u8>,
        pitch: Option<u8>,
        appended_silence_ms: Option<u32>,
    ) -> Self {
        Self(AudioOutputConfig {
            rate,
            volume,
            pitch,
            appended_silence_ms,
        })
    }
}

impl From<PyAudioOutputConfig> for AudioOutputConfig {
    fn from(other: PyAudioOutputConfig) -> Self {
        other.0
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
    fn save_to_file(&self, filename: &str) -> PyPiperResult<()> {
        Ok(self.0.save_to_file(filename)?)
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
    #[getter]
    fn inference_ms(&self) -> Option<f32> {
        self.0.inference_ms()
    }
    #[getter]
    fn duration_ms(&self) -> f32 {
        self.0.duration_ms()
    }
    #[getter]
    fn real_time_factor(&self) -> Option<f32> {
        self.0.real_time_factor()
    }
}

#[pyclass(weakref, module = "piper")]
struct LazySpeechStream(PiperSpeechStreamLazy);

impl From<PiperSpeechStreamLazy> for LazySpeechStream {
    fn from(other: PiperSpeechStreamLazy) -> Self {
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
struct ParallelSpeechStream(PiperSpeechStreamParallel);

impl From<PiperSpeechStreamParallel> for ParallelSpeechStream {
    fn from(other: PiperSpeechStreamParallel) -> Self {
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
struct BatchedSpeechStream(PiperSpeechStreamBatched);

impl From<PiperSpeechStreamBatched> for BatchedSpeechStream {
    fn from(other: PiperSpeechStreamBatched) -> Self {
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
        synth_config: Option<PySynthConfig>,
        audio_output_config: Option<PyAudioOutputConfig>,
    ) -> PyPiperResult<LazySpeechStream> {
        self.synthesize_lazy(text, synth_config, audio_output_config)
    }

    fn synthesize_lazy(
        &self,
        text: String,
        synth_config: Option<PySynthConfig>,
        audio_output_config: Option<PyAudioOutputConfig>,
    ) -> PyPiperResult<LazySpeechStream> {
        Ok(self
            .0
            .synthesize_lazy(
                text,
                synth_config.map(|s| s.into()),
                audio_output_config.map(|o| o.into()),
            )?
            .into())
    }

    fn synthesize_parallel(
        &self,
        text: String,
        synth_config: Option<PySynthConfig>,
        audio_output_config: Option<PyAudioOutputConfig>,
    ) -> PyPiperResult<ParallelSpeechStream> {
        Ok(self
            .0
            .synthesize_parallel(
                text,
                synth_config.map(|s| s.into()),
                audio_output_config.map(|o| o.into()),
            )?
            .into())
    }

    fn synthesize_batched(
        &self,
        text: String,
        synth_config: Option<PySynthConfig>,
        audio_output_config: Option<PyAudioOutputConfig>,
        batch_size: Option<usize>,
    ) -> PyPiperResult<BatchedSpeechStream> {
        Ok(self
            .0
            .synthesize_batched(
                text,
                synth_config.map(|s| s.into()),
                audio_output_config.map(|o| o.into()),
                batch_size,
            )?
            .into())
    }

    fn synthesize_to_file(
        &self,
        filename: &str,
        text: String,
        synth_config: Option<PySynthConfig>,
        audio_output_config: Option<PyAudioOutputConfig>,
    ) -> PyPiperResult<()> {
        Ok(self
            .0
            .synthesize_to_file(
                filename,
                text,
                synth_config.map(|s| s.into()),
                audio_output_config.map(|o| o.into()),
            )?
            .into())
    }
}

/// A fast, local neural text-to-speech system
#[pymodule]
fn pyper(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySynthConfig>()?;
    m.add_class::<PyAudioOutputConfig>()?;
    m.add_class::<WaveSamples>()?;
    m.add_class::<LazySpeechStream>()?;
    m.add_class::<ParallelSpeechStream>()?;
    m.add_class::<BatchedSpeechStream>()?;
    m.add_class::<Piper>()?;
    Ok(())
}
