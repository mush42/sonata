#![feature(trait_upcasting)]

use once_cell::sync::OnceCell;
use piper_core::{PiperError, PiperWaveInfo, PiperWaveSamples, RawWaveSamples};
use piper_synth::{
    AudioOutputConfig, PiperSpeechStreamBatched, PiperSpeechStreamLazy, PiperSpeechStreamParallel,
    PiperSpeechSynthesizer, SYNTHESIS_THREAD_POOL,
};
use piper_vits::VitsVoice;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

type PyPiperResult<T> = Result<T, PyPiperError>;
static ORT_ENVIRONMENT: OnceCell<Arc<ort::Environment>> = OnceCell::new();

create_exception!(
    piper,
    PiperException,
    PyException,
    "Base Exception for all exceptions raised by piper."
);

struct PyPiperError(PiperError);

impl From<PyPiperError> for PyErr {
    fn from(other: PyPiperError) -> Self {
        PiperException::new_err(other.0.to_string())
    }
}

impl From<PiperError> for PyPiperError {
    fn from(other: PiperError) -> Self {
        Self(other)
    }
}

#[pyclass(weakref, module = "piper", frozen)]
#[pyo3(name = "PiperWaveInfo")]
struct PyWaveInfo(PiperWaveInfo);

#[pymethods]
impl PyWaveInfo {
    #[getter]
    fn get_sample_rate(&self) -> usize {
        self.0.sample_rate
    }
    #[getter]
    fn get_num_channels(&self) -> usize {
        self.0.num_channels
    }
    #[getter]
    fn get_sample_width(&self) -> usize {
        self.0.sample_width
    }
}

impl From<PiperWaveInfo> for PyWaveInfo {
    fn from(other: PiperWaveInfo) -> Self {
        Self(other)
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
        self.0.info.sample_rate
    }
    #[getter]
    fn num_channels(&self) -> usize {
        self.0.info.num_channels
    }
    #[getter]
    fn sample_width(&self) -> usize {
        self.0.info.sample_width
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
                PyErr::from(PyPiperError::from(e)).restore(py);
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
                PyErr::from(PyPiperError::from(e)).restore(py);
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
                PyErr::from(PyPiperError::from(e)).restore(py);
                None
            }
        }
    }
}

#[pyclass(weakref, module = "piper")]
struct RealtimeSpeechStream(std::sync::mpsc::IntoIter<PyPiperResult<RawWaveSamples>>);

impl RealtimeSpeechStream {
    fn new(receiver: std::sync::mpsc::Receiver<PyPiperResult<RawWaveSamples>>) -> Self {
        Self(receiver.into_iter())
    }
}

#[pymethods]
impl RealtimeSpeechStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> Option<PyObject> {
        let result = py.allow_threads(|| self.0.next())?;
        match result {
            Ok(samples) => Some(PyBytes::new(py, &samples.as_wave_bytes()).into()),
            Err(e) => {
                PyErr::from(e).restore(py);
                None
            }
        }
    }
}

#[pyclass(weakref, module = "piper")]
#[pyo3(name = "VitsModel")]
struct PyVitsModel(Arc<dyn VitsVoice + Send + Sync>);

impl PyVitsModel {
    fn get_ort_environment() -> &'static Arc<ort::Environment> {
        ORT_ENVIRONMENT.get_or_init(|| {
            Arc::new(
                ort::Environment::builder()
                    .with_name("piper")
                    .with_execution_providers([ort::ExecutionProvider::CPU(Default::default())])
                    .build()
                    .unwrap(),
            )
        })
    }
}

#[pymethods]
impl PyVitsModel {
    #[new]
    fn new(config_path: &str) -> PyPiperResult<Self> {
        let vits =
            piper_vits::from_config_path(&PathBuf::from(config_path), Self::get_ort_environment())?;
        Ok(Self(vits))
    }
    #[getter]
    fn speakers(&self) -> PyPiperResult<HashMap<i64, String>> {
        Ok(self.0.speakers()?)
    }
    #[getter]
    fn get_speaker(&self) -> PyPiperResult<Option<String>> {
        Ok(self.0.get_speaker()?)
    }
    #[setter]
    fn set_speaker(&self, name: String) -> PyPiperResult<()> {
        Ok(self.0.set_speaker(name)?)
    }
    #[getter]
    fn get_noise_scale(&self) -> PyPiperResult<f32> {
        Ok(self.0.get_noise_scale()?)
    }
    #[setter]
    fn set_noise_scale(&self, value: f32) -> PyPiperResult<()> {
        Ok(self.0.set_noise_scale(value)?)
    }
    #[getter]
    fn get_length_scale(&self) -> PyPiperResult<f32> {
        Ok(self.0.get_length_scale()?)
    }
    #[setter]
    fn set_length_scale(&self, value: f32) -> PyPiperResult<()> {
        Ok(self.0.set_length_scale(value)?)
    }
    #[getter]
    fn get_noise_w(&self) -> PyPiperResult<f32> {
        Ok(self.0.get_noise_w()?)
    }
    #[setter]
    fn set_noise_w(&self, value: f32) -> PyPiperResult<()> {
        Ok(self.0.set_noise_w(value)?)
    }
    fn get_wave_output_info(&self) -> PyPiperResult<PyWaveInfo> {
        Ok(self.0.wave_info()?.into())
    }
}

#[pyclass(weakref, module = "piper", frozen)]
struct Piper(Arc<PiperSpeechSynthesizer>);

#[pymethods]
impl Piper {
    #[staticmethod]
    fn with_vits(vits_model: &PyVitsModel) -> PyPiperResult<Self> {
        let model = Arc::clone(&vits_model.0);
        let synthesizer = Arc::new(PiperSpeechSynthesizer::new(model)?);
        Ok(Self(synthesizer))
    }
    fn synthesize(
        &self,
        text: String,
        audio_output_config: Option<PyAudioOutputConfig>,
    ) -> PyPiperResult<LazySpeechStream> {
        self.synthesize_lazy(text, audio_output_config)
    }

    fn synthesize_lazy(
        &self,
        text: String,
        audio_output_config: Option<PyAudioOutputConfig>,
    ) -> PyPiperResult<LazySpeechStream> {
        Ok(self
            .0
            .synthesize_lazy(text, audio_output_config.map(|o| o.into()))?
            .into())
    }

    fn synthesize_parallel(
        &self,
        text: String,
        audio_output_config: Option<PyAudioOutputConfig>,
    ) -> PyPiperResult<ParallelSpeechStream> {
        Ok(self
            .0
            .synthesize_parallel(text, audio_output_config.map(|o| o.into()))?
            .into())
    }

    fn synthesize_batched(
        &self,
        text: String,
        audio_output_config: Option<PyAudioOutputConfig>,
        batch_size: Option<usize>,
    ) -> PyPiperResult<BatchedSpeechStream> {
        Ok(self
            .0
            .synthesize_batched(text, audio_output_config.map(|o| o.into()), batch_size)?
            .into())
    }

    fn synthesize_streamed(
        &self,
        text: String,
        audio_output_config: Option<PyAudioOutputConfig>,
        chunk_size: Option<usize>,
        chunk_padding: Option<usize>,
    ) -> PyPiperResult<RealtimeSpeechStream> {
        let synth = Arc::clone(&self.0);
        let (tx, rx) = std::sync::mpsc::channel::<PyPiperResult<RawWaveSamples>>();
        SYNTHESIS_THREAD_POOL.spawn_fifo(move || {
            let stream_result = synth.synthesize_streamed(
                text,
                audio_output_config.map(|o| o.into()),
                chunk_size.unwrap_or(45),
                chunk_padding.unwrap_or(3),
            );
            let stream = match stream_result {
                Ok(stream) => stream,
                Err(e) => {
                    tx.send(Err(e.into())).unwrap();
                    return;
                }
            };
            for result in stream {
                let samples: PyPiperResult<RawWaveSamples> = result.map_err(|e| e.into());
                tx.send(samples).unwrap();
            }
        });
        Ok(RealtimeSpeechStream::new(rx))
    }

    fn synthesize_to_file(
        &self,
        filename: &str,
        text: String,
        audio_output_config: Option<PyAudioOutputConfig>,
    ) -> PyPiperResult<()> {
        self.0
            .synthesize_to_file(filename, text, audio_output_config.map(|o| o.into()))?;
        Ok(())
    }
}

/// A fast, local neural text-to-speech system
#[pymodule]
fn pyper(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("PiperException", _py.get_type::<PiperException>())?;
    m.add_class::<PyVitsModel>()?;
    m.add_class::<PyAudioOutputConfig>()?;
    m.add_class::<WaveSamples>()?;
    m.add_class::<LazySpeechStream>()?;
    m.add_class::<ParallelSpeechStream>()?;
    m.add_class::<BatchedSpeechStream>()?;
    m.add_class::<Piper>()?;
    Ok(())
}
