mod utils;

use once_cell::sync::OnceCell;
use piper_model::vits::{SynthesisConfig, VitsModel};
use piper_model::{PiperError, PiperModel, PiperResult, PiperWaveResult, PiperWaveSamples};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::vec_deque::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

const RATE_RANGE: (f32, f32) = (0.0f32, 5.0f32);
const VOLUME_RANGE: (f32, f32) = (0.1f32, 1.9f32);
const PITCH_RANGE: (f32, f32) = (0.5f32, 1.5f32);
/// Batch size when using batched synthesis mode
const SPEECH_STREAM_BATCH_SIZE: usize = 4;

#[derive(Clone)]
pub struct AudioOutputConfig {
    pub rate: Option<u8>,
    pub volume: Option<u8>,
    pub pitch: Option<u8>,
    pub appended_silence_ms: Option<u32>,
}

impl AudioOutputConfig {
    fn apply(&self, audio: PiperWaveSamples) -> PiperWaveResult {
        let input_len = audio.len();
        if input_len == 0 {
            return Ok(audio);
        }
        let (samples, info) = audio.into_raw();
        let mut out_buf: Vec<i16> = Vec::new();
        unsafe {
            let stream =
                sonic_sys::sonicCreateStream(info.sample_rate as i32, info.num_channels as i32);
            if let Some(rate) = self.rate {
                sonic_sys::sonicSetSpeed(
                    stream,
                    utils::percent_to_param(rate, RATE_RANGE.0, RATE_RANGE.1),
                );
            }
            if let Some(volume) = self.volume {
                sonic_sys::sonicSetVolume(
                    stream,
                    utils::percent_to_param(volume, VOLUME_RANGE.0, VOLUME_RANGE.1),
                );
            }
            if let Some(pitch) = self.pitch {
                sonic_sys::sonicSetPitch(
                    stream,
                    utils::percent_to_param(pitch, PITCH_RANGE.0, PITCH_RANGE.1),
                );
            }
            sonic_sys::sonicWriteShortToStream(stream, samples.as_ptr(), input_len as i32);
            sonic_sys::sonicFlushStream(stream);
            let num_samples = sonic_sys::sonicSamplesAvailable(stream);
            if num_samples <= 0 {
                return Err(
                    PiperError::OperationError("Sonic Error: failed to apply audio config. Invalid parameter value for rate, volume, or pitch".to_string())
                );
            }
            out_buf.reserve_exact(num_samples as usize);
            sonic_sys::sonicReadShortFromStream(
                stream,
                out_buf.spare_capacity_mut().as_mut_ptr().cast(),
                num_samples,
            );
            sonic_sys::sonicDestroyStream(stream);
            out_buf.set_len(num_samples as usize);
        }
        if let Some(time_ms) = self.appended_silence_ms {
            let num_samples = (time_ms * info.sample_rate as u32) / 1000u32;
            out_buf.append(&mut vec![0i16; num_samples as usize]);
        };
        Ok(PiperWaveSamples::from_raw(out_buf, info))
    }
}

pub struct PiperSpeechSynthesizer {
    model: Arc<VitsModel>,
    thread_pool: OnceCell<Arc<ThreadPool>>,
}

impl PiperSpeechSynthesizer {
    pub fn new(config_path: PathBuf, onnx_path: PathBuf) -> PiperResult<Self> {
        let model = VitsModel::new(config_path, onnx_path)?;
        Ok(Self {
            model: Arc::new(model),
            thread_pool: OnceCell::new(),
        })
    }

    fn create_synthesis_task_provider(
        &self,
        text: String,
        synth_config: Option<SynthesisConfig>,
        output_config: Option<AudioOutputConfig>,
    ) -> SpeechSynthesisTaskProvider {
        SpeechSynthesisTaskProvider {
            model: Arc::clone(&self.model),
            text,
            synth_config,
            output_config,
        }
    }

    fn get_or_create_thread_pool(&self) -> Arc<ThreadPool> {
        Arc::clone(self.thread_pool.get_or_init(|| {
            let thread_pool = ThreadPoolBuilder::new()
                .thread_name(|i| format!("piper_synth_{}", i))
                .build()
                .unwrap();
            Arc::new(thread_pool)
        }))
    }

    pub fn synthesize_lazy(
        &self,
        text: String,
        synth_config: Option<SynthesisConfig>,
        output_config: Option<AudioOutputConfig>,
    ) -> PiperResult<PiperSpeechStreamLazy> {
        PiperSpeechStreamLazy::new(self.create_synthesis_task_provider(
            text,
            synth_config,
            output_config,
        ))
    }

    pub fn synthesize_parallel(
        &self,
        text: String,
        synth_config: Option<SynthesisConfig>,
        output_config: Option<AudioOutputConfig>,
    ) -> PiperResult<PiperSpeechStreamParallel> {
        PiperSpeechStreamParallel::new(self.create_synthesis_task_provider(
            text,
            synth_config,
            output_config,
        ))
    }

    pub fn synthesize_batched(
        &self,
        text: String,
        synth_config: Option<SynthesisConfig>,
        output_config: Option<AudioOutputConfig>,
        batch_size: Option<usize>,
    ) -> PiperResult<PiperSpeechStreamBatched> {
        let mut batch_size = batch_size.unwrap_or(SPEECH_STREAM_BATCH_SIZE);
        if batch_size == 0 {
            batch_size = SPEECH_STREAM_BATCH_SIZE;
        }
        let thread_pool = self.get_or_create_thread_pool();
        PiperSpeechStreamBatched::new(
            self.create_synthesis_task_provider(text, synth_config, output_config),
            thread_pool,
            batch_size,
        )
    }

    pub fn synthesize_to_file(
        &self,
        filename: &str,
        text: String,
        synth_config: Option<SynthesisConfig>,
        output_config: Option<AudioOutputConfig>,
    ) -> PiperResult<()> {
        let mut samples: Vec<i16> = Vec::new();
        for result in self.synthesize_parallel(text, synth_config, output_config)? {
            match result {
                Ok(ws) => {
                    samples.append(&mut ws.into_raw().0);
                }
                Err(e) => return Err(e),
            };
        }
        if samples.is_empty() {
            return Err(PiperError::OperationError(
                "No speech data to write".to_string(),
            ));
        }
        Ok(wave_writer::write_wave_samples_to_file(
            filename.into(),
            samples.iter(),
            self.model.config.audio.sample_rate,
            1u32,
            2u32,
        )?)
    }

    pub fn info(&self) -> PiperResult<String> {
        self.model.info()
    }
}

struct SpeechSynthesisTaskProvider {
    model: Arc<VitsModel>,
    text: String,
    synth_config: Option<SynthesisConfig>,
    output_config: Option<AudioOutputConfig>,
}

impl SpeechSynthesisTaskProvider {
    fn get_phonemes(&self) -> PiperResult<Vec<String>> {
        Ok(self.model.phonemize_text(&self.text)?.to_vec())
    }
    fn process_phonemes(&self, phonemes: String) -> PiperWaveResult {
        let audio = self.model.speak_phonemes(phonemes, &self.synth_config)?;
        match self.output_config {
            Some(ref config) => Ok(config.apply(audio)?),
            None => Ok(audio),
        }
    }
}

pub struct PiperSpeechStreamLazy {
    provider: SpeechSynthesisTaskProvider,
    sentence_phonemes: std::vec::IntoIter<String>,
}

impl PiperSpeechStreamLazy {
    fn new(provider: SpeechSynthesisTaskProvider) -> PiperResult<Self> {
        let sentence_phonemes = provider.get_phonemes()?.into_iter();
        Ok(Self {
            provider,
            sentence_phonemes,
        })
    }
}

impl Iterator for PiperSpeechStreamLazy {
    type Item = PiperWaveResult;

    fn next(&mut self) -> Option<Self::Item> {
        self.sentence_phonemes
            .next()
            .map(|p| self.provider.process_phonemes(p))
    }
}

pub struct PiperSpeechStreamParallel {
    precalculated_results: std::vec::IntoIter<PiperWaveResult>,
}

impl PiperSpeechStreamParallel {
    fn new(provider: SpeechSynthesisTaskProvider) -> PiperResult<Self> {
        let calculated_result: Vec<PiperWaveResult> = provider
            .get_phonemes()?
            .par_iter()
            .map(|ph| provider.process_phonemes(ph.to_string()))
            .collect();
        Ok(Self {
            precalculated_results: calculated_result.into_iter(),
        })
    }
}

impl Iterator for PiperSpeechStreamParallel {
    type Item = PiperWaveResult;

    fn next(&mut self) -> Option<Self::Item> {
        self.precalculated_results.next()
    }
}

pub struct PiperSpeechStreamBatched {
    provider: Arc<SpeechSynthesisTaskProvider>,
    sentence_phonemes: std::vec::IntoIter<String>,
    channel: SpeechSynthesisChannel,
    batch_size: usize,
}

impl PiperSpeechStreamBatched {
    fn new(
        provider: SpeechSynthesisTaskProvider,
        thread_pool: Arc<ThreadPool>,
        batch_size: usize,
    ) -> PiperResult<Self> {
        let sentence_phonemes = provider.get_phonemes()?.into_iter();
        let mut instance = Self {
            provider: Arc::new(provider),
            sentence_phonemes,
            channel: SpeechSynthesisChannel::new(thread_pool, batch_size)?,
            batch_size,
        };
        instance.send_batch();
        Ok(instance)
    }
    fn send_batch(&mut self) {
        (&mut self.sentence_phonemes)
            .take(self.batch_size)
            .for_each(|p| {
                let provider = Arc::clone(&self.provider);
                self.channel.put(provider, p);
            });
    }
}

impl Iterator for PiperSpeechStreamBatched {
    type Item = PiperWaveResult;

    fn next(&mut self) -> Option<Self::Item> {
        self.send_batch();
        self.channel.get()
    }
}

struct SpeechSynthesisTask(Arc<OnceCell<PiperWaveResult>>);

impl SpeechSynthesisTask {
    fn new(
        provider: Arc<SpeechSynthesisTaskProvider>,
        phonemes: String,
        thread_pool: &ThreadPool,
    ) -> Self {
        let instance = Self(Arc::new(OnceCell::new()));
        let result = Arc::clone(&instance.0);
        thread_pool.spawn_fifo(move || {
            result.set(provider.process_phonemes(phonemes)).unwrap();
        });
        instance
    }
    fn get_result(self) -> PiperWaveResult {
        self.0.wait();
        if let Ok(result) = Arc::try_unwrap(self.0) {
            result.into_inner().unwrap()
        } else {
            Err(PiperError::OperationError(
                "Failed to obtain results".to_string(),
            ))
        }
    }
}

struct SpeechSynthesisChannel {
    worker_queue: VecDeque<SpeechSynthesisTask>,
    thread_pool: Arc<ThreadPool>,
}

impl SpeechSynthesisChannel {
    fn new(thread_pool: Arc<ThreadPool>, batch_size: usize) -> PiperResult<Self> {
        Ok(Self {
            worker_queue: VecDeque::with_capacity(batch_size * 2),
            thread_pool,
        })
    }
    fn put(&mut self, provider: Arc<SpeechSynthesisTaskProvider>, phonemes: String) {
        self.worker_queue.push_back(SpeechSynthesisTask::new(
            provider,
            phonemes,
            &self.thread_pool,
        ));
    }
    fn get(&mut self) -> Option<PiperWaveResult> {
        self.worker_queue.pop_front().map(|task| task.get_result())
    }
}
