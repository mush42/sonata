mod utils;

use once_cell::sync::{Lazy, OnceCell};
use piper_core::{PiperError, PiperModel, PiperResult, PiperWaveResult, PiperWaveSamples};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::vec_deque::VecDeque;
use std::sync::Arc;

const RATE_RANGE: (f32, f32) = (0.0f32, 5.0f32);
const VOLUME_RANGE: (f32, f32) = (0.1f32, 1.9f32);
const PITCH_RANGE: (f32, f32) = (0.5f32, 1.5f32);
/// Batch size when using batched synthesis mode
const SPEECH_STREAM_BATCH_SIZE: usize = 4;

static SYNTHESIS_THREAD_POOL: Lazy<ThreadPool> = Lazy::new(|| {
    ThreadPoolBuilder::new()
        .thread_name(|i| format!("piper_synth_{}", i))
        .num_threads(num_cpus::get())
        .build()
        .unwrap()
});

#[derive(Clone)]
pub struct AudioOutputConfig {
    pub rate: Option<u8>,
    pub volume: Option<u8>,
    pub pitch: Option<u8>,
    pub appended_silence_ms: Option<u32>,
}

impl AudioOutputConfig {
    fn apply(&self, mut audio: PiperWaveSamples) -> PiperWaveResult {
        let input_len = audio.len();
        if input_len == 0 {
            return Ok(audio);
        }
        let samples = std::mem::take(&mut audio.samples);
        let mut out_buf: Vec<i16> = Vec::new();
        unsafe {
            let stream = sonic_sys::sonicCreateStream(
                audio.info.sample_rate as i32,
                audio.info.num_channels as i32,
            );
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
            let num_samples = (time_ms * audio.info.sample_rate as u32) / 1000u32;
            out_buf.append(&mut vec![0i16; num_samples as usize]);
        };
        audio.samples = out_buf;
        Ok(audio)
    }
}

pub struct PiperSpeechSynthesizer(Arc<dyn PiperModel + Sync + Send>);

impl PiperSpeechSynthesizer {
    pub fn new(model: Arc<dyn PiperModel + Sync + Send>) -> PiperResult<Self> {
        Ok(Self(model))
    }

    fn create_synthesis_task_provider(
        &self,
        text: String,
        output_config: Option<AudioOutputConfig>,
    ) -> SpeechSynthesisTaskProvider {
        SpeechSynthesisTaskProvider {
            model: Arc::clone(&self.0),
            text,
            output_config,
        }
    }

    pub fn synthesize_lazy(
        &self,
        text: String,
        output_config: Option<AudioOutputConfig>,
    ) -> PiperResult<PiperSpeechStreamLazy> {
        PiperSpeechStreamLazy::new(self.create_synthesis_task_provider(text, output_config))
    }
    pub fn synthesize_parallel(
        &self,
        text: String,
        output_config: Option<AudioOutputConfig>,
    ) -> PiperResult<PiperSpeechStreamParallel> {
        PiperSpeechStreamParallel::new(self.create_synthesis_task_provider(text, output_config))
    }
    pub fn synthesize_batched(
        &self,
        text: String,
        output_config: Option<AudioOutputConfig>,
        batch_size: Option<usize>,
    ) -> PiperResult<PiperSpeechStreamBatched> {
        let mut batch_size = batch_size.unwrap_or(SPEECH_STREAM_BATCH_SIZE);
        if batch_size == 0 {
            batch_size = SPEECH_STREAM_BATCH_SIZE;
        }
        PiperSpeechStreamBatched::new(
            self.create_synthesis_task_provider(text, output_config),
            batch_size,
        )
    }
    pub fn synthesize_to_file(
        &self,
        filename: &str,
        text: String,
        output_config: Option<AudioOutputConfig>,
    ) -> PiperResult<()> {
        let mut samples: Vec<i16> = Vec::new();
        for result in self.synthesize_parallel(text, output_config)? {
            match result {
                Ok(ws) => {
                    samples.append(&mut ws.to_vec());
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
            self.0.wave_info()?.sample_rate as u32,
            self.0.wave_info()?.num_channels.try_into().unwrap(),
            self.0.wave_info()?.sample_width.try_into().unwrap(),
        )?)
    }
}

struct SpeechSynthesisTaskProvider {
    model: Arc<dyn PiperModel + Sync + Send>,
    text: String,
    output_config: Option<AudioOutputConfig>,
}

impl SpeechSynthesisTaskProvider {
    fn get_phonemes(&self) -> PiperResult<Vec<String>> {
        Ok(self.model.phonemize_text(&self.text)?.to_vec())
    }
    fn process_one_sentence(&self, phonemes: String) -> PiperWaveResult {
        let wave_samples = self.model.speak_one_sentence(vec![phonemes])?.pop().unwrap();
        match self.output_config {
            Some(ref config) => config.apply(wave_samples),
            None => Ok(wave_samples),
        }
    }
    #[allow(dead_code)]
    fn process_batches(&self, phonemes: Vec<String>) -> PiperResult<Vec<PiperWaveSamples>> {
        let wave_samples = self.model.speak_batch(phonemes)?;
        match self.output_config {
            Some(ref config) => {
                let mut processed: Vec<PiperWaveSamples> = Vec::with_capacity(wave_samples.len());
                for samples in wave_samples.into_iter() {
                    processed.push(config.apply(samples)?);
                }
                Ok(processed)
            }
            None => Ok(wave_samples),
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
        let next_batch = self.sentence_phonemes.next()?;
        match  self.provider.process_one_sentence(next_batch) {
            Ok(ws) => Some(Ok(ws)),
            Err(e) => Some(Err(e)),
        }
    }
}

#[must_use]
pub struct PiperSpeechStreamParallel {
    precalculated_results: std::vec::IntoIter<PiperWaveResult>,
}

impl PiperSpeechStreamParallel {
    fn new(provider: SpeechSynthesisTaskProvider) -> PiperResult<Self> {
        let calculated_result: Vec<PiperWaveResult> = provider.get_phonemes()?
            .par_iter()
            .map(|ph| provider.process_one_sentence(ph.to_string()))
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

#[must_use]
pub struct PiperSpeechStreamBatched {
    provider: Arc<SpeechSynthesisTaskProvider>,
    sentence_phonemes: std::vec::IntoIter<String>,
    channel: SpeechSynthesisChannel,
    batch_size: usize,
}

impl PiperSpeechStreamBatched {
    fn new(provider: SpeechSynthesisTaskProvider, batch_size: usize) -> PiperResult<Self> {
        let sentence_phonemes = provider.get_phonemes()?.into_iter();
        let mut instance = Self {
            provider: Arc::new(provider),
            sentence_phonemes,
            channel: SpeechSynthesisChannel::new(batch_size)?,
            batch_size,
        };
        instance.send_batch();
        Ok(instance)
    }
    fn send_batch(&mut self) {
        let next_batch = Vec::from_iter((&mut self.sentence_phonemes).take(self.batch_size));
        if !next_batch.is_empty() {
            let provider = Arc::clone(&self.provider);
            self.channel.put(provider, next_batch);
        }
    }
}

impl Iterator for PiperSpeechStreamBatched {
    type Item = PiperWaveResult;

    fn next(&mut self) -> Option<Self::Item> {
        self.send_batch();
        self.channel.get()
    }
}

struct SpeechSynthesisTask(Arc<OnceCell<Vec<PiperWaveResult>>>);

impl SpeechSynthesisTask {
    fn new(provider: Arc<SpeechSynthesisTaskProvider>, batch: Vec<String>) -> Self {
        let instance = Self(Arc::new(OnceCell::new()));
        let result = Arc::clone(&instance.0);
        SYNTHESIS_THREAD_POOL.spawn_fifo(move || {
            let wave_samples: Vec<PiperWaveResult> = batch
                .par_iter()
                .map(|ph| provider.process_one_sentence(ph.to_string()))
                .collect();
            result.set(wave_samples).unwrap();
        });
        instance
    }
    fn get_result(self) -> PiperResult<Vec<PiperWaveResult>> {
        self.0.wait();
        if let Ok(result) = Arc::try_unwrap(self.0) {
            Ok(result.into_inner().unwrap())
        } else {
            Err(PiperError::OperationError(
                "Failed to obtain results".to_string(),
            ))
        }
    }
}

struct SpeechSynthesisChannel {
    task_queue: VecDeque<SpeechSynthesisTask>,
    result_queue: VecDeque<PiperWaveResult>,
}

impl SpeechSynthesisChannel {
    fn new(batch_size: usize) -> PiperResult<Self> {
        Ok(Self {
            result_queue: VecDeque::with_capacity(batch_size * 4),
            task_queue: VecDeque::with_capacity(batch_size * 4),
        })
    }
    fn put(&mut self, provider: Arc<SpeechSynthesisTaskProvider>, batch: Vec<String>) {
        self.task_queue
            .push_back(SpeechSynthesisTask::new(provider, batch));
    }
    fn get(&mut self) -> Option<PiperWaveResult> {
        if let Some(result) = self.result_queue.pop_front() {
            return Some(result);
        }
        if let Some(task) = self.task_queue.pop_front() {
            let results = match task.get_result() {
                Ok(res) => res,
                Err(e) => return Some(Err(e)),
            };
            for audio in results {
                self.result_queue.push_back(audio);
            }
            self.get()
        } else {
            None
        }
    }
}
