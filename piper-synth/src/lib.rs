mod utils;

use once_cell::sync::OnceCell;
use piper_model::{
    PiperError, PiperModel, PiperResult, PiperWaveResult, PiperWaveSamples, SynthesisConfig,
};
use rayon::prelude::*;
use std::collections::vec_deque::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

const RATE_RANGE: (f32, f32) = (0.0f32, 5.0f32);
const VOLUME_RANGE: (f32, f32) = (0.1f32, 1.9f32);
const PITCH_RANGE: (f32, f32) = (0.5f32, 1.5f32);
/// Batch size when using batched synthesis mode
const SPEECH_STREAM_BATCH_SIZE: usize = 4;

pub struct AudioOutputConfig {
    rate: Option<u8>,
    volume: Option<u8>,
    pitch: Option<u8>,
}

impl AudioOutputConfig {
    pub fn new(rate: Option<u8>, volume: Option<u8>, pitch: Option<u8>) -> Self {
        Self {
            rate,
            volume,
            pitch,
        }
    }

    fn has_any_option_set(&self) -> bool {
        self.rate.is_some() || self.volume.is_some() || self.pitch.is_some()
    }

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
        Ok(PiperWaveSamples::from_raw(out_buf, info))
    }
}

pub struct PiperSpeechSynthesizer(Arc<PiperModel>);

impl PiperSpeechSynthesizer {
    pub fn new(config_path: PathBuf, onnx_path: PathBuf) -> PiperResult<Self> {
        let model = PiperModel::new(config_path, onnx_path)?;
        Ok(Self(Arc::new(model)))
    }

    fn create_synthesis_task_provider(
        &self,
        text: String,
        synth_config: Option<SynthesisConfig>,
        output_config: Option<AudioOutputConfig>,
    ) -> SpeechSynthesisTaskProvider {
        SpeechSynthesisTaskProvider {
            model: Arc::clone(&self.0),
            text,
            synth_config,
            output_config,
        }
    }

    pub fn synthesize_lazy(
        &self,
        text: String,
        synth_config: Option<SynthesisConfig>,
        output_config: Option<AudioOutputConfig>,
    ) -> PiperResult<PiperSpeechStream<Lazy>> {
        PiperSpeechStream::<Lazy>::new(self.create_synthesis_task_provider(
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
    ) -> PiperResult<PiperSpeechStream<Parallel>> {
        PiperSpeechStream::<Parallel>::new(self.create_synthesis_task_provider(
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
    ) -> PiperResult<PiperSpeechStream<Batched>> {
        PiperSpeechStream::<Batched>::new_batched(
            self.create_synthesis_task_provider(text, synth_config, output_config),
            batch_size.unwrap_or(SPEECH_STREAM_BATCH_SIZE),
        )
    }

    pub fn info(&self) -> PiperResult<Vec<String>> {
        self.0.info()
    }
}

/// The following marker types represent how the speech stream generate it's results
/// assuming that it takes t_i to speak each sentence
/// Lazy: takes sum(t_i) to speak the whole text
pub struct Lazy;
/// Parallel: takes utmost max(t_i) to speak the whole text
pub struct Parallel;
/// Batched: takes at least  max(t_i) to speak the whole text
/// there is a good chance that the next sentence's speech whill be ready when requested
pub struct Batched;

struct SpeechSynthesisTaskProvider {
    pub model: Arc<PiperModel>,
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
            Some(ref config) => {
                if !config.has_any_option_set() {
                    return Ok(audio);
                }
                Ok(config.apply(audio)?)
            }
            None => Ok(audio),
        }
    }
}

pub struct PiperSpeechStream<Mode> {
    provider: Arc<SpeechSynthesisTaskProvider>,
    sentence_phonemes: Option<std::vec::IntoIter<String>>,
    precalculated_results: Option<std::vec::IntoIter<PiperWaveResult>>,
    channel: Option<SpeechSynthesisChannel>,
    batch_size: usize,
    mode: std::marker::PhantomData<Mode>,
}

impl<Mode> PiperSpeechStream<Mode> {
    fn new(provider: SpeechSynthesisTaskProvider) -> PiperResult<Self> {
        Ok(Self {
            provider: Arc::new(provider),
            sentence_phonemes: None,
            precalculated_results: None,
            channel: None,
            batch_size: 0,
            mode: std::marker::PhantomData,
        })
    }
}

impl PiperSpeechStream<Batched> {
    fn new_batched(provider: SpeechSynthesisTaskProvider, batch_size: usize) -> PiperResult<Self> {
        let mut instance = Self::new(provider)?;
        instance.batch_size = batch_size;
        Ok(instance)
    }
}

impl Iterator for PiperSpeechStream<Lazy> {
    type Item = PiperWaveResult;

    fn next(&mut self) -> Option<Self::Item> {
        let sent_phoneme_iter = match self.sentence_phonemes {
            Some(ref mut ph) => ph,
            None => match self.provider.get_phonemes() {
                Ok(ph) => self.sentence_phonemes.insert(ph.into_iter()),
                Err(e) => return Some(Err(e)),
            },
        };
        match sent_phoneme_iter.next() {
            Some(sent_phonemes) => Some(self.provider.process_phonemes(sent_phonemes)),
            None => None,
        }
    }
}

impl Iterator for PiperSpeechStream<Parallel> {
    type Item = PiperWaveResult;

    fn next(&mut self) -> Option<Self::Item> {
        if self.precalculated_results.is_none() {
            let phonemes = match self.provider.get_phonemes() {
                Ok(ph) => ph,
                Err(e) => return Some(Err(e)),
            };
            let calculated_result: Vec<PiperWaveResult> = phonemes
                .par_iter()
                .map(|ph| self.provider.process_phonemes(ph.to_string()))
                .collect();
            self.precalculated_results = Some(calculated_result.into_iter());
        }
        match self.precalculated_results {
            Some(ref mut res_iter) => res_iter.next(),
            None => None,
        }
    }
}

impl Iterator for PiperSpeechStream<Batched> {
    type Item = PiperWaveResult;

    fn next(&mut self) -> Option<Self::Item> {
        if self.channel.is_none() {
            match SpeechSynthesisChannel::new(self.batch_size) {
                Ok(ch) => self.channel = Some(ch),
                Err(e) => return Some(Err(e)),
            }
        }
        let channel: &mut SpeechSynthesisChannel;
        if let Some(ref mut ch) = self.channel {
            channel = ch;
        } else {
            return None;
        }
        let sent_phoneme_iter = match self.sentence_phonemes {
            Some(ref mut ph) => ph,
            None => match self.provider.get_phonemes() {
                Ok(ph) => self.sentence_phonemes.insert(ph.into_iter()),
                Err(e) => return Some(Err(e)),
            },
        };
        sent_phoneme_iter.take(self.batch_size).for_each(|p| {
            let provider = Arc::clone(&self.provider);
            channel.put(provider, p);
        });
        channel.get()
    }
}

struct SpeechSynthesisTask(Arc<OnceCell<PiperWaveResult>>);

impl SpeechSynthesisTask {
    fn new(
        provider: Arc<SpeechSynthesisTaskProvider>,
        phonemes: String,
        thread_pool: &rayon::ThreadPool,
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
    thread_pool: rayon::ThreadPool,
}

impl SpeechSynthesisChannel {
    fn new(batch_size: usize) -> PiperResult<Self> {
        let thread_pool_builder = rayon::ThreadPoolBuilder::new()
            .num_threads(batch_size)
            .thread_name(|i| format!("piper_synth_{}", i))
            .build();
        let thread_pool = match thread_pool_builder {
            Ok(tp) => tp,
            Err(e) => {
                return Err(PiperError::OperationError(format!(
                    "Failed to build thread pool. Error: {}",
                    e
                )))
            }
        };
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
