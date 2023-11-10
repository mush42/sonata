mod utils;

use once_cell::sync::{Lazy, OnceCell};
use piper_core::{
    Phonemes, PiperError, PiperModel, PiperResult, PiperWaveInfo, PiperWaveResult,
    PiperWaveSamples, RawWaveSamples,
};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::any::Any;
use std::collections::{vec_deque::VecDeque, HashMap};
use std::sync::Arc;

const RATE_RANGE: (f32, f32) = (0.0f32, 5.0f32);
const VOLUME_RANGE: (f32, f32) = (0.1f32, 1.9f32);
const PITCH_RANGE: (f32, f32) = (0.5f32, 1.5f32);
/// Batch size when using batched synthesis mode
const SPEECH_STREAM_BATCH_SIZE: usize = 4;

pub static SYNTHESIS_THREAD_POOL: Lazy<ThreadPool> = Lazy::new(|| {
    let num_cpus = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4);
    ThreadPoolBuilder::new()
        .thread_name(|i| format!("piper_synth_{}", i))
        .num_threads(num_cpus * 3)
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
        let mut samples = audio.samples.take();
        if let Some(time_ms) = self.appended_silence_ms {
            let mut silence_samples = self.generate_silence(
                time_ms as usize,
                audio.info.sample_rate,
                audio.info.num_channels
            )?;
            samples.append(silence_samples.take().as_mut());
        }
        let mut samples =
            self.apply_to_raw_samples(samples.into(), audio.info.sample_rate, audio.info.num_channels)?;
        audio.samples.as_mut_vec().append(samples.as_mut_vec());
        Ok(audio)
    }
    fn apply_to_raw_samples(
        &self,
        samples: RawWaveSamples,
        sample_rate: usize,
        num_channels: usize,
    ) -> PiperResult<RawWaveSamples> {
        let samples = samples.into_vec();
        let input_len = samples.len();
        if input_len == 0 {
            return Ok(samples.into());
        }
        let mut out_buf: Vec<f32> = Vec::new();
        unsafe {
            let stream = sonic_sys::sonicCreateStream(sample_rate as i32, num_channels as i32);
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
            sonic_sys::sonicWriteFloatToStream(stream, samples.as_ptr(), input_len as i32);
            sonic_sys::sonicFlushStream(stream);
            let num_samples = sonic_sys::sonicSamplesAvailable(stream);
            if num_samples <= 0 {
                return Err(
                    PiperError::OperationError("Sonic Error: failed to apply audio config. Invalid parameter value for rate, volume, or pitch".to_string())
                );
            }
            out_buf.reserve_exact(num_samples as usize);
            sonic_sys::sonicReadFloatFromStream(
                stream,
                out_buf.spare_capacity_mut().as_mut_ptr().cast(),
                num_samples,
            );
            sonic_sys::sonicDestroyStream(stream);
            out_buf.set_len(num_samples as usize);
        }
        Ok(out_buf.into())
    }
    #[inline(always)]
    fn generate_silence(
        &self,
        time_ms: usize,
        sample_rate: usize,
        num_channels: usize
    ) -> PiperResult<RawWaveSamples> {
        let num_samples = (time_ms * sample_rate) / 1000;
        let silence_samples = vec![0f32; num_samples];
        self.apply_to_raw_samples(
            silence_samples.into(),
            sample_rate,
            num_channels,
        )
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
            model: self.clone_model(),
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
    pub fn synthesize_streamed(
        &self,
        text: String,
        output_config: Option<AudioOutputConfig>,
        chunk_size: usize,
        chunk_padding: usize,
    ) -> PiperResult<RealtimeSpeechStream> {
        let phonemes = self.0.phonemize_text(&text)?;
        let stream = self
            .0
            .stream_synthesis(phonemes.to_string(), chunk_size, chunk_padding)?;
        let wavinfo = self.0.wave_info()?;
        Ok(RealtimeSpeechStream::new(
            stream,
            output_config,
            wavinfo.sample_rate,
            wavinfo.num_channels,
        ))
    }

    pub fn synthesize_to_file(
        &self,
        filename: &str,
        text: String,
        output_config: Option<AudioOutputConfig>,
    ) -> PiperResult<()> {
        let mut samples: Vec<f32> = Vec::new();
        for result in self.synthesize_parallel(text, output_config)? {
            match result {
                Ok(ws) => {
                    samples.append(&mut ws.into_vec());
                }
                Err(e) => return Err(e),
            };
        }
        if samples.is_empty() {
            return Err(PiperError::OperationError(
                "No speech data to write".to_string(),
            ));
        }
        let audio = RawWaveSamples::from(samples);
        Ok(wave_writer::write_wave_samples_to_file(
            filename.into(),
            audio.to_i16_vec().iter(),
            self.0.wave_info()?.sample_rate as u32,
            self.0.wave_info()?.num_channels.try_into().unwrap(),
            self.0.wave_info()?.sample_width.try_into().unwrap(),
        )?)
    }
    #[inline(always)]
    pub fn clone_model(&self) -> Arc<dyn PiperModel + Send + Sync> {
        Arc::clone(&self.0)
    }
}

impl PiperModel for PiperSpeechSynthesizer {
    fn wave_info(&self) -> PiperResult<PiperWaveInfo> {
        self.0.wave_info()
    }
    fn phonemize_text(&self, text: &str) -> PiperResult<Phonemes> {
        self.0.phonemize_text(text)
    }
    fn speak_batch(&self, phoneme_batches: Vec<String>) -> PiperResult<Vec<PiperWaveSamples>> {
        self.0.speak_batch(phoneme_batches)
    }
    fn speak_one_sentence(&self, phonemes: String) -> PiperWaveResult {
        self.0.speak_one_sentence(phonemes)
    }
    fn get_default_synthesis_config(&self) -> PiperResult<Box<dyn Any>> {
        self.0.get_default_synthesis_config()
    }
    fn get_fallback_synthesis_config(&self) -> PiperResult<Box<dyn Any>> {
        self.0.get_fallback_synthesis_config()
    }
    fn set_fallback_synthesis_config(&self, synthesis_config: &dyn Any) -> PiperResult<()> {
        self.0.set_fallback_synthesis_config(synthesis_config)
    }
    fn get_language(&self) -> PiperResult<Option<String>> {
        self.0.get_language()
    }
    fn get_speakers(&self) -> PiperResult<Option<&HashMap<i64, String>>> {
        self.0.get_speakers()
    }
    fn properties(&self) -> PiperResult<HashMap<String, String>> {
        self.0.properties()
    }
    fn supports_streaming_output(&self) -> bool {
        self.0.supports_streaming_output()
    }
    fn stream_synthesis<'a>(
        &'a self,
        #[allow(unused_variables)] phonemes: String,
        #[allow(unused_variables)] chunk_size: usize,
        #[allow(unused_variables)] chunk_padding: usize,
    ) -> PiperResult<Box<dyn Iterator<Item = PiperResult<RawWaveSamples>> + Send + Sync + 'a>> {
        self.0.stream_synthesis(phonemes, chunk_size, chunk_padding)
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
        let wave_samples = self.model.speak_one_sentence(phonemes)?;
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
        let phonemes = self.sentence_phonemes.next()?;
        match self.provider.process_one_sentence(phonemes) {
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
        let calculated_result: Vec<PiperWaveResult> = provider
            .get_phonemes()?
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

struct SpeechSynthesisTask(Arc<OnceCell<PiperWaveResult>>);

impl SpeechSynthesisTask {
    fn new(provider: Arc<SpeechSynthesisTaskProvider>, phonemes: String) -> Self {
        let instance = Self(Arc::new(OnceCell::new()));
        let result = Arc::clone(&instance.0);
        SYNTHESIS_THREAD_POOL.spawn_fifo(move || {
            let wave_result = provider.process_one_sentence(phonemes);
            result.set(wave_result).unwrap();
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
    task_queue: VecDeque<SpeechSynthesisTask>,
}

impl SpeechSynthesisChannel {
    fn new(batch_size: usize) -> PiperResult<Self> {
        Ok(Self {
            task_queue: VecDeque::with_capacity(batch_size * 4),
        })
    }
    fn put(&mut self, provider: Arc<SpeechSynthesisTaskProvider>, batch: Vec<String>) {
        for phonemes in batch.into_iter() {
            self.task_queue
                .push_back(SpeechSynthesisTask::new(Arc::clone(&provider), phonemes));
        }
    }
    fn get(&mut self) -> Option<PiperWaveResult> {
        self.task_queue.pop_front().map(|task| task.get_result())
    }
}

pub struct RealtimeSpeechStream<'a> {
    stream: Box<dyn Iterator<Item = PiperResult<RawWaveSamples>> + Send + Sync + 'a>,
    output_config: Option<AudioOutputConfig>,
    sample_rate: usize,
    num_channels: usize,
    finished: bool,
}

impl<'a> RealtimeSpeechStream<'a> {
    fn new(
        stream: Box<dyn Iterator<Item = PiperResult<RawWaveSamples>> + Send + Sync + 'a>,
        output_config: Option<AudioOutputConfig>,
        sample_rate: usize,
        num_channels: usize,
    ) -> Self {
        Self {
            stream,
            output_config,
            sample_rate,
            num_channels,
            finished: false,
        }
    }
    fn apply_audio_output_config(
        &self,
        wav_result: PiperResult<RawWaveSamples>,
    ) -> PiperResult<RawWaveSamples> {
        if let Some(ref output_config) = self.output_config {
            output_config.apply_to_raw_samples(wav_result?, self.sample_rate, self.num_channels)
        } else {
            wav_result
        }
    }
}

impl<'a> Iterator for RealtimeSpeechStream<'a> {
    type Item = PiperResult<RawWaveSamples>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.stream.next() {
            Some(raw_samples) => Some(self.apply_audio_output_config(raw_samples)),
            None => {
                self.finished = true;
                self.output_config.as_ref().and_then(|output_config| {
                    output_config.appended_silence_ms.map(|time_ms| {
                        output_config.generate_silence(
                            time_ms as usize,
                            self.sample_rate,
                            self.num_channels,
                        )
                    })
                })
            }
        }
    }
}

impl std::fmt::Debug for RealtimeSpeechStream<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "RealtimeSpeechStream Iterator")
    }
}
