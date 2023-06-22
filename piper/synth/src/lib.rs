mod utils;

use piper_core::{PiperError, PiperModel, PiperResult, PiperWaveResult, PiperWaveSamples};
use std::collections::vec_deque::VecDeque;
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
            1u32,
            2u32,
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
    fn process_batches(&self, phonemes: Vec<String>) -> PiperResult<Vec<PiperWaveSamples>> {
        let wave_samples = self.model.speak_phonemes(phonemes)?;
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
        match self.provider.process_batches(vec![next_batch]) {
            Ok(ws) => ws.into_iter().next().map(Ok),
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
            .process_batches(provider.get_phonemes()?)?
            .into_iter()
            .map(Ok)
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
    batch_size: usize,
    queue: VecDeque<PiperWaveResult>,
}

impl PiperSpeechStreamBatched {
    fn new(provider: SpeechSynthesisTaskProvider, batch_size: usize) -> PiperResult<Self> {
        let sentence_phonemes = provider.get_phonemes()?.into_iter();
        let mut instance = Self {
            provider: Arc::new(provider),
            sentence_phonemes,
            batch_size,
            queue: VecDeque::new(),
        };
        instance.send_batch()?;
        Ok(instance)
    }
    fn send_batch(&mut self) -> PiperResult<()> {
        let next_batch = Vec::from_iter((&mut self.sentence_phonemes).take(self.batch_size));
        if !next_batch.is_empty() {
            for wave_samples in self.provider.process_batches(next_batch)? {
                self.queue.push_back(Ok(wave_samples));
            }
        }
        Ok(())
    }
}

impl Iterator for PiperSpeechStreamBatched {
    type Item = PiperWaveResult;

    fn next(&mut self) -> Option<Self::Item> {
        if let Err(e) = self.send_batch() {
            return Some(Err(e))
        }
        self.queue.pop_front()
    }
}
