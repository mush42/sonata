mod utils;

use piper_model::{PiperError, PiperModel, PiperResult, PiperWaveSamples, SynthesisConfig};
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use sonic_sys;
use std::path::PathBuf;
use std::sync::Arc;

const RATE_RANGE: (f32, f32) = (0.0f32, 5.0f32);
const VOLUME_RANGE: (f32, f32) = (0.1f32, 1.9f32);
const PITCH_RANGE: (f32, f32) = (0.5f32, 1.5f32);

// const DEFAULT_RATE: u8 = 20;
// const DEFAULT_VOLUME: u8 = 75;
// const DEFAULT_PITCH: u8 = 50;

pub struct AudioOutputConfig {
    rate: Option<u8>,
    volume: Option<u8>,
    pitch: Option<u8>,
}

impl AudioOutputConfig {
    pub fn new(rate: Option<u8>, volume: Option<u8>, pitch: Option<u8>) -> Self {
        Self {
            rate: rate,
            volume: volume,
            pitch: pitch,
        }
    }

    fn has_any_option_set(&self) -> bool {
        self.rate.is_some() || self.volume.is_some() || self.pitch.is_some()
    }

    fn apply(&self, audio: Vec<i16>, sample_rate: u32) -> PiperResult<Vec<i16>> {
        if audio.len() == 0 {
            return Ok(audio);
        }
        let mut out_buf: Vec<i16> = Vec::new();
        unsafe {
            let stream = sonic_sys::sonicCreateStream(sample_rate as i32, 1);
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
            sonic_sys::sonicWriteShortToStream(stream, audio.as_ptr(), audio.len() as i32);
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
        Ok(out_buf)
    }
}

pub struct PiperSpeechSynthesizer(Arc<PiperModel>);

impl PiperSpeechSynthesizer {
    pub fn new(config_path: PathBuf, onnx_path: PathBuf) -> PiperResult<Self> {
        let model = PiperModel::new(config_path, onnx_path)?;
        Ok(Self(Arc::new(model)))
    }

    pub fn synthesize(
        &self,
        text: String,
        synth_config: Option<SynthesisConfig>,
        output_config: Option<AudioOutputConfig>,
    ) -> PiperResult<PiperSpeechStream> {
        let task = PiperSpeechSynthesisTask {
            model: Arc::clone(&self.0),
            text,
            synth_config,
            output_config
        };
        PiperSpeechStream::<Lazy>::new(task)
    }

    pub fn info(&self) -> PiperResult<Vec<String>> {
        self.0.info()
    }
}

pub enum PiperStreamingMode {
    Lazy,
    Parallel,
    Batched,
}

/// The following marker types represent how the speech stream generate it's results
/// assuming that it takes t_i to speak each sentence
/// Lazy: takes sum(t_i) to speak the whole text
pub struct Lazy;
/// Parallel: takes utmost max(t_i) to speak the whole text
pub struct Parallel;
/// Incremental: takes at least  max(t_i) to speak the whole text
/// there is a good chance that the next sentence's speech whill be ready when requested
pub struct Incremental;

pub struct PiperSpeechSynthesisTask {
    pub model: Arc<PiperModel>,
    text: String,
    synth_config: Option<SynthesisConfig>,
    output_config: Option<AudioOutputConfig>,
}

impl PiperSpeechSynthesisTask {
    pub fn get_sample_rate(&self) -> u32 {
        self.model.config.audio.sample_rate
    }
    fn get_phonemes(&self) -> PiperResult<Vec<String>> {
        Ok(self.model.phonemize_text(&self.text)?.to_vec())
    }
    fn process_phonemes(&self, phonemes: String) -> PiperResult<PiperWaveSamples> {
        let audio = self.model.speak_phonemes(phonemes, &self.synth_config)?;
        match self.output_config {
            Some(ref config) => {
                if !config.has_any_option_set() {
                    return Ok(audio);
                }
                Ok(config
                    .apply(audio.into(), self.model.config.audio.sample_rate)?
                    .into())
            }
            None => Ok(audio),
        }
    }
}


pub struct PiperSpeechStream<Mode = Lazy> {
    task: Arc<PiperSpeechSynthesisTask>,
    sentence_phonemes: Option<std::vec::IntoIter<String>>,
    precalculated_results: Option<std::vec::IntoIter<PiperResult<PiperWaveSamples>>>,
    channel: Option<Channel>,
    mode: std::marker::PhantomData<Mode>,
}


impl<Mode> PiperSpeechStream<Mode> {
    fn new(task: PiperSpeechSynthesisTask) -> PiperResult<Self> {
        Ok(Self {
            task: Arc::new(task),
            sentence_phonemes: None,
            precalculated_results: None,
            channel: None,
            mode: std::marker::PhantomData,
        })
    }
}

impl Iterator for PiperSpeechStream<Lazy> {
    type Item = PiperResult<PiperWaveSamples>;

    fn next(&mut self) -> Option<Self::Item> {
        let sent_phoneme_iter = match self.sentence_phonemes {
            Some(ref mut ph) => ph,
            None => match self.task.get_phonemes() {
                Ok(ph) => self.sentence_phonemes.insert(ph.into_iter()),
                Err(e) => return Some(Err(e)),
            },
        };
        match sent_phoneme_iter.next() {
            Some(sent_phonemes) => Some(self.task.process_phonemes(sent_phonemes)),
            None => None,
        }
    }
}

impl Iterator for PiperSpeechStream<Parallel> {
    type Item = PiperResult<PiperWaveSamples>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.precalculated_results.is_none() {
            let phonemes = match self.task.get_phonemes() {
                Ok(ph) => ph,
                Err(e) => return Some(Err(e)),
            };
            let calculated_result: Vec<PiperResult<PiperWaveSamples>> = phonemes
                .par_iter()
                .map(|ph| self.task.process_phonemes(ph.to_string()))
                .collect();
            self.precalculated_results = Some(calculated_result.into_iter());
        }
        match self.precalculated_results {
            Some(ref mut res_iter) => res_iter.next(),
            None => None,
        }
    }
}

impl Iterator for PiperSpeechStream<Incremental> {
    type Item = PiperResult<PiperWaveSamples>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.channel.is_none() {
            let (sender, receiver) = crossbeam_channel::unbounded();
            let ch = Channel { sender: sender, receiver: receiver};
            self.channel = Some(ch);
        }
        let sent_phoneme_iter = match self.sentence_phonemes {
            Some(ref mut ph) => ph,
            None => match self.task.get_phonemes() {
                Ok(ph) => self.sentence_phonemes.insert(ph.into_iter()),
                Err(e) => return Some(Err(e)),
            },
        };
        match sent_phoneme_iter.next() {
            Some(item) => {
                match self.channel {
                    Some(ref ch) => Some(ch.receiver.recv().unwrap().result.into_inner().unwrap()),
                    None => None
                }
            },
            None => None
        }
    }
}


struct SpeechSynthesisResult {
    task: Arc<PiperSpeechSynthesisTask>,
    phonemes: String,
    result: OnceCell<PiperResult<PiperWaveSamples>>
}

impl SpeechSynthesisResult {
    fn new(task: Arc<PiperSpeechSynthesisTask>, phonemes: String) -> Self {
        Self { task, phonemes, result: OnceCell::new() }
    }
    fn generate(&self)  {
        self.result.set(self.task.process_phonemes(self.phonemes.clone()));
    }
}


struct Channel {
    sender: crossbeam_channel::Sender<SpeechSynthesisResult>,
    receiver: crossbeam_channel::Receiver<SpeechSynthesisResult>
}