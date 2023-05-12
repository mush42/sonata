mod utils;

use piper_model::{PiperError, PiperModel, PiperResult, PiperWaveSamples, SynthesisConfig};
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
    ) -> PiperSpeechStream {
        PiperSpeechStream::<Lazy>::new(Arc::clone(&self.0), text, synth_config, output_config)
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


pub struct Lazy;
pub struct Parallel;
pub struct Batched;


pub struct PiperSpeechStream<Mode = Lazy> {
    model: Arc<PiperModel>,
    text: String,
    synth_config: Option<SynthesisConfig>,
    output_config: Option<AudioOutputConfig>,
    sentence_phonemes: Option<std::vec::IntoIter<String>>,
    mode: std::marker::PhantomData<Mode>,
}



impl<Mode> PiperSpeechStream<Mode>{
    fn new(
        model: Arc<PiperModel>,
        text: String,
        synth_config: Option<SynthesisConfig>,
        output_config: Option<AudioOutputConfig>,
    ) -> Self{
        Self {
            model,
            text,
            synth_config,
            output_config,
            sentence_phonemes: None,
            mode: std::marker::PhantomData
        }
    }

    pub fn get_sample_rate(&self) -> u32 {
        self.model.config.audio.sample_rate
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


impl Iterator for PiperSpeechStream<Lazy> {
    type Item = PiperResult<PiperWaveSamples>;

    fn next(&mut self) -> Option<Self::Item> {
        let sent_phonemes = match self.sentence_phonemes {
            Some(ref mut ph) => ph,
            None => match self.model.phonemize_text(&self.text) {
                Ok(ph) => self.sentence_phonemes.insert(ph.to_vec().into_iter()),
                Err(e) => return Some(Err(e)),
            },
        };
        match sent_phonemes.next() {
            Some(sent_phonemes) => Some(self.process_phonemes(sent_phonemes)),
            None => None,
        }
    }
}




// ====================
impl Iterator for PiperSpeechStream<Parallel> {
    type Item = PiperResult<PiperWaveSamples>;

    fn next(&mut self) -> Option<Self::Item> {
        let sent_phonemes = match self.sentence_phonemes {
            Some(ref mut ph) => ph,
            None => match self.model.phonemize_text(&self.text) {
                Ok(ph) => self.sentence_phonemes.insert(ph.to_vec().into_iter()),
                Err(e) => return Some(Err(e)),
            },
        };
        match sent_phonemes.next() {
            Some(sent_phonemes) => Some(self.process_phonemes(sent_phonemes)),
            None => None,
        }
    }
}





