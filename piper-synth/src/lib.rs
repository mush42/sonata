mod utils;

use piper_model::{PiperModel, PiperResult, SynthesisConfig};
use sonic_sys;
use std::path::PathBuf;

pub const DEFAULT_RATE: u8 = 50;
pub const DEFAULT_VOLUME: u8 = 75;
pub const DEFAULT_PITCH: u8 = 50;


pub struct AudioOutputConfig {
    volume: u8,
    rate: u8,
    pitch: u8,
}

impl AudioOutputConfig {
    pub fn new(volume: Option<u8>, rate: Option<u8>, pitch: Option<u8>) -> Self {
        Self {
            volume: volume.unwrap_or(DEFAULT_VOLUME),
            rate: rate.unwrap_or(DEFAULT_RATE),
            pitch: pitch.unwrap_or(DEFAULT_PITCH),
        }
    }

    fn apply(&self, mut audio: Vec<i16>, sample_rate: u32) -> PiperResult<Vec<i16>> {
        audio.reserve(audio.len() * 2);
        unsafe {
            sonic_sys::sonicChangeShortSpeed(
                audio.as_mut_ptr(),
                audio.len() as i32 / sample_rate as i32,
                utils::percent_to_param(self.rate, 0.0f32, 4.0f32),
                utils::percent_to_param(self.pitch, 0.0f32, 2.0f32),
                1.0,
                utils::percent_to_param(self.volume, 0.0f32, 1.0f32),
                0,
                sample_rate as i32,
                1,
            );
        }
        Ok(audio)
    }
}

pub struct PiperSpeechSynthesizer(PiperModel);

impl PiperSpeechSynthesizer {
    pub fn new(config_path: PathBuf, onnx_path: PathBuf) -> PiperResult<Self> {
        Ok(Self(PiperModel::new(config_path, onnx_path)?))
    }

    pub fn synthesize(
        &self,
        text: &str,
        synth_config: Option<SynthesisConfig>,
        output_config: Option<AudioOutputConfig>,
    ) -> PiperResult<Vec<i16>> {
        let audio = self.0.generate_speech(text, synth_config)?;
        if let Some(cfg) = output_config {
            cfg.apply(audio, self.0.config.audio.sample_rate)
        } else {
            Ok(audio)
        }
    }

    pub fn info(&self) -> PiperResult<Vec<String>> {
        self.0.info()
    }
}
