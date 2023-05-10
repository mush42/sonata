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
        let mut out_buf: Vec<i16> = vec![0i16; audio.len() * 2];
        let input_buf = unsafe { std::slice::from_raw_parts_mut(audio.as_mut_ptr(), audio.len()) };
        unsafe {
            let stream = sonic_sys::sonicCreateStream(sample_rate as i32, 1);
            sonic_sys::sonicSetSpeed(stream, utils::percent_to_param(self.rate, 0.0f32, 4.0f32));
            sonic_sys::sonicSetVolume(stream, utils::percent_to_param(self.volume, 0.0f32, 2.0f32));
            sonic_sys::sonicSetPitch(stream, utils::percent_to_param(self.pitch, 0.0f32, 2.0f32));
            sonic_sys::sonicWriteShortToStream(stream, input_buf.as_ptr(), input_buf.len() as i32/ sample_rate as i32);
            sonic_sys::sonicFlushStream(stream);
            let num_samples = sonic_sys::sonicSamplesAvailable(stream);
            sonic_sys::sonicReadShortFromStream(stream, out_buf.as_mut_ptr(), num_samples);
            sonic_sys::sonicDestroyStream(stream);
            out_buf.set_len(num_samples as usize * sample_rate as usize);
        }
        Ok(out_buf)
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
