mod utils;

use piper_model::{PiperModel, PiperResult, PiperError, PiperWaveSamples, SynthesisConfig};
use sonic_sys;
use std::path::PathBuf;

const RATE_RANGE: (f32, f32) = (0.0f32, 5.0f32);
const VOLUME_RANGE: (f32, f32) = (0.1f32, 1.9f32);
const PITCH_RANGE: (f32, f32) = (0.5f32, 1.5f32);

const DEFAULT_RATE: u8 = 20;
const DEFAULT_VOLUME: u8 = 75;
const DEFAULT_PITCH: u8 = 50;


pub struct AudioOutputConfig {
    rate: u8,
    volume: u8,
    pitch: u8,
}

impl AudioOutputConfig {
    pub fn new(rate: Option<u8>, volume: Option<u8>, pitch: Option<u8>) -> Self {
        Self {
            rate: rate.unwrap_or(DEFAULT_RATE),
            volume: volume.unwrap_or(DEFAULT_VOLUME),
            pitch: pitch.unwrap_or(DEFAULT_PITCH),
        }
    }

    fn apply(&self, audio: Vec<i16>, sample_rate: u32) -> PiperResult<Vec<i16>> {
        if audio.len() == 0 {
            return Ok(audio);
        }
        let (rate, volume, pitch) = (
            utils::percent_to_param(self.rate, RATE_RANGE.0, RATE_RANGE.1),
            utils::percent_to_param(self.volume, VOLUME_RANGE.0, VOLUME_RANGE.1),
            utils::percent_to_param(self.pitch, PITCH_RANGE.0, PITCH_RANGE.1),
        );
        let mut out_buf: Vec<i16> = Vec::new();
        unsafe {
            let stream = sonic_sys::sonicCreateStream(sample_rate as i32, 1);
            sonic_sys::sonicSetSpeed(stream, rate);
            sonic_sys::sonicSetVolume(stream, volume);
            sonic_sys::sonicSetPitch(stream, pitch);
            sonic_sys::sonicWriteShortToStream(
                stream,
                audio.as_ptr(),
                audio.len() as i32
            );
            sonic_sys::sonicFlushStream(stream);
            let num_samples = sonic_sys::sonicSamplesAvailable(stream);
            if num_samples <= 0 {
                return Err(
                    PiperError::OperationError("Sonic Error: failed to apply audio config. Invalid parameter value for rate, volume, or pitch".to_string())
                );
            }
            out_buf.reserve_exact(num_samples as usize);
            sonic_sys::sonicReadShortFromStream(stream, out_buf.spare_capacity_mut().as_mut_ptr().cast(), num_samples);
            sonic_sys::sonicDestroyStream(stream);
            out_buf.set_len(num_samples as usize);
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
    ) -> PiperResult<PiperWaveSamples> {
        let audio = self.0.speak_text(text, &synth_config)?;
        match output_config {
            Some(config) => Ok(config.apply(audio.into(), self.0.config.audio.sample_rate)?.into()),
            None => Ok(audio)
        }
    }

    pub fn info(&self) -> PiperResult<Vec<String>> {
        self.0.info()
    }
}
