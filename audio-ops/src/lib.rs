mod samples;
mod wave_writer;

pub use samples::{AudioInfo, AudioSamples, Audio};
pub use wave_writer::{write_wave_samples_to_buffer, write_wave_samples_to_file, WaveWriterError};
