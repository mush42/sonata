mod samples;
mod wave_writer;

pub use samples::{Audio, AudioInfo, AudioSamples};
pub use wave_writer::{write_wave_samples_to_buffer, write_wave_samples_to_file, WaveWriterError};
