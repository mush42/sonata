/*
git clone --recursive https://github.com/mush42/sonata
cd sonata

wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2
tar xf espeak-ng-data.tar.bz2
export ESPEAK_DATA_PATH=$(pwd)/espeak-ng-data

cargo run en_US-hfc_female-medium.onnx.json
*/

use rodio::buffer::SamplesBuffer;
use sonata_synth::SonataSpeechSynthesizer;
use std::path::Path;

fn init_ort_environment() {
    ort::init()
        .with_name("sonata")
        .with_execution_providers([ort::ExecutionProviderDispatch::CPU(Default::default())])
        .commit()
        .expect("Failed to initialize onnxruntime");
}

fn main() {
    init_ort_environment();
    let config_path = std::env::args().nth(1).expect("Please specify config path");
    let text = "Hello! i'm playing audio from memory directly.".to_string();

    let voice = sonata_piper::from_config_path(Path::new(&config_path)).unwrap();
    let synth = SonataSpeechSynthesizer::new(voice).unwrap();
    let mut samples: Vec<f32> = Vec::new();
    let audio = synth.synthesize_parallel(text,  None).unwrap();
    for result in audio {
        samples.append(&mut result.unwrap().into_vec());
    }
        
    let (_stream, handle) = rodio::OutputStream::try_default().unwrap();
    let sink = rodio::Sink::try_new(&handle).unwrap();

    let buf = SamplesBuffer::new(1, 22050, samples);
    sink.append(buf);

    sink.sleep_until_end();
}
