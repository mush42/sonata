use divan::black_box;
use sonata_core::{SonataModel, SonataResult, AudioSamples};
use sonata_synth::{SonataSpeechSynthesizer, AudioOutputConfig};
use sonata_piper::from_config_path as voice_from_config_path;
use once_cell::sync::{OnceCell, Lazy};
use std::path::PathBuf;
use std::sync::Arc;


const TEXT: &'static str = "Technology is not inevitable.
Powerful drivers must exist in order for people to keep pushing the envelope and continue demanding more and more from a particular field of knowledge.
Cheaper Communications 
The first and most important driver is our demand for ever cheaper and easier communications.
All of human society depends on communications.";

const CRATE_DIR: &'static str = env!("CARGO_MANIFEST_DIR");
static ORT_ENVIRONMENT: OnceCell<Arc<ort::Environment>> = OnceCell::new();
static STD_VOICE: Lazy<Arc<dyn SonataModel + Send + Sync>> = Lazy::new(|| {
    let config_path = model_directory("std").join("model.onnx.json");
    voice_from_config_path(&config_path, get_ort_environment()).unwrap()
});
static RT_VOICE: Lazy<Arc<dyn SonataModel + Send + Sync>> = Lazy::new(|| {
    let config_path = model_directory("rt").join("config.json");
    voice_from_config_path(&config_path, get_ort_environment()).unwrap()
});

fn main() {
    Lazy::force(&STD_VOICE);
    Lazy::force(&RT_VOICE);
    divan::main();
}


fn get_ort_environment() -> &'static Arc<ort::Environment> {
    ORT_ENVIRONMENT.get_or_init(|| {
        Arc::new(
            ort::Environment::builder()
                .with_name("sonata")
                .with_execution_providers([ort::ExecutionProvider::CPU(Default::default())])
                .build()
                .unwrap(),
        )
    })
}

fn model_directory(kind: &str) -> PathBuf {
    PathBuf::from(CRATE_DIR).join("benches").join("models").join(kind)
}

fn load_synth_with_std_model() -> impl Fn() -> (SonataSpeechSynthesizer, String, Option<AudioOutputConfig>) {
    move || {
        let model = Arc::clone(&STD_VOICE);
        let output_config = Some(AudioOutputConfig {
            rate: Some(50),
            volume: Some(50),
            pitch: Some(50),
            appended_silence_ms: None,
        });
        let synth = SonataSpeechSynthesizer::new(model).unwrap();
        (synth, TEXT.to_string(), output_config)
    }
}

#[inline(always)]
fn iterate_stream(stream: impl Iterator<Item = SonataResult<AudioSamples>>)  {
    for result in stream {
        if let Ok(audio) = result {
            let wav_bytes = black_box(audio.as_wave_bytes());
            wav_bytes.len();
        }
    }
}

#[divan::bench_group(sample_count=20, sample_size=5)]
mod speech_streams {
    use super::*;
    use divan::{Bencher, black_box};

    #[divan::bench(threads=4)]
    fn bench_lazy_stream(bencher: Bencher) {
        bencher
            .with_inputs(load_synth_with_std_model())
            .bench_local_refs(|(synth, text, output_config)| {
                let stream = synth.synthesize_lazy(text.clone(), output_config.clone())
                    .unwrap()
                    .map(|res| res.map(|a| a.samples));
                iterate_stream(black_box(stream));
            });
    }

    #[divan::bench(threads=4)]
    fn bench_batched_stream(bencher: Bencher) {
        bencher
            .with_inputs(load_synth_with_std_model())
            .bench_local_refs(|(synth, text, output_config)| {
                let stream = synth.synthesize_batched(text.clone(), output_config.clone(), None)
                    .unwrap()
                    .map(|res| res.map(|a| a.samples));
                iterate_stream(black_box(stream));
            });
    }

    #[divan::bench]
    fn bench_parallel_stream(bencher: Bencher) {
        bencher
            .with_inputs(load_synth_with_std_model())
            .bench_local_refs(|(synth, text, output_config)| {
                let stream = synth.synthesize_parallel(text.clone(), output_config.clone())
                    .unwrap()
                    .map(|res| res.map(|a| a.samples));
                iterate_stream(black_box(stream));
            });
    }
    #[divan::bench]
    fn bench_realtime_stream(bencher: Bencher) {
        bencher
            .with_inputs(load_synth_with_std_model())
            .bench_local_refs(|(synth, text, output_config)| {
                let stream = synth.synthesize_streamed(text.clone(), output_config.clone(), 72, 3)
                    .unwrap();
                iterate_stream(black_box(stream));
            });
    }
}
