use crate::{InferenceEngine, LibtashkeelResult};
use std::path::PathBuf;

pub struct DynamicInferenceEngine(Box<dyn InferenceEngine + Send + Sync>);

impl DynamicInferenceEngine {
    pub fn new(engine: Box<dyn InferenceEngine + Send + Sync>) -> Self {
        Self(engine)
    }
}

impl InferenceEngine for DynamicInferenceEngine {
    fn infer(
        &self,
        input_ids: Vec<i64>,
        seq_length: usize,
    ) -> LibtashkeelResult<(Vec<u8>, Vec<f32>)> {
        self.0.infer(input_ids, seq_length)
    }
}

#[cfg(feature = "tract")]
mod tract;

#[cfg(feature = "tract")]
pub fn create_inference_engine(
    model_path: Option<PathBuf>,
) -> LibtashkeelResult<DynamicInferenceEngine> {
    use self::tract::TractEngine;

    log::info!("Built with `Tract` inference backend.");

    match model_path {
        Some(path) => {
            log::info!("Loading model from path: `{}`", path.display());
            let engine = TractEngine::from_path(&path)?;
            Ok(DynamicInferenceEngine::new(Box::new(engine)))
        }
        None => {
            log::info!("Using bundled model");
            let engine = TractEngine::with_bundled_model()?;
            Ok(DynamicInferenceEngine::new(Box::new(engine)))
        }
    }
}

#[cfg(feature = "ort")]
mod ort;

#[cfg(feature = "ort")]
pub fn create_inference_engine(
    model_path: Option<PathBuf>,
) -> LibtashkeelResult<DynamicInferenceEngine> {
    use self::ort::{OrtEngineWithModelBytes, OrtEngineWithModelPath};

    log::info!("Built with `ORT` inference backend.");

    match model_path {
        Some(path) => {
            log::info!("Loading model from path: `{}`", path.display());
            let engine = OrtEngineWithModelPath::from_path(&path)?;
            Ok(DynamicInferenceEngine::new(Box::new(engine)))
        }
        None => {
            log::info!("Using bundled model");
            let engine = OrtEngineWithModelBytes::with_bundled_model()?;
            Ok(DynamicInferenceEngine::new(Box::new(engine)))
        }
    }
}
