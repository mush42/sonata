use crate::{InferenceEngine, LibtashkeelError, LibtashkeelResult, CHAR_LIMIT};
use bytes::{Buf, Bytes};
use std::path::Path;
use tract_onnx::prelude::*;

impl From<tract_data::anyhow::Error> for LibtashkeelError {
    fn from(other: tract_data::anyhow::Error) -> Self {
        LibtashkeelError::InferenceError(format!(
            "Failed to run model using Tract. Caused by {:#}",
            other
        ))
    }
}

type TractModelType = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;
const MODEL_BYTES: &[u8] = include_bytes!("../../data/tract/model.onnx");

pub struct TractEngine(TractModelType);

impl TractEngine {
    pub fn from_bytes(model_bytes: &'static [u8]) -> LibtashkeelResult<Self> {
        let model_bytes = Bytes::from_static(model_bytes);
        let model = tract_onnx::onnx()
            .model_for_read(&mut model_bytes.reader())?
            .into_optimized()?
            .into_compact()?
            .into_runnable()?;
        Ok(Self(model))
    }
    pub fn from_path(model_path: impl AsRef<Path>) -> LibtashkeelResult<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .into_optimized()?
            .into_compact()?
            .into_runnable()?;
        Ok(Self(model))
    }
    pub fn with_bundled_model() -> LibtashkeelResult<Self> {
        Self::from_bytes(MODEL_BYTES)
    }
}

impl InferenceEngine for TractEngine {
    fn infer(
        &self,
        mut input_ids: Vec<i64>,
        seq_length: usize,
    ) -> LibtashkeelResult<(Vec<u8>, Vec<f32>)> {
        input_ids.resize(CHAR_LIMIT, 0);

        let input = tract_ndarray::Array1::from_iter(input_ids).into_tensor();
        let input_length = tract_ndarray::Array1::from_iter([seq_length as i64]).into_tensor();

        let (target_ids, logits): (Vec<u8>, Vec<f32>) = {
            let result = &mut self.0.run(tvec![input.into(), input_length.into()])?;
            let logits = result.pop().unwrap();
            let preds = result.pop().unwrap();
            let preds = Vec::from_iter(preds.as_slice::<u8>()?.iter().take(seq_length).cloned());
            let logits = Vec::from_iter(logits.as_slice::<f32>()?.iter().take(seq_length).cloned());
            (preds, logits)
        };

        Ok((target_ids, logits))
    }
}
