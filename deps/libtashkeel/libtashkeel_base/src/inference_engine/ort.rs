use crate::{InferenceEngine, LibtashkeelError, LibtashkeelResult};
use ndarray::{Array1, Array2, CowArray};
use once_cell::sync::Lazy;
use ort::{tensor::OrtOwnedTensor, GraphOptimizationLevel, SessionBuilder, Value};
use std::path::Path;
use std::sync::Arc;

impl From<ort::OrtError> for LibtashkeelError {
    fn from(other: ort::OrtError) -> Self {
        LibtashkeelError::InferenceError(format!(
            "Failed to run model using onnxruntime via ort. Caused by {}",
            other
        ))
    }
}

fn ort_session_run(
    session: &ort::Session,
    input_ids: Vec<i64>,
    seq_length: usize,
) -> LibtashkeelResult<(Vec<u8>, Vec<f32>)> {
    let input_ids =
        CowArray::from(Array2::<i64>::from_shape_vec((1, seq_length), input_ids).unwrap())
            .into_dyn();
    let input_length = CowArray::from(Array1::<i64>::from_iter([seq_length as i64])).into_dyn();

    let (target_ids, logits): (Vec<u8>, Vec<f32>) = {
        let inputs = vec![
            Value::from_array(session.allocator(), &input_ids).unwrap(),
            Value::from_array(session.allocator(), &input_length).unwrap(),
        ];
        let outputs = session.run(inputs)?;
        let target_ids: OrtOwnedTensor<u8, _> = outputs[0].try_extract()?;
        let logits: OrtOwnedTensor<f32, _> = outputs[1].try_extract()?;
        let target_ids_vec = Vec::from_iter(target_ids.view().iter().copied());
        let logits_vec = Vec::from_iter(logits.view().iter().copied());
        (target_ids_vec, logits_vec)
    };

    Ok((target_ids, logits))
}

const MODEL_BYTES: &[u8] = include_bytes!("../../data/ort/model.onnx");
static ORT_ENV: Lazy<Arc<ort::Environment>> = Lazy::new(|| {
    ort::Environment::builder()
        .with_name("libtashkeel")
        .with_execution_providers([ort::ExecutionProvider::CPU(Default::default())])
        .build()
        .unwrap()
        .into_arc()
});

pub struct OrtEngineWithModelBytes<'a>(ort::InMemorySession<'a>);

impl<'a> OrtEngineWithModelBytes<'a> {
    pub fn from_bytes(model_bytes: &'a [u8]) -> LibtashkeelResult<OrtEngineWithModelBytes<'a>> {
        let session = SessionBuilder::new(&ORT_ENV)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_allocator(ort::AllocatorType::Arena)?
            .with_memory_pattern(true)?
            .with_parallel_execution(true)?
            .with_inter_threads(2)?
            .with_intra_threads(2)?
            .with_model_from_memory(model_bytes)?;

        Ok(Self(session))
    }
    #[allow(dead_code)]
    pub fn from_session(
        session: ort::InMemorySession<'a>,
    ) -> LibtashkeelResult<OrtEngineWithModelBytes<'a>> {
        Ok(Self(session))
    }
    pub fn with_bundled_model() -> LibtashkeelResult<OrtEngineWithModelBytes<'a>> {
        Self::from_bytes(MODEL_BYTES)
    }
}

impl InferenceEngine for OrtEngineWithModelBytes<'_> {
    fn infer(
        &self,
        input_ids: Vec<i64>,
        seq_length: usize,
    ) -> LibtashkeelResult<(Vec<u8>, Vec<f32>)> {
        ort_session_run(&self.0, input_ids, seq_length)
    }
}

pub struct OrtEngineWithModelPath(ort::Session);

impl OrtEngineWithModelPath {
    pub fn from_path(model_path: impl AsRef<Path>) -> LibtashkeelResult<Self> {
        let session = SessionBuilder::new(&ORT_ENV)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_allocator(ort::AllocatorType::Arena)?
            .with_memory_pattern(true)?
            .with_parallel_execution(true)?
            .with_inter_threads(2)?
            .with_intra_threads(2)?
            .with_model_from_file(model_path)?;

        Ok(Self(session))
    }
    #[allow(dead_code)]
    pub fn from_session(session: ort::Session) -> LibtashkeelResult<Self> {
        Ok(Self(session))
    }
}

impl InferenceEngine for OrtEngineWithModelPath {
    fn infer(
        &self,
        input_ids: Vec<i64>,
        seq_length: usize,
    ) -> LibtashkeelResult<(Vec<u8>, Vec<f32>)> {
        ort_session_run(&self.0, input_ids, seq_length)
    }
}
