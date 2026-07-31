//! Deferred TensorFlow / NPZ support (Milestone 5).
//!
//! Present so the workspace layout matches the specification; probing always
//! returns no-match until NPZ/MLX readers are implemented.

#![forbid(unsafe_code)]

use dyninfer_checkpoint::{
    BuiltinCheckpointSupport, CheckpointContainerReader, InspectionLimits, ProbeScore,
    RandomAccessSource, RawCheckpointIndex, RuntimeProviderPlan,
};
use dyninfer_core::ContainerFormatId;
use dyninfer_error::{DynInferError, Result, UnsupportedContainerError};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct TensorflowContainer;

impl CheckpointContainerReader for TensorflowContainer {
    fn format_id(&self) -> ContainerFormatId {
        ContainerFormatId::new("tensorflow-npz")
    }

    fn probe(&self, _source: &dyn RandomAccessSource) -> Result<ProbeScore> {
        Ok(ProbeScore::NONE)
    }

    fn index(
        &self,
        _source: Arc<dyn RandomAccessSource>,
        _limits: &InspectionLimits,
    ) -> Result<RawCheckpointIndex> {
        Err(DynInferError::UnsupportedContainer(UnsupportedContainerError {
            message: "TensorFlow/NPZ containers are deferred past version 1".into(),
            path: None,
            probed_formats: vec!["tensorflow-npz".into()],
        }))
    }

    fn runtime_provider_plan(&self, _index: &RawCheckpointIndex) -> Result<RuntimeProviderPlan> {
        Err(DynInferError::internal("tensorflow provider not implemented"))
    }
}

pub fn register(support: &mut BuiltinCheckpointSupport) {
    support.register_container(TensorflowContainer::default());
}
