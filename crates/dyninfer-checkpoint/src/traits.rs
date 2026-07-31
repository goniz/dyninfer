//! Container reader and convention decoder traits.

use crate::catalog::{
    DecodeContext, MatchScore, ParameterCatalog, ProbeScore, RawCheckpointIndex, RuntimeProviderPlan,
};
use crate::limits::InspectionLimits;
use crate::source::RandomAccessSource;
use dyninfer_core::{ContainerFormatId, ConventionId};
use dyninfer_error::Result;
use std::sync::Arc;

/// Reads a checkpoint container's metadata index.
pub trait CheckpointContainerReader: Send + Sync {
    fn format_id(&self) -> ContainerFormatId;

    fn probe(&self, source: &dyn RandomAccessSource) -> Result<ProbeScore>;

    fn index(
        &self,
        source: Arc<dyn RandomAccessSource>,
        limits: &InspectionLimits,
    ) -> Result<RawCheckpointIndex>;

    fn runtime_provider_plan(&self, index: &RawCheckpointIndex) -> Result<RuntimeProviderPlan>;
}

/// Decodes container-specific encodings into logical parameters.
pub trait CheckpointConventionDecoder: Send + Sync {
    fn convention_id(&self) -> ConventionId;

    fn match_score(
        &self,
        index: &RawCheckpointIndex,
        context: &DecodeContext,
    ) -> Result<MatchScore>;

    fn decode(
        &self,
        index: &RawCheckpointIndex,
        context: &DecodeContext,
    ) -> Result<ParameterCatalog>;
}

/// Optional materializer for derived parameter artifacts.
pub trait ParameterMaterializer: Send + Sync {
    fn id(&self) -> &str;
    fn can_materialize(&self, encoding_name: &str) -> bool;
}
