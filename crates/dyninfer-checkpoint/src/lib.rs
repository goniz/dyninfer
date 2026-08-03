//! Checkpoint container/convention traits and catalog model.
//!
//! Format-specific readers live in sibling crates and are registered into
//! [`BuiltinCheckpointSupport`] at compile time.

#![forbid(unsafe_code)]

mod catalog;
mod fingerprint;
mod limits;
mod role;
mod source;
mod support;
mod traits;

pub use catalog::{
    CheckpointCatalog, ContainerIdentity, DecodeContext, LogicalParameter, MatchScore,
    ParameterCatalog, ProbeScore, RawCheckpointIndex, RawTensorEntry, RuntimeProviderPlan,
};
pub use fingerprint::schema_fingerprint_from_parameters;
pub use limits::InspectionLimits;
pub use role::infer_role;
pub use source::{BytesSource, FileSource, RandomAccessSource};
pub use support::{BuiltinCheckpointSupport, empty_support};
pub use traits::{CheckpointContainerReader, CheckpointConventionDecoder, ParameterMaterializer};
