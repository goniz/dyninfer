//! Stable shared types used across compiler orchestration and runtime manifests.
//!
//! This crate has no IREE or MLIR dependencies.

#![forbid(unsafe_code)]

mod fingerprint;
mod ids;
mod scalar;
mod shape;
mod types;

pub use fingerprint::{content_digest, Digest, SchemaFingerprint};
pub use ids::{
    ArchitectureId, CanonicalParameterName, CodecId, ContainerFormatId, ConventionId,
    ParameterSlotId, TiedParameterGroup,
};
pub use scalar::{Endianness, ScalarType, StorageElementType, TensorOrder};
pub use shape::{ByteRange, Range64, Shape, ShapeProfile};
pub use types::{
    BindingPlan, BindingTransform, ExecutableManifest, KvCacheDescriptor, KvCacheLayout,
    LogicalTensorConstraint, LogicalTensorType, MaterializationPolicy, MaterializationRequest,
    MetadataMap, ModelMetadata, ParameterBinding, ParameterRole, ParameterSlot, PhysicalEncoding,
    SessionConfig, SourceFile, StorageComponent, TargetProfile, TokenId, ZeroPointMode,
};
