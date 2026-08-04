//! Stable shared types used across compiler orchestration and runtime manifests.
//!
//! This crate has no IREE or MLIR dependencies.

#![forbid(unsafe_code)]

mod fingerprint;
mod ids;
mod ir;
mod scalar;
mod shape;
mod types;

pub use fingerprint::{Digest, SchemaFingerprint, content_digest};
pub use ids::{
    ArchitectureId, CanonicalParameterName, CodecId, ContainerFormatId, ConventionId, EncodingId,
    GraphValueId, KernelId, LoweringId, OperationId, ParameterSlotId, TiedParameterGroup,
};
pub use ir::{
    ArchitectureExport, ArchitectureGraph, ArchitectureOperation, BoundModel, ElementwiseFunction,
    ExecutionMode, GraphValue, KvCacheComponent, ModelInputKind, OperationKind, PrecisionPolicy,
    SelectedKernel, SemanticElementType, SemanticTensorType, SpecializedExecutionShape,
    TensorDimension,
};
pub use scalar::{Endianness, ScalarType, StorageElementType, TensorOrder};
pub use shape::{ByteRange, Range64, Shape, ShapeProfile};
pub use types::{
    BindingPlan, BindingTransform, BlockLayoutField, ExecutableManifest, KvCacheDescriptor,
    KvCacheLayout, KvCacheStorage, LogicalTensorConstraint, LogicalTensorType,
    ManifestParameterComponent, MetadataMap, ModelMetadata, ParameterBinding,
    ParameterComponentBinding, ParameterRole, ParameterSlot, PhysicalEncoding, SessionConfig,
    SourceFile, StorageComponent, TargetProfile, TokenId, VULKAN_LDS_CLAMP_TOOL, ZeroPointMode,
    vulkan_executable_flags,
};
