//! Higher-level shared domain types.

use crate::fingerprint::{Digest, SchemaFingerprint};
use crate::ids::{
    ArchitectureId, CanonicalParameterName, CodecId, ParameterSlotId, TiedParameterGroup,
};
use crate::scalar::{Endianness, ScalarType, StorageElementType, TensorOrder};
use crate::shape::{ByteRange, Range64, Shape, ShapeProfile};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Arbitrary string metadata map with stable ordering.
pub type MetadataMap = BTreeMap<String, serde_json::Value>;

/// Token identifier used by the public runtime API.
pub type TokenId = u32;

/// Source file referenced by a checkpoint catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceFile {
    pub path: PathBuf,
    pub size_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_digest: Option<Digest>,
}

/// Role of a logical parameter in the architecture.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterRole {
    Embedding,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
    FfnGate,
    FfnUp,
    FfnDown,
    Norm,
    Output,
    Bias,
    RopeFreqs,
    Other(String),
}

impl ParameterRole {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Embedding => "embedding",
            Self::AttentionQ => "attention.q",
            Self::AttentionK => "attention.k",
            Self::AttentionV => "attention.v",
            Self::AttentionO => "attention.o",
            Self::FfnGate => "ffn.gate",
            Self::FfnUp => "ffn.up",
            Self::FfnDown => "ffn.down",
            Self::Norm => "norm",
            Self::Output => "output",
            Self::Bias => "bias",
            Self::RopeFreqs => "rope.freqs",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// Logical tensor type visible to the architecture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalTensorType {
    pub shape: Shape,
    pub element_type: ScalarType,
}

/// Constraint used when declaring architecture parameter slots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalTensorConstraint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<Shape>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub element_types: Vec<ScalarType>,
}

/// Zero-point behavior for group quantization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ZeroPointMode {
    None,
    Symmetric,
    Asymmetric { zero_point_type: ScalarType },
    Constant { value: i32 },
}

/// Physical storage encoding for a logical parameter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PhysicalEncoding {
    Plain {
        storage_type: ScalarType,
        order: TensorOrder,
    },
    GroupQuantized {
        logical_type: ScalarType,
        storage_bits: u8,
        signed: bool,
        axis: i32,
        group_size: u32,
        scale_type: ScalarType,
        zero_point: ZeroPointMode,
        packing: String,
    },
    BlockQuantized {
        logical_type: ScalarType,
        block_shape: Vec<u32>,
        codec: CodecId,
        codec_version: u32,
        components: Vec<String>,
    },
    Sparse {
        logical_type: ScalarType,
        format: String,
        block_shape: Vec<u32>,
    },
    Opaque {
        codec: CodecId,
        codec_version: u32,
        descriptor: serde_json::Value,
    },
}

impl PhysicalEncoding {
    pub fn plain(storage_type: ScalarType) -> Self {
        Self::Plain {
            storage_type,
            order: TensorOrder::RowMajor,
        }
    }

    pub fn gguf_q4_0() -> Self {
        Self::BlockQuantized {
            logical_type: ScalarType::F16,
            block_shape: vec![32],
            codec: CodecId::new("gguf.q4_0"),
            codec_version: 1,
            components: vec!["scale_f16".into(), "qs_u4".into()],
        }
    }

    pub fn is_supported_v1(&self) -> bool {
        match self {
            Self::Plain { .. } => true,
            Self::BlockQuantized { codec, .. } => codec.as_str() == "gguf.q4_0",
            Self::Opaque { .. } | Self::GroupQuantized { .. } | Self::Sparse { .. } => false,
        }
    }
}

/// One storage component of a logical parameter (weights, scales, …).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageComponent {
    pub name: String,
    pub key: String,
    pub shape: Shape,
    pub storage_type: StorageElementType,
    pub byte_ranges: Vec<ByteRange>,
    pub alignment: u64,
    pub endianness: Endianness,
}

/// Architecture-declared parameter slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterSlot {
    pub id: ParameterSlotId,
    pub canonical_name: CanonicalParameterName,
    pub role: ParameterRole,
    pub expected_type: LogicalTensorConstraint,
    #[serde(default)]
    pub supported_encodings: Vec<String>,
    pub optional: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tied_group: Option<TiedParameterGroup>,
}

/// How a binding maps architecture slots onto checkpoint storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BindingTransform {
    Identity,
    Rename,
    Reshape { shape: Vec<u64> },
    LogicalTranspose { permutation: Vec<u32> },
    Slice { ranges: Vec<Range64> },
    Concatenate { axis: u32 },
    Split { axis: u32, segments: Vec<u64> },
    Permute { permutation: Vec<u32> },
    Alias,
    Repack { target_encoding: PhysicalEncoding },
}

/// Materialization policy for bound parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterializationPolicy {
    DirectView,
    CopyAligned,
    DecodeOnTheFly,
    PrepackToCache,
}

/// One bound parameter association.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterBinding {
    pub slot_id: ParameterSlotId,
    pub canonical_name: CanonicalParameterName,
    pub checkpoint_keys: Vec<String>,
    pub encoding: PhysicalEncoding,
    pub logical_shape: Shape,
    pub logical_type: ScalarType,
    pub transform: BindingTransform,
    pub materialization: MaterializationPolicy,
    pub scope: String,
    pub parameter_key: String,
    pub storage_bytes: u64,
    pub alignment: u64,
}

/// Request to materialize derived parameter bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationRequest {
    pub slot_id: ParameterSlotId,
    pub policy: MaterializationPolicy,
    pub reason: String,
}

/// Complete binding plan produced by the binder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BindingPlan {
    pub architecture_id: ArchitectureId,
    pub checkpoint_schema: SchemaFingerprint,
    pub bindings: Vec<ParameterBinding>,
    pub unresolved_optional_slots: Vec<ParameterSlotId>,
    pub materializations: Vec<MaterializationRequest>,
}

/// Hardware / execution target profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetProfile {
    pub driver: String,
    pub device_id: Option<u32>,
    pub triple: Option<String>,
    pub features: Vec<String>,
    pub capability_fingerprint: Digest,
}

impl TargetProfile {
    /// Default AMDGPU arch when `--target rocm` / `hip` omits an explicit chip.
    pub const DEFAULT_ROCM_TARGET: &'static str = "gfx1151";

    pub fn llvm_cpu_host() -> Self {
        let triple = Some(host_triple().to_string());
        let features: Vec<String> = Vec::new();
        let capability_fingerprint = Digest::from_bytes(
            format!("llvm-cpu|{}|{}", triple.as_deref().unwrap_or("unknown"), features.join(","))
                .as_bytes(),
        );
        Self {
            driver: "local-task".into(),
            device_id: Some(0),
            triple,
            features,
            capability_fingerprint,
        }
    }

    /// Vulkan SPIR-V target. `chip` is an IREE Vulkan arch (e.g. `gfx1151`,
    /// `rdna3`); empty uses [`Self::DEFAULT_ROCM_TARGET`] (same AMDGPU SKUs).
    pub fn vulkan(chip: &str) -> Self {
        let chip = if chip.is_empty() {
            Self::DEFAULT_ROCM_TARGET
        } else {
            chip
        };
        Self {
            driver: "vulkan".into(),
            device_id: Some(0),
            triple: Some(chip.to_string()),
            features: vec!["spirv".into(), format!("vulkan-target={chip}")],
            // Include promote-bf16 so cache keys diverge from pre-promote VMFBs.
            capability_fingerprint: Digest::from_bytes(
                format!("vulkan|{chip}|promote-bf16-f32").as_bytes(),
            ),
        }
    }

    /// Vulkan with the default desktop AMDGPU arch (not Android baseline).
    pub fn vulkan_generic() -> Self {
        Self::vulkan(Self::DEFAULT_ROCM_TARGET)
    }

    /// Default NVPTX arch when `--target cuda` omits an explicit SM.
    pub const DEFAULT_CUDA_TARGET: &'static str = "sm_80";

    /// HIP / ROCm target. `chip` is an LLVM AMDGPU target (e.g. `gfx1151`).
    pub fn hip_rocm(chip: &str) -> Self {
        let chip = if chip.is_empty() {
            Self::DEFAULT_ROCM_TARGET
        } else {
            chip
        };
        Self {
            driver: "hip".into(),
            device_id: Some(0),
            triple: Some(chip.to_string()),
            features: vec![format!("rocm-target={chip}")],
            capability_fingerprint: Digest::from_bytes(format!("hip|{chip}").as_bytes()),
        }
    }

    /// CUDA target. `arch` is an NVPTX target (e.g. `sm_80`).
    pub fn cuda(arch: &str) -> Self {
        let arch = if arch.is_empty() {
            Self::DEFAULT_CUDA_TARGET
        } else {
            arch
        };
        Self {
            driver: "cuda".into(),
            device_id: Some(0),
            triple: Some(arch.to_string()),
            features: vec![format!("cuda-target={arch}")],
            capability_fingerprint: Digest::from_bytes(format!("cuda|{arch}").as_bytes()),
        }
    }

    /// ROCm chip / SKU for `--iree-rocm-target`, if this is a HIP profile.
    pub fn rocm_target(&self) -> Option<&str> {
        if self.driver == "hip" || self.driver == "rocm" {
            self.triple.as_deref().or(Some(Self::DEFAULT_ROCM_TARGET))
        } else {
            None
        }
    }

    /// CUDA SM arch for `--iree-cuda-target`, if this is a CUDA profile.
    pub fn cuda_target(&self) -> Option<&str> {
        if self.driver == "cuda" {
            self.triple.as_deref().or(Some(Self::DEFAULT_CUDA_TARGET))
        } else {
            None
        }
    }

    /// Vulkan GPU arch for `--iree-vulkan-target`, if this is a Vulkan profile.
    pub fn vulkan_target(&self) -> Option<&str> {
        if self.driver == "vulkan" {
            self.triple.as_deref().or(Some(Self::DEFAULT_ROCM_TARGET))
        } else {
            None
        }
    }

    /// GPU arch string passed to IREE compile (`gfx*` / `sm_*`), if any.
    pub fn gpu_compile_arch(&self) -> Option<&str> {
        self.rocm_target()
            .or_else(|| self.cuda_target())
            .or_else(|| self.vulkan_target())
    }
}

fn host_triple() -> &'static str {
    if cfg!(all(target_arch = "x86_64", target_os = "linux")) {
        "x86_64-unknown-linux-gnu"
    } else if cfg!(all(target_arch = "aarch64", target_os = "linux")) {
        "aarch64-unknown-linux-gnu"
    } else if cfg!(all(target_arch = "x86_64", target_os = "macos")) {
        "x86_64-apple-darwin"
    } else if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        "aarch64-apple-darwin"
    } else {
        "unknown"
    }
}

/// KV-cache layout descriptor from the executable manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheLayout {
    LayersHeadsSeqDim,
    LayersSeqHeadsDim,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheDescriptor {
    pub layer_count: u32,
    pub max_batch_size: u32,
    pub max_sequence_length: u32,
    pub kv_head_count: u32,
    pub head_dimension: u32,
    pub element_type: ScalarType,
    pub layout: KvCacheLayout,
    pub alignment: u64,
}

/// Model metadata exposed by the runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub architecture_id: ArchitectureId,
    pub architecture_revision: String,
    pub vocabulary_size: u32,
    pub context_length: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub hidden_size: u32,
    #[serde(default)]
    pub extra: MetadataMap,
}

/// Session configuration for inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionConfig {
    pub max_sequence_length: u32,
    pub batch_size: u32,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_sequence_length: 2048,
            batch_size: 1,
            seed: None,
        }
    }
}

/// Executable bundle manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutableManifest {
    pub format: String,
    pub version: u32,
    pub architecture_id: ArchitectureId,
    pub architecture_revision: String,
    pub checkpoint_schema: SchemaFingerprint,
    pub target: TargetProfile,
    pub shape_profile: ShapeProfile,
    pub entrypoints: Vec<String>,
    pub kv_cache: KvCacheDescriptor,
    pub parameter_scope: String,
    pub vmfb_path: String,
    /// Static `@prefill` token window compiled into the VMFB.
    #[serde(default = "default_prefill_window")]
    pub prefill_window: u32,
    #[serde(default)]
    pub diagnostics: Vec<String>,
}

fn default_prefill_window() -> u32 {
    4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_encoding_q4_0_is_supported() {
        assert!(PhysicalEncoding::gguf_q4_0().is_supported_v1());
        assert!(PhysicalEncoding::plain(ScalarType::Bf16).is_supported_v1());
    }

    #[test]
    fn target_profile_roundtrips() {
        let t = TargetProfile::llvm_cpu_host();
        let json = serde_json::to_string(&t).unwrap();
        let back: TargetProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(t.driver, back.driver);
    }
}
