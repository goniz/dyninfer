//! Higher-level shared domain types.

use crate::fingerprint::{Digest, SchemaFingerprint};
use crate::ids::{
    ArchitectureId, CanonicalParameterName, CodecId, ParameterSlotId, TiedParameterGroup,
};
use crate::scalar::{Endianness, ScalarType, StorageElementType, TensorOrder};
use crate::shape::{ByteRange, Range64, Shape, ShapeProfile};
use crate::{PrecisionPolicy, SelectedKernel};
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

/// One byte-addressed field inside a block-quantized storage block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockLayoutField {
    pub name: String,
    pub byte_offset: u32,
    pub byte_length: u32,
    pub storage_type: StorageElementType,
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
        storage_container: ScalarType,
        signed: bool,
        axis: i32,
        group_size: u32,
        scale_type: ScalarType,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bias_type: Option<ScalarType>,
        zero_point: ZeroPointMode,
        packing: String,
        order: TensorOrder,
        components: Vec<String>,
    },
    BlockQuantized {
        logical_type: ScalarType,
        block_shape: Vec<u32>,
        bytes_per_block: u32,
        codec: CodecId,
        codec_version: u32,
        components: Vec<String>,
        layout: Vec<BlockLayoutField>,
        order: TensorOrder,
        endianness: Endianness,
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
}

/// One storage component of a logical parameter (weights, scales, …).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageComponent {
    pub name: String,
    pub key: String,
    #[serde(default)]
    pub source_file_index: u32,
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
}

/// One bound physical component addressed by a stable external parameter key.
/// Checkpoint paths and byte offsets remain in the associated catalog/runtime
/// provider plan rather than entering compiler IR.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterComponentBinding {
    pub component_name: String,
    pub external_key: String,
    pub checkpoint_key: String,
    pub source_file_index: u32,
    pub shape: Shape,
    pub storage_type: StorageElementType,
    pub byte_lengths: Vec<u64>,
    pub alignment: u64,
    pub endianness: Endianness,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub components: Vec<ParameterComponentBinding>,
    pub transform: BindingTransform,
    pub scope: String,
    pub parameter_key: String,
    pub storage_bytes: u64,
    pub alignment: u64,
}

/// Complete binding plan produced by the binder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BindingPlan {
    pub architecture_id: ArchitectureId,
    pub checkpoint_schema: SchemaFingerprint,
    pub bindings: Vec<ParameterBinding>,
    pub unresolved_optional_slots: Vec<ParameterSlotId>,
}

/// Hardware / execution target profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetProfile {
    pub driver: String,
    pub device_id: Option<u32>,
    /// Host CPU triple. GPU architecture is recorded separately.
    pub triple: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    pub features: Vec<String>,
    pub capability_fingerprint: Digest,
    /// Full HAL device URI from discovery (`vulkan://GPU-…`, `hip://0`, …).
    /// When set, the runtime MUST open this device instead of the driver default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subgroup_size: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_workgroup_invocations: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub executable_target_flags: Vec<String>,
    /// True only when all facts required for local compilation were detected
    /// or, for host CPU, queried from the running process.
    #[serde(default)]
    pub verified: bool,
}

impl TargetProfile {
    pub fn llvm_cpu_host() -> Self {
        let triple = Some(host_triple().to_string());
        let features = detected_host_features();
        let mut profile = Self {
            driver: "local-task".into(),
            device_id: Some(0),
            triple,
            architecture: None,
            features,
            capability_fingerprint: Digest::from_bytes(b"pending"),
            device_uri: Some("local-task://".into()),
            device_name: Some("host-cpu".into()),
            subgroup_size: None,
            max_workgroup_invocations: None,
            executable_target_flags: vec![
                "--iree-hal-target-device=local".into(),
                "--iree-hal-local-target-device-backends=llvm-cpu".into(),
                "--iree-llvmcpu-target-cpu=host".into(),
            ],
            verified: host_triple() != "unknown",
        };
        profile.refresh_fingerprint();
        profile
    }

    /// Vulkan SPIR-V target with an exact architecture reported by discovery.
    /// An empty architecture produces an unverified profile that compilation
    /// must reject.
    pub fn vulkan(chip: &str) -> Self {
        let architecture = nonempty(chip);
        let mut profile = Self {
            driver: "vulkan".into(),
            device_id: Some(0),
            triple: None,
            architecture: architecture.clone(),
            features: architecture
                .iter()
                .map(|chip| format!("vulkan-target={chip}"))
                .chain(std::iter::once("spirv".into()))
                .collect(),
            capability_fingerprint: Digest::from_bytes(b"pending"),
            device_uri: None,
            device_name: None,
            subgroup_size: None,
            max_workgroup_invocations: None,
            executable_target_flags: architecture
                .as_ref()
                .map(|chip| {
                    vec![
                        "--iree-hal-target-device=vulkan".into(),
                        format!("--iree-vulkan-target={chip}"),
                    ]
                })
                .unwrap_or_default(),
            verified: architecture.is_some(),
        };
        profile.refresh_fingerprint();
        profile
    }

    /// HIP / ROCm target. `chip` is an LLVM AMDGPU target (e.g. `gfx1151`).
    pub fn hip_rocm(chip: &str) -> Self {
        let architecture = nonempty(chip);
        let mut profile = Self {
            driver: "hip".into(),
            device_id: Some(0),
            triple: None,
            architecture: architecture.clone(),
            features: architecture
                .iter()
                .map(|chip| format!("rocm-target={chip}"))
                .collect(),
            capability_fingerprint: Digest::from_bytes(b"pending"),
            device_uri: None,
            device_name: None,
            subgroup_size: None,
            max_workgroup_invocations: None,
            executable_target_flags: architecture
                .as_ref()
                .map(|chip| {
                    vec![
                        "--iree-hal-target-device=hip".into(),
                        format!("--iree-rocm-target={chip}"),
                    ]
                })
                .unwrap_or_default(),
            verified: architecture.is_some(),
        };
        profile.refresh_fingerprint();
        profile
    }

    /// CUDA target. `arch` is an NVPTX target (e.g. `sm_80`).
    pub fn cuda(arch: &str) -> Self {
        let architecture = nonempty(arch);
        let mut features = architecture
            .iter()
            .map(|arch| format!("cuda-target={arch}"))
            .collect::<Vec<_>>();
        if let Some(sm) = architecture.as_deref().and_then(cuda_sm) {
            if sm >= 53 {
                features.push("native_f16".into());
            }
            if sm >= 61 {
                features.push("integer_dot_product".into());
            }
            if sm >= 70 {
                features.push("tensor_cores".into());
            }
            if sm >= 80 {
                features.push("native_bf16".into());
            }
            if sm >= 100 {
                features.push("nvfp4".into());
            }
        }
        let mut profile = Self {
            driver: "cuda".into(),
            device_id: Some(0),
            triple: None,
            architecture: architecture.clone(),
            features,
            capability_fingerprint: Digest::from_bytes(b"pending"),
            device_uri: None,
            device_name: None,
            subgroup_size: Some(32),
            max_workgroup_invocations: None,
            executable_target_flags: architecture
                .as_ref()
                .map(|arch| {
                    vec![
                        "--iree-hal-target-device=cuda".into(),
                        format!("--iree-cuda-target={arch}"),
                    ]
                })
                .unwrap_or_default(),
            verified: architecture.is_some(),
        };
        profile.refresh_fingerprint();
        profile
    }

    pub fn with_device_identity(
        mut self,
        device_id: u32,
        device_uri: impl Into<String>,
        device_name: impl Into<String>,
    ) -> Self {
        self.device_id = Some(device_id);
        self.device_uri = nonempty(&device_uri.into());
        self.device_name = nonempty(&device_name.into());
        self.refresh_fingerprint();
        self
    }

    pub fn with_discovered_features(mut self, features: impl IntoIterator<Item = String>) -> Self {
        self.features.extend(features);
        self.refresh_fingerprint();
        self
    }

    pub fn with_execution_limits(
        mut self,
        subgroup_size: Option<u32>,
        max_workgroup_invocations: Option<u32>,
    ) -> Self {
        self.subgroup_size = subgroup_size.or(self.subgroup_size);
        self.max_workgroup_invocations =
            max_workgroup_invocations.or(self.max_workgroup_invocations);
        self.refresh_fingerprint();
        self
    }

    pub fn is_compile_ready(&self) -> bool {
        self.verified
            && (!matches!(self.driver.as_str(), "cuda" | "hip" | "rocm" | "vulkan")
                || self.architecture.is_some())
    }

    /// Preferred runtime device string: full URI when known, else driver name.
    pub fn runtime_device(&self) -> &str {
        self.device_uri
            .as_deref()
            .filter(|u| !u.is_empty())
            .unwrap_or(self.driver.as_str())
    }

    /// ROCm chip / SKU for `--iree-rocm-target`, if this is a HIP profile.
    pub fn rocm_target(&self) -> Option<&str> {
        if self.driver == "hip" || self.driver == "rocm" {
            self.architecture.as_deref()
        } else {
            None
        }
    }

    /// CUDA SM arch for `--iree-cuda-target`, if this is a CUDA profile.
    pub fn cuda_target(&self) -> Option<&str> {
        if self.driver == "cuda" {
            self.architecture.as_deref()
        } else {
            None
        }
    }

    /// Vulkan GPU arch for `--iree-vulkan-target`, if this is a Vulkan profile.
    pub fn vulkan_target(&self) -> Option<&str> {
        if self.driver == "vulkan" {
            self.architecture.as_deref()
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

    fn refresh_fingerprint(&mut self) {
        self.features.sort();
        self.features.dedup();
        self.executable_target_flags.sort();
        self.executable_target_flags.dedup();
        self.capability_fingerprint = Digest::from_bytes(
            format!(
                "driver={}|device_id={:?}|triple={:?}|architecture={:?}|features={:?}|uri={:?}|name={:?}|subgroup={:?}|max_workgroup={:?}|flags={:?}|verified={}",
                self.driver,
                self.device_id,
                self.triple,
                self.architecture,
                self.features,
                self.device_uri,
                self.device_name,
                self.subgroup_size,
                self.max_workgroup_invocations,
                self.executable_target_flags,
                self.verified
            )
            .as_bytes(),
        );
    }
}

fn nonempty(value: &str) -> Option<String> {
    (!value.trim().is_empty()).then(|| value.trim().to_string())
}

fn cuda_sm(architecture: &str) -> Option<u32> {
    architecture
        .strip_prefix("sm_")
        .or_else(|| architecture.strip_prefix("sm"))?
        .parse()
        .ok()
}

fn detected_host_features() -> Vec<String> {
    let mut features = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        for (name, detected) in [
            ("sse4.2", std::is_x86_feature_detected!("sse4.2")),
            ("avx", std::is_x86_feature_detected!("avx")),
            ("avx2", std::is_x86_feature_detected!("avx2")),
            ("fma", std::is_x86_feature_detected!("fma")),
            ("f16c", std::is_x86_feature_detected!("f16c")),
            ("avx512f", std::is_x86_feature_detected!("avx512f")),
            ("avx512bw", std::is_x86_feature_detected!("avx512bw")),
            ("avx512vl", std::is_x86_feature_detected!("avx512vl")),
            ("avx512vnni", std::is_x86_feature_detected!("avx512vnni")),
        ] {
            if detected {
                features.push(name.into());
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        for (name, detected) in [
            ("neon", std::arch::is_aarch64_feature_detected!("neon")),
            (
                "native_f16",
                std::arch::is_aarch64_feature_detected!("fp16"),
            ),
            (
                "native_bf16",
                std::arch::is_aarch64_feature_detected!("bf16"),
            ),
            (
                "dotprod",
                std::arch::is_aarch64_feature_detected!("dotprod"),
            ),
            ("i8mm", std::arch::is_aarch64_feature_detected!("i8mm")),
        ] {
            if detected {
                features.push(name.into());
            }
        }
    }
    features
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

/// Stable external component contract recorded in an executable manifest.
/// Paths and offsets are intentionally absent so schema-compatible checkpoint
/// values can reuse the same VMFB.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ManifestParameterComponent {
    pub scope: String,
    pub key: String,
    pub byte_length: u64,
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
    #[serde(default)]
    pub precision_policy: PrecisionPolicy,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selected_kernels: Vec<SelectedKernel>,
    pub shape_profile: ShapeProfile,
    pub entrypoints: Vec<String>,
    pub kv_cache: KvCacheDescriptor,
    pub parameter_scope: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parameter_components: Vec<ManifestParameterComponent>,
    #[serde(default)]
    pub derived_parameters_required: bool,
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
    fn target_profile_roundtrips() {
        let t = TargetProfile::llvm_cpu_host();
        let json = serde_json::to_string(&t).unwrap();
        let back: TargetProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(t.driver, back.driver);
    }

    #[test]
    fn target_fingerprint_is_deterministic_and_capability_sensitive() {
        let a = TargetProfile::cuda("sm_100")
            .with_device_identity(2, "cuda://GPU-stable", "example")
            .with_discovered_features(["feature_b".into(), "feature_a".into()]);
        let b = TargetProfile::cuda("sm_100")
            .with_device_identity(2, "cuda://GPU-stable", "example")
            .with_discovered_features(["feature_a".into(), "feature_b".into()]);
        assert_eq!(a.capability_fingerprint, b.capability_fingerprint);
        assert!(a.features.iter().any(|feature| feature == "nvfp4"));

        let different_arch =
            TargetProfile::cuda("sm_90").with_device_identity(2, "cuda://GPU-stable", "example");
        assert_ne!(
            a.capability_fingerprint,
            different_arch.capability_fingerprint
        );
    }
}
