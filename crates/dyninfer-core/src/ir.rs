//! Serializable semantic and bound model intermediate representations.
//!
//! These types are deliberately data-only. Architecture definitions construct
//! [`ArchitectureGraph`] values; later compiler phases attach checkpoint
//! bindings, execution shapes, target facts, precision policy, and selected
//! kernels in [`BoundModel`]. No type in this module owns lowering callbacks or
//! host-specific checkpoint paths.

use crate::{
    ArchitectureId, BindingPlan, GraphValueId, KernelId, LoweringId, MetadataMap, OperationId,
    ParameterRole, ParameterSlot, ParameterSlotId, ScalarType, TargetProfile,
};
use serde::{Deserialize, Serialize};

/// A dimension in a checkpoint-independent architecture graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum TensorDimension {
    Static(u64),
    Symbol(String),
}

impl TensorDimension {
    pub fn symbol(name: impl Into<String>) -> Self {
        Self::Symbol(name.into())
    }
}

/// Semantic element categories before operation-level precision selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticElementType {
    TokenId,
    Floating,
}

/// Tensor type attached to every graph value.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemanticTensorType {
    pub shape: Vec<TensorDimension>,
    pub element_type: SemanticElementType,
}

impl SemanticTensorType {
    pub fn tokens() -> Self {
        Self {
            shape: vec![
                TensorDimension::symbol("batch"),
                TensorDimension::symbol("sequence"),
            ],
            element_type: SemanticElementType::TokenId,
        }
    }

    pub fn activations(width: u32) -> Self {
        Self {
            shape: vec![
                TensorDimension::symbol("batch"),
                TensorDimension::symbol("sequence"),
                TensorDimension::Static(u64::from(width)),
            ],
            element_type: SemanticElementType::Floating,
        }
    }

    pub fn kv_cache(head_count: u32, head_dim: u32) -> Self {
        Self {
            shape: vec![
                TensorDimension::symbol("batch"),
                TensorDimension::Static(u64::from(head_count)),
                TensorDimension::symbol("kv_sequence"),
                TensorDimension::Static(u64::from(head_dim)),
            ],
            element_type: SemanticElementType::Floating,
        }
    }
}

/// One named value in the semantic graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphValue {
    pub id: GraphValueId,
    pub tensor_type: SemanticTensorType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelInputKind {
    Tokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheComponent {
    Key,
    Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementwiseFunction {
    Silu,
    Multiply,
}

/// Typed semantic operation kind. Physical checkpoint encodings and lowering
/// choices intentionally do not appear here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OperationKind {
    Input {
        input: ModelInputKind,
    },
    Embedding,
    Linear {
        role: ParameterRole,
    },
    RmsNorm {
        epsilon: f64,
    },
    PerHeadRmsNorm {
        epsilon: f64,
        head_count: u32,
        head_dim: u32,
    },
    Rope {
        head_count: u32,
        head_dim: u32,
        theta: f64,
    },
    KvCacheWrite {
        layer: u32,
        component: KvCacheComponent,
    },
    KvCacheRead {
        layer: u32,
        component: KvCacheComponent,
    },
    Attention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        causal: bool,
    },
    /// LFM2-style gated causal depthwise short convolution (`Lfm2ShortConv`).
    ///
    /// The single input is the `3 * channels`-wide projection `[B, C, x]`. The
    /// operation computes `C * depthwise_conv1d(B * x)` over the causal window
    /// `kernel_size`, so the gating multiplies and the channel-local recurrence
    /// stay fused. `layer` addresses the per-layer convolution state that
    /// carries the trailing `kernel_size` inputs across decode steps, the
    /// same role [`KvCacheWrite`](Self::KvCacheWrite) plays for attention.
    ShortConv {
        layer: u32,
        channels: u32,
        kernel_size: u32,
    },
    Elementwise {
        function: ElementwiseFunction,
    },
    Residual,
    OutputProjection,
}

/// A semantic operation with explicit dataflow and parameter dependencies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureOperation {
    pub id: OperationId,
    pub kind: OperationKind,
    pub inputs: Vec<GraphValueId>,
    pub outputs: Vec<GraphValueId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parameters: Vec<ParameterSlotId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    Prefill,
    Decode,
}

/// Public semantic export before its concrete ABI shapes are specialized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchitectureExport {
    pub name: String,
    pub mode: ExecutionMode,
    pub inputs: Vec<GraphValueId>,
    pub outputs: Vec<GraphValueId>,
}

/// Checkpoint- and quantization-independent architecture graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureGraph {
    pub version: u32,
    pub architecture_id: ArchitectureId,
    pub values: Vec<GraphValue>,
    pub operations: Vec<ArchitectureOperation>,
    pub parameter_slots: Vec<ParameterSlot>,
    pub exports: Vec<ArchitectureExport>,
}

/// Versioned policy used to choose operation-local compute types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrecisionPolicy {
    pub id: String,
    pub version: u32,
    pub sensitive_internal_type: ScalarType,
}

impl PrecisionPolicy {
    pub const CONSERVATIVE_SENSITIVE_OPS_ID: &'static str = "conservative_sensitive_ops";

    pub fn conservative_sensitive_ops() -> Self {
        Self {
            id: Self::CONSERVATIVE_SENSITIVE_OPS_ID.into(),
            version: 1,
            sensitive_internal_type: ScalarType::F32,
        }
    }
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self::conservative_sensitive_ops()
    }
}

/// One concrete static request generated from a semantic export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecializedExecutionShape {
    pub mode: ExecutionMode,
    pub batch_size: u32,
    pub sequence_length: u32,
    pub max_kv_length: u32,
}

/// Serializable result of kernel selection for one operation and execution
/// mode. Lowering implementation objects remain in their owning registries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedKernel {
    pub operation_id: OperationId,
    pub mode: ExecutionMode,
    pub kernel_id: KernelId,
    pub lowering_id: LoweringId,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parameter_slots: Vec<ParameterSlotId>,
    pub input_type: ScalarType,
    pub output_type: ScalarType,
    pub activation_type: ScalarType,
    pub accumulator_type: ScalarType,
}

/// Canonical compiler input after semantic binding and specialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundModel {
    pub version: u32,
    pub architecture: ArchitectureGraph,
    pub resolved_config: MetadataMap,
    pub binding: BindingPlan,
    pub execution_shapes: Vec<SpecializedExecutionShape>,
    pub target: TargetProfile,
    pub precision_policy: PrecisionPolicy,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selected_kernels: Vec<SelectedKernel>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conservative_policy_is_explicit_and_versioned() {
        let policy = PrecisionPolicy::default();
        assert_eq!(policy.id, "conservative_sensitive_ops");
        assert_eq!(policy.version, 1);
        assert_eq!(policy.sensitive_internal_type, ScalarType::F32);
    }

    #[test]
    fn symbolic_tensor_types_roundtrip() {
        let ty = SemanticTensorType::activations(4096);
        let json = serde_json::to_string(&ty).unwrap();
        let decoded: SemanticTensorType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, ty);
    }
}
