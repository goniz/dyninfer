//! Typed production-kernel candidate registry and deterministic selection.
//!
//! This crate owns selection mechanics only. Encoding definitions contribute
//! candidate descriptors, while compiler crates own implementations addressed
//! by each descriptor's lowering ID.

#![forbid(unsafe_code)]

use dyninfer_core::{
    EncodingId, ExecutionMode, KernelId, LoweringId, OperationId, OperationKind, ParameterSlotId,
    PrecisionPolicy, ScalarType, Shape, TargetProfile,
};
use dyninfer_error::{CompilationError, Diagnostic, DynInferError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Kernel-selection operation categories, independent of display strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelOperationKind {
    Input,
    Embedding,
    Linear,
    RmsNorm,
    PerHeadRmsNorm,
    Rope,
    KvCacheWrite,
    KvCacheRead,
    Attention,
    ShortConv,
    Silu,
    Multiply,
    Residual,
    OutputProjection,
}

impl KernelOperationKind {
    pub fn from_semantic(operation: &OperationKind) -> Self {
        match operation {
            OperationKind::Input { .. } => Self::Input,
            OperationKind::Embedding => Self::Embedding,
            OperationKind::Linear { .. } => Self::Linear,
            OperationKind::RmsNorm { .. } => Self::RmsNorm,
            OperationKind::PerHeadRmsNorm { .. } => Self::PerHeadRmsNorm,
            OperationKind::Rope { .. } => Self::Rope,
            OperationKind::KvCacheWrite { .. } => Self::KvCacheWrite,
            OperationKind::KvCacheRead { .. } => Self::KvCacheRead,
            OperationKind::Attention { .. } => Self::Attention,
            OperationKind::ShortConv { .. } => Self::ShortConv,
            OperationKind::Elementwise {
                function: dyninfer_core::ElementwiseFunction::Silu,
            } => Self::Silu,
            OperationKind::Elementwise {
                function: dyninfer_core::ElementwiseFunction::Multiply,
            } => Self::Multiply,
            OperationKind::Residual => Self::Residual,
            OperationKind::OutputProjection => Self::OutputProjection,
        }
    }

    pub fn is_precision_sensitive(self) -> bool {
        matches!(
            self,
            Self::RmsNorm | Self::PerHeadRmsNorm | Self::Rope | Self::Attention
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EncodingKey {
    pub id: EncodingId,
    pub version: u32,
}

impl EncodingKey {
    pub fn new(id: impl Into<String>, version: u32) -> Self {
        Self {
            id: EncodingId::new(id),
            version,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterOrientation {
    Native,
    LogicalTranspose,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisMultiple {
    pub axis: usize,
    pub multiple_of: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ShapeConstraint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_rank: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub axis_multiples: Vec<AxisMultiple>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetConstraint {
    pub backends: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exact_architectures: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_features: Vec<String>,
}

impl TargetConstraint {
    pub fn any() -> Self {
        Self {
            backends: vec!["any".into()],
            exact_architectures: vec![],
            required_features: vec![],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductionReadiness {
    Prototype,
    Production,
}

/// Serializable kernel candidate contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelCandidateDescriptor {
    pub id: KernelId,
    pub operation: KernelOperationKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<EncodingKey>,
    pub input_types: Vec<ScalarType>,
    pub output_types: Vec<ScalarType>,
    pub accumulator_types: Vec<ScalarType>,
    pub shape: ShapeConstraint,
    pub orientations: Vec<ParameterOrientation>,
    pub target: TargetConstraint,
    pub modes: Vec<ExecutionMode>,
    pub lowering: LoweringId,
    pub deterministic_score: i64,
    pub readiness: ProductionReadiness,
    pub notes: String,
}

/// Complete operation-local selection request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelRequest {
    pub operation_id: OperationId,
    pub operation: KernelOperationKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<EncodingKey>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_shape: Option<Shape>,
    pub orientation: ParameterOrientation,
    pub mode: ExecutionMode,
    pub precision_policy: PrecisionPolicy,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parameter_slots: Vec<ParameterSlotId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checkpoint_component_keys: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum CandidateRejectionReason {
    EncodingMismatch,
    NotProductionReady,
    ModeMismatch,
    BackendMismatch {
        selected: String,
    },
    ArchitectureMismatch {
        selected: Option<String>,
    },
    MissingTargetFeature {
        feature: String,
    },
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    AxisNotDivisible {
        axis: usize,
        dimension: u64,
        multiple_of: u64,
    },
    OrientationMismatch,
    PrecisionFloor {
        required: ScalarType,
        supported: Vec<ScalarType>,
    },
    MissingTypeChoice,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateRejection {
    pub candidate_id: KernelId,
    pub reasons: Vec<CandidateRejectionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedKernelCandidate {
    pub descriptor: KernelCandidateDescriptor,
    pub input_type: ScalarType,
    pub output_type: ScalarType,
    pub activation_type: ScalarType,
    pub accumulator_type: ScalarType,
    pub score: i64,
}

impl SelectedKernelCandidate {
    pub fn to_manifest_record(&self, request: &KernelRequest) -> dyninfer_core::SelectedKernel {
        dyninfer_core::SelectedKernel {
            operation_id: request.operation_id.clone(),
            mode: request.mode,
            kernel_id: self.descriptor.id.clone(),
            lowering_id: self.descriptor.lowering.clone(),
            parameter_slots: request.parameter_slots.clone(),
            input_type: self.input_type,
            output_type: self.output_type,
            activation_type: self.activation_type,
            accumulator_type: self.accumulator_type,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectionReport {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected: Option<SelectedKernelCandidate>,
    pub rejected: Vec<CandidateRejection>,
}

pub trait KernelCostModel: Send + Sync {
    fn score(
        &self,
        candidate: &KernelCandidateDescriptor,
        request: &KernelRequest,
        target: &TargetProfile,
    ) -> Result<i64>;
}

#[derive(Debug, Default)]
pub struct DeterministicCostModel;

impl KernelCostModel for DeterministicCostModel {
    fn score(
        &self,
        candidate: &KernelCandidateDescriptor,
        _request: &KernelRequest,
        target: &TargetProfile,
    ) -> Result<i64> {
        let exact_backend_bonus = i64::from(
            candidate
                .target
                .backends
                .iter()
                .any(|backend| backend == &target.driver),
        ) * 1000;
        Ok(candidate.deterministic_score + exact_backend_bonus)
    }
}

#[derive(Debug, Default)]
pub struct KernelRegistry {
    candidates: Vec<KernelCandidateDescriptor>,
}

impl KernelRegistry {
    /// Bump whenever candidate interpretation or selection ordering changes.
    pub const VERSION: &'static str = "5";

    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, candidate: KernelCandidateDescriptor) -> Result<()> {
        if self
            .candidates
            .iter()
            .any(|existing| existing.id == candidate.id)
        {
            return Err(DynInferError::internal(format!(
                "duplicate kernel candidate `{}`",
                candidate.id
            )));
        }
        self.candidates.push(candidate);
        Ok(())
    }

    pub fn candidates(&self) -> &[KernelCandidateDescriptor] {
        &self.candidates
    }

    pub fn select(
        &self,
        request: &KernelRequest,
        target: &TargetProfile,
        cost: &dyn KernelCostModel,
    ) -> Result<SelectionReport> {
        let mut accepted = Vec::new();
        let mut rejected = Vec::new();
        for candidate in self
            .candidates
            .iter()
            .filter(|candidate| candidate.operation == request.operation)
        {
            let reasons = rejection_reasons(candidate, request, target);
            if reasons.is_empty() {
                let accumulator_type = if request.operation.is_precision_sensitive() {
                    request.precision_policy.sensitive_internal_type
                } else {
                    candidate.accumulator_types[0]
                };
                let selected = SelectedKernelCandidate {
                    descriptor: candidate.clone(),
                    input_type: candidate.input_types[0],
                    output_type: candidate.output_types[0],
                    activation_type: candidate.input_types[0],
                    accumulator_type,
                    score: cost.score(candidate, request, target)?,
                };
                accepted.push(selected);
            } else {
                rejected.push(CandidateRejection {
                    candidate_id: candidate.id.clone(),
                    reasons,
                });
            }
        }
        accepted.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| left.descriptor.id.cmp(&right.descriptor.id))
        });
        Ok(SelectionReport {
            selected: accepted.into_iter().next(),
            rejected,
        })
    }

    pub fn require(
        &self,
        request: &KernelRequest,
        target: &TargetProfile,
        cost: &dyn KernelCostModel,
    ) -> Result<SelectedKernelCandidate> {
        let report = self.select(request, target, cost)?;
        if let Some(selected) = report.selected {
            return Ok(selected);
        }
        let encoding = request
            .encoding
            .as_ref()
            .map(|encoding| format!("{}@{}", encoding.id, encoding.version))
            .unwrap_or_else(|| "none".into());
        let rejected = report
            .rejected
            .iter()
            .map(|candidate| format!("{}: {:?}", candidate.candidate_id, candidate.reasons))
            .collect::<Vec<_>>();
        let mut diagnostic = Diagnostic::error(
            "E_KERNEL_COVERAGE",
            format!(
                "no production kernel for {:?} with encoding {encoding} in {:?} mode",
                request.operation, request.mode
            ),
        );
        diagnostic.architecture_op = Some(request.operation_id.to_string());
        diagnostic.parameter_slot = request.parameter_slots.first().map(ToString::to_string);
        diagnostic.checkpoint_key = request.checkpoint_component_keys.first().cloned();
        diagnostic.actual = Some(format!(
            "target={} architecture={:?} triple={:?} features={:?}",
            target.driver, target.architecture, target.triple, target.features
        ));
        diagnostic.pass_name = Some("kernel.coverage".into());
        diagnostic.suggestion = Some(format!(
            "implement and qualify a {:?}/{encoding} kernel for this target; rejected candidates: {}",
            request.operation,
            if rejected.is_empty() {
                "none registered".into()
            } else {
                rejected.join("; ")
            }
        ));
        Err(DynInferError::Compilation(CompilationError {
            message: diagnostic.message.clone(),
            pass: Some("kernel.coverage".into()),
            diagnostics: vec![diagnostic],
        }))
    }
}

/// Register architecture-independent production candidates for operations that
/// do not consume encoded checkpoint parameters.
pub fn register_builtin_semantic_candidates(registry: &mut KernelRegistry) -> Result<()> {
    for (operation, id, lowering, scalar, score) in [
        (
            KernelOperationKind::Input,
            "model.input.abi",
            "model.input.abi",
            ScalarType::I64,
            100,
        ),
        (
            KernelOperationKind::Rope,
            "rope.generated.f32",
            "rope.generated",
            ScalarType::F32,
            100,
        ),
        (
            KernelOperationKind::KvCacheWrite,
            "kv_cache.write.generated",
            "kv_cache.write.generated",
            ScalarType::F32,
            100,
        ),
        (
            KernelOperationKind::KvCacheRead,
            "kv_cache.read.generated",
            "kv_cache.read.generated",
            ScalarType::F32,
            100,
        ),
        (
            KernelOperationKind::Attention,
            "attention.gqa.generated.f32",
            "attention.gqa.generated",
            ScalarType::F32,
            100,
        ),
        (
            KernelOperationKind::Silu,
            "elementwise.silu.generated.f32",
            "elementwise.silu.generated",
            ScalarType::F32,
            100,
        ),
        (
            KernelOperationKind::Multiply,
            "elementwise.multiply.generated.f32",
            "elementwise.multiply.generated",
            ScalarType::F32,
            100,
        ),
        (
            KernelOperationKind::Residual,
            "residual.add.generated.f32",
            "residual.add.generated",
            ScalarType::F32,
            100,
        ),
    ] {
        registry.register(KernelCandidateDescriptor {
            id: KernelId::new(id),
            operation,
            encoding: None,
            input_types: vec![scalar],
            output_types: vec![scalar],
            accumulator_types: vec![scalar],
            shape: ShapeConstraint::default(),
            orientations: vec![ParameterOrientation::Native],
            target: TargetConstraint::any(),
            modes: vec![ExecutionMode::Prefill, ExecutionMode::Decode],
            lowering: LoweringId::new(lowering),
            deterministic_score: score,
            readiness: ProductionReadiness::Production,
            notes: "Architecture-independent generated lowering".into(),
        })?;
    }
    registry.register(KernelCandidateDescriptor {
        id: KernelId::new("attention.online_paged.generated.f32"),
        operation: KernelOperationKind::Attention,
        encoding: None,
        input_types: vec![ScalarType::F32],
        output_types: vec![ScalarType::F32],
        accumulator_types: vec![ScalarType::F32],
        shape: ShapeConstraint::default(),
        orientations: vec![ParameterOrientation::Native],
        target: TargetConstraint::any(),
        modes: vec![ExecutionMode::Prefill, ExecutionMode::Decode],
        lowering: LoweringId::new("attention.online_paged.generated"),
        deterministic_score: 90,
        readiness: ProductionReadiness::Production,
        notes: "Runtime-paged online-softmax attention".into(),
    })?;
    Ok(())
}

fn rejection_reasons(
    candidate: &KernelCandidateDescriptor,
    request: &KernelRequest,
    target: &TargetProfile,
) -> Vec<CandidateRejectionReason> {
    let mut reasons = Vec::new();
    if candidate.encoding != request.encoding {
        reasons.push(CandidateRejectionReason::EncodingMismatch);
    }
    if candidate.readiness != ProductionReadiness::Production {
        reasons.push(CandidateRejectionReason::NotProductionReady);
    }
    if !candidate.modes.contains(&request.mode) {
        reasons.push(CandidateRejectionReason::ModeMismatch);
    }
    if !candidate
        .target
        .backends
        .iter()
        .any(|backend| backend == "any" || backend == &target.driver)
    {
        reasons.push(CandidateRejectionReason::BackendMismatch {
            selected: target.driver.clone(),
        });
    }
    if !candidate.target.exact_architectures.is_empty()
        && !target
            .architecture
            .as_ref()
            .is_some_and(|architecture| candidate.target.exact_architectures.contains(architecture))
    {
        reasons.push(CandidateRejectionReason::ArchitectureMismatch {
            selected: target.architecture.clone(),
        });
    }
    let target_features: BTreeSet<_> = target.features.iter().map(String::as_str).collect();
    for feature in &candidate.target.required_features {
        if !target_features.contains(feature.as_str()) {
            reasons.push(CandidateRejectionReason::MissingTargetFeature {
                feature: feature.clone(),
            });
        }
    }
    if let (Some(shape), Some(expected)) = (&request.logical_shape, candidate.shape.logical_rank) {
        if shape.rank() != expected {
            reasons.push(CandidateRejectionReason::RankMismatch {
                expected,
                actual: shape.rank(),
            });
        }
    }
    if let Some(shape) = &request.logical_shape {
        for constraint in &candidate.shape.axis_multiples {
            if let Some(&dimension) = shape.dims().get(constraint.axis) {
                if constraint.multiple_of == 0 || !dimension.is_multiple_of(constraint.multiple_of)
                {
                    reasons.push(CandidateRejectionReason::AxisNotDivisible {
                        axis: constraint.axis,
                        dimension,
                        multiple_of: constraint.multiple_of,
                    });
                }
            }
        }
    }
    if !candidate.orientations.is_empty() && !candidate.orientations.contains(&request.orientation)
    {
        reasons.push(CandidateRejectionReason::OrientationMismatch);
    }
    if candidate.input_types.is_empty()
        || candidate.output_types.is_empty()
        || candidate.accumulator_types.is_empty()
    {
        reasons.push(CandidateRejectionReason::MissingTypeChoice);
    } else if request.operation.is_precision_sensitive()
        && !candidate
            .accumulator_types
            .contains(&request.precision_policy.sensitive_internal_type)
    {
        reasons.push(CandidateRejectionReason::PrecisionFloor {
            required: request.precision_policy.sensitive_internal_type,
            supported: candidate.accumulator_types.clone(),
        });
    }
    reasons
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, readiness: ProductionReadiness) -> KernelCandidateDescriptor {
        KernelCandidateDescriptor {
            id: KernelId::new(id),
            operation: KernelOperationKind::Linear,
            encoding: Some(EncodingKey::new("plain.f16", 1)),
            input_types: vec![ScalarType::F32],
            output_types: vec![ScalarType::F32],
            accumulator_types: vec![ScalarType::F32],
            shape: ShapeConstraint {
                logical_rank: Some(2),
                axis_multiples: vec![],
            },
            orientations: vec![ParameterOrientation::Native],
            target: TargetConstraint::any(),
            modes: vec![ExecutionMode::Prefill, ExecutionMode::Decode],
            lowering: LoweringId::new("dense.matmul.linalg"),
            deterministic_score: 100,
            readiness,
            notes: String::new(),
        }
    }

    fn request() -> KernelRequest {
        KernelRequest {
            operation_id: OperationId::new("blk.0.attn_q"),
            operation: KernelOperationKind::Linear,
            encoding: Some(EncodingKey::new("plain.f16", 1)),
            logical_shape: Some(Shape::new(vec![64, 64])),
            orientation: ParameterOrientation::Native,
            mode: ExecutionMode::Decode,
            precision_policy: PrecisionPolicy::default(),
            parameter_slots: vec![ParameterSlotId::new("blk.0.attn_q.weight")],
            checkpoint_component_keys: vec!["weights::blk.0.attn_q.weight::data".into()],
        }
    }

    #[test]
    fn only_production_candidates_can_be_selected() {
        let mut registry = KernelRegistry::new();
        registry
            .register(candidate("prototype", ProductionReadiness::Prototype))
            .unwrap();
        registry
            .register(candidate("production", ProductionReadiness::Production))
            .unwrap();
        let report = registry
            .select(
                &request(),
                &TargetProfile::llvm_cpu_host(),
                &DeterministicCostModel,
            )
            .unwrap();
        assert_eq!(
            report.selected.unwrap().descriptor.id.as_str(),
            "production"
        );
        assert!(matches!(
            report.rejected[0].reasons.as_slice(),
            [CandidateRejectionReason::NotProductionReady]
        ));
    }

    #[test]
    fn tie_breaking_is_stable_by_kernel_id() {
        let mut registry = KernelRegistry::new();
        registry
            .register(candidate("z.kernel", ProductionReadiness::Production))
            .unwrap();
        registry
            .register(candidate("a.kernel", ProductionReadiness::Production))
            .unwrap();
        let selected = registry
            .require(
                &request(),
                &TargetProfile::llvm_cpu_host(),
                &DeterministicCostModel,
            )
            .unwrap();
        assert_eq!(selected.descriptor.id.as_str(), "a.kernel");
    }

    #[test]
    fn unsupported_encoding_returns_structured_coverage_error() {
        let mut registry = KernelRegistry::new();
        registry
            .register(candidate("dense", ProductionReadiness::Production))
            .unwrap();
        let mut unsupported = request();
        unsupported.encoding = Some(EncodingKey::new("gguf.q8_0", 1));
        let error = registry
            .require(
                &unsupported,
                &TargetProfile::llvm_cpu_host(),
                &DeterministicCostModel,
            )
            .unwrap_err();
        let DynInferError::Compilation(error) = error else {
            panic!("expected compilation error");
        };
        assert_eq!(error.diagnostics[0].code, "E_KERNEL_COVERAGE");
        assert_eq!(
            error.diagnostics[0].architecture_op.as_deref(),
            Some("blk.0.attn_q")
        );
    }

    #[test]
    fn sensitive_operation_rejects_candidate_below_precision_floor() {
        let mut low_precision = candidate("rms.f16", ProductionReadiness::Production);
        low_precision.operation = KernelOperationKind::RmsNorm;
        low_precision.shape.logical_rank = Some(1);
        low_precision.accumulator_types = vec![ScalarType::F16];
        let mut sensitive = request();
        sensitive.operation = KernelOperationKind::RmsNorm;
        sensitive.logical_shape = Some(Shape::new(vec![64]));

        let mut registry = KernelRegistry::new();
        registry.register(low_precision).unwrap();
        let report = registry
            .select(
                &sensitive,
                &TargetProfile::llvm_cpu_host(),
                &DeterministicCostModel,
            )
            .unwrap();
        assert!(report.selected.is_none());
        assert!(matches!(
            report.rejected[0].reasons.as_slice(),
            [CandidateRejectionReason::PrecisionFloor {
                required: ScalarType::F32,
                ..
            }]
        ));
    }

    #[test]
    fn exact_gpu_architecture_matches_architecture_not_cpu_triple() {
        let mut exact = candidate("cuda.sm100", ProductionReadiness::Production);
        exact.target = TargetConstraint {
            backends: vec!["cuda".into()],
            exact_architectures: vec!["sm_100".into()],
            required_features: vec!["nvfp4".into()],
        };
        let mut registry = KernelRegistry::new();
        registry.register(exact).unwrap();
        let target = TargetProfile::cuda("sm_100");
        let report = registry
            .select(&request(), &target, &DeterministicCostModel)
            .unwrap();
        assert!(report.selected.is_some());
    }
}
