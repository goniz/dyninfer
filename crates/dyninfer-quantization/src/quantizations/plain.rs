use crate::{EncodingDefinitionDescriptor, ExternalEncodingTag, QuantizationDefinition};
use dyninfer_checkpoint::LogicalParameter;
use dyninfer_core::{PhysicalEncoding, ScalarType};
use dyninfer_error::{DynInferError, Result, UnsupportedEncodingError};
use dyninfer_kernel_registry::{
    EncodingKey, KernelCandidateDescriptor, KernelOperationKind, ParameterOrientation,
    ProductionReadiness, ShapeConstraint, TargetConstraint,
};

#[derive(Debug)]
pub struct PlainDefinition {
    storage_type: ScalarType,
}

impl PlainDefinition {
    pub fn new(storage_type: ScalarType) -> Self {
        assert!(matches!(
            storage_type,
            ScalarType::F32 | ScalarType::F16 | ScalarType::Bf16
        ));
        Self { storage_type }
    }

    fn id(&self) -> String {
        format!("plain.{}", self.storage_type)
    }

    /// Rank of the checkpoint parameter consumed by `operation`.
    fn logical_rank(operation: KernelOperationKind) -> usize {
        match operation {
            KernelOperationKind::RmsNorm | KernelOperationKind::PerHeadRmsNorm => 1,
            // Depthwise short-conv weights are stored as `[channels, 1, kernel]`.
            KernelOperationKind::ShortConv => 3,
            _ => 2,
        }
    }

    fn candidate(
        &self,
        operation: KernelOperationKind,
        suffix: &str,
        lowering: &str,
    ) -> KernelCandidateDescriptor {
        KernelCandidateDescriptor {
            id: dyninfer_core::KernelId::new(format!(
                "dense.{suffix}.{}.portable_f32",
                self.storage_type
            )),
            operation,
            encoding: Some(EncodingKey::new(self.id(), 1)),
            input_types: vec![ScalarType::F32],
            output_types: vec![ScalarType::F32],
            accumulator_types: vec![ScalarType::F32],
            shape: ShapeConstraint {
                logical_rank: Some(Self::logical_rank(operation)),
                axis_multiples: vec![],
            },
            orientations: vec![ParameterOrientation::Native],
            target: TargetConstraint::any(),
            modes: vec![
                dyninfer_core::ExecutionMode::Prefill,
                dyninfer_core::ExecutionMode::Decode,
            ],
            lowering: dyninfer_core::LoweringId::new(lowering),
            deterministic_score: 100,
            readiness: ProductionReadiness::Production,
            notes: format!(
                "Consumes plain {} storage directly and computes conservatively in f32",
                self.storage_type
            ),
        }
    }

    fn native_candidate(
        &self,
        operation: KernelOperationKind,
    ) -> Option<KernelCandidateDescriptor> {
        let (feature, compute_type) = match self.storage_type {
            ScalarType::F16 => ("native_f16", ScalarType::F16),
            ScalarType::Bf16 => ("native_bf16", ScalarType::Bf16),
            _ => return None,
        };
        let suffix = match operation {
            KernelOperationKind::Embedding => "gather",
            KernelOperationKind::Linear => "matmul",
            KernelOperationKind::RmsNorm => "rms_norm",
            KernelOperationKind::PerHeadRmsNorm => "per_head_rms_norm",
            KernelOperationKind::OutputProjection => "output_projection",
            KernelOperationKind::ShortConv => "short_conv",
            _ => unreachable!(),
        };
        Some(KernelCandidateDescriptor {
            id: dyninfer_core::KernelId::new(format!(
                "dense.{suffix}.{}.native",
                self.storage_type
            )),
            operation,
            encoding: Some(EncodingKey::new(self.id(), 1)),
            input_types: vec![compute_type],
            output_types: vec![compute_type],
            // Sensitive norm reductions remain f32 even when their surrounding
            // activation type is native f16/bf16.
            accumulator_types: vec![ScalarType::F32],
            shape: ShapeConstraint {
                logical_rank: Some(Self::logical_rank(operation)),
                axis_multiples: vec![],
            },
            orientations: vec![ParameterOrientation::Native],
            target: TargetConstraint {
                backends: vec!["any".into()],
                exact_architectures: vec![],
                required_features: vec![feature.into()],
            },
            modes: vec![
                dyninfer_core::ExecutionMode::Prefill,
                dyninfer_core::ExecutionMode::Decode,
            ],
            lowering: dyninfer_core::LoweringId::new(format!("dense.{suffix}.native")),
            deterministic_score: 200,
            readiness: ProductionReadiness::Production,
            notes: format!(
                "Native {} activation path gated by verified target capability",
                self.storage_type
            ),
        })
    }
}

impl QuantizationDefinition for PlainDefinition {
    fn descriptor(&self) -> EncodingDefinitionDescriptor {
        EncodingDefinitionDescriptor {
            key: EncodingKey::new(self.id(), 1),
            external_tags: vec![ExternalEncodingTag {
                family: "scalar".into(),
                value: self.storage_type.to_string(),
            }],
        }
    }

    fn matches(&self, encoding: &PhysicalEncoding) -> bool {
        matches!(
            encoding,
            PhysicalEncoding::Plain { storage_type, .. } if storage_type == &self.storage_type
        )
    }

    fn validate(&self, parameter: &LogicalParameter) -> Result<()> {
        if !self.matches(&parameter.encoding) {
            return Err(DynInferError::UnsupportedEncoding(
                UnsupportedEncodingError {
                    message: format!(
                        "parameter `{}` does not use {} storage",
                        parameter.canonical_name, self.storage_type
                    ),
                    key: Some(parameter.canonical_name.to_string()),
                    codec: Some(self.id()),
                    codec_version: Some(1),
                    expected: Some(self.storage_type.to_string()),
                    actual: Some(format!("{:?}", parameter.encoding)),
                },
            ));
        }
        if parameter.logical_type.element_type != self.storage_type {
            return Err(DynInferError::UnsupportedEncoding(
                UnsupportedEncodingError {
                    message: "plain logical and storage scalar types differ".into(),
                    key: Some(parameter.canonical_name.to_string()),
                    codec: Some(self.id()),
                    codec_version: Some(1),
                    expected: Some(self.storage_type.to_string()),
                    actual: Some(parameter.logical_type.element_type.to_string()),
                },
            ));
        }
        let expected = parameter
            .logical_type
            .shape
            .numel()
            .and_then(|numel| {
                self.storage_type
                    .size_bytes()
                    .and_then(|size| numel.checked_mul(u64::from(size)))
            })
            .ok_or_else(|| DynInferError::internal("plain parameter byte size overflow"))?;
        let actual = parameter
            .components
            .iter()
            .flat_map(|component| &component.byte_ranges)
            .try_fold(0u64, |total, range| total.checked_add(range.length))
            .ok_or_else(|| DynInferError::internal("plain component byte size overflow"))?;
        if parameter.components.len() != 1 || actual != expected {
            return Err(DynInferError::UnsupportedEncoding(
                UnsupportedEncodingError {
                    message: "plain parameter must have one complete storage component".into(),
                    key: Some(parameter.canonical_name.to_string()),
                    codec: Some(self.id()),
                    codec_version: Some(1),
                    expected: Some(format!("1 component, {expected} bytes")),
                    actual: Some(format!(
                        "{} components, {actual} bytes",
                        parameter.components.len()
                    )),
                },
            ));
        }
        Ok(())
    }

    fn kernel_candidates(&self) -> Vec<KernelCandidateDescriptor> {
        let operations = [
            (
                KernelOperationKind::Embedding,
                "gather",
                "dense.gather.linalg",
            ),
            (KernelOperationKind::Linear, "matmul", "dense.matmul.linalg"),
            (
                KernelOperationKind::RmsNorm,
                "rms_norm",
                "dense.rms_norm.generated",
            ),
            (
                KernelOperationKind::PerHeadRmsNorm,
                "per_head_rms_norm",
                "dense.per_head_rms_norm.generated",
            ),
            (
                KernelOperationKind::OutputProjection,
                "output_projection",
                "dense.matmul.linalg",
            ),
            (
                KernelOperationKind::ShortConv,
                "short_conv",
                "short_conv.gated.generated",
            ),
        ];
        let mut candidates = Vec::new();
        for (operation, suffix, lowering) in operations {
            candidates.push(self.candidate(operation, suffix, lowering));
            if let Some(native) = self.native_candidate(operation) {
                candidates.push(native);
            }
        }
        candidates
    }
}
