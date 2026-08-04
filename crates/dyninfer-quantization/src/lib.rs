//! Physical encoding definitions and dry-run kernel coverage.

#![forbid(unsafe_code)]

mod quantizations;

use dyninfer_checkpoint::LogicalParameter;
use dyninfer_core::{
    ArchitectureGraph, BindingPlan, BindingTransform, EncodingId, ExecutionMode, LoweringId,
    OperationId, ParameterBinding, ParameterSlotId, PhysicalEncoding, PrecisionPolicy,
    SelectedKernel, Shape, TargetProfile,
};
use dyninfer_error::{CompilationError, Diagnostic, DynInferError, Result};
use dyninfer_kernel_registry::{
    CandidateRejection, DeterministicCostModel, EncodingKey, KernelCandidateDescriptor,
    KernelOperationKind, KernelRegistry, KernelRequest, ParameterOrientation,
    SelectedKernelCandidate,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Static operation shape passed to an encoding-owned MLIR lowering hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterLoweringOperation {
    Embedding,
    Linear { rows: u32 },
    OutputProjection,
}

/// A selected physical-parameter lowering. This contains no checkpoint path or
/// byte offset; the binding exposes only stable external component keys.
#[derive(Debug, Clone, Copy)]
pub struct ParameterLoweringSpec<'a> {
    pub binding: &'a ParameterBinding,
    pub lowering_id: &'a LoweringId,
    pub mode: ExecutionMode,
    pub operation: ParameterLoweringOperation,
}

#[derive(Debug)]
pub struct LinearCallSpec<'a> {
    pub lowering: ParameterLoweringSpec<'a>,
    pub result_ssa: &'a str,
    pub input_ssa: &'a str,
    pub parameter_ssa: &'a str,
}

#[derive(Debug)]
pub struct EmbeddingCallSpec<'a> {
    pub lowering: ParameterLoweringSpec<'a>,
    pub result_ssa: &'a str,
    pub index_ssa: &'a str,
    pub parameter_ssa: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalEncodingTag {
    pub family: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodingDefinitionDescriptor {
    pub key: EncodingKey,
    pub external_tags: Vec<ExternalEncodingTag>,
}

/// Implementation contract kept separate from serializable encoding data.
pub trait QuantizationDefinition: Send + Sync + std::fmt::Debug {
    fn descriptor(&self) -> EncodingDefinitionDescriptor;
    fn matches(&self, encoding: &PhysicalEncoding) -> bool;
    fn validate(&self, parameter: &LogicalParameter) -> Result<()>;
    fn kernel_candidates(&self) -> Vec<KernelCandidateDescriptor>;

    /// Emit stable external globals for a non-plain encoded parameter.
    /// Returns false when this definition has no executable lowering.
    fn emit_external_globals(
        &self,
        _builder: &mut dyninfer_mlir::ModuleBuilder,
        _binding: &ParameterBinding,
        _symbol: &str,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Emit function-local loads for the globals declared above.
    fn emit_parameter_load(
        &self,
        _function: &mut dyninfer_mlir::FuncBuilder,
        _binding: &ParameterBinding,
        _ssa: &str,
        _symbol: &str,
    ) -> Result<bool> {
        Ok(false)
    }

    /// A deterministic key for deduplicating helper functions in one module.
    fn helper_key(&self, _spec: &ParameterLoweringSpec<'_>) -> Option<String> {
        None
    }

    fn emit_helper(
        &self,
        _module: &mut dyninfer_mlir::ModuleBuilder,
        _spec: &ParameterLoweringSpec<'_>,
    ) -> Result<bool> {
        Ok(false)
    }

    fn emit_linear_call(
        &self,
        _function: &mut dyninfer_mlir::FuncBuilder,
        _spec: &LinearCallSpec<'_>,
    ) -> Result<bool> {
        Ok(false)
    }

    fn emit_embedding_call(
        &self,
        _function: &mut dyninfer_mlir::FuncBuilder,
        _spec: &EmbeddingCallSpec<'_>,
    ) -> Result<bool> {
        Ok(false)
    }
}

#[derive(Debug, Default)]
pub struct QuantizationRegistry {
    definitions: Vec<Arc<dyn QuantizationDefinition>>,
}

impl QuantizationRegistry {
    pub const VERSION: &'static str = "2";

    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, definition: impl QuantizationDefinition + 'static) -> Result<()> {
        let key = definition.descriptor().key;
        if self
            .definitions
            .iter()
            .any(|existing| existing.descriptor().key == key)
        {
            return Err(DynInferError::internal(format!(
                "duplicate encoding definition `{}@{}`",
                key.id, key.version
            )));
        }
        self.definitions.push(Arc::new(definition));
        Ok(())
    }

    pub fn definitions(&self) -> &[Arc<dyn QuantizationDefinition>] {
        &self.definitions
    }

    pub fn resolve(&self, encoding: &PhysicalEncoding) -> Option<Arc<dyn QuantizationDefinition>> {
        self.definitions
            .iter()
            .find(|definition| definition.matches(encoding))
            .cloned()
    }

    pub fn register_kernel_candidates(&self, kernels: &mut KernelRegistry) -> Result<()> {
        for definition in &self.definitions {
            for candidate in definition.kernel_candidates() {
                kernels.register(candidate)?;
            }
        }
        Ok(())
    }
}

/// Explicit static registration for all physical encoding definitions.
pub fn register_all(registry: &mut QuantizationRegistry) -> Result<()> {
    registry.register(quantizations::plain::PlainDefinition::new(
        dyninfer_core::ScalarType::F32,
    ))?;
    registry.register(quantizations::plain::PlainDefinition::new(
        dyninfer_core::ScalarType::F16,
    ))?;
    registry.register(quantizations::plain::PlainDefinition::new(
        dyninfer_core::ScalarType::Bf16,
    ))?;
    registry.register(quantizations::q4_0::GgufQ40Definition)?;
    registry.register(quantizations::q4_1::GgufQ41Definition)?;
    registry.register(quantizations::q6_k::GgufQ6KDefinition)?;
    registry.register(quantizations::q8_0::GgufQ80Definition)?;
    for definition in quantizations::gguf_block::SCHEMA_ONLY_DEFINITIONS {
        registry.register(*definition)?;
    }
    for definition in quantizations::mlx_affine::SCHEMA_DEFINITIONS {
        registry.register(*definition)?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationCoverage {
    pub operation_id: OperationId,
    pub operation: KernelOperationKind,
    pub mode: ExecutionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<EncodingKey>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected: Option<SelectedKernelCandidate>,
    pub rejected: Vec<CandidateRejection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_shape: Option<Shape>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parameter_slots: Vec<ParameterSlotId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checkpoint_component_keys: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoverageReport {
    pub target: TargetProfile,
    pub target_fingerprint: String,
    pub precision_policy: PrecisionPolicy,
    pub operations: Vec<OperationCoverage>,
}

impl CoverageReport {
    pub fn is_complete(&self) -> bool {
        self.operations
            .iter()
            .all(|operation| operation.selected.is_some() && operation.validation_error.is_none())
    }

    pub fn selected_kernels(&self) -> Vec<SelectedKernel> {
        self.operations
            .iter()
            .filter_map(|operation| {
                operation.selected.as_ref().map(|selected| SelectedKernel {
                    operation_id: operation.operation_id.clone(),
                    mode: operation.mode,
                    kernel_id: selected.descriptor.id.clone(),
                    lowering_id: selected.descriptor.lowering.clone(),
                    parameter_slots: operation.parameter_slots.clone(),
                    input_type: selected.input_type,
                    output_type: selected.output_type,
                    activation_type: selected.activation_type,
                    accumulator_type: selected.accumulator_type,
                })
            })
            .collect()
    }

    pub fn require_complete(&self) -> Result<()> {
        if self.is_complete() {
            return Ok(());
        }
        let diagnostics = self
            .operations
            .iter()
            .filter(|operation| {
                operation.selected.is_none() || operation.validation_error.is_some()
            })
            .map(|operation| {
                let mut diagnostic = Diagnostic::error(
                    "E_KERNEL_COVERAGE",
                    operation.validation_error.clone().unwrap_or_else(|| {
                        format!(
                            "no production kernel selected for operation `{}` in {:?} mode",
                            operation.operation_id, operation.mode
                        )
                    }),
                );
                diagnostic.architecture_op = Some(operation.operation_id.to_string());
                diagnostic.parameter_slot =
                    operation.parameter_slots.first().map(ToString::to_string);
                diagnostic.checkpoint_key = operation.checkpoint_component_keys.first().cloned();
                diagnostic.expected = operation.encoding.as_ref().map(|encoding| {
                    format!(
                        "production {:?} kernel for {}@{} and shape {:?}",
                        operation.operation, encoding.id, encoding.version, operation.logical_shape
                    )
                });
                diagnostic.actual = Some(format!(
                    "target={} architecture={:?} triple={:?} features={:?} fingerprint={}",
                    self.target.driver,
                    self.target.architecture,
                    self.target.triple,
                    self.target.features,
                    self.target_fingerprint
                ));
                diagnostic.pass_name = Some("kernel.coverage".into());
                diagnostic.suggestion = Some(format!(
                    "add and qualify a matching kernel; rejected candidates: {:?}",
                    operation.rejected
                ));
                diagnostic
            })
            .collect();
        Err(DynInferError::Compilation(CompilationError {
            message: "bound model has incomplete production kernel coverage".into(),
            pass: Some("kernel.coverage".into()),
            diagnostics,
        }))
    }
}

/// Select kernels without generating MLIR. Every semantic operation is checked
/// independently for both exported execution modes.
pub fn dry_run_coverage(
    graph: &ArchitectureGraph,
    parameters: &[LogicalParameter],
    binding: &BindingPlan,
    encodings: &QuantizationRegistry,
    kernels: &KernelRegistry,
    target: &TargetProfile,
    precision_policy: &PrecisionPolicy,
) -> CoverageReport {
    let definitions_by_slot: BTreeMap<_, _> = binding
        .bindings
        .iter()
        .filter_map(|bound| {
            parameters
                .iter()
                .find(|parameter| parameter.canonical_name == bound.canonical_name)
                .map(|parameter| (bound.slot_id.clone(), parameter))
        })
        .collect();
    let bindings_by_slot: BTreeMap<_, _> = binding
        .bindings
        .iter()
        .map(|bound| (bound.slot_id.clone(), bound))
        .collect();
    let modes: BTreeSet<_> = graph.exports.iter().map(|export| export.mode).collect();
    let cost = DeterministicCostModel;
    let mut operations = Vec::new();

    for operation in &graph.operations {
        for mode in &modes {
            let parameter = operation
                .parameters
                .first()
                .and_then(|slot| definitions_by_slot.get(slot).copied());
            let bound = operation
                .parameters
                .first()
                .and_then(|slot| bindings_by_slot.get(slot).copied());
            let encoding = parameter.map(|parameter| physical_encoding_key(&parameter.encoding));
            let mut validation_error = None;
            if let Some(parameter) = parameter {
                match encodings.resolve(&parameter.encoding) {
                    Some(definition) => {
                        if let Err(error) = definition.validate(parameter) {
                            validation_error = Some(error.to_string());
                        }
                    }
                    None => {
                        let key = physical_encoding_key(&parameter.encoding);
                        validation_error = Some(format!(
                            "no encoding definition registered for `{}@{}`",
                            key.id, key.version
                        ));
                    }
                }
            } else if !operation.parameters.is_empty() {
                validation_error =
                    Some("operation parameter is not present in binding plan".into());
            }
            let orientation = bound
                .map(|bound| match bound.transform {
                    BindingTransform::LogicalTranspose { .. } => {
                        ParameterOrientation::LogicalTranspose
                    }
                    _ => ParameterOrientation::Native,
                })
                .unwrap_or(ParameterOrientation::Native);
            let request = KernelRequest {
                operation_id: operation.id.clone(),
                operation: KernelOperationKind::from_semantic(&operation.kind),
                encoding: encoding.clone(),
                logical_shape: bound.map(|bound| bound.logical_shape.clone()),
                orientation,
                mode: *mode,
                precision_policy: precision_policy.clone(),
                parameter_slots: operation.parameters.clone(),
                checkpoint_component_keys: bound
                    .map(|bound| {
                        bound
                            .components
                            .iter()
                            .map(|component| component.external_key.clone())
                            .collect()
                    })
                    .unwrap_or_default(),
            };
            let selection = kernels
                .select(&request, target, &cost)
                .unwrap_or_else(|error| {
                    validation_error = Some(error.to_string());
                    dyninfer_kernel_registry::SelectionReport {
                        selected: None,
                        rejected: vec![],
                    }
                });
            operations.push(OperationCoverage {
                operation_id: operation.id.clone(),
                operation: request.operation,
                mode: *mode,
                encoding,
                selected: selection.selected,
                rejected: selection.rejected,
                logical_shape: request.logical_shape,
                parameter_slots: request.parameter_slots,
                checkpoint_component_keys: request.checkpoint_component_keys,
                validation_error,
            });
        }
    }
    CoverageReport {
        target: target.clone(),
        target_fingerprint: target.capability_fingerprint.to_string(),
        precision_policy: precision_policy.clone(),
        operations,
    }
}

fn physical_encoding_key(encoding: &PhysicalEncoding) -> EncodingKey {
    match encoding {
        PhysicalEncoding::Plain { storage_type, .. } => {
            EncodingKey::new(format!("plain.{storage_type}"), 1)
        }
        PhysicalEncoding::BlockQuantized {
            codec,
            codec_version,
            ..
        }
        | PhysicalEncoding::Opaque {
            codec,
            codec_version,
            ..
        } => EncodingKey {
            id: EncodingId::new(codec.to_string()),
            version: *codec_version,
        },
        PhysicalEncoding::GroupQuantized { packing, .. } => EncodingKey::new(packing.clone(), 1),
        PhysicalEncoding::Sparse { format, .. } => EncodingKey::new(format.clone(), 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dyninfer_checkpoint::LogicalParameter;
    use dyninfer_core::{
        ArchitectureExport, ArchitectureId, ArchitectureOperation, BlockLayoutField, ByteRange,
        CanonicalParameterName, CodecId, Endianness, GraphValue, GraphValueId, LogicalTensorType,
        OperationKind, ParameterBinding, ParameterRole, ParameterSlot, ParameterSlotId, ScalarType,
        SchemaFingerprint, SemanticTensorType, Shape, StorageComponent, StorageElementType,
        TensorOrder,
    };

    fn dense_fixture() -> (ArchitectureGraph, Vec<LogicalParameter>, BindingPlan) {
        let slot_id = ParameterSlotId::new("weight");
        let graph = ArchitectureGraph {
            version: 1,
            architecture_id: ArchitectureId::new("test"),
            values: vec![
                GraphValue {
                    id: GraphValueId::new("input"),
                    tensor_type: SemanticTensorType::activations(64),
                },
                GraphValue {
                    id: GraphValueId::new("output"),
                    tensor_type: SemanticTensorType::activations(64),
                },
            ],
            operations: vec![ArchitectureOperation {
                id: OperationId::new("linear"),
                kind: OperationKind::Linear {
                    role: ParameterRole::AttentionQ,
                },
                inputs: vec![GraphValueId::new("input")],
                outputs: vec![GraphValueId::new("output")],
                parameters: vec![slot_id.clone()],
            }],
            parameter_slots: vec![ParameterSlot {
                id: slot_id.clone(),
                canonical_name: CanonicalParameterName::new("weight"),
                role: ParameterRole::AttentionQ,
                expected_type: dyninfer_core::LogicalTensorConstraint {
                    rank: Some(2),
                    shape: None,
                    element_types: vec![ScalarType::F16],
                },
                optional: false,
                tied_group: None,
            }],
            exports: vec![ArchitectureExport {
                name: "decode".into(),
                mode: ExecutionMode::Decode,
                inputs: vec![GraphValueId::new("input")],
                outputs: vec![GraphValueId::new("output")],
            }],
        };
        let parameter = LogicalParameter {
            canonical_name: CanonicalParameterName::new("weight"),
            role: ParameterRole::AttentionQ,
            logical_type: LogicalTensorType {
                shape: Shape::new(vec![64, 64]),
                element_type: ScalarType::F16,
            },
            encoding: PhysicalEncoding::Plain {
                storage_type: ScalarType::F16,
                order: TensorOrder::RowMajor,
            },
            components: vec![StorageComponent {
                name: "data".into(),
                key: "weight".into(),
                source_file_index: 0,
                shape: Shape::new(vec![64, 64]),
                storage_type: StorageElementType::scalar(ScalarType::F16),
                byte_ranges: vec![ByteRange::new(0, 64 * 64 * 2)],
                alignment: 1,
                endianness: Endianness::Little,
            }],
            aliases: vec!["weight".into()],
        };
        let binding = BindingPlan {
            architecture_id: ArchitectureId::new("test"),
            checkpoint_schema: SchemaFingerprint {
                digest: dyninfer_core::Digest::from_bytes(b"schema"),
                entry_count: 1,
                total_bytes: 64 * 64 * 2,
            },
            bindings: vec![ParameterBinding {
                slot_id,
                canonical_name: CanonicalParameterName::new("weight"),
                checkpoint_keys: vec!["weight".into()],
                encoding: parameter.encoding.clone(),
                logical_shape: Shape::new(vec![64, 64]),
                logical_type: ScalarType::F16,
                components: vec![dyninfer_core::ParameterComponentBinding {
                    component_name: "data".into(),
                    external_key: "weights::weight::data".into(),
                    checkpoint_key: "weight".into(),
                    source_file_index: 0,
                    shape: Shape::new(vec![64, 64]),
                    storage_type: StorageElementType::scalar(ScalarType::F16),
                    byte_lengths: vec![64 * 64 * 2],
                    alignment: 1,
                    endianness: Endianness::Little,
                }],
                transform: BindingTransform::Identity,
                scope: "weights".into(),
                parameter_key: "weight".into(),
                storage_bytes: 64 * 64 * 2,
                alignment: 1,
            }],
            unresolved_optional_slots: vec![],
        };
        (graph, vec![parameter], binding)
    }

    #[test]
    fn dense_f16_fixture_has_complete_coverage() {
        let mut encodings = QuantizationRegistry::new();
        register_all(&mut encodings).unwrap();
        let mut kernels = KernelRegistry::new();
        encodings.register_kernel_candidates(&mut kernels).unwrap();
        let (graph, parameters, binding) = dense_fixture();
        let report = dry_run_coverage(
            &graph,
            &parameters,
            &binding,
            &encodings,
            &kernels,
            &TargetProfile::llvm_cpu_host(),
            &PrecisionPolicy::default(),
        );
        report.require_complete().unwrap();
        assert_eq!(
            report.operations[0]
                .selected
                .as_ref()
                .unwrap()
                .descriptor
                .id
                .as_str(),
            "dense.matmul.f16.portable_f32"
        );
    }

    #[test]
    fn q4_0_direct_kernel_has_complete_coverage() {
        let mut encodings = QuantizationRegistry::new();
        register_all(&mut encodings).unwrap();
        let mut kernels = KernelRegistry::new();
        encodings.register_kernel_candidates(&mut kernels).unwrap();
        let (graph, mut parameters, mut binding) = dense_fixture();
        let q4_0 = PhysicalEncoding::BlockQuantized {
            logical_type: ScalarType::F16,
            block_shape: vec![32],
            bytes_per_block: 18,
            codec: CodecId::new("gguf.q4_0"),
            codec_version: 1,
            components: vec!["scale_f16".into(), "quants_u4".into()],
            layout: vec![
                BlockLayoutField {
                    name: "scale_f16".into(),
                    byte_offset: 0,
                    byte_length: 2,
                    storage_type: StorageElementType::scalar(ScalarType::F16),
                },
                BlockLayoutField {
                    name: "quants_u4".into(),
                    byte_offset: 2,
                    byte_length: 16,
                    storage_type: StorageElementType::Opaque {
                        codec: "packed.u4".into(),
                    },
                },
            ],
            order: TensorOrder::RowMajor,
            endianness: Endianness::Little,
        };
        parameters[0].encoding = q4_0.clone();
        parameters[0].components[0].storage_type = StorageElementType::Opaque {
            codec: "gguf.q4_0".into(),
        };
        parameters[0].components[0].byte_ranges = vec![ByteRange::new(0, 64 * 64 / 32 * 18)];
        binding.bindings[0].encoding = q4_0;
        binding.bindings[0].storage_bytes = 64 * 64 / 32 * 18;
        let report = dry_run_coverage(
            &graph,
            &parameters,
            &binding,
            &encodings,
            &kernels,
            &TargetProfile::llvm_cpu_host(),
            &PrecisionPolicy::default(),
        );
        report.require_complete().unwrap();
        assert_eq!(
            report.operations[0]
                .selected
                .as_ref()
                .unwrap()
                .descriptor
                .id
                .as_str(),
            "gguf.q4_0.matmul.iree_block_cpu"
        );
    }
}
