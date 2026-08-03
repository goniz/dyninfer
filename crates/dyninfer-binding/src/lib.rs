//! Name matching, shape validation, and materialization planning.

#![forbid(unsafe_code)]

use dyninfer_architecture::ArchitecturePackage;
use dyninfer_checkpoint::{CheckpointCatalog, LogicalParameter};
use dyninfer_core::{
    BindingPlan, BindingTransform, MaterializationPolicy, MaterializationRequest, ParameterBinding,
    ParameterSlot, PhysicalEncoding,
};
use dyninfer_error::{BindingError, DynInferError, Result};
use std::collections::{BTreeMap, BTreeSet};
use tracing::info_span;

pub struct Binder {
    pub parameter_scope: String,
}

impl Default for Binder {
    fn default() -> Self {
        Self {
            parameter_scope: "weights".into(),
        }
    }
}

impl Binder {
    pub fn bind(
        &self,
        architecture: &ArchitecturePackage,
        catalog: &CheckpointCatalog,
    ) -> Result<BindingPlan> {
        let _span = info_span!(
            "binding.resolve",
            architecture = %architecture.id,
            params = catalog.parameters.len()
        )
        .entered();

        let mut by_name: BTreeMap<&str, &LogicalParameter> = BTreeMap::new();
        for p in &catalog.parameters {
            by_name.insert(p.canonical_name.as_str(), p);
            for alias in &p.aliases {
                by_name.insert(alias.as_str(), p);
            }
        }

        let mut bindings = Vec::new();
        let mut unresolved_optional = Vec::new();
        let mut materializations = Vec::new();
        let mut used = BTreeSet::new();

        for slot in &architecture.parameter_slots {
            match self.bind_slot(slot, &by_name)? {
                Some((binding, mat)) => {
                    used.insert(binding.parameter_key.clone());
                    if let Some(m) = mat {
                        materializations.push(m);
                    }
                    bindings.push(binding);
                }
                None => {
                    if slot.optional {
                        unresolved_optional.push(slot.id.clone());
                    } else {
                        return Err(DynInferError::Binding(BindingError {
                            message: format!(
                                "required parameter slot `{}` not found in checkpoint",
                                slot.canonical_name
                            ),
                            slot: Some(slot.id.to_string()),
                            checkpoint_key: None,
                            expected: Some(slot.canonical_name.to_string()),
                            actual: None,
                        }));
                    }
                }
            }
        }

        Ok(BindingPlan {
            architecture_id: architecture.id.clone(),
            checkpoint_schema: catalog.schema_fingerprint.clone(),
            bindings,
            unresolved_optional_slots: unresolved_optional,
            materializations,
        })
    }

    fn bind_slot(
        &self,
        slot: &ParameterSlot,
        by_name: &BTreeMap<&str, &LogicalParameter>,
    ) -> Result<Option<(ParameterBinding, Option<MaterializationRequest>)>> {
        let Some(param) = by_name.get(slot.canonical_name.as_str()).copied() else {
            return Ok(None);
        };

        if let Some(rank) = slot.expected_type.rank {
            if param.logical_type.shape.rank() != rank {
                return Err(DynInferError::Binding(BindingError {
                    message: "parameter rank mismatch".into(),
                    slot: Some(slot.id.to_string()),
                    checkpoint_key: Some(param.aliases.first().cloned().unwrap_or_default()),
                    expected: Some(format!("rank {rank}")),
                    actual: Some(format!("rank {}", param.logical_type.shape.rank())),
                }));
            }
        }

        if !slot.expected_type.element_types.is_empty()
            && !slot
                .expected_type
                .element_types
                .contains(&param.logical_type.element_type)
        {
            // Encoding may remap logical type (e.g. Q4_0 -> f16); allow if encoding supported.
            if !encoding_allowed(slot, &param.encoding) {
                return Err(DynInferError::Binding(BindingError {
                    message: "parameter element type mismatch".into(),
                    slot: Some(slot.id.to_string()),
                    checkpoint_key: Some(param.canonical_name.to_string()),
                    expected: Some(format!("{:?}", slot.expected_type.element_types)),
                    actual: Some(param.logical_type.element_type.to_string()),
                }));
            }
        }

        if !encoding_allowed(slot, &param.encoding) {
            return Err(DynInferError::Binding(BindingError {
                message: "parameter encoding not supported for slot".into(),
                slot: Some(slot.id.to_string()),
                checkpoint_key: Some(param.canonical_name.to_string()),
                expected: Some(slot.supported_encodings.join("|")),
                actual: Some(format!("{:?}", param.encoding)),
            }));
        }

        if !param.encoding.is_supported_v1() {
            let message = if param.encoding.is_planned_v1() {
                "encoding gguf.q4_0 requires qkernel lowering (Milestone 2); \
                 refuse to bind rather than silently emit dense f32"
                    .into()
            } else {
                "encoding is not supported in version 1".into()
            };
            return Err(DynInferError::Binding(BindingError {
                message,
                slot: Some(slot.id.to_string()),
                checkpoint_key: Some(param.canonical_name.to_string()),
                expected: Some("plain".into()),
                actual: Some(format!("{:?}", param.encoding)),
            }));
        }

        let storage_bytes = param
            .components
            .iter()
            .flat_map(|c| c.byte_ranges.iter())
            .map(|r| r.length)
            .fold(0u64, |a, b| a.saturating_add(b));

        let materialization = match &param.encoding {
            PhysicalEncoding::Plain { .. } => MaterializationPolicy::DirectView,
            PhysicalEncoding::BlockQuantized { .. } => MaterializationPolicy::DecodeOnTheFly,
            _ => MaterializationPolicy::PrepackToCache,
        };

        let mat_req = if materialization == MaterializationPolicy::PrepackToCache {
            Some(MaterializationRequest {
                slot_id: slot.id.clone(),
                policy: materialization,
                reason: "unsupported direct encoding requires derived cache".into(),
            })
        } else {
            None
        };

        let key = param
            .components
            .first()
            .map(|c| c.key.clone())
            .unwrap_or_else(|| param.canonical_name.to_string());

        let binding = ParameterBinding {
            slot_id: slot.id.clone(),
            canonical_name: slot.canonical_name.clone(),
            checkpoint_keys: param.aliases.clone(),
            encoding: param.encoding.clone(),
            logical_shape: param.logical_type.shape.clone(),
            logical_type: param.logical_type.element_type,
            transform: BindingTransform::Identity,
            materialization,
            scope: self.parameter_scope.clone(),
            parameter_key: key,
            storage_bytes,
            alignment: param.components.first().map(|c| c.alignment).unwrap_or(1),
        };
        Ok(Some((binding, mat_req)))
    }
}

fn encoding_allowed(slot: &ParameterSlot, encoding: &PhysicalEncoding) -> bool {
    if slot.supported_encodings.is_empty() {
        return encoding.is_supported_v1();
    }
    let name = match encoding {
        PhysicalEncoding::Plain { .. } => "plain".to_string(),
        PhysicalEncoding::BlockQuantized { codec, .. } => codec.to_string(),
        PhysicalEncoding::GroupQuantized { packing, .. } => packing.clone(),
        PhysicalEncoding::Opaque { codec, .. } => codec.to_string(),
        PhysicalEncoding::Sparse { format, .. } => format.clone(),
    };
    slot.supported_encodings.iter().any(|e| e == &name || e == "plain" && matches!(encoding, PhysicalEncoding::Plain { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dyninfer_architecture::{ArchitecturePackage, ResolvedModelConfig};
    use dyninfer_checkpoint::{CheckpointCatalog, ContainerIdentity, LogicalParameter};
    use dyninfer_core::{
        ArchitectureId, CanonicalParameterName, Digest, LogicalTensorConstraint, LogicalTensorType,
        ParameterRole, ParameterSlotId, PhysicalEncoding, ScalarType, SchemaFingerprint, Shape,
        ContainerFormatId, ConventionId,
    };
    use std::collections::BTreeMap;

    #[test]
    fn binds_matching_name() {
        let slot = ParameterSlot {
            id: ParameterSlotId::new("tok"),
            canonical_name: CanonicalParameterName::new("token_embd.weight"),
            role: ParameterRole::Embedding,
            expected_type: LogicalTensorConstraint {
                rank: Some(2),
                shape: None,
                element_types: vec![ScalarType::F32],
            },
            supported_encodings: vec!["plain".into()],
            optional: false,
            tied_group: None,
        };
        let arch = ArchitecturePackage {
            id: ArchitectureId::new("llama.decoder"),
            revision: "0.1.0".into(),
            mlir_text: String::new(),
            parameter_slots: vec![slot],
            resolved_config: ResolvedModelConfig { values: BTreeMap::new() },
        };
        let param = LogicalParameter {
            canonical_name: CanonicalParameterName::new("token_embd.weight"),
            role: ParameterRole::Embedding,
            logical_type: LogicalTensorType {
                shape: Shape::new(vec![32, 16]),
                element_type: ScalarType::F32,
            },
            encoding: PhysicalEncoding::plain(ScalarType::F32),
            components: vec![],
            aliases: vec!["token_embd.weight".into()],
        };
        let catalog = CheckpointCatalog {
            container: ContainerIdentity {
                format_id: ContainerFormatId::new("safetensors"),
                version: Some(1),
                magic: None,
            },
            convention_id: ConventionId::new("safetensors.dense"),
            source_files: vec![],
            metadata: Default::default(),
            raw_entries: vec![],
            parameters: vec![param],
            schema_fingerprint: SchemaFingerprint {
                digest: Digest::from_bytes(b"x"),
                entry_count: 1,
                total_bytes: 0,
            },
        };
        let plan = Binder::default().bind(&arch, &catalog).unwrap();
        assert_eq!(plan.bindings.len(), 1);
    }

    #[test]
    fn rejects_q4_0_until_qkernel() {
        let slot = ParameterSlot {
            id: ParameterSlotId::new("tok"),
            canonical_name: CanonicalParameterName::new("token_embd.weight"),
            role: ParameterRole::Embedding,
            expected_type: LogicalTensorConstraint {
                rank: Some(2),
                shape: None,
                element_types: vec![ScalarType::F16],
            },
            supported_encodings: vec!["plain".into(), "gguf.q4_0".into()],
            optional: false,
            tied_group: None,
        };
        let arch = ArchitecturePackage {
            id: ArchitectureId::new("llama.decoder"),
            revision: "0.1.0".into(),
            mlir_text: String::new(),
            parameter_slots: vec![slot],
            resolved_config: ResolvedModelConfig {
                values: BTreeMap::new(),
            },
        };
        let param = LogicalParameter {
            canonical_name: CanonicalParameterName::new("token_embd.weight"),
            role: ParameterRole::Embedding,
            logical_type: LogicalTensorType {
                shape: Shape::new(vec![32, 16]),
                element_type: ScalarType::F16,
            },
            encoding: PhysicalEncoding::gguf_q4_0(),
            components: vec![],
            aliases: vec!["token_embd.weight".into()],
        };
        let catalog = CheckpointCatalog {
            container: ContainerIdentity {
                format_id: ContainerFormatId::new("gguf"),
                version: Some(3),
                magic: None,
            },
            convention_id: ConventionId::new("gguf.q4_0"),
            source_files: vec![],
            metadata: Default::default(),
            raw_entries: vec![],
            parameters: vec![param],
            schema_fingerprint: SchemaFingerprint {
                digest: Digest::from_bytes(b"x"),
                entry_count: 1,
                total_bytes: 0,
            },
        };
        let err = Binder::default().bind(&arch, &catalog).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("qkernel") || msg.contains("gguf.q4_0"),
            "unexpected error: {msg}"
        );
    }
}
