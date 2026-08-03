//! Name matching, shape validation, and materialization planning.

#![forbid(unsafe_code)]

use dyninfer_architecture::ArchitecturePackage;
use dyninfer_checkpoint::{CheckpointCatalog, LogicalParameter};
use dyninfer_core::{
    BindingPlan, BindingTransform, MaterializationPolicy, MaterializationRequest, ParameterBinding,
    ParameterRole, ParameterSlot, PhysicalEncoding, Shape,
};
use dyninfer_error::{BindingError, DynInferError, Result};
use std::collections::BTreeMap;
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
        let mut tied_shapes: BTreeMap<String, (String, Shape)> = BTreeMap::new();

        for slot in &architecture.parameter_slots {
            match self.bind_slot(slot, &by_name)? {
                Some((binding, mat)) => {
                    if let Some(group) = &slot.tied_group {
                        let group_key = group.to_string();
                        let shape = binding.logical_shape.clone();
                        if let Some((other_slot, other_shape)) = tied_shapes.get(&group_key) {
                            if other_shape != &shape {
                                return Err(DynInferError::Binding(BindingError {
                                    message: format!(
                                        "tied group `{group_key}` shape mismatch between `{other_slot}` and `{}`",
                                        slot.id
                                    ),
                                    slot: Some(slot.id.to_string()),
                                    checkpoint_key: Some(binding.parameter_key.clone()),
                                    expected: Some(other_shape.to_string()),
                                    actual: Some(shape.to_string()),
                                }));
                            }
                        } else {
                            tied_shapes.insert(group_key, (slot.id.to_string(), shape));
                        }
                    }
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

        if let Some(expected_shape) = &slot.expected_type.shape {
            if !param.logical_type.shape.is_compatible_with(expected_shape) {
                return Err(DynInferError::Binding(BindingError {
                    message: "parameter shape mismatch".into(),
                    slot: Some(slot.id.to_string()),
                    checkpoint_key: Some(param.canonical_name.to_string()),
                    expected: Some(expected_shape.to_string()),
                    actual: Some(param.logical_type.shape.to_string()),
                }));
            }
        }

        // Role mismatch is a hard error when the checkpoint declared a concrete role
        // (not Other) that disagrees with the architecture slot.
        if !roles_compatible(&slot.role, &param.role) {
            return Err(DynInferError::Binding(BindingError {
                message: "parameter role mismatch".into(),
                slot: Some(slot.id.to_string()),
                checkpoint_key: Some(param.canonical_name.to_string()),
                expected: Some(slot.role.as_str().into()),
                actual: Some(param.role.as_str().into()),
            }));
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
            return Err(DynInferError::Binding(BindingError {
                message: "encoding is not supported in version 1".into(),
                slot: Some(slot.id.to_string()),
                checkpoint_key: Some(param.canonical_name.to_string()),
                expected: Some("plain or gguf.q4_0".into()),
                actual: Some(format!("{:?}", param.encoding)),
            }));
        }

        let storage_bytes = param
            .components
            .iter()
            .flat_map(|c| c.byte_ranges.iter())
            .map(|r| r.length)
            .fold(0u64, |a, b| a.saturating_add(b));

        // For plain encodings with declared storage, bytes must match numel * elem size.
        // Q4_0 packed size is numel/32*18.
        if !param.components.is_empty() {
            match &param.encoding {
                PhysicalEncoding::Plain { storage_type, .. } => {
                    if let Some(elem) = storage_type.size_bytes() {
                        let Some(numel) = param.logical_type.shape.numel() else {
                            return Err(DynInferError::Binding(BindingError {
                                message: "parameter shape numel overflow".into(),
                                slot: Some(slot.id.to_string()),
                                checkpoint_key: Some(param.canonical_name.to_string()),
                                expected: None,
                                actual: Some(param.logical_type.shape.to_string()),
                            }));
                        };
                        let expected = numel.saturating_mul(u64::from(elem));
                        if storage_bytes != expected {
                            return Err(DynInferError::Binding(BindingError {
                                message: "parameter storage byte length mismatch".into(),
                                slot: Some(slot.id.to_string()),
                                checkpoint_key: Some(param.canonical_name.to_string()),
                                expected: Some(format!("{expected} bytes")),
                                actual: Some(format!("{storage_bytes} bytes")),
                            }));
                        }
                    }
                }
                PhysicalEncoding::BlockQuantized { codec, .. } if codec.as_str() == "gguf.q4_0" => {
                    let Some(numel) = param.logical_type.shape.numel() else {
                        return Err(DynInferError::Binding(BindingError {
                            message: "parameter shape numel overflow".into(),
                            slot: Some(slot.id.to_string()),
                            checkpoint_key: Some(param.canonical_name.to_string()),
                            expected: None,
                            actual: Some(param.logical_type.shape.to_string()),
                        }));
                    };
                    if !numel.is_multiple_of(32) {
                        return Err(DynInferError::Binding(BindingError {
                            message: "Q4_0 numel must be divisible by 32".into(),
                            slot: Some(slot.id.to_string()),
                            checkpoint_key: Some(param.canonical_name.to_string()),
                            expected: Some("numel % 32 == 0".into()),
                            actual: Some(format!("numel {numel}")),
                        }));
                    }
                    let expected = (numel / 32) * 18;
                    if storage_bytes != expected {
                        return Err(DynInferError::Binding(BindingError {
                            message: "Q4_0 storage byte length mismatch".into(),
                            slot: Some(slot.id.to_string()),
                            checkpoint_key: Some(param.canonical_name.to_string()),
                            expected: Some(format!("{expected} bytes")),
                            actual: Some(format!("{storage_bytes} bytes")),
                        }));
                    }
                }
                _ => {}
            }
        }

        let alignment = param.components.first().map(|c| c.alignment).unwrap_or(1);
        if alignment == 0 {
            return Err(DynInferError::Binding(BindingError {
                message: "parameter alignment must be non-zero".into(),
                slot: Some(slot.id.to_string()),
                checkpoint_key: Some(param.canonical_name.to_string()),
                expected: Some("alignment >= 1".into()),
                actual: Some("0".into()),
            }));
        }
        for comp in &param.components {
            for range in &comp.byte_ranges {
                if range.offset % alignment != 0 {
                    return Err(DynInferError::Binding(BindingError {
                        message: "parameter byte range is misaligned".into(),
                        slot: Some(slot.id.to_string()),
                        checkpoint_key: Some(param.canonical_name.to_string()),
                        expected: Some(format!("offset % {alignment} == 0")),
                        actual: Some(format!("offset {}", range.offset)),
                    }));
                }
            }
        }

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
            alignment,
        };
        Ok(Some((binding, mat_req)))
    }
}

fn roles_compatible(slot: &ParameterRole, param: &ParameterRole) -> bool {
    match (slot, param) {
        (_, ParameterRole::Other(_)) => true,
        (ParameterRole::Other(_), _) => true,
        (a, b) => a == b,
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
    slot.supported_encodings
        .iter()
        .any(|e| e == &name || e == "plain" && matches!(encoding, PhysicalEncoding::Plain { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dyninfer_architecture::{ArchitecturePackage, ResolvedModelConfig};
    use dyninfer_checkpoint::{CheckpointCatalog, ContainerIdentity, LogicalParameter};
    use dyninfer_core::{
        ArchitectureId, CanonicalParameterName, ContainerFormatId, ConventionId, Digest,
        LogicalTensorConstraint, LogicalTensorType, ParameterRole, ParameterSlotId,
        PhysicalEncoding, ScalarType, SchemaFingerprint, Shape,
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
            resolved_config: ResolvedModelConfig {
                values: BTreeMap::new(),
            },
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
    fn binds_q4_0_weights() {
        let slot = ParameterSlot {
            id: ParameterSlotId::new("q"),
            canonical_name: CanonicalParameterName::new("blk.0.attn_q.weight"),
            role: ParameterRole::AttentionQ,
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
        let numel = 64u64 * 64;
        let nbytes = numel / 32 * 18;
        let param = LogicalParameter {
            canonical_name: CanonicalParameterName::new("blk.0.attn_q.weight"),
            role: ParameterRole::AttentionQ,
            logical_type: LogicalTensorType {
                shape: Shape::new(vec![64, 64]),
                element_type: ScalarType::F16,
            },
            encoding: PhysicalEncoding::gguf_q4_0(),
            components: vec![dyninfer_core::StorageComponent {
                name: "data".into(),
                key: "blk.0.attn_q.weight".into(),
                shape: Shape::new(vec![64, 64]),
                storage_type: dyninfer_core::StorageElementType::Opaque {
                    codec: "gguf.q4_0".into(),
                },
                byte_ranges: vec![dyninfer_core::ByteRange::new(0, nbytes)],
                alignment: 32,
                endianness: dyninfer_core::Endianness::Little,
            }],
            aliases: vec!["blk.0.attn_q.weight".into()],
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
                total_bytes: nbytes,
            },
        };
        let plan = Binder::default().bind(&arch, &catalog).unwrap();
        assert_eq!(plan.bindings.len(), 1);
        assert_eq!(
            plan.bindings[0].materialization,
            MaterializationPolicy::DecodeOnTheFly
        );
    }
}
