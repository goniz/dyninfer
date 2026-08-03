use crate::{CheckpointCatalog, ProviderParameterDescriptor, RuntimeProviderPlan};
use dyninfer_core::BindingPlan;
use dyninfer_error::{BindingError, DynInferError, Result};
use std::collections::BTreeSet;

/// Builds a direct, container-independent provider plan from bound logical
/// components. The plan contains paths and byte ranges only; it never reads a
/// checkpoint payload or creates derived storage.
pub fn build_runtime_provider_plan(
    catalog: &CheckpointCatalog,
    binding: &BindingPlan,
) -> Result<RuntimeProviderPlan> {
    if catalog.source_files.is_empty() {
        return Err(provider_error("checkpoint has no source files", None, None));
    }

    let mut parameters = Vec::new();
    let mut external_keys = BTreeSet::new();
    for bound in &binding.bindings {
        let parameter = catalog
            .parameters
            .iter()
            .find(|parameter| parameter.canonical_name == bound.canonical_name)
            .ok_or_else(|| {
                provider_error(
                    "bound logical parameter is absent from checkpoint catalog",
                    Some(bound.slot_id.to_string()),
                    Some(bound.canonical_name.to_string()),
                )
            })?;
        for component_binding in &bound.components {
            let component = parameter
                .components
                .iter()
                .find(|component| {
                    component.name == component_binding.component_name
                        && component.key == component_binding.checkpoint_key
                })
                .ok_or_else(|| {
                    provider_error(
                        "bound storage component is absent from checkpoint catalog",
                        Some(bound.slot_id.to_string()),
                        Some(component_binding.checkpoint_key.clone()),
                    )
                })?;
            if component.source_file_index != component_binding.source_file_index {
                return Err(provider_error(
                    "bound component source file differs from checkpoint catalog",
                    Some(bound.slot_id.to_string()),
                    Some(component.key.clone()),
                ));
            }
            let [range] = component.byte_ranges.as_slice() else {
                return Err(provider_error(
                    "direct parameter components must occupy one contiguous byte range",
                    Some(bound.slot_id.to_string()),
                    Some(component.key.clone()),
                ));
            };
            if component_binding.byte_lengths.as_slice() != [range.length] {
                return Err(provider_error(
                    "bound component length differs from checkpoint catalog",
                    Some(bound.slot_id.to_string()),
                    Some(component.key.clone()),
                ));
            }
            let source = catalog
                .source_files
                .get(component.source_file_index as usize)
                .ok_or_else(|| {
                    provider_error(
                        "component source_file_index is out of range",
                        Some(bound.slot_id.to_string()),
                        Some(component.key.clone()),
                    )
                })?;
            let end = range.offset.checked_add(range.length).ok_or_else(|| {
                provider_error(
                    "component byte range overflows",
                    Some(bound.slot_id.to_string()),
                    Some(component.key.clone()),
                )
            })?;
            if range.length == 0 || end > source.size_bytes {
                return Err(provider_error(
                    format!(
                        "component byte range [{}, {end}) exceeds source size {}",
                        range.offset, source.size_bytes
                    ),
                    Some(bound.slot_id.to_string()),
                    Some(component.key.clone()),
                ));
            }
            if !external_keys.insert(component_binding.external_key.clone()) {
                return Err(provider_error(
                    "duplicate stable external parameter key",
                    Some(bound.slot_id.to_string()),
                    Some(component_binding.external_key.clone()),
                ));
            }
            parameters.push(ProviderParameterDescriptor {
                external_key: component_binding.external_key.clone(),
                aliases: vec![component_binding.checkpoint_key.clone()],
                source_file_index: component.source_file_index,
                offset: range.offset,
                length: range.length,
            });
        }
    }
    parameters.sort_by(|left, right| left.external_key.cmp(&right.external_key));

    Ok(RuntimeProviderPlan {
        kind: "direct-file-ranges-v1".into(),
        scope: "weights".into(),
        file_paths: catalog
            .source_files
            .iter()
            .map(|source| source.path.clone())
            .collect(),
        parameters,
        notes: vec![
            "Entries reference original checkpoint files; aliases serve legacy MLIR keys".into(),
        ],
    })
}

fn provider_error(
    message: impl Into<String>,
    slot: Option<String>,
    checkpoint_key: Option<String>,
) -> DynInferError {
    DynInferError::Binding(BindingError {
        message: message.into(),
        slot,
        checkpoint_key,
        expected: None,
        actual: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContainerIdentity, LogicalParameter};
    use dyninfer_core::{
        ArchitectureId, BindingPlan, BindingTransform, ByteRange, CanonicalParameterName, CodecId,
        ContainerFormatId, ConventionId, Digest, Endianness, LogicalTensorType, ParameterBinding,
        ParameterComponentBinding, ParameterRole, ParameterSlotId, PhysicalEncoding, ScalarType,
        SchemaFingerprint, Shape, SourceFile, StorageComponent, StorageElementType,
    };
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn storage_component(
        name: &str,
        key: &str,
        source_file_index: u32,
        offset: u64,
        length: u64,
    ) -> StorageComponent {
        StorageComponent {
            name: name.into(),
            key: key.into(),
            source_file_index,
            shape: Shape::new(vec![length]),
            storage_type: StorageElementType::Opaque {
                codec: "test".into(),
            },
            byte_ranges: vec![ByteRange { offset, length }],
            alignment: 1,
            endianness: Endianness::Little,
        }
    }

    #[test]
    fn builds_multi_file_multi_component_plan_without_reading_payloads() {
        let components = vec![
            storage_component("data", "layer.weight", 0, 10, 20),
            storage_component("scales", "layer.scales", 1, 30, 8),
        ];
        let parameter = LogicalParameter {
            canonical_name: CanonicalParameterName::new("layer.weight"),
            role: ParameterRole::AttentionQ,
            logical_type: LogicalTensorType {
                shape: Shape::new(vec![4, 4]),
                element_type: ScalarType::F16,
            },
            encoding: PhysicalEncoding::Opaque {
                codec: CodecId::new("test.mixed"),
                codec_version: 1,
                descriptor: serde_json::json!({}),
            },
            components: components.clone(),
            aliases: vec![],
        };
        let component_bindings = components
            .iter()
            .map(|component| ParameterComponentBinding {
                component_name: component.name.clone(),
                external_key: format!("weights::layer.weight::{}", component.name),
                checkpoint_key: component.key.clone(),
                source_file_index: component.source_file_index,
                shape: component.shape.clone(),
                storage_type: component.storage_type.clone(),
                byte_lengths: vec![component.byte_ranges[0].length],
                alignment: component.alignment,
                endianness: component.endianness.clone(),
            })
            .collect();
        let binding = BindingPlan {
            architecture_id: ArchitectureId::new("test"),
            checkpoint_schema: SchemaFingerprint {
                digest: Digest::from_bytes(b"schema"),
                entry_count: 1,
                total_bytes: 28,
            },
            bindings: vec![ParameterBinding {
                slot_id: ParameterSlotId::new("layer.weight"),
                canonical_name: CanonicalParameterName::new("layer.weight"),
                checkpoint_keys: vec!["layer.weight".into()],
                encoding: parameter.encoding.clone(),
                logical_shape: parameter.logical_type.shape.clone(),
                logical_type: ScalarType::F16,
                components: component_bindings,
                transform: BindingTransform::Identity,
                scope: "weights".into(),
                parameter_key: "layer.weight".into(),
                storage_bytes: 28,
                alignment: 1,
            }],
            unresolved_optional_slots: vec![],
        };
        let catalog = CheckpointCatalog {
            container: ContainerIdentity {
                format_id: ContainerFormatId::new("test"),
                version: Some(1),
                magic: None,
            },
            convention_id: ConventionId::new("test.mixed"),
            source_files: vec![
                SourceFile {
                    path: PathBuf::from("shard-0.bin"),
                    size_bytes: 100,
                    content_digest: None,
                },
                SourceFile {
                    path: PathBuf::from("shard-1.bin"),
                    size_bytes: 100,
                    content_digest: None,
                },
            ],
            metadata: BTreeMap::new(),
            raw_entries: vec![],
            parameters: vec![parameter],
            schema_fingerprint: binding.checkpoint_schema.clone(),
        };

        let plan = build_runtime_provider_plan(&catalog, &binding).unwrap();
        assert_eq!(plan.kind, "direct-file-ranges-v1");
        assert_eq!(plan.file_paths.len(), 2);
        assert_eq!(plan.parameters.len(), 2);
        assert_eq!(plan.parameters[0].offset, 10);
        assert_eq!(plan.parameters[1].source_file_index, 1);
        assert_eq!(plan.parameters[1].aliases, ["layer.scales"]);
    }
}
