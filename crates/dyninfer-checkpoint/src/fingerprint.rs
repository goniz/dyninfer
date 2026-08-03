//! Code-relevant, value-independent checkpoint schema fingerprints.

use crate::catalog::LogicalParameter;
use dyninfer_core::{
    Endianness, ParameterRole, PhysicalEncoding, SchemaFingerprint, StorageElementType,
    content_digest,
};
use dyninfer_error::Result;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Serialize)]
struct FingerprintComponent<'a> {
    name: &'a str,
    shape: &'a [u64],
    storage_type: &'a StorageElementType,
    byte_lengths: Vec<u64>,
    alignment: u64,
    endianness: &'a Endianness,
    /// Stable label for shared storage. Raw keys and offsets are intentionally
    /// excluded; only the fact that two logical components alias is relevant.
    #[serde(skip_serializing_if = "Option::is_none")]
    alias_group: Option<&'a str>,
}

#[derive(Serialize)]
struct FingerprintEntry<'a> {
    canonical_name: &'a str,
    role: &'a ParameterRole,
    logical_shape: &'a [u64],
    logical_element_type: dyninfer_core::ScalarType,
    encoding: &'a PhysicalEncoding,
    components: Vec<FingerprintComponent<'a>>,
}

pub fn schema_fingerprint_from_parameters(
    parameters: &[LogicalParameter],
) -> Result<SchemaFingerprint> {
    // Discover storage aliases using raw identities, then serialize only a
    // canonical semantic label for each shared group. This preserves tied
    // weights without making paths, keys, or physical offsets code-relevant.
    let mut identities: BTreeMap<(u32, String, Vec<(u64, u64)>), Vec<String>> = BTreeMap::new();
    for parameter in parameters {
        for component in &parameter.components {
            identities
                .entry((
                    component.source_file_index,
                    component.key.clone(),
                    component
                        .byte_ranges
                        .iter()
                        .map(|range| (range.offset, range.length))
                        .collect(),
                ))
                .or_default()
                .push(format!("{}::{}", parameter.canonical_name, component.name));
        }
    }
    let alias_labels: BTreeMap<_, _> = identities
        .into_iter()
        .filter_map(|(identity, mut members)| {
            if members.len() < 2 {
                return None;
            }
            members.sort();
            Some((identity, members[0].clone()))
        })
        .collect();

    let mut entries: Vec<FingerprintEntry<'_>> = parameters
        .iter()
        .map(|parameter| FingerprintEntry {
            canonical_name: parameter.canonical_name.as_str(),
            role: &parameter.role,
            logical_shape: parameter.logical_type.shape.dims(),
            logical_element_type: parameter.logical_type.element_type,
            encoding: &parameter.encoding,
            components: parameter
                .components
                .iter()
                .map(|component| {
                    let identity = (
                        component.source_file_index,
                        component.key.clone(),
                        component
                            .byte_ranges
                            .iter()
                            .map(|range| (range.offset, range.length))
                            .collect(),
                    );
                    FingerprintComponent {
                        name: &component.name,
                        shape: component.shape.dims(),
                        storage_type: &component.storage_type,
                        byte_lengths: component
                            .byte_ranges
                            .iter()
                            .map(|range| range.length)
                            .collect(),
                        alignment: component.alignment,
                        endianness: &component.endianness,
                        alias_group: alias_labels.get(&identity).map(String::as_str),
                    }
                })
                .collect(),
        })
        .collect();
    entries.sort_by(|left, right| left.canonical_name.cmp(right.canonical_name));

    let digest = content_digest(&entries)?;
    let total_bytes = parameters
        .iter()
        .flat_map(|parameter| &parameter.components)
        .flat_map(|component| &component.byte_ranges)
        .fold(0u64, |total, range| total.saturating_add(range.length));

    Ok(SchemaFingerprint {
        digest,
        entry_count: parameters.len() as u64,
        total_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dyninfer_core::{
        ByteRange, CanonicalParameterName, LogicalTensorType, ScalarType, Shape, StorageComponent,
    };

    fn parameter(name: &str, key: &str, offset: u64) -> LogicalParameter {
        LogicalParameter {
            canonical_name: CanonicalParameterName::new(name),
            role: ParameterRole::AttentionQ,
            logical_type: LogicalTensorType {
                shape: Shape::new(vec![4, 8]),
                element_type: ScalarType::F16,
            },
            encoding: PhysicalEncoding::plain(ScalarType::F16),
            components: vec![StorageComponent {
                name: "data".into(),
                key: key.into(),
                source_file_index: 0,
                shape: Shape::new(vec![4, 8]),
                storage_type: StorageElementType::scalar(ScalarType::F16),
                byte_ranges: vec![ByteRange::new(offset, 64)],
                alignment: 1,
                endianness: Endianness::Little,
            }],
            aliases: vec![key.into()],
        }
    }

    #[test]
    fn ignores_raw_keys_offsets_and_values() {
        let first = schema_fingerprint_from_parameters(&[parameter("weight", "a", 128)]).unwrap();
        let second =
            schema_fingerprint_from_parameters(&[parameter("weight", "renamed", 4096)]).unwrap();
        assert_eq!(first.digest, second.digest);
    }

    #[test]
    fn component_contract_changes_are_code_relevant() {
        let baseline = parameter("weight", "a", 128);
        let mut changed = baseline.clone();
        changed.components[0].alignment = 32;
        let first = schema_fingerprint_from_parameters(&[baseline]).unwrap();
        let second = schema_fingerprint_from_parameters(&[changed]).unwrap();
        assert_ne!(first.digest, second.digest);
    }

    #[test]
    fn tied_storage_relationship_is_code_relevant_but_key_is_not() {
        let tied_a = vec![parameter("a", "shared", 128), parameter("b", "shared", 128)];
        let tied_b = vec![parameter("a", "other", 4096), parameter("b", "other", 4096)];
        let untied = vec![parameter("a", "a", 128), parameter("b", "b", 256)];
        let first = schema_fingerprint_from_parameters(&tied_a).unwrap();
        let second = schema_fingerprint_from_parameters(&tied_b).unwrap();
        let third = schema_fingerprint_from_parameters(&untied).unwrap();
        assert_eq!(first.digest, second.digest);
        assert_ne!(first.digest, third.digest);
    }
}
