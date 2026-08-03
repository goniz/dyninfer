//! Schema fingerprint helpers for checkpoint catalogs.

use crate::catalog::LogicalParameter;
use dyninfer_core::{SchemaFingerprint, content_digest};
use dyninfer_error::Result;
use serde::Serialize;

#[derive(Serialize)]
struct FingerprintEntry<'a> {
    canonical_name: &'a str,
    role: String,
    shape: &'a [u64],
    element_type: String,
    encoding: &'a dyninfer_core::PhysicalEncoding,
    keys: Vec<&'a str>,
}

pub fn schema_fingerprint_from_parameters(
    parameters: &[LogicalParameter],
) -> Result<SchemaFingerprint> {
    let mut entries: Vec<FingerprintEntry<'_>> = parameters
        .iter()
        .map(|p| FingerprintEntry {
            canonical_name: p.canonical_name.as_str(),
            role: p.role.as_str().to_string(),
            shape: p.logical_type.shape.dims(),
            element_type: p.logical_type.element_type.to_string(),
            encoding: &p.encoding,
            keys: p.components.iter().map(|c| c.key.as_str()).collect(),
        })
        .collect();
    entries.sort_by(|a, b| a.canonical_name.cmp(b.canonical_name));

    let digest = content_digest(&entries)?;
    let mut total_bytes = 0u64;
    for p in parameters {
        for c in &p.components {
            for r in &c.byte_ranges {
                total_bytes = total_bytes.saturating_add(r.length);
            }
        }
    }

    Ok(SchemaFingerprint {
        digest,
        entry_count: parameters.len() as u64,
        total_bytes,
    })
}
