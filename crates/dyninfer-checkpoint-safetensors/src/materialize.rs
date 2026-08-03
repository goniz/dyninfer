//! Materialize / resolve runtime SafeTensors for IREE `--parameters=`.

use crate::fixture::write_safetensors;
use dyninfer_checkpoint::{CheckpointCatalog, FileSource, RandomAccessSource};
use dyninfer_core::ScalarType;
use dyninfer_error::{DynInferError, Result};
use half::{bf16, f16};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

/// Prefer the original checkpoint file so IREE loads weights in their on-disk
/// dtypes (whatever the catalog reports) without a host-side expansion.
///
/// Requires the compiled module's `#stream.parameter.named` keys and element
/// types to match the checkpoint (`StorageComponent.key` / scalar dtype).
pub fn resolve_runtime_parameters(catalog: &CheckpointCatalog) -> Result<PathBuf> {
    let path = catalog
        .source_files
        .first()
        .map(|f| f.path.clone())
        .ok_or_else(|| DynInferError::io("checkpoint has no source file for parameters"))?;
    if !path.is_file() {
        return Err(DynInferError::io_path(
            path.display().to_string(),
            "parameter file missing",
        ));
    }
    info!(
        path = %path.display(),
        "using native checkpoint parameters (checkpoint dtypes, no host materialize)"
    );
    Ok(path)
}

/// Host-side bf16/f16→f32 expansion for IREE parameter bind (no disk write).
///
/// Keys are the on-disk SafeTensors / storage-component keys (what
/// `#stream.parameter.named<"weights"::…>` references), not canonical names.
/// Each value is little-endian f32 bytes (`numel * 4`).
///
/// Used when the VMFB was compiled with `--iree-input-promote-bf16-to-f32`
/// (Vulkan): the module expects f32 parameter blobs while the checkpoint file
/// remains bf16/f16.
pub fn decode_parameters_as_f32_host(catalog: &CheckpointCatalog) -> Result<HostF32Parameters> {
    let mut entries = Vec::with_capacity(catalog.parameters.len());
    let source_path = catalog
        .source_files
        .first()
        .map(|f| f.path.as_path())
        .ok_or_else(|| DynInferError::io("checkpoint has no source file for parameters"))?;
    let source = FileSource::open(source_path)?;

    for param in &catalog.parameters {
        let comp = param.components.first().ok_or_else(|| {
            DynInferError::io(format!(
                "parameter {} has no storage component",
                param.canonical_name
            ))
        })?;
        let range = comp.byte_ranges.first().ok_or_else(|| {
            DynInferError::io(format!(
                "parameter {} has no byte range",
                param.canonical_name
            ))
        })?;
        let bytes = source.read_range(range.offset, range.length)?;
        let ty = match &comp.storage_type {
            dyninfer_core::StorageElementType::Scalar { ty } => *ty,
            other => {
                return Err(DynInferError::io(format!(
                    "unsupported storage for {}: {other}",
                    param.canonical_name
                )));
            }
        };
        let values = decode_to_f32(&bytes, ty)?;
        let numel = param.logical_type.shape.numel().ok_or_else(|| {
            DynInferError::io(format!(
                "parameter {}: shape overflow",
                param.canonical_name
            ))
        })? as usize;
        if values.len() != numel {
            return Err(DynInferError::io(format!(
                "parameter {}: decoded {} values, expected {numel}",
                param.canonical_name,
                values.len()
            )));
        }
        let mut le = Vec::with_capacity(values.len() * 4);
        for v in values {
            le.extend_from_slice(&v.to_le_bytes());
        }
        entries.push((comp.key.clone(), le));
    }

    info!(
        tensors = entries.len(),
        "decoded checkpoint parameters to host f32 (in-memory, no disk materialize)"
    );
    Ok(HostF32Parameters { entries })
}

/// Owned f32 parameter blobs keyed for IREE `#stream.parameter.named`.
#[derive(Debug, Clone)]
pub struct HostF32Parameters {
    pub entries: Vec<(String, Vec<u8>)>,
}

/// Convert catalog parameters to an F32 SafeTensors file keyed by canonical names.
///
/// Output is written under `cache_dir` (created if needed) with a content hash
/// so repeated loads reuse the same file.
///
/// Prefer [`resolve_runtime_parameters`] when the VMFB binds native dtypes/keys,
/// or [`decode_parameters_as_f32_host`] when only an in-memory promote is needed.
pub fn materialize_f32_safetensors(
    catalog: &CheckpointCatalog,
    cache_dir: impl AsRef<Path>,
) -> Result<PathBuf> {
    let cache_dir = cache_dir.as_ref();
    fs::create_dir_all(cache_dir).map_err(|e| {
        DynInferError::io_path(cache_dir.display().to_string(), format!("mkdir: {e}"))
    })?;

    let digest = catalog.schema_fingerprint.digest.as_str();
    let out_path = cache_dir.join(format!("params-f32-{digest}.safetensors"));
    if out_path.is_file() {
        return Ok(out_path);
    }

    let source_path = catalog
        .source_files
        .first()
        .map(|f| f.path.as_path())
        .ok_or_else(|| DynInferError::io("checkpoint has no source file for materialization"))?;
    let source = FileSource::open(source_path)?;

    let mut tensors: BTreeMap<String, (Vec<u64>, Vec<f32>)> = BTreeMap::new();
    for param in &catalog.parameters {
        let comp = param.components.first().ok_or_else(|| {
            DynInferError::io(format!(
                "parameter {} has no storage component",
                param.canonical_name
            ))
        })?;
        let range = comp.byte_ranges.first().ok_or_else(|| {
            DynInferError::io(format!(
                "parameter {} has no byte range",
                param.canonical_name
            ))
        })?;
        let bytes = source.read_range(range.offset, range.length)?;
        let ty = match &comp.storage_type {
            dyninfer_core::StorageElementType::Scalar { ty } => *ty,
            other => {
                return Err(DynInferError::io(format!(
                    "unsupported storage for {}: {other}",
                    param.canonical_name
                )));
            }
        };
        let values = decode_to_f32(&bytes, ty)?;
        let numel = param.logical_type.shape.numel().ok_or_else(|| {
            DynInferError::io(format!(
                "parameter {}: shape overflow",
                param.canonical_name
            ))
        })? as usize;
        if values.len() != numel {
            return Err(DynInferError::io(format!(
                "parameter {}: decoded {} values, expected {numel}",
                param.canonical_name,
                values.len()
            )));
        }
        tensors.insert(
            param.canonical_name.to_string(),
            (param.logical_type.shape.dims().to_vec(), values),
        );
    }

    let meta = serde_json::to_value(&catalog.metadata).unwrap_or(serde_json::Value::Null);
    let blob = write_safetensors(&tensors, meta);
    let tmp = cache_dir.join(format!(
        "params-f32-{digest}.{}.tmp",
        hex::encode(Sha256::digest(&blob))
    ));
    fs::write(&tmp, &blob)
        .map_err(|e| DynInferError::io_path(tmp.display().to_string(), format!("write: {e}")))?;
    fs::rename(&tmp, &out_path).map_err(|e| {
        DynInferError::io_path(out_path.display().to_string(), format!("rename: {e}"))
    })?;
    info!(
        path = %out_path.display(),
        tensors = tensors.len(),
        "materialized f32 SafeTensors parameters"
    );
    Ok(out_path)
}

fn decode_to_f32(bytes: &[u8], ty: ScalarType) -> Result<Vec<f32>> {
    match ty {
        ScalarType::F32 => {
            if bytes.len() % 4 != 0 {
                return Err(DynInferError::io("F32 byte length not multiple of 4"));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        ScalarType::F16 => {
            if bytes.len() % 2 != 0 {
                return Err(DynInferError::io("F16 byte length not multiple of 2"));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        ScalarType::Bf16 => {
            if bytes.len() % 2 != 0 {
                return Err(DynInferError::io("BF16 byte length not multiple of 2"));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        other => Err(DynInferError::io(format!(
            "cannot materialize dtype {other} to f32"
        ))),
    }
}
