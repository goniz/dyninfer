//! GGUF container reader and Q4_0 / dense convention decoders.

#![forbid(unsafe_code)]

mod container;
mod convention;
mod q4;
mod types;

pub use container::GgufContainer;
pub use convention::{GgufDenseConvention, GgufQ40Convention};
pub use q4::{
    MetaValue, Q4_0_BLOCK, Q4_0_TYPE_SIZE, dequant_q4_0, fill_f32, pack_q4_0, q4_0_nbytes,
    tiny_llama_q4_0, write_gguf,
};
pub use types::GgufType;

use dyninfer_checkpoint::BuiltinCheckpointSupport;

/// Register GGUF support into a builtin registry.
pub fn register(support: &mut BuiltinCheckpointSupport) {
    support.register_container(GgufContainer::default());
    support.register_convention(GgufQ40Convention::default());
    support.register_convention(GgufDenseConvention::default());
}

/// Read GGUF parameters as host f32 blobs (Q4_0 dequantized via qkernel reference).
pub fn decode_parameters_as_f32_host(
    catalog: &dyninfer_checkpoint::CheckpointCatalog,
) -> dyninfer_error::Result<Vec<(String, Vec<u8>)>> {
    use dyninfer_checkpoint::{FileSource, RandomAccessSource};
    use dyninfer_core::{PhysicalEncoding, StorageElementType};
    use tracing::info;

    let source_path = catalog
        .source_files
        .first()
        .map(|f| f.path.as_path())
        .ok_or_else(|| dyninfer_error::DynInferError::io("checkpoint has no source file"))?;
    let source = FileSource::open(source_path)?;
    let mut entries = Vec::with_capacity(catalog.parameters.len());
    for param in &catalog.parameters {
        let comp = param.components.first().ok_or_else(|| {
            dyninfer_error::DynInferError::io(format!(
                "parameter {} has no storage component",
                param.canonical_name
            ))
        })?;
        let range = comp.byte_ranges.first().ok_or_else(|| {
            dyninfer_error::DynInferError::io(format!(
                "parameter {} has no byte range",
                param.canonical_name
            ))
        })?;
        let bytes = source.read_range(range.offset, range.length)?;
        let numel = param.logical_type.shape.numel().ok_or_else(|| {
            dyninfer_error::DynInferError::io(format!(
                "parameter {}: shape overflow",
                param.canonical_name
            ))
        })? as usize;

        let values = match &param.encoding {
            PhysicalEncoding::BlockQuantized { codec, .. } if codec.as_str() == "gguf.q4_0" => {
                dequant_q4_0(&bytes, numel)?
            }
            PhysicalEncoding::Plain { .. } => {
                let ty = match &comp.storage_type {
                    StorageElementType::Scalar { ty } => *ty,
                    other => {
                        return Err(dyninfer_error::DynInferError::io(format!(
                            "unsupported dense storage for {}: {other}",
                            param.canonical_name
                        )));
                    }
                };
                decode_dense_to_f32(&bytes, ty, numel)?
            }
            other => {
                return Err(dyninfer_error::DynInferError::io(format!(
                    "unsupported GGUF encoding for {}: {other:?}",
                    param.canonical_name
                )));
            }
        };
        let mut le = Vec::with_capacity(values.len() * 4);
        for v in values {
            le.extend_from_slice(&v.to_le_bytes());
        }
        entries.push((comp.key.clone(), le));
    }
    info!(
        params = entries.len(),
        "GGUF host parameters ready (Q4_0 dequantized via qkernel)"
    );
    Ok(entries)
}

fn decode_dense_to_f32(
    bytes: &[u8],
    ty: dyninfer_core::ScalarType,
    numel: usize,
) -> dyninfer_error::Result<Vec<f32>> {
    use dyninfer_core::ScalarType;
    use half::{bf16, f16};
    match ty {
        ScalarType::F32 => {
            if bytes.len() != numel * 4 {
                return Err(dyninfer_error::DynInferError::io(format!(
                    "f32 bytes {} != numel {numel} * 4",
                    bytes.len()
                )));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        ScalarType::F16 => {
            if bytes.len() != numel * 2 {
                return Err(dyninfer_error::DynInferError::io("f16 size mismatch"));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        ScalarType::Bf16 => {
            if bytes.len() != numel * 2 {
                return Err(dyninfer_error::DynInferError::io("bf16 size mismatch"));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        other => Err(dyninfer_error::DynInferError::io(format!(
            "unsupported dense scalar {other}"
        ))),
    }
}

/// Read packed / dense parameter bytes for IREE host binding from a GGUF catalog.
pub fn decode_parameters_as_host(
    catalog: &dyninfer_checkpoint::CheckpointCatalog,
) -> dyninfer_error::Result<Vec<(String, Vec<u8>)>> {
    decode_parameters_as_f32_host(catalog)
}
