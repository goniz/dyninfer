//! SafeTensors container indexing.

use dyninfer_checkpoint::{
    CheckpointContainerReader, ContainerIdentity, InspectionLimits, ProbeScore, RandomAccessSource,
    RawCheckpointIndex, RawTensorEntry, RuntimeProviderPlan,
};
use dyninfer_core::{
    ByteRange, ContainerFormatId, Endianness, MetadataMap, ScalarType, SourceFile, StorageElementType,
};
use dyninfer_error::{CheckpointValidationError, DynInferError, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use tracing::debug;

#[derive(Debug, Default)]
pub struct SafeTensorsContainer;

#[derive(Debug, Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

fn parse_dtype(dtype: &str) -> Result<ScalarType> {
    match dtype {
        "F32" => Ok(ScalarType::F32),
        "F16" => Ok(ScalarType::F16),
        "BF16" => Ok(ScalarType::Bf16),
        "F64" => Ok(ScalarType::F64),
        "I8" => Ok(ScalarType::I8),
        "I16" => Ok(ScalarType::I16),
        "I32" => Ok(ScalarType::I32),
        "I64" => Ok(ScalarType::I64),
        "U8" => Ok(ScalarType::U8),
        "U16" => Ok(ScalarType::U16),
        "U32" => Ok(ScalarType::U32),
        "U64" => Ok(ScalarType::U64),
        "BOOL" => Ok(ScalarType::Bool),
        other => Err(DynInferError::InvalidCheckpoint(CheckpointValidationError {
            message: format!("unsupported SafeTensors dtype: {other}"),
            key: None,
            detail: None,
        })),
    }
}

impl CheckpointContainerReader for SafeTensorsContainer {
    fn format_id(&self) -> ContainerFormatId {
        ContainerFormatId::new("safetensors")
    }

    fn probe(&self, source: &dyn RandomAccessSource) -> Result<ProbeScore> {
        if source.len() < 8 {
            return Ok(ProbeScore::NONE);
        }
        let mut hdr_len_buf = [0u8; 8];
        source.read_exact_at(0, &mut hdr_len_buf)?;
        let header_len = u64::from_le_bytes(hdr_len_buf);
        if header_len == 0 || header_len > 64 * 1024 * 1024 {
            return Ok(ProbeScore::NONE);
        }
        if source.len() < 8 + header_len {
            return Ok(ProbeScore::NONE);
        }
        let header = source.read_range(8, header_len.min(64))?;
        if header.first() == Some(&b'{') {
            // Stronger score when path ends with .safetensors
            if source
                .path()
                .and_then(|p| p.extension())
                .is_some_and(|e| e == "safetensors")
            {
                return Ok(ProbeScore::STRONG);
            }
            return Ok(ProbeScore { score: 80 });
        }
        Ok(ProbeScore::NONE)
    }

    fn index(
        &self,
        source: Arc<dyn RandomAccessSource>,
        limits: &InspectionLimits,
    ) -> Result<RawCheckpointIndex> {
        let mut hdr_len_buf = [0u8; 8];
        source.read_exact_at(0, &mut hdr_len_buf)?;
        let header_len = u64::from_le_bytes(hdr_len_buf);
        if header_len == 0 || header_len > limits.max_header_bytes {
            return Err(DynInferError::InvalidCheckpoint(CheckpointValidationError {
                message: format!("invalid SafeTensors header length: {header_len}"),
                key: None,
                detail: None,
            }));
        }
        if source.len() < 8 + header_len {
            return Err(DynInferError::InvalidCheckpoint(CheckpointValidationError {
                message: "SafeTensors header extends past end of file".into(),
                key: None,
                detail: None,
            }));
        }

        let header_bytes = source.read_range(8, header_len)?;
        let header_value: Value = serde_json::from_slice(&header_bytes).map_err(|e| {
            DynInferError::InvalidCheckpoint(CheckpointValidationError {
                message: format!("SafeTensors header JSON invalid: {e}"),
                key: None,
                detail: None,
            })
        })?;
        let obj = header_value.as_object().ok_or_else(|| {
            DynInferError::InvalidCheckpoint(CheckpointValidationError {
                message: "SafeTensors header must be a JSON object".into(),
                key: None,
                detail: None,
            })
        })?;

        let mut metadata: MetadataMap = BTreeMap::new();
        let mut entries = Vec::new();
        let data_base = 8 + header_len;

        for (key, value) in obj {
            if key == "__metadata__" {
                if let Some(meta_obj) = value.as_object() {
                    for (mk, mv) in meta_obj {
                        metadata.insert(mk.clone(), mv.clone());
                    }
                }
                continue;
            }
            limits.validate_key(key)?;
            let info: TensorInfo = serde_json::from_value(value.clone()).map_err(|e| {
                DynInferError::InvalidCheckpoint(CheckpointValidationError {
                    message: format!("invalid tensor info: {e}"),
                    key: Some(key.clone()),
                    detail: None,
                })
            })?;
            limits.validate_shape(&info.shape)?;
            if info.data_offsets[1] < info.data_offsets[0] {
                return Err(DynInferError::InvalidCheckpoint(CheckpointValidationError {
                    message: "data_offsets end < start".into(),
                    key: Some(key.clone()),
                    detail: None,
                }));
            }
            let length = info.data_offsets[1] - info.data_offsets[0];
            let abs_offset = data_base
                .checked_add(info.data_offsets[0])
                .ok_or_else(|| {
                    DynInferError::InvalidCheckpoint(CheckpointValidationError {
                        message: "tensor offset overflow".into(),
                        key: Some(key.clone()),
                        detail: None,
                    })
                })?;
            if abs_offset.checked_add(length).unwrap_or(u64::MAX) > source.len() {
                return Err(DynInferError::InvalidCheckpoint(CheckpointValidationError {
                    message: "tensor data range extends past end of file".into(),
                    key: Some(key.clone()),
                    detail: None,
                }));
            }

            let ty = parse_dtype(&info.dtype)?;
            entries.push(RawTensorEntry {
                key: key.clone(),
                shape: info.shape,
                storage_type: StorageElementType::scalar(ty),
                byte_ranges: vec![ByteRange::new(abs_offset, length)],
                alignment: ty.size_bytes().unwrap_or(1) as u64,
                endianness: Endianness::Little,
                metadata: MetadataMap::new(),
            });
        }

        if entries.len() as u64 > limits.max_tensor_count {
            return Err(DynInferError::InvalidCheckpoint(CheckpointValidationError {
                message: format!(
                    "tensor count {} exceeds limit {}",
                    entries.len(),
                    limits.max_tensor_count
                ),
                key: None,
                detail: None,
            }));
        }

        entries.sort_by(|a, b| a.key.cmp(&b.key));
        debug!(count = entries.len(), "indexed safetensors tensors");

        let source_files = source
            .path()
            .map(|p| {
                vec![SourceFile {
                    path: p.to_path_buf(),
                    size_bytes: source.len(),
                    content_digest: None,
                }]
            })
            .unwrap_or_default();

        Ok(RawCheckpointIndex {
            container: ContainerIdentity {
                format_id: self.format_id(),
                version: Some(1),
                magic: Some("safetensors".into()),
            },
            source_files,
            metadata,
            entries,
            data_offset: data_base,
        })
    }

    fn runtime_provider_plan(&self, index: &RawCheckpointIndex) -> Result<RuntimeProviderPlan> {
        Ok(RuntimeProviderPlan {
            kind: "file-mapped-external-parameters".into(),
            scope: "weights".into(),
            file_paths: index
                .source_files
                .iter()
                .map(|f| f.path.display().to_string())
                .collect(),
            notes: vec!["SafeTensors dense tensors use direct byte ranges".into()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dyninfer_checkpoint::BytesSource;
    use serde_json::json;

    fn make_st(tensors: serde_json::Value) -> Vec<u8> {
        let header = tensors.to_string();
        let header_len = header.len() as u64;
        let mut out = Vec::new();
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        // pad data for one f32[2] tensor = 8 bytes
        out.extend_from_slice(&[0u8; 8]);
        out
    }

    #[test]
    fn indexes_dense_tensor() {
        let bytes = make_st(json!({
            "w.weight": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 8]
            }
        }));
        let source = Arc::new(BytesSource::new(bytes)) as Arc<dyn RandomAccessSource>;
        let c = SafeTensorsContainer;
        assert!(c.probe(source.as_ref()).unwrap().is_match());
        let index = c.index(source, &InspectionLimits::default()).unwrap();
        assert_eq!(index.entries.len(), 1);
        assert_eq!(index.entries[0].key, "w.weight");
        assert_eq!(index.entries[0].shape, vec![2]);
    }
}
