//! GGUF container indexing (metadata + tensor directory only).

use crate::types::GgufType;
use byteorder::{ByteOrder, LittleEndian};
use dyninfer_checkpoint::{
    CheckpointContainerReader, ContainerIdentity, InspectionLimits, ProbeScore, RandomAccessSource,
    RawCheckpointIndex, RawTensorEntry, RuntimeProviderPlan,
};
use dyninfer_core::{ByteRange, ContainerFormatId, Endianness, MetadataMap, SourceFile};
use dyninfer_error::{CheckpointValidationError, DynInferError, Result};
use std::collections::BTreeMap;
use std::sync::Arc;
use tracing::debug;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const ALIGNMENT_DEFAULT: u64 = 32;

#[derive(Debug, Default)]
pub struct GgufContainer;

struct Cursor<'a> {
    source: &'a dyn RandomAccessSource,
    pos: u64,
    limits: &'a InspectionLimits,
}

impl<'a> Cursor<'a> {
    fn read_exact(&mut self, len: usize) -> Result<Vec<u8>> {
        let buf = self.source.read_range(self.pos, len as u64)?;
        self.pos = self
            .pos
            .checked_add(len as u64)
            .ok_or_else(|| DynInferError::io("GGUF cursor overflow"))?;
        Ok(buf)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_exact(4)?;
        Ok(LittleEndian::read_u32(&b))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_exact(8)?;
        Ok(LittleEndian::read_u64(&b))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let b = self.read_exact(4)?;
        Ok(LittleEndian::read_i32(&b))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let b = self.read_exact(4)?;
        Ok(LittleEndian::read_f32(&b))
    }

    fn read_bool(&mut self) -> Result<bool> {
        let b = self.read_exact(1)?;
        Ok(b[0] != 0)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        if len > self.limits.max_key_len as u64 {
            return Err(DynInferError::InvalidCheckpoint(
                CheckpointValidationError {
                    message: format!("GGUF string length {len} exceeds limit"),
                    key: None,
                    detail: None,
                },
            ));
        }
        let bytes = self.read_exact(len as usize)?;
        String::from_utf8(bytes).map_err(|e| {
            DynInferError::InvalidCheckpoint(CheckpointValidationError {
                message: format!("GGUF string is not UTF-8: {e}"),
                key: None,
                detail: None,
            })
        })
    }

    fn read_value(&mut self, value_type: u32) -> Result<serde_json::Value> {
        // GGUF value types: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
        Ok(match value_type {
            0 => serde_json::Value::Number(self.read_u8()?.into()), // UINT8
            1 => serde_json::Value::Number((self.read_i8()? as i64).into()), // INT8
            2 => serde_json::Value::Number(self.read_u16()?.into()), // UINT16
            3 => serde_json::Value::Number((self.read_i16()? as i64).into()), // INT16
            4 => serde_json::Value::Number(self.read_u32()?.into()), // UINT32
            5 => serde_json::Value::Number(self.read_i32()?.into()), // INT32
            6 => serde_json::Number::from_f64(self.read_f32()? as f64)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null), // FLOAT32
            7 => serde_json::Value::Bool(self.read_bool()?),        // BOOL
            8 => serde_json::Value::String(self.read_string()?),    // STRING
            9 => {
                // ARRAY
                let elem_type = self.read_u32()?;
                let count = self.read_u64()?;
                if count > self.limits.max_metadata_entries {
                    return Err(DynInferError::InvalidCheckpoint(
                        CheckpointValidationError {
                            message: format!("GGUF array length {count} exceeds limit"),
                            key: None,
                            detail: None,
                        },
                    ));
                }
                let mut arr = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    arr.push(self.read_value(elem_type)?);
                }
                serde_json::Value::Array(arr)
            }
            10 => serde_json::Value::Number(self.read_u64()?.into()), // UINT64
            11 => {
                let v = self.read_exact(8)?;
                serde_json::Value::Number(LittleEndian::read_i64(&v).into())
            } // INT64
            12 => {
                let v = self.read_exact(8)?;
                serde_json::Number::from_f64(LittleEndian::read_f64(&v))
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            } // FLOAT64
            other => {
                return Err(DynInferError::InvalidCheckpoint(
                    CheckpointValidationError {
                        message: format!("unsupported GGUF metadata value type {other}"),
                        key: None,
                        detail: None,
                    },
                ));
            }
        })
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }
    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_exact(1)?[0] as i8)
    }
    fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_exact(2)?;
        Ok(LittleEndian::read_u16(&b))
    }
    fn read_i16(&mut self) -> Result<i16> {
        let b = self.read_exact(2)?;
        Ok(LittleEndian::read_i16(&b))
    }
}

fn align_up(value: u64, align: u64) -> u64 {
    if align == 0 {
        return value;
    }
    value.div_ceil(align) * align
}

impl CheckpointContainerReader for GgufContainer {
    fn format_id(&self) -> ContainerFormatId {
        ContainerFormatId::new("gguf")
    }

    fn probe(&self, source: &dyn RandomAccessSource) -> Result<ProbeScore> {
        if source.len() < 4 {
            return Ok(ProbeScore::NONE);
        }
        let mut magic = [0u8; 4];
        source.read_exact_at(0, &mut magic)?;
        if &magic == GGUF_MAGIC {
            return Ok(ProbeScore::STRONG);
        }
        Ok(ProbeScore::NONE)
    }

    fn index(
        &self,
        source: Arc<dyn RandomAccessSource>,
        limits: &InspectionLimits,
    ) -> Result<RawCheckpointIndex> {
        let mut cur = Cursor {
            source: source.as_ref(),
            pos: 0,
            limits,
        };
        let magic = cur.read_exact(4)?;
        if magic.as_slice() != GGUF_MAGIC {
            return Err(DynInferError::InvalidCheckpoint(
                CheckpointValidationError {
                    message: "not a GGUF file".into(),
                    key: None,
                    detail: None,
                },
            ));
        }
        let version = cur.read_u32()?;
        if version < 2 || version > 3 {
            return Err(DynInferError::InvalidCheckpoint(
                CheckpointValidationError {
                    message: format!("unsupported GGUF version {version}"),
                    key: None,
                    detail: Some("supported: 2, 3".into()),
                },
            ));
        }
        let tensor_count = cur.read_u64()?;
        let kv_count = cur.read_u64()?;
        if tensor_count > limits.max_tensor_count || kv_count > limits.max_metadata_entries {
            return Err(DynInferError::InvalidCheckpoint(
                CheckpointValidationError {
                    message: format!(
                        "GGUF counts exceed limits: tensors={tensor_count} kv={kv_count}"
                    ),
                    key: None,
                    detail: None,
                },
            ));
        }

        let mut metadata: MetadataMap = BTreeMap::new();
        for _ in 0..kv_count {
            let key = cur.read_string()?;
            limits.validate_key(&key)?;
            let value_type = cur.read_u32()?;
            let value = cur.read_value(value_type)?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(ALIGNMENT_DEFAULT);

        #[derive(Clone)]
        struct TensorDir {
            name: String,
            shape: Vec<u64>,
            gguf_type: GgufType,
            offset: u64,
        }

        let mut dirs = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            limits.validate_key(&name)?;
            let n_dims = cur.read_u32()? as usize;
            if n_dims > limits.max_shape_rank {
                return Err(DynInferError::InvalidCheckpoint(
                    CheckpointValidationError {
                        message: format!("GGUF tensor rank {n_dims} exceeds limit"),
                        key: Some(name),
                        detail: None,
                    },
                ));
            }
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(cur.read_u64()?);
            }
            // GGUF stores dims in reverse order relative to row-major logical shape.
            shape.reverse();
            limits.validate_shape(&shape)?;
            let type_code = cur.read_u32()?;
            let gguf_type = GgufType::from_u32(type_code)?;
            let offset = cur.read_u64()?;
            dirs.push(TensorDir {
                name,
                shape,
                gguf_type,
                offset,
            });
        }

        let data_offset = align_up(cur.pos, alignment);
        let mut entries = Vec::with_capacity(dirs.len());
        for dir in dirs {
            let nbytes = dir.gguf_type.nbytes_for_shape(&dir.shape)?;
            let abs = data_offset.checked_add(dir.offset).ok_or_else(|| {
                DynInferError::InvalidCheckpoint(CheckpointValidationError {
                    message: "tensor absolute offset overflow".into(),
                    key: Some(dir.name.clone()),
                    detail: None,
                })
            })?;
            if abs.checked_add(nbytes).unwrap_or(u64::MAX) > source.len() {
                return Err(DynInferError::InvalidCheckpoint(
                    CheckpointValidationError {
                        message: "tensor data extends past end of file".into(),
                        key: Some(dir.name.clone()),
                        detail: None,
                    },
                ));
            }
            let mut meta = MetadataMap::new();
            meta.insert(
                "gguf.type".into(),
                serde_json::Value::String(dir.gguf_type.name().into()),
            );
            meta.insert(
                "gguf.type_code".into(),
                serde_json::Value::from(dir.gguf_type as u32),
            );
            entries.push(RawTensorEntry {
                key: dir.name,
                source_file_index: 0,
                shape: dir.shape,
                storage_type: dir.gguf_type.storage_element_type(),
                byte_ranges: vec![ByteRange::new(abs, nbytes)],
                alignment,
                endianness: Endianness::Little,
                metadata: meta,
            });
        }

        entries.sort_by(|a, b| a.key.cmp(&b.key));
        debug!(count = entries.len(), version, "indexed gguf tensors");

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
                version: Some(version),
                magic: Some("GGUF".into()),
            },
            source_files,
            metadata,
            entries,
            data_offset,
        })
    }

    fn runtime_provider_plan(&self, index: &RawCheckpointIndex) -> Result<RuntimeProviderPlan> {
        Ok(RuntimeProviderPlan {
            kind: "file-mapped-external-parameters".into(),
            scope: "weights".into(),
            file_paths: index.source_files.iter().map(|f| f.path.clone()).collect(),
            parameters: vec![],
            notes: vec!["GGUF tensors addressed by absolute file offsets".into()],
        })
    }
}
