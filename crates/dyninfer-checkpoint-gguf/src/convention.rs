//! GGUF convention decoders (Q4_0 and dense).

use crate::types::GgufType;
use dyninfer_checkpoint::{
    CheckpointConventionDecoder, DecodeContext, LogicalParameter, MatchScore, ParameterCatalog,
    RawCheckpointIndex, infer_role,
};
use dyninfer_core::{
    CanonicalParameterName, ConventionId, Endianness, LogicalTensorType, PhysicalEncoding,
    ScalarType, Shape, StorageComponent,
};
use dyninfer_error::{DynInferError, Result, UnsupportedEncodingError};

fn entry_gguf_type(entry: &dyninfer_checkpoint::RawTensorEntry) -> Option<GgufType> {
    entry
        .metadata
        .get("gguf.type_code")
        .and_then(|v| v.as_u64())
        .and_then(|c| GgufType::from_u32(c as u32).ok())
}

#[derive(Debug, Default)]
pub struct GgufQ40Convention;

impl CheckpointConventionDecoder for GgufQ40Convention {
    fn convention_id(&self) -> ConventionId {
        ConventionId::new("gguf.q4_0")
    }

    fn match_score(
        &self,
        index: &RawCheckpointIndex,
        _context: &DecodeContext,
    ) -> Result<MatchScore> {
        if index.container.format_id.as_str() != "gguf" {
            return Ok(MatchScore::NONE);
        }
        let q40 = index
            .entries
            .iter()
            .filter(|e| entry_gguf_type(e).is_some_and(|t| t.is_q4_0()))
            .count();
        if q40 == 0 {
            return Ok(MatchScore::NONE);
        }
        // Prefer Q4_0 when present (higher than dense).
        Ok(MatchScore {
            score: 70 + q40.min(20) as u32,
        })
    }

    fn decode(
        &self,
        index: &RawCheckpointIndex,
        _context: &DecodeContext,
    ) -> Result<ParameterCatalog> {
        let mut parameters = Vec::new();
        for entry in &index.entries {
            let Some(gguf_type) = entry_gguf_type(entry) else {
                return Err(DynInferError::UnsupportedEncoding(
                    UnsupportedEncodingError {
                        message: "missing gguf.type_code on tensor entry".into(),
                        key: Some(entry.key.clone()),
                        codec: None,
                        codec_version: None,
                        expected: None,
                        actual: None,
                    },
                ));
            };

            let (encoding, logical_ty) = if gguf_type.is_q4_0() {
                (PhysicalEncoding::gguf_q4_0(), ScalarType::F16)
            } else if gguf_type.is_dense_v1() {
                let ty = match gguf_type {
                    GgufType::F32 => ScalarType::F32,
                    GgufType::F16 => ScalarType::F16,
                    GgufType::BF16 => ScalarType::Bf16,
                    _ => unreachable!(),
                };
                (PhysicalEncoding::plain(ty), ty)
            } else {
                return Err(DynInferError::UnsupportedEncoding(
                    UnsupportedEncodingError {
                        message: format!(
                            "GGUF encoding {} not supported in version-1 Q4_0 path",
                            gguf_type.name()
                        ),
                        key: Some(entry.key.clone()),
                        codec: Some(format!("gguf.{}", gguf_type.name())),
                        codec_version: Some(1),
                        expected: Some("gguf.q4_0 or dense f16/bf16/f32".into()),
                        actual: Some(gguf_type.name().into()),
                    },
                ));
            };

            let shape = Shape::new(entry.shape.clone());
            parameters.push(LogicalParameter {
                canonical_name: CanonicalParameterName::new(entry.key.clone()),
                role: infer_role(&entry.key),
                logical_type: LogicalTensorType {
                    shape: shape.clone(),
                    element_type: logical_ty,
                },
                encoding,
                components: vec![StorageComponent {
                    name: "data".into(),
                    key: entry.key.clone(),
                    shape,
                    storage_type: entry.storage_type.clone(),
                    byte_ranges: entry.byte_ranges.clone(),
                    alignment: entry.alignment,
                    endianness: Endianness::Little,
                }],
                aliases: vec![entry.key.clone()],
            });
        }

        Ok(ParameterCatalog {
            convention_id: self.convention_id(),
            parameters,
            metadata: index.metadata.clone(),
        })
    }
}

#[derive(Debug, Default)]
pub struct GgufDenseConvention;

impl CheckpointConventionDecoder for GgufDenseConvention {
    fn convention_id(&self) -> ConventionId {
        ConventionId::new("gguf.dense")
    }

    fn match_score(
        &self,
        index: &RawCheckpointIndex,
        _context: &DecodeContext,
    ) -> Result<MatchScore> {
        if index.container.format_id.as_str() != "gguf" {
            return Ok(MatchScore::NONE);
        }
        let dense = index
            .entries
            .iter()
            .filter(|e| entry_gguf_type(e).is_some_and(|t| t.is_dense_v1()))
            .count();
        let quantized = index
            .entries
            .iter()
            .filter(|e| entry_gguf_type(e).is_some_and(|t| !t.is_dense_v1()))
            .count();
        if dense == 0 || quantized > 0 {
            return Ok(MatchScore::NONE);
        }
        Ok(MatchScore {
            score: 40 + dense.min(20) as u32,
        })
    }

    fn decode(
        &self,
        index: &RawCheckpointIndex,
        context: &DecodeContext,
    ) -> Result<ParameterCatalog> {
        // Reuse Q4_0 decoder path which also accepts dense tensors.
        GgufQ40Convention.decode(index, context).map(|mut c| {
            c.convention_id = self.convention_id();
            c
        })
    }
}
