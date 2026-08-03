//! Mixed per-tensor GGUF convention decoding.

use crate::types::GgufType;
use dyninfer_checkpoint::{
    CheckpointConventionDecoder, DecodeContext, LogicalParameter, MatchScore, ParameterCatalog,
    RawCheckpointIndex, infer_role,
};
use dyninfer_core::{
    BlockLayoutField, CanonicalParameterName, CodecId, ConventionId, Endianness, LogicalTensorType,
    MetadataMap, PhysicalEncoding, ScalarType, Shape, StorageComponent, StorageElementType,
    TensorOrder,
};
use dyninfer_error::{DynInferError, Result, UnsupportedEncodingError};

fn entry_gguf_type(entry: &dyninfer_checkpoint::RawTensorEntry) -> Result<GgufType> {
    let code = entry
        .metadata
        .get("gguf.type_code")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| {
            DynInferError::UnsupportedEncoding(UnsupportedEncodingError {
                message: "missing gguf.type_code on tensor entry".into(),
                key: Some(entry.key.clone()),
                codec: None,
                codec_version: None,
                expected: Some("known GGUF ggml_type".into()),
                actual: None,
            })
        })?;
    GgufType::from_u32(code as u32)
}

#[derive(Debug, Default)]
pub struct GgufMixedConvention;

impl CheckpointConventionDecoder for GgufMixedConvention {
    fn convention_id(&self) -> ConventionId {
        ConventionId::new("gguf.mixed")
    }

    fn match_score(
        &self,
        index: &RawCheckpointIndex,
        _context: &DecodeContext,
    ) -> Result<MatchScore> {
        if index.container.format_id.as_str() != "gguf" || index.entries.is_empty() {
            return Ok(MatchScore::NONE);
        }
        Ok(MatchScore { score: 100 })
    }

    fn decode(
        &self,
        index: &RawCheckpointIndex,
        _context: &DecodeContext,
    ) -> Result<ParameterCatalog> {
        let mut parameters = Vec::with_capacity(index.entries.len());
        for entry in &index.entries {
            let gguf_type = entry_gguf_type(entry)?;
            let (encoding, logical_type) = physical_encoding(gguf_type)?;
            let shape = Shape::new(entry.shape.clone());
            parameters.push(LogicalParameter {
                canonical_name: CanonicalParameterName::new(entry.key.clone()),
                role: infer_role(&entry.key),
                logical_type: LogicalTensorType {
                    shape: shape.clone(),
                    element_type: logical_type,
                },
                encoding,
                components: vec![StorageComponent {
                    name: "data".into(),
                    key: entry.key.clone(),
                    source_file_index: entry.source_file_index,
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
            metadata: normalized_metadata(&index.metadata),
        })
    }
}

fn physical_encoding(gguf_type: GgufType) -> Result<(PhysicalEncoding, ScalarType)> {
    let scalar = match gguf_type {
        GgufType::F32 => Some(ScalarType::F32),
        GgufType::F16 => Some(ScalarType::F16),
        GgufType::BF16 => Some(ScalarType::Bf16),
        GgufType::F64 => Some(ScalarType::F64),
        GgufType::I8 => Some(ScalarType::I8),
        GgufType::I16 => Some(ScalarType::I16),
        GgufType::I32 => Some(ScalarType::I32),
        GgufType::I64 => Some(ScalarType::I64),
        _ => None,
    };
    if let Some(scalar) = scalar {
        return Ok((PhysicalEncoding::plain(scalar), scalar));
    }
    let block = gguf_type.block_size().ok_or_else(|| {
        DynInferError::UnsupportedEncoding(UnsupportedEncodingError {
            message: format!("GGUF type {} has no block layout", gguf_type.name()),
            key: None,
            codec: Some(format!("gguf.{}", gguf_type.name())),
            codec_version: Some(1),
            expected: Some("registered scalar or block layout".into()),
            actual: Some(gguf_type.name().into()),
        })
    })?;
    let bytes_per_block = gguf_type.type_size().ok_or_else(|| {
        DynInferError::UnsupportedEncoding(UnsupportedEncodingError {
            message: format!("GGUF type {} has no byte size", gguf_type.name()),
            key: None,
            codec: Some(format!("gguf.{}", gguf_type.name())),
            codec_version: Some(1),
            expected: Some("registered block byte size".into()),
            actual: None,
        })
    })?;
    Ok((
        PhysicalEncoding::BlockQuantized {
            logical_type: ScalarType::F16,
            block_shape: vec![block as u32],
            bytes_per_block: bytes_per_block as u32,
            codec: CodecId::new(format!("gguf.{}", gguf_type.name())),
            codec_version: 1,
            components: layout_components(gguf_type)
                .iter()
                .map(ToString::to_string)
                .collect(),
            layout: block_layout(gguf_type)?,
            order: TensorOrder::RowMajor,
            endianness: Endianness::Little,
        },
        ScalarType::F16,
    ))
}

fn block_layout(gguf_type: GgufType) -> Result<Vec<BlockLayoutField>> {
    let names = layout_components(gguf_type);
    let lengths = layout_byte_lengths(gguf_type);
    if names.len() != lengths.len() {
        return Err(DynInferError::internal(format!(
            "GGUF {} layout name/length count mismatch",
            gguf_type.name()
        )));
    }
    let mut offset = 0u32;
    let layout: Vec<_> = names
        .iter()
        .zip(lengths)
        .map(|(name, &byte_length)| {
            let field = BlockLayoutField {
                name: (*name).into(),
                byte_offset: offset,
                byte_length,
                storage_type: layout_storage_type(name),
            };
            offset += byte_length;
            field
        })
        .collect();
    if Some(u64::from(offset)) != gguf_type.type_size() {
        return Err(DynInferError::internal(format!(
            "GGUF {} layout totals {offset} bytes, expected {:?}",
            gguf_type.name(),
            gguf_type.type_size()
        )));
    }
    Ok(layout)
}

fn layout_storage_type(name: &str) -> StorageElementType {
    if name.ends_with("f16") {
        StorageElementType::scalar(ScalarType::F16)
    } else if name.ends_with("f32") {
        StorageElementType::scalar(ScalarType::F32)
    } else if name.ends_with("i8") {
        StorageElementType::scalar(ScalarType::I8)
    } else if name.ends_with("i16") {
        StorageElementType::scalar(ScalarType::I16)
    } else if name.ends_with("u8") {
        StorageElementType::scalar(ScalarType::U8)
    } else if name.ends_with("u16") {
        StorageElementType::scalar(ScalarType::U16)
    } else {
        StorageElementType::Opaque {
            codec: format!("packed.{name}"),
        }
    }
}

fn layout_byte_lengths(gguf_type: GgufType) -> &'static [u32] {
    match gguf_type {
        GgufType::Q4_0 => &[2, 16],
        GgufType::Q4_1 => &[2, 2, 16],
        GgufType::Q5_0 => &[2, 4, 16],
        GgufType::Q5_1 => &[2, 2, 4, 16],
        GgufType::Q8_0 => &[2, 32],
        GgufType::Q8_1 => &[2, 2, 32],
        GgufType::Q2_K => &[16, 64, 4],
        GgufType::Q3_K => &[32, 64, 12, 2],
        GgufType::Q4_K => &[4, 12, 128],
        GgufType::Q5_K => &[4, 12, 32, 128],
        GgufType::Q6_K => &[128, 64, 16, 2],
        GgufType::Q8_K => &[4, 256, 32],
        GgufType::IQ2_XXS => &[2, 64],
        GgufType::IQ2_XS => &[2, 64, 8],
        GgufType::IQ3_XXS => &[2, 96],
        GgufType::IQ1_S => &[2, 32, 16],
        GgufType::IQ4_NL => &[2, 16],
        GgufType::IQ3_S => &[2, 64, 8, 32, 4],
        GgufType::IQ2_S => &[2, 64, 8, 8],
        GgufType::IQ4_XS => &[2, 2, 4, 128],
        GgufType::IQ1_M => &[32, 16, 8],
        GgufType::TQ1_0 => &[2, 4, 48],
        GgufType::TQ2_0 => &[2, 64],
        GgufType::MXFP4 => &[1, 16],
        _ => &[],
    }
}

fn layout_components(gguf_type: GgufType) -> &'static [&'static str] {
    match gguf_type {
        GgufType::Q4_0 => &["scale_f16", "quants_u4"],
        GgufType::Q4_1 => &["scale_f16", "minimum_f16", "quants_u4"],
        GgufType::Q5_0 => &["scale_f16", "high_bits", "quants_u4"],
        GgufType::Q5_1 => &["scale_f16", "minimum_f16", "high_bits", "quants_u4"],
        GgufType::Q8_0 => &["scale_f16", "quants_i8"],
        GgufType::Q8_1 => &["scale_f16", "sum_f16", "quants_i8"],
        GgufType::Q2_K => &["scales_and_mins_u4", "quants_u2", "scale_min_f16"],
        GgufType::Q3_K => &["high_bits", "quants_u2", "scales_u6", "scale_f16"],
        GgufType::Q4_K => &["scale_min_f16", "scales_and_mins_u6", "quants_u4"],
        GgufType::Q5_K => &[
            "scale_min_f16",
            "scales_and_mins_u6",
            "high_bits",
            "quants_u4",
        ],
        GgufType::Q6_K => &["quants_low_u4", "quants_high_u2", "scales_i8", "scale_f16"],
        GgufType::Q8_K => &["scale_f32", "quants_i8", "block_sums_i16"],
        GgufType::IQ2_XXS => &["scale_f16", "grid_indices_and_signs_u16"],
        GgufType::IQ2_XS => &["scale_f16", "grid_indices_and_signs_u16", "scales_u8"],
        GgufType::IQ3_XXS => &["scale_f16", "grid_indices_and_signs_u8"],
        GgufType::IQ1_S => &["scale_f16", "grid_indices_u8", "high_bits_and_delta_u16"],
        GgufType::IQ4_NL => &["scale_f16", "nonlinear_quants_u4"],
        GgufType::IQ3_S => &[
            "scale_f16",
            "grid_indices_low_u8",
            "grid_indices_high_u8",
            "signs_u8",
            "scales_u8",
        ],
        GgufType::IQ2_S => &[
            "scale_f16",
            "grid_indices_low_u8",
            "grid_indices_high_u8",
            "scales_u8",
        ],
        GgufType::IQ4_XS => &[
            "scale_f16",
            "scales_high_u16",
            "scales_low_u8",
            "nonlinear_quants_u4",
        ],
        GgufType::IQ1_M => &["grid_indices_u8", "high_bits_and_delta_u8", "scales_u8"],
        GgufType::TQ1_0 => &["scale_f16", "block_scales_u8", "ternary_quants_base3"],
        GgufType::TQ2_0 => &["scale_f16", "ternary_quants_u2"],
        GgufType::MXFP4 => &["e8m0_scale_u8", "mxfp4_quants_u4"],
        _ => &[],
    }
}

fn normalized_metadata(metadata: &MetadataMap) -> MetadataMap {
    let mut normalized = metadata.clone();
    let architecture = metadata
        .get("general.architecture")
        .and_then(|value| value.as_str());
    if let Some(architecture) = architecture {
        normalized.insert("model_type".into(), architecture.into());
        for (source_suffix, destination) in [
            ("block_count", "num_layers"),
            ("embedding_length", "hidden_size"),
            ("feed_forward_length", "intermediate_size"),
            ("attention.head_count", "num_heads"),
            ("attention.head_count_kv", "num_kv_heads"),
            ("attention.key_length", "head_dim"),
            ("context_length", "context_length"),
            ("attention.layer_norm_rms_epsilon", "rms_norm_eps"),
            ("rope.freq_base", "rope_theta"),
        ] {
            if let Some(value) = metadata.get(&format!("{architecture}.{source_suffix}")) {
                normalized.insert(destination.into(), value.clone());
            }
        }
        if !normalized.contains_key("head_dim") {
            if let Some(value) = metadata.get(&format!("{architecture}.rope.dimension_count")) {
                normalized.insert("head_dim".into(), value.clone());
            }
        }
    }
    if let Some(tokens) = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|value| value.as_array())
    {
        normalized.insert("vocab_size".into(), (tokens.len() as u64).into());
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q6_k_descriptor_is_schema_only_and_complete() {
        let (encoding, logical) = physical_encoding(GgufType::Q6_K).unwrap();
        assert_eq!(logical, ScalarType::F16);
        let PhysicalEncoding::BlockQuantized {
            block_shape,
            codec,
            components,
            ..
        } = encoding
        else {
            panic!("expected block encoding");
        };
        assert_eq!(block_shape, vec![256]);
        assert_eq!(codec.as_str(), "gguf.q6_k");
        assert_eq!(components.len(), 4);
    }

    #[test]
    fn every_sized_block_type_has_an_explicit_layout() {
        for code in 0..40 {
            let Ok(gguf_type) = GgufType::from_u32(code) else {
                continue;
            };
            if gguf_type.block_size().is_some() {
                assert!(
                    !layout_components(gguf_type).is_empty(),
                    "missing layout for {}",
                    gguf_type.name()
                );
                let layout = block_layout(gguf_type).unwrap();
                let mut offset = 0;
                for field in &layout {
                    assert_eq!(field.byte_offset, offset);
                    offset += field.byte_length;
                }
                assert_eq!(u64::from(offset), gguf_type.type_size().unwrap());
            }
        }
    }
}
