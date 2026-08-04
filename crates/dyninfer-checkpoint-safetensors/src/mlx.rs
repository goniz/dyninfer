//! MLX SafeTensors convention with compound affine-quantized parameters.

use crate::config_json::load_hf_config_metadata;
use dyninfer_checkpoint::{
    CheckpointConventionDecoder, DecodeContext, LogicalParameter, MatchScore, ParameterCatalog,
    RawCheckpointIndex, RawTensorEntry, infer_role,
};
use dyninfer_core::{
    CanonicalParameterName, ConventionId, LogicalTensorType, MetadataMap, PhysicalEncoding,
    ScalarType, Shape, StorageComponent, StorageElementType, TensorOrder, ZeroPointMode,
};
use dyninfer_error::{DynInferError, Result, UnsupportedEncodingError};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Default)]
pub struct MlxSafeTensorsConvention;

impl CheckpointConventionDecoder for MlxSafeTensorsConvention {
    fn convention_id(&self) -> ConventionId {
        ConventionId::new("safetensors.mlx.mixed")
    }

    fn match_score(
        &self,
        index: &RawCheckpointIndex,
        _context: &DecodeContext,
    ) -> Result<MatchScore> {
        if !index
            .container
            .format_id
            .as_str()
            .starts_with("safetensors")
        {
            return Ok(MatchScore::NONE);
        }
        let declares_mlx = index
            .metadata
            .get("format")
            .and_then(|value| value.as_str())
            .is_some_and(|format| format.eq_ignore_ascii_case("mlx"));
        let has_compound_weights = index
            .entries
            .iter()
            .any(|entry| entry.key.ends_with(".scales") || entry.key.ends_with(".biases"));
        Ok(if declares_mlx && has_compound_weights {
            MatchScore { score: 150 }
        } else {
            MatchScore::NONE
        })
    }

    fn decode(
        &self,
        index: &RawCheckpointIndex,
        _context: &DecodeContext,
    ) -> Result<ParameterCatalog> {
        let mut metadata = index.metadata.clone();
        if let Some(source) = index.source_files.first() {
            for (key, value) in load_hf_config_metadata(&source.path) {
                metadata.entry(key).or_insert(value);
            }
        }
        let bits = required_u32(&metadata, "mlx.quantization.bits")?;
        let group_size = required_u32(&metadata, "mlx.quantization.group_size")?;
        if !matches!(bits, 2 | 3 | 4 | 6 | 8) || group_size == 0 {
            return Err(unsupported(
                None,
                format!(
                    "unsupported MLX affine configuration bits={bits}, group_size={group_size}"
                ),
                Some(format!("bits in 2|3|4|6|8 and group_size > 0")),
                None,
            ));
        }

        let by_key: BTreeMap<_, _> = index
            .entries
            .iter()
            .map(|entry| (entry.key.as_str(), entry))
            .collect();
        let mut consumed_components = BTreeSet::new();
        let mut parameters = Vec::new();

        for entry in &index.entries {
            if entry.key.ends_with(".scales") || entry.key.ends_with(".biases") {
                continue;
            }
            if entry.key.contains("rotary_emb") || entry.key.ends_with(".inv_freq") {
                continue;
            }

            let scale_key = entry
                .key
                .strip_suffix(".weight")
                .map(|base| format!("{base}.scales"));
            let bias_key = entry
                .key
                .strip_suffix(".weight")
                .map(|base| format!("{base}.biases"));
            let scale = scale_key
                .as_deref()
                .and_then(|key| by_key.get(key).copied());
            let bias = bias_key.as_deref().and_then(|key| by_key.get(key).copied());

            match (scale, bias) {
                (Some(scale), Some(bias)) => {
                    let parameter = decode_affine_weight(entry, scale, bias, bits, group_size)?;
                    consumed_components.insert(scale.key.clone());
                    consumed_components.insert(bias.key.clone());
                    parameters.push(parameter);
                }
                (None, None) => parameters.push(decode_plain(entry)?),
                _ => {
                    return Err(unsupported(
                        Some(&entry.key),
                        "MLX quantized weight must have both .scales and .biases components",
                        Some("weight + scales + biases".into()),
                        Some(format!(
                            "scales={}, biases={}",
                            scale.is_some(),
                            bias.is_some()
                        )),
                    ));
                }
            }
        }

        let orphan: Vec<_> = index
            .entries
            .iter()
            .filter(|entry| entry.key.ends_with(".scales") || entry.key.ends_with(".biases"))
            .filter(|entry| !consumed_components.contains(&entry.key))
            .map(|entry| entry.key.clone())
            .collect();
        if !orphan.is_empty() {
            return Err(unsupported(
                orphan.first().map(String::as_str),
                format!("orphan MLX quantization components: {orphan:?}"),
                Some("components paired with a .weight tensor".into()),
                None,
            ));
        }

        Ok(ParameterCatalog {
            convention_id: self.convention_id(),
            parameters,
            metadata,
        })
    }
}

fn decode_affine_weight(
    weight: &RawTensorEntry,
    scales: &RawTensorEntry,
    biases: &RawTensorEntry,
    bits: u32,
    group_size: u32,
) -> Result<LogicalParameter> {
    if weight.shape.is_empty() {
        return Err(unsupported(
            Some(&weight.key),
            "MLX packed weight must have rank >= 1",
            None,
            None,
        ));
    }
    if !matches!(
        weight.storage_type,
        StorageElementType::Scalar {
            ty: ScalarType::U32
        }
    ) {
        return Err(unsupported(
            Some(&weight.key),
            "MLX packed weight storage must be u32",
            Some("u32".into()),
            Some(weight.storage_type.to_string()),
        ));
    }
    let scale_type = scalar_type(scales)?;
    let bias_type = scalar_type(biases)?;
    if scale_type != bias_type
        || !matches!(
            scale_type,
            ScalarType::F16 | ScalarType::Bf16 | ScalarType::F32
        )
    {
        return Err(unsupported(
            Some(&weight.key),
            "MLX scales and biases must use the same floating-point type",
            Some("matching f16|bf16|f32".into()),
            Some(format!("scales={scale_type}, biases={bias_type}")),
        ));
    }

    let mut logical_shape = weight.shape.clone();
    let packed_last = *logical_shape.last().unwrap();
    let logical_numerator = packed_last
        .checked_mul(32)
        .ok_or_else(|| DynInferError::internal("MLX logical shape overflow"))?;
    if !logical_numerator.is_multiple_of(u64::from(bits)) {
        return Err(unsupported(
            Some(&weight.key),
            "MLX packed axis cannot be expanded exactly",
            Some(format!("packed_last * 32 divisible by {bits}")),
            Some(packed_last.to_string()),
        ));
    }
    *logical_shape.last_mut().unwrap() = logical_numerator / u64::from(bits);
    let logical_last = *logical_shape.last().unwrap();
    if !logical_last.is_multiple_of(u64::from(group_size)) {
        return Err(unsupported(
            Some(&weight.key),
            "MLX logical quantization axis is not group divisible",
            Some(format!("multiple of {group_size}")),
            Some(logical_last.to_string()),
        ));
    }
    let mut expected_group_shape = logical_shape.clone();
    *expected_group_shape.last_mut().unwrap() = logical_last / u64::from(group_size);
    if scales.shape != expected_group_shape || biases.shape != expected_group_shape {
        return Err(unsupported(
            Some(&weight.key),
            "MLX scales/biases shape does not match logical groups",
            Some(format!("{expected_group_shape:?}")),
            Some(format!(
                "scales={:?}, biases={:?}",
                scales.shape, biases.shape
            )),
        ));
    }

    Ok(LogicalParameter {
        canonical_name: CanonicalParameterName::new(weight.key.clone()),
        role: infer_role(&weight.key),
        logical_type: LogicalTensorType {
            shape: Shape::new(logical_shape),
            element_type: scale_type,
        },
        encoding: PhysicalEncoding::GroupQuantized {
            logical_type: scale_type,
            storage_bits: bits as u8,
            storage_container: ScalarType::U32,
            signed: false,
            axis: -1,
            group_size,
            scale_type,
            bias_type: Some(bias_type),
            zero_point: ZeroPointMode::None,
            packing: format!("mlx.affine.u{bits}"),
            order: TensorOrder::RowMajor,
            components: vec!["packed".into(), "scales".into(), "biases".into()],
        },
        components: vec![
            component("packed", weight),
            component("scales", scales),
            component("biases", biases),
        ],
        aliases: vec![weight.key.clone()],
    })
}

fn decode_plain(entry: &RawTensorEntry) -> Result<LogicalParameter> {
    let ty = scalar_type(entry)?;
    if !matches!(ty, ScalarType::Bf16 | ScalarType::F16 | ScalarType::F32) {
        return Err(unsupported(
            Some(&entry.key),
            "unpaired MLX tensor is not a supported dense float",
            Some("bf16|f16|f32".into()),
            Some(ty.to_string()),
        ));
    }
    Ok(LogicalParameter {
        canonical_name: CanonicalParameterName::new(entry.key.clone()),
        role: infer_role(&entry.key),
        logical_type: LogicalTensorType {
            shape: Shape::new(entry.shape.clone()),
            element_type: ty,
        },
        encoding: PhysicalEncoding::plain(ty),
        components: vec![component("data", entry)],
        aliases: vec![entry.key.clone()],
    })
}

fn scalar_type(entry: &RawTensorEntry) -> Result<ScalarType> {
    match entry.storage_type {
        StorageElementType::Scalar { ty } => Ok(ty),
        _ => Err(unsupported(
            Some(&entry.key),
            "MLX SafeTensors component must use scalar storage",
            None,
            Some(entry.storage_type.to_string()),
        )),
    }
}

fn component(name: &str, entry: &RawTensorEntry) -> StorageComponent {
    StorageComponent {
        name: name.into(),
        key: entry.key.clone(),
        source_file_index: entry.source_file_index,
        shape: Shape::new(entry.shape.clone()),
        storage_type: entry.storage_type.clone(),
        byte_ranges: entry.byte_ranges.clone(),
        alignment: entry.alignment,
        endianness: entry.endianness.clone(),
    }
}

fn required_u32(metadata: &MetadataMap, key: &str) -> Result<u32> {
    metadata
        .get(key)
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            unsupported(
                None,
                format!("missing required MLX metadata `{key}`"),
                Some("explicit config.json quantization metadata".into()),
                None,
            )
        })
}

fn unsupported(
    key: Option<&str>,
    message: impl Into<String>,
    expected: Option<String>,
    actual: Option<String>,
) -> DynInferError {
    DynInferError::UnsupportedEncoding(UnsupportedEncodingError {
        message: message.into(),
        key: key.map(ToString::to_string),
        codec: Some("mlx.affine".into()),
        codec_version: Some(1),
        expected,
        actual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dyninfer_checkpoint::{ContainerIdentity, RawCheckpointIndex};
    use dyninfer_core::{ByteRange, ContainerFormatId, Endianness};

    fn entry(key: &str, shape: Vec<u64>, ty: ScalarType, offset: u64) -> RawTensorEntry {
        let bytes = shape.iter().product::<u64>() * u64::from(ty.size_bytes().unwrap());
        RawTensorEntry {
            key: key.into(),
            source_file_index: 0,
            shape,
            storage_type: StorageElementType::scalar(ty),
            byte_ranges: vec![ByteRange::new(offset, bytes)],
            alignment: 1,
            endianness: Endianness::Little,
            metadata: MetadataMap::new(),
        }
    }

    fn mixed_index() -> RawCheckpointIndex {
        RawCheckpointIndex {
            container: ContainerIdentity {
                format_id: ContainerFormatId::new("safetensors"),
                version: Some(1),
                magic: None,
            },
            source_files: vec![],
            metadata: MetadataMap::from([
                ("format".into(), serde_json::json!("mlx")),
                ("mlx.quantization.bits".into(), serde_json::json!(4)),
                ("mlx.quantization.group_size".into(), serde_json::json!(64)),
            ]),
            entries: vec![
                entry("layer.weight", vec![2, 8], ScalarType::U32, 0),
                entry("layer.scales", vec![2, 1], ScalarType::Bf16, 64),
                entry("layer.biases", vec![2, 1], ScalarType::Bf16, 68),
                entry("layer_norm.weight", vec![64], ScalarType::Bf16, 72),
            ],
            data_offset: 0,
        }
    }

    #[test]
    fn groups_real_mlx_weight_scale_bias_topology() {
        let decoder = MlxSafeTensorsConvention;
        assert_eq!(
            decoder
                .match_score(&mixed_index(), &DecodeContext::default())
                .unwrap()
                .score,
            150
        );
        let catalog = decoder
            .decode(&mixed_index(), &DecodeContext::default())
            .unwrap();
        assert_eq!(catalog.parameters.len(), 2);
        let weight = catalog
            .parameters
            .iter()
            .find(|parameter| parameter.canonical_name.as_str() == "layer.weight")
            .unwrap();
        assert_eq!(weight.logical_type.shape.dims(), &[2, 64]);
        assert_eq!(weight.components.len(), 3);
        assert!(matches!(
            &weight.encoding,
            PhysicalEncoding::GroupQuantized {
                storage_bits: 4,
                group_size: 64,
                packing,
                ..
            } if packing == "mlx.affine.u4"
        ));
    }

    #[test]
    fn rejects_partial_compound_weight() {
        let mut index = mixed_index();
        index.entries.retain(|entry| entry.key != "layer.biases");
        let error = MlxSafeTensorsConvention
            .decode(&index, &DecodeContext::default())
            .unwrap_err();
        assert_eq!(error.code(), "E_UNSUPPORTED_ENCODING");
    }
}
