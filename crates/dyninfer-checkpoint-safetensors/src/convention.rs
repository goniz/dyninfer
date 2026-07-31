//! Dense SafeTensors convention decoder.

use dyninfer_checkpoint::{
    infer_role, CheckpointConventionDecoder, DecodeContext, LogicalParameter, MatchScore,
    ParameterCatalog, RawCheckpointIndex,
};
use dyninfer_core::{
    CanonicalParameterName, ConventionId, Endianness, LogicalTensorType, PhysicalEncoding, ScalarType,
    Shape, StorageComponent, StorageElementType,
};
use dyninfer_error::{DynInferError, UnsupportedEncodingError, Result};

#[derive(Debug, Default)]
pub struct DenseSafetensorsConvention;

impl CheckpointConventionDecoder for DenseSafetensorsConvention {
    fn convention_id(&self) -> ConventionId {
        ConventionId::new("safetensors.dense")
    }

    fn match_score(
        &self,
        index: &RawCheckpointIndex,
        _context: &DecodeContext,
    ) -> Result<MatchScore> {
        if index.container.format_id.as_str() != "safetensors" {
            return Ok(MatchScore::NONE);
        }
        let dense_count = index
            .entries
            .iter()
            .filter(|e| matches!(e.storage_type, StorageElementType::Scalar { .. }))
            .count();
        if dense_count == 0 {
            return Ok(MatchScore::NONE);
        }
        Ok(MatchScore {
            score: 50 + dense_count.min(40) as u32,
        })
    }

    fn decode(
        &self,
        index: &RawCheckpointIndex,
        _context: &DecodeContext,
    ) -> Result<ParameterCatalog> {
        let mut parameters = Vec::with_capacity(index.entries.len());
        for entry in &index.entries {
            let StorageElementType::Scalar { ty } = &entry.storage_type else {
                return Err(DynInferError::UnsupportedEncoding(UnsupportedEncodingError {
                    message: "non-scalar SafeTensors entry not supported by dense convention"
                        .into(),
                    key: Some(entry.key.clone()),
                    codec: None,
                    codec_version: None,
                    expected: Some("scalar dense".into()),
                    actual: Some(entry.storage_type.to_string()),
                }));
            };
            // Version 1 prefers BF16; F16/F32 are accepted for inspection/binding.
            if !matches!(ty, ScalarType::Bf16 | ScalarType::F16 | ScalarType::F32) {
                return Err(DynInferError::UnsupportedEncoding(UnsupportedEncodingError {
                    message: format!("dense SafeTensors dtype {ty} not in version-1 path"),
                    key: Some(entry.key.clone()),
                    codec: None,
                    codec_version: None,
                    expected: Some("bf16|f16|f32".into()),
                    actual: Some(ty.to_string()),
                }));
            }

            let shape = Shape::new(entry.shape.clone());
            parameters.push(LogicalParameter {
                canonical_name: CanonicalParameterName::new(entry.key.clone()),
                role: infer_role(&entry.key),
                logical_type: LogicalTensorType {
                    shape: shape.clone(),
                    element_type: *ty,
                },
                encoding: PhysicalEncoding::plain(*ty),
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
