//! Schema-only definitions for recognized GGUF block encodings.
//!
//! These definitions make mixed GGUF layouts inspectable and strictly
//! validate their per-tensor storage contracts. They deliberately contribute
//! no production kernels: parsing a ggml type is not executable support.

use crate::{EncodingDefinitionDescriptor, ExternalEncodingTag, QuantizationDefinition};
use dyninfer_checkpoint::LogicalParameter;
use dyninfer_core::{Endianness, PhysicalEncoding, ScalarType, StorageElementType, TensorOrder};
use dyninfer_error::{DynInferError, Result, UnsupportedEncodingError};
use dyninfer_kernel_registry::{EncodingKey, KernelCandidateDescriptor};

#[derive(Debug, Clone, Copy)]
pub struct GgufBlockDefinition {
    name: &'static str,
    type_code: u32,
    block_size: u64,
    bytes_per_block: u64,
    layout_components: &'static [&'static str],
}

impl GgufBlockDefinition {
    pub const fn new(
        name: &'static str,
        type_code: u32,
        block_size: u64,
        bytes_per_block: u64,
        layout_components: &'static [&'static str],
    ) -> Self {
        Self {
            name,
            type_code,
            block_size,
            bytes_per_block,
            layout_components,
        }
    }

    fn codec(&self) -> String {
        format!("gguf.{}", self.name)
    }
}

impl QuantizationDefinition for GgufBlockDefinition {
    fn descriptor(&self) -> EncodingDefinitionDescriptor {
        EncodingDefinitionDescriptor {
            key: EncodingKey::new(self.codec(), 1),
            external_tags: vec![ExternalEncodingTag {
                family: "gguf.ggml_type".into(),
                value: self.type_code.to_string(),
            }],
        }
    }

    fn matches(&self, encoding: &PhysicalEncoding) -> bool {
        matches!(
            encoding,
            PhysicalEncoding::BlockQuantized {
                codec,
                codec_version: 1,
                ..
            } if codec.as_str() == self.codec()
        )
    }

    fn validate(&self, parameter: &LogicalParameter) -> Result<()> {
        let PhysicalEncoding::BlockQuantized {
            logical_type,
            block_shape,
            bytes_per_block,
            components,
            layout,
            order,
            endianness,
            ..
        } = &parameter.encoding
        else {
            return Err(self.unsupported(parameter, "expected block-quantized storage"));
        };
        let expected_layout: Vec<_> = self
            .layout_components
            .iter()
            .map(ToString::to_string)
            .collect();
        if !self.matches(&parameter.encoding)
            || logical_type != &ScalarType::F16
            || block_shape != &[self.block_size as u32]
            || bytes_per_block != &(self.bytes_per_block as u32)
            || components != &expected_layout
            || order != &TensorOrder::RowMajor
            || endianness != &Endianness::Little
            || layout.len() != expected_layout.len()
            || !layout
                .iter()
                .zip(&expected_layout)
                .all(|(field, expected_name)| field.name == expected_name.as_str())
            || layout.iter().try_fold(0u32, |offset, field| {
                (field.byte_offset == offset)
                    .then(|| offset.checked_add(field.byte_length))
                    .flatten()
            }) != Some(self.bytes_per_block as u32)
        {
            return Err(self.unsupported(
                parameter,
                format!(
                    "invalid {} descriptor; expected logical=f16 block=[{}] layout={expected_layout:?}",
                    self.name, self.block_size
                ),
            ));
        }
        let numel = parameter
            .logical_type
            .shape
            .numel()
            .ok_or_else(|| DynInferError::internal(format!("{} numel overflow", self.name)))?;
        if !numel.is_multiple_of(self.block_size) {
            return Err(self.unsupported(
                parameter,
                format!(
                    "{} numel {numel} is not divisible by {}",
                    self.name, self.block_size
                ),
            ));
        }
        let expected_bytes = (numel / self.block_size)
            .checked_mul(self.bytes_per_block)
            .ok_or_else(|| DynInferError::internal(format!("{} byte size overflow", self.name)))?;
        let actual_bytes = parameter
            .components
            .iter()
            .flat_map(|component| &component.byte_ranges)
            .try_fold(0u64, |total, range| total.checked_add(range.length))
            .ok_or_else(|| {
                DynInferError::internal(format!("{} component byte size overflow", self.name))
            })?;
        let storage_codec_matches = matches!(
            parameter.components.as_slice(),
            [component]
                if component.name == "data"
                    && matches!(
                        &component.storage_type,
                        StorageElementType::Opaque { codec } if codec == &self.codec()
                    )
        );
        if !storage_codec_matches || actual_bytes != expected_bytes {
            return Err(self.unsupported(
                parameter,
                format!(
                    "{} requires one interleaved data component of {expected_bytes} bytes, got {} components and {actual_bytes} bytes",
                    self.name,
                    parameter.components.len()
                ),
            ));
        }
        Ok(())
    }

    fn kernel_candidates(&self) -> Vec<KernelCandidateDescriptor> {
        vec![]
    }
}

impl GgufBlockDefinition {
    fn unsupported(
        &self,
        parameter: &LogicalParameter,
        message: impl Into<String>,
    ) -> DynInferError {
        DynInferError::UnsupportedEncoding(UnsupportedEncodingError {
            message: message.into(),
            key: Some(parameter.canonical_name.to_string()),
            codec: Some(self.codec()),
            codec_version: Some(1),
            expected: Some(format!(
                "{} values per {}-byte interleaved block",
                self.block_size, self.bytes_per_block
            )),
            actual: Some(format!("{:?}", parameter.encoding)),
        })
    }
}

/// All current ggml block formats except executable Q4_0, Q4_1, Q8_0, and Q6_K.
pub const SCHEMA_ONLY_DEFINITIONS: &[GgufBlockDefinition] = &[
    GgufBlockDefinition::new("q5_0", 6, 32, 22, &["scale_f16", "high_bits", "quants_u4"]),
    GgufBlockDefinition::new(
        "q5_1",
        7,
        32,
        24,
        &["scale_f16", "minimum_f16", "high_bits", "quants_u4"],
    ),
    GgufBlockDefinition::new("q8_1", 9, 32, 36, &["scale_f16", "sum_f16", "quants_i8"]),
    GgufBlockDefinition::new(
        "q2_k",
        10,
        256,
        84,
        &["scales_and_mins_u4", "quants_u2", "scale_min_f16"],
    ),
    GgufBlockDefinition::new(
        "q3_k",
        11,
        256,
        110,
        &["high_bits", "quants_u2", "scales_u6", "scale_f16"],
    ),
    GgufBlockDefinition::new(
        "q4_k",
        12,
        256,
        144,
        &["scale_min_f16", "scales_and_mins_u6", "quants_u4"],
    ),
    GgufBlockDefinition::new(
        "q5_k",
        13,
        256,
        176,
        &[
            "scale_min_f16",
            "scales_and_mins_u6",
            "high_bits",
            "quants_u4",
        ],
    ),
    GgufBlockDefinition::new(
        "q8_k",
        15,
        256,
        292,
        &["scale_f32", "quants_i8", "block_sums_i16"],
    ),
    GgufBlockDefinition::new(
        "iq2_xxs",
        16,
        256,
        66,
        &["scale_f16", "grid_indices_and_signs_u16"],
    ),
    GgufBlockDefinition::new(
        "iq2_xs",
        17,
        256,
        74,
        &["scale_f16", "grid_indices_and_signs_u16", "scales_u8"],
    ),
    GgufBlockDefinition::new(
        "iq3_xxs",
        18,
        256,
        98,
        &["scale_f16", "grid_indices_and_signs_u8"],
    ),
    GgufBlockDefinition::new(
        "iq1_s",
        19,
        256,
        50,
        &["scale_f16", "grid_indices_u8", "high_bits_and_delta_u16"],
    ),
    GgufBlockDefinition::new("iq4_nl", 20, 32, 18, &["scale_f16", "nonlinear_quants_u4"]),
    GgufBlockDefinition::new(
        "iq3_s",
        21,
        256,
        110,
        &[
            "scale_f16",
            "grid_indices_low_u8",
            "grid_indices_high_u8",
            "signs_u8",
            "scales_u8",
        ],
    ),
    GgufBlockDefinition::new(
        "iq2_s",
        22,
        256,
        82,
        &[
            "scale_f16",
            "grid_indices_low_u8",
            "grid_indices_high_u8",
            "scales_u8",
        ],
    ),
    GgufBlockDefinition::new(
        "iq4_xs",
        23,
        256,
        136,
        &[
            "scale_f16",
            "scales_high_u16",
            "scales_low_u8",
            "nonlinear_quants_u4",
        ],
    ),
    GgufBlockDefinition::new(
        "iq1_m",
        29,
        256,
        56,
        &["grid_indices_u8", "high_bits_and_delta_u8", "scales_u8"],
    ),
    GgufBlockDefinition::new(
        "tq1_0",
        34,
        256,
        54,
        &["scale_f16", "block_scales_u8", "ternary_quants_base3"],
    ),
    GgufBlockDefinition::new("tq2_0", 35, 256, 66, &["scale_f16", "ternary_quants_u2"]),
    GgufBlockDefinition::new("mxfp4", 39, 32, 17, &["e8m0_scale_u8", "mxfp4_quants_u4"]),
];
