use super::gguf_codegen::{self, BlockLayout};
use crate::{
    EmbeddingCallSpec, EncodingDefinitionDescriptor, ExternalEncodingTag, LinearCallSpec,
    ParameterLoweringSpec, QuantizationDefinition,
};
use dyninfer_checkpoint::LogicalParameter;
use dyninfer_core::{
    Endianness, ParameterBinding, PhysicalEncoding, ScalarType, StorageElementType, TensorOrder,
};
use dyninfer_error::{DynInferError, Result, UnsupportedEncodingError};
use dyninfer_kernel_registry::{EncodingKey, KernelCandidateDescriptor};

/// GGUF Q4_0 schema and direct interleaved-block CPU lowering.
#[derive(Debug, Default)]
pub struct GgufQ40Definition;

impl QuantizationDefinition for GgufQ40Definition {
    fn descriptor(&self) -> EncodingDefinitionDescriptor {
        EncodingDefinitionDescriptor {
            key: EncodingKey::new("gguf.q4_0", 1),
            external_tags: vec![ExternalEncodingTag {
                family: "gguf.ggml_type".into(),
                value: "2".into(),
            }],
        }
    }

    fn matches(&self, encoding: &PhysicalEncoding) -> bool {
        matches!(
            encoding,
            PhysicalEncoding::BlockQuantized {
                codec,
                codec_version: 1,
                block_shape,
                ..
            } if codec.as_str() == "gguf.q4_0" && block_shape == &[32]
        )
    }

    fn validate(&self, parameter: &LogicalParameter) -> Result<()> {
        if !self.matches(&parameter.encoding) {
            return Err(unsupported(parameter, "invalid Q4_0 encoding descriptor"));
        }
        let PhysicalEncoding::BlockQuantized {
            logical_type,
            bytes_per_block,
            components,
            layout,
            order,
            endianness,
            ..
        } = &parameter.encoding
        else {
            unreachable!("matches checked the variant")
        };
        if logical_type != &ScalarType::F16
            || bytes_per_block != &18
            || components != &["scale_f16".to_string(), "quants_u4".to_string()]
            || layout.len() != 2
            || layout[0].name != "scale_f16"
            || layout[0].byte_offset != 0
            || layout[0].byte_length != 2
            || layout[1].name != "quants_u4"
            || layout[1].byte_offset != 2
            || layout[1].byte_length != 16
            || order != &TensorOrder::RowMajor
            || endianness != &Endianness::Little
        {
            return Err(unsupported(
                parameter,
                "Q4_0 requires logical f16 with scale_f16/quants_u4 layout",
            ));
        }
        let numel = parameter
            .logical_type
            .shape
            .numel()
            .ok_or_else(|| DynInferError::internal("Q4_0 parameter numel overflow"))?;
        if !numel.is_multiple_of(32) {
            return Err(unsupported(
                parameter,
                format!("Q4_0 numel {numel} is not divisible by 32"),
            ));
        }
        let expected = (numel / 32)
            .checked_mul(18)
            .ok_or_else(|| DynInferError::internal("Q4_0 byte size overflow"))?;
        let actual = parameter
            .components
            .iter()
            .flat_map(|component| &component.byte_ranges)
            .try_fold(0u64, |total, range| total.checked_add(range.length))
            .ok_or_else(|| DynInferError::internal("Q4_0 component byte size overflow"))?;
        let storage_codec_matches = matches!(
            parameter.components.as_slice(),
            [component]
                if component.name == "data"
                    && matches!(
                        &component.storage_type,
                        StorageElementType::Opaque { codec } if codec == "gguf.q4_0"
                    )
        );
        if !storage_codec_matches || actual != expected {
            return Err(unsupported(
                parameter,
                format!(
                    "Q4_0 requires one interleaved component of {expected} bytes, got {} components and {actual} bytes",
                    parameter.components.len()
                ),
            ));
        }
        Ok(())
    }

    fn kernel_candidates(&self) -> Vec<KernelCandidateDescriptor> {
        gguf_codegen::kernel_candidates(BlockLayout::Q40, "gguf.q4_0")
    }

    fn emit_external_globals(
        &self,
        builder: &mut dyninfer_mlir::ModuleBuilder,
        binding: &ParameterBinding,
        symbol: &str,
    ) -> Result<bool> {
        gguf_codegen::emit_external_globals(builder, binding, symbol)
    }

    fn emit_parameter_load(
        &self,
        function: &mut dyninfer_mlir::FuncBuilder,
        binding: &ParameterBinding,
        ssa: &str,
        symbol: &str,
    ) -> Result<bool> {
        gguf_codegen::emit_parameter_load(function, binding, ssa, symbol)
    }

    fn helper_key(&self, spec: &ParameterLoweringSpec<'_>) -> Option<String> {
        gguf_codegen::helper_key(BlockLayout::Q40, spec)
    }

    fn emit_helper(
        &self,
        module: &mut dyninfer_mlir::ModuleBuilder,
        spec: &ParameterLoweringSpec<'_>,
    ) -> Result<bool> {
        gguf_codegen::emit_helper(module, BlockLayout::Q40, spec)
    }

    fn emit_linear_call(
        &self,
        function: &mut dyninfer_mlir::FuncBuilder,
        spec: &LinearCallSpec<'_>,
    ) -> Result<bool> {
        gguf_codegen::emit_linear_call(function, BlockLayout::Q40, spec)
    }

    fn emit_embedding_call(
        &self,
        function: &mut dyninfer_mlir::FuncBuilder,
        spec: &EmbeddingCallSpec<'_>,
    ) -> Result<bool> {
        gguf_codegen::emit_embedding_call(function, BlockLayout::Q40, spec)
    }
}

fn unsupported(parameter: &LogicalParameter, message: impl Into<String>) -> DynInferError {
    DynInferError::UnsupportedEncoding(UnsupportedEncodingError {
        message: message.into(),
        key: Some(parameter.canonical_name.to_string()),
        codec: Some("gguf.q4_0".into()),
        codec_version: Some(1),
        expected: Some("32 values per 18-byte interleaved block".into()),
        actual: Some(format!("{:?}", parameter.encoding)),
    })
}
