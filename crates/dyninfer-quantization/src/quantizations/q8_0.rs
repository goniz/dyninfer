//! Executable GGUF Q8_0 definition.

use super::gguf_block::GgufBlockDefinition;
use super::gguf_codegen::{self, BlockLayout};
use crate::{
    EmbeddingCallSpec, EncodingDefinitionDescriptor, ExternalEncodingTag, LinearCallSpec,
    ParameterLoweringSpec, QuantizationDefinition,
};
use dyninfer_checkpoint::LogicalParameter;
use dyninfer_core::{ParameterBinding, PhysicalEncoding};
use dyninfer_error::Result;
use dyninfer_kernel_registry::{EncodingKey, KernelCandidateDescriptor};

#[derive(Debug, Default)]
pub struct GgufQ80Definition;

const SCHEMA: GgufBlockDefinition =
    GgufBlockDefinition::new("q8_0", 8, 32, 34, &["scale_f16", "quants_i8"]);

impl QuantizationDefinition for GgufQ80Definition {
    fn descriptor(&self) -> EncodingDefinitionDescriptor {
        EncodingDefinitionDescriptor {
            key: EncodingKey::new("gguf.q8_0", 1),
            external_tags: vec![ExternalEncodingTag {
                family: "gguf.ggml_type".into(),
                value: "8".into(),
            }],
        }
    }

    fn matches(&self, encoding: &PhysicalEncoding) -> bool {
        SCHEMA.matches(encoding)
    }

    fn validate(&self, parameter: &LogicalParameter) -> Result<()> {
        SCHEMA.validate(parameter)
    }

    fn kernel_candidates(&self) -> Vec<KernelCandidateDescriptor> {
        gguf_codegen::kernel_candidates(BlockLayout::Q80, "gguf.q8_0")
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
        gguf_codegen::helper_key(BlockLayout::Q80, spec)
    }

    fn emit_helper(
        &self,
        module: &mut dyninfer_mlir::ModuleBuilder,
        spec: &ParameterLoweringSpec<'_>,
    ) -> Result<bool> {
        gguf_codegen::emit_helper(module, BlockLayout::Q80, spec)
    }

    fn emit_linear_call(
        &self,
        function: &mut dyninfer_mlir::FuncBuilder,
        spec: &LinearCallSpec<'_>,
    ) -> Result<bool> {
        gguf_codegen::emit_linear_call(function, BlockLayout::Q80, spec)
    }

    fn emit_embedding_call(
        &self,
        function: &mut dyninfer_mlir::FuncBuilder,
        spec: &EmbeddingCallSpec<'_>,
    ) -> Result<bool> {
        gguf_codegen::emit_embedding_call(function, BlockLayout::Q80, spec)
    }
}
