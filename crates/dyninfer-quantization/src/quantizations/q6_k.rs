//! Executable GGUF Q6_K definition.

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
pub struct GgufQ6KDefinition;

const SCHEMA: GgufBlockDefinition = GgufBlockDefinition::new(
    "q6_k",
    14,
    256,
    210,
    &["quants_low_u4", "quants_high_u2", "scales_i8", "scale_f16"],
);

impl QuantizationDefinition for GgufQ6KDefinition {
    fn descriptor(&self) -> EncodingDefinitionDescriptor {
        EncodingDefinitionDescriptor {
            key: EncodingKey::new("gguf.q6_k", 1),
            external_tags: vec![ExternalEncodingTag {
                family: "gguf.ggml_type".into(),
                value: "14".into(),
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
        gguf_codegen::kernel_candidates(BlockLayout::Q6K, "gguf.q6_k")
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
        gguf_codegen::helper_key(BlockLayout::Q6K, spec)
    }

    fn emit_helper(
        &self,
        module: &mut dyninfer_mlir::ModuleBuilder,
        spec: &ParameterLoweringSpec<'_>,
    ) -> Result<bool> {
        gguf_codegen::emit_helper(module, BlockLayout::Q6K, spec)
    }

    fn emit_linear_call(
        &self,
        function: &mut dyninfer_mlir::FuncBuilder,
        spec: &LinearCallSpec<'_>,
    ) -> Result<bool> {
        gguf_codegen::emit_linear_call(function, BlockLayout::Q6K, spec)
    }

    fn emit_embedding_call(
        &self,
        function: &mut dyninfer_mlir::FuncBuilder,
        spec: &EmbeddingCallSpec<'_>,
    ) -> Result<bool> {
        gguf_codegen::emit_embedding_call(function, BlockLayout::Q6K, spec)
    }
}
