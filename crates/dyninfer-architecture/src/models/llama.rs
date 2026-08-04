//! Llama decoder architecture (`llama.decoder`).
//!
//! Covers Meta Llama / OpenLLaMA / Mistral-layout checkpoints and the synthetic
//! Milestone-1 fixture. Q/K norms are optional (absent for classic Llama).

use crate::naming::canonicalize_hf_family;
use crate::slots::field;
use crate::{
    ArchitectureDefinition, ConfigSchema, DecoderBlockSpec, ModelBuilder, ModelModule,
    ResolvedModelConfig,
};
use dyninfer_checkpoint::ParameterCatalog;
use dyninfer_core::ArchitectureId;
use dyninfer_error::Result;
use std::sync::LazyLock;

/// Declares which hyperparams this arch reads from `--set` / config.json / GGUF
/// metadata. Defaults are used only when a key is absent (synthetic fixture /
/// incomplete metadata) — real HF checkpoints override every required field.
static CONFIG_SCHEMA: LazyLock<ConfigSchema> = LazyLock::new(|| ConfigSchema {
    fields: vec![
        field("num_layers", "u32", true, Some(serde_json::json!(2))),
        field("num_heads", "u32", true, Some(serde_json::json!(4))),
        field("num_kv_heads", "u32", true, Some(serde_json::json!(4))),
        field("head_dim", "u32", true, Some(serde_json::json!(64))),
        field("hidden_size", "u32", true, Some(serde_json::json!(256))),
        field(
            "intermediate_size",
            "u32",
            true,
            Some(serde_json::json!(512)),
        ),
        field("vocab_size", "u32", true, Some(serde_json::json!(32000))),
        field("context_length", "u32", true, Some(serde_json::json!(2048))),
        field("rms_norm_eps", "f64", false, Some(serde_json::json!(1e-5))),
        field("rope_theta", "f64", false, None),
        // Compile-time specialization shapes (override dense-emitter heuristics).
        field("prefill_window", "u32", false, None),
        field("max_kv", "u32", false, None),
    ],
});

#[derive(Debug, Default)]
pub struct LlamaArchitecture;

impl ArchitectureDefinition for LlamaArchitecture {
    fn id(&self) -> ArchitectureId {
        ArchitectureId::new("llama.decoder")
    }

    fn revision(&self) -> &str {
        "0.2.0"
    }

    fn config_schema(&self) -> &ConfigSchema {
        &CONFIG_SCHEMA
    }

    /// HF `model_type` / `architectures[]` stems routed here.
    ///
    /// Mistral shares Llama's Transformers weight layout
    /// (`self_attn.*_proj`, SwiGLU `mlp.*`, same layernorm names), so it maps to
    /// this arch — same choice as mlx-lm / llama.cpp convert scripts.
    fn model_types(&self) -> &[&str] {
        &["llama", "mistral", "LlamaForCausalLM", "MistralForCausalLM"]
    }

    fn build(&self, config: &ResolvedModelConfig, m: &mut ModelBuilder) -> Result<ModelModule> {
        let num_layers = config.num_layers()?;
        let hidden = config.get_u32("hidden_size")?;
        let vocab = config.get_u32("vocab_size")?;
        let block = DecoderBlockSpec {
            hidden_size: hidden,
            intermediate_size: config.get_u32("intermediate_size")?,
            num_heads: config.get_u32("num_heads")?,
            num_kv_heads: config.get_u32("num_kv_heads")?,
            head_dim: config.get_u32("head_dim")?,
            rms_norm_epsilon: config.get_f64("rms_norm_eps")?,
            rope_theta: config
                .values
                .get("rope_theta")
                .and_then(|value| value.as_f64()),
        };

        let tokens = m.input_tokens("tokens")?;
        let mut x = m.embedding("token_embedding", tokens, "token_embd.weight", hidden)?;
        for layer in 0..num_layers {
            x = m.decoder_block(x, layer, /*has_qk_norm=*/ false, &block)?;
        }
        x = m.final_rms_norm(
            "output_norm",
            x,
            "output_norm.weight",
            block.rms_norm_epsilon,
        )?;
        let logits = m.output_projection("output_projection", x, "output.weight", vocab)?;
        m.export_prefill_and_decode(logits)?;
        m.finish()
    }

    fn canonicalize_param(&self, key: &str) -> Option<String> {
        // HF → GGUF-style names; verified against llama.cpp TENSOR_NAMES.
        canonicalize_hf_family(key)
    }

    fn sanitize_catalog(&self, catalog: &mut ParameterCatalog) {
        crate::naming::tie_output_to_embed(catalog);
    }
}
