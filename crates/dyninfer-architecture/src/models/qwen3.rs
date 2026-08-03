//! Qwen3 decoder architecture (`qwen3.decoder`).
//!
//! GQA, independent `head_dim`, per-head Q/K RMSNorm, often tied embeddings.

use crate::naming::{canonicalize_hf_family, tie_output_to_embed};
use crate::slots::field;
use crate::{
    ArchitectureDefinition, ConfigSchema, DecoderBlockSpec, ModelBuilder, ModelModule,
    ResolvedModelConfig,
};
use dyninfer_checkpoint::ParameterCatalog;
use dyninfer_core::ArchitectureId;
use dyninfer_error::Result;
use std::sync::LazyLock;

static CONFIG_SCHEMA: LazyLock<ConfigSchema> = LazyLock::new(|| ConfigSchema {
    fields: vec![
        field("num_layers", "u32", true, Some(serde_json::json!(28))),
        field("num_heads", "u32", true, Some(serde_json::json!(16))),
        field("num_kv_heads", "u32", true, Some(serde_json::json!(8))),
        field("head_dim", "u32", true, Some(serde_json::json!(128))),
        field("hidden_size", "u32", true, Some(serde_json::json!(1024))),
        field(
            "intermediate_size",
            "u32",
            true,
            Some(serde_json::json!(3072)),
        ),
        field("vocab_size", "u32", true, Some(serde_json::json!(151936))),
        field(
            "context_length",
            "u32",
            true,
            Some(serde_json::json!(40960)),
        ),
        field("rms_norm_eps", "f64", false, Some(serde_json::json!(1e-6))),
        field(
            "rope_theta",
            "f64",
            false,
            Some(serde_json::json!(1_000_000.0)),
        ),
        field("prefill_window", "u32", false, None),
        field("max_kv", "u32", false, None),
    ],
});

#[derive(Debug, Default)]
pub struct Qwen3Architecture;

impl ArchitectureDefinition for Qwen3Architecture {
    fn id(&self) -> ArchitectureId {
        ArchitectureId::new("qwen3.decoder")
    }

    fn revision(&self) -> &str {
        "0.1.0"
    }

    fn config_schema(&self) -> &ConfigSchema {
        &CONFIG_SCHEMA
    }

    fn model_types(&self) -> &[&str] {
        &["qwen3", "Qwen3ForCausalLM"]
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
            x = m.decoder_block(x, layer, /*has_qk_norm=*/ true, &block)?;
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
        canonicalize_hf_family(key)
    }

    fn sanitize_catalog(&self, catalog: &mut ParameterCatalog) {
        tie_output_to_embed(catalog);
    }
}
