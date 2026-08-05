//! LFM2 hybrid decoder architecture (`lfm2.hybrid`).
//!
//! Liquid AI's LFM2 / LFM2.5 checkpoints interleave two operator kinds behind
//! an otherwise standard pre-norm SwiGLU decoder:
//!
//! - `full_attention` layers: GQA with per-head Q/K RMSNorm and RoPE, matching
//!   Qwen3's attention shape but named `out_proj` / `q_layernorm` upstream.
//! - `conv` layers: `Lfm2ShortConv`, a gated causal depthwise convolution.
//!   `in_proj` produces `[B, C, x]`, the operator computes
//!   `C * depthwise_conv1d(B * x)` over a `conv_L_cache` window, and `out_proj`
//!   projects back to the residual stream.
//!
//! The per-layer schedule is data, not a stride: `config.json` carries an
//! explicit `layer_types` list, so this file refuses to guess when it is
//! missing or the wrong length.
//!
//! Both operator kinds share `operator_norm` (canonicalized to `attn_norm`) and
//! the layer's `feed_forward` block, so only the operator sublayer differs.
//! LFM2's final norm is named `embedding_norm` upstream because embeddings are
//! tied; semantically it is the output norm and maps to `output_norm.weight`.

use crate::naming::tie_output_to_embed;
use crate::slots::field;
use crate::{
    ArchitectureDefinition, ConfigSchema, DecoderBlockSpec, ModelBuilder, ModelModule,
    ResolvedModelConfig, ShortConvBlockSpec,
};
use dyninfer_checkpoint::ParameterCatalog;
use dyninfer_core::ArchitectureId;
use dyninfer_error::{ConfigError, DynInferError, Result};
use std::sync::LazyLock;

/// Upstream `layer_types` token for an attention layer.
const ATTENTION_LAYER: &str = "full_attention";
/// Upstream `layer_types` token for a short-convolution layer.
const CONV_LAYER: &str = "conv";

/// Fixture-sized defaults. Real checkpoints override every required field from
/// `config.json`; the default schedule keeps both operator kinds represented so
/// conformance tests exercise the hybrid path.
static CONFIG_SCHEMA: LazyLock<ConfigSchema> = LazyLock::new(|| ConfigSchema {
    fields: vec![
        field("num_layers", "u32", true, Some(serde_json::json!(3))),
        field("num_heads", "u32", true, Some(serde_json::json!(4))),
        field("num_kv_heads", "u32", true, Some(serde_json::json!(2))),
        field("head_dim", "u32", true, Some(serde_json::json!(64))),
        field("hidden_size", "u32", true, Some(serde_json::json!(256))),
        field("intermediate_size", "u32", true, Some(serde_json::json!(512))),
        field("vocab_size", "u32", true, Some(serde_json::json!(32000))),
        field("context_length", "u32", true, Some(serde_json::json!(4096))),
        field(
            "layer_types",
            "array<string>",
            true,
            Some(serde_json::json!([CONV_LAYER, CONV_LAYER, ATTENTION_LAYER])),
        ),
        field("rms_norm_eps", "f64", false, Some(serde_json::json!(1e-5))),
        field(
            "rope_theta",
            "f64",
            false,
            Some(serde_json::json!(1_000_000.0)),
        ),
        // `conv_L_cache` upstream: causal window of the depthwise convolution.
        field("conv_kernel", "u32", false, Some(serde_json::json!(3))),
        // `conv_dim` upstream; defaults to `hidden_size` when absent.
        field("conv_dim", "u32", false, None),
        field("prefill_window", "u32", false, None),
        field("max_kv", "u32", false, None),
    ],
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerKind {
    Attention,
    ShortConv,
}

/// Read the explicit per-layer operator schedule from `layer_types`.
fn layer_schedule(config: &ResolvedModelConfig, num_layers: u32) -> Result<Vec<LayerKind>> {
    let invalid = |message: String| DynInferError::Config(ConfigError { message });
    let entries = config
        .values
        .get("layer_types")
        .and_then(|value| value.as_array())
        .ok_or_else(|| invalid("config field `layer_types` missing or not an array".into()))?;
    if entries.len() != num_layers as usize {
        return Err(invalid(format!(
            "`layer_types` has {} entries but `num_layers` is {num_layers}",
            entries.len()
        )));
    }
    entries
        .iter()
        .enumerate()
        .map(|(index, entry)| match entry.as_str() {
            Some(ATTENTION_LAYER) => Ok(LayerKind::Attention),
            Some(CONV_LAYER) => Ok(LayerKind::ShortConv),
            other => Err(invalid(format!(
                "unsupported `layer_types[{index}]` value {other:?}; expected `{ATTENTION_LAYER}` or `{CONV_LAYER}`"
            ))),
        })
        .collect()
}

#[derive(Debug, Default)]
pub struct Lfm2Architecture;

impl ArchitectureDefinition for Lfm2Architecture {
    fn id(&self) -> ArchitectureId {
        ArchitectureId::new("lfm2.hybrid")
    }

    fn revision(&self) -> &str {
        "0.1.0"
    }

    fn config_schema(&self) -> &ConfigSchema {
        &CONFIG_SCHEMA
    }

    fn model_types(&self) -> &[&str] {
        &["lfm2", "Lfm2ForCausalLM"]
    }

    fn build(&self, config: &ResolvedModelConfig, m: &mut ModelBuilder) -> Result<ModelModule> {
        let num_layers = config.num_layers()?;
        let hidden = config.get_u32("hidden_size")?;
        let vocab = config.get_u32("vocab_size")?;
        let rms_norm_epsilon = config.get_f64("rms_norm_eps")?;
        let schedule = layer_schedule(config, num_layers)?;

        let block = DecoderBlockSpec {
            hidden_size: hidden,
            intermediate_size: config.get_u32("intermediate_size")?,
            num_heads: config.get_u32("num_heads")?,
            num_kv_heads: config.get_u32("num_kv_heads")?,
            head_dim: config.get_u32("head_dim")?,
            rms_norm_epsilon,
            rope_theta: config
                .values
                .get("rope_theta")
                .and_then(|value| value.as_f64()),
        };
        let short_conv = ShortConvBlockSpec {
            hidden_size: hidden,
            channels: config.get_u32("conv_dim").unwrap_or(hidden),
            kernel_size: config.get_u32("conv_kernel")?,
            rms_norm_epsilon,
        };

        let tokens = m.input_tokens("tokens")?;
        let mut x = m.embedding("token_embedding", tokens, "token_embd.weight", hidden)?;
        for (layer, kind) in schedule.into_iter().enumerate() {
            let layer = layer as u32;
            x = match kind {
                LayerKind::Attention => {
                    m.attention_sublayer(x, layer, /*has_qk_norm=*/ true, &block)?
                }
                LayerKind::ShortConv => m.short_conv_sublayer(x, layer, &short_conv)?,
            };
            x = m.feed_forward_sublayer(x, layer, &block)?;
        }
        x = m.final_rms_norm("output_norm", x, "output_norm.weight", rms_norm_epsilon)?;
        let logits = m.output_projection("output_projection", x, "output.weight", vocab)?;
        m.export_prefill_and_decode(logits)?;
        m.finish()
    }

    /// LFM2's weight tree diverges from the Llama/Qwen3 table
    /// (`operator_norm`, `feed_forward.w*`, `out_proj`, `q_layernorm`, and the
    /// `conv` submodule), so it carries its own map instead of reusing
    /// [`crate::naming::canonicalize_hf_family`].
    fn canonicalize_param(&self, key: &str) -> Option<String> {
        canonicalize_lfm2(key)
    }

    fn sanitize_catalog(&self, catalog: &mut ParameterCatalog) {
        tie_output_to_embed(catalog);
    }
}

fn canonicalize_lfm2(key: &str) -> Option<String> {
    if key.contains("rotary_emb") || key.ends_with(".inv_freq") {
        return None;
    }
    match key {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        // Final norm, named for the tied embedding it re-normalizes upstream.
        "model.embedding_norm.weight" => return Some("output_norm.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        _ => {}
    }

    let Some(rest) = key.strip_prefix("model.layers.") else {
        // GGUF-style / already canonical keys pass through unchanged.
        return Some(key.to_string());
    };
    let Some((index, suffix)) = rest.split_once('.') else {
        return Some(key.to_string());
    };
    let Ok(layer) = index.parse::<u32>() else {
        return Some(key.to_string());
    };
    let canonical_suffix = match suffix {
        "operator_norm.weight" => "attn_norm.weight",
        "ffn_norm.weight" => "ffn_norm.weight",
        "feed_forward.w1.weight" => "ffn_gate.weight",
        "feed_forward.w3.weight" => "ffn_up.weight",
        "feed_forward.w2.weight" => "ffn_down.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.out_proj.weight" => "attn_output.weight",
        "self_attn.q_layernorm.weight" => "attn_q_norm.weight",
        "self_attn.k_layernorm.weight" => "attn_k_norm.weight",
        "conv.in_proj.weight" => "shortconv.in_proj.weight",
        "conv.conv.weight" => "shortconv.conv.weight",
        "conv.out_proj.weight" => "shortconv.out_proj.weight",
        _ => return Some(key.to_string()),
    };
    Some(format!("blk.{layer}.{canonical_suffix}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_both_operator_kinds() {
        for (key, expected) in [
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.embedding_norm.weight", "output_norm.weight"),
            ("model.layers.0.operator_norm.weight", "blk.0.attn_norm.weight"),
            (
                "model.layers.0.conv.conv.weight",
                "blk.0.shortconv.conv.weight",
            ),
            (
                "model.layers.0.conv.in_proj.weight",
                "blk.0.shortconv.in_proj.weight",
            ),
            (
                "model.layers.0.feed_forward.w2.weight",
                "blk.0.ffn_down.weight",
            ),
            (
                "model.layers.2.self_attn.out_proj.weight",
                "blk.2.attn_output.weight",
            ),
            (
                "model.layers.2.self_attn.q_layernorm.weight",
                "blk.2.attn_q_norm.weight",
            ),
            // Already-canonical keys are untouched.
            ("blk.2.attn_q.weight", "blk.2.attn_q.weight"),
        ] {
            assert_eq!(canonicalize_lfm2(key).as_deref(), Some(expected), "{key}");
        }
        assert!(canonicalize_lfm2("model.layers.0.self_attn.rotary_emb.inv_freq").is_none());
    }

    #[test]
    fn layer_schedule_must_match_num_layers() {
        let config = ResolvedModelConfig {
            values: [(
                "layer_types".to_string(),
                serde_json::json!([CONV_LAYER, ATTENTION_LAYER]),
            )]
            .into_iter()
            .collect(),
        };
        assert_eq!(
            layer_schedule(&config, 2).unwrap(),
            vec![LayerKind::ShortConv, LayerKind::Attention]
        );
        assert!(layer_schedule(&config, 3).is_err());
    }
}
