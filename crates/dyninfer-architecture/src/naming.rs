//! HF Transformers → GGUF-style canonical names (llama.cpp `TENSOR_NAMES`).
//!
//! # Why this exists
//!
//! Checkpoints arrive with **container-native** keys:
//! - SafeTensors / HF: `model.layers.0.self_attn.q_proj.weight`
//! - GGUF: `blk.0.attn_q.weight`
//!
//! Architecture slots and the dense emitter speak **one** canonical vocabulary
//! (GGUF-style). This module is the HF→canonical table for decoder layouts that
//! share Transformers' Llama/Mistral/Qwen3 module paths. GGUF keys pass through
//! unchanged. Returns `None` only for non-weights (RoPE caches / `inv_freq`).
//!
//! Tied embeddings (`lm_head` absent) are handled later by
//! [`tie_output_to_embed`], not here.
//!
//! # Onboarding a new architecture
//!
//! Reuse this helper **only** if the HF weight tree matches this table.
//! Otherwise implement `ArchitectureDefinition::canonicalize_param` in the
//! arch file with its own map — do not grow this into a mega-registry.
//! Verified against llama.cpp `gguf/constants.py` (`TENSOR_NAMES`) /
//! `tensor_mapping.py`.

/// Skip tensor, or map HF key → canonical, or keep non-HF keys as-is.
pub fn canonicalize_hf_family(key: &str) -> Option<String> {
    if key.contains("rotary_emb") || key.ends_with(".inv_freq") {
        return None;
    }
    if let Some(mapped) = try_hf_remap(key) {
        return Some(mapped);
    }
    Some(key.to_string())
}

fn try_hf_remap(key: &str) -> Option<String> {
    if key == "model.embed_tokens.weight" || key == "embed_tokens.weight" {
        return Some("token_embd.weight".into());
    }
    if key == "model.norm.weight" || key == "norm.weight" {
        return Some("output_norm.weight".into());
    }
    if key == "lm_head.weight" {
        return Some("output.weight".into());
    }

    let rest = key
        .strip_prefix("model.layers.")
        .or_else(|| key.strip_prefix("layers."))?;
    let (idx_str, suffix) = rest.split_once('.')?;
    let layer: u32 = idx_str.parse().ok()?;
    let canonical_suffix = match suffix {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => return None,
    };
    Some(format!("blk.{layer}.{canonical_suffix}"))
}

/// Synthesize `output.weight` from `token_embd.weight` when lm_head is absent.
pub fn tie_output_to_embed(catalog: &mut dyninfer_checkpoint::ParameterCatalog) {
    let has_output = catalog
        .parameters
        .iter()
        .any(|p| p.canonical_name.as_str() == "output.weight");
    if has_output {
        return;
    }
    let Some(emb) = catalog
        .parameters
        .iter()
        .find(|p| p.canonical_name.as_str() == "token_embd.weight")
        .cloned()
    else {
        return;
    };
    let mut tied = emb;
    tied.canonical_name = dyninfer_core::CanonicalParameterName::new("output.weight");
    tied.role = dyninfer_core::ParameterRole::Output;
    tied.aliases.push("output.weight".into());
    tied.aliases.push("lm_head.weight".into());
    catalog.parameters.push(tied);
    catalog
        .metadata
        .entry("tie_word_embeddings".into())
        .or_insert(serde_json::Value::Bool(true));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaps_hf_and_keeps_gguf() {
        assert_eq!(
            canonicalize_hf_family("model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            canonicalize_hf_family("token_embd.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            canonicalize_hf_family("blk.0.attn_q.weight").as_deref(),
            Some("blk.0.attn_q.weight")
        );
        assert!(canonicalize_hf_family("model.layers.0.self_attn.rotary_emb.inv_freq").is_none());
        assert_eq!(
            canonicalize_hf_family("model.layers.0.self_attn.q_norm.weight").as_deref(),
            Some("blk.0.attn_q_norm.weight")
        );
    }
}
