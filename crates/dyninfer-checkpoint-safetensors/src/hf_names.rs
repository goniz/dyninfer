//! HuggingFace Transformers → GGUF-style canonical parameter names.

/// Map a Transformers / HF tensor key to the dyninfer Llama canonical name.
///
/// Returns `None` for tensors that should be skipped (e.g. cached RoPE freqs).
pub fn hf_to_canonical(key: &str) -> Option<String> {
    if key.contains("rotary_emb") || key.ends_with(".inv_freq") {
        return None;
    }
    if key == "model.embed_tokens.weight" || key == "embed_tokens.weight" {
        return Some("token_embd.weight".into());
    }
    if key == "model.norm.weight" || key == "norm.weight" {
        return Some("output_norm.weight".into());
    }
    if key == "lm_head.weight" {
        return Some("output.weight".into());
    }

    // model.layers.{i}....
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
        // Qwen3 / Qwen2.5: per-head RMSNorm on Q/K (before RoPE).
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => return None,
    };
    Some(format!("blk.{layer}.{canonical_suffix}"))
}

/// Whether `key` looks like a HuggingFace Transformers Llama weight name.
pub fn looks_like_hf_llama(key: &str) -> bool {
    key.starts_with("model.") || key == "lm_head.weight" || key.contains("rotary_emb")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaps_common_keys() {
        assert_eq!(
            hf_to_canonical("model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            hf_to_canonical("model.layers.3.self_attn.q_proj.weight").as_deref(),
            Some("blk.3.attn_q.weight")
        );
        assert_eq!(
            hf_to_canonical("lm_head.weight").as_deref(),
            Some("output.weight")
        );
        assert!(hf_to_canonical("model.layers.0.self_attn.rotary_emb.inv_freq").is_none());
        assert_eq!(
            hf_to_canonical("model.layers.0.self_attn.q_norm.weight").as_deref(),
            Some("blk.0.attn_q_norm.weight")
        );
        assert_eq!(
            hf_to_canonical("model.layers.2.self_attn.k_norm.weight").as_deref(),
            Some("blk.2.attn_k_norm.weight")
        );
    }
}
