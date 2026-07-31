//! Heuristic role inference from tensor key names.

use dyninfer_core::ParameterRole;

/// Infer a [`ParameterRole`] from a common checkpoint key / canonical name.
pub fn infer_role(name: &str) -> ParameterRole {
    let n = name.to_ascii_lowercase();
    if n.contains("embed") || n.contains("token_embd") || n.contains("tok_embeddings") {
        return ParameterRole::Embedding;
    }
    // Qwen3 q_norm/k_norm contain "attn_q"/"attn_k" substrings — check before Q/K.
    if n.contains("q_norm") || n.contains("k_norm") {
        return ParameterRole::Norm;
    }
    if n.contains("attn_q") || n.contains("attention.wq") || n.contains("q_proj") || n.ends_with(".q.weight")
    {
        return ParameterRole::AttentionQ;
    }
    if n.contains("attn_k") || n.contains("attention.wk") || n.contains("k_proj") || n.ends_with(".k.weight")
    {
        return ParameterRole::AttentionK;
    }
    if n.contains("attn_v") || n.contains("attention.wv") || n.contains("v_proj") || n.ends_with(".v.weight")
    {
        return ParameterRole::AttentionV;
    }
    if n.contains("attn_output")
        || n.contains("attn_o")
        || n.contains("attention.wo")
        || n.contains("o_proj")
        || n.ends_with(".o.weight")
    {
        return ParameterRole::AttentionO;
    }
    if n.contains("ffn_gate") || n.contains("w1") || n.contains("gate_proj") {
        return ParameterRole::FfnGate;
    }
    if n.contains("ffn_up") || n.contains("w3") || n.contains("up_proj") {
        return ParameterRole::FfnUp;
    }
    if n.contains("ffn_down") || n.contains("w2") || n.contains("down_proj") {
        return ParameterRole::FfnDown;
    }
    if n.contains("norm") || n.contains("ln_") || n.contains("layernorm") {
        return ParameterRole::Norm;
    }
    if n.contains("output") || n.contains("lm_head") || n.ends_with("output.weight") {
        return ParameterRole::Output;
    }
    if n.contains("bias") {
        return ParameterRole::Bias;
    }
    if n.contains("rope") || n.contains("freq") {
        return ParameterRole::RopeFreqs;
    }
    ParameterRole::Other(name.to_string())
}
