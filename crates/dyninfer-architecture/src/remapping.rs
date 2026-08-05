//! HF `model_type` / class → dyninfer architecture id (mlx-lm style).

/// Alias table: source identifier → module stem (`llama`, `qwen3`, …).
pub const MODEL_REMAPPING: &[(&str, &str)] = &[
    ("llama", "llama"),
    ("mistral", "llama"),
    ("LlamaForCausalLM", "llama"),
    ("MistralForCausalLM", "llama"),
    ("qwen3", "qwen3"),
    ("Qwen3ForCausalLM", "qwen3"),
    ("lfm2", "lfm2"),
    ("Lfm2ForCausalLM", "lfm2"),
];

/// Map a remapped stem to a stable architecture id.
pub fn architecture_id_for_stem(stem: &str) -> Option<&'static str> {
    match stem {
        "llama" => Some("llama.decoder"),
        "qwen3" => Some("qwen3.decoder"),
        "lfm2" => Some("lfm2.hybrid"),
        _ => None,
    }
}

/// Resolve an architecture id from a raw HF model_type / class name.
pub fn remap_model_type(model_type: &str) -> Option<&'static str> {
    let mt = model_type.trim();
    if mt.is_empty() {
        return None;
    }
    let stem = MODEL_REMAPPING
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(mt))
        .map(|(_, v)| *v)
        .or_else(|| {
            mt.strip_suffix("ForCausalLM")
                .map(|s| s.to_ascii_lowercase())
                .and_then(|s| {
                    MODEL_REMAPPING
                        .iter()
                        .find(|(k, _)| k.eq_ignore_ascii_case(&s))
                        .map(|(_, v)| *v)
                })
        })
        .unwrap_or(mt);
    architecture_id_for_stem(stem).or_else(|| {
        // Direct stem match when not in alias table.
        architecture_id_for_stem(&stem.to_ascii_lowercase())
    })
}
