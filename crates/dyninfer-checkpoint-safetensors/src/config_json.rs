//! Load HuggingFace `config.json` sitting next to a SafeTensors checkpoint.

use dyninfer_core::MetadataMap;
use serde_json::Value;
use std::fs;
use std::path::Path;

/// Read sibling `config.json` and map common Transformers Llama fields into
/// dyninfer / GGUF-style metadata keys.
pub fn load_hf_config_metadata(checkpoint: &Path) -> MetadataMap {
    let mut out = MetadataMap::new();
    let Some(dir) = checkpoint.parent() else {
        return out;
    };
    let path = dir.join("config.json");
    let Ok(bytes) = fs::read(&path) else {
        return out;
    };
    let Ok(Value::Object(obj)) = serde_json::from_slice::<Value>(&bytes) else {
        return out;
    };

    let u = |k: &str| obj.get(k).and_then(|v| v.as_u64());
    let f = |k: &str| obj.get(k).and_then(|v| v.as_f64());

    if let Some(v) = u("num_hidden_layers")
        .or_else(|| u("n_layer"))
        .or_else(|| u("num_layers"))
    {
        out.insert("num_layers".into(), Value::from(v));
        out.insert("n_layer".into(), Value::from(v));
        out.insert("block_count".into(), Value::from(v));
        out.insert("llama.block_count".into(), Value::from(v));
    }
    if let Some(v) = u("hidden_size").or_else(|| u("n_embd")) {
        out.insert("hidden_size".into(), Value::from(v));
        out.insert("llama.embedding_length".into(), Value::from(v));
    }
    if let Some(v) = u("intermediate_size") {
        out.insert("intermediate_size".into(), Value::from(v));
    }
    if let Some(v) = u("num_attention_heads")
        .or_else(|| u("n_head"))
        .or_else(|| u("num_heads"))
    {
        out.insert("num_heads".into(), Value::from(v));
        out.insert("llama.attention.head_count".into(), Value::from(v));
    }
    let kv_heads = u("num_key_value_heads")
        .or_else(|| u("num_kv_heads"))
        .or_else(|| out.get("num_heads").and_then(|v| v.as_u64()));
    if let Some(v) = kv_heads {
        out.insert("num_kv_heads".into(), Value::from(v));
        out.insert("llama.attention.head_count_kv".into(), Value::from(v));
    }
    if let Some(v) = u("vocab_size") {
        out.insert("vocab_size".into(), Value::from(v));
        out.insert("llama.vocab_size".into(), Value::from(v));
    }
    if let Some(v) = u("max_position_embeddings").or_else(|| u("context_length")) {
        out.insert("context_length".into(), Value::from(v));
        out.insert("llama.context_length".into(), Value::from(v));
    }
    // `norm_eps` is the LFM2 spelling of the same RMSNorm epsilon.
    if let Some(v) = f("rms_norm_eps").or_else(|| f("norm_eps")) {
        out.insert("rms_norm_eps".into(), Value::from(v));
    }
    // Transformers v5 nests RoPE settings under `rope_parameters`.
    let nested_rope_theta = obj
        .get("rope_parameters")
        .or_else(|| obj.get("rope_scaling"))
        .and_then(Value::as_object)
        .and_then(|rope| rope.get("rope_theta"))
        .and_then(|v| v.as_f64());
    if let Some(v) = f("rope_theta")
        .or_else(|| u("rope_theta").map(|x| x as f64))
        .or(nested_rope_theta)
    {
        out.insert("rope_theta".into(), Value::from(v));
    } else {
        out.insert("rope_theta".into(), Value::from(10000.0));
    }
    // LFM2 hybrid schedule: explicit per-layer operator kinds plus the causal
    // window of the short convolution.
    if let Some(v) = obj.get("layer_types").and_then(|v| v.as_array()) {
        out.insert("layer_types".into(), Value::Array(v.clone()));
    }
    if let Some(v) = u("conv_L_cache").or_else(|| u("conv_kernel")) {
        out.insert("conv_kernel".into(), Value::from(v));
    }
    if let Some(v) = u("conv_dim") {
        out.insert("conv_dim".into(), Value::from(v));
    }
    if let Some(v) = u("bos_token_id") {
        out.insert("bos_token_id".into(), Value::from(v));
    }
    if let Some(v) = u("eos_token_id") {
        out.insert("eos_token_id".into(), Value::from(v));
    }
    if let Some(v) = u("pad_token_id") {
        out.insert("pad_token_id".into(), Value::from(v));
    }

    // Prefer explicit head_dim (Qwen3: 128 with hidden=1024, heads=16).
    if let Some(v) = u("head_dim") {
        out.insert("head_dim".into(), Value::from(v));
    } else {
        let hidden = out.get("hidden_size").and_then(|v| v.as_u64());
        let heads = out.get("num_heads").and_then(|v| v.as_u64());
        if let (Some(h), Some(n)) = (hidden, heads) {
            if n > 0 && h % n == 0 {
                out.insert("head_dim".into(), Value::from(h / n));
            }
        }
    }

    if let Some(v) = obj.get("architectures").and_then(|v| v.as_array()) {
        if let Some(Value::String(name)) = v.first() {
            out.insert("hf_architecture".into(), Value::String(name.clone()));
        }
    }
    if let Some(v) = obj.get("model_type").and_then(|v| v.as_str()) {
        out.insert("model_type".into(), Value::from(v));
    }
    if let Some(v) = obj.get("tie_word_embeddings").and_then(|v| v.as_bool()) {
        out.insert("tie_word_embeddings".into(), Value::from(v));
    }

    let quantization = obj
        .get("quantization_config")
        .or_else(|| obj.get("quantization"))
        .and_then(Value::as_object);
    if let Some(quantization) = quantization {
        if let Some(bits) = quantization.get("bits").and_then(Value::as_u64) {
            out.insert("mlx.quantization.bits".into(), Value::from(bits));
        }
        if let Some(group_size) = quantization.get("group_size").and_then(Value::as_u64) {
            out.insert(
                "mlx.quantization.group_size".into(),
                Value::from(group_size),
            );
        }
    }

    out
}
