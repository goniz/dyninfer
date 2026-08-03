//! Tiny SafeTensors fixture helpers for tests and local demos.

use serde_json::{Value, json};
use std::collections::BTreeMap;

enum FixtureTensorData {
    F32(Vec<f32>),
    U32(Vec<u32>),
}

fn write_mixed_safetensors(
    tensors: &BTreeMap<String, (Vec<u64>, FixtureTensorData)>,
    metadata: Value,
) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    if !metadata.is_null() {
        header.insert("__metadata__".into(), metadata);
    }

    let mut data = Vec::new();
    for (name, (shape, values)) in tensors {
        let start = data.len() as u64;
        let dtype = match values {
            FixtureTensorData::F32(values) => {
                for value in values {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                "F32"
            }
            FixtureTensorData::U32(values) => {
                for value in values {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                "U32"
            }
        };
        let end = data.len() as u64;
        header.insert(
            name.clone(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }

    let header_bytes = Value::Object(header).to_string().into_bytes();
    let mut out = Vec::with_capacity(8 + header_bytes.len() + data.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&data);
    out
}

/// Build an in-memory SafeTensors blob with dense F32 tensors.
pub fn write_safetensors(
    tensors: &BTreeMap<String, (Vec<u64>, Vec<f32>)>,
    metadata: Value,
) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    if !metadata.is_null() {
        header.insert("__metadata__".into(), metadata);
    }

    let mut data = Vec::new();
    for (name, (shape, values)) in tensors {
        let start = data.len() as u64;
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let end = data.len() as u64;
        header.insert(
            name.clone(),
            json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }

    let header_bytes = Value::Object(header).to_string().into_bytes();
    let mut out = Vec::with_capacity(8 + header_bytes.len() + data.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&data);
    out
}

/// Deterministic pseudo-random fills used by the M1 reference + IREE path.
pub fn fill_f32(n: usize, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut x = seed;
    for i in 0..n {
        x = x
            .wrapping_mul(1664525)
            .wrapping_add(1013904223)
            .wrapping_add(i as u32);
        out.push(((x % 1000) as f32) / 1000.0 - 0.5);
    }
    out
}

/// Minimal Llama-shaped dense checkpoint (1 layer, tiny dims, varied weights).
pub fn tiny_llama_dense_f32() -> Vec<u8> {
    let vocab = 32u64;
    let hidden = 64u64;
    let intermediate = 128u64;

    let meta = json!({
        "num_layers": 1,
        "n_layer": 1,
        "block_count": 1,
        "num_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 16,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "context_length": 128,
        "llama.vocab_size": vocab,
        "llama.embedding_length": hidden,
        "llama.attention.head_count": 4,
        "llama.block_count": 1,
        "rms_norm_eps": 1e-5,
    });

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "token_embd.weight".into(),
        (vec![vocab, hidden], fill_f32((vocab * hidden) as usize, 1)),
    );
    tensors.insert(
        "blk.0.attn_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "blk.0.attn_q.weight".into(),
        (
            vec![hidden, hidden],
            fill_f32((hidden * hidden) as usize, 2),
        ),
    );
    tensors.insert(
        "blk.0.attn_k.weight".into(),
        (
            vec![hidden, hidden],
            fill_f32((hidden * hidden) as usize, 3),
        ),
    );
    tensors.insert(
        "blk.0.attn_v.weight".into(),
        (
            vec![hidden, hidden],
            fill_f32((hidden * hidden) as usize, 4),
        ),
    );
    tensors.insert(
        "blk.0.attn_output.weight".into(),
        (
            vec![hidden, hidden],
            fill_f32((hidden * hidden) as usize, 5),
        ),
    );
    tensors.insert(
        "blk.0.ffn_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "blk.0.ffn_gate.weight".into(),
        (
            vec![intermediate, hidden],
            fill_f32((intermediate * hidden) as usize, 6),
        ),
    );
    tensors.insert(
        "blk.0.ffn_up.weight".into(),
        (
            vec![intermediate, hidden],
            fill_f32((intermediate * hidden) as usize, 7),
        ),
    );
    tensors.insert(
        "blk.0.ffn_down.weight".into(),
        (
            vec![hidden, intermediate],
            fill_f32((hidden * intermediate) as usize, 8),
        ),
    );
    tensors.insert(
        "output_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "output.weight".into(),
        (vec![vocab, hidden], fill_f32((vocab * hidden) as usize, 9)),
    );

    write_safetensors(&tensors, meta)
}

/// Minimal Llama fixture using the exact MLX affine-U4 storage topology.
///
/// Every rank-2 weight remains packed as U32 words with one F32 scale and bias
/// per 64-value group. Rank-1 normalization weights stay dense. This is used to
/// differentially exercise direct grouped dequantization in the compiled path.
pub fn tiny_llama_mlx_affine_u4() -> Vec<u8> {
    const VOCAB: u64 = 32;
    const HIDDEN: u64 = 64;
    const INTERMEDIATE: u64 = 128;
    const GROUP: usize = 64;

    let meta = json!({
        "format": "mlx",
        "mlx.quantization.bits": 4,
        "mlx.quantization.group_size": GROUP,
        "num_layers": 1,
        "n_layer": 1,
        "block_count": 1,
        "num_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 16,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "context_length": 128,
        "llama.vocab_size": VOCAB,
        "llama.embedding_length": HIDDEN,
        "llama.attention.head_count": 4,
        "llama.block_count": 1,
        "rms_norm_eps": 1e-5,
    });

    let specs = [
        ("token_embd.weight", VOCAB, HIDDEN, 1),
        ("blk.0.attn_q.weight", HIDDEN, HIDDEN, 2),
        ("blk.0.attn_k.weight", HIDDEN, HIDDEN, 3),
        ("blk.0.attn_v.weight", HIDDEN, HIDDEN, 4),
        ("blk.0.attn_output.weight", HIDDEN, HIDDEN, 5),
        ("blk.0.ffn_gate.weight", INTERMEDIATE, HIDDEN, 6),
        ("blk.0.ffn_up.weight", INTERMEDIATE, HIDDEN, 7),
        ("blk.0.ffn_down.weight", HIDDEN, INTERMEDIATE, 8),
        ("output.weight", VOCAB, HIDDEN, 9),
    ];
    let mut tensors = BTreeMap::new();
    for (name, rows, cols, seed) in specs {
        let values = fill_f32((rows * cols) as usize, seed);
        let (packed, scales, biases) =
            quantize_affine_u4(&values, rows as usize, cols as usize, GROUP);
        let base = name.strip_suffix(".weight").unwrap();
        tensors.insert(
            name.into(),
            (vec![rows, cols / 8], FixtureTensorData::U32(packed)),
        );
        tensors.insert(
            format!("{base}.scales"),
            (
                vec![rows, cols / GROUP as u64],
                FixtureTensorData::F32(scales),
            ),
        );
        tensors.insert(
            format!("{base}.biases"),
            (
                vec![rows, cols / GROUP as u64],
                FixtureTensorData::F32(biases),
            ),
        );
    }
    for name in [
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "output_norm.weight",
    ] {
        tensors.insert(
            name.into(),
            (
                vec![HIDDEN],
                FixtureTensorData::F32(vec![1.0; HIDDEN as usize]),
            ),
        );
    }
    write_mixed_safetensors(&tensors, meta)
}

fn quantize_affine_u4(
    values: &[f32],
    rows: usize,
    columns: usize,
    group_size: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    assert_eq!(values.len(), rows * columns);
    assert!(columns.is_multiple_of(group_size));
    assert!(group_size.is_multiple_of(8));
    let mut packed = Vec::with_capacity(values.len() / 8);
    let mut scales = Vec::with_capacity(values.len() / group_size);
    let mut biases = Vec::with_capacity(values.len() / group_size);
    for row in values.chunks_exact(columns) {
        for group in row.chunks_exact(group_size) {
            let min = group.iter().copied().fold(f32::INFINITY, f32::min);
            let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = ((max - min) / 15.0).max(1.0e-7);
            scales.push(scale);
            biases.push(min);
            for lanes in group.chunks_exact(8) {
                let mut word = 0u32;
                for (lane, value) in lanes.iter().enumerate() {
                    let quantized = ((*value - min) / scale).round().clamp(0.0, 15.0) as u32;
                    word |= quantized << (lane * 4);
                }
                packed.push(word);
            }
        }
    }
    (packed, scales, biases)
}

/// Tiny GQA + RoPE + Q/K-norm fixture (not the synthetic no-RoPE M1 shape).
///
/// Dims deliberately differ from the M1 synthetic heuristic (`vocab=32, hidden=64,
/// heads=4`) so shared specialization keeps RoPE enabled.
pub fn tiny_gqa_rope_f32() -> Vec<u8> {
    let vocab = 48u64;
    let hidden = 32u64;
    let intermediate = 64u64;
    let num_heads = 4u64;
    let num_kv_heads = 2u64;
    let head_dim = 8u64;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let meta = json!({
        "num_layers": 1,
        "n_layer": 1,
        "block_count": 1,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "context_length": 128,
        "llama.vocab_size": vocab,
        "llama.embedding_length": hidden,
        "llama.attention.head_count": num_heads,
        "llama.attention.head_count_kv": num_kv_heads,
        "llama.block_count": 1,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "model_type": "qwen3",
        "hf_architecture": "Qwen3ForCausalLM",
        "prefill_window": 4,
        "max_kv": 8,
    });

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "token_embd.weight".into(),
        (vec![vocab, hidden], fill_f32((vocab * hidden) as usize, 1)),
    );
    tensors.insert(
        "blk.0.attn_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "blk.0.attn_q.weight".into(),
        (vec![q_dim, hidden], fill_f32((q_dim * hidden) as usize, 2)),
    );
    tensors.insert(
        "blk.0.attn_k.weight".into(),
        (
            vec![kv_dim, hidden],
            fill_f32((kv_dim * hidden) as usize, 3),
        ),
    );
    tensors.insert(
        "blk.0.attn_v.weight".into(),
        (
            vec![kv_dim, hidden],
            fill_f32((kv_dim * hidden) as usize, 4),
        ),
    );
    tensors.insert(
        "blk.0.attn_output.weight".into(),
        (vec![hidden, q_dim], fill_f32((hidden * q_dim) as usize, 5)),
    );
    tensors.insert(
        "blk.0.attn_q_norm.weight".into(),
        (vec![head_dim], vec![1.0f32; head_dim as usize]),
    );
    tensors.insert(
        "blk.0.attn_k_norm.weight".into(),
        (vec![head_dim], vec![1.0f32; head_dim as usize]),
    );
    tensors.insert(
        "blk.0.ffn_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "blk.0.ffn_gate.weight".into(),
        (
            vec![intermediate, hidden],
            fill_f32((intermediate * hidden) as usize, 6),
        ),
    );
    tensors.insert(
        "blk.0.ffn_up.weight".into(),
        (
            vec![intermediate, hidden],
            fill_f32((intermediate * hidden) as usize, 7),
        ),
    );
    tensors.insert(
        "blk.0.ffn_down.weight".into(),
        (
            vec![hidden, intermediate],
            fill_f32((hidden * intermediate) as usize, 8),
        ),
    );
    tensors.insert(
        "output_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    // Tied output omitted — explicit lm_head for the fixture.
    tensors.insert(
        "output.weight".into(),
        (vec![vocab, hidden], fill_f32((vocab * hidden) as usize, 9)),
    );

    write_safetensors(&tensors, meta)
}

/// GQA like [`tiny_gqa_rope_f32`] but no RoPE metadata and no Q/K-norm (llama path).
pub fn tiny_gqa_plain_f32() -> Vec<u8> {
    let vocab = 48u64;
    let hidden = 32u64;
    let intermediate = 64u64;
    let num_heads = 4u64;
    let num_kv_heads = 2u64;
    let head_dim = 8u64;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let meta = json!({
        "num_layers": 1,
        "n_layer": 1,
        "block_count": 1,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "context_length": 128,
        "llama.vocab_size": vocab,
        "llama.embedding_length": hidden,
        "llama.attention.head_count": num_heads,
        "llama.attention.head_count_kv": num_kv_heads,
        "llama.block_count": 1,
        "rms_norm_eps": 1e-5,
        "rope_theta": 0.0,
        "prefill_window": 4,
        "max_kv": 8,
    });

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "token_embd.weight".into(),
        (vec![vocab, hidden], fill_f32((vocab * hidden) as usize, 1)),
    );
    tensors.insert(
        "blk.0.attn_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "blk.0.attn_q.weight".into(),
        (vec![q_dim, hidden], fill_f32((q_dim * hidden) as usize, 2)),
    );
    tensors.insert(
        "blk.0.attn_k.weight".into(),
        (
            vec![kv_dim, hidden],
            fill_f32((kv_dim * hidden) as usize, 3),
        ),
    );
    tensors.insert(
        "blk.0.attn_v.weight".into(),
        (
            vec![kv_dim, hidden],
            fill_f32((kv_dim * hidden) as usize, 4),
        ),
    );
    tensors.insert(
        "blk.0.attn_output.weight".into(),
        (vec![hidden, q_dim], fill_f32((hidden * q_dim) as usize, 5)),
    );
    tensors.insert(
        "blk.0.ffn_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "blk.0.ffn_gate.weight".into(),
        (
            vec![intermediate, hidden],
            fill_f32((intermediate * hidden) as usize, 6),
        ),
    );
    tensors.insert(
        "blk.0.ffn_up.weight".into(),
        (
            vec![intermediate, hidden],
            fill_f32((intermediate * hidden) as usize, 7),
        ),
    );
    tensors.insert(
        "blk.0.ffn_down.weight".into(),
        (
            vec![hidden, intermediate],
            fill_f32((hidden * intermediate) as usize, 8),
        ),
    );
    tensors.insert(
        "output_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "output.weight".into(),
        (vec![vocab, hidden], fill_f32((vocab * hidden) as usize, 9)),
    );
    write_safetensors(&tensors, meta)
}

/// Same dims as [`tiny_gqa_rope_f32`] but MHA (kv_heads == heads), no Q/K-norm.
/// Used to bisect decode parity: RoPE without GQA/qk_norm.
pub fn tiny_mha_rope_f32() -> Vec<u8> {
    let vocab = 48u64;
    let hidden = 32u64;
    let intermediate = 64u64;
    let num_heads = 4u64;
    let head_dim = 8u64;
    let q_dim = num_heads * head_dim;

    let meta = json!({
        "num_layers": 1,
        "n_layer": 1,
        "block_count": 1,
        "num_heads": num_heads,
        "num_kv_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "context_length": 128,
        "llama.vocab_size": vocab,
        "llama.embedding_length": hidden,
        "llama.attention.head_count": num_heads,
        "llama.attention.head_count_kv": num_heads,
        "llama.block_count": 1,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "prefill_window": 4,
        "max_kv": 8,
    });

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "token_embd.weight".into(),
        (vec![vocab, hidden], fill_f32((vocab * hidden) as usize, 1)),
    );
    tensors.insert(
        "blk.0.attn_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "blk.0.attn_q.weight".into(),
        (vec![q_dim, hidden], fill_f32((q_dim * hidden) as usize, 2)),
    );
    tensors.insert(
        "blk.0.attn_k.weight".into(),
        (vec![q_dim, hidden], fill_f32((q_dim * hidden) as usize, 3)),
    );
    tensors.insert(
        "blk.0.attn_v.weight".into(),
        (vec![q_dim, hidden], fill_f32((q_dim * hidden) as usize, 4)),
    );
    tensors.insert(
        "blk.0.attn_output.weight".into(),
        (vec![hidden, q_dim], fill_f32((hidden * q_dim) as usize, 5)),
    );
    tensors.insert(
        "blk.0.ffn_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "blk.0.ffn_gate.weight".into(),
        (
            vec![intermediate, hidden],
            fill_f32((intermediate * hidden) as usize, 6),
        ),
    );
    tensors.insert(
        "blk.0.ffn_up.weight".into(),
        (
            vec![intermediate, hidden],
            fill_f32((intermediate * hidden) as usize, 7),
        ),
    );
    tensors.insert(
        "blk.0.ffn_down.weight".into(),
        (
            vec![hidden, intermediate],
            fill_f32((hidden * intermediate) as usize, 8),
        ),
    );
    tensors.insert(
        "output_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "output.weight".into(),
        (vec![vocab, hidden], fill_f32((vocab * hidden) as usize, 9)),
    );

    write_safetensors(&tensors, meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SafeTensorsContainer;
    use dyninfer_checkpoint::{CheckpointContainerReader, InspectionLimits};
    use std::sync::Arc;

    #[test]
    fn tiny_llama_indexes() {
        let bytes = tiny_llama_dense_f32();
        let source = Arc::new(dyninfer_checkpoint::BytesSource::new(bytes));
        let index = SafeTensorsContainer
            .index(source, &InspectionLimits::default())
            .unwrap();
        assert!(index.entries.len() >= 12);
        assert_eq!(
            index.metadata.get("num_layers").and_then(|v| v.as_u64()),
            Some(1)
        );
    }

    #[test]
    fn tiny_mlx_u4_indexes_mixed_components() {
        let bytes = tiny_llama_mlx_affine_u4();
        let source = Arc::new(dyninfer_checkpoint::BytesSource::new(bytes));
        let index = SafeTensorsContainer
            .index(source, &InspectionLimits::default())
            .unwrap();
        assert_eq!(index.entries.len(), 30);
        assert_eq!(
            index
                .metadata
                .get("mlx.quantization.bits")
                .and_then(|value| value.as_u64()),
            Some(4)
        );
        assert!(index.entries.iter().any(|entry| {
            entry.key == "token_embd.weight"
                && entry.shape == [32, 8]
                && entry.storage_type
                    == dyninfer_core::StorageElementType::scalar(dyninfer_core::ScalarType::U32)
        }));
    }
}
