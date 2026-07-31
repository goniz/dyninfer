//! Tiny SafeTensors fixture helpers for tests and local demos.

use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Build an in-memory SafeTensors blob with dense F32 tensors.
pub fn write_safetensors(tensors: &BTreeMap<String, (Vec<u64>, Vec<f32>)>, metadata: Value) -> Vec<u8> {
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

/// Minimal Llama-shaped dense checkpoint (1 layer, tiny dims).
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
    });

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "token_embd.weight".into(),
        (vec![vocab, hidden], vec![0.01f32; (vocab * hidden) as usize]),
    );
    tensors.insert(
        "blk.0.attn_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "blk.0.attn_q.weight".into(),
        (vec![hidden, hidden], vec![0.02f32; (hidden * hidden) as usize]),
    );
    tensors.insert(
        "blk.0.attn_k.weight".into(),
        (vec![hidden, hidden], vec![0.02f32; (hidden * hidden) as usize]),
    );
    tensors.insert(
        "blk.0.attn_v.weight".into(),
        (vec![hidden, hidden], vec![0.02f32; (hidden * hidden) as usize]),
    );
    tensors.insert(
        "blk.0.attn_output.weight".into(),
        (vec![hidden, hidden], vec![0.02f32; (hidden * hidden) as usize]),
    );
    tensors.insert(
        "blk.0.ffn_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "blk.0.ffn_gate.weight".into(),
        (
            vec![intermediate, hidden],
            vec![0.03f32; (intermediate * hidden) as usize],
        ),
    );
    tensors.insert(
        "blk.0.ffn_up.weight".into(),
        (
            vec![intermediate, hidden],
            vec![0.03f32; (intermediate * hidden) as usize],
        ),
    );
    tensors.insert(
        "blk.0.ffn_down.weight".into(),
        (
            vec![hidden, intermediate],
            vec![0.03f32; (hidden * intermediate) as usize],
        ),
    );
    tensors.insert(
        "output_norm.weight".into(),
        (vec![hidden], vec![1.0f32; hidden as usize]),
    );
    tensors.insert(
        "output.weight".into(),
        (vec![vocab, hidden], vec![0.01f32; (vocab * hidden) as usize]),
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
}
