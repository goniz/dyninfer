//! GGUF Q4_0 pack / dequant (reference + fixture helpers).

use crate::types::GgufType;
use byteorder::{LittleEndian, WriteBytesExt};
use dyninfer_error::{DynInferError, Result};
use half::f16;
use std::collections::BTreeMap;
use std::io::Write;

pub const Q4_0_BLOCK: usize = 32;
pub const Q4_0_TYPE_SIZE: usize = 18; // f16 scale + 16 packed nibbles
pub const Q4_1_TYPE_SIZE: usize = 20; // f16 scale + f16 minimum + 16 nibbles
pub const Q8_0_TYPE_SIZE: usize = 34; // f16 scale + 32 signed bytes

/// Pack row-major f32 weights into GGUF Q4_0 bytes (numel must be divisible by 32).
pub fn pack_q4_0(weights: &[f32]) -> Result<Vec<u8>> {
    if !weights.len().is_multiple_of(Q4_0_BLOCK) {
        return Err(DynInferError::io(format!(
            "Q4_0 pack: numel {} not divisible by {Q4_0_BLOCK}",
            weights.len()
        )));
    }
    let nblocks = weights.len() / Q4_0_BLOCK;
    let mut out = Vec::with_capacity(nblocks * Q4_0_TYPE_SIZE);
    for block in weights.chunks_exact(Q4_0_BLOCK) {
        let mut amax = 0.0f32;
        for &w in block {
            amax = amax.max(w.abs());
        }
        let scale = if amax > 0.0 { amax / 7.0 } else { 0.0 };
        let scale_f16 = f16::from_f32(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let mut qs = [0u8; 16];
        // GGML stores values 0..16 in the low nibbles and values 16..32
        // in the high nibbles of the same 16 bytes. It does not interleave
        // adjacent logical values in one byte.
        for i in 0..16 {
            let low = ((block[i] * inv).round() as i32).clamp(-8, 7) + 8;
            let high = ((block[i + 16] * inv).round() as i32).clamp(-8, 7) + 8;
            qs[i] = low as u8 | ((high as u8) << 4);
        }
        out.extend_from_slice(&qs);
    }
    Ok(out)
}

fn pack_q4_1(weights: &[f32]) -> Result<Vec<u8>> {
    if !weights.len().is_multiple_of(Q4_0_BLOCK) {
        return Err(DynInferError::io(format!(
            "Q4_1 pack: numel {} not divisible by {Q4_0_BLOCK}",
            weights.len()
        )));
    }
    let mut out = Vec::with_capacity(weights.len() / Q4_0_BLOCK * Q4_1_TYPE_SIZE);
    for block in weights.chunks_exact(Q4_0_BLOCK) {
        let minimum = block.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if maximum > minimum {
            (maximum - minimum) / 15.0
        } else {
            0.0
        };
        out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(minimum).to_le_bytes());
        let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for lane in 0..16 {
            let low = ((block[lane] - minimum) * inverse).round().clamp(0.0, 15.0) as u8;
            let high = ((block[lane + 16] - minimum) * inverse)
                .round()
                .clamp(0.0, 15.0) as u8;
            out.push(low | (high << 4));
        }
    }
    Ok(out)
}

fn pack_q8_0(weights: &[f32]) -> Result<Vec<u8>> {
    if !weights.len().is_multiple_of(Q4_0_BLOCK) {
        return Err(DynInferError::io(format!(
            "Q8_0 pack: numel {} not divisible by {Q4_0_BLOCK}",
            weights.len()
        )));
    }
    let mut out = Vec::with_capacity(weights.len() / Q4_0_BLOCK * Q8_0_TYPE_SIZE);
    for block in weights.chunks_exact(Q4_0_BLOCK) {
        let amax = block.iter().map(|value| value.abs()).fold(0.0, f32::max);
        let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        out.extend(
            block
                .iter()
                .map(|value| ((*value * inverse).round() as i32).clamp(-127, 127) as i8 as u8),
        );
    }
    Ok(out)
}

/// Dequantize GGUF Q4_0 packed bytes to f32 (reference / host path).
#[cfg(test)]
pub fn dequant_q4_0(packed: &[u8], numel: usize) -> Result<Vec<f32>> {
    if !numel.is_multiple_of(Q4_0_BLOCK) {
        return Err(DynInferError::io(format!(
            "Q4_0 dequant: numel {numel} not divisible by {Q4_0_BLOCK}"
        )));
    }
    let nblocks = numel / Q4_0_BLOCK;
    let expected = nblocks * Q4_0_TYPE_SIZE;
    if packed.len() != expected {
        return Err(DynInferError::io(format!(
            "Q4_0 dequant: packed {} bytes, expected {expected}",
            packed.len()
        )));
    }
    let mut out = Vec::with_capacity(numel);
    for bi in 0..nblocks {
        let off = bi * Q4_0_TYPE_SIZE;
        let scale = f16::from_le_bytes([packed[off], packed[off + 1]]).to_f32();
        let qs = &packed[off + 2..off + 18];
        for i in 0..Q4_0_BLOCK {
            let byte = qs[i % 16];
            let nibble = if i < 16 { byte & 0x0f } else { byte >> 4 };
            out.push((nibble as i32 - 8) as f32 * scale);
        }
    }
    Ok(out)
}

/// Bytes required for a logical numel under Q4_0.
pub fn q4_0_nbytes(numel: u64) -> Result<u64> {
    GgufType::Q4_0.nbytes_for_shape(&[numel])
}

/// Write a minimal GGUF v3 file.
///
/// `tensors`: name → (shape, ggml type code, raw bytes).
/// `metadata`: string/u32/f32 values under common keys.
pub fn write_gguf(
    tensors: &BTreeMap<String, (Vec<u64>, u32, Vec<u8>)>,
    metadata: &BTreeMap<String, MetaValue>,
) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    body.write_all(b"GGUF")?;
    body.write_u32::<LittleEndian>(3)?; // version
    body.write_u64::<LittleEndian>(tensors.len() as u64)?;
    body.write_u64::<LittleEndian>(metadata.len() as u64)?;

    for (k, v) in metadata {
        write_string(&mut body, k)?;
        match v {
            MetaValue::U32(x) => {
                body.write_u32::<LittleEndian>(4)?; // UINT32
                body.write_u32::<LittleEndian>(*x)?;
            }
            MetaValue::U64(x) => {
                body.write_u32::<LittleEndian>(10)?; // UINT64
                body.write_u64::<LittleEndian>(*x)?;
            }
            MetaValue::F32(x) => {
                body.write_u32::<LittleEndian>(6)?; // FLOAT32
                body.write_f32::<LittleEndian>(*x)?;
            }
            MetaValue::String(s) => {
                body.write_u32::<LittleEndian>(8)?; // STRING
                write_string(&mut body, s)?;
            }
        }
    }

    // Tensor infos with relative data offsets (from start of data section).
    let mut data = Vec::new();
    let alignment = 32u64;
    for (name, (shape, type_code, bytes)) in tensors {
        write_string(&mut body, name)?;
        body.write_u32::<LittleEndian>(shape.len() as u32)?;
        // GGUF on-disk dims are reversed vs row-major logical shape (reader reverses).
        for &d in shape.iter().rev() {
            body.write_u64::<LittleEndian>(d)?;
        }
        body.write_u32::<LittleEndian>(*type_code)?;
        // Align each tensor start within the data blob.
        let pad = (alignment - (data.len() as u64 % alignment)) % alignment;
        data.extend(std::iter::repeat_n(0u8, pad as usize));
        let offset = data.len() as u64;
        body.write_u64::<LittleEndian>(offset)?;
        data.extend_from_slice(bytes);
    }

    // Align data section start.
    let pad = (alignment - (body.len() as u64 % alignment)) % alignment;
    body.extend(std::iter::repeat_n(0u8, pad as usize));
    body.extend_from_slice(&data);
    Ok(body)
}

fn write_string(w: &mut Vec<u8>, s: &str) -> Result<()> {
    w.write_u64::<LittleEndian>(s.len() as u64)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

#[derive(Debug, Clone)]
pub enum MetaValue {
    U32(u32),
    U64(u64),
    F32(f32),
    String(String),
}

/// Deterministic pseudo-random fills (shared with SafeTensors fixtures).
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

/// Tiny Llama-shaped mixed GGUF: Q4_0/Q4_1/Q8_0 linears and plain embedding.
///
/// Dims match `tiny_llama_dense_f32` so the same architecture emit path applies.
pub fn tiny_llama_q4_0() -> Result<Vec<u8>> {
    let vocab = 32u64;
    let hidden = 64u64;
    let intermediate = 128u64;

    let mut meta = BTreeMap::new();
    meta.insert(
        "general.architecture".into(),
        MetaValue::String("llama".into()),
    );
    meta.insert("llama.block_count".into(), MetaValue::U32(1));
    meta.insert(
        "llama.embedding_length".into(),
        MetaValue::U32(hidden as u32),
    );
    meta.insert(
        "llama.feed_forward_length".into(),
        MetaValue::U32(intermediate as u32),
    );
    meta.insert("llama.attention.head_count".into(), MetaValue::U32(4));
    meta.insert("llama.attention.head_count_kv".into(), MetaValue::U32(4));
    meta.insert("llama.vocab_size".into(), MetaValue::U32(vocab as u32));
    meta.insert("llama.context_length".into(), MetaValue::U32(128));
    meta.insert("llama.rope.dimension_count".into(), MetaValue::U32(16));
    meta.insert(
        "llama.attention.layer_norm_rms_epsilon".into(),
        MetaValue::F32(1e-5),
    );

    let mut tensors = BTreeMap::new();

    tensors.insert(
        "token_embd.weight".into(),
        (
            vec![vocab, hidden],
            GgufType::F32 as u32,
            f32_bytes(&fill_f32((vocab * hidden) as usize, 1)),
        ),
    );

    let ones = vec![1.0f32; hidden as usize];
    tensors.insert(
        "blk.0.attn_norm.weight".into(),
        (vec![hidden], GgufType::F32 as u32, f32_bytes(&ones)),
    );

    for (name, seed, rows, cols) in [
        ("blk.0.attn_q.weight", 2u32, hidden, hidden),
        ("blk.0.attn_k.weight", 3, hidden, hidden),
        ("blk.0.attn_v.weight", 4, hidden, hidden),
        ("blk.0.attn_output.weight", 5, hidden, hidden),
        ("blk.0.ffn_gate.weight", 6, intermediate, hidden),
        ("blk.0.ffn_up.weight", 7, intermediate, hidden),
        ("blk.0.ffn_down.weight", 8, hidden, intermediate),
        ("output.weight", 9, vocab, hidden),
    ] {
        let w = fill_f32((rows * cols) as usize, seed);
        let (kind, packed) = match name {
            "blk.0.attn_q.weight" => (GgufType::Q4_1, pack_q4_1(&w)?),
            "blk.0.attn_v.weight" => (GgufType::Q8_0, pack_q8_0(&w)?),
            _ => (GgufType::Q4_0, pack_q4_0(&w)?),
        };
        tensors.insert(name.into(), (vec![rows, cols], kind as u32, packed));
    }

    tensors.insert(
        "blk.0.ffn_norm.weight".into(),
        (vec![hidden], GgufType::F32 as u32, f32_bytes(&ones)),
    );
    tensors.insert(
        "output_norm.weight".into(),
        (vec![hidden], GgufType::F32 as u32, f32_bytes(&ones)),
    );

    write_gguf(&tensors, &meta)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_dequant_roundtrip_rough() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let packed = pack_q4_0(&w).unwrap();
        assert_eq!(packed.len(), 2 * Q4_0_TYPE_SIZE);
        let back = dequant_q4_0(&packed, 64).unwrap();
        let mut max_err = 0.0f32;
        for (a, b) in w.iter().zip(back.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        // Q4_0 is lossy; scale is max/7 so error is bounded.
        assert!(max_err < 0.15, "max_err={max_err}");
    }

    #[test]
    fn tiny_gguf_indexes() {
        use dyninfer_checkpoint::{BytesSource, CheckpointContainerReader, InspectionLimits};
        use std::sync::Arc;

        let bytes = tiny_llama_q4_0().unwrap();
        let source =
            Arc::new(BytesSource::new(bytes)) as Arc<dyn dyninfer_checkpoint::RandomAccessSource>;
        let c = crate::GgufContainer;
        assert!(c.probe(source.as_ref()).unwrap().is_match());
        let index = c.index(source, &InspectionLimits::default()).unwrap();
        assert!(index.entries.len() >= 12);
        let q = index
            .entries
            .iter()
            .filter(|e| {
                e.metadata.get("gguf.type_code").and_then(|v| v.as_u64())
                    == Some(GgufType::Q4_0 as u64)
            })
            .count();
        assert!(q >= 6, "expected mixed Q4_0 tensors, got {q}");
        assert!(index.entries.iter().any(|entry| {
            entry
                .metadata
                .get("gguf.type_code")
                .and_then(|value| value.as_u64())
                == Some(GgufType::Q8_0 as u64)
        }));
    }

    #[test]
    fn mixed_gguf_decodes_each_tensor_type_independently() {
        use dyninfer_checkpoint::{
            BuiltinCheckpointSupport, BytesSource, DecodeContext, InspectionLimits,
        };
        use dyninfer_core::PhysicalEncoding;
        use std::sync::Arc;

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "dense.weight".into(),
            (vec![32], GgufType::F32 as u32, vec![0; 32 * 4]),
        );
        tensors.insert(
            "classic.weight".into(),
            (vec![32], GgufType::Q4_0 as u32, vec![0; 18]),
        );
        tensors.insert(
            "k_quant.weight".into(),
            (vec![256], GgufType::Q6_K as u32, vec![0; 210]),
        );
        tensors.insert(
            "ud_iq.weight".into(),
            (vec![256], GgufType::IQ1_S as u32, vec![0; 50]),
        );
        let bytes = write_gguf(&tensors, &BTreeMap::new()).unwrap();
        let source = Arc::new(BytesSource::new(bytes));
        let mut support = BuiltinCheckpointSupport::new();
        crate::register(&mut support);
        let catalog = support
            .inspect_source(
                source,
                &InspectionLimits::default(),
                &DecodeContext::default(),
            )
            .unwrap();
        assert_eq!(catalog.convention_id.as_str(), "gguf.mixed");
        let codecs: BTreeMap<_, _> = catalog
            .parameters
            .iter()
            .map(|parameter| {
                let codec = match &parameter.encoding {
                    PhysicalEncoding::Plain { storage_type, .. } => {
                        format!("plain.{storage_type}")
                    }
                    PhysicalEncoding::BlockQuantized { codec, .. } => codec.to_string(),
                    other => panic!("unexpected encoding: {other:?}"),
                };
                (parameter.canonical_name.to_string(), codec)
            })
            .collect();
        assert_eq!(codecs.get("dense.weight").unwrap(), "plain.f32");
        assert_eq!(codecs.get("classic.weight").unwrap(), "gguf.q4_0");
        assert_eq!(codecs.get("k_quant.weight").unwrap(), "gguf.q6_k");
        assert_eq!(codecs.get("ud_iq.weight").unwrap(), "gguf.iq1_s");
    }
}
