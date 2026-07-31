//! CPU reference forward for the tiny dense Llama fixture (Milestone 1).

use dyninfer_checkpoint_safetensors::fill_f32;
use dyninfer_error::{DynInferError, Result};

const V: usize = 32;
const H: usize = 64;
const I: usize = 128;
const NH: usize = 4;
const D: usize = 16;
const S: usize = 4;
const EPS: f32 = 1e-5;

#[derive(Clone)]
struct Weights {
    emb: Vec<f32>,       // V*H
    attn_norm: Vec<f32>, // H
    wq: Vec<f32>,        // H*H (out*in)
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    ffn_norm: Vec<f32>,
    wgate: Vec<f32>, // I*H
    wup: Vec<f32>,
    wdown: Vec<f32>, // H*I
    out_norm: Vec<f32>,
    wout: Vec<f32>, // V*H
}

impl Weights {
    fn tiny_fixture() -> Self {
        Self {
            emb: fill_f32(V * H, 1),
            attn_norm: vec![1.0; H],
            wq: fill_f32(H * H, 2),
            wk: fill_f32(H * H, 3),
            wv: fill_f32(H * H, 4),
            wo: fill_f32(H * H, 5),
            ffn_norm: vec![1.0; H],
            wgate: fill_f32(I * H, 6),
            wup: fill_f32(I * H, 7),
            wdown: fill_f32(H * I, 8),
            out_norm: vec![1.0; H],
            wout: fill_f32(V * H, 9),
        }
    }
}

/// Reference logits for a padded 4-token window (matches `@prefill`).
pub fn tiny_llama_prefill_logits(tokens: &[u32]) -> Result<Vec<f32>> {
    let mut window = [0u32; S];
    for (i, t) in tokens.iter().take(S).enumerate() {
        window[i] = *t;
    }
    Ok(forward(&Weights::tiny_fixture(), &window))
}

fn forward(w: &Weights, tokens: &[u32; S]) -> Vec<f32> {
    let mut h = vec![0.0f32; S * H];
    for (s, &tok) in tokens.iter().enumerate() {
        let src = &w.emb[tok as usize * H..(tok as usize + 1) * H];
        h[s * H..(s + 1) * H].copy_from_slice(src);
    }
    let h_in = h.clone();
    let xn = rmsnorm_2d(&h, &w.attn_norm, S);
    let q = linear(&xn, &w.wq, S, H, H);
    let k = linear(&xn, &w.wk, S, H, H);
    let v = linear(&xn, &w.wv, S, H, H);
    let ctx = attention(&q, &k, &v);
    let o = linear(&ctx, &w.wo, S, H, H);
    let mut h2 = vec![0.0f32; S * H];
    for i in 0..S * H {
        h2[i] = h_in[i] + o[i];
    }
    let fn_ = rmsnorm_2d(&h2, &w.ffn_norm, S);
    let gate = linear(&fn_, &w.wgate, S, H, I);
    let up = linear(&fn_, &w.wup, S, H, I);
    let mut ff = vec![0.0f32; S * I];
    for i in 0..S * I {
        ff[i] = silu(gate[i]) * up[i];
    }
    let down = linear(&ff, &w.wdown, S, I, H);
    let mut h3 = vec![0.0f32; S * H];
    for i in 0..S * H {
        h3[i] = h2[i] + down[i];
    }
    let last = &h3[(S - 1) * H..S * H];
    let ln = rmsnorm_1d(last, &w.out_norm);
    linear(&ln, &w.wout, 1, H, V)
}

fn linear(x: &[f32], w: &[f32], batch: usize, inn: usize, out: usize) -> Vec<f32> {
    // W is [out, in]; y = x @ W.T
    let mut y = vec![0.0f32; batch * out];
    for b in 0..batch {
        for o in 0..out {
            let mut acc = 0.0;
            for k in 0..inn {
                acc += x[b * inn + k] * w[o * inn + k];
            }
            y[b * out + o] = acc;
        }
    }
    y
}

fn rmsnorm_2d(x: &[f32], w: &[f32], rows: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * H];
    for r in 0..rows {
        let row = &x[r * H..(r + 1) * H];
        let n = rmsnorm_1d(row, w);
        y[r * H..(r + 1) * H].copy_from_slice(&n);
    }
    y
}

fn rmsnorm_1d(x: &[f32], w: &[f32]) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + EPS).sqrt();
    x.iter()
        .zip(w.iter())
        .map(|(a, ww)| a * inv * ww)
        .collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn attention(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let scale = 1.0 / (D as f32).sqrt();
    let mut ctx = vec![0.0f32; S * H];
    for hd in 0..NH {
        let mut scores = vec![0.0f32; S * S];
        for i in 0..S {
            for j in 0..S {
                let mut s = 0.0;
                for t in 0..D {
                    s += q[i * H + hd * D + t] * k[j * H + hd * D + t];
                }
                s *= scale;
                if j > i {
                    s = -1.0e30;
                }
                scores[i * S + j] = s;
            }
            // softmax row
            let row = &mut scores[i * S..(i + 1) * S];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = (*v - m).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        for i in 0..S {
            for t in 0..D {
                let mut acc = 0.0;
                for j in 0..S {
                    acc += scores[i * S + j] * v[j * H + hd * D + t];
                }
                ctx[i * H + hd * D + t] = acc;
            }
        }
    }
    ctx
}

/// Max abs error helper for tests.
pub fn max_abs_err(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(DynInferError::Internal(format!(
            "length mismatch {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max))
}
