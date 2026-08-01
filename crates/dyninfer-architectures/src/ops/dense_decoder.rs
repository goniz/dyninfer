//! Shared dense causal-decoder MLIR emitter (ops library).
//!
//! Architecture files (`models/*`) configure [`DenseDecoderConfig`] and call
//! [`emit_dense_decoder`]. Supports MHA/GQA, independent `head_dim`, optional
//! Q/K RMSNorm, and HuggingFace Llama/Qwen-style (`rotate_half`) RoPE.
//!
//! Weight globals always use the per-tensor dtype from the checkpoint catalog.
//! Activations / logits use [`COMPUTE_DTYPE`] (currently f32); narrower float
//! weights are cast after load.

use crate::ops::kernels;
use dyninfer_architecture::ArchitecturePackage;
use dyninfer_checkpoint::CheckpointCatalog;
use dyninfer_core::{ScalarType, StorageElementType};
use dyninfer_error::Result;
use dyninfer_mlir::{FuncBuilder, ModuleBuilder};
use std::collections::BTreeMap;

/// Default static prefill window for mid-size dense models.
pub const PREFILL_WINDOW: u32 = 64;
/// Smaller window for large models (Qwen3-0.6B) to keep compile/runtime tractable.
pub const LARGE_PREFILL_WINDOW: u32 = 32;
/// Window used by the synthetic Milestone-1 fixture (fast differential tests).
pub const TINY_PREFILL_WINDOW: u32 = 4;

/// Default mutable KV capacity (decode can grow past the prefill window).
pub const PREFILL_MAX_KV: u32 = 128;
pub const LARGE_MAX_KV: u32 = 256;
/// Matches [`TINY_PREFILL_WINDOW`]: decode-at-last fills a dense softmax domain.
/// Larger models pass a host `attn_bias` for padded KV slots.
pub const TINY_MAX_KV: u32 = 4;

/// Activation / logits compute type. Independent of on-disk weight dtypes.
pub const COMPUTE_DTYPE: ScalarType = ScalarType::F32;

#[derive(Debug, Clone, PartialEq)]
pub struct DenseDecoderConfig {
    pub vocab: u32,
    pub hidden: u32,
    pub intermediate: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub num_layers: u32,
    /// Static prefill token window (prompt bucket).
    pub seq: u32,
    /// Mutable KV capacity (`>= seq`); decode positions use `[0, max_kv)`.
    pub max_kv: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: Option<f32>,
    /// Qwen3-style RMSNorm on Q/K heads before RoPE.
    pub has_qk_norm: bool,
    /// Canonical slot name → tensor key in the parameter file (often HF names).
    pub param_keys: BTreeMap<String, String>,
    /// Canonical slot name → on-disk scalar dtype from the checkpoint catalog.
    pub param_dtypes: BTreeMap<String, ScalarType>,
}

impl DenseDecoderConfig {
    pub fn q_dim(&self) -> u32 {
        self.num_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }

    pub fn gqa_group(&self) -> u32 {
        self.num_heads / self.num_kv_heads.max(1)
    }

    pub fn from_package(package: &ArchitecturePackage, catalog: &CheckpointCatalog) -> Self {
        let cfg = &package.resolved_config;
        let meta = &catalog.metadata;
        let u = |keys: &[&str], default: u32| -> u32 {
            for k in keys {
                if let Some(v) = cfg.values.get(*k).and_then(|v| v.as_u64()) {
                    return v as u32;
                }
                if let Some(v) = meta.get(*k).and_then(|v| v.as_u64()) {
                    return v as u32;
                }
            }
            default
        };
        let f = |keys: &[&str], default: f32| -> f32 {
            for k in keys {
                if let Some(v) = cfg.values.get(*k).and_then(|v| v.as_f64()) {
                    return v as f32;
                }
                if let Some(v) = meta.get(*k).and_then(|v| v.as_f64()) {
                    return v as f32;
                }
            }
            default
        };
        let hidden = u(&["hidden_size", "llama.embedding_length"], 64);
        let num_heads = u(&["num_heads", "llama.attention.head_count"], 4);
        let num_kv_heads = u(
            &["num_kv_heads", "llama.attention.head_count_kv"],
            num_heads,
        );
        let vocab = u(&["vocab_size", "llama.vocab_size"], 32);
        let num_layers = u(
            &["num_layers", "n_layer", "block_count", "llama.block_count"],
            1,
        );
        let head_dim = u(&["head_dim"], hidden / num_heads.max(1));
        let is_tiny = vocab == 32 && hidden == 64 && num_heads == 4;
        let is_large = vocab > 50_000 || num_layers > 16 || hidden >= 1024;
        let (seq, max_kv) = if is_tiny {
            (TINY_PREFILL_WINDOW, TINY_MAX_KV)
        } else if is_large {
            (LARGE_PREFILL_WINDOW, LARGE_MAX_KV)
        } else {
            (PREFILL_WINDOW, PREFILL_MAX_KV)
        };
        let rope_theta = meta
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .or_else(|| cfg.values.get("rope_theta").and_then(|v| v.as_f64()))
            .map(|v| v as f32);
        let has_qk_norm = catalog.parameters.iter().any(|p| {
            p.canonical_name
                .as_str()
                .contains("attn_q_norm.weight")
        });
        let mut param_keys = BTreeMap::new();
        let mut param_dtypes = BTreeMap::new();
        for p in &catalog.parameters {
            let key = p
                .components
                .first()
                .map(|c| c.key.clone())
                .unwrap_or_else(|| p.canonical_name.to_string());
            param_keys.insert(p.canonical_name.to_string(), key);
            if let Some(ty) = catalog_param_dtype(p) {
                param_dtypes.insert(p.canonical_name.to_string(), ty);
            }
        }
        Self {
            vocab,
            hidden,
            intermediate: u(&["intermediate_size"], hidden * 2),
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            seq,
            max_kv: max_kv.max(seq),
            rms_norm_eps: f(&["rms_norm_eps"], 1e-5),
            rope_theta: if is_tiny {
                None
            } else {
                rope_theta.or(Some(10000.0))
            },
            has_qk_norm,
            param_keys,
            param_dtypes,
        }
    }

    fn param_key(&self, canonical: &str) -> String {
        self.param_keys
            .get(canonical)
            .cloned()
            .unwrap_or_else(|| canonical.to_string())
    }

    /// On-disk dtype for a parameter. Always sourced from the checkpoint catalog
    /// when present; synthetic fixtures without catalog entries default to f32.
    fn param_dtype(&self, canonical: &str) -> ScalarType {
        self.param_dtypes
            .get(canonical)
            .copied()
            .unwrap_or(ScalarType::F32)
    }

    pub fn is_tiny_m1(&self) -> bool {
        self.vocab == 32
            && self.hidden == 64
            && self.intermediate == 128
            && self.num_heads == 4
            && self.head_dim == 16
            && self.num_layers == 1
            && self.seq == TINY_PREFILL_WINDOW
            && self.rope_theta.is_none()
    }

    pub fn supports_dense_emit(&self) -> bool {
        self.num_heads > 0
            && self.num_kv_heads > 0
            && self.num_heads % self.num_kv_heads == 0
            && self.head_dim > 0
            && self.head_dim % 2 == 0
            && self.q_dim() > 0
            && self.kv_dim() > 0
            && self.intermediate > 0
            && self.num_layers >= 1
            && self.num_layers <= 32
            && self.vocab >= 1
            && self.vocab <= 160_000
            && self.hidden <= 2048
            && self.head_dim <= 256
            && self.q_dim() <= 4096
            && (self.seq == 4 || self.seq == 32 || self.seq == 64)
            && self.max_kv >= self.seq
            && self.max_kv <= 512
    }
}

/// Emit using an explicit config (architecture files may override flags).
///
/// Builds an in-memory MLIR module via [`ModuleBuilder`], verifies, then prints
/// for the IREE compile boundary (spec §8.3.1).
pub fn emit_dense_decoder_cfg(arch_id: &str, c: &DenseDecoderConfig) -> Result<String> {
    assert!(
        c.supports_dense_emit(),
        "unsupported dense decoder emit config: {c:?}"
    );
    let mut builder = ModuleBuilder::new()?;
    build_dense_decoder(&mut builder, arch_id, c)?;
    Ok(builder.finish()?.mlir_text)
}

/// Build the dense decoder into an existing [`ModuleBuilder`].
pub fn build_dense_decoder(
    builder: &mut ModuleBuilder,
    arch_id: &str,
    c: &DenseDecoderConfig,
) -> Result<()> {
    let _ = arch_id; // retained for call-site tracing / future module attrs
    emit_globals(builder, c)?;
    emit_helpers(builder, c)?;
    emit_prefill(builder, c)?;
    emit_decode(builder, c)?;
    kernels::emit_add_smoke(builder)?;
    Ok(())
}

fn catalog_param_dtype(param: &dyninfer_checkpoint::LogicalParameter) -> Option<ScalarType> {
    let comp = param.components.first()?;
    match &comp.storage_type {
        StorageElementType::Scalar { ty } => Some(*ty),
        _ => None,
    }
}

fn mlir_ty(ty: ScalarType) -> String {
    ty.to_string()
}

fn emit_global(
    builder: &mut ModuleBuilder,
    c: &DenseDecoderConfig,
    sym: &str,
    canonical: &str,
    shape: &str,
) -> Result<()> {
    let key = c.param_key(canonical);
    let wt = mlir_ty(c.param_dtype(canonical));
    builder.util_global_parameter(sym, &key, &format!("tensor<{shape}x{wt}>"))
}

/// Load a weight global and cast to [`COMPUTE_DTYPE`] when the checkpoint dtype differs.
fn emit_load_compute(
    f: &mut FuncBuilder,
    c: &DenseDecoderConfig,
    ssa: &str,
    sym: &str,
    canonical: &str,
    shape: &str,
) {
    let storage = c.param_dtype(canonical);
    let wt = mlir_ty(storage);
    let ct = mlir_ty(COMPUTE_DTYPE);
    match storage {
        ScalarType::F16 | ScalarType::Bf16 if COMPUTE_DTYPE == ScalarType::F32 => {
            kernels::load_compute(f, ssa, sym, &wt, &ct, shape);
        }
        ScalarType::F32 if COMPUTE_DTYPE == ScalarType::F32 => {
            kernels::load_compute(f, ssa, sym, &wt, &ct, shape);
        }
        other => panic!(
            "dense decoder: cannot cast checkpoint dtype {other} → compute {COMPUTE_DTYPE} for {canonical}"
        ),
    }
}


fn emit_globals(builder: &mut ModuleBuilder, c: &DenseDecoderConfig) -> Result<()> {
    let (v, h, i) = (c.vocab, c.hidden, c.intermediate);
    let (q, kv, d) = (c.q_dim(), c.kv_dim(), c.head_dim);
    emit_global(
        builder,
        c,
        "token_embd_weight",
        "token_embd.weight",
        &format!("{v}x{h}"),
    )?;
    for layer in 0..c.num_layers {
        let p = format!("blk{layer}");
        let n = format!("blk.{layer}");
        emit_global(
            builder,
            c,
            &format!("{p}_attn_norm_weight"),
            &format!("{n}.attn_norm.weight"),
            &format!("{h}"),
        )?;
        emit_global(
            builder,
            c,
            &format!("{p}_attn_q_weight"),
            &format!("{n}.attn_q.weight"),
            &format!("{q}x{h}"),
        )?;
        emit_global(
            builder,
            c,
            &format!("{p}_attn_k_weight"),
            &format!("{n}.attn_k.weight"),
            &format!("{kv}x{h}"),
        )?;
        emit_global(
            builder,
            c,
            &format!("{p}_attn_v_weight"),
            &format!("{n}.attn_v.weight"),
            &format!("{kv}x{h}"),
        )?;
        emit_global(
            builder,
            c,
            &format!("{p}_attn_output_weight"),
            &format!("{n}.attn_output.weight"),
            &format!("{h}x{q}"),
        )?;
        if c.has_qk_norm {
            emit_global(
                builder,
                c,
                &format!("{p}_attn_q_norm_weight"),
                &format!("{n}.attn_q_norm.weight"),
                &format!("{d}"),
            )?;
            emit_global(
                builder,
                c,
                &format!("{p}_attn_k_norm_weight"),
                &format!("{n}.attn_k_norm.weight"),
                &format!("{d}"),
            )?;
        }
        emit_global(
            builder,
            c,
            &format!("{p}_ffn_norm_weight"),
            &format!("{n}.ffn_norm.weight"),
            &format!("{h}"),
        )?;
        emit_global(
            builder,
            c,
            &format!("{p}_ffn_gate_weight"),
            &format!("{n}.ffn_gate.weight"),
            &format!("{i}x{h}"),
        )?;
        emit_global(
            builder,
            c,
            &format!("{p}_ffn_up_weight"),
            &format!("{n}.ffn_up.weight"),
            &format!("{i}x{h}"),
        )?;
        emit_global(
            builder,
            c,
            &format!("{p}_ffn_down_weight"),
            &format!("{n}.ffn_down.weight"),
            &format!("{h}x{i}"),
        )?;
    }
    emit_global(
        builder,
        c,
        "output_norm_weight",
        "output_norm.weight",
        &format!("{h}"),
    )?;
    emit_global(
        builder,
        c,
        "output_weight",
        "output.weight",
        &format!("{v}x{h}"),
    )?;
    // Mutable KV cache (f32 compute). Prefill seeds [0, seq); decode grows to max_kv.
    let (mk, nkv, d) = (c.max_kv, c.num_kv_heads, c.head_dim);
    for layer in 0..c.num_layers {
        builder.util_global_mutable_zero(
            &format!("kv_k{layer}"),
            &format!("tensor<{mk}x{nkv}x{d}xf32>"),
        )?;
        builder.util_global_mutable_zero(
            &format!("kv_v{layer}"),
            &format!("tensor<{mk}x{nkv}x{d}xf32>"),
        )?;
    }
    Ok(())
}

fn emit_helpers(module: &mut ModuleBuilder, c: &DenseDecoderConfig) -> Result<()> {
    let (s, h, i, nh, nkv, d) = (
        c.seq,
        c.hidden,
        c.intermediate,
        c.num_heads,
        c.num_kv_heads,
        c.head_dim,
    );
    let (q, kv) = (c.q_dim(), c.kv_dim());
    let g = c.gqa_group();

    kernels::emit_rms_norm(module, "rms_norm", s, h, c.rms_norm_eps)?;
    kernels::emit_linear(module, "linear_hq", s, h, q, true)?;
    kernels::emit_linear(module, "linear_hkv", s, h, kv, true)?;
    kernels::emit_linear(module, "linear_qh", s, q, h, true)?;
    kernels::emit_linear(module, "linear_hi", s, h, i, true)?;
    kernels::emit_linear(module, "linear_ih", s, i, h, true)?;

    if c.has_qk_norm {
        kernels::emit_rms_norm_heads(module, "rms_norm_q_heads", s, nh, d, c.rms_norm_eps)?;
        if nkv != nh {
            kernels::emit_rms_norm_heads(module, "rms_norm_kv_heads", s, nkv, d, c.rms_norm_eps)?;
        }
    }

    if nkv != nh {
        kernels::emit_repeat_kv(module, "repeat_kv", s, nkv, nh, d, g)?;
    }

    if let Some(theta) = c.rope_theta {
        kernels::emit_rope(module, "apply_rope_q", s, nh, d, theta)?;
        if nkv != nh {
            kernels::emit_rope(module, "apply_rope_kv", s, nkv, d, theta)?;
        }
    }

    emit_attn(module, c)?;
    emit_decode_helpers(module, c)?;
    Ok(())
}

fn emit_attn(module: &mut ModuleBuilder, c: &DenseDecoderConfig) -> Result<()> {
    let (s, nh, nkv, d) = (c.seq, c.num_heads, c.num_kv_heads, c.head_dim);
    let (q, kv) = (c.q_dim(), c.kv_dim());
    let scale = 1.0 / (d as f32).sqrt();
    // Returns (context, k_post_rope [S×nkv×d], v [S×nkv×d]) so callers can seed KV cache.
    let mut f = module.func_private("attn");
    f.arg("q", format!("tensor<{s}x{q}xf32>"));
    f.arg("k", format!("tensor<{s}x{kv}xf32>"));
    f.arg("v", format!("tensor<{s}x{kv}xf32>"));
    if c.has_qk_norm {
        f.arg("q_norm", format!("tensor<{d}xf32>"));
        f.arg("k_norm", format!("tensor<{d}xf32>"));
    }
    f.result_ty(format!("tensor<{s}x{q}xf32>"));
    f.result_ty(format!("tensor<{s}x{nkv}x{d}xf32>"));
    f.result_ty(format!("tensor<{s}x{nkv}x{d}xf32>"));
    f.op_asm(format!(
        r#"  %q3 = tensor.expand_shape %q [[0], [1, 2]] output_shape [{s}, {nh}, {d}] : tensor<{s}x{q}xf32> into tensor<{s}x{nh}x{d}xf32>
  %k3 = tensor.expand_shape %k [[0], [1, 2]] output_shape [{s}, {nkv}, {d}] : tensor<{s}x{kv}xf32> into tensor<{s}x{nkv}x{d}xf32>
  %v3 = tensor.expand_shape %v [[0], [1, 2]] output_shape [{s}, {nkv}, {d}] : tensor<{s}x{kv}xf32> into tensor<{s}x{nkv}x{d}xf32>
"#,
        s = s,
        q = q,
        kv = kv,
        nh = nh,
        nkv = nkv,
        d = d,
    ));

    let q_heads = if c.has_qk_norm {
        f.op_asm(format!(
            "  %qn = func.call @rms_norm_q_heads(%q3, %q_norm) : (tensor<{s}x{nh}x{d}xf32>, tensor<{d}xf32>) -> tensor<{s}x{nh}x{d}xf32>\n",
            s = s, nh = nh, d = d,
        ));
        "qn"
    } else {
        "q3"
    };
    let k_heads = if c.has_qk_norm {
        let kn_fn = if nkv != nh {
            "rms_norm_kv_heads"
        } else {
            "rms_norm_q_heads"
        };
        f.op_asm(format!(
            "  %kn = func.call @{kn_fn}(%k3, %k_norm) : (tensor<{s}x{nkv}x{d}xf32>, tensor<{d}xf32>) -> tensor<{s}x{nkv}x{d}xf32>\n",
            kn_fn = kn_fn,
            s = s,
            nkv = nkv,
            d = d,
        ));
        "kn"
    } else {
        "k3"
    };

    let (q_ssa, k_ssa) = if c.rope_theta.is_some() {
        f.op_asm(format!(
            r#"  %qr = func.call @apply_rope_q(%{q_heads}) : (tensor<{s}x{nh}x{d}xf32>) -> tensor<{s}x{nh}x{d}xf32>
"#,
            q_heads = q_heads,
            s = s,
            nh = nh,
            d = d,
        ));
        let rope_kv = if nkv != nh {
            "apply_rope_kv"
        } else {
            "apply_rope_q"
        };
        f.op_asm(format!(
            r#"  %kr = func.call @{rope_kv}(%{k_heads}) : (tensor<{s}x{nkv}x{d}xf32>) -> tensor<{s}x{nkv}x{d}xf32>
"#,
            rope_kv = rope_kv,
            k_heads = k_heads,
            s = s,
            nkv = nkv,
            d = d,
        ));
        ("qr", "kr")
    } else {
        (q_heads, k_heads)
    };

    let k_full = if nkv != nh {
        f.op_asm(format!(
            "  %krep = func.call @repeat_kv(%{k_ssa}) : (tensor<{s}x{nkv}x{d}xf32>) -> tensor<{s}x{nh}x{d}xf32>\n",
            k_ssa = k_ssa,
            s = s,
            nkv = nkv,
            nh = nh,
            d = d,
        ));
        f.op_asm(format!(
            "  %vrep = func.call @repeat_kv(%v3) : (tensor<{s}x{nkv}x{d}xf32>) -> tensor<{s}x{nh}x{d}xf32>\n",
            s = s,
            nkv = nkv,
            nh = nh,
            d = d,
        ));
        ("krep", "vrep")
    } else {
        (k_ssa, "v3")
    };

    f.op_asm(format!(
        r#"  %q_ti = tensor.empty() : tensor<{nh}x{s}x{d}xf32>
  %k_ti = tensor.empty() : tensor<{nh}x{s}x{d}xf32>
  %v_ti = tensor.empty() : tensor<{nh}x{s}x{d}xf32>
  %qb = linalg.transpose ins(%{q_ssa} : tensor<{s}x{nh}x{d}xf32>) outs(%q_ti : tensor<{nh}x{s}x{d}xf32>) permutation = [1, 0, 2]
  %kb = linalg.transpose ins(%{k_full} : tensor<{s}x{nh}x{d}xf32>) outs(%k_ti : tensor<{nh}x{s}x{d}xf32>) permutation = [1, 0, 2]
  %vb = linalg.transpose ins(%{v_full} : tensor<{s}x{nh}x{d}xf32>) outs(%v_ti : tensor<{nh}x{s}x{d}xf32>) permutation = [1, 0, 2]
  %kt_i = tensor.empty() : tensor<{nh}x{d}x{s}xf32>
  %kt = linalg.transpose ins(%kb : tensor<{nh}x{s}x{d}xf32>) outs(%kt_i : tensor<{nh}x{d}x{s}xf32>) permutation = [0, 2, 1]
  %c0 = arith.constant 0.0 : f32
  %neg = arith.constant -1.0e+7 : f32
  %scale = arith.constant {scale:.8e} : f32
  %sc_i = tensor.empty() : tensor<{nh}x{s}x{s}xf32>
  %sc_z = linalg.fill ins(%c0 : f32) outs(%sc_i : tensor<{nh}x{s}x{s}xf32>) -> tensor<{nh}x{s}x{s}xf32>
  %scores = linalg.batch_matmul ins(%qb, %kt : tensor<{nh}x{s}x{d}xf32>, tensor<{nh}x{d}x{s}xf32>) outs(%sc_z : tensor<{nh}x{s}x{s}xf32>) -> tensor<{nh}x{s}x{s}xf32>
  %scores_s = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%scores : tensor<{nh}x{s}x{s}xf32>) outs(%scores : tensor<{nh}x{s}x{s}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %m = arith.mulf %a, %scale : f32
      linalg.yield %m : f32
  }} -> tensor<{nh}x{s}x{s}xf32>
  %masked = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%scores_s : tensor<{nh}x{s}x{s}xf32>) outs(%scores_s : tensor<{nh}x{s}x{s}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %ii = linalg.index 1 : index
      %jj = linalg.index 2 : index
      %cmp = arith.cmpi sgt, %jj, %ii : index
      %sel = arith.select %cmp, %neg, %a : f32
      linalg.yield %sel : f32
  }} -> tensor<{nh}x{s}x{s}xf32>
  %sm_i = tensor.empty() : tensor<{nh}x{s}x{s}xf32>
  %attn = linalg.softmax dimension(2) ins(%masked : tensor<{nh}x{s}x{s}xf32>) outs(%sm_i : tensor<{nh}x{s}x{s}xf32>) -> tensor<{nh}x{s}x{s}xf32>
  %ctx_i = tensor.empty() : tensor<{nh}x{s}x{d}xf32>
  %ctx_z = linalg.fill ins(%c0 : f32) outs(%ctx_i : tensor<{nh}x{s}x{d}xf32>) -> tensor<{nh}x{s}x{d}xf32>
  %ctx_b = linalg.batch_matmul ins(%attn, %vb : tensor<{nh}x{s}x{s}xf32>, tensor<{nh}x{s}x{d}xf32>) outs(%ctx_z : tensor<{nh}x{s}x{d}xf32>) -> tensor<{nh}x{s}x{d}xf32>
  %ctx_t_i = tensor.empty() : tensor<{s}x{nh}x{d}xf32>
  %ctx_t = linalg.transpose ins(%ctx_b : tensor<{nh}x{s}x{d}xf32>) outs(%ctx_t_i : tensor<{s}x{nh}x{d}xf32>) permutation = [1, 0, 2]
  %ctx = tensor.collapse_shape %ctx_t [[0], [1, 2]] : tensor<{s}x{nh}x{d}xf32> into tensor<{s}x{q}xf32>
  return %ctx, %{k_ssa}, %v3 : tensor<{s}x{q}xf32>, tensor<{s}x{nkv}x{d}xf32>, tensor<{s}x{nkv}x{d}xf32>
"#,
        s = s,
        q = q,
        nh = nh,
        nkv = nkv,
        d = d,
        scale = scale,
        q_ssa = q_ssa,
        k_ssa = k_ssa,
        k_full = k_full.0,
        v_full = k_full.1,
    ));

    f.finish(module)
}

fn emit_decode_helpers(module: &mut ModuleBuilder, c: &DenseDecoderConfig) -> Result<()> {
    let (mk, h, i, nh, nkv, d) = (
        c.max_kv,
        c.hidden,
        c.intermediate,
        c.num_heads,
        c.num_kv_heads,
        c.head_dim,
    );
    let (q, kv) = (c.q_dim(), c.kv_dim());
    let g = c.gqa_group();
    let scale = 1.0 / (d as f32).sqrt();

    kernels::emit_rms_norm(module, "rms_norm_tok", 1, h, c.rms_norm_eps)?;
    kernels::emit_linear(module, "linear_hq_tok", 1, h, q, true)?;
    kernels::emit_linear(module, "linear_hkv_tok", 1, h, kv, true)?;
    kernels::emit_linear(module, "linear_qh_tok", 1, q, h, true)?;
    kernels::emit_linear(module, "linear_hi_tok", 1, h, i, true)?;
    kernels::emit_linear(module, "linear_ih_tok", 1, i, h, true)?;

    if c.has_qk_norm {
        kernels::emit_rms_norm_heads(module, "rms_norm_q_heads_tok", 1, nh, d, c.rms_norm_eps)?;
        if nkv != nh {
            kernels::emit_rms_norm_heads(module, "rms_norm_kv_heads_tok", 1, nkv, d, c.rms_norm_eps)?;
        }
    }

    if nkv != nh {
        kernels::emit_repeat_kv(module, "repeat_kv_mk", mk, nkv, nh, d, g)?;
    }

    if let Some(theta) = c.rope_theta {
        kernels::emit_rope_at(module, "apply_rope_q_at", mk, nh, d, theta)?;
        if nkv != nh {
            kernels::emit_rope_at(module, "apply_rope_kv_at", mk, nkv, d, theta)?;
        }
    }

    {
        let mut f = module.func_private("kv_update_row");
        f.arg("cache", format!("tensor<{mk}x{nkv}x{d}xf32>"));
        f.arg("row", format!("tensor<1x{nkv}x{d}xf32>"));
        f.arg("pos_t", "tensor<i64>");
        f.result_ty(format!("tensor<{mk}x{nkv}x{d}xf32>"));
        f.op_asm(format!(
            r#"  %pos_i64 = tensor.extract %pos_t[] : tensor<i64>
  %pos = arith.index_cast %pos_i64 : i64 to index
  %c0 = arith.constant 0 : index
  %y = tensor.insert_slice %row into %cache[%pos, %c0, %c0] [1, {nkv}, {d}] [1, 1, 1] : tensor<1x{nkv}x{d}xf32> into tensor<{mk}x{nkv}x{d}xf32>
  return %y : tensor<{mk}x{nkv}x{d}xf32>"#,
            mk = mk,
            nkv = nkv,
            d = d,
        ));
        f.finish(module)?;
    }

    let mut f = module.func_private("attn_decode");
    f.arg("q", format!("tensor<1x{q}xf32>"));
    f.arg("k_cache", format!("tensor<{mk}x{nkv}x{d}xf32>"));
    f.arg("v_cache", format!("tensor<{mk}x{nkv}x{d}xf32>"));
    f.arg("k_new", format!("tensor<1x{nkv}x{d}xf32>"));
    f.arg("v_new", format!("tensor<1x{nkv}x{d}xf32>"));
    f.arg("pos_t", "tensor<i64>");
    f.arg("attn_bias", format!("tensor<{mk}xf32>"));
    if c.has_qk_norm {
        f.arg("q_norm", format!("tensor<{d}xf32>"));
        f.arg("k_norm", format!("tensor<{d}xf32>"));
    }
    f.result_ty(format!("tensor<1x{q}xf32>"));
    f.result_ty(format!("tensor<{mk}x{nkv}x{d}xf32>"));
    f.result_ty(format!("tensor<{mk}x{nkv}x{d}xf32>"));
    f.op_asm(format!(
        r#"  %pos_i64 = tensor.extract %pos_t[] : tensor<i64>
  %pos = arith.index_cast %pos_i64 : i64 to index
  %q3 = tensor.expand_shape %q [[0], [1, 2]] output_shape [1, {nh}, {d}] : tensor<1x{q}xf32> into tensor<1x{nh}x{d}xf32>
"#,
        q = q,
        d = d,
        nh = nh,
    ));

    let q_heads = if c.has_qk_norm {
        f.op_asm(format!(
            "  %qn = func.call @rms_norm_q_heads_tok(%q3, %q_norm) : (tensor<1x{nh}x{d}xf32>, tensor<{d}xf32>) -> tensor<1x{nh}x{d}xf32>\n",
            nh = nh,
            d = d,
        ));
        "qn"
    } else {
        "q3"
    };
    let k_heads = if c.has_qk_norm {
        let kn_fn = if nkv != nh {
            "rms_norm_kv_heads_tok"
        } else {
            "rms_norm_q_heads_tok"
        };
        f.op_asm(format!(
            "  %kn = func.call @{kn_fn}(%k_new, %k_norm) : (tensor<1x{nkv}x{d}xf32>, tensor<{d}xf32>) -> tensor<1x{nkv}x{d}xf32>\n",
            kn_fn = kn_fn,
            nkv = nkv,
            d = d,
        ));
        "kn"
    } else {
        "k_new"
    };

    let (q_ssa, k_ssa) = if c.rope_theta.is_some() {
        f.op_asm(format!(
            "  %qr = func.call @apply_rope_q_at(%{q_heads}, %pos) : (tensor<1x{nh}x{d}xf32>, index) -> tensor<1x{nh}x{d}xf32>\n",
            q_heads = q_heads,
            nh = nh,
            d = d,
        ));
        let rope_kv = if nkv != nh {
            "apply_rope_kv_at"
        } else {
            "apply_rope_q_at"
        };
        f.op_asm(format!(
            "  %kr = func.call @{rope_kv}(%{k_heads}, %pos) : (tensor<1x{nkv}x{d}xf32>, index) -> tensor<1x{nkv}x{d}xf32>\n",
            rope_kv = rope_kv,
            k_heads = k_heads,
            nkv = nkv,
            d = d,
        ));
        ("qr", "kr")
    } else {
        (q_heads, k_heads)
    };

    f.op_asm(format!(
        r#"  %k_upd = func.call @kv_update_row(%k_cache, %{k_ssa}, %pos_t) : (tensor<{mk}x{nkv}x{d}xf32>, tensor<1x{nkv}x{d}xf32>, tensor<i64>) -> tensor<{mk}x{nkv}x{d}xf32>
  %v_upd = func.call @kv_update_row(%v_cache, %v_new, %pos_t) : (tensor<{mk}x{nkv}x{d}xf32>, tensor<1x{nkv}x{d}xf32>, tensor<i64>) -> tensor<{mk}x{nkv}x{d}xf32>
"#,
        k_ssa = k_ssa,
        nkv = nkv,
        d = d,
        mk = mk,
    ));

    let (k_full, v_full) = if nkv != nh {
        f.op_asm(format!(
            "  %krep = func.call @repeat_kv_mk(%k_upd) : (tensor<{mk}x{nkv}x{d}xf32>) -> tensor<{mk}x{nh}x{d}xf32>\n",
            mk = mk,
            nkv = nkv,
            nh = nh,
            d = d,
        ));
        f.op_asm(format!(
            "  %vrep = func.call @repeat_kv_mk(%v_upd) : (tensor<{mk}x{nkv}x{d}xf32>) -> tensor<{mk}x{nh}x{d}xf32>\n",
            mk = mk,
            nkv = nkv,
            nh = nh,
            d = d,
        ));
        ("krep", "vrep")
    } else {
        ("k_upd", "v_upd")
    };

    f.op_asm(format!(
        r#"  %q_ti = tensor.empty() : tensor<{nh}x1x{d}xf32>
  %k_ti = tensor.empty() : tensor<{nh}x{mk}x{d}xf32>
  %v_ti = tensor.empty() : tensor<{nh}x{mk}x{d}xf32>
  %qb = linalg.transpose ins(%{q_ssa} : tensor<1x{nh}x{d}xf32>) outs(%q_ti : tensor<{nh}x1x{d}xf32>) permutation = [1, 0, 2]
  %kb = linalg.transpose ins(%{k_full} : tensor<{mk}x{nh}x{d}xf32>) outs(%k_ti : tensor<{nh}x{mk}x{d}xf32>) permutation = [1, 0, 2]
  %vb = linalg.transpose ins(%{v_full} : tensor<{mk}x{nh}x{d}xf32>) outs(%v_ti : tensor<{nh}x{mk}x{d}xf32>) permutation = [1, 0, 2]
  %kt_i = tensor.empty() : tensor<{nh}x{d}x{mk}xf32>
  %kt = linalg.transpose ins(%kb : tensor<{nh}x{mk}x{d}xf32>) outs(%kt_i : tensor<{nh}x{d}x{mk}xf32>) permutation = [0, 2, 1]
  %c0 = arith.constant 0.0 : f32
  %scale = arith.constant {scale:.8e} : f32
  %sc_i = tensor.empty() : tensor<{nh}x1x{mk}xf32>
  %sc_z = linalg.fill ins(%c0 : f32) outs(%sc_i : tensor<{nh}x1x{mk}xf32>) -> tensor<{nh}x1x{mk}xf32>
  %scores = linalg.batch_matmul ins(%qb, %kt : tensor<{nh}x1x{d}xf32>, tensor<{nh}x{d}x{mk}xf32>) outs(%sc_z : tensor<{nh}x1x{mk}xf32>) -> tensor<{nh}x1x{mk}xf32>
  %scores_s = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%scores : tensor<{nh}x1x{mk}xf32>) outs(%scores : tensor<{nh}x1x{mk}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %m = arith.mulf %a, %scale : f32
      linalg.yield %m : f32
  }} -> tensor<{nh}x1x{mk}xf32>
  %bias_i = tensor.empty() : tensor<{nh}x1x{mk}xf32>
  %bias_3d = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%attn_bias : tensor<{mk}xf32>) outs(%bias_i : tensor<{nh}x1x{mk}xf32>) {{
    ^bb0(%b: f32, %o: f32):
      linalg.yield %b : f32
  }} -> tensor<{nh}x1x{mk}xf32>
  %masked_i = tensor.empty() : tensor<{nh}x1x{mk}xf32>
  %masked = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%scores_s, %bias_3d : tensor<{nh}x1x{mk}xf32>, tensor<{nh}x1x{mk}xf32>) outs(%masked_i : tensor<{nh}x1x{mk}xf32>) {{
    ^bb0(%a: f32, %b: f32, %o: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  }} -> tensor<{nh}x1x{mk}xf32>
  %sm_i = tensor.empty() : tensor<{nh}x1x{mk}xf32>
  %attn = linalg.softmax dimension(2) ins(%masked : tensor<{nh}x1x{mk}xf32>) outs(%sm_i : tensor<{nh}x1x{mk}xf32>) -> tensor<{nh}x1x{mk}xf32>
  %ctx_i = tensor.empty() : tensor<{nh}x1x{d}xf32>
  %ctx_z = linalg.fill ins(%c0 : f32) outs(%ctx_i : tensor<{nh}x1x{d}xf32>) -> tensor<{nh}x1x{d}xf32>
  %ctx_b = linalg.batch_matmul ins(%attn, %vb : tensor<{nh}x1x{mk}xf32>, tensor<{nh}x{mk}x{d}xf32>) outs(%ctx_z : tensor<{nh}x1x{d}xf32>) -> tensor<{nh}x1x{d}xf32>
  %ctx_t_i = tensor.empty() : tensor<1x{nh}x{d}xf32>
  %ctx_t = linalg.transpose ins(%ctx_b : tensor<{nh}x1x{d}xf32>) outs(%ctx_t_i : tensor<1x{nh}x{d}xf32>) permutation = [1, 0, 2]
  %ctx = tensor.collapse_shape %ctx_t [[0], [1, 2]] : tensor<1x{nh}x{d}xf32> into tensor<1x{q}xf32>
  return %ctx, %k_upd, %v_upd : tensor<1x{q}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<{mk}x{nkv}x{d}xf32>
"#,
        q = q,
        nh = nh,
        nkv = nkv,
        d = d,
        mk = mk,
        scale = scale,
        q_ssa = q_ssa,
        k_full = k_full,
        v_full = v_full,
    ));

    f.finish(module)
}

fn emit_prefill(module: &mut ModuleBuilder, c: &DenseDecoderConfig) -> Result<()> {

    let (s, h, v) = (c.seq, c.hidden, c.vocab);
    let (q, kv) = (c.q_dim(), c.kv_dim());
    let d = c.head_dim;
    // Tokens are left-aligned (right-padded). `%last` is the index of the
    // newest real token so RoPE/causal attention ignore trailing pads.
    let mut f = module.func("prefill");
    f.arg("tokens", format!("tensor<{s}xi64>"));
    f.arg("last", "tensor<i64>");
    f.result_ty(format!("tensor<{v}xf32>"));
    emit_load_compute(
        &mut f,
        c,
        "emb_t",
        "token_embd_weight",
        "token_embd.weight",
        &format!("{v}x{h}"),
    );

    // Embedding gather (unrolled).
    f.op_asm(format!(
        "  %h_acc0 = tensor.empty() : tensor<{s}x{h}xf32>\n"
    ));
    for pos in 0..s {
        f.op_asm(format!("  %c{pos}i = arith.constant {pos} : index\n"));
        f.op_asm(format!(
            "  %t{pos} = tensor.extract %tokens[%c{pos}i] : tensor<{s}xi64>\n"
        ));
        f.op_asm(format!(
            "  %i{pos} = arith.index_cast %t{pos} : i64 to index\n"
        ));
        f.op_asm(format!(
            "  %r{pos} = tensor.extract_slice %emb_t[%i{pos}, 0] [1, {h}] [1, 1] : tensor<{v}x{h}xf32> to tensor<1x{h}xf32>\n"
        ));
        let prev = if pos == 0 {
            "h_acc0".to_string()
        } else {
            format!("h_acc{pos}")
        };
        let next = format!("h_acc{}", pos + 1);
        f.op_asm(format!(
            "  %{next} = tensor.insert_slice %r{pos} into %{prev}[{pos}, 0] [1, {h}] [1, 1] : tensor<1x{h}xf32> into tensor<{s}x{h}xf32>\n"
        ));
    }
    let mut h_name = format!("h_acc{s}");

    for layer in 0..c.num_layers {
        let p = format!("blk{layer}");
        let n = format!("blk.{layer}");
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_attn_nw"),
            &format!("{p}_attn_norm_weight"),
            &format!("{n}.attn_norm.weight"),
            &format!("{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wq"),
            &format!("{p}_attn_q_weight"),
            &format!("{n}.attn_q.weight"),
            &format!("{q}x{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wk"),
            &format!("{p}_attn_k_weight"),
            &format!("{n}.attn_k.weight"),
            &format!("{kv}x{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wv"),
            &format!("{p}_attn_v_weight"),
            &format!("{n}.attn_v.weight"),
            &format!("{kv}x{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wo"),
            &format!("{p}_attn_output_weight"),
            &format!("{n}.attn_output.weight"),
            &format!("{h}x{q}"),
        );
        if c.has_qk_norm {
            emit_load_compute(
                &mut f,
                c,
                &format!("{p}_qnw"),
                &format!("{p}_attn_q_norm_weight"),
                &format!("{n}.attn_q_norm.weight"),
                &format!("{d}"),
            );
            emit_load_compute(
                &mut f,
                c,
                &format!("{p}_knw"),
                &format!("{p}_attn_k_norm_weight"),
                &format!("{n}.attn_k_norm.weight"),
                &format!("{d}"),
            );
        }
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_ffn_nw"),
            &format!("{p}_ffn_norm_weight"),
            &format!("{n}.ffn_norm.weight"),
            &format!("{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wgate"),
            &format!("{p}_ffn_gate_weight"),
            &format!("{n}.ffn_gate.weight"),
            &format!("{}x{h}", c.intermediate),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wup"),
            &format!("{p}_ffn_up_weight"),
            &format!("{n}.ffn_up.weight"),
            &format!("{}x{h}", c.intermediate),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wdown"),
            &format!("{p}_ffn_down_weight"),
            &format!("{n}.ffn_down.weight"),
            &format!("{h}x{}", c.intermediate),
        );

        let xin = h_name.clone();
        f.op_asm(format!(
            "  %{p}_xn = func.call @rms_norm(%{xin}, %{p}_attn_nw) : (tensor<{s}x{h}xf32>, tensor<{h}xf32>) -> tensor<{s}x{h}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_q = func.call @linear_hq(%{p}_xn, %{p}_wq) : (tensor<{s}x{h}xf32>, tensor<{q}x{h}xf32>) -> tensor<{s}x{q}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_k = func.call @linear_hkv(%{p}_xn, %{p}_wk) : (tensor<{s}x{h}xf32>, tensor<{kv}x{h}xf32>) -> tensor<{s}x{kv}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_v = func.call @linear_hkv(%{p}_xn, %{p}_wv) : (tensor<{s}x{h}xf32>, tensor<{kv}x{h}xf32>) -> tensor<{s}x{kv}xf32>\n"
        ));
        let nkv = c.num_kv_heads;
        let mk = c.max_kv;
        if c.has_qk_norm {
            f.op_asm(format!(
                "  %{p}_ctx, %{p}_kc, %{p}_vc = func.call @attn(%{p}_q, %{p}_k, %{p}_v, %{p}_qnw, %{p}_knw) : (tensor<{s}x{q}xf32>, tensor<{s}x{kv}xf32>, tensor<{s}x{kv}xf32>, tensor<{d}xf32>, tensor<{d}xf32>) -> (tensor<{s}x{q}xf32>, tensor<{s}x{nkv}x{d}xf32>, tensor<{s}x{nkv}x{d}xf32>)\n"
            ));
        } else {
            f.op_asm(format!(
                "  %{p}_ctx, %{p}_kc, %{p}_vc = func.call @attn(%{p}_q, %{p}_k, %{p}_v) : (tensor<{s}x{q}xf32>, tensor<{s}x{kv}xf32>, tensor<{s}x{kv}xf32>) -> (tensor<{s}x{q}xf32>, tensor<{s}x{nkv}x{d}xf32>, tensor<{s}x{nkv}x{d}xf32>)\n"
            ));
        }
        // Seed mutable KV at positions [0, seq).
        f.op_asm(format!(
            "  %{p}_k_old = util.global.load @kv_k{layer} : tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_v_old = util.global.load @kv_v{layer} : tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_k_new = tensor.insert_slice %{p}_kc into %{p}_k_old[0, 0, 0] [{s}, {nkv}, {d}] [1, 1, 1] : tensor<{s}x{nkv}x{d}xf32> into tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_v_new = tensor.insert_slice %{p}_vc into %{p}_v_old[0, 0, 0] [{s}, {nkv}, {d}] [1, 1, 1] : tensor<{s}x{nkv}x{d}xf32> into tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  util.global.store %{p}_k_new, @kv_k{layer} : tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  util.global.store %{p}_v_new, @kv_v{layer} : tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_o = func.call @linear_qh(%{p}_ctx, %{p}_wo) : (tensor<{s}x{q}xf32>, tensor<{h}x{q}xf32>) -> tensor<{s}x{h}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_h2 = arith.addf %{xin}, %{p}_o : tensor<{s}x{h}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_fn = func.call @rms_norm(%{p}_h2, %{p}_ffn_nw) : (tensor<{s}x{h}xf32>, tensor<{h}xf32>) -> tensor<{s}x{h}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_gate = func.call @linear_hi(%{p}_fn, %{p}_wgate) : (tensor<{s}x{h}xf32>, tensor<{i}x{h}xf32>) -> tensor<{s}x{i}xf32>\n",
            i = c.intermediate
        ));
        f.op_asm(format!(
            "  %{p}_up = func.call @linear_hi(%{p}_fn, %{p}_wup) : (tensor<{s}x{h}xf32>, tensor<{i}x{h}xf32>) -> tensor<{s}x{i}xf32>\n",
            i = c.intermediate
        ));
        f.op_asm(format!(
            r#"  %{p}_silu = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}}
    ins(%{p}_gate : tensor<{s}x{i}xf32>) outs(%{p}_gate : tensor<{s}x{i}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %n = arith.negf %a : f32
      %e = math.exp %n : f32
      %one = arith.constant 1.0 : f32
      %den = arith.addf %one, %e : f32
      %sg = arith.divf %a, %den : f32
      linalg.yield %sg : f32
  }} -> tensor<{s}x{i}xf32>
"#,
            i = c.intermediate
        ));
        f.op_asm(format!(
            "  %{p}_ff = arith.mulf %{p}_silu, %{p}_up : tensor<{s}x{i}xf32>\n",
            i = c.intermediate
        ));
        f.op_asm(format!(
            "  %{p}_down = func.call @linear_ih(%{p}_ff, %{p}_wdown) : (tensor<{s}x{i}xf32>, tensor<{h}x{i}xf32>) -> tensor<{s}x{h}xf32>\n",
            i = c.intermediate
        ));
        f.op_asm(format!(
            "  %{p}_hout = arith.addf %{p}_h2, %{p}_down : tensor<{s}x{h}xf32>\n"
        ));
        h_name = format!("{p}_hout");
    }

    emit_load_compute(
        &mut f,
        c,
        "out_nw",
        "output_norm_weight",
        "output_norm.weight",
        &format!("{h}"),
    );
    emit_load_compute(
        &mut f,
        c,
        "wout",
        "output_weight",
        "output.weight",
        &format!("{v}x{h}"),
    );
    f.op_asm("  %last_i64 = tensor.extract %last[] : tensor<i64>\n");
    f.op_asm("  %li = arith.index_cast %last_i64 : i64 to index\n");
    f.op_asm(format!(
        "  %last_row = tensor.extract_slice %{h_name}[%li, 0] [1, {h}] [1, 1] : tensor<{s}x{h}xf32> to tensor<1x{h}xf32>\n"
    ));
    // Tile to S for rms_norm helper, take row 0.
    f.op_asm(format!(
        "  %last_tile = tensor.empty() : tensor<{s}x{h}xf32>\n"
    ));
    f.op_asm(format!(
        "  %last_s = tensor.insert_slice %last_row into %last_tile[0, 0] [1, {h}] [1, 1] : tensor<1x{h}xf32> into tensor<{s}x{h}xf32>\n"
    ));
    f.op_asm(format!(
        "  %ln = func.call @rms_norm(%last_s, %out_nw) : (tensor<{s}x{h}xf32>, tensor<{h}xf32>) -> tensor<{s}x{h}xf32>\n"
    ));
    f.op_asm(format!(
        "  %ln1 = tensor.extract_slice %ln[0, 0] [1, {h}] [1, 1] : tensor<{s}x{h}xf32> to tensor<1x{h}xf32>\n"
    ));
    f.op_asm("  %c0f = arith.constant 0.0 : f32\n");
    f.op_asm(format!(
        "  %wt_i = tensor.empty() : tensor<{h}x{v}xf32>\n"
    ));
    f.op_asm(format!(
        "  %wt = linalg.transpose ins(%wout : tensor<{v}x{h}xf32>) outs(%wt_i : tensor<{h}x{v}xf32>) permutation = [1, 0]\n"
    ));
    f.op_asm(format!(
        "  %yi = tensor.empty() : tensor<1x{v}xf32>\n"
    ));
    f.op_asm(format!(
        "  %yz = linalg.fill ins(%c0f : f32) outs(%yi : tensor<1x{v}xf32>) -> tensor<1x{v}xf32>\n"
    ));
    f.op_asm(format!(
        "  %y = linalg.matmul ins(%ln1, %wt : tensor<1x{h}xf32>, tensor<{h}x{v}xf32>) outs(%yz : tensor<1x{v}xf32>) -> tensor<1x{v}xf32>\n"
    ));
    f.op_asm(format!(
        "  %logits = tensor.collapse_shape %y [[0, 1]] : tensor<1x{v}xf32> into tensor<{v}xf32>\n"
    ));
    f.op_asm(format!("  return %logits : tensor<{v}xf32>"));

    f.finish(module)
}

fn emit_decode(module: &mut ModuleBuilder, c: &DenseDecoderConfig) -> Result<()> {

    let (h, v, mk) = (c.hidden, c.vocab, c.max_kv);
    let (q, kv) = (c.q_dim(), c.kv_dim());
    let d = c.head_dim;
    let nkv = c.num_kv_heads;
    let mut f = module.func("decode");
    f.arg("token", "tensor<i64>");
    f.arg("pos", "tensor<i64>");
    f.arg("attn_bias", format!("tensor<{mk}xf32>"));
    f.result_ty(format!("tensor<{v}xf32>"));
    emit_load_compute(
        &mut f,
        c,
        "emb_t",
        "token_embd_weight",
        "token_embd.weight",
        &format!("{v}x{h}"),
    );
    f.op_asm("  %tok = tensor.extract %token[] : tensor<i64>\n");
    f.op_asm("  %ti = arith.index_cast %tok : i64 to index\n");
    f.op_asm(format!(
        "  %row = tensor.extract_slice %emb_t[%ti, 0] [1, {h}] [1, 1] : tensor<{v}x{h}xf32> to tensor<1x{h}xf32>\n"
    ));
    let mut h_name = "row".to_string();

    for layer in 0..c.num_layers {
        let p = format!("blk{layer}");
        let n = format!("blk.{layer}");
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_attn_nw"),
            &format!("{p}_attn_norm_weight"),
            &format!("{n}.attn_norm.weight"),
            &format!("{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wq"),
            &format!("{p}_attn_q_weight"),
            &format!("{n}.attn_q.weight"),
            &format!("{q}x{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wk"),
            &format!("{p}_attn_k_weight"),
            &format!("{n}.attn_k.weight"),
            &format!("{kv}x{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wv"),
            &format!("{p}_attn_v_weight"),
            &format!("{n}.attn_v.weight"),
            &format!("{kv}x{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wo"),
            &format!("{p}_attn_output_weight"),
            &format!("{n}.attn_output.weight"),
            &format!("{h}x{q}"),
        );
        if c.has_qk_norm {
            emit_load_compute(
                &mut f,
                c,
                &format!("{p}_qnw"),
                &format!("{p}_attn_q_norm_weight"),
                &format!("{n}.attn_q_norm.weight"),
                &format!("{d}"),
            );
            emit_load_compute(
                &mut f,
                c,
                &format!("{p}_knw"),
                &format!("{p}_attn_k_norm_weight"),
                &format!("{n}.attn_k_norm.weight"),
                &format!("{d}"),
            );
        }
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_ffn_nw"),
            &format!("{p}_ffn_norm_weight"),
            &format!("{n}.ffn_norm.weight"),
            &format!("{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wgate"),
            &format!("{p}_ffn_gate_weight"),
            &format!("{n}.ffn_gate.weight"),
            &format!("{}x{h}", c.intermediate),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wup"),
            &format!("{p}_ffn_up_weight"),
            &format!("{n}.ffn_up.weight"),
            &format!("{}x{h}", c.intermediate),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wdown"),
            &format!("{p}_ffn_down_weight"),
            &format!("{n}.ffn_down.weight"),
            &format!("{h}x{}", c.intermediate),
        );

        let xin = h_name.clone();
        f.op_asm(format!(
            "  %{p}_xn = func.call @rms_norm_tok(%{xin}, %{p}_attn_nw) : (tensor<1x{h}xf32>, tensor<{h}xf32>) -> tensor<1x{h}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_q = func.call @linear_hq_tok(%{p}_xn, %{p}_wq) : (tensor<1x{h}xf32>, tensor<{q}x{h}xf32>) -> tensor<1x{q}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_k = func.call @linear_hkv_tok(%{p}_xn, %{p}_wk) : (tensor<1x{h}xf32>, tensor<{kv}x{h}xf32>) -> tensor<1x{kv}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_v = func.call @linear_hkv_tok(%{p}_xn, %{p}_wv) : (tensor<1x{h}xf32>, tensor<{kv}x{h}xf32>) -> tensor<1x{kv}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_k3 = tensor.expand_shape %{p}_k [[0], [1, 2]] output_shape [1, {nkv}, {d}] : tensor<1x{kv}xf32> into tensor<1x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_v3 = tensor.expand_shape %{p}_v [[0], [1, 2]] output_shape [1, {nkv}, {d}] : tensor<1x{kv}xf32> into tensor<1x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_k_old = util.global.load @kv_k{layer} : tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_v_old = util.global.load @kv_v{layer} : tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        if c.has_qk_norm {
            f.op_asm(format!(
                "  %{p}_ctx, %{p}_k_new, %{p}_v_new = func.call @attn_decode(%{p}_q, %{p}_k_old, %{p}_v_old, %{p}_k3, %{p}_v3, %pos, %attn_bias, %{p}_qnw, %{p}_knw) : (tensor<1x{q}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<1x{nkv}x{d}xf32>, tensor<1x{nkv}x{d}xf32>, tensor<i64>, tensor<{mk}xf32>, tensor<{d}xf32>, tensor<{d}xf32>) -> (tensor<1x{q}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<{mk}x{nkv}x{d}xf32>)\n"
            ));
        } else {
            f.op_asm(format!(
                "  %{p}_ctx, %{p}_k_new, %{p}_v_new = func.call @attn_decode(%{p}_q, %{p}_k_old, %{p}_v_old, %{p}_k3, %{p}_v3, %pos, %attn_bias) : (tensor<1x{q}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<1x{nkv}x{d}xf32>, tensor<1x{nkv}x{d}xf32>, tensor<i64>, tensor<{mk}xf32>) -> (tensor<1x{q}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<{mk}x{nkv}x{d}xf32>)\n"
            ));
        }
        f.op_asm(format!(
            "  util.global.store %{p}_k_new, @kv_k{layer} : tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  util.global.store %{p}_v_new, @kv_v{layer} : tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_o = func.call @linear_qh_tok(%{p}_ctx, %{p}_wo) : (tensor<1x{q}xf32>, tensor<{h}x{q}xf32>) -> tensor<1x{h}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_h2 = arith.addf %{xin}, %{p}_o : tensor<1x{h}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_fn = func.call @rms_norm_tok(%{p}_h2, %{p}_ffn_nw) : (tensor<1x{h}xf32>, tensor<{h}xf32>) -> tensor<1x{h}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_gate = func.call @linear_hi_tok(%{p}_fn, %{p}_wgate) : (tensor<1x{h}xf32>, tensor<{i}x{h}xf32>) -> tensor<1x{i}xf32>\n",
            i = c.intermediate
        ));
        f.op_asm(format!(
            "  %{p}_up = func.call @linear_hi_tok(%{p}_fn, %{p}_wup) : (tensor<1x{h}xf32>, tensor<{i}x{h}xf32>) -> tensor<1x{i}xf32>\n",
            i = c.intermediate
        ));
        f.op_asm(format!(
            r#"  %{p}_silu = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}}
    ins(%{p}_gate : tensor<1x{i}xf32>) outs(%{p}_gate : tensor<1x{i}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %n = arith.negf %a : f32
      %e = math.exp %n : f32
      %one = arith.constant 1.0 : f32
      %den = arith.addf %one, %e : f32
      %sg = arith.divf %a, %den : f32
      linalg.yield %sg : f32
  }} -> tensor<1x{i}xf32>
"#,
            i = c.intermediate
        ));
        f.op_asm(format!(
            "  %{p}_ff = arith.mulf %{p}_silu, %{p}_up : tensor<1x{i}xf32>\n",
            i = c.intermediate
        ));
        f.op_asm(format!(
            "  %{p}_down = func.call @linear_ih_tok(%{p}_ff, %{p}_wdown) : (tensor<1x{i}xf32>, tensor<{h}x{i}xf32>) -> tensor<1x{h}xf32>\n",
            i = c.intermediate
        ));
        f.op_asm(format!(
            "  %{p}_hout = arith.addf %{p}_h2, %{p}_down : tensor<1x{h}xf32>\n"
        ));
        h_name = format!("{p}_hout");
    }

    emit_load_compute(
        &mut f,
        c,
        "out_nw",
        "output_norm_weight",
        "output_norm.weight",
        &format!("{h}"),
    );
    emit_load_compute(
        &mut f,
        c,
        "wout",
        "output_weight",
        "output.weight",
        &format!("{v}x{h}"),
    );
    f.op_asm(format!(
        "  %ln = func.call @rms_norm_tok(%{h_name}, %out_nw) : (tensor<1x{h}xf32>, tensor<{h}xf32>) -> tensor<1x{h}xf32>\n"
    ));
    f.op_asm("  %c0f = arith.constant 0.0 : f32\n");
    f.op_asm(format!("  %wt_i = tensor.empty() : tensor<{h}x{v}xf32>\n"));
    f.op_asm(format!(
        "  %wt = linalg.transpose ins(%wout : tensor<{v}x{h}xf32>) outs(%wt_i : tensor<{h}x{v}xf32>) permutation = [1, 0]\n"
    ));
    f.op_asm(format!("  %yi = tensor.empty() : tensor<1x{v}xf32>\n"));
    f.op_asm(format!(
        "  %yz = linalg.fill ins(%c0f : f32) outs(%yi : tensor<1x{v}xf32>) -> tensor<1x{v}xf32>\n"
    ));
    f.op_asm(format!(
        "  %y = linalg.matmul ins(%ln, %wt : tensor<1x{h}xf32>, tensor<{h}x{v}xf32>) outs(%yz : tensor<1x{v}xf32>) -> tensor<1x{v}xf32>\n"
    ));
    f.op_asm(format!(
        "  %logits = tensor.collapse_shape %y [[0, 1]] : tensor<1x{v}xf32> into tensor<{v}xf32>\n"
    ));
    f.op_asm(format!("  return %logits : tensor<{v}xf32>"));

    f.finish(module)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_0_6b_shape_supported() {
        let c = DenseDecoderConfig {
            vocab: 151_936,
            hidden: 1024,
            intermediate: 3072,
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 128,
            num_layers: 28,
            seq: LARGE_PREFILL_WINDOW,
            max_kv: LARGE_MAX_KV,
            rms_norm_eps: 1e-6,
            rope_theta: Some(1_000_000.0),
            has_qk_norm: true,
            param_keys: BTreeMap::new(),
            param_dtypes: BTreeMap::from([(
                "token_embd.weight".into(),
                ScalarType::Bf16,
            )]),
        };
        assert!(c.supports_dense_emit());
        assert_eq!(c.q_dim(), 2048);
        assert_eq!(c.kv_dim(), 1024);
        assert_eq!(c.gqa_group(), 2);
        assert_eq!(c.param_dtype("token_embd.weight"), ScalarType::Bf16);
        assert_eq!(c.param_dtype("missing.weight"), ScalarType::F32);
    }

    #[test]
    fn tiny_mha_still_supported() {
        let c = DenseDecoderConfig {
            vocab: 32,
            hidden: 64,
            intermediate: 128,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            num_layers: 1,
            seq: TINY_PREFILL_WINDOW,
            max_kv: TINY_MAX_KV,
            rms_norm_eps: 1e-5,
            rope_theta: None,
            has_qk_norm: false,
            param_keys: BTreeMap::new(),
            param_dtypes: BTreeMap::new(),
        };
        assert!(c.supports_dense_emit());
        assert!(c.is_tiny_m1());
    }

    #[test]
    fn emit_uses_per_tensor_checkpoint_dtypes() {
        let mut param_dtypes = BTreeMap::new();
        param_dtypes.insert("token_embd.weight".into(), ScalarType::Bf16);
        param_dtypes.insert("blk.0.attn_norm.weight".into(), ScalarType::F16);
        param_dtypes.insert("blk.0.attn_q.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.attn_k.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.attn_v.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.attn_output.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.ffn_norm.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.ffn_gate.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.ffn_up.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.ffn_down.weight".into(), ScalarType::F32);
        param_dtypes.insert("output_norm.weight".into(), ScalarType::F32);
        param_dtypes.insert("output.weight".into(), ScalarType::Bf16);
        let c = DenseDecoderConfig {
            vocab: 32,
            hidden: 64,
            intermediate: 128,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            num_layers: 1,
            seq: TINY_PREFILL_WINDOW,
            max_kv: TINY_MAX_KV,
            rms_norm_eps: 1e-5,
            rope_theta: None,
            has_qk_norm: false,
            param_keys: BTreeMap::new(),
            param_dtypes,
        };
        let mlir = emit_dense_decoder_cfg("test.decoder", &c).expect("mlir verify");
        assert!(
            mlir.contains("@token_embd_weight") && mlir.contains("tensor<32x64xbf16>"),
            "missing bf16 token embd global: {mlir}"
        );
        assert!(
            mlir.contains("@blk0_attn_norm_weight") && mlir.contains("tensor<64xf16>"),
            "missing f16 attn norm global: {mlir}"
        );
        assert!(
            mlir.contains("@blk0_attn_q_weight") && mlir.contains("tensor<64x64xf32>"),
            "missing f32 q weight global: {mlir}"
        );
        assert!(
            mlir.contains("arith.extf") && mlir.contains("bf16") && mlir.contains("f32"),
            "missing bf16→f32 cast: {mlir}"
        );
        assert!(
            mlir.contains("@prefill") || mlir.contains("func.func @prefill"),
            "verified module missing prefill: {mlir}"
        );
    }
}
