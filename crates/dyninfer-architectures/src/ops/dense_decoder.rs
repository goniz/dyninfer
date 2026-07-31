//! Shared dense causal-decoder MLIR emitter (ops library).
//!
//! Architecture files (`models/*`) configure [`DenseDecoderConfig`] and call
//! [`emit_dense_decoder`]. Supports MHA/GQA, independent `head_dim`, optional
//! Q/K RMSNorm, and HuggingFace Llama/Qwen-style (`rotate_half`) RoPE.
//!
//! Weight globals always use the per-tensor dtype from the checkpoint catalog.
//! Activations / logits use [`COMPUTE_DTYPE`] (currently f32); narrower float
//! weights are cast after load.

use dyninfer_architecture::ArchitecturePackage;
use dyninfer_checkpoint::CheckpointCatalog;
use dyninfer_core::{ScalarType, StorageElementType};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

/// Default static prefill window for mid-size dense models.
pub const PREFILL_WINDOW: u32 = 64;
/// Smaller window for large models (Qwen3-0.6B) to keep compile/runtime tractable.
pub const LARGE_PREFILL_WINDOW: u32 = 32;
/// Window used by the synthetic Milestone-1 fixture (fast differential tests).
pub const TINY_PREFILL_WINDOW: u32 = 4;

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
    pub seq: u32,
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
        let seq = if is_tiny {
            TINY_PREFILL_WINDOW
        } else if is_large {
            LARGE_PREFILL_WINDOW
        } else {
            PREFILL_WINDOW
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

    fn weight_dtypes_summary(&self) -> String {
        let mut set = BTreeSet::new();
        for ty in self.param_dtypes.values() {
            set.insert(ty.to_string());
        }
        if set.is_empty() {
            return COMPUTE_DTYPE.to_string();
        }
        set.into_iter().collect::<Vec<_>>().join(",")
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
    }
}

/// Emit using an explicit config (architecture files may override flags).
pub fn emit_dense_decoder_cfg(arch_id: &str, c: &DenseDecoderConfig) -> String {
    assert!(
        c.supports_dense_emit(),
        "unsupported dense decoder emit config: {c:?}"
    );
    let mut out = String::new();
    let _ = writeln!(
        out,
        "// dyninfer dense decoder arch={} layers={} seq={} gqa={}/{} head_dim={} qk_norm={} rope={:?} weight_dtypes=[{}] compute={}",
        arch_id,
        c.num_layers,
        c.seq,
        c.num_heads,
        c.num_kv_heads,
        c.head_dim,
        c.has_qk_norm,
        c.rope_theta,
        c.weight_dtypes_summary(),
        COMPUTE_DTYPE,
    );
    emit_globals(&mut out, c);
    emit_helpers(&mut out, c);
    emit_prefill(&mut out, c);
    emit_decode(&mut out, c);
    out.push_str(
        r#"
func.func @add(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}
"#,
    );
    out
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

fn emit_global(out: &mut String, c: &DenseDecoderConfig, sym: &str, canonical: &str, shape: &str) {
    let key = c.param_key(canonical);
    let wt = mlir_ty(c.param_dtype(canonical));
    out.push_str(&format!(
        "util.global private @{sym} = #stream.parameter.named<\"weights\"::\"{key}\"> : tensor<{shape}x{wt}>\n"
    ));
}

/// Load a weight global and cast to [`COMPUTE_DTYPE`] when the checkpoint dtype differs.
fn emit_load_compute(
    out: &mut String,
    c: &DenseDecoderConfig,
    ssa: &str,
    sym: &str,
    canonical: &str,
    shape: &str,
) {
    let storage = c.param_dtype(canonical);
    let wt = mlir_ty(storage);
    let ct = mlir_ty(COMPUTE_DTYPE);
    if storage == COMPUTE_DTYPE {
        out.push_str(&format!(
            "  %{ssa} = util.global.load @{sym} : tensor<{shape}x{ct}>\n"
        ));
        return;
    }
    match storage {
        ScalarType::F16 | ScalarType::Bf16 if COMPUTE_DTYPE == ScalarType::F32 => {
            out.push_str(&format!(
                "  %{ssa}_native = util.global.load @{sym} : tensor<{shape}x{wt}>\n"
            ));
            out.push_str(&format!(
                "  %{ssa} = arith.extf %{ssa}_native : tensor<{shape}x{wt}> to tensor<{shape}x{ct}>\n"
            ));
        }
        other => panic!(
            "dense decoder: cannot cast checkpoint dtype {other} → compute {COMPUTE_DTYPE} for {canonical}"
        ),
    }
}

fn emit_globals(out: &mut String, c: &DenseDecoderConfig) {
    let (v, h, i) = (c.vocab, c.hidden, c.intermediate);
    let (q, kv, d) = (c.q_dim(), c.kv_dim(), c.head_dim);
    emit_global(out, c, "token_embd_weight", "token_embd.weight", &format!("{v}x{h}"));
    for layer in 0..c.num_layers {
        let p = format!("blk{layer}");
        let n = format!("blk.{layer}");
        emit_global(
            out,
            c,
            &format!("{p}_attn_norm_weight"),
            &format!("{n}.attn_norm.weight"),
            &format!("{h}"),
        );
        emit_global(
            out,
            c,
            &format!("{p}_attn_q_weight"),
            &format!("{n}.attn_q.weight"),
            &format!("{q}x{h}"),
        );
        emit_global(
            out,
            c,
            &format!("{p}_attn_k_weight"),
            &format!("{n}.attn_k.weight"),
            &format!("{kv}x{h}"),
        );
        emit_global(
            out,
            c,
            &format!("{p}_attn_v_weight"),
            &format!("{n}.attn_v.weight"),
            &format!("{kv}x{h}"),
        );
        emit_global(
            out,
            c,
            &format!("{p}_attn_output_weight"),
            &format!("{n}.attn_output.weight"),
            &format!("{h}x{q}"),
        );
        if c.has_qk_norm {
            emit_global(
                out,
                c,
                &format!("{p}_attn_q_norm_weight"),
                &format!("{n}.attn_q_norm.weight"),
                &format!("{d}"),
            );
            emit_global(
                out,
                c,
                &format!("{p}_attn_k_norm_weight"),
                &format!("{n}.attn_k_norm.weight"),
                &format!("{d}"),
            );
        }
        emit_global(
            out,
            c,
            &format!("{p}_ffn_norm_weight"),
            &format!("{n}.ffn_norm.weight"),
            &format!("{h}"),
        );
        emit_global(
            out,
            c,
            &format!("{p}_ffn_gate_weight"),
            &format!("{n}.ffn_gate.weight"),
            &format!("{i}x{h}"),
        );
        emit_global(
            out,
            c,
            &format!("{p}_ffn_up_weight"),
            &format!("{n}.ffn_up.weight"),
            &format!("{i}x{h}"),
        );
        emit_global(
            out,
            c,
            &format!("{p}_ffn_down_weight"),
            &format!("{n}.ffn_down.weight"),
            &format!("{h}x{i}"),
        );
    }
    emit_global(
        out,
        c,
        "output_norm_weight",
        "output_norm.weight",
        &format!("{h}"),
    );
    emit_global(
        out,
        c,
        "output_weight",
        "output.weight",
        &format!("{v}x{h}"),
    );
    out.push('\n');
}

fn emit_helpers(out: &mut String, c: &DenseDecoderConfig) {
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
    let eps = format!("{:.8e}", c.rms_norm_eps);
    let h_f = format!("{:.1}", h as f32);
    let d_f = format!("{:.1}", d as f32);

    out.push_str(&format!(
        r#"func.func private @rms_norm(%x: tensor<{s}x{h}xf32>, %w: tensor<{h}xf32>) -> tensor<{s}x{h}xf32> {{
  %c0 = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %eps = arith.constant {eps} : f32
  %ch = arith.constant {h_f} : f32
  %sq = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}}
    ins(%x : tensor<{s}x{h}xf32>) outs(%x : tensor<{s}x{h}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %p = arith.mulf %a, %a : f32
      linalg.yield %p : f32
  }} -> tensor<{s}x{h}xf32>
  %init = tensor.empty() : tensor<{s}xf32>
  %z = linalg.fill ins(%c0 : f32) outs(%init : tensor<{s}xf32>) -> tensor<{s}xf32>
  %ms = linalg.reduce ins(%sq : tensor<{s}x{h}xf32>) outs(%z : tensor<{s}xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {{
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }}
  %inv = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}}
    ins(%ms : tensor<{s}xf32>) outs(%ms : tensor<{s}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %m = arith.divf %a, %ch : f32
      %meps = arith.addf %m, %eps : f32
      %root = math.sqrt %meps : f32
      %i = arith.divf %one, %root : f32
      linalg.yield %i : f32
  }} -> tensor<{s}xf32>
  %y = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}}
    ins(%x, %inv, %w : tensor<{s}x{h}xf32>, tensor<{s}xf32>, tensor<{h}xf32>) outs(%x : tensor<{s}x{h}xf32>) {{
    ^bb0(%a: f32, %i: f32, %ww: f32, %o: f32):
      %t = arith.mulf %a, %i : f32
      %r = arith.mulf %t, %ww : f32
      linalg.yield %r : f32
  }} -> tensor<{s}x{h}xf32>
  return %y : tensor<{s}x{h}xf32>
}}

func.func private @linear_hq(%x: tensor<{s}x{h}xf32>, %w: tensor<{q}x{h}xf32>) -> tensor<{s}x{q}xf32> {{
  %c0 = arith.constant 0.0 : f32
  %ti = tensor.empty() : tensor<{h}x{q}xf32>
  %wt = linalg.transpose ins(%w : tensor<{q}x{h}xf32>) outs(%ti : tensor<{h}x{q}xf32>) permutation = [1, 0]
  %init = tensor.empty() : tensor<{s}x{q}xf32>
  %z = linalg.fill ins(%c0 : f32) outs(%init : tensor<{s}x{q}xf32>) -> tensor<{s}x{q}xf32>
  %y = linalg.matmul ins(%x, %wt : tensor<{s}x{h}xf32>, tensor<{h}x{q}xf32>) outs(%z : tensor<{s}x{q}xf32>) -> tensor<{s}x{q}xf32>
  return %y : tensor<{s}x{q}xf32>
}}

func.func private @linear_hkv(%x: tensor<{s}x{h}xf32>, %w: tensor<{kv}x{h}xf32>) -> tensor<{s}x{kv}xf32> {{
  %c0 = arith.constant 0.0 : f32
  %ti = tensor.empty() : tensor<{h}x{kv}xf32>
  %wt = linalg.transpose ins(%w : tensor<{kv}x{h}xf32>) outs(%ti : tensor<{h}x{kv}xf32>) permutation = [1, 0]
  %init = tensor.empty() : tensor<{s}x{kv}xf32>
  %z = linalg.fill ins(%c0 : f32) outs(%init : tensor<{s}x{kv}xf32>) -> tensor<{s}x{kv}xf32>
  %y = linalg.matmul ins(%x, %wt : tensor<{s}x{h}xf32>, tensor<{h}x{kv}xf32>) outs(%z : tensor<{s}x{kv}xf32>) -> tensor<{s}x{kv}xf32>
  return %y : tensor<{s}x{kv}xf32>
}}

func.func private @linear_qh(%x: tensor<{s}x{q}xf32>, %w: tensor<{h}x{q}xf32>) -> tensor<{s}x{h}xf32> {{
  %c0 = arith.constant 0.0 : f32
  %ti = tensor.empty() : tensor<{q}x{h}xf32>
  %wt = linalg.transpose ins(%w : tensor<{h}x{q}xf32>) outs(%ti : tensor<{q}x{h}xf32>) permutation = [1, 0]
  %init = tensor.empty() : tensor<{s}x{h}xf32>
  %z = linalg.fill ins(%c0 : f32) outs(%init : tensor<{s}x{h}xf32>) -> tensor<{s}x{h}xf32>
  %y = linalg.matmul ins(%x, %wt : tensor<{s}x{q}xf32>, tensor<{q}x{h}xf32>) outs(%z : tensor<{s}x{h}xf32>) -> tensor<{s}x{h}xf32>
  return %y : tensor<{s}x{h}xf32>
}}

func.func private @linear_hi(%x: tensor<{s}x{h}xf32>, %w: tensor<{i}x{h}xf32>) -> tensor<{s}x{i}xf32> {{
  %c0 = arith.constant 0.0 : f32
  %ti = tensor.empty() : tensor<{h}x{i}xf32>
  %wt = linalg.transpose ins(%w : tensor<{i}x{h}xf32>) outs(%ti : tensor<{h}x{i}xf32>) permutation = [1, 0]
  %init = tensor.empty() : tensor<{s}x{i}xf32>
  %z = linalg.fill ins(%c0 : f32) outs(%init : tensor<{s}x{i}xf32>) -> tensor<{s}x{i}xf32>
  %y = linalg.matmul ins(%x, %wt : tensor<{s}x{h}xf32>, tensor<{h}x{i}xf32>) outs(%z : tensor<{s}x{i}xf32>) -> tensor<{s}x{i}xf32>
  return %y : tensor<{s}x{i}xf32>
}}

func.func private @linear_ih(%x: tensor<{s}x{i}xf32>, %w: tensor<{h}x{i}xf32>) -> tensor<{s}x{h}xf32> {{
  %c0 = arith.constant 0.0 : f32
  %ti = tensor.empty() : tensor<{i}x{h}xf32>
  %wt = linalg.transpose ins(%w : tensor<{h}x{i}xf32>) outs(%ti : tensor<{i}x{h}xf32>) permutation = [1, 0]
  %init = tensor.empty() : tensor<{s}x{h}xf32>
  %z = linalg.fill ins(%c0 : f32) outs(%init : tensor<{s}x{h}xf32>) -> tensor<{s}x{h}xf32>
  %y = linalg.matmul ins(%x, %wt : tensor<{s}x{i}xf32>, tensor<{i}x{h}xf32>) outs(%z : tensor<{s}x{h}xf32>) -> tensor<{s}x{h}xf32>
  return %y : tensor<{s}x{h}xf32>
}}

"#,
        s = s,
        h = h,
        i = i,
        q = q,
        kv = kv,
        eps = eps,
        h_f = h_f,
    ));

    if c.has_qk_norm {
        emit_rms_norm_heads(out, s, nh, d, &eps, &d_f, "rms_norm_q_heads");
        if nkv != nh {
            emit_rms_norm_heads(out, s, nkv, d, &eps, &d_f, "rms_norm_kv_heads");
        }
    }

    if nkv != nh {
        out.push_str(&format!(
            r#"func.func private @repeat_kv(%x: tensor<{s}x{nkv}x{d}xf32>) -> tensor<{s}x{nh}x{d}xf32> {{
  %init = tensor.empty() : tensor<{s}x{nh}x{d}xf32>
  %y = linalg.generic {{
      indexing_maps = [
        affine_map<(p, h, dim) -> (p, h floordiv {g}, dim)>,
        affine_map<(p, h, dim) -> (p, h, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%x : tensor<{s}x{nkv}x{d}xf32>) outs(%init : tensor<{s}x{nh}x{d}xf32>) {{
    ^bb0(%a: f32, %o: f32):
      linalg.yield %a : f32
  }} -> tensor<{s}x{nh}x{d}xf32>
  return %y : tensor<{s}x{nh}x{d}xf32>
}}

"#,
            s = s,
            nkv = nkv,
            nh = nh,
            d = d,
            g = g,
        ));
    }

    if let Some(theta) = c.rope_theta {
        emit_rope_helper(out, s, nh, d, theta, "apply_rope_q");
        if nkv != nh {
            emit_rope_helper(out, s, nkv, d, theta, "apply_rope_kv");
        }
    }

    let scale = 1.0 / (d as f32).sqrt();
    out.push_str(&format!(
        r#"func.func private @attn(%q: tensor<{s}x{q}xf32>, %k: tensor<{s}x{kv}xf32>, %v: tensor<{s}x{kv}xf32>"#
    ));
    if c.has_qk_norm {
        out.push_str(&format!(
            ", %q_norm: tensor<{d}xf32>, %k_norm: tensor<{d}xf32>"
        ));
    }
    out.push_str(&format!(
        r#") -> tensor<{s}x{q}xf32> {{
  %q3 = tensor.expand_shape %q [[0], [1, 2]] output_shape [{s}, {nh}, {d}] : tensor<{s}x{q}xf32> into tensor<{s}x{nh}x{d}xf32>
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
        out.push_str(&format!(
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
        out.push_str(&format!(
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
        out.push_str(&format!(
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
        out.push_str(&format!(
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
        out.push_str(&format!(
            "  %krep = func.call @repeat_kv(%{k_ssa}) : (tensor<{s}x{nkv}x{d}xf32>) -> tensor<{s}x{nh}x{d}xf32>\n",
            k_ssa = k_ssa,
            s = s,
            nkv = nkv,
            nh = nh,
            d = d,
        ));
        out.push_str(&format!(
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

    out.push_str(&format!(
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
  return %ctx : tensor<{s}x{q}xf32>
}}

"#,
        s = s,
        q = q,
        nh = nh,
        d = d,
        scale = scale,
        q_ssa = q_ssa,
        k_full = k_full.0,
        v_full = k_full.1,
    ));
}

fn emit_rms_norm_heads(
    out: &mut String,
    s: u32,
    nh: u32,
    d: u32,
    eps: &str,
    d_f: &str,
    name: &str,
) {
    out.push_str(&format!(
        r#"func.func private @{name}(%x: tensor<{s}x{nh}x{d}xf32>, %w: tensor<{d}xf32>) -> tensor<{s}x{nh}x{d}xf32> {{
  %c0 = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %eps = arith.constant {eps} : f32
  %cd = arith.constant {d_f} : f32
  %sq = linalg.generic {{
      indexing_maps = [affine_map<(p, h, dim) -> (p, h, dim)>, affine_map<(p, h, dim) -> (p, h, dim)>],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%x : tensor<{s}x{nh}x{d}xf32>) outs(%x : tensor<{s}x{nh}x{d}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %p = arith.mulf %a, %a : f32
      linalg.yield %p : f32
  }} -> tensor<{s}x{nh}x{d}xf32>
  %init = tensor.empty() : tensor<{s}x{nh}xf32>
  %z = linalg.fill ins(%c0 : f32) outs(%init : tensor<{s}x{nh}xf32>) -> tensor<{s}x{nh}xf32>
  %ms = linalg.reduce ins(%sq : tensor<{s}x{nh}x{d}xf32>) outs(%z : tensor<{s}x{nh}xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {{
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }}
  %inv = linalg.generic {{
      indexing_maps = [affine_map<(p, h) -> (p, h)>, affine_map<(p, h) -> (p, h)>],
      iterator_types = ["parallel", "parallel"]}}
    ins(%ms : tensor<{s}x{nh}xf32>) outs(%ms : tensor<{s}x{nh}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %m = arith.divf %a, %cd : f32
      %meps = arith.addf %m, %eps : f32
      %root = math.sqrt %meps : f32
      %i = arith.divf %one, %root : f32
      linalg.yield %i : f32
  }} -> tensor<{s}x{nh}xf32>
  %y = linalg.generic {{
      indexing_maps = [
        affine_map<(p, h, dim) -> (p, h, dim)>,
        affine_map<(p, h, dim) -> (p, h)>,
        affine_map<(p, h, dim) -> (dim)>,
        affine_map<(p, h, dim) -> (p, h, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%x, %inv, %w : tensor<{s}x{nh}x{d}xf32>, tensor<{s}x{nh}xf32>, tensor<{d}xf32>) outs(%x : tensor<{s}x{nh}x{d}xf32>) {{
    ^bb0(%a: f32, %i: f32, %ww: f32, %o: f32):
      %t = arith.mulf %a, %i : f32
      %r = arith.mulf %t, %ww : f32
      linalg.yield %r : f32
  }} -> tensor<{s}x{nh}x{d}xf32>
  return %y : tensor<{s}x{nh}x{d}xf32>
}}

"#,
        name = name,
        s = s,
        nh = nh,
        d = d,
        eps = eps,
        d_f = d_f,
    ));
}

fn emit_rope_helper(out: &mut String, s: u32, nh: u32, d: u32, theta: f32, name: &str) {
    // HuggingFace Llama/Qwen RoPE: rotate_half.
    //   x1, x2 = x[..., :D/2], x[..., D/2:]
    //   out = cat(x1*cos - x2*sin, x1*sin + x2*cos)
    // Cos/sin tables are [S, D/2] with freq_i = theta^(-2i/D).
    let half = (d / 2) as usize;
    let mut cos_rows = Vec::with_capacity(s as usize);
    let mut sin_rows = Vec::with_capacity(s as usize);
    for pos in 0..s {
        let mut cos_row = Vec::with_capacity(half);
        let mut sin_row = Vec::with_capacity(half);
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / d as f32);
            let angle = pos as f32 * freq;
            cos_row.push(format!("{:.8e}", angle.cos()));
            sin_row.push(format!("{:.8e}", angle.sin()));
        }
        cos_rows.push(format!("[{}]", cos_row.join(", ")));
        sin_rows.push(format!("[{}]", sin_row.join(", ")));
    }
    let cos_lit = cos_rows.join(", ");
    let sin_lit = sin_rows.join(", ");
    out.push_str(&format!(
        r#"func.func private @{name}(%x: tensor<{s}x{nh}x{d}xf32>) -> tensor<{s}x{nh}x{d}xf32> {{
  %cos = arith.constant dense<[{cos_lit}]> : tensor<{s}x{half}xf32>
  %sin = arith.constant dense<[{sin_lit}]> : tensor<{s}x{half}xf32>
  %init = tensor.empty() : tensor<{s}x{nh}x{d}xf32>
  %y = linalg.generic {{
      indexing_maps = [
        affine_map<(p, h, dim) -> (p, h, dim)>,
        affine_map<(p, h, dim) -> (p, h, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%x : tensor<{s}x{nh}x{d}xf32>) outs(%init : tensor<{s}x{nh}x{d}xf32>) {{
    ^bb0(%a: f32, %o: f32):
      %p = linalg.index 0 : index
      %hh = linalg.index 1 : index
      %dim = linalg.index 2 : index
      %half_i = arith.constant {half} : index
      %in_first = arith.cmpi ult, %dim, %half_i : index
      %pair_lo = arith.remui %dim, %half_i : index
      %dim_hi = arith.addi %pair_lo, %half_i : index
      %x1 = tensor.extract %x[%p, %hh, %pair_lo] : tensor<{s}x{nh}x{d}xf32>
      %x2 = tensor.extract %x[%p, %hh, %dim_hi] : tensor<{s}x{nh}x{d}xf32>
      %cv = tensor.extract %cos[%p, %pair_lo] : tensor<{s}x{half}xf32>
      %sv = tensor.extract %sin[%p, %pair_lo] : tensor<{s}x{half}xf32>
      %x1c = arith.mulf %x1, %cv : f32
      %x2s = arith.mulf %x2, %sv : f32
      %x1s = arith.mulf %x1, %sv : f32
      %x2c = arith.mulf %x2, %cv : f32
      %lo = arith.subf %x1c, %x2s : f32
      %hi = arith.addf %x1s, %x2c : f32
      %r = arith.select %in_first, %lo, %hi : f32
      linalg.yield %r : f32
  }} -> tensor<{s}x{nh}x{d}xf32>
  return %y : tensor<{s}x{nh}x{d}xf32>
}}

"#,
        name = name,
        s = s,
        nh = nh,
        d = d,
        half = half,
        cos_lit = cos_lit,
        sin_lit = sin_lit,
    ));
}

fn emit_prefill(out: &mut String, c: &DenseDecoderConfig) {
    let (s, h, v) = (c.seq, c.hidden, c.vocab);
    let (q, kv) = (c.q_dim(), c.kv_dim());
    let d = c.head_dim;
    // Tokens are left-aligned (right-padded). `%last` is the index of the
    // newest real token so RoPE/causal attention ignore trailing pads.
    out.push_str(&format!(
        "func.func @prefill(%tokens: tensor<{s}xi64>, %last: tensor<i64>) -> tensor<{v}xf32> {{\n"
    ));
    emit_load_compute(
        out,
        c,
        "emb_t",
        "token_embd_weight",
        "token_embd.weight",
        &format!("{v}x{h}"),
    );

    // Embedding gather (unrolled).
    out.push_str(&format!(
        "  %h_acc0 = tensor.empty() : tensor<{s}x{h}xf32>\n"
    ));
    for pos in 0..s {
        out.push_str(&format!("  %c{pos}i = arith.constant {pos} : index\n"));
        out.push_str(&format!(
            "  %t{pos} = tensor.extract %tokens[%c{pos}i] : tensor<{s}xi64>\n"
        ));
        out.push_str(&format!(
            "  %i{pos} = arith.index_cast %t{pos} : i64 to index\n"
        ));
        out.push_str(&format!(
            "  %r{pos} = tensor.extract_slice %emb_t[%i{pos}, 0] [1, {h}] [1, 1] : tensor<{v}x{h}xf32> to tensor<1x{h}xf32>\n"
        ));
        let prev = if pos == 0 {
            "h_acc0".to_string()
        } else {
            format!("h_acc{pos}")
        };
        let next = format!("h_acc{}", pos + 1);
        out.push_str(&format!(
            "  %{next} = tensor.insert_slice %r{pos} into %{prev}[{pos}, 0] [1, {h}] [1, 1] : tensor<1x{h}xf32> into tensor<{s}x{h}xf32>\n"
        ));
    }
    let mut h_name = format!("h_acc{s}");

    for layer in 0..c.num_layers {
        let p = format!("blk{layer}");
        let n = format!("blk.{layer}");
        emit_load_compute(
            out,
            c,
            &format!("{p}_attn_nw"),
            &format!("{p}_attn_norm_weight"),
            &format!("{n}.attn_norm.weight"),
            &format!("{h}"),
        );
        emit_load_compute(
            out,
            c,
            &format!("{p}_wq"),
            &format!("{p}_attn_q_weight"),
            &format!("{n}.attn_q.weight"),
            &format!("{q}x{h}"),
        );
        emit_load_compute(
            out,
            c,
            &format!("{p}_wk"),
            &format!("{p}_attn_k_weight"),
            &format!("{n}.attn_k.weight"),
            &format!("{kv}x{h}"),
        );
        emit_load_compute(
            out,
            c,
            &format!("{p}_wv"),
            &format!("{p}_attn_v_weight"),
            &format!("{n}.attn_v.weight"),
            &format!("{kv}x{h}"),
        );
        emit_load_compute(
            out,
            c,
            &format!("{p}_wo"),
            &format!("{p}_attn_output_weight"),
            &format!("{n}.attn_output.weight"),
            &format!("{h}x{q}"),
        );
        if c.has_qk_norm {
            emit_load_compute(
                out,
                c,
                &format!("{p}_qnw"),
                &format!("{p}_attn_q_norm_weight"),
                &format!("{n}.attn_q_norm.weight"),
                &format!("{d}"),
            );
            emit_load_compute(
                out,
                c,
                &format!("{p}_knw"),
                &format!("{p}_attn_k_norm_weight"),
                &format!("{n}.attn_k_norm.weight"),
                &format!("{d}"),
            );
        }
        emit_load_compute(
            out,
            c,
            &format!("{p}_ffn_nw"),
            &format!("{p}_ffn_norm_weight"),
            &format!("{n}.ffn_norm.weight"),
            &format!("{h}"),
        );
        emit_load_compute(
            out,
            c,
            &format!("{p}_wgate"),
            &format!("{p}_ffn_gate_weight"),
            &format!("{n}.ffn_gate.weight"),
            &format!("{}x{h}", c.intermediate),
        );
        emit_load_compute(
            out,
            c,
            &format!("{p}_wup"),
            &format!("{p}_ffn_up_weight"),
            &format!("{n}.ffn_up.weight"),
            &format!("{}x{h}", c.intermediate),
        );
        emit_load_compute(
            out,
            c,
            &format!("{p}_wdown"),
            &format!("{p}_ffn_down_weight"),
            &format!("{n}.ffn_down.weight"),
            &format!("{h}x{}", c.intermediate),
        );

        let xin = h_name.clone();
        out.push_str(&format!(
            "  %{p}_xn = func.call @rms_norm(%{xin}, %{p}_attn_nw) : (tensor<{s}x{h}xf32>, tensor<{h}xf32>) -> tensor<{s}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_q = func.call @linear_hq(%{p}_xn, %{p}_wq) : (tensor<{s}x{h}xf32>, tensor<{q}x{h}xf32>) -> tensor<{s}x{q}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_k = func.call @linear_hkv(%{p}_xn, %{p}_wk) : (tensor<{s}x{h}xf32>, tensor<{kv}x{h}xf32>) -> tensor<{s}x{kv}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_v = func.call @linear_hkv(%{p}_xn, %{p}_wv) : (tensor<{s}x{h}xf32>, tensor<{kv}x{h}xf32>) -> tensor<{s}x{kv}xf32>\n"
        ));
        if c.has_qk_norm {
            out.push_str(&format!(
                "  %{p}_ctx = func.call @attn(%{p}_q, %{p}_k, %{p}_v, %{p}_qnw, %{p}_knw) : (tensor<{s}x{q}xf32>, tensor<{s}x{kv}xf32>, tensor<{s}x{kv}xf32>, tensor<{d}xf32>, tensor<{d}xf32>) -> tensor<{s}x{q}xf32>\n"
            ));
        } else {
            out.push_str(&format!(
                "  %{p}_ctx = func.call @attn(%{p}_q, %{p}_k, %{p}_v) : (tensor<{s}x{q}xf32>, tensor<{s}x{kv}xf32>, tensor<{s}x{kv}xf32>) -> tensor<{s}x{q}xf32>\n"
            ));
        }
        out.push_str(&format!(
            "  %{p}_o = func.call @linear_qh(%{p}_ctx, %{p}_wo) : (tensor<{s}x{q}xf32>, tensor<{h}x{q}xf32>) -> tensor<{s}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_h2 = arith.addf %{xin}, %{p}_o : tensor<{s}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_fn = func.call @rms_norm(%{p}_h2, %{p}_ffn_nw) : (tensor<{s}x{h}xf32>, tensor<{h}xf32>) -> tensor<{s}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_gate = func.call @linear_hi(%{p}_fn, %{p}_wgate) : (tensor<{s}x{h}xf32>, tensor<{i}x{h}xf32>) -> tensor<{s}x{i}xf32>\n",
            i = c.intermediate
        ));
        out.push_str(&format!(
            "  %{p}_up = func.call @linear_hi(%{p}_fn, %{p}_wup) : (tensor<{s}x{h}xf32>, tensor<{i}x{h}xf32>) -> tensor<{s}x{i}xf32>\n",
            i = c.intermediate
        ));
        out.push_str(&format!(
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
        out.push_str(&format!(
            "  %{p}_ff = arith.mulf %{p}_silu, %{p}_up : tensor<{s}x{i}xf32>\n",
            i = c.intermediate
        ));
        out.push_str(&format!(
            "  %{p}_down = func.call @linear_ih(%{p}_ff, %{p}_wdown) : (tensor<{s}x{i}xf32>, tensor<{h}x{i}xf32>) -> tensor<{s}x{h}xf32>\n",
            i = c.intermediate
        ));
        out.push_str(&format!(
            "  %{p}_hout = arith.addf %{p}_h2, %{p}_down : tensor<{s}x{h}xf32>\n"
        ));
        h_name = format!("{p}_hout");
    }

    emit_load_compute(
        out,
        c,
        "out_nw",
        "output_norm_weight",
        "output_norm.weight",
        &format!("{h}"),
    );
    emit_load_compute(
        out,
        c,
        "wout",
        "output_weight",
        "output.weight",
        &format!("{v}x{h}"),
    );
    out.push_str("  %last_i64 = tensor.extract %last[] : tensor<i64>\n");
    out.push_str("  %li = arith.index_cast %last_i64 : i64 to index\n");
    out.push_str(&format!(
        "  %last_row = tensor.extract_slice %{h_name}[%li, 0] [1, {h}] [1, 1] : tensor<{s}x{h}xf32> to tensor<1x{h}xf32>\n"
    ));
    // Tile to S for rms_norm helper, take row 0.
    out.push_str(&format!(
        "  %last_tile = tensor.empty() : tensor<{s}x{h}xf32>\n"
    ));
    out.push_str(&format!(
        "  %last_s = tensor.insert_slice %last_row into %last_tile[0, 0] [1, {h}] [1, 1] : tensor<1x{h}xf32> into tensor<{s}x{h}xf32>\n"
    ));
    out.push_str(&format!(
        "  %ln = func.call @rms_norm(%last_s, %out_nw) : (tensor<{s}x{h}xf32>, tensor<{h}xf32>) -> tensor<{s}x{h}xf32>\n"
    ));
    out.push_str(&format!(
        "  %ln1 = tensor.extract_slice %ln[0, 0] [1, {h}] [1, 1] : tensor<{s}x{h}xf32> to tensor<1x{h}xf32>\n"
    ));
    out.push_str("  %c0f = arith.constant 0.0 : f32\n");
    out.push_str(&format!(
        "  %wt_i = tensor.empty() : tensor<{h}x{v}xf32>\n"
    ));
    out.push_str(&format!(
        "  %wt = linalg.transpose ins(%wout : tensor<{v}x{h}xf32>) outs(%wt_i : tensor<{h}x{v}xf32>) permutation = [1, 0]\n"
    ));
    out.push_str(&format!(
        "  %yi = tensor.empty() : tensor<1x{v}xf32>\n"
    ));
    out.push_str(&format!(
        "  %yz = linalg.fill ins(%c0f : f32) outs(%yi : tensor<1x{v}xf32>) -> tensor<1x{v}xf32>\n"
    ));
    out.push_str(&format!(
        "  %y = linalg.matmul ins(%ln1, %wt : tensor<1x{h}xf32>, tensor<{h}x{v}xf32>) outs(%yz : tensor<1x{v}xf32>) -> tensor<1x{v}xf32>\n"
    ));
    out.push_str(&format!(
        "  %logits = tensor.collapse_shape %y [[0, 1]] : tensor<1x{v}xf32> into tensor<{v}xf32>\n"
    ));
    out.push_str(&format!("  return %logits : tensor<{v}xf32>\n}}\n\n"));
}

fn emit_decode(out: &mut String, c: &DenseDecoderConfig) {
    let (s, v) = (c.seq, c.vocab);
    out.push_str(&format!(
        r#"func.func @decode(%token: tensor<i64>) -> tensor<{v}xf32> {{
  %pad = arith.constant dense<0> : tensor<{s}xi64>
  %tok = tensor.extract %token[] : tensor<i64>
  %c0 = arith.constant 0 : index
  %tokens = tensor.insert %tok into %pad[%c0] : tensor<{s}xi64>
  %last = arith.constant dense<0> : tensor<i64>
  %logits = func.call @prefill(%tokens, %last) : (tensor<{s}xi64>, tensor<i64>) -> tensor<{v}xf32>
  return %logits : tensor<{v}xf32>
}}
"#,
        s = s,
        v = v,
    ));
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
            rms_norm_eps: 1e-5,
            rope_theta: None,
            has_qk_norm: false,
            param_keys: BTreeMap::new(),
            param_dtypes,
        };
        let mlir = emit_dense_decoder_cfg("test.decoder", &c);
        assert!(mlir.contains("weight_dtypes=[bf16,f16,f32]"));
        assert!(mlir.contains(
            "util.global private @token_embd_weight = #stream.parameter.named<\"weights\"::\"token_embd.weight\"> : tensor<32x64xbf16>"
        ));
        assert!(mlir.contains(
            "util.global private @blk0_attn_norm_weight = #stream.parameter.named<\"weights\"::\"blk.0.attn_norm.weight\"> : tensor<64xf16>"
        ));
        assert!(mlir.contains(
            "util.global private @blk0_attn_q_weight = #stream.parameter.named<\"weights\"::\"blk.0.attn_q.weight\"> : tensor<64x64xf32>"
        ));
        assert!(mlir.contains("arith.extf %emb_t_native : tensor<32x64xbf16> to tensor<32x64xf32>"));
        assert!(mlir.contains(
            "arith.extf %blk0_attn_nw_native : tensor<64xf16> to tensor<64xf32>"
        ));
    }
}
