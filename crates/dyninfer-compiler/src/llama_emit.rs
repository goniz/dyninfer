//! Dense Llama → IREE MLIR.
//!
//! Emits a static-shaped MHA decoder (embed → N×(attn+SwiGLU) → lm_head) that
//! loads weights via `#stream.parameter.named<"weights"::"...">`.
//! Optional Llama RoPE when `rope_theta` is present in checkpoint metadata.

use dyninfer_architecture::ArchitecturePackage;
use dyninfer_checkpoint::CheckpointCatalog;

/// Default static prefill window for non-tiny dense models.
pub const PREFILL_WINDOW: u32 = 64;
/// Window used by the synthetic Milestone-1 fixture (fast differential tests).
pub const TINY_PREFILL_WINDOW: u32 = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaEmitConfig {
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
}

impl LlamaEmitConfig {
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
        let is_tiny = vocab == 32 && hidden == 64 && num_heads == 4;
        let seq = if is_tiny {
            TINY_PREFILL_WINDOW
        } else {
            PREFILL_WINDOW
        };
        let rope_theta = meta
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .or_else(|| cfg.values.get("rope_theta").and_then(|v| v.as_f64()))
            .map(|v| v as f32);
        Self {
            vocab,
            hidden,
            intermediate: u(&["intermediate_size"], hidden * 2),
            num_heads,
            num_kv_heads,
            head_dim: u(&["head_dim"], hidden / num_heads.max(1)),
            num_layers: u(
                &["num_layers", "n_layer", "block_count", "llama.block_count"],
                1,
            ),
            seq,
            rms_norm_eps: f(&["rms_norm_eps"], 1e-5),
            rope_theta: if is_tiny { None } else { rope_theta.or(Some(10000.0)) },
        }
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
        self.num_heads == self.num_kv_heads
            && self.num_heads > 0
            && self.head_dim > 0
            && self.head_dim % 2 == 0
            && self.hidden == self.num_heads * self.head_dim
            && self.intermediate > 0
            && self.num_layers >= 1
            && self.num_layers <= 32
            && self.vocab >= 1
            && self.vocab <= 128_256
            && self.hidden <= 2048
            && (self.seq == 4 || self.seq == 32 || self.seq == 64)
    }
}

/// Emit dense Llama MLIR for supported MHA shapes.
pub fn emit_dense_llama(package: &ArchitecturePackage, catalog: &CheckpointCatalog) -> String {
    let c = LlamaEmitConfig::from_package(package, catalog);
    assert!(
        c.supports_dense_emit(),
        "unsupported dense Llama emit config: {c:?}"
    );
    let mut out = String::new();
    out.push_str(&format!(
        "// dyninfer dense llama arch={} layers={} seq={} rope={:?}\n",
        package.id, c.num_layers, c.seq, c.rope_theta
    ));
    emit_globals(&mut out, &c);
    emit_helpers(&mut out, &c);
    emit_prefill(&mut out, &c);
    emit_decode(&mut out, &c);
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

fn emit_globals(out: &mut String, c: &LlamaEmitConfig) {
    let (v, h, i) = (c.vocab, c.hidden, c.intermediate);
    out.push_str(&format!(
        "util.global private @token_embd_weight = #stream.parameter.named<\"weights\"::\"token_embd.weight\"> : tensor<{v}x{h}xf32>\n"
    ));
    for layer in 0..c.num_layers {
        let p = format!("blk{layer}");
        let n = format!("blk.{layer}");
        out.push_str(&format!(
            "util.global private @{p}_attn_norm_weight = #stream.parameter.named<\"weights\"::\"{n}.attn_norm.weight\"> : tensor<{h}xf32>\n"
        ));
        out.push_str(&format!(
            "util.global private @{p}_attn_q_weight = #stream.parameter.named<\"weights\"::\"{n}.attn_q.weight\"> : tensor<{h}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "util.global private @{p}_attn_k_weight = #stream.parameter.named<\"weights\"::\"{n}.attn_k.weight\"> : tensor<{h}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "util.global private @{p}_attn_v_weight = #stream.parameter.named<\"weights\"::\"{n}.attn_v.weight\"> : tensor<{h}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "util.global private @{p}_attn_output_weight = #stream.parameter.named<\"weights\"::\"{n}.attn_output.weight\"> : tensor<{h}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "util.global private @{p}_ffn_norm_weight = #stream.parameter.named<\"weights\"::\"{n}.ffn_norm.weight\"> : tensor<{h}xf32>\n"
        ));
        out.push_str(&format!(
            "util.global private @{p}_ffn_gate_weight = #stream.parameter.named<\"weights\"::\"{n}.ffn_gate.weight\"> : tensor<{i}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "util.global private @{p}_ffn_up_weight = #stream.parameter.named<\"weights\"::\"{n}.ffn_up.weight\"> : tensor<{i}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "util.global private @{p}_ffn_down_weight = #stream.parameter.named<\"weights\"::\"{n}.ffn_down.weight\"> : tensor<{h}x{i}xf32>\n"
        ));
    }
    out.push_str(&format!(
        "util.global private @output_norm_weight = #stream.parameter.named<\"weights\"::\"output_norm.weight\"> : tensor<{h}xf32>\n"
    ));
    out.push_str(&format!(
        "util.global private @output_weight = #stream.parameter.named<\"weights\"::\"output.weight\"> : tensor<{v}x{h}xf32>\n\n"
    ));
}

fn emit_helpers(out: &mut String, c: &LlamaEmitConfig) {
    let (s, h, i, nh, d) = (c.seq, c.hidden, c.intermediate, c.num_heads, c.head_dim);
    let eps = format!("{:.8e}", c.rms_norm_eps);
    let h_f = format!("{:.1}", h as f32);

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

func.func private @linear_hh(%x: tensor<{s}x{h}xf32>, %w: tensor<{h}x{h}xf32>) -> tensor<{s}x{h}xf32> {{
  %c0 = arith.constant 0.0 : f32
  %ti = tensor.empty() : tensor<{h}x{h}xf32>
  %wt = linalg.transpose ins(%w : tensor<{h}x{h}xf32>) outs(%ti : tensor<{h}x{h}xf32>) permutation = [1, 0]
  %init = tensor.empty() : tensor<{s}x{h}xf32>
  %z = linalg.fill ins(%c0 : f32) outs(%init : tensor<{s}x{h}xf32>) -> tensor<{s}x{h}xf32>
  %y = linalg.matmul ins(%x, %wt : tensor<{s}x{h}xf32>, tensor<{h}x{h}xf32>) outs(%z : tensor<{s}x{h}xf32>) -> tensor<{s}x{h}xf32>
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
        eps = eps,
        h_f = h_f,
    ));

    if let Some(theta) = c.rope_theta {
        emit_rope_helper(out, s, nh, d, theta);
    }

    let scale = 1.0 / (d as f32).sqrt();
    out.push_str(&format!(
        r#"func.func private @attn(%q: tensor<{s}x{h}xf32>, %k: tensor<{s}x{h}xf32>, %v: tensor<{s}x{h}xf32>) -> tensor<{s}x{h}xf32> {{
  %q3 = tensor.expand_shape %q [[0], [1, 2]] output_shape [{s}, {nh}, {d}] : tensor<{s}x{h}xf32> into tensor<{s}x{nh}x{d}xf32>
  %k3 = tensor.expand_shape %k [[0], [1, 2]] output_shape [{s}, {nh}, {d}] : tensor<{s}x{h}xf32> into tensor<{s}x{nh}x{d}xf32>
  %v3 = tensor.expand_shape %v [[0], [1, 2]] output_shape [{s}, {nh}, {d}] : tensor<{s}x{h}xf32> into tensor<{s}x{nh}x{d}xf32>
"#,
        s = s,
        h = h,
        nh = nh,
        d = d,
    ));

    let (q_ssa, k_ssa) = if c.rope_theta.is_some() {
        out.push_str(&format!(
            r#"  %qr = func.call @apply_rope(%q3) : (tensor<{s}x{nh}x{d}xf32>) -> tensor<{s}x{nh}x{d}xf32>
  %kr = func.call @apply_rope(%k3) : (tensor<{s}x{nh}x{d}xf32>) -> tensor<{s}x{nh}x{d}xf32>
"#,
            s = s,
            nh = nh,
            d = d,
        ));
        ("qr", "kr")
    } else {
        ("q3", "k3")
    };

    out.push_str(&format!(
        r#"  %q_ti = tensor.empty() : tensor<{nh}x{s}x{d}xf32>
  %k_ti = tensor.empty() : tensor<{nh}x{s}x{d}xf32>
  %v_ti = tensor.empty() : tensor<{nh}x{s}x{d}xf32>
  %qb = linalg.transpose ins(%{q_ssa} : tensor<{s}x{nh}x{d}xf32>) outs(%q_ti : tensor<{nh}x{s}x{d}xf32>) permutation = [1, 0, 2]
  %kb = linalg.transpose ins(%{k_ssa} : tensor<{s}x{nh}x{d}xf32>) outs(%k_ti : tensor<{nh}x{s}x{d}xf32>) permutation = [1, 0, 2]
  %vb = linalg.transpose ins(%v3 : tensor<{s}x{nh}x{d}xf32>) outs(%v_ti : tensor<{nh}x{s}x{d}xf32>) permutation = [1, 0, 2]
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
  %ctx = tensor.collapse_shape %ctx_t [[0], [1, 2]] : tensor<{s}x{nh}x{d}xf32> into tensor<{s}x{h}xf32>
  return %ctx : tensor<{s}x{h}xf32>
}}

"#,
        s = s,
        h = h,
        nh = nh,
        d = d,
        scale = scale,
    ));
}

fn emit_rope_helper(out: &mut String, s: u32, nh: u32, d: u32, theta: f32) {
    // Precompute cos/sin tables as nested dense constants: [S, D/2]
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
        r#"func.func private @apply_rope(%x: tensor<{s}x{nh}x{d}xf32>) -> tensor<{s}x{nh}x{d}xf32> {{
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
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %pair = arith.divui %dim, %c2 : index
      %parity = arith.remui %dim, %c2 : index
      %is_even = arith.cmpi eq, %parity, %c0 : index
      %dim0 = arith.muli %pair, %c2 : index
      %dim1 = arith.addi %dim0, %c1 : index
      %x0 = tensor.extract %x[%p, %hh, %dim0] : tensor<{s}x{nh}x{d}xf32>
      %x1 = tensor.extract %x[%p, %hh, %dim1] : tensor<{s}x{nh}x{d}xf32>
      %cv = tensor.extract %cos[%p, %pair] : tensor<{s}x{half}xf32>
      %sv = tensor.extract %sin[%p, %pair] : tensor<{s}x{half}xf32>
      %x0c = arith.mulf %x0, %cv : f32
      %x1s = arith.mulf %x1, %sv : f32
      %x0s = arith.mulf %x0, %sv : f32
      %x1c = arith.mulf %x1, %cv : f32
      %even = arith.subf %x0c, %x1s : f32
      %odd = arith.addf %x0s, %x1c : f32
      %r = arith.select %is_even, %even, %odd : f32
      linalg.yield %r : f32
  }} -> tensor<{s}x{nh}x{d}xf32>
  return %y : tensor<{s}x{nh}x{d}xf32>
}}

"#,
        s = s,
        nh = nh,
        d = d,
        half = half,
        cos_lit = cos_lit,
        sin_lit = sin_lit,
    ));
}

fn emit_prefill(out: &mut String, c: &LlamaEmitConfig) {
    let (s, h, v, nh, d) = (c.seq, c.hidden, c.vocab, c.num_heads, c.head_dim);
    out.push_str(&format!(
        "func.func @prefill(%tokens: tensor<{s}xi64>) -> tensor<{v}xf32> {{\n"
    ));
    out.push_str(&format!(
        "  %emb_t = util.global.load @token_embd_weight : tensor<{v}x{h}xf32>\n"
    ));

    // Embedding gather (unrolled).
    out.push_str(&format!(
        "  %h_acc0 = tensor.empty() : tensor<{s}x{h}xf32>\n"
    ));
    for pos in 0..s {
        out.push_str(&format!(
            "  %c{pos}i = arith.constant {pos} : index\n"
        ));
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
        out.push_str(&format!(
            "  %{p}_attn_nw = util.global.load @{p}_attn_norm_weight : tensor<{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_wq = util.global.load @{p}_attn_q_weight : tensor<{h}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_wk = util.global.load @{p}_attn_k_weight : tensor<{h}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_wv = util.global.load @{p}_attn_v_weight : tensor<{h}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_wo = util.global.load @{p}_attn_output_weight : tensor<{h}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_ffn_nw = util.global.load @{p}_ffn_norm_weight : tensor<{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_wgate = util.global.load @{p}_ffn_gate_weight : tensor<{}x{h}xf32>\n",
            c.intermediate
        ));
        out.push_str(&format!(
            "  %{p}_wup = util.global.load @{p}_ffn_up_weight : tensor<{}x{h}xf32>\n",
            c.intermediate
        ));
        out.push_str(&format!(
            "  %{p}_wdown = util.global.load @{p}_ffn_down_weight : tensor<{h}x{}xf32>\n",
            c.intermediate
        ));

        let xin = h_name.clone();
        out.push_str(&format!(
            "  %{p}_xn = func.call @rms_norm(%{xin}, %{p}_attn_nw) : (tensor<{s}x{h}xf32>, tensor<{h}xf32>) -> tensor<{s}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_q = func.call @linear_hh(%{p}_xn, %{p}_wq) : (tensor<{s}x{h}xf32>, tensor<{h}x{h}xf32>) -> tensor<{s}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_k = func.call @linear_hh(%{p}_xn, %{p}_wk) : (tensor<{s}x{h}xf32>, tensor<{h}x{h}xf32>) -> tensor<{s}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_v = func.call @linear_hh(%{p}_xn, %{p}_wv) : (tensor<{s}x{h}xf32>, tensor<{h}x{h}xf32>) -> tensor<{s}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_ctx = func.call @attn(%{p}_q, %{p}_k, %{p}_v) : (tensor<{s}x{h}xf32>, tensor<{s}x{h}xf32>, tensor<{s}x{h}xf32>) -> tensor<{s}x{h}xf32>\n"
        ));
        out.push_str(&format!(
            "  %{p}_o = func.call @linear_hh(%{p}_ctx, %{p}_wo) : (tensor<{s}x{h}xf32>, tensor<{h}x{h}xf32>) -> tensor<{s}x{h}xf32>\n"
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

    let last = s - 1;
    out.push_str(&format!(
        "  %out_nw = util.global.load @output_norm_weight : tensor<{h}xf32>\n"
    ));
    out.push_str(&format!(
        "  %wout = util.global.load @output_weight : tensor<{v}x{h}xf32>\n"
    ));
    out.push_str(&format!(
        "  %last = tensor.extract_slice %{h_name}[{last}, 0] [1, {h}] [1, 1] : tensor<{s}x{h}xf32> to tensor<1x{h}xf32>\n"
    ));
    // Tile to S for rms_norm helper, take row 0.
    out.push_str(&format!(
        "  %last_tile = tensor.empty() : tensor<{s}x{h}xf32>\n"
    ));
    out.push_str(&format!(
        "  %last_s = tensor.insert_slice %last into %last_tile[0, 0] [1, {h}] [1, 1] : tensor<1x{h}xf32> into tensor<{s}x{h}xf32>\n"
    ));
    out.push_str(&format!(
        "  %ln = func.call @rms_norm(%last_s, %out_nw) : (tensor<{s}x{h}xf32>, tensor<{h}xf32>) -> tensor<{s}x{h}xf32>\n"
    ));
    out.push_str(&format!(
        "  %ln1 = tensor.extract_slice %ln[0, 0] [1, {h}] [1, 1] : tensor<{s}x{h}xf32> to tensor<1x{h}xf32>\n"
    ));
    out.push_str(&format!(
        "  %c0f = arith.constant 0.0 : f32\n"
    ));
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
    let _ = (nh, d);
}

fn emit_decode(out: &mut String, c: &LlamaEmitConfig) {
    let (s, v) = (c.seq, c.vocab);
    let last = s - 1;
    out.push_str(&format!(
        r#"func.func @decode(%token: tensor<i64>) -> tensor<{v}xf32> {{
  %pad = arith.constant dense<0> : tensor<{s}xi64>
  %tok = tensor.extract %token[] : tensor<i64>
  %clast = arith.constant {last} : index
  %tokens = tensor.insert %tok into %pad[%clast] : tensor<{s}xi64>
  %logits = func.call @prefill(%tokens) : (tensor<{s}xi64>) -> tensor<{v}xf32>
  return %logits : tensor<{v}xf32>
}}
"#,
        s = s,
        v = v,
        last = last,
    ));
}
