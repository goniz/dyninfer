//! Dense-decoder helper kernels built with [`dyninfer_mlir::FuncBuilder`].

use dyninfer_error::Result;
use dyninfer_mlir::{FuncBuilder, ModuleBuilder, Value};

/// `func.func private @name(%x, %w) -> …` matmul-after-transpose linear.
pub fn emit_linear(
    module: &mut ModuleBuilder,
    name: &str,
    s: u32,
    in_dim: u32,
    out_dim: u32,
    // Weight layout is `out × in` (same as current emitter).
    weight_out_by_in: bool,
) -> Result<()> {
    let (w_rows, w_cols) = if weight_out_by_in {
        (out_dim, in_dim)
    } else {
        (in_dim, out_dim)
    };
    let x_ty = format!("tensor<{s}x{in_dim}xf32>");
    let w_ty = format!("tensor<{w_rows}x{w_cols}xf32>");
    let y_ty = format!("tensor<{s}x{out_dim}xf32>");
    let wt_ty = format!("tensor<{in_dim}x{out_dim}xf32>");

    let mut f = module.func_private(name);
    let x = f.arg("x", &x_ty);
    let w = f.arg("w", &w_ty);
    f.result_ty(&y_ty);
    let c0 = f.constant_f32("0.0");
    let ti = f.tensor_empty(&wt_ty);
    let wt = f.linalg_transpose(&w, &ti, "[1, 0]", &w_ty, &wt_ty);
    let init = f.tensor_empty(&y_ty);
    let z = f.linalg_fill(&c0, &init, &y_ty);
    let y = f.linalg_matmul(&x, &wt, &z, &x_ty, &wt_ty, &y_ty);
    f.ret_ty(&[&y], &y_ty);
    f.finish(module)
}

pub fn emit_rms_norm(
    module: &mut ModuleBuilder,
    name: &str,
    s: u32,
    h: u32,
    eps: f32,
) -> Result<()> {
    let x_ty = format!("tensor<{s}x{h}xf32>");
    let w_ty = format!("tensor<{h}xf32>");
    let red_ty = format!("tensor<{s}xf32>");
    let eps_lit = format!("{eps:.8e}");
    let h_f = format!("{:.1}", h as f32);

    let mut f = module.func_private(name);
    let x = f.arg("x", &x_ty);
    let w = f.arg("w", &w_ty);
    f.result_ty(&x_ty);
    // Region-bearing ops stay as asm fragments inside the FuncBuilder.
    f.op_asm(format!(
        r#"  %c0 = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %eps = arith.constant {eps_lit} : f32
  %ch = arith.constant {h_f} : f32
  %sq = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}}
    ins({x} : {x_ty}) outs({x} : {x_ty}) {{
    ^bb0(%a: f32, %b: f32):
      %p = arith.mulf %a, %a : f32
      linalg.yield %p : f32
  }} -> {x_ty}
  %init = tensor.empty() : {red_ty}
  %z = linalg.fill ins(%c0 : f32) outs(%init : {red_ty}) -> {red_ty}
  %ms = linalg.reduce ins(%sq : {x_ty}) outs(%z : {red_ty}) dimensions = [1]
    (%in: f32, %acc: f32) {{
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }}
  %inv = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}}
    ins(%ms : {red_ty}) outs(%ms : {red_ty}) {{
    ^bb0(%a: f32, %b: f32):
      %m = arith.divf %a, %ch : f32
      %meps = arith.addf %m, %eps : f32
      %root = math.sqrt %meps : f32
      %i = arith.divf %one, %root : f32
      linalg.yield %i : f32
  }} -> {red_ty}
  %y = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}}
    ins({x}, %inv, {w} : {x_ty}, {red_ty}, {w_ty}) outs({x} : {x_ty}) {{
    ^bb0(%a: f32, %i: f32, %ww: f32, %o: f32):
      %t = arith.mulf %a, %i : f32
      %r = arith.mulf %t, %ww : f32
      linalg.yield %r : f32
  }} -> {x_ty}
  return %y : {x_ty}"#
    ));
    f.finish(module)
}

pub fn emit_rms_norm_heads(
    module: &mut ModuleBuilder,
    name: &str,
    s: u32,
    nh: u32,
    d: u32,
    eps: f32,
) -> Result<()> {
    let x_ty = format!("tensor<{s}x{nh}x{d}xf32>");
    let w_ty = format!("tensor<{d}xf32>");
    let red_ty = format!("tensor<{s}x{nh}xf32>");
    let eps_lit = format!("{eps:.8e}");
    let d_f = format!("{:.1}", d as f32);

    let mut f = module.func_private(name);
    let x = f.arg("x", &x_ty);
    let w = f.arg("w", &w_ty);
    f.result_ty(&x_ty);
    f.op_asm(format!(
        r#"  %c0 = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %eps = arith.constant {eps_lit} : f32
  %cd = arith.constant {d_f} : f32
  %sq = linalg.generic {{
      indexing_maps = [affine_map<(p, h, dim) -> (p, h, dim)>, affine_map<(p, h, dim) -> (p, h, dim)>],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins({x} : {x_ty}) outs({x} : {x_ty}) {{
    ^bb0(%a: f32, %b: f32):
      %p = arith.mulf %a, %a : f32
      linalg.yield %p : f32
  }} -> {x_ty}
  %init = tensor.empty() : {red_ty}
  %z = linalg.fill ins(%c0 : f32) outs(%init : {red_ty}) -> {red_ty}
  %ms = linalg.reduce ins(%sq : {x_ty}) outs(%z : {red_ty}) dimensions = [2]
    (%in: f32, %acc: f32) {{
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }}
  %inv = linalg.generic {{
      indexing_maps = [affine_map<(p, h) -> (p, h)>, affine_map<(p, h) -> (p, h)>],
      iterator_types = ["parallel", "parallel"]}}
    ins(%ms : {red_ty}) outs(%ms : {red_ty}) {{
    ^bb0(%a: f32, %b: f32):
      %m = arith.divf %a, %cd : f32
      %meps = arith.addf %m, %eps : f32
      %root = math.sqrt %meps : f32
      %i = arith.divf %one, %root : f32
      linalg.yield %i : f32
  }} -> {red_ty}
  %y = linalg.generic {{
      indexing_maps = [
        affine_map<(p, h, dim) -> (p, h, dim)>,
        affine_map<(p, h, dim) -> (p, h)>,
        affine_map<(p, h, dim) -> (dim)>,
        affine_map<(p, h, dim) -> (p, h, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins({x}, %inv, {w} : {x_ty}, {red_ty}, {w_ty}) outs({x} : {x_ty}) {{
    ^bb0(%a: f32, %i: f32, %ww: f32, %o: f32):
      %t = arith.mulf %a, %i : f32
      %r = arith.mulf %t, %ww : f32
      linalg.yield %r : f32
  }} -> {x_ty}
  return %y : {x_ty}"#
    ));
    f.finish(module)
}

pub fn emit_repeat_kv(
    module: &mut ModuleBuilder,
    name: &str,
    s: u32,
    nkv: u32,
    nh: u32,
    d: u32,
    g: u32,
) -> Result<()> {
    let in_ty = format!("tensor<{s}x{nkv}x{d}xf32>");
    let out_ty = format!("tensor<{s}x{nh}x{d}xf32>");
    let mut f = module.func_private(name);
    let x = f.arg("x", &in_ty);
    f.result_ty(&out_ty);
    f.op_asm(format!(
        r#"  %init = tensor.empty() : {out_ty}
  %y = linalg.generic {{
      indexing_maps = [
        affine_map<(p, h, dim) -> (p, h floordiv {g}, dim)>,
        affine_map<(p, h, dim) -> (p, h, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins({x} : {in_ty}) outs(%init : {out_ty}) {{
    ^bb0(%a: f32, %o: f32):
      linalg.yield %a : f32
  }} -> {out_ty}
  return %y : {out_ty}"#
    ));
    f.finish(module)
}

pub fn emit_rope(
    module: &mut ModuleBuilder,
    name: &str,
    s: u32,
    nh: u32,
    d: u32,
    theta: f32,
) -> Result<()> {
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
    let x_ty = format!("tensor<{s}x{nh}x{d}xf32>");
    let half_ty = format!("tensor<{s}x{half}xf32>");

    let mut f = module.func_private(name);
    let x = f.arg("x", &x_ty);
    f.result_ty(&x_ty);
    f.op_asm(format!(
        r#"  %cos = arith.constant dense<[{cos_lit}]> : {half_ty}
  %sin = arith.constant dense<[{sin_lit}]> : {half_ty}
  %init = tensor.empty() : {x_ty}
  %y = linalg.generic {{
      indexing_maps = [
        affine_map<(p, h, dim) -> (p, h, dim)>,
        affine_map<(p, h, dim) -> (p, h, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins({x} : {x_ty}) outs(%init : {x_ty}) {{
    ^bb0(%a: f32, %o: f32):
      %p = linalg.index 0 : index
      %hh = linalg.index 1 : index
      %dim = linalg.index 2 : index
      %half_i = arith.constant {half} : index
      %in_first = arith.cmpi ult, %dim, %half_i : index
      %pair_lo = arith.remui %dim, %half_i : index
      %dim_hi = arith.addi %pair_lo, %half_i : index
      %x1 = tensor.extract {x}[%p, %hh, %pair_lo] : {x_ty}
      %x2 = tensor.extract {x}[%p, %hh, %dim_hi] : {x_ty}
      %cv = tensor.extract %cos[%p, %pair_lo] : {half_ty}
      %sv = tensor.extract %sin[%p, %pair_lo] : {half_ty}
      %x1c = arith.mulf %x1, %cv : f32
      %x2s = arith.mulf %x2, %sv : f32
      %x1s = arith.mulf %x1, %sv : f32
      %x2c = arith.mulf %x2, %cv : f32
      %lo = arith.subf %x1c, %x2s : f32
      %hi = arith.addf %x1s, %x2c : f32
      %r = arith.select %in_first, %lo, %hi : f32
      linalg.yield %r : f32
  }} -> {x_ty}
  return %y : {x_ty}"#
    ));
    f.finish(module)
}

/// RoPE at absolute position `pos` (decode path). Tables sized `[max_kv, D/2]`.
///
/// Takes `tensor<i64>` to match the `@decode` ABI (same as `pos_t` elsewhere).
pub fn emit_rope_at(
    module: &mut ModuleBuilder,
    name: &str,
    mk: u32,
    nh: u32,
    d: u32,
    theta: f32,
) -> Result<()> {
    let half = (d / 2) as usize;
    let mut cos_rows = Vec::with_capacity(mk as usize);
    let mut sin_rows = Vec::with_capacity(mk as usize);
    for pos in 0..mk {
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
    let x_ty = format!("tensor<1x{nh}x{d}xf32>");
    let half_ty = format!("tensor<{mk}x{half}xf32>");

    let mut f = module.func_private(name);
    let x = f.arg("x", &x_ty);
    let _pos_t = f.arg("pos_t", "tensor<i64>");
    f.result_ty(&x_ty);
    f.op_asm(format!(
        r#"  %cos = arith.constant dense<[{cos_lit}]> : {half_ty}
  %sin = arith.constant dense<[{sin_lit}]> : {half_ty}
  %pos_i64 = tensor.extract %pos_t[] : tensor<i64>
  %pos = arith.index_cast %pos_i64 : i64 to index
  %init = tensor.empty() : {x_ty}
  %y = linalg.generic {{
      indexing_maps = [
        affine_map<(p, h, dim) -> (p, h, dim)>,
        affine_map<(p, h, dim) -> (p, h, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins({x} : {x_ty}) outs(%init : {x_ty}) {{
    ^bb0(%a: f32, %o: f32):
      %c0i = arith.constant 0 : index
      %hh = linalg.index 1 : index
      %dim = linalg.index 2 : index
      %half_i = arith.constant {half} : index
      %in_first = arith.cmpi ult, %dim, %half_i : index
      %pair_lo = arith.remui %dim, %half_i : index
      %dim_hi = arith.addi %pair_lo, %half_i : index
      %x1 = tensor.extract {x}[%c0i, %hh, %pair_lo] : {x_ty}
      %x2 = tensor.extract {x}[%c0i, %hh, %dim_hi] : {x_ty}
      %cv = tensor.extract %cos[%pos, %pair_lo] : {half_ty}
      %sv = tensor.extract %sin[%pos, %pair_lo] : {half_ty}
      %x1c = arith.mulf %x1, %cv : f32
      %x2s = arith.mulf %x2, %sv : f32
      %x1s = arith.mulf %x1, %sv : f32
      %x2c = arith.mulf %x2, %cv : f32
      %lo = arith.subf %x1c, %x2s : f32
      %hi = arith.addf %x1s, %x2c : f32
      %r = arith.select %in_first, %lo, %hi : f32
      linalg.yield %r : f32
  }} -> {x_ty}
  return %y : {x_ty}"#
    ));
    f.finish(module)
}

pub fn emit_add_smoke(module: &mut ModuleBuilder) -> Result<()> {
    let mut f = module.func("add");
    let a = f.arg("a", "tensor<4xf32>");
    let b = f.arg("b", "tensor<4xf32>");
    f.result_ty("tensor<4xf32>");
    let y = f.addf(&a, &b, "tensor<4xf32>");
    f.ret_ty(&[&y], "tensor<4xf32>");
    f.finish(module)
}

/// Load a weight global, casting to f32 compute when needed.
pub fn load_compute(
    f: &mut FuncBuilder,
    ssa: &str,
    sym: &str,
    storage_ty: &str,
    compute_ty: &str,
    shape: &str,
) -> Value {
    let storage = format!("tensor<{shape}x{storage_ty}>");
    let compute = format!("tensor<{shape}x{compute_ty}>");
    if storage_ty == compute_ty {
        f.global_load_as(ssa, sym, &compute)
    } else {
        let native = f.global_load_as(&format!("{ssa}_native"), sym, &storage);
        f.extf_as(ssa, &native, &storage, &compute)
    }
}
