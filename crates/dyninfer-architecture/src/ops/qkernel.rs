//! Portable `dyninfer.qkernel` lowering for GGUF Q4_0 (Rust-emitted MLIR).
//!
//! Spec path: `linear` → `qkernel.quantized_matmul` → block unpack + matmul.
//! Device path binds packed `tensor<Nxi8>` weights and fuses unpack into the
//! reduction (no full f32 weight materialization).

#![allow(dead_code)] // emit_q4_0_dequant retained for materializing diagnostics

use dyninfer_error::Result;
use dyninfer_mlir::{FuncBuilder, ModuleBuilder, Value};

/// Bytes for a Q4_0 tensor with `rows * cols` logical elements.
pub fn q4_0_packed_len(rows: u32, cols: u32) -> u32 {
    let numel = rows.saturating_mul(cols);
    debug_assert!(numel.is_multiple_of(32));
    (numel / 32) * 18
}

/// Emit `@name(%packed: tensor<Nxi8>) -> tensor<RxCxf32>` Q4_0 dequant.
///
/// Block layout matches GGUF: `[f16 scale | 16×u8 nibbles]` per 32 weights.
/// Kept for tests / rare materializing call sites (e.g. diagnostics).
pub fn emit_q4_0_dequant(
    module: &mut ModuleBuilder,
    name: &str,
    rows: u32,
    cols: u32,
) -> Result<()> {
    let numel = rows * cols;
    assert!(
        numel.is_multiple_of(32),
        "Q4_0 dequant shape {rows}x{cols} numel not divisible by 32"
    );
    let nbytes = q4_0_packed_len(rows, cols);
    let nblocks = numel / 32;
    let packed_ty = format!("tensor<{nbytes}xi8>");
    let out_ty = format!("tensor<{rows}x{cols}xf32>");

    let mut f = module.func_private(name);
    let q = f.arg("q", &packed_ty);
    f.result_ty(&out_ty);
    f.op_asm(format!(
        r#"  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4_i32 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c8_i16 = arith.constant 8 : i16
  %c15_i32 = arith.constant 15 : i32
  %c32 = arith.constant 32 : index
  %c18 = arith.constant 18 : index
  %nblocks = arith.constant {nblocks} : index
  %cols = arith.constant {cols} : index
  %init = tensor.empty() : {out_ty}
  %out = scf.for %bi = %c0 to %nblocks step %c1 iter_args(%acc = %init) -> ({out_ty}) {{
    %boff = arith.muli %bi, %c18 : index
    %b0i = tensor.extract {q}[%boff] : {packed_ty}
    %b0 = arith.extui %b0i : i8 to i16
    %boff1 = arith.addi %boff, %c1 : index
    %b1i = tensor.extract {q}[%boff1] : {packed_ty}
    %b1 = arith.extui %b1i : i8 to i16
    %b1s = arith.shli %b1, %c8_i16 : i16
    %bits = arith.ori %b0, %b1s : i16
    %scale_f16 = arith.bitcast %bits : i16 to f16
    %scale = arith.extf %scale_f16 : f16 to f32
    %qs_base = arith.addi %boff, %c2 : index
    %acc2 = scf.for %j = %c0 to %c32 step %c1 iter_args(%a2 = %acc) -> ({out_ty}) {{
      %byte_i = arith.divui %j, %c2 : index
      %qoff = arith.addi %qs_base, %byte_i : index
      %qi = tensor.extract {q}[%qoff] : {packed_ty}
      %qu = arith.extui %qi : i8 to i32
      %odd = arith.remui %j, %c2 : index
      %is_odd = arith.cmpi eq, %odd, %c1 : index
      %hi = arith.shrui %qu, %c4_i32 : i32
      %lo = arith.andi %qu, %c15_i32 : i32
      %nibble = arith.select %is_odd, %hi, %lo : i32
      %qsigned = arith.subi %nibble, %c8 : i32
      %qf = arith.sitofp %qsigned : i32 to f32
      %w = arith.mulf %qf, %scale : f32
      %lin = arith.muli %bi, %c32 : index
      %elem = arith.addi %lin, %j : index
      %r = arith.divui %elem, %cols : index
      %c = arith.remui %elem, %cols : index
      %a3 = tensor.insert %w into %a2[%r, %c] : {out_ty}
      scf.yield %a3 : {out_ty}
    }}
    scf.yield %acc2 : {out_ty}
  }}
  return %out : {out_ty}"#
    ));
    f.finish(module)
}

/// Emit fused `@name(%x, %wq) -> y` Q4_0 unpack+dot.
///
/// Weight layout is logical `out × in` row-major Q4_0 (GGUF). Requires
/// `in_dim % 32 == 0` so each output row is an integer number of blocks.
pub fn emit_q4_0_linear(
    module: &mut ModuleBuilder,
    name: &str,
    s: u32,
    in_dim: u32,
    out_dim: u32,
) -> Result<()> {
    assert!(
        in_dim.is_multiple_of(32),
        "Q4_0 fused linear in_dim {in_dim} must be divisible by 32"
    );
    let nbytes = q4_0_packed_len(out_dim, in_dim);
    let x_ty = format!("tensor<{s}x{in_dim}xf32>");
    let q_ty = format!("tensor<{nbytes}xi8>");
    let y_ty = format!("tensor<{s}x{out_dim}xf32>");

    let mut f = module.func_private(name);
    let x = f.arg("x", &x_ty);
    let wq = f.arg("wq", &q_ty);
    f.result_ty(&y_ty);
    // linalg.generic reduction with on-the-fly Q4 unpack (IREE-friendly).
    f.op_asm(format!(
        r#"  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4_i32 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c8_i16 = arith.constant 8 : i16
  %c15_i32 = arith.constant 15 : i32
  %c18 = arith.constant 18 : index
  %c32 = arith.constant 32 : index
  %in_dim = arith.constant {in_dim} : index
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : {y_ty}
  %y0 = linalg.fill ins(%zero : f32) outs(%init : {y_ty}) -> {y_ty}
  %y = linalg.generic {{
      indexing_maps = [
        affine_map<(b, o, i) -> (b, i)>,
        affine_map<(b, o, i) -> (b, o)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]}}
    ins({x} : {x_ty}) outs(%y0 : {y_ty}) {{
    ^bb0(%xv: f32, %acc: f32):
      %o = linalg.index 1 : index
      %i = linalg.index 2 : index
      %lin = arith.muli %o, %in_dim : index
      %elem = arith.addi %lin, %i : index
      %bi = arith.divui %elem, %c32 : index
      %j = arith.remui %elem, %c32 : index
      %boff = arith.muli %bi, %c18 : index
      %b0i = tensor.extract {wq}[%boff] : {q_ty}
      %b0 = arith.extui %b0i : i8 to i16
      %boff1 = arith.addi %boff, %c1 : index
      %b1i = tensor.extract {wq}[%boff1] : {q_ty}
      %b1 = arith.extui %b1i : i8 to i16
      %b1s = arith.shli %b1, %c8_i16 : i16
      %bits = arith.ori %b0, %b1s : i16
      %scale_f16 = arith.bitcast %bits : i16 to f16
      %scale = arith.extf %scale_f16 : f16 to f32
      %qs_base = arith.addi %boff, %c2 : index
      %byte_i = arith.divui %j, %c2 : index
      %qoff = arith.addi %qs_base, %byte_i : index
      %qi = tensor.extract {wq}[%qoff] : {q_ty}
      %qu = arith.extui %qi : i8 to i32
      %odd = arith.remui %j, %c2 : index
      %is_odd = arith.cmpi eq, %odd, %c1 : index
      %hi = arith.shrui %qu, %c4_i32 : i32
      %lo = arith.andi %qu, %c15_i32 : i32
      %nibble = arith.select %is_odd, %hi, %lo : i32
      %qsigned = arith.subi %nibble, %c8 : i32
      %qf = arith.sitofp %qsigned : i32 to f32
      %w = arith.mulf %qf, %scale : f32
      %prod = arith.mulf %xv, %w : f32
      %sum = arith.addf %acc, %prod : f32
      linalg.yield %sum : f32
  }} -> {y_ty}
  return %y : {y_ty}"#
    ));
    f.finish(module)
}

/// Load a packed Q4_0 global as `tensor<Nxi8>`.
pub fn load_q4_0_packed(f: &mut FuncBuilder, ssa: &str, sym: &str, nbytes: u32) -> Value {
    let q_ty = format!("tensor<{nbytes}xi8>");
    f.global_load_as(ssa, sym, &q_ty)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_0_dequant_mlir_verifies() {
        let mut builder = ModuleBuilder::new().unwrap();
        emit_q4_0_dequant(&mut builder, "q4_0_dequant_32x32", 32, 32).unwrap();
        let verified = builder.finish().unwrap();
        assert!(verified.mlir_text.contains("q4_0_dequant_32x32"));
        assert!(verified.mlir_text.contains("arith.bitcast"));
    }

    #[test]
    fn q4_0_linear_mlir_verifies() {
        let mut builder = ModuleBuilder::new().unwrap();
        emit_q4_0_linear(&mut builder, "q4_linear_1x64x32", 1, 64, 32).unwrap();
        let verified = builder.finish().unwrap();
        assert!(verified.mlir_text.contains("q4_linear_1x64x32"));
        assert!(verified.mlir_text.contains("linalg.generic"));
        assert!(verified.mlir_text.contains("arith.bitcast"));
    }
}
