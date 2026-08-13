//! Shared dense-decoder helper kernels built with [`dyninfer_mlir::FuncBuilder`].

use dyninfer_error::Result;
use dyninfer_mlir::{FuncBuilder, ModuleBuilder, Value};

/// WMMA-R3 tiles are 16 along N (`head_dim`). The previous hardcoded `64`
/// failed to distribute on TinyLlama (`head_dim=16`) on gfx1151.
#[allow(dead_code)]
fn flash_n_tile(head_dim: u32) -> u32 {
    let cap = 64.min(head_dim);
    let mut tile = cap - (cap % 16);
    if tile == 0 {
        tile = 16;
    }
    while tile > 16 && head_dim % tile != 0 {
        tile -= 16;
    }
    tile
}

/// One page of numerically stable online attention. The returned accumulator,
/// row maximum, and row sum can be fed into the next page without materializing
/// a query-by-context score tensor.
///
/// Kept for HIP experiments; gfx1151 KernelConfig does not select
/// VectorDistribute for `online_attention` (iree#24064). Production HIP uses
/// [`emit_iree_attention`] instead.
#[allow(dead_code)]
pub fn emit_iree_online_attention_page(
    module: &mut ModuleBuilder,
    name: &str,
    query_len: u32,
    page_size: u32,
    kv_heads: u32,
    gqa_group: u32,
    head_dim: u32,
) -> Result<()> {
    let flash_queries = gqa_group * query_len;
    let kernel_queries = if query_len == 1 { 16 } else { flash_queries };
    let q_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}x{head_dim}xf32>");
    let q_flash_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}x{head_dim}xf16>");
    let q_flat_ty = format!("tensor<{kv_heads}x{flash_queries}x{head_dim}xf16>");
    let output_flat_ty = format!("tensor<{kv_heads}x{flash_queries}x{head_dim}xf32>");
    let q_kernel_ty = format!("tensor<{kv_heads}x{kernel_queries}x{head_dim}xf16>");
    let output_kernel_ty = format!("tensor<{kv_heads}x{kernel_queries}x{head_dim}xf32>");
    let kv_ty = format!("tensor<{kv_heads}x{page_size}x{head_dim}xf32>");
    let kv_flash_ty = format!("tensor<{kv_heads}x{page_size}x{head_dim}xf16>");
    let mask_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}x{page_size}xf32>");
    let mask_flash_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}x{page_size}xf16>");
    let mask_flat_ty = format!("tensor<{kv_heads}x{flash_queries}x{page_size}xf16>");
    let mask_kernel_ty = format!("tensor<{kv_heads}x{kernel_queries}x{page_size}xf16>");
    let row_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}xf32>");
    let row_flat_ty = format!("tensor<{kv_heads}x{flash_queries}xf32>");
    let row_kernel_ty = format!("tensor<{kv_heads}x{kernel_queries}xf32>");
    let n_tile = flash_n_tile(head_dim);
    // gfx1151 KernelConfig matches `iree_linalg_ext.attention`, not
    // `online_attention` (iree#24064). These compilation_info attrs force
    // LLVMGPUVectorDistribute when we must emit the online op for page merge.
    let flash_config = if query_len == 1 {
        format!(
            r#"compilation_info = #iree_codegen.compilation_info<
        lowering_config = #iree_gpu.lowering_config<{{
          promote_operands = [0, 1, 2],
          reduction = [0, 0, 0, 64, 0],
          workgroup = [1, 16, 0, 0, {n_tile}]}}>,
        translation_info = #iree_codegen.translation_info<
          pipeline = LLVMGPUVectorDistribute
          workgroup_size = [32, 1, 1]
          subgroup_size = 32,
          {{iree_codegen.denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<"preserve-sign">}}>>,
      decomposition_config = {{
        pv_attrs = {{lowering_config = #iree_gpu.lowering_config<{{
          mma_kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>,
          promote_operands = [1],
          subgroup_basis = [[1, 1, 1, 1, 1], [0, 1, 3, 4]]}}>}},
        qk_attrs = {{lowering_config = #iree_gpu.lowering_config<{{
          mma_kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>,
          promote_operands = [0, 1],
          subgroup_basis = [[1, 1, 1, 1, 1], [0, 1, 2, 3]]}}>}}}},
"#
        )
    } else {
        format!(
            r#"compilation_info = #iree_codegen.compilation_info<
        lowering_config = #iree_gpu.lowering_config<{{
          promote_operands = [0, 1, 2],
          reduction = [0, 0, 0, 64, 0],
          workgroup = [1, 64, 0, 0, {n_tile}]}}>,
        translation_info = #iree_codegen.translation_info<
          pipeline = LLVMGPUVectorDistribute
          workgroup_size = [128, 1, 1]
          subgroup_size = 32,
          {{iree_codegen.denormal_fp_math_f32 = #iree_codegen.denormal_fp_math<"preserve-sign">}}>>,
      decomposition_config = {{
        pv_attrs = {{lowering_config = #iree_gpu.lowering_config<{{
          mma_kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>,
          promote_operands = [1],
          subgroup_basis = [[1, 4, 1, 1, 1], [0, 1, 3, 4]]}}>}},
        qk_attrs = {{lowering_config = #iree_gpu.lowering_config<{{
          mma_kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>,
          promote_operands = [0, 1],
          subgroup_basis = [[1, 4, 1, 1, 1], [0, 1, 2, 3]]}}>}}}},
"#
        )
    };

    let mut f = module.func_private(name);
    f.arg("q", &q_ty);
    f.arg("k", &kv_ty);
    f.arg("v", &kv_ty);
    f.arg("scale", "f16");
    f.arg("mask", &mask_ty);
    f.arg("output", &q_ty);
    f.arg("row_max", &row_ty);
    f.arg("row_sum", &row_ty);
    f.result_ty(&q_ty);
    f.result_ty(&row_ty);
    f.result_ty(&row_ty);
    let (padding, q_input, mask_input, output_input, max_input, sum_input, result_slices) =
        if query_len == 1 {
            // WMMA decode tiles are 16-wide; pad extra query rows so they do not
            // participate in softmax (mask=-inf) or seed row_max with 0.
            (
                format!(
                    r#"  %zero16 = arith.constant 0.0 : f16
  %zero32 = arith.constant 0.0 : f32
  %neg16 = arith.constant 0xFC00 : f16
  %neg32 = arith.constant -3.40282347E+38 : f32
  %q_pad_e = tensor.empty() : {q_kernel_ty}
  %mask_pad_e = tensor.empty() : {mask_kernel_ty}
  %output_pad_e = tensor.empty() : {output_kernel_ty}
  %row_max_pad_e = tensor.empty() : {row_kernel_ty}
  %row_sum_pad_e = tensor.empty() : {row_kernel_ty}
  %q_pad = linalg.fill ins(%zero16 : f16) outs(%q_pad_e : {q_kernel_ty}) -> {q_kernel_ty}
  %mask_pad = linalg.fill ins(%neg16 : f16) outs(%mask_pad_e : {mask_kernel_ty}) -> {mask_kernel_ty}
  %output_pad = linalg.fill ins(%zero32 : f32) outs(%output_pad_e : {output_kernel_ty}) -> {output_kernel_ty}
  %row_max_pad = linalg.fill ins(%neg32 : f32) outs(%row_max_pad_e : {row_kernel_ty}) -> {row_kernel_ty}
  %row_sum_pad = linalg.fill ins(%zero32 : f32) outs(%row_sum_pad_e : {row_kernel_ty}) -> {row_kernel_ty}
  %q_kernel = tensor.insert_slice %q_flat into %q_pad[0, 0, 0] [{kv_heads}, {flash_queries}, {head_dim}] [1, 1, 1] : {q_flat_ty} into {q_kernel_ty}
  %mask_kernel = tensor.insert_slice %mask_flat into %mask_pad[0, 0, 0] [{kv_heads}, {flash_queries}, {page_size}] [1, 1, 1] : {mask_flat_ty} into {mask_kernel_ty}
  %output_kernel = tensor.insert_slice %output_flat into %output_pad[0, 0, 0] [{kv_heads}, {flash_queries}, {head_dim}] [1, 1, 1] : {output_flat_ty} into {output_kernel_ty}
  %max_kernel = tensor.insert_slice %max_flat into %row_max_pad[0, 0] [{kv_heads}, {flash_queries}] [1, 1] : {row_flat_ty} into {row_kernel_ty}
  %sum_kernel = tensor.insert_slice %sum_flat into %row_sum_pad[0, 0] [{kv_heads}, {flash_queries}] [1, 1] : {row_flat_ty} into {row_kernel_ty}
"#
                ),
                "q_kernel",
                "mask_kernel",
                "output_kernel",
                "max_kernel",
                "sum_kernel",
                format!(
                    r#"  %output_real = tensor.extract_slice %next#0[0, 0, 0] [{kv_heads}, {flash_queries}, {head_dim}] [1, 1, 1] : {output_kernel_ty} to {output_flat_ty}
  %max_real = tensor.extract_slice %next#1[0, 0] [{kv_heads}, {flash_queries}] [1, 1] : {row_kernel_ty} to {row_flat_ty}
  %sum_real = tensor.extract_slice %next#2[0, 0] [{kv_heads}, {flash_queries}] [1, 1] : {row_kernel_ty} to {row_flat_ty}
"#
                ),
            )
        } else {
            (
                String::new(),
                "q_flat",
                "mask_flat",
                "output_flat",
                "max_flat",
                "sum_flat",
                String::new(),
            )
        };
    let output_result = if query_len == 1 {
        "output_real"
    } else {
        "next#0"
    };
    let max_result = if query_len == 1 { "max_real" } else { "next#1" };
    let sum_result = if query_len == 1 { "sum_real" } else { "next#2" };
    f.op_asm(format!(
        r#"  %q16 = arith.truncf %q : {q_ty} to {q_flash_ty}
  %k16 = arith.truncf %k : {kv_ty} to {kv_flash_ty}
  %v16 = arith.truncf %v : {kv_ty} to {kv_flash_ty}
  %mask16 = arith.truncf %mask : {mask_ty} to {mask_flash_ty}
  %q_flat = tensor.collapse_shape %q16 [[0], [1, 2], [3]] : {q_flash_ty} into {q_flat_ty}
  %mask_flat = tensor.collapse_shape %mask16 [[0], [1, 2], [3]] : {mask_flash_ty} into {mask_flat_ty}
  %output_flat = tensor.collapse_shape %output [[0], [1, 2], [3]] : {q_ty} into {output_flat_ty}
  %max_flat = tensor.collapse_shape %row_max [[0], [1, 2]] : {row_ty} into {row_flat_ty}
  %sum_flat = tensor.collapse_shape %row_sum [[0], [1, 2]] : {row_ty} into {row_flat_ty}
{padding}
  %next:3 = iree_linalg_ext.online_attention {{
      {flash_config}
      indexing_maps = [
        affine_map<(h, q, d, p, n) -> (h, q, d)>,
        affine_map<(h, q, d, p, n) -> (h, p, d)>,
        affine_map<(h, q, d, p, n) -> (h, p, n)>,
        affine_map<(h, q, d, p, n) -> ()>,
        affine_map<(h, q, d, p, n) -> (h, q, p)>,
        affine_map<(h, q, d, p, n) -> (h, q, n)>,
        affine_map<(h, q, d, p, n) -> (h, q)>,
        affine_map<(h, q, d, p, n) -> (h, q)>
      ]
    }} ins(%{q_input}, %k16, %v16, %scale, %{mask_input} : {q_kernel_ty}, {kv_flash_ty}, {kv_flash_ty}, f16, {mask_kernel_ty})
       outs(%{output_input}, %{max_input}, %{sum_input} : {output_kernel_ty}, {row_kernel_ty}, {row_kernel_ty}) {{
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  }} -> {output_kernel_ty}, {row_kernel_ty}, {row_kernel_ty}
{result_slices}  %output_next = tensor.expand_shape %{output_result} [[0], [1, 2], [3]] output_shape [{kv_heads}, {gqa_group}, {query_len}, {head_dim}] : {output_flat_ty} into {q_ty}
  %max_next = tensor.expand_shape %{max_result} [[0], [1, 2]] output_shape [{kv_heads}, {gqa_group}, {query_len}] : {row_flat_ty} into {row_ty}
  %sum_next = tensor.expand_shape %{sum_result} [[0], [1, 2]] output_shape [{kv_heads}, {gqa_group}, {query_len}] : {row_flat_ty} into {row_ty}
  return %output_next, %max_next, %sum_next : {q_ty}, {row_ty}, {row_ty}
"#,
    ));
    f.finish(module)
}

/// Full-context `iree_linalg_ext.attention`. gfx1151 KernelConfig selects
/// VectorDistribute for AttentionOp and then converts it to online attention
/// internally (iree#24064). Prefer this over emitting `online_attention`.
pub fn emit_iree_attention(
    module: &mut ModuleBuilder,
    name: &str,
    query_len: u32,
    kv_len: u32,
    kv_heads: u32,
    gqa_group: u32,
    head_dim: u32,
) -> Result<()> {
    let flash_queries = gqa_group * query_len;
    let q_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}x{head_dim}xf32>");
    let q_flash_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}x{head_dim}xf16>");
    let q_flat_ty = format!("tensor<{kv_heads}x{flash_queries}x{head_dim}xf16>");
    let output_flat_ty = format!("tensor<{kv_heads}x{flash_queries}x{head_dim}xf32>");
    let kv_flash_ty = format!("tensor<{kv_heads}x{kv_len}x{head_dim}xf16>");
    let mask_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}x{kv_len}xf32>");
    let mask_flash_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}x{kv_len}xf16>");
    let mask_flat_ty = format!("tensor<{kv_heads}x{flash_queries}x{kv_len}xf16>");
    let mut f = module.func_private(name);
    f.arg("q", &q_ty);
    f.arg("k", &kv_flash_ty);
    f.arg("v", &kv_flash_ty);
    f.arg("scale", "f16");
    f.arg("mask", &mask_ty);
    f.arg("output", &q_ty);
    f.result_ty(&q_ty);
    f.op_asm(format!(
        r#"  %q16 = arith.truncf %q : {q_ty} to {q_flash_ty}
  %mask16 = arith.truncf %mask : {mask_ty} to {mask_flash_ty}
  %q_flat = tensor.collapse_shape %q16 [[0], [1, 2], [3]] : {q_flash_ty} into {q_flat_ty}
  %mask_flat = tensor.collapse_shape %mask16 [[0], [1, 2], [3]] : {mask_flash_ty} into {mask_flat_ty}
  %output_flat = tensor.collapse_shape %output [[0], [1, 2], [3]] : {q_ty} into {output_flat_ty}
  %next = iree_linalg_ext.attention {{
      indexing_maps = [
        affine_map<(h, q, d, p, n) -> (h, q, d)>,
        affine_map<(h, q, d, p, n) -> (h, p, d)>,
        affine_map<(h, q, d, p, n) -> (h, p, n)>,
        affine_map<(h, q, d, p, n) -> ()>,
        affine_map<(h, q, d, p, n) -> (h, q, p)>,
        affine_map<(h, q, d, p, n) -> (h, q, n)>
      ]
    }} ins(%q_flat, %k, %v, %scale, %mask_flat : {q_flat_ty}, {kv_flash_ty}, {kv_flash_ty}, f16, {mask_flat_ty})
       outs(%output_flat : {output_flat_ty}) {{
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  }} -> {output_flat_ty}
  %output_next = tensor.expand_shape %next [[0], [1, 2], [3]] output_shape [{kv_heads}, {gqa_group}, {query_len}, {head_dim}] : {output_flat_ty} into {q_ty}
  return %output_next : {q_ty}
"#
    ));
    f.finish(module)
}

/// Causal + written-range mask over a packed KV length (`num_pages * page`).
pub fn emit_full_causal_mask(
    module: &mut ModuleBuilder,
    name: &str,
    query_len: u32,
    kv_len: u32,
    kv_heads: u32,
    gqa_group: u32,
) -> Result<()> {
    let mask_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}x{kv_len}xf32>");
    let mut f = module.func_private(name);
    f.arg("start_pos", "tensor<i64>");
    f.arg("valid_count", "tensor<i64>");
    f.result_ty(&mask_ty);
    f.op_asm(format!(
        r#"  %start64 = tensor.extract %start_pos[] : tensor<i64>
  %valid64 = tensor.extract %valid_count[] : tensor<i64>
  %seq_end = arith.addi %start64, %valid64 : i64
  %zero = arith.constant 0.0 : f32
  %neg = arith.constant -3.40282347E+38 : f32
  %empty = tensor.empty() : {mask_ty}
  %mask = linalg.generic {{
      indexing_maps = [affine_map<(kh, g, q, k) -> (kh, g, q, k)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}}
    outs(%empty : {mask_ty}) {{
    ^bb0(%o: f32):
      %q = linalg.index 2 : index
      %k = linalg.index 3 : index
      %q64 = arith.index_cast %q : index to i64
      %k64 = arith.index_cast %k : index to i64
      %abs_q = arith.addi %start64, %q64 : i64
      %causal = arith.cmpi ule, %k64, %abs_q : i64
      %written = arith.cmpi ult, %k64, %seq_end : i64
      %visible = arith.andi %causal, %written : i1
      %value = arith.select %visible, %zero, %neg : f32
      linalg.yield %value : f32
  }} -> {mask_ty}
  return %mask : {mask_ty}"#
    ));
    f.finish(module)
}

/// Backend-portable online attention fallback. It materializes scores for one
/// fixed page only, then merges page-local state with the running softmax.
pub fn emit_online_attention_page(
    module: &mut ModuleBuilder,
    name: &str,
    query_len: u32,
    page_size: u32,
    kv_heads: u32,
    gqa_group: u32,
    head_dim: u32,
) -> Result<()> {
    let q_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}x{head_dim}xf32>");
    let kv_ty = format!("tensor<{kv_heads}x{page_size}x{head_dim}xf32>");
    let score_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}x{page_size}xf32>");
    let row_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}xf32>");
    let mut f = module.func_private(name);
    f.arg("q", &q_ty);
    f.arg("k", &kv_ty);
    f.arg("v", &kv_ty);
    f.arg("scale", "f32");
    f.arg("mask", &score_ty);
    f.arg("output", &q_ty);
    f.arg("row_max", &row_ty);
    f.arg("row_sum", &row_ty);
    f.result_ty(&q_ty);
    f.result_ty(&row_ty);
    f.result_ty(&row_ty);
    f.op_asm(format!(
        r#"  %zero = arith.constant 0.0 : f32
  %neg = arith.constant -3.40282347E+38 : f32
  %scores_e = tensor.empty() : {score_ty}
  %scores_z = linalg.fill ins(%zero : f32) outs(%scores_e : {score_ty}) -> {score_ty}
  %dots = linalg.generic {{
      indexing_maps = [
        affine_map<(kh, g, q, p, d) -> (kh, g, q, d)>,
        affine_map<(kh, g, q, p, d) -> (kh, p, d)>,
        affine_map<(kh, g, q, p, d) -> (kh, g, q, p)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}}
    ins(%q, %k : {q_ty}, {kv_ty}) outs(%scores_z : {score_ty}) {{
    ^bb0(%qv: f32, %kv: f32, %acc: f32):
      %product = arith.mulf %qv, %kv : f32
      %next = arith.addf %acc, %product : f32
      linalg.yield %next : f32
  }} -> {score_ty}
  %masked_e = tensor.empty() : {score_ty}
  %scores = linalg.generic {{
      indexing_maps = [
        affine_map<(kh, g, q, p) -> (kh, g, q, p)>,
        affine_map<(kh, g, q, p) -> (kh, g, q, p)>,
        affine_map<(kh, g, q, p) -> (kh, g, q, p)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}}
    ins(%dots, %mask : {score_ty}, {score_ty}) outs(%masked_e : {score_ty}) {{
    ^bb0(%dot: f32, %bias: f32, %o: f32):
      %scaled = arith.mulf %dot, %scale : f32
      %value = arith.addf %scaled, %bias : f32
      linalg.yield %value : f32
  }} -> {score_ty}
  %row_e = tensor.empty() : {row_ty}
  %page_max_z = linalg.fill ins(%neg : f32) outs(%row_e : {row_ty}) -> {row_ty}
  %page_max = linalg.reduce ins(%scores : {score_ty}) outs(%page_max_z : {row_ty}) dimensions = [3]
    (%value: f32, %acc: f32) {{
      %next = arith.maximumf %value, %acc : f32
      linalg.yield %next : f32
    }}
  %new_max_e = tensor.empty() : {row_ty}
  %new_max = arith.maximumf %row_max, %page_max : {row_ty}
  %weights_e = tensor.empty() : {score_ty}
  %weights = linalg.generic {{
      indexing_maps = [
        affine_map<(kh, g, q, p) -> (kh, g, q, p)>,
        affine_map<(kh, g, q, p) -> (kh, g, q)>,
        affine_map<(kh, g, q, p) -> (kh, g, q, p)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}}
    ins(%scores, %new_max : {score_ty}, {row_ty}) outs(%weights_e : {score_ty}) {{
    ^bb0(%score: f32, %maximum: f32, %o: f32):
      %shifted = arith.subf %score, %maximum : f32
      %weight = math.exp %shifted : f32
      linalg.yield %weight : f32
  }} -> {score_ty}
  %page_sum_z = linalg.fill ins(%zero : f32) outs(%row_e : {row_ty}) -> {row_ty}
  %page_sum = linalg.reduce ins(%weights : {score_ty}) outs(%page_sum_z : {row_ty}) dimensions = [3]
    (%value: f32, %acc: f32) {{
      %next = arith.addf %value, %acc : f32
      linalg.yield %next : f32
    }}
  %page_out_e = tensor.empty() : {q_ty}
  %page_out_z = linalg.fill ins(%zero : f32) outs(%page_out_e : {q_ty}) -> {q_ty}
  %page_out = linalg.generic {{
      indexing_maps = [
        affine_map<(kh, g, q, d, p) -> (kh, g, q, p)>,
        affine_map<(kh, g, q, d, p) -> (kh, p, d)>,
        affine_map<(kh, g, q, d, p) -> (kh, g, q, d)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}}
    ins(%weights, %v : {score_ty}, {kv_ty}) outs(%page_out_z : {q_ty}) {{
    ^bb0(%weight: f32, %value: f32, %acc: f32):
      %product = arith.mulf %weight, %value : f32
      %next = arith.addf %acc, %product : f32
      linalg.yield %next : f32
  }} -> {q_ty}
  %old_scale_e = tensor.empty() : {row_ty}
  %old_scale = linalg.generic {{
      indexing_maps = [
        affine_map<(kh, g, q) -> (kh, g, q)>,
        affine_map<(kh, g, q) -> (kh, g, q)>,
        affine_map<(kh, g, q) -> (kh, g, q)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%row_max, %new_max : {row_ty}, {row_ty}) outs(%old_scale_e : {row_ty}) {{
    ^bb0(%old: f32, %new: f32, %o: f32):
      %delta = arith.subf %old, %new : f32
      %factor = math.exp %delta : f32
      linalg.yield %factor : f32
  }} -> {row_ty}
  %sum_e = tensor.empty() : {row_ty}
  %combined_sum = linalg.generic {{
      indexing_maps = [
        affine_map<(kh, g, q) -> (kh, g, q)>,
        affine_map<(kh, g, q) -> (kh, g, q)>,
        affine_map<(kh, g, q) -> (kh, g, q)>,
        affine_map<(kh, g, q) -> (kh, g, q)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%row_sum, %old_scale, %page_sum : {row_ty}, {row_ty}, {row_ty}) outs(%sum_e : {row_ty}) {{
    ^bb0(%old: f32, %factor: f32, %page: f32, %o: f32):
      %scaled = arith.mulf %old, %factor : f32
      %value = arith.addf %scaled, %page : f32
      linalg.yield %value : f32
  }} -> {row_ty}
  %out_e = tensor.empty() : {q_ty}
  %combined_out = linalg.generic {{
      indexing_maps = [
        affine_map<(kh, g, q, d) -> (kh, g, q, d)>,
        affine_map<(kh, g, q, d) -> (kh, g, q)>,
        affine_map<(kh, g, q, d) -> (kh, g, q, d)>,
        affine_map<(kh, g, q, d) -> (kh, g, q, d)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}}
    ins(%output, %old_scale, %page_out : {q_ty}, {row_ty}, {q_ty}) outs(%out_e : {q_ty}) {{
    ^bb0(%old: f32, %factor: f32, %page: f32, %o: f32):
      %scaled = arith.mulf %old, %factor : f32
      %value = arith.addf %scaled, %page : f32
      linalg.yield %value : f32
  }} -> {q_ty}
  return %combined_out, %new_max, %combined_sum : {q_ty}, {row_ty}, {row_ty}"#
    ));
    f.finish(module)
}

pub fn emit_paged_causal_mask(
    module: &mut ModuleBuilder,
    name: &str,
    query_len: u32,
    page_size: u32,
    kv_heads: u32,
    gqa_group: u32,
) -> Result<()> {
    let mask_ty = format!("tensor<{kv_heads}x{gqa_group}x{query_len}x{page_size}xf32>");
    let mut f = module.func_private(name);
    f.arg("start_pos", "tensor<i64>");
    f.arg("page_index", "index");
    // Host/chunk valid length: only keys in `[start, start+valid)` are written.
    f.arg("valid_count", "tensor<i64>");
    f.result_ty(&mask_ty);
    let body = format!(
        r#"  %start64 = tensor.extract %start_pos[] : tensor<i64>
  %valid64 = tensor.extract %valid_count[] : tensor<i64>
  %seq_end = arith.addi %start64, %valid64 : i64
  %page64 = arith.index_cast %page_index : index to i64
  %page_size64 = arith.constant {page_size} : i64
  %page_start = arith.muli %page64, %page_size64 : i64
  %zero = arith.constant 0.0 : f32
  %neg = arith.constant -3.40282347E+38 : f32
  %empty = tensor.empty() : {mask_ty}
  %mask = linalg.generic {{
      indexing_maps = [affine_map<(kh, g, q, k) -> (kh, g, q, k)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}}
    outs(%empty : {mask_ty}) {{
    ^bb0(%o: f32):
      %q = linalg.index 2 : index
      %k = linalg.index 3 : index
      %q64 = arith.index_cast %q : index to i64
      %k64 = arith.index_cast %k : index to i64
      %abs_q = arith.addi %start64, %q64 : i64
      %abs_k = arith.addi %page_start, %k64 : i64
      %causal = arith.cmpi ule, %abs_k, %abs_q : i64
      %written = arith.cmpi ult, %abs_k, %seq_end : i64
      %visible = arith.andi %causal, %written : i1
      %value = arith.select %visible, %zero, %neg : f32
      linalg.yield %value : f32
  }} -> {mask_ty}
  return %mask : {mask_ty}"#
    );
    f.op_asm(body);
    f.finish(module)
}

#[cfg(test)]
mod online_attention_tests {
    use super::*;

    fn q_value(head: usize, pos: usize, dim: usize) -> f64 {
        (((head * 17 + pos * 3 + dim * 11) % 29) as f64 - 14.0) * 0.75
    }

    fn k_value(kv_head: usize, pos: usize, dim: usize) -> f64 {
        (((kv_head * 13 + pos * 7 + dim * 5) % 31) as f64 - 15.0) * 0.5
    }

    fn v_value(kv_head: usize, pos: usize, dim: usize) -> f64 {
        ((kv_head * 19 + pos * 11 + dim * 2) % 37) as f64 / 37.0
    }

    fn direct_attention(head: usize, query_pos: usize, length: usize, dim: usize) -> Vec<f64> {
        let kv_head = head / 2;
        let mut scores = Vec::with_capacity(query_pos + 1);
        for key_pos in 0..length {
            if key_pos > query_pos {
                break;
            }
            let score = (0..dim)
                .map(|d| q_value(head, query_pos, d) * k_value(kv_head, key_pos, d))
                .sum::<f64>()
                / (dim as f64).sqrt();
            scores.push(score);
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let sum: f64 = weights.iter().sum();
        (0..dim)
            .map(|d| {
                weights
                    .iter()
                    .enumerate()
                    .map(|(key_pos, weight)| weight * v_value(kv_head, key_pos, d))
                    .sum::<f64>()
                    / sum
            })
            .collect()
    }

    fn paged_attention(head: usize, query_pos: usize, length: usize, dim: usize) -> Vec<f64> {
        let kv_head = head / 2;
        let mut running_max = f64::NEG_INFINITY;
        let mut running_sum = 0.0;
        let mut output = vec![0.0; dim];
        for page_start in (0..length).step_by(256) {
            let page_end = (page_start + 256).min(length).min(query_pos + 1);
            if page_start >= page_end {
                continue;
            }
            let scores: Vec<_> = (page_start..page_end)
                .map(|key_pos| {
                    (0..dim)
                        .map(|d| q_value(head, query_pos, d) * k_value(kv_head, key_pos, d))
                        .sum::<f64>()
                        / (dim as f64).sqrt()
                })
                .collect();
            let page_max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let next_max = running_max.max(page_max);
            let old_scale = (running_max - next_max).exp();
            for value in &mut output {
                *value *= old_scale;
            }
            running_sum *= old_scale;
            for (offset, score) in scores.into_iter().enumerate() {
                let weight = (score - next_max).exp();
                let key_pos = page_start + offset;
                running_sum += weight;
                for (d, value) in output.iter_mut().enumerate() {
                    *value += weight * v_value(kv_head, key_pos, d);
                }
            }
            running_max = next_max;
        }
        for value in &mut output {
            *value /= running_sum;
        }
        output
    }

    #[test]
    fn page_kernel_verifies_without_quadratic_context_tensor() {
        let mut module = ModuleBuilder::new().unwrap();
        emit_online_attention_page(&mut module, "online_page", 32, 256, 8, 2, 128).unwrap();
        emit_iree_online_attention_page(&mut module, "iree_online_page", 32, 256, 8, 2, 128)
            .unwrap();
        emit_paged_causal_mask(&mut module, "paged_mask", 32, 256, 8, 2).unwrap();
        emit_rope_chunk(&mut module, "rope_chunk", 32, 8, 128, 1_000_000.0).unwrap();
        let text = module.finish().unwrap().mlir_text;
        assert!(text.contains("iree_linalg_ext.online_attention"));
        assert!(text.contains("math.powf"));
        assert!(!text.contains("tensor<8x4096x4096"));
        let mut attn_mod = ModuleBuilder::new().unwrap();
        emit_iree_attention(&mut attn_mod, "iree_attn", 1, 1024, 8, 2, 128).unwrap();
        emit_full_causal_mask(&mut attn_mod, "full_mask", 1, 1024, 8, 2).unwrap();
        let attn_text = attn_mod.finish().unwrap().mlir_text;
        assert!(attn_text.contains("iree_linalg_ext.attention"));
        assert!(!attn_text.contains("iree_linalg_ext.online_attention"));
        assert!(attn_text.contains("tensor<8x1024x128xf16>"));
        assert!(!attn_text.contains("%k16 = arith.truncf"));
    }

    #[test]
    fn tiny_head_dim_flash_uses_matching_n_tile() {
        let mut module = ModuleBuilder::new().unwrap();
        emit_iree_online_attention_page(&mut module, "tiny_flash", 32, 256, 4, 1, 16).unwrap();
        let text = module.finish().unwrap().mlir_text;
        assert!(
            text.contains("workgroup = [1, 64, 0, 0, 16]"),
            "head_dim=16 must not request an N tile of 64"
        );
    }

    #[test]
    fn decode_flash_pads_masked_rows_with_neg_inf() {
        let mut module = ModuleBuilder::new().unwrap();
        emit_iree_online_attention_page(&mut module, "decode_flash_pad", 1, 256, 8, 2, 128)
            .unwrap();
        let text = module.finish().unwrap().mlir_text;
        assert!(
            text.contains("0xFC00 : f16"),
            "padded decode mask rows must be f16 -inf"
        );
        assert!(
            text.contains("-3.40282347E+38 : f32"),
            "padded decode row_max must be -inf"
        );
        // Regression: filling pad mask/max with 0 lets dummy WMMA rows attend.
        assert!(!text.contains(
            "linalg.fill ins(%zero16 : f16) outs(%mask_pad_e"
        ));
    }

    #[test]
    fn hip_decode_flash_compiles_when_requested() {
        if std::env::var_os("DYNINFER_HIP_FLASH_DECODE").is_none() {
            return;
        }
        let mut module = ModuleBuilder::new().unwrap();
        emit_iree_online_attention_page(&mut module, "decode_flash_impl", 1, 256, 8, 2, 128)
            .unwrap();
        let q = "tensor<8x2x1x128xf32>";
        let kv = "tensor<8x256x128xf32>";
        let mask = "tensor<8x2x1x256xf32>";
        let row = "tensor<8x2x1xf32>";
        let mut f = module.func("decode_flash");
        f.arg("q", q);
        f.arg("k", kv);
        f.arg("v", kv);
        f.arg("scale", "f16");
        f.arg("mask", mask);
        f.arg("output", q);
        f.arg("row_max", row);
        f.arg("row_sum", row);
        f.result_ty(q);
        f.result_ty(row);
        f.result_ty(row);
        f.op_asm(format!(
            "  %next:3 = func.call @decode_flash_impl(%q, %k, %v, %scale, %mask, %output, %row_max, %row_sum) : ({q}, {kv}, {kv}, f16, {mask}, {q}, {row}, {row}) -> ({q}, {row}, {row})\n  return %next#0, %next#1, %next#2 : {q}, {row}, {row}"
        ));
        f.finish(&mut module).unwrap();
        let mlir = module.finish().unwrap().mlir_text;
        let arch = std::env::var("DYNINFER_HIP_ARCH").unwrap_or_else(|_| "gfx1151".into());
        let mut flags = vec![
            "--iree-hal-target-device=hip".into(),
            format!("--iree-rocm-target={arch}"),
        ];
        if let Some(path) = iree_compiler_sys::discover_rocm_bc_dir() {
            flags.push(format!("--iree-rocm-bc-dir={}", path.display()));
        }
        iree_compiler_sys::compile_mlir_to_vmfb(&mlir, &flags)
            .expect("one-token HIP Flash Attention must compile");
    }

    #[test]
    fn hip_iree_attention_compiles_when_requested() {
        if std::env::var_os("DYNINFER_HIP_FLASH_DECODE").is_none() {
            return;
        }
        let mut module = ModuleBuilder::new().unwrap();
        emit_iree_attention(&mut module, "attn_impl", 1, 1024, 8, 2, 128).unwrap();
        let q = "tensor<8x2x1x128xf32>";
        let kv = "tensor<8x1024x128xf16>";
        let mask = "tensor<8x2x1x1024xf32>";
        let mut f = module.func("decode_attn");
        f.arg("q", q);
        f.arg("k", kv);
        f.arg("v", kv);
        f.arg("scale", "f16");
        f.arg("mask", mask);
        f.arg("output", q);
        f.result_ty(q);
        f.op_asm(format!(
            "  %next = func.call @attn_impl(%q, %k, %v, %scale, %mask, %output) : ({q}, {kv}, {kv}, f16, {mask}, {q}) -> {q}\n  return %next : {q}"
        ));
        f.finish(&mut module).unwrap();
        let mlir = module.finish().unwrap().mlir_text;
        let arch = std::env::var("DYNINFER_HIP_ARCH").unwrap_or_else(|_| "gfx1151".into());
        let mut flags = vec![
            "--iree-hal-target-device=hip".into(),
            format!("--iree-rocm-target={arch}"),
        ];
        if let Some(path) = iree_compiler_sys::discover_rocm_bc_dir() {
            flags.push(format!("--iree-rocm-bc-dir={}", path.display()));
        }
        iree_compiler_sys::compile_mlir_to_vmfb(&mlir, &flags)
            .expect("HIP iree_linalg_ext.attention must compile on gfx1151");
    }

    #[test]
    fn online_merge_matches_f64_reference_at_page_boundaries() {
        for length in [255, 256, 257, 4096] {
            for query_pos in [0, 255.min(length - 1), 256.min(length - 1), length - 1] {
                for head in 0..4 {
                    let direct = direct_attention(head, query_pos, length, 4);
                    let paged = paged_attention(head, query_pos, length, 4);
                    for (expected, actual) in direct.iter().zip(&paged) {
                        assert!((expected - actual).abs() < 1e-12);
                    }
                }
            }
        }
    }
}

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

/// RoPE for a fixed-size chunk at a runtime absolute token offset. Unlike the
/// legacy helpers this computes frequencies on device and emits no context-size
/// lookup table.
pub fn emit_rope_chunk(
    module: &mut ModuleBuilder,
    name: &str,
    chunk: u32,
    heads: u32,
    head_dim: u32,
    theta: f32,
) -> Result<()> {
    let half = head_dim / 2;
    let x_ty = format!("tensor<{chunk}x{heads}x{head_dim}xf32>");
    let mut f = module.func_private(name);
    f.arg("x", &x_ty);
    f.arg("start_pos", "tensor<i64>");
    f.result_ty(&x_ty);
    f.op_asm(format!(
        r#"  %start64 = tensor.extract %start_pos[] : tensor<i64>
  %start = arith.index_cast %start64 : i64 to index
  %init = tensor.empty() : {x_ty}
  %theta = arith.constant {theta:.8e} : f32
  %two = arith.constant 2.0 : f32
  %dimf = arith.constant {head_dim}.0 : f32
  %one = arith.constant 1.0 : f32
  %y = linalg.generic {{
      indexing_maps = [
        affine_map<(p, h, dim) -> (p, h, dim)>,
        affine_map<(p, h, dim) -> (p, h, dim)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%x : {x_ty}) outs(%init : {x_ty}) {{
    ^bb0(%a: f32, %o: f32):
      %p = linalg.index 0 : index
      %hh = linalg.index 1 : index
      %dim = linalg.index 2 : index
      %half_i = arith.constant {half} : index
      %in_first = arith.cmpi ult, %dim, %half_i : index
      %pair = arith.remui %dim, %half_i : index
      %pair_hi = arith.addi %pair, %half_i : index
      %abs_pos = arith.addi %start, %p : index
      %pos64 = arith.index_cast %abs_pos : index to i64
      %pair64 = arith.index_cast %pair : index to i64
      %posf = arith.sitofp %pos64 : i64 to f32
      %pairf = arith.sitofp %pair64 : i64 to f32
      %twopair = arith.mulf %two, %pairf : f32
      %exponent = arith.divf %twopair, %dimf : f32
      %den = math.powf %theta, %exponent : f32
      %freq = arith.divf %one, %den : f32
      %angle = arith.mulf %posf, %freq : f32
      %cv = math.cos %angle : f32
      %sv = math.sin %angle : f32
      %x1 = tensor.extract %x[%p, %hh, %pair] : {x_ty}
      %x2 = tensor.extract %x[%p, %hh, %pair_hi] : {x_ty}
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
