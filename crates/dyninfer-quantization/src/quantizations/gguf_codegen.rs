//! Shared MLIR building blocks for executable, interleaved GGUF encodings.
//!
//! Format definitions still own candidate registration and delegate here with
//! an explicit layout. This module never dispatches on a codec string.

use crate::{EmbeddingCallSpec, LinearCallSpec, ParameterLoweringOperation, ParameterLoweringSpec};
use dyninfer_core::{ParameterBinding, StorageElementType};
use dyninfer_error::{DynInferError, Result};
use dyninfer_kernel_registry::{
    AxisMultiple, EncodingKey, KernelCandidateDescriptor, KernelOperationKind,
    ParameterOrientation, ProductionReadiness, ShapeConstraint, TargetConstraint,
};
use dyninfer_mlir::{FuncBuilder, ModuleBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockLayout {
    Q40,
    Q41,
    Q80,
    Q6K,
}

impl BlockLayout {
    pub const fn block_size(self) -> u32 {
        match self {
            Self::Q40 | Self::Q41 | Self::Q80 => 32,
            Self::Q6K => 256,
        }
    }

    pub const fn bytes_per_block(self) -> u32 {
        match self {
            Self::Q40 => 18,
            Self::Q41 => 20,
            Self::Q80 => 34,
            Self::Q6K => 210,
        }
    }

    const fn stem(self) -> &'static str {
        match self {
            Self::Q40 => "gguf_q4_0",
            Self::Q41 => "gguf_q4_1",
            Self::Q80 => "gguf_q8_0",
            Self::Q6K => "gguf_q6_k",
        }
    }
}

pub fn kernel_candidates(
    layout: BlockLayout,
    codec: &'static str,
) -> Vec<KernelCandidateDescriptor> {
    let operations = [
        (KernelOperationKind::Embedding, "gather"),
        (KernelOperationKind::Linear, "matmul"),
        (KernelOperationKind::OutputProjection, "output"),
    ];
    let targets = [
        (
            "cpu",
            TargetConstraint {
                backends: vec!["local-task".into(), "local-sync".into()],
                exact_architectures: vec![],
                required_features: vec!["avx2".into()],
            },
        ),
        (
            "hip",
            TargetConstraint {
                backends: vec!["hip".into(), "rocm".into()],
                exact_architectures: vec![],
                required_features: vec![],
            },
        ),
        (
            "vulkan",
            TargetConstraint {
                backends: vec!["vulkan".into()],
                exact_architectures: vec![],
                required_features: vec!["spirv".into()],
            },
        ),
    ];
    operations
        .into_iter()
        .flat_map(|(operation, suffix)| {
            targets
                .clone()
                .into_iter()
                .map(move |(backend, target)| (operation, suffix, backend, target))
        })
        .map(|(operation, suffix, backend, target)| KernelCandidateDescriptor {
        id: dyninfer_core::KernelId::new(format!("{codec}.{suffix}.iree_block_{backend}")),
        operation,
        encoding: Some(EncodingKey::new(codec, 1)),
        input_types: vec![dyninfer_core::ScalarType::F32],
        output_types: vec![dyninfer_core::ScalarType::F32],
        accumulator_types: vec![dyninfer_core::ScalarType::F32],
        shape: ShapeConstraint {
            logical_rank: Some(2),
            axis_multiples: vec![AxisMultiple {
                axis: 1,
                multiple_of: u64::from(layout.block_size()),
            }],
        },
        orientations: vec![ParameterOrientation::Native],
        target,
        modes: vec![
            dyninfer_core::ExecutionMode::Prefill,
            dyninfer_core::ExecutionMode::Decode,
        ],
        lowering: dyninfer_core::LoweringId::new(format!("{codec}.{suffix}.iree_block")),
        deterministic_score: 300,
        readiness: ProductionReadiness::Production,
        notes: format!(
            "Consumes official GGML {codec} interleaved blocks directly; dequantization remains adjacent to gather/matmul"
        ),
        })
        .collect()
}

pub fn emit_external_globals(
    builder: &mut ModuleBuilder,
    binding: &ParameterBinding,
    symbol: &str,
) -> Result<bool> {
    let component = data_component(binding)?;
    let bytes = component_bytes(binding)?;
    builder.util_global_parameter(
        &format!("{symbol}_data"),
        &component.external_key,
        &format!("tensor<{bytes}xi8>"),
    )?;
    Ok(true)
}

pub fn emit_parameter_load(
    function: &mut FuncBuilder,
    binding: &ParameterBinding,
    ssa: &str,
    symbol: &str,
) -> Result<bool> {
    let bytes = component_bytes(binding)?;
    function.op_asm(format!(
        "  %{ssa}_data = util.global.load @{symbol}_data : tensor<{bytes}xi8>\n"
    ));
    Ok(true)
}

pub fn helper_key(layout: BlockLayout, spec: &ParameterLoweringSpec<'_>) -> Option<String> {
    let [output, input] = spec.binding.logical_shape.dims() else {
        return None;
    };
    if !input.is_multiple_of(u64::from(layout.block_size())) {
        return None;
    }
    match spec.operation {
        ParameterLoweringOperation::Embedding => {
            Some(format!("{}-gather-{output}-{input}", layout.stem()))
        }
        ParameterLoweringOperation::Linear { rows } => {
            Some(format!("{}-linear-{rows}-{input}-{output}", layout.stem()))
        }
        ParameterLoweringOperation::OutputProjection => {
            Some(format!("{}-linear-1-{input}-{output}", layout.stem()))
        }
    }
}

pub fn emit_helper(
    module: &mut ModuleBuilder,
    layout: BlockLayout,
    spec: &ParameterLoweringSpec<'_>,
) -> Result<bool> {
    if helper_key(layout, spec).is_none() {
        return Ok(false);
    }
    let [output, input] = spec.binding.logical_shape.dims() else {
        unreachable!()
    };
    let (output, input) = (*output as u32, *input as u32);
    match spec.operation {
        ParameterLoweringOperation::Embedding => {
            emit_gather(module, layout, output, input)?;
        }
        ParameterLoweringOperation::Linear { rows } => {
            emit_linear(module, layout, rows, input, output)?;
        }
        ParameterLoweringOperation::OutputProjection => {
            emit_linear(module, layout, 1, input, output)?;
        }
    }
    Ok(true)
}

pub fn emit_linear_call(
    function: &mut FuncBuilder,
    layout: BlockLayout,
    spec: &LinearCallSpec<'_>,
) -> Result<bool> {
    if helper_key(layout, &spec.lowering).is_none() {
        return Ok(false);
    }
    let [output, input] = spec.lowering.binding.logical_shape.dims() else {
        unreachable!()
    };
    let rows = match spec.lowering.operation {
        ParameterLoweringOperation::Linear { rows } => rows,
        ParameterLoweringOperation::OutputProjection => 1,
        ParameterLoweringOperation::Embedding => return Ok(false),
    };
    let bytes = component_bytes(spec.lowering.binding)?;
    let helper = linear_fn(layout, rows, *input as u32, *output as u32);
    function.op_asm(format!(
        "  %{} = func.call @{helper}(%{}, %{}_data) : (tensor<{rows}x{input}xf32>, tensor<{bytes}xi8>) -> tensor<{rows}x{output}xf32>\n",
        spec.result_ssa, spec.input_ssa, spec.parameter_ssa,
    ));
    Ok(true)
}

pub fn emit_embedding_call(
    function: &mut FuncBuilder,
    layout: BlockLayout,
    spec: &EmbeddingCallSpec<'_>,
) -> Result<bool> {
    if helper_key(layout, &spec.lowering).is_none() {
        return Ok(false);
    }
    let [vocab, width] = spec.lowering.binding.logical_shape.dims() else {
        unreachable!()
    };
    let bytes = component_bytes(spec.lowering.binding)?;
    let helper = gather_fn(layout, *vocab as u32, *width as u32);
    function.op_asm(format!(
        "  %{} = func.call @{helper}(%{}_data, %{}) : (tensor<{bytes}xi8>, index) -> tensor<1x{width}xf32>\n",
        spec.result_ssa, spec.parameter_ssa, spec.index_ssa,
    ));
    Ok(true)
}

fn data_component(binding: &ParameterBinding) -> Result<&dyninfer_core::ParameterComponentBinding> {
    let [component] = binding.components.as_slice() else {
        return Err(DynInferError::internal(format!(
            "GGUF block parameter `{}` must have one data component",
            binding.canonical_name
        )));
    };
    if component.component_name != "data"
        || !matches!(component.storage_type, StorageElementType::Opaque { .. })
    {
        return Err(DynInferError::internal(format!(
            "GGUF block parameter `{}` has an invalid data component",
            binding.canonical_name
        )));
    }
    Ok(component)
}

fn component_bytes(binding: &ParameterBinding) -> Result<u64> {
    data_component(binding)?
        .byte_lengths
        .iter()
        .try_fold(0u64, |sum, bytes| sum.checked_add(*bytes))
        .ok_or_else(|| DynInferError::internal("GGUF block component size overflow"))
}

fn linear_fn(layout: BlockLayout, rows: u32, input: u32, output: u32) -> String {
    format!("{}_linear_{rows}_{input}_{output}", layout.stem())
}

fn gather_fn(layout: BlockLayout, vocab: u32, width: u32) -> String {
    format!("{}_gather_{vocab}_{width}", layout.stem())
}

fn emit_linear(
    module: &mut ModuleBuilder,
    layout: BlockLayout,
    rows: u32,
    input: u32,
    output: u32,
) -> Result<()> {
    let block = layout.block_size();
    let block_bytes = layout.bytes_per_block();
    assert!(input.is_multiple_of(block));
    let blocks = input / block;
    let bytes = u64::from(output) * u64::from(blocks) * u64::from(block_bytes);
    let x_ty = format!("tensor<{rows}x{input}xf32>");
    let xg_ty = format!("tensor<{rows}x{blocks}x{block}xf32>");
    let data_ty = format!("tensor<{bytes}xi8>");
    let w_ty = format!("tensor<{output}x{blocks}x{block}xf32>");
    let y_ty = format!("tensor<{rows}x{output}xf32>");

    let mut function = module.func_private(&linear_fn(layout, rows, input, output));
    let x = function.arg("x", &x_ty);
    let _data = function.arg("data", &data_ty);
    function.result_ty(&y_ty);
    function.op_asm(format!(
        "  %xg = tensor.expand_shape {x} [[0], [1, 2]] output_shape [{rows}, {blocks}, {block}] : {x_ty} into {xg_ty}\n"
    ));
    emit_dequant(&mut function, layout, "%data", output, blocks, "w")?;
    function.op_asm(format!(
        r#"  %zero = arith.constant 0.0 : f32
  %y_init = tensor.empty() : {y_ty}
  %y_zero = linalg.fill ins(%zero : f32) outs(%y_init : {y_ty}) -> {y_ty}
  %y = linalg.generic {{
      indexing_maps = [
        affine_map<(b, o, g, k) -> (b, g, k)>,
        affine_map<(b, o, g, k) -> (o, g, k)>,
        affine_map<(b, o, g, k) -> (b, o)>
      ],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]}}
    ins(%xg, %w : {xg_ty}, {w_ty}) outs(%y_zero : {y_ty}) {{
    ^bb0(%xv: f32, %wv: f32, %acc: f32):
      %prod = arith.mulf %xv, %wv : f32
      %sum = arith.addf %acc, %prod : f32
      linalg.yield %sum : f32
  }} -> {y_ty}
  return %y : {y_ty}"#
    ));
    function.finish(module)
}

fn emit_gather(
    module: &mut ModuleBuilder,
    layout: BlockLayout,
    vocab: u32,
    width: u32,
) -> Result<()> {
    let block = layout.block_size();
    let block_bytes = layout.bytes_per_block();
    assert!(width.is_multiple_of(block));
    let blocks = width / block;
    let row_bytes = u64::from(blocks) * u64::from(block_bytes);
    let bytes = u64::from(vocab) * row_bytes;
    let data_ty = format!("tensor<{bytes}xi8>");
    let row_data_ty = format!("tensor<{row_bytes}xi8>");
    let row_ty = format!("tensor<1x{width}xf32>");
    let row3_ty = format!("tensor<1x{blocks}x{block}xf32>");

    let mut function = module.func_private(&gather_fn(layout, vocab, width));
    let data = function.arg("data", &data_ty);
    let index = function.arg("index", "index");
    function.result_ty(&row_ty);
    function.op_asm(format!(
        r#"  %row_bytes = arith.constant {row_bytes} : index
  %offset = arith.muli {index}, %row_bytes : index
  %row_data = tensor.extract_slice {data}[%offset] [{row_bytes}] [1] : {data_ty} to {row_data_ty}
"#
    ));
    emit_dequant(&mut function, layout, "%row_data", 1, blocks, "row3")?;
    function.op_asm(format!(
        "  %row = tensor.collapse_shape %row3 [[0], [1, 2]] : {row3_ty} into {row_ty}\n  return %row : {row_ty}"
    ));
    function.finish(module)
}

fn emit_dequant(
    function: &mut FuncBuilder,
    layout: BlockLayout,
    data: &str,
    output: u32,
    blocks: u32,
    result: &str,
) -> Result<()> {
    let block = layout.block_size();
    let block_bytes = layout.bytes_per_block();
    let bytes = u64::from(output) * u64::from(blocks) * u64::from(block_bytes);
    let words = block_bytes / 2;
    let data_ty = format!("tensor<{bytes}xi8>");
    let bytes3_ty = format!("tensor<{output}x{blocks}x{block_bytes}xi8>");
    let words_ty = format!("tensor<{output}x{blocks}x{words}xf16>");
    let result_ty = format!("tensor<{output}x{blocks}x{block}xf32>");
    function.op_asm(format!(
        "  %{result}_bytes = tensor.expand_shape {data} [[0, 1, 2]] output_shape [{output}, {blocks}, {block_bytes}] : {data_ty} into {bytes3_ty}\n  %{result}_words = iree_tensor_ext.bitcast {data} : {data_ty} -> {words_ty}\n  %{result}_init = tensor.empty() : {result_ty}\n"
    ));
    match layout {
        BlockLayout::Q40 => emit_q4_linalg(
            function, result, &bytes3_ty, &words_ty, &result_ty, 2, false,
        ),
        BlockLayout::Q41 => {
            emit_q4_linalg(function, result, &bytes3_ty, &words_ty, &result_ty, 4, true)
        }
        BlockLayout::Q80 => emit_q8_linalg(function, result, &bytes3_ty, &words_ty, &result_ty),
        BlockLayout::Q6K => emit_q6k_linalg(function, result, &bytes3_ty, &words_ty, &result_ty),
    }
    Ok(())
}

fn emit_q8_linalg(
    function: &mut FuncBuilder,
    result: &str,
    bytes_ty: &str,
    words_ty: &str,
    result_ty: &str,
) {
    let (output, blocks) = parse_block_dimensions(result_ty, 32);
    let quants_ty = format!("tensor<{output}x{blocks}x32xi8>");
    function.op_asm(format!(
        r#"  %{result}_quants = tensor.extract_slice %{result}_bytes[0, 0, 2] [{output}, {blocks}, 32] [1, 1, 1] : {bytes_ty} to {quants_ty}
  %{result} = linalg.generic {{
      indexing_maps = [
        affine_map<(o, b, k) -> (o, b, k)>,
        affine_map<(o, b, k) -> (o, b, 0)>,
        affine_map<(o, b, k) -> (o, b, k)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%{result}_quants, %{result}_words : {quants_ty}, {words_ty}) outs(%{result}_init : {result_ty}) {{
    ^bb0(%q: i8, %scale: f16, %unused: f32):
      %qi = arith.extsi %q : i8 to i32
      %qf = arith.sitofp %qi : i32 to f32
      %scale_f32 = arith.extf %scale : f16 to f32
      %dequant = arith.mulf %qf, %scale_f32 : f32
      linalg.yield %dequant : f32
  }} -> {result_ty}
"#
    ));
}

fn emit_q4_linalg(
    function: &mut FuncBuilder,
    result: &str,
    bytes_ty: &str,
    words_ty: &str,
    result_ty: &str,
    quant_offset: u32,
    has_minimum: bool,
) {
    let (output, blocks) = parse_block_dimensions(result_ty, 32);
    let qbytes_ty = format!("tensor<{output}x{blocks}x16xi8>");
    let qlanes_ty = format!("tensor<{output}x{blocks}x16x2xi4>");
    let result4_ty = format!("tensor<{output}x{blocks}x2x16xf32>");
    let minimum_map = if has_minimum {
        ",\n        affine_map<(o, b, h, l) -> (o, b, 1)>"
    } else {
        ""
    };
    let minimum_arg = if has_minimum {
        format!(", %{result}_words")
    } else {
        String::new()
    };
    let minimum_type = if has_minimum {
        format!(", {words_ty}")
    } else {
        String::new()
    };
    let minimum_bb = if has_minimum { ", %minimum: f16" } else { "" };
    let zero_or_min = if has_minimum {
        "      %bias = arith.extf %minimum : f16 to f32\n"
    } else {
        "      %bias = arith.constant 0.0 : f32\n"
    };
    function.op_asm(format!(
        r#"  %{result}_qbytes = tensor.extract_slice %{result}_bytes[0, 0, {quant_offset}] [{output}, {blocks}, 16] [1, 1, 1] : {bytes_ty} to {qbytes_ty}
  %{result}_qlanes = iree_tensor_ext.bitcast %{result}_qbytes : {qbytes_ty} -> {qlanes_ty}
  %{result}_4_init = tensor.empty() : {result4_ty}
  %{result}_4 = linalg.generic {{
      indexing_maps = [
        affine_map<(o, b, h, l) -> (o, b, l, h)>,
        affine_map<(o, b, h, l) -> (o, b, 0)>{minimum_map},
        affine_map<(o, b, h, l) -> (o, b, h, l)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}}
    ins(%{result}_qlanes, %{result}_words{minimum_arg} : {qlanes_ty}, {words_ty}{minimum_type}) outs(%{result}_4_init : {result4_ty}) {{
    ^bb0(%q: i4, %scale: f16{minimum_bb}, %unused: f32):
      %qi0 = arith.extui %q : i4 to i32
      %zero_point = arith.constant {zero_point} : i32
      %qi = arith.subi %qi0, %zero_point : i32
      %qf = arith.sitofp %qi : i32 to f32
      %scale_f32 = arith.extf %scale : f16 to f32
      %scaled = arith.mulf %qf, %scale_f32 : f32
{zero_or_min}      %dequant = arith.addf %scaled, %bias : f32
      linalg.yield %dequant : f32
  }} -> {result4_ty}
  %{result} = tensor.collapse_shape %{result}_4 [[0], [1], [2, 3]] : {result4_ty} into {result_ty}
"#,
        zero_point = if has_minimum { 0 } else { 8 },
    ));
}

fn emit_q6k_linalg(
    function: &mut FuncBuilder,
    result: &str,
    bytes_ty: &str,
    words_ty: &str,
    result_ty: &str,
) {
    let (output, blocks) = parse_block_dimensions(result_ty, 256);
    let ql_bytes_ty = format!("tensor<{output}x{blocks}x128xi8>");
    let qh_bytes_ty = format!("tensor<{output}x{blocks}x64xi8>");
    let scales_ty = format!("tensor<{output}x{blocks}x16xi8>");
    let ql_ty = format!("tensor<{output}x{blocks}x2x2x32x2xi4>");
    let qh_ty = format!("tensor<{output}x{blocks}x2x32x2x2xi2>");
    let scales6_ty = format!("tensor<{output}x{blocks}x2x2x2x2xi8>");
    let result7_ty = format!("tensor<{output}x{blocks}x2x2x2x2x16xf32>");
    function.op_asm(format!(
        r#"  %{result}_ql_bytes = tensor.extract_slice %{result}_bytes[0, 0, 0] [{output}, {blocks}, 128] [1, 1, 1] : {bytes_ty} to {ql_bytes_ty}
  %{result}_qh_bytes = tensor.extract_slice %{result}_bytes[0, 0, 128] [{output}, {blocks}, 64] [1, 1, 1] : {bytes_ty} to {qh_bytes_ty}
  %{result}_scale_bytes = tensor.extract_slice %{result}_bytes[0, 0, 192] [{output}, {blocks}, 16] [1, 1, 1] : {bytes_ty} to {scales_ty}
  %{result}_ql = iree_tensor_ext.bitcast %{result}_ql_bytes : {ql_bytes_ty} -> {ql_ty}
  %{result}_qh = iree_tensor_ext.bitcast %{result}_qh_bytes : {qh_bytes_ty} -> {qh_ty}
  %{result}_scales = tensor.expand_shape %{result}_scale_bytes [[0], [1], [2, 3, 4, 5]] output_shape [{output}, {blocks}, 2, 2, 2, 2] : {scales_ty} into {scales6_ty}
  %{result}_7_init = tensor.empty() : {result7_ty}
  %{result}_7 = linalg.generic {{
      indexing_maps = [
        affine_map<(o, b, n, h, p, s, l) -> (o, b, n, p, s * 16 + l, h)>,
        affine_map<(o, b, n, h, p, s, l) -> (o, b, n, s * 16 + l, h, p)>,
        affine_map<(o, b, n, h, p, s, l) -> (o, b, n, h, p, s)>,
        affine_map<(o, b, n, h, p, s, l) -> (o, b, 104)>,
        affine_map<(o, b, n, h, p, s, l) -> (o, b, n, h, p, s, l)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}}
    ins(%{result}_ql, %{result}_qh, %{result}_scales, %{result}_words : {ql_ty}, {qh_ty}, {scales6_ty}, {words_ty}) outs(%{result}_7_init : {result7_ty}) {{
    ^bb0(%low: i4, %high2: i2, %scale_byte: i8, %super_scale: f16, %unused: f32):
      %low_i32 = arith.extui %low : i4 to i32
      %high_i32 = arith.extui %high2 : i2 to i32
      %sixteen = arith.constant 16 : i32
      %high = arith.muli %high_i32, %sixteen : i32
      %q_i32_u = arith.addi %low_i32, %high : i32
      %thirty_two = arith.constant 32 : i32
      %q_i32 = arith.subi %q_i32_u, %thirty_two : i32
      %qf = arith.sitofp %q_i32 : i32 to f32
      %scale_i32 = arith.extsi %scale_byte : i8 to i32
      %scale_f32 = arith.sitofp %scale_i32 : i32 to f32
      %super_f32 = arith.extf %super_scale : f16 to f32
      %local_scaled = arith.mulf %qf, %scale_f32 : f32
      %dequant = arith.mulf %local_scaled, %super_f32 : f32
      linalg.yield %dequant : f32
  }} -> {result7_ty}
  %{result} = tensor.collapse_shape %{result}_7 [[0], [1], [2, 3, 4, 5, 6]] : {result7_ty} into {result_ty}
"#
    ));
}

fn parse_block_dimensions(result_ty: &str, block: u32) -> (u32, u32) {
    let body = result_ty
        .strip_prefix("tensor<")
        .and_then(|value| value.strip_suffix("xf32>"))
        .expect("internal static GGUF result tensor type");
    let dimensions: Vec<u32> = body
        .split('x')
        .map(|dimension| dimension.parse().expect("internal static GGUF dimension"))
        .collect();
    assert_eq!(dimensions.len(), 3);
    assert_eq!(dimensions[2], block);
    (dimensions[0], dimensions[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executable_gguf_helpers_verify() {
        let mut builder = ModuleBuilder::new().unwrap();
        emit_linear(&mut builder, BlockLayout::Q40, 4, 64, 32).unwrap();
        emit_gather(&mut builder, BlockLayout::Q40, 32, 64).unwrap();
        emit_linear(&mut builder, BlockLayout::Q41, 4, 64, 32).unwrap();
        emit_gather(&mut builder, BlockLayout::Q41, 32, 64).unwrap();
        emit_linear(&mut builder, BlockLayout::Q80, 4, 64, 32).unwrap();
        emit_gather(&mut builder, BlockLayout::Q80, 32, 64).unwrap();
        emit_linear(&mut builder, BlockLayout::Q6K, 4, 256, 32).unwrap();
        emit_gather(&mut builder, BlockLayout::Q6K, 32, 256).unwrap();
        let mlir = builder.finish().unwrap().mlir_text;
        assert!(mlir.contains("@gguf_q4_0_linear_4_64_32"));
        assert!(mlir.contains("@gguf_q4_1_linear_4_64_32"));
        assert!(mlir.contains("@gguf_q8_0_linear_4_64_32"));
        assert!(mlir.contains("@gguf_q6_k_linear_4_256_32"));
    }
}
