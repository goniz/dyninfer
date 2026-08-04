//! MLX affine group-quantization schemas.

use crate::{
    EmbeddingCallSpec, EncodingDefinitionDescriptor, ExternalEncodingTag, LinearCallSpec,
    ParameterLoweringOperation, ParameterLoweringSpec, QuantizationDefinition,
};
use dyninfer_checkpoint::LogicalParameter;
use dyninfer_core::{
    ParameterBinding, ParameterComponentBinding, PhysicalEncoding, ScalarType, StorageElementType,
    TensorOrder, ZeroPointMode,
};
use dyninfer_error::{DynInferError, Result, UnsupportedEncodingError};
use dyninfer_kernel_registry::{
    AxisMultiple, EncodingKey, KernelCandidateDescriptor, KernelOperationKind,
    ParameterOrientation, ProductionReadiness, ShapeConstraint, TargetConstraint,
};

#[derive(Debug, Clone, Copy)]
pub struct MlxAffineDefinition {
    bits: u8,
}

impl MlxAffineDefinition {
    pub const fn new(bits: u8) -> Self {
        Self { bits }
    }

    fn id(&self) -> String {
        format!("mlx.affine.u{}", self.bits)
    }
}

impl QuantizationDefinition for MlxAffineDefinition {
    fn descriptor(&self) -> EncodingDefinitionDescriptor {
        EncodingDefinitionDescriptor {
            key: EncodingKey::new(self.id(), 1),
            external_tags: vec![ExternalEncodingTag {
                family: "mlx.quantization.bits".into(),
                value: self.bits.to_string(),
            }],
        }
    }

    fn matches(&self, encoding: &PhysicalEncoding) -> bool {
        matches!(
            encoding,
            PhysicalEncoding::GroupQuantized {
                storage_bits,
                packing,
                ..
            } if storage_bits == &self.bits && packing == &self.id()
        )
    }

    fn validate(&self, parameter: &LogicalParameter) -> Result<()> {
        let PhysicalEncoding::GroupQuantized {
            logical_type,
            storage_bits,
            storage_container,
            signed,
            axis,
            group_size,
            scale_type,
            bias_type,
            zero_point,
            packing,
            order,
            components,
        } = &parameter.encoding
        else {
            return Err(self.unsupported(parameter, "expected group-quantized encoding"));
        };
        if !self.matches(&parameter.encoding)
            || logical_type != scale_type
            || storage_bits != &self.bits
            || storage_container != &ScalarType::U32
            || *signed
            || axis != &-1
            || *group_size == 0
            || bias_type != &Some(*scale_type)
            || zero_point != &ZeroPointMode::None
            || packing != &self.id()
            || order != &TensorOrder::RowMajor
            || components != &["packed", "scales", "biases"]
        {
            return Err(self.unsupported(parameter, "invalid MLX affine encoding descriptor"));
        }
        let [packed, scales, biases] = parameter.components.as_slice() else {
            return Err(self.unsupported(
                parameter,
                "MLX affine parameter requires packed/scales/biases components",
            ));
        };
        if packed.name != "packed" || scales.name != "scales" || biases.name != "biases" {
            return Err(self.unsupported(parameter, "MLX affine component names are invalid"));
        }
        if !matches!(
            packed.storage_type,
            StorageElementType::Scalar {
                ty: ScalarType::U32
            }
        ) || scales.storage_type != StorageElementType::scalar(*scale_type)
            || biases.storage_type != StorageElementType::scalar(*scale_type)
        {
            return Err(self.unsupported(parameter, "MLX affine component dtypes are invalid"));
        }
        let logical_shape = parameter.logical_type.shape.dims();
        if logical_shape.is_empty() || packed.shape.rank() != logical_shape.len() {
            return Err(self.unsupported(parameter, "MLX affine logical/packed ranks differ"));
        }
        let mut expected_packed = logical_shape.to_vec();
        let logical_last = *expected_packed.last().unwrap();
        let packed_numerator = logical_last
            .checked_mul(u64::from(self.bits))
            .ok_or_else(|| DynInferError::internal("MLX packed shape overflow"))?;
        if !packed_numerator.is_multiple_of(32) {
            return Err(self.unsupported(
                parameter,
                "MLX logical axis cannot be packed into complete u32 words",
            ));
        }
        *expected_packed.last_mut().unwrap() = packed_numerator / 32;
        let mut expected_groups = logical_shape.to_vec();
        if !logical_last.is_multiple_of(u64::from(*group_size)) {
            return Err(self.unsupported(parameter, "MLX logical axis is not group divisible"));
        }
        *expected_groups.last_mut().unwrap() = logical_last / u64::from(*group_size);
        if packed.shape.dims() != expected_packed
            || scales.shape.dims() != expected_groups
            || biases.shape.dims() != expected_groups
        {
            return Err(self.unsupported(parameter, "MLX affine component shapes are invalid"));
        }
        for component in [packed, scales, biases] {
            let element_bytes = match component.storage_type {
                StorageElementType::Scalar { ty } => ty.size_bytes().map(u64::from),
                _ => None,
            }
            .ok_or_else(|| self.unsupported(parameter, "component has unsized storage type"))?;
            let expected_bytes = component
                .shape
                .numel()
                .and_then(|numel| numel.checked_mul(element_bytes))
                .ok_or_else(|| DynInferError::internal("MLX component byte size overflow"))?;
            let actual_bytes = component
                .byte_ranges
                .iter()
                .try_fold(0u64, |total, range| total.checked_add(range.length))
                .ok_or_else(|| DynInferError::internal("MLX byte range size overflow"))?;
            if actual_bytes != expected_bytes {
                return Err(self.unsupported(parameter, "MLX component byte length is invalid"));
            }
        }
        Ok(())
    }

    fn kernel_candidates(&self) -> Vec<KernelCandidateDescriptor> {
        if self.bits != 4 {
            return vec![];
        }
        let operations = [
            (
                KernelOperationKind::Embedding,
                "embedding",
                "mlx.affine.u4.gather.iree_grouped",
            ),
            (
                KernelOperationKind::Linear,
                "matmul",
                "mlx.affine.u4.matmul.iree_grouped",
            ),
            (
                KernelOperationKind::OutputProjection,
                "output",
                "mlx.affine.u4.matmul.iree_grouped",
            ),
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
        ];
        operations
        .into_iter()
        .flat_map(|(operation, suffix, lowering)| {
            targets.clone().into_iter().map(move |(backend, target)| {
                (operation, suffix, lowering, backend, target)
            })
        })
        .map(|(operation, suffix, lowering, backend, target)| {
            let lowering = if backend == "cpu" {
                lowering.to_string()
            } else {
                format!("{lowering}_gpu")
            };
            KernelCandidateDescriptor {
                id: dyninfer_core::KernelId::new(format!(
                    "mlx.affine.u4.{suffix}.iree_grouped_{backend}"
                )),
                operation,
                encoding: Some(EncodingKey::new("mlx.affine.u4", 1)),
                input_types: vec![ScalarType::F32],
                output_types: vec![ScalarType::F32],
                accumulator_types: vec![ScalarType::F32],
                shape: ShapeConstraint {
                    logical_rank: Some(2),
                    axis_multiples: vec![AxisMultiple {
                        axis: 1,
                        multiple_of: 32,
                    }],
                },
                orientations: vec![ParameterOrientation::Native],
                target,
                modes: vec![
                    dyninfer_core::ExecutionMode::Prefill,
                    dyninfer_core::ExecutionMode::Decode,
                ],
                lowering: dyninfer_core::LoweringId::new(lowering),
                deterministic_score: 300,
                readiness: ProductionReadiness::Production,
                notes: format!("Direct MLX U4 words with groupwise scale/bias on {backend}; CPU permits fused dequantization while GPU uses an on-device dequantization boundary before contraction"),
            }
        })
        .collect()
    }

    fn emit_external_globals(
        &self,
        builder: &mut dyninfer_mlir::ModuleBuilder,
        binding: &ParameterBinding,
        symbol: &str,
    ) -> Result<bool> {
        if self.bits != 4 {
            return Ok(false);
        }
        for component_name in ["packed", "scales", "biases"] {
            let component = component(binding, component_name)?;
            builder.util_global_parameter(
                &format!("{symbol}_{component_name}"),
                &component.external_key,
                &component_tensor_type(component)?,
            )?;
        }
        Ok(true)
    }

    fn emit_parameter_load(
        &self,
        function: &mut dyninfer_mlir::FuncBuilder,
        binding: &ParameterBinding,
        ssa: &str,
        symbol: &str,
    ) -> Result<bool> {
        if self.bits != 4 {
            return Ok(false);
        }
        let [output, input] = binding.logical_shape.dims() else {
            return Err(DynInferError::internal(format!(
                "MLX affine parameter `{}` must have rank 2",
                binding.canonical_name
            )));
        };
        let group = mlx_group_size(binding)?;
        let groups = *input / u64::from(group);
        let packed = component(binding, "packed")?;
        let scales = component(binding, "scales")?;
        let biases = component(binding, "biases")?;
        let packed_ty = component_tensor_type(packed)?;
        let scales_ty = component_tensor_type(scales)?;
        let biases_ty = component_tensor_type(biases)?;
        let lanes_ty = format!("tensor<{output}x{groups}x{group}xi4>");
        function.op_asm(format!(
            "  %{ssa}_packed_words = util.global.load @{symbol}_packed : {packed_ty}\n"
        ));
        function.op_asm(format!(
            "  %{ssa}_packed = iree_tensor_ext.bitcast %{ssa}_packed_words : {packed_ty} -> {lanes_ty}\n"
        ));
        function.op_asm(format!(
            "  %{ssa}_scales_storage = util.global.load @{symbol}_scales : {scales_ty}\n"
        ));
        function.op_asm(format!(
            "  %{ssa}_biases_storage = util.global.load @{symbol}_biases : {biases_ty}\n"
        ));
        emit_aux_f32(
            function,
            &format!("{ssa}_scales"),
            &format!("{ssa}_scales_storage"),
            scales,
            &scales_ty,
            *output,
            groups,
        );
        emit_aux_f32(
            function,
            &format!("{ssa}_biases"),
            &format!("{ssa}_biases_storage"),
            biases,
            &biases_ty,
            *output,
            groups,
        );
        Ok(true)
    }

    fn helper_key(&self, spec: &ParameterLoweringSpec<'_>) -> Option<String> {
        let [output, input] = spec.binding.logical_shape.dims() else {
            return None;
        };
        let group = mlx_group_size(spec.binding).ok()?;
        let gpu = spec.lowering_id.as_str().ends_with("_gpu");
        match spec.operation {
            ParameterLoweringOperation::Embedding
                if matches_mlx_lowering(
                    spec.lowering_id.as_str(),
                    "mlx.affine.u4.gather.iree_grouped",
                ) =>
            {
                Some(format!(
                    "mlx-u4-gather-{output}-{input}-g{group}-{}",
                    if gpu { "gpu" } else { "cpu" }
                ))
            }
            ParameterLoweringOperation::Linear { rows }
                if matches_mlx_lowering(
                    spec.lowering_id.as_str(),
                    "mlx.affine.u4.matmul.iree_grouped",
                ) =>
            {
                Some(format!(
                    "mlx-u4-linear-{rows}-{input}-{output}-g{group}-{}",
                    if gpu { "gpu" } else { "cpu" }
                ))
            }
            ParameterLoweringOperation::OutputProjection
                if matches_mlx_lowering(
                    spec.lowering_id.as_str(),
                    "mlx.affine.u4.matmul.iree_grouped",
                ) =>
            {
                Some(format!(
                    "mlx-u4-linear-1-{input}-{output}-g{group}-{}",
                    if gpu { "gpu" } else { "cpu" }
                ))
            }
            _ => None,
        }
    }

    fn emit_helper(
        &self,
        module: &mut dyninfer_mlir::ModuleBuilder,
        spec: &ParameterLoweringSpec<'_>,
    ) -> Result<bool> {
        let Some(_) = self.helper_key(spec) else {
            return Ok(false);
        };
        let [output, input] = spec.binding.logical_shape.dims() else {
            unreachable!()
        };
        let (output, input) = (*output as u32, *input as u32);
        let group = mlx_group_size(spec.binding)?;
        let gpu = spec.lowering_id.as_str().ends_with("_gpu");
        match spec.operation {
            ParameterLoweringOperation::Embedding => {
                emit_gather_u4(module, &mlx_gather_fn(gpu), output, input, group)?;
            }
            ParameterLoweringOperation::Linear { rows } => {
                emit_linear_u4(
                    module,
                    &mlx_linear_fn(rows, input, output, group, gpu),
                    rows,
                    input,
                    output,
                    group,
                    gpu,
                )?;
            }
            ParameterLoweringOperation::OutputProjection => {
                emit_linear_u4(
                    module,
                    &mlx_linear_fn(1, input, output, group, gpu),
                    1,
                    input,
                    output,
                    group,
                    gpu,
                )?;
            }
        }
        Ok(true)
    }

    fn emit_linear_call(
        &self,
        function: &mut dyninfer_mlir::FuncBuilder,
        spec: &LinearCallSpec<'_>,
    ) -> Result<bool> {
        if self.helper_key(&spec.lowering).is_none() {
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
        let (output, input) = (*output as u32, *input as u32);
        let group = mlx_group_size(spec.lowering.binding)?;
        let groups = input / group;
        let gpu = spec.lowering.lowering_id.as_str().ends_with("_gpu");
        let helper = mlx_linear_fn(rows, input, output, group, gpu);
        function.op_asm(format!(
            "  %{} = func.call @{helper}(%{}, %{}_packed, %{}_scales, %{}_biases) : (tensor<{rows}x{input}xf32>, tensor<{output}x{groups}x{group}xi4>, tensor<{output}x{groups}xf32>, tensor<{output}x{groups}xf32>) -> tensor<{rows}x{output}xf32>\n",
            spec.result_ssa,
            spec.input_ssa,
            spec.parameter_ssa,
            spec.parameter_ssa,
            spec.parameter_ssa,
        ));
        Ok(true)
    }

    fn emit_embedding_call(
        &self,
        function: &mut dyninfer_mlir::FuncBuilder,
        spec: &EmbeddingCallSpec<'_>,
    ) -> Result<bool> {
        if self.helper_key(&spec.lowering).is_none() {
            return Ok(false);
        }
        let [vocab, width] = spec.lowering.binding.logical_shape.dims() else {
            unreachable!()
        };
        let group = mlx_group_size(spec.lowering.binding)?;
        let groups = *width / u64::from(group);
        let gpu = spec.lowering.lowering_id.as_str().ends_with("_gpu");
        let helper = mlx_gather_fn(gpu);
        function.op_asm(format!(
            "  %{} = func.call @{helper}(%{}_packed, %{}_scales, %{}_biases, %{}) : (tensor<{vocab}x{groups}x{group}xi4>, tensor<{vocab}x{groups}xf32>, tensor<{vocab}x{groups}xf32>, index) -> tensor<1x{width}xf32>\n",
            spec.result_ssa,
            spec.parameter_ssa,
            spec.parameter_ssa,
            spec.parameter_ssa,
            spec.index_ssa,
        ));
        Ok(true)
    }
}

fn component<'a>(
    binding: &'a ParameterBinding,
    name: &str,
) -> Result<&'a ParameterComponentBinding> {
    binding
        .components
        .iter()
        .find(|component| component.component_name == name)
        .ok_or_else(|| {
            DynInferError::internal(format!(
                "MLX affine parameter `{}` is missing `{name}`",
                binding.canonical_name
            ))
        })
}

fn mlx_group_size(binding: &ParameterBinding) -> Result<u32> {
    match binding.encoding {
        PhysicalEncoding::GroupQuantized {
            storage_bits: 4,
            group_size,
            ref packing,
            ..
        } if packing == "mlx.affine.u4" => Ok(group_size),
        _ => Err(DynInferError::internal(format!(
            "parameter `{}` is not MLX affine U4",
            binding.canonical_name
        ))),
    }
}

fn component_tensor_type(component: &ParameterComponentBinding) -> Result<String> {
    let shape = component
        .shape
        .dims()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("x");
    let StorageElementType::Scalar { ty } = component.storage_type else {
        return Err(DynInferError::internal(format!(
            "component `{}` has non-scalar storage {}",
            component.external_key, component.storage_type
        )));
    };
    let ty = match ty {
        ScalarType::U32 => "ui32".into(),
        other => other.to_string(),
    };
    Ok(format!("tensor<{shape}x{ty}>"))
}

fn emit_aux_f32(
    function: &mut dyninfer_mlir::FuncBuilder,
    result: &str,
    input: &str,
    component: &ParameterComponentBinding,
    storage_ty: &str,
    output: u64,
    groups: u64,
) {
    let f32_ty = format!("tensor<{output}x{groups}xf32>");
    match component.storage_type {
        StorageElementType::Scalar {
            ty: ScalarType::F32,
        } => function.op_asm(format!(
            "  %{result} = tensor.cast %{input} : {storage_ty} to {f32_ty}\n"
        )),
        _ => function.op_asm(format!(
            "  %{result} = arith.extf %{input} : {storage_ty} to {f32_ty}\n"
        )),
    };
}

fn matches_mlx_lowering(actual: &str, base: &str) -> bool {
    actual == base || actual == format!("{base}_gpu")
}

fn mlx_linear_fn(rows: u32, input: u32, output: u32, group: u32, gpu: bool) -> String {
    format!(
        "mlx_u4_linear_{rows}_{input}_{output}_g{group}_{}",
        if gpu { "gpu" } else { "cpu" }
    )
}

fn mlx_gather_fn(gpu: bool) -> String {
    format!(
        "mlx_u4_embedding_gather_{}",
        if gpu { "gpu" } else { "cpu" }
    )
}

fn emit_linear_u4(
    module: &mut dyninfer_mlir::ModuleBuilder,
    name: &str,
    rows: u32,
    input: u32,
    output: u32,
    group: u32,
    gpu: bool,
) -> Result<()> {
    assert!(group > 0 && input.is_multiple_of(group));
    let groups = input / group;
    let x_ty = format!("tensor<{rows}x{input}xf32>");
    let xg_ty = format!("tensor<{rows}x{groups}x{group}xf32>");
    let q_ty = format!("tensor<{output}x{groups}x{group}xi4>");
    let aux_ty = format!("tensor<{output}x{groups}xf32>");
    let w_ty = format!("tensor<{output}x{groups}x{group}xf32>");
    let y_ty = format!("tensor<{rows}x{output}xf32>");

    let mut function = module.func_private(name);
    let x = function.arg("x", &x_ty);
    let q = function.arg("q", &q_ty);
    let scales = function.arg("scales", &aux_ty);
    let biases = function.arg("biases", &aux_ty);
    function.result_ty(&y_ty);
    function.op_asm(format!(
        r#"  %xg = tensor.expand_shape {x} [[0], [1, 2]] output_shape [{rows}, {groups}, {group}] : {x_ty} into {xg_ty}
  %w_init = tensor.empty() : {w_ty}
  %w = linalg.generic {{
      indexing_maps = [
        affine_map<(o, g, k) -> (o, g, k)>,
        affine_map<(o, g, k) -> (o, g)>,
        affine_map<(o, g, k) -> (o, g)>,
        affine_map<(o, g, k) -> (o, g, k)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins({q}, {scales}, {biases} : {q_ty}, {aux_ty}, {aux_ty}) outs(%w_init : {w_ty}) {{
    ^bb0(%qv: i4, %scale: f32, %bias: f32, %unused: f32):
      %qi = arith.extui %qv : i4 to i32
      %qf = arith.uitofp %qi : i32 to f32
      %scaled = arith.mulf %qf, %scale : f32
      %dequant = arith.addf %scaled, %bias : f32
      linalg.yield %dequant : f32
  }} -> {w_ty}
  {barrier}
  %zero = arith.constant 0.0 : f32
  %y_init = tensor.empty() : {y_ty}
  %y_zero = linalg.fill ins(%zero : f32) outs(%y_init : {y_ty}) -> {y_ty}
  %y = linalg.generic {{
      indexing_maps = [
        affine_map<(b, o, g, k) -> (b, g, k)>,
        affine_map<(b, o, g, k) -> (o, g, k)>,
        affine_map<(b, o, g, k) -> (b, o)>
      ],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]}}
    ins(%xg, %{weight} : {xg_ty}, {w_ty}) outs(%y_zero : {y_ty}) {{
    ^bb0(%xv: f32, %wv: f32, %acc: f32):
      %prod = arith.mulf %xv, %wv : f32
      %sum = arith.addf %acc, %prod : f32
      linalg.yield %sum : f32
  }} -> {y_ty}
  return %y : {y_ty}"#,
        barrier = if gpu {
            format!("%w_ready = util.optimization_barrier %w : {w_ty}")
        } else {
            String::new()
        },
        weight = if gpu { "w_ready" } else { "w" },
    ));
    function.finish(module)
}

fn emit_gather_u4(
    module: &mut dyninfer_mlir::ModuleBuilder,
    name: &str,
    vocab: u32,
    width: u32,
    group: u32,
) -> Result<()> {
    assert!(group > 0 && width.is_multiple_of(group));
    let groups = width / group;
    let q_ty = format!("tensor<{vocab}x{groups}x{group}xi4>");
    let aux_ty = format!("tensor<{vocab}x{groups}xf32>");
    let qrow_ty = format!("tensor<1x{groups}x{group}xi4>");
    let auxrow_ty = format!("tensor<1x{groups}xf32>");
    let row3_ty = format!("tensor<1x{groups}x{group}xf32>");
    let row_ty = format!("tensor<1x{width}xf32>");

    let mut function = module.func_private(name);
    let q = function.arg("q", &q_ty);
    let scales = function.arg("scales", &aux_ty);
    let biases = function.arg("biases", &aux_ty);
    let index = function.arg("index", "index");
    function.result_ty(&row_ty);
    function.op_asm(format!(
        r#"  %qrow = tensor.extract_slice {q}[{index}, 0, 0] [1, {groups}, {group}] [1, 1, 1] : {q_ty} to {qrow_ty}
  %srow = tensor.extract_slice {scales}[{index}, 0] [1, {groups}] [1, 1] : {aux_ty} to {auxrow_ty}
  %brow = tensor.extract_slice {biases}[{index}, 0] [1, {groups}] [1, 1] : {aux_ty} to {auxrow_ty}
  %row_init = tensor.empty() : {row3_ty}
  %row3 = linalg.generic {{
      indexing_maps = [
        affine_map<(b, g, k) -> (b, g, k)>,
        affine_map<(b, g, k) -> (b, g)>,
        affine_map<(b, g, k) -> (b, g)>,
        affine_map<(b, g, k) -> (b, g, k)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%qrow, %srow, %brow : {qrow_ty}, {auxrow_ty}, {auxrow_ty}) outs(%row_init : {row3_ty}) {{
    ^bb0(%qv: i4, %scale: f32, %bias: f32, %unused: f32):
      %qi = arith.extui %qv : i4 to i32
      %qf = arith.uitofp %qi : i32 to f32
      %scaled = arith.mulf %qf, %scale : f32
      %dequant = arith.addf %scaled, %bias : f32
      linalg.yield %dequant : f32
  }} -> {row3_ty}
  %row = tensor.collapse_shape %row3 [[0], [1, 2]] : {row3_ty} into {row_ty}
  return %row : {row_ty}"#
    ));
    function.finish(module)
}

impl MlxAffineDefinition {
    fn unsupported(
        &self,
        parameter: &LogicalParameter,
        message: impl Into<String>,
    ) -> DynInferError {
        DynInferError::UnsupportedEncoding(UnsupportedEncodingError {
            message: message.into(),
            key: Some(parameter.canonical_name.to_string()),
            codec: Some(self.id()),
            codec_version: Some(1),
            expected: Some(format!(
                "MLX unsigned {}-bit affine groups in u32 storage",
                self.bits
            )),
            actual: Some(format!("{:?}", parameter.encoding)),
        })
    }
}

pub const SCHEMA_DEFINITIONS: &[MlxAffineDefinition] = &[
    MlxAffineDefinition::new(2),
    MlxAffineDefinition::new(3),
    MlxAffineDefinition::new(4),
    MlxAffineDefinition::new(6),
    MlxAffineDefinition::new(8),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_owned_grouped_u4_lowerings_verify() {
        let mut builder = dyninfer_mlir::ModuleBuilder::new().unwrap();
        emit_linear_u4(&mut builder, "mlx_linear", 4, 64, 32, 32, false).unwrap();
        emit_gather_u4(&mut builder, "mlx_gather", 32, 64, 32).unwrap();
        let mlir = builder.finish().unwrap().mlir_text;
        assert!(mlir.contains("tensor<32x2x32xi4>"));
        assert!(mlir.contains("arith.extui") && mlir.contains("i4 to i32"));
        assert!(mlir.contains(
            "iterator_types = [\"parallel\", \"parallel\", \"reduction\", \"reduction\"]"
        ));
    }

    #[test]
    fn gpu_grouped_u4_linear_has_dequantization_fusion_boundary() {
        let mut builder = dyninfer_mlir::ModuleBuilder::new().unwrap();
        emit_linear_u4(&mut builder, "mlx_linear_gpu", 1, 64, 32, 32, true).unwrap();
        let mlir = builder.finish().unwrap().mlir_text;
        assert!(mlir.contains("util.optimization_barrier"));
        assert!(mlir.contains("arith.extui") && mlir.contains("i4 to i32"));
    }
}
