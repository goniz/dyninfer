//! Encoding-independent dispatch from selected parameter lowerings to the
//! implementation object registered by `dyninfer-quantization`.

use super::{dense_decoder::DenseDecoderConfig, kernels};
use dyninfer_core::{ExecutionMode, LoweringId, PhysicalEncoding, ScalarType};
use dyninfer_error::{CompilationError, DynInferError, Result};
use dyninfer_kernel_registry::KernelOperationKind;
use dyninfer_mlir::{FuncBuilder, ModuleBuilder};
use dyninfer_quantization::{
    EmbeddingCallSpec, LinearCallSpec, ParameterLoweringOperation, ParameterLoweringSpec,
    QuantizationDefinition, QuantizationRegistry,
};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::OnceLock;

pub struct ParameterLowerings {
    registry: QuantizationRegistry,
}

pub fn default_parameter_lowerings() -> &'static ParameterLowerings {
    static LOWERINGS: OnceLock<ParameterLowerings> = OnceLock::new();
    LOWERINGS.get_or_init(|| {
        ParameterLowerings::new().expect("built-in quantization registrations must be valid")
    })
}

impl ParameterLowerings {
    pub fn new() -> Result<Self> {
        let mut registry = QuantizationRegistry::new();
        dyninfer_quantization::register_all(&mut registry)?;
        Ok(Self { registry })
    }

    pub fn emit_global(
        &self,
        builder: &mut ModuleBuilder,
        config: &DenseDecoderConfig,
        symbol: &str,
        canonical: &str,
        shape: &str,
    ) -> Result<()> {
        if let Some(binding) = config.param_binding(canonical) {
            if !matches!(binding.encoding, PhysicalEncoding::Plain { .. }) {
                let definition = self.definition(binding)?;
                if definition.emit_external_globals(builder, binding, symbol)? {
                    return Ok(());
                }
                return Err(no_lowering(canonical, "external-global declaration"));
            }
        }
        let key = config.param_key(canonical);
        let storage = mlir_ty(config.param_dtype(canonical));
        builder.util_global_parameter(symbol, &key, &format!("tensor<{shape}x{storage}>"))
    }

    pub fn emit_load(
        &self,
        function: &mut FuncBuilder,
        config: &DenseDecoderConfig,
        ssa: &str,
        symbol: &str,
        canonical: &str,
        shape: &str,
    ) -> Result<()> {
        if let Some(binding) = config.param_binding(canonical) {
            if !matches!(binding.encoding, PhysicalEncoding::Plain { .. }) {
                let definition = self.definition(binding)?;
                if definition.emit_parameter_load(function, binding, ssa, symbol)? {
                    return Ok(());
                }
                return Err(no_lowering(canonical, "external-component load"));
            }
        }
        let storage = config.param_dtype(canonical);
        let compute = config.param_compute_dtype(canonical);
        let storage_ty = mlir_ty(storage);
        let compute_ty = mlir_ty(compute);
        match (storage, compute) {
            (ScalarType::F16 | ScalarType::Bf16, ScalarType::F32)
            | (ScalarType::F32, ScalarType::F32) => {
                kernels::load_compute(function, ssa, symbol, &storage_ty, &compute_ty, shape);
                Ok(())
            }
            _ => Err(no_lowering(
                canonical,
                &format!("storage-to-compute conversion {storage}->{compute}"),
            )),
        }
    }

    pub fn emit_quantized_helpers(
        &self,
        module: &mut ModuleBuilder,
        config: &DenseDecoderConfig,
    ) -> Result<()> {
        let mut emitted = BTreeSet::new();
        for (canonical, binding) in &config.param_bindings {
            if matches!(binding.encoding, PhysicalEncoding::Plain { .. }) {
                continue;
            }
            let definition = self.definition(binding)?;
            for mode in [ExecutionMode::Prefill, ExecutionMode::Decode] {
                let Some(lowering) = config.param_lowering(canonical, mode) else {
                    continue;
                };
                let operation = if canonical == "token_embd.weight" {
                    ParameterLoweringOperation::Embedding
                } else if canonical == "output.weight" {
                    ParameterLoweringOperation::OutputProjection
                } else if binding.logical_shape.rank() == 2 {
                    ParameterLoweringOperation::Linear {
                        rows: if mode == ExecutionMode::Prefill {
                            config.seq
                        } else {
                            1
                        },
                    }
                } else {
                    continue;
                };
                let lowering_id = LoweringId::new(lowering);
                let spec = ParameterLoweringSpec {
                    binding,
                    lowering_id: &lowering_id,
                    mode,
                    operation,
                };
                let Some(key) = definition.helper_key(&spec) else {
                    return Err(no_lowering(canonical, "selected helper"));
                };
                if emitted.insert(key) && !definition.emit_helper(module, &spec)? {
                    return Err(no_lowering(canonical, "selected helper"));
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn emit_linear_call(
        &self,
        function: &mut FuncBuilder,
        config: &DenseDecoderConfig,
        result: &str,
        dense_function: &str,
        input_ssa: &str,
        parameter_ssa: &str,
        canonical: &str,
        mode: ExecutionMode,
        rows: u32,
        input: u32,
        output: u32,
    ) -> Result<()> {
        if let Some(binding) = config.param_binding(canonical) {
            if !matches!(binding.encoding, PhysicalEncoding::Plain { .. }) {
                let lowering_id =
                    LoweringId::new(config.param_lowering(canonical, mode).ok_or_else(|| {
                        no_lowering(canonical, "operation-local selected lowering")
                    })?);
                let definition = self.definition(binding)?;
                let spec = LinearCallSpec {
                    lowering: ParameterLoweringSpec {
                        binding,
                        lowering_id: &lowering_id,
                        mode,
                        operation: ParameterLoweringOperation::Linear { rows },
                    },
                    result_ssa: result,
                    input_ssa,
                    parameter_ssa,
                };
                if definition.emit_linear_call(function, &spec)? {
                    return Ok(());
                }
                return Err(no_lowering(canonical, "linear call"));
            }
        }
        function.op_asm(format!(
            "  %{result} = func.call @{dense_function}(%{input_ssa}, %{parameter_ssa}) : (tensor<{rows}x{input}xf32>, tensor<{output}x{input}xf32>) -> tensor<{rows}x{output}xf32>\n"
        ));
        Ok(())
    }

    pub fn emit_embedding_call(
        &self,
        function: &mut FuncBuilder,
        config: &DenseDecoderConfig,
        result: &str,
        index_ssa: &str,
        parameter_ssa: &str,
        mode: ExecutionMode,
    ) -> Result<bool> {
        let Some(binding) = config.param_binding("token_embd.weight") else {
            return Ok(false);
        };
        if matches!(binding.encoding, PhysicalEncoding::Plain { .. }) {
            return Ok(false);
        }
        let lowering_id = LoweringId::new(
            config
                .param_lowering("token_embd.weight", mode)
                .ok_or_else(|| no_lowering("token_embd.weight", "embedding lowering"))?,
        );
        let definition = self.definition(binding)?;
        let spec = EmbeddingCallSpec {
            lowering: ParameterLoweringSpec {
                binding,
                lowering_id: &lowering_id,
                mode,
                operation: ParameterLoweringOperation::Embedding,
            },
            result_ssa: result,
            index_ssa,
            parameter_ssa,
        };
        if definition.emit_embedding_call(function, &spec)? {
            Ok(true)
        } else {
            Err(no_lowering("token_embd.weight", "embedding call"))
        }
    }

    pub fn emit_output_projection(
        &self,
        function: &mut FuncBuilder,
        config: &DenseDecoderConfig,
        hidden_ssa: &str,
        parameter_ssa: &str,
        mode: ExecutionMode,
    ) -> Result<bool> {
        let Some(binding) = config.param_binding("output.weight") else {
            return Ok(false);
        };
        if matches!(binding.encoding, PhysicalEncoding::Plain { .. }) {
            return Ok(false);
        }
        let lowering_id = LoweringId::new(
            config
                .param_lowering("output.weight", mode)
                .ok_or_else(|| no_lowering("output.weight", "output-projection lowering"))?,
        );
        let definition = self.definition(binding)?;
        let spec = LinearCallSpec {
            lowering: ParameterLoweringSpec {
                binding,
                lowering_id: &lowering_id,
                mode,
                operation: ParameterLoweringOperation::OutputProjection,
            },
            result_ssa: "y",
            input_ssa: hidden_ssa,
            parameter_ssa,
        };
        if definition.emit_linear_call(function, &spec)? {
            function.op_asm(format!(
                "  %logits = tensor.collapse_shape %y [[0, 1]] : tensor<1x{}xf32> into tensor<{}xf32>\n",
                config.vocab, config.vocab
            ));
            Ok(true)
        } else {
            Err(no_lowering("output.weight", "output-projection call"))
        }
    }

    fn definition(
        &self,
        binding: &dyninfer_core::ParameterBinding,
    ) -> Result<Arc<dyn QuantizationDefinition>> {
        self.registry.resolve(&binding.encoding).ok_or_else(|| {
            no_lowering(
                binding.canonical_name.as_str(),
                "registered encoding definition",
            )
        })
    }

    fn implements_lowering(&self, operation: KernelOperationKind, lowering: &str) -> bool {
        self.registry.definitions().iter().any(|definition| {
            definition.kernel_candidates().iter().any(|candidate| {
                candidate.operation == operation && candidate.lowering.as_str() == lowering
            })
        })
    }
}

pub(crate) fn registered_parameter_lowering_matches(
    operation: &dyninfer_core::OperationKind,
    lowering: &str,
) -> bool {
    default_parameter_lowerings()
        .implements_lowering(KernelOperationKind::from_semantic(operation), lowering)
}

fn mlir_ty(ty: ScalarType) -> String {
    match ty {
        ScalarType::U8 => "ui8".into(),
        ScalarType::U16 => "ui16".into(),
        ScalarType::U32 => "ui32".into(),
        ScalarType::U64 => "ui64".into(),
        ScalarType::Bool => "i1".into(),
        other => other.to_string(),
    }
}

fn no_lowering(canonical: &str, requirement: &str) -> DynInferError {
    DynInferError::Compilation(CompilationError {
        message: format!(
            "parameter `{canonical}` has no registered implementation for {requirement}"
        ),
        pass: Some("lower.parameter.dispatch".into()),
        diagnostics: vec![],
    })
}
