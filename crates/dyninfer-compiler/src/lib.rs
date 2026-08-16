//! Safe Rust compiler wrapper — prefers in-process `libIREECompiler`, with
//! `iree-compile` subprocess fallback.
//!
//! Architecture-specific MLIR is supplied by the caller (from
//! `dyninfer-architecture`); this crate only drives IREE.

#![forbid(unsafe_code)]

mod iree_tools;
mod lowering;
mod mlir_emit;

pub use iree_tools::IreeTools;
pub use lowering::{
    DenseDecoderConfig, LARGE_PREFILL_WINDOW, PAGED_PREFILL_CHUNK_SIZE, PREFILL_WINDOW,
    TINY_PREFILL_WINDOW,
};
pub use mlir_emit::{emit_add_smoke_module, emit_bridge_module};

use dyninfer_architecture::ArchitecturePackage;
use dyninfer_core::{
    BindingPlan, BoundModel, ExecutableManifest, ExecutionMode, KernelId, KvCacheDescriptor,
    KvCacheLayout, KvCacheStorage, LoweringId, ManifestParameterComponent, OperationKind,
    PrecisionPolicy, ScalarType, SchemaFingerprint, SelectedKernel, ShapeProfile,
    SpecializedExecutionShape, TargetProfile,
};
use dyninfer_error::{CompilationError, Diagnostic, DynInferError, Result, Severity};
use serde::{Deserialize, Serialize};
use tracing::{info, info_span};

pub const COMPILER_VERSION: &str = "0.3.0-iree-3.11.0-paged-kv-v11.01-bool-mask";
/// Pinned IREE pip / source revision identity for executable cache keys (spec §19.1).
pub const IREE_REVISION: &str = "3.11.0+e4a3b0405d7d";
/// Kernel registry policy version included in executable cache keys.
pub const KERNEL_REGISTRY_VERSION: &str = dyninfer_kernel_registry::KernelRegistry::VERSION;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompileOptions {
    pub mode: String,
    pub dump_ir: bool,
    /// If true, compile only the trivial `@add` smoke module.
    #[serde(default)]
    pub smoke_only: bool,
    /// Force subprocess `iree-compile` even when the shared library is available.
    #[serde(default)]
    pub force_subprocess: bool,
}

#[derive(Debug, Clone)]
pub struct VmfbArtifact {
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct CompileRequest<'a> {
    pub bound_model: &'a BoundModel,
    pub architecture_revision: &'a str,
    pub checkpoint_schema: &'a SchemaFingerprint,
    pub shape_profile: &'a ShapeProfile,
    pub options: &'a CompileOptions,
}

#[derive(Debug, Clone)]
pub struct CompileOutput {
    pub executable: VmfbArtifact,
    pub decode_executable: Option<VmfbArtifact>,
    pub manifest: ExecutableManifest,
    pub diagnostics: Vec<Diagnostic>,
    pub mlir_text: String,
}

#[derive(Debug, Clone)]
pub struct LoweringOutput {
    pub mlir_text: String,
    pub decode_mlir_text: Option<String>,
    pub prefill_window: u32,
    pub max_kv: u32,
    pub page_size: u32,
    pub paged_kv: bool,
    pub num_layers: u32,
    pub kv_element_type: ScalarType,
}

/// Resolve the bounded static shapes used by the initial prefill/decode ABI.
/// User-provided config wins; otherwise deterministic size buckets keep local
/// compilation tractable without inspecting checkpoint payloads.
pub fn default_shape_profile(architecture: &ArchitecturePackage) -> ShapeProfile {
    let values = &architecture.resolved_config.values;
    let get = |key: &str| {
        values
            .get(key)
            .and_then(|value| value.as_u64())
            .map(|v| v as u32)
    };
    let vocab = get("vocab_size").unwrap_or(32);
    let hidden = get("hidden_size").unwrap_or(64);
    let heads = get("num_heads").unwrap_or(4);
    let layers = get("num_layers").unwrap_or(1);
    let synthetic = vocab == 32 && hidden == 64 && heads == 4;
    let large = vocab > 50_000 || layers > 16 || hidden >= 1024;
    let default_prefill = if synthetic {
        TINY_PREFILL_WINDOW
    } else if large {
        LARGE_PREFILL_WINDOW
    } else {
        PREFILL_WINDOW
    };
    let default_max_kv = if synthetic {
        lowering::TINY_MAX_KV
    } else if large {
        lowering::LARGE_MAX_KV
    } else {
        lowering::PREFILL_MAX_KV
    };
    let prefill = get("prefill_window").unwrap_or(default_prefill).max(1);
    let max_kv = get("max_kv").unwrap_or(default_max_kv).max(prefill);
    ShapeProfile {
        batch_sizes: vec![1],
        sequence_buckets: vec![prefill],
        max_sequence_length: max_kv,
        extra: None,
    }
}

pub fn build_bound_model(
    architecture: &ArchitecturePackage,
    binding: &BindingPlan,
    target: &TargetProfile,
    precision_policy: &PrecisionPolicy,
    mut selected_kernels: Vec<SelectedKernel>,
    shape_profile: &ShapeProfile,
) -> Result<BoundModel> {
    dyninfer_architecture::verify_architecture_graph(&architecture.graph)?;
    let batch_size = shape_profile.batch_sizes.first().copied().unwrap_or(1);
    let requested_prefill = shape_profile
        .sequence_buckets
        .first()
        .copied()
        .unwrap_or(1)
        .max(1);
    let max_kv = shape_profile.max_sequence_length.max(requested_prefill);
    let paged = max_kv > PAGED_PREFILL_CHUNK_SIZE;
    let prefill = if paged {
        lowering::PAGED_PREFILL_CHUNK_SIZE
    } else {
        requested_prefill
    };
    if paged {
        let attention_ops: std::collections::BTreeSet<_> = architecture
            .graph
            .operations
            .iter()
            .filter(|operation| matches!(operation.kind, OperationKind::Attention { .. }))
            .map(|operation| operation.id.clone())
            .collect();
        for selected in &mut selected_kernels {
            if attention_ops.contains(&selected.operation_id) {
                selected.kernel_id = KernelId::new("attention.online_paged.generated.f32");
                selected.lowering_id = LoweringId::new("attention.online_paged.generated");
            }
        }
    }
    Ok(BoundModel {
        version: 1,
        architecture: architecture.graph.clone(),
        resolved_config: architecture.resolved_config.values.clone(),
        binding: binding.clone(),
        execution_shapes: vec![
            SpecializedExecutionShape {
                mode: ExecutionMode::Prefill,
                batch_size,
                sequence_length: prefill,
                max_kv_length: max_kv,
            },
            SpecializedExecutionShape {
                mode: ExecutionMode::Decode,
                batch_size,
                sequence_length: 1,
                max_kv_length: max_kv,
            },
        ],
        target: target.clone(),
        precision_policy: precision_policy.clone(),
        selected_kernels,
    })
}

pub fn lower_bound_model(bound: &BoundModel) -> Result<LoweringOutput> {
    validate_bound_model_lowerings(bound)?;
    reject_operations_outside_dense_decoder(bound)?;
    let config = DenseDecoderConfig::from_bound_model(bound);
    if let Some(binding) = bound.binding.bindings.iter().find(|binding| {
        !config
            .param_compute_dtypes
            .contains_key(binding.canonical_name.as_str())
    }) {
        return Err(DynInferError::Compilation(CompilationError {
            message: format!(
                "parameter `{}` has no operation-local compute dtype selection",
                binding.canonical_name
            ),
            pass: Some("lower.precision".into()),
            diagnostics: vec![],
        }));
    }
    if !config.supports_dense_emit() {
        return Err(DynInferError::Compilation(CompilationError {
            message: format!(
                "shared transformer lowering does not support specialization {config:?}"
            ),
            pass: Some("lower.bound_model".into()),
            diagnostics: vec![],
        }));
    }
    let (mlir_text, decode_mlir_text) = if config.paged_kv {
        if config.has_short_conv() {
            // Hybrid short-conv keeps rolling `conv_state*` util.globals that
            // must be shared by prefill and decode — emit one combined module.
            (
                lowering::emit_dense_decoder_cfg_program(
                    bound.architecture.architecture_id.as_str(),
                    &config,
                    lowering::PagedProgram::Combined,
                )?,
                None,
            )
        } else {
            (
                lowering::emit_dense_decoder_cfg_program(
                    bound.architecture.architecture_id.as_str(),
                    &config,
                    lowering::PagedProgram::Prefill,
                )?,
                Some(lowering::emit_dense_decoder_cfg_program(
                    bound.architecture.architecture_id.as_str(),
                    &config,
                    lowering::PagedProgram::Decode,
                )?),
            )
        }
    } else {
        (
            lowering::emit_dense_decoder_cfg(bound.architecture.architecture_id.as_str(), &config)?,
            None,
        )
    };
    Ok(LoweringOutput {
        mlir_text,
        decode_mlir_text,
        prefill_window: config.seq,
        max_kv: config.max_kv,
        page_size: config.page_size,
        paged_kv: config.paged_kv,
        num_layers: config.num_layers,
        kv_element_type: config.paged_kv_element_type(),
    })
}

/// Reject operators the shared dense decoder emitter cannot lower. Short-conv
/// hybrid schedules are supported via [`DenseDecoderConfig::resolved_layer_types`];
/// keep this hook for future operator kinds that still need an explicit refuse.
fn reject_operations_outside_dense_decoder(_bound: &BoundModel) -> Result<()> {
    Ok(())
}

fn validate_bound_model_lowerings(bound: &BoundModel) -> Result<()> {
    let modes = bound
        .architecture
        .exports
        .iter()
        .map(|export| export.mode)
        .collect::<std::collections::BTreeSet<_>>();
    for operation in &bound.architecture.operations {
        for mode in &modes {
            let selected = bound
                .selected_kernels
                .iter()
                .find(|selected| selected.operation_id == operation.id && selected.mode == *mode)
                .ok_or_else(|| {
                    DynInferError::Compilation(CompilationError {
                        message: format!(
                            "operation `{}` has no selected lowering for {:?}",
                            operation.id, mode
                        ),
                        pass: Some("lower.bound_model".into()),
                        diagnostics: vec![],
                    })
                })?;
            if selected.input_type != ScalarType::I64
                && (selected.input_type != ScalarType::F32
                    || selected.output_type != ScalarType::F32
                    || selected.activation_type != ScalarType::F32
                    || selected.accumulator_type != ScalarType::F32)
            {
                return Err(DynInferError::Compilation(CompilationError {
                    message: format!(
                        "selected kernel `{}` requires {:?}/{:?}/{:?}/{:?}, but the current shared lowering is qualified only for f32 compute",
                        selected.kernel_id,
                        selected.input_type,
                        selected.output_type,
                        selected.activation_type,
                        selected.accumulator_type
                    ),
                    pass: Some("lower.precision".into()),
                    diagnostics: vec![],
                }));
            }
            if !lowering_matches_operation(&operation.kind, selected.lowering_id.as_str()) {
                return Err(DynInferError::Compilation(CompilationError {
                    message: format!(
                        "selected lowering `{}` does not implement operation `{}` ({:?})",
                        selected.lowering_id, operation.id, operation.kind
                    ),
                    pass: Some("lower.dispatch".into()),
                    diagnostics: vec![],
                }));
            }
        }
    }
    Ok(())
}

fn lowering_matches_operation(operation: &OperationKind, lowering: &str) -> bool {
    match operation {
        OperationKind::Input { .. } => lowering == "model.input.abi",
        OperationKind::Embedding => {
            lowering == "dense.gather.linalg"
                || lowering::parameter::registered_parameter_lowering_matches(operation, lowering)
        }
        OperationKind::Linear { .. } | OperationKind::OutputProjection => {
            lowering == "dense.matmul.linalg"
                || lowering::parameter::registered_parameter_lowering_matches(operation, lowering)
        }
        OperationKind::RmsNorm { .. } => lowering == "dense.rms_norm.generated",
        OperationKind::PerHeadRmsNorm { .. } => lowering == "dense.per_head_rms_norm.generated",
        OperationKind::Rope { .. } => lowering == "rope.generated",
        OperationKind::KvCacheWrite { .. } => lowering == "kv_cache.write.generated",
        OperationKind::KvCacheRead { .. } => lowering == "kv_cache.read.generated",
        OperationKind::Attention { .. } => {
            lowering == "attention.gqa.generated" || lowering == "attention.online_paged.generated"
        }
        OperationKind::ShortConv { .. } => lowering == "short_conv.gated.generated",
        OperationKind::Elementwise {
            function: dyninfer_core::ElementwiseFunction::Silu,
        } => lowering == "elementwise.silu.generated",
        OperationKind::Elementwise {
            function: dyninfer_core::ElementwiseFunction::Multiply,
        } => lowering == "elementwise.multiply.generated",
        OperationKind::Residual => lowering == "residual.add.generated",
    }
}

pub trait ModelCompiler: Send + Sync {
    fn compile(&self, request: &CompileRequest<'_>) -> Result<CompileOutput>;
}

/// Local compiler using in-process IREE (bindgen + libIREECompiler) by default.
pub struct LocalCompiler {
    tools: Option<IreeTools>,
    force_subprocess: bool,
}

impl LocalCompiler {
    pub fn new(options: &CompileOptions) -> Result<Self> {
        let tools = match IreeTools::discover() {
            Ok(t) => Some(t),
            Err(_) if !options.force_subprocess => None,
            Err(e) => return Err(e),
        };
        Ok(Self {
            tools,
            force_subprocess: options.force_subprocess,
        })
    }

    pub fn tools(&self) -> Option<&IreeTools> {
        self.tools.as_ref()
    }

    /// Compile an arbitrary MLIR string to VMFB for the given target.
    pub fn compile_mlir(&self, mlir: &str, target: &TargetProfile) -> Result<Vec<u8>> {
        compile_mlir_prefer_inprocess(mlir, target, self.force_subprocess, self.tools.as_ref())
    }
}

fn compile_flags_for(target: &TargetProfile) -> Result<Vec<String>> {
    if !target.is_compile_ready() || target.executable_target_flags.is_empty() {
        return Err(DynInferError::Compilation(CompilationError {
            message: format!(
                "target `{}` is not backed by verified local compile facts (architecture={:?})",
                target.driver, target.architecture
            ),
            pass: Some("target.validate".into()),
            diagnostics: vec![],
        }));
    }
    let mut flags = target.executable_target_flags.clone();
    if matches!(target.driver.as_str(), "hip" | "rocm") {
        if let Some(bc) = iree_compiler_sys::discover_rocm_bc_dir() {
            flags.push(format!("--iree-rocm-bc-dir={}", bc.display()));
        }
    } else if matches!(target.driver.as_str(), "local" | "local-task" | "local-sync")
        && !flags
            .iter()
            .any(|flag| flag.starts_with("--iree-llvmcpu-embedded-linker-path="))
        && let Some(sdk) = dyninfer_rocm::RocmSdk::discover()
    {
        flags.push(format!(
            "--iree-llvmcpu-embedded-linker-path={}",
            sdk.linker().display()
        ));
    }
    Ok(flags)
}

fn compile_mlir_prefer_inprocess(
    mlir: &str,
    target: &TargetProfile,
    force_subprocess: bool,
    tools: Option<&IreeTools>,
) -> Result<Vec<u8>> {
    let flags = compile_flags_for(target)?;
    // The ROCm backend locates lld from the process-wide PATH. The Bazel-pinned
    // TheRock SDK lives in a separate runfiles tree, so use the subprocess path
    // where we can configure PATH without mutating a multithreaded process.
    let inprocess_supported = !matches!(target.driver.as_str(), "hip" | "rocm");
    if !force_subprocess && inprocess_supported {
        match iree_compiler_sys::compile_mlir_to_vmfb(mlir, &flags) {
            Ok(bytes) => {
                info!(
                    bytes = bytes.len(),
                    rev = iree_compiler_sys::revision().unwrap_or_default(),
                    driver = %target.driver,
                    "compiled via in-process libIREECompiler"
                );
                return Ok(bytes);
            }
            Err(e) => {
                info!(error = %e, "in-process IREE compile unavailable; trying iree-compile");
            }
        }
    }
    let tools = tools.ok_or_else(|| {
        DynInferError::Compilation(CompilationError {
            message: "IREE compiler unavailable (libIREECompiler + iree-compile)".into(),
            pass: Some("iree-compile".into()),
            diagnostics: vec![],
        })
    })?;
    tools.compile_mlir_with_flags(mlir, &flags)
}

impl ModelCompiler for LocalCompiler {
    fn compile(&self, request: &CompileRequest<'_>) -> Result<CompileOutput> {
        let _span = info_span!(
            "compile.specialize",
            architecture = %request.bound_model.architecture.architecture_id,
            target = %request.bound_model.target.driver
        )
        .entered();

        let lowering = if request.options.smoke_only {
            LoweringOutput {
                mlir_text: emit_add_smoke_module().to_string(),
                decode_mlir_text: None,
                prefill_window: 4,
                max_kv: 4,
                page_size: lowering::PAGED_KV_PAGE_SIZE,
                paged_kv: false,
                num_layers: 0,
                kv_element_type: ScalarType::F32,
            }
        } else {
            lower_bound_model(request.bound_model)?
        };
        let mlir = lowering.mlir_text;
        let decode_mlir = lowering.decode_mlir_text;

        let _iree = info_span!("compile.iree").entered();
        let annotate_error = |err| match err {
            DynInferError::Compilation(mut c) => {
                c.diagnostics
                    .push(Diagnostic::error("E_IREE_COMPILE", c.message.clone()));
                DynInferError::Compilation(c)
            }
            other => other,
        };
        let (vmfb, decode_vmfb) = if let Some(decode_mlir) = decode_mlir.as_deref() {
            std::thread::scope(|scope| {
                let decode =
                    scope.spawn(|| self.compile_mlir(decode_mlir, &request.bound_model.target));
                let prefill = self.compile_mlir(&mlir, &request.bound_model.target);
                let decode = decode.join().map_err(|_| {
                    DynInferError::Compilation(CompilationError {
                        message: "parallel decode compilation panicked".into(),
                        pass: Some("iree-compile.decode".into()),
                        diagnostics: vec![],
                    })
                })?;
                Ok::<_, DynInferError>((prefill?, Some(decode?)))
            })
            .map_err(annotate_error)?
        } else {
            (
                self.compile_mlir(&mlir, &request.bound_model.target)
                    .map_err(annotate_error)?,
                None,
            )
        };

        let backend = if request.options.force_subprocess {
            "iree-compile".to_string()
        } else {
            "libIREECompiler".to_string()
        };

        let mut diagnostics = vec![Diagnostic {
            code: "R_IREE_COMPILE".into(),
            severity: Severity::Remark,
            message: format!(
                "compiled {} bytes via {backend}",
                vmfb.len() + decode_vmfb.as_ref().map_or(0, Vec::len)
            ),
            architecture_op: None,
            parameter_slot: None,
            checkpoint_key: None,
            expected: None,
            actual: None,
            pass_name: Some("iree-compile".into()),
            suggestion: None,
        }];
        if request.options.dump_ir {
            diagnostics.push(Diagnostic::warning(
                "W_DUMP_IR",
                "dump_ir requested; MLIR retained in CompileOutput.mlir_text",
            ));
        }

        let layers = request
            .bound_model
            .resolved_config
            .get("num_layers")
            .and_then(|value| value.as_u64())
            .map(|value| value as u32)
            .unwrap_or(1);
        let prefill_window = lowering.prefill_window;
        let paged_kv = lowering.paged_kv;
        validate_binding_for_compile(&request.bound_model.binding)?;
        let manifest = ExecutableManifest {
            format: "dyninfer.bundle".into(),
            version: if paged_kv { 11 } else { 2 },
            architecture_id: request.bound_model.architecture.architecture_id.clone(),
            architecture_revision: request.architecture_revision.into(),
            checkpoint_schema: request.checkpoint_schema.clone(),
            target: request.bound_model.target.clone(),
            precision_policy: request.bound_model.precision_policy.clone(),
            selected_kernels: request.bound_model.selected_kernels.clone(),
            shape_profile: if paged_kv {
                let mut profile = request.shape_profile.clone();
                profile.sequence_buckets = vec![lowering::PAGED_PREFILL_CHUNK_SIZE];
                profile
            } else {
                request.shape_profile.clone()
            },
            entrypoints: if request.options.smoke_only {
                vec!["add".into()]
            } else if paged_kv {
                vec![
                    "prefill_chunk".into(),
                    "decode_chunk".into(),
                    "add".into(),
                ]
            } else {
                vec!["prefill".into(), "decode".into(), "add".into()]
            },
            kv_cache: KvCacheDescriptor {
                layer_count: layers,
                max_batch_size: request
                    .shape_profile
                    .batch_sizes
                    .first()
                    .copied()
                    .unwrap_or(1),
                max_sequence_length: lowering.max_kv,
                kv_head_count: request
                    .bound_model
                    .resolved_config
                    .get("num_kv_heads")
                    .and_then(|value| value.as_u64())
                    .map(|value| value as u32)
                    .unwrap_or(1),
                head_dimension: request
                    .bound_model
                    .resolved_config
                    .get("head_dim")
                    .and_then(|value| value.as_u64())
                    .map(|value| value as u32)
                    .unwrap_or(64),
                element_type: lowering.kv_element_type,
                layout: KvCacheLayout::LayersHeadsSeqDim,
                alignment: 64,
                storage: if paged_kv {
                    KvCacheStorage::Paged {
                        page_size: lowering.page_size,
                        chunk_size: lowering::PAGED_PREFILL_CHUNK_SIZE,
                    }
                } else {
                    KvCacheStorage::StaticGlobals
                },
            },
            parameter_scope: "weights".into(),
            parameter_components: request
                .bound_model
                .binding
                .bindings
                .iter()
                .flat_map(|binding| {
                    binding
                        .components
                        .iter()
                        .map(|component| ManifestParameterComponent {
                            scope: binding.scope.clone(),
                            key: component.external_key.clone(),
                            byte_length: component.byte_lengths.iter().copied().sum(),
                        })
                })
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            derived_parameters_required: false,
            vmfb_path: "executables/model.vmfb".into(),
            prefill_window,
            diagnostics: diagnostics.iter().map(|d| d.to_string()).collect(),
        };

        Ok(CompileOutput {
            executable: VmfbArtifact { bytes: vmfb },
            decode_executable: decode_vmfb.map(|bytes| VmfbArtifact { bytes }),
            manifest,
            diagnostics,
            mlir_text: mlir,
        })
    }
}

/// Validate the direct component contract used by selected lowerings.
fn validate_binding_for_compile(plan: &BindingPlan) -> Result<()> {
    for b in &plan.bindings {
        if b.components.is_empty() {
            return Err(DynInferError::Compilation(CompilationError {
                message: format!(
                    "binding for `{}` has no storage components",
                    b.canonical_name
                ),
                pass: Some("binding.validate".into()),
                diagnostics: vec![],
            }));
        }
    }
    Ok(())
}

/// Compile the built-in add smoke module (no architecture required).
pub fn compile_add_smoke(target: &TargetProfile) -> Result<Vec<u8>> {
    compile_mlir_prefer_inprocess(emit_add_smoke_module(), target, false, None).or_else(|_| {
        let tools = IreeTools::discover()?;
        let flags = compile_flags_for(target)?;
        tools.compile_mlir_with_flags(emit_add_smoke_module(), &flags)
    })
}
