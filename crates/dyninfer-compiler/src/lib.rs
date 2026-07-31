//! Safe Rust compiler wrapper — prefers in-process `libIREECompiler`, with
//! `iree-compile` subprocess fallback.

#![forbid(unsafe_code)]

mod iree_tools;
mod llama_emit;
mod mlir_emit;

pub use iree_tools::IreeTools;
pub use llama_emit::{LlamaEmitConfig, PREFILL_WINDOW, TINY_PREFILL_WINDOW};
pub use mlir_emit::{emit_add_smoke_module, emit_bridge_module, emit_model_module};

use dyninfer_architecture::ArchitecturePackage;
use dyninfer_checkpoint::CheckpointCatalog;
use dyninfer_core::{
    BindingPlan, ExecutableManifest, KvCacheDescriptor, KvCacheLayout, ScalarType, ShapeProfile,
    TargetProfile,
};
use dyninfer_error::{CompilationError, Diagnostic, DynInferError, Result, Severity};
use serde::{Deserialize, Serialize};
use tracing::{info, info_span};

pub const COMPILER_VERSION: &str = "0.1.0-iree-3.11.0";

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
    pub architecture: &'a ArchitecturePackage,
    pub checkpoint: &'a CheckpointCatalog,
    pub binding: &'a BindingPlan,
    pub target: &'a TargetProfile,
    pub shape_profile: &'a ShapeProfile,
    pub options: &'a CompileOptions,
}

#[derive(Debug, Clone)]
pub struct CompileOutput {
    pub executable: VmfbArtifact,
    pub manifest: ExecutableManifest,
    pub diagnostics: Vec<Diagnostic>,
    pub mlir_text: String,
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

    /// Compile an arbitrary MLIR string to VMFB for the given target driver.
    pub fn compile_mlir(&self, mlir: &str, driver: &str) -> Result<Vec<u8>> {
        compile_mlir_prefer_inprocess(mlir, driver, self.force_subprocess, self.tools.as_ref())
    }
}

fn compile_mlir_prefer_inprocess(
    mlir: &str,
    driver: &str,
    force_subprocess: bool,
    tools: Option<&IreeTools>,
) -> Result<Vec<u8>> {
    if !force_subprocess {
        let flags = iree_compiler_sys::flags_for_driver(driver);
        match iree_compiler_sys::compile_mlir_to_vmfb(mlir, &flags) {
            Ok(bytes) => {
                info!(
                    bytes = bytes.len(),
                    rev = iree_compiler_sys::revision().unwrap_or_default(),
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
    tools.compile_mlir(mlir, driver)
}

impl ModelCompiler for LocalCompiler {
    fn compile(&self, request: &CompileRequest<'_>) -> Result<CompileOutput> {
        let _span = info_span!(
            "compile.specialize",
            architecture = %request.architecture.id,
            target = %request.target.driver
        )
        .entered();

        let mlir = if request.options.smoke_only {
            emit_add_smoke_module().to_string()
        } else {
            emit_model_module(request.architecture, request.checkpoint)
        };

        let _iree = info_span!("compile.iree").entered();
        let vmfb = self
            .compile_mlir(&mlir, &request.target.driver)
            .map_err(|err| match err {
                DynInferError::Compilation(mut c) => {
                    c.diagnostics.push(Diagnostic::error(
                        "E_IREE_COMPILE",
                        c.message.clone(),
                    ));
                    DynInferError::Compilation(c)
                }
                other => other,
            })?;

        let backend = if request.options.force_subprocess {
            "iree-compile".to_string()
        } else {
            "libIREECompiler".to_string()
        };

        let mut diagnostics = vec![Diagnostic {
            code: "R_IREE_COMPILE".into(),
            severity: Severity::Remark,
            message: format!("compiled {} bytes via {backend}", vmfb.len()),
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
            .architecture
            .resolved_config
            .num_layers()
            .unwrap_or(1);
        let emit_cfg = LlamaEmitConfig::from_package(request.architecture, request.checkpoint);
        let prefill_window = if emit_cfg.supports_dense_emit() {
            emit_cfg.seq
        } else {
            TINY_PREFILL_WINDOW
        };
        let manifest = ExecutableManifest {
            format: "dyninfer.bundle".into(),
            version: 1,
            architecture_id: request.architecture.id.clone(),
            architecture_revision: request.architecture.revision.clone(),
            checkpoint_schema: request.checkpoint.schema_fingerprint.clone(),
            target: request.target.clone(),
            shape_profile: request.shape_profile.clone(),
            entrypoints: if request.options.smoke_only {
                vec!["add".into()]
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
                max_sequence_length: request.shape_profile.max_sequence_length,
                kv_head_count: request
                    .architecture
                    .resolved_config
                    .get_u32("num_kv_heads")
                    .unwrap_or(1),
                head_dimension: request
                    .architecture
                    .resolved_config
                    .get_u32("head_dim")
                    .unwrap_or(64),
                element_type: ScalarType::F32,
                layout: KvCacheLayout::LayersHeadsSeqDim,
                alignment: 64,
            },
            parameter_scope: "weights".into(),
            vmfb_path: "executables/model.vmfb".into(),
            prefill_window,
            diagnostics: diagnostics.iter().map(|d| d.to_string()).collect(),
        };

        Ok(CompileOutput {
            executable: VmfbArtifact { bytes: vmfb },
            manifest,
            diagnostics,
            mlir_text: mlir,
        })
    }
}

/// Compile the built-in add smoke module (no architecture required).
pub fn compile_add_smoke(driver: &str) -> Result<Vec<u8>> {
    compile_mlir_prefer_inprocess(emit_add_smoke_module(), driver, false, None).or_else(|_| {
        let tools = IreeTools::discover()?;
        tools.compile_mlir(emit_add_smoke_module(), driver)
    })
}
