//! Emit IREE-legal MLIR for model executables.

use dyninfer_architecture::ArchitecturePackage;
use dyninfer_checkpoint::CheckpointCatalog;

use crate::llama_emit::{emit_dense_llama, LlamaEmitConfig};

/// Emit the best available module for this architecture/checkpoint.
pub fn emit_model_module(package: &ArchitecturePackage, catalog: &CheckpointCatalog) -> String {
    let cfg = LlamaEmitConfig::from_package(package, catalog);
    if cfg.supports_dense_emit() {
        emit_dense_llama(package, catalog)
    } else {
        emit_bridge_module(package)
    }
}

/// Constant-zero logits bridge (non-tiny shapes / smoke scaffolding).
pub fn emit_bridge_module(package: &ArchitecturePackage) -> String {
    let vocab = package
        .resolved_config
        .get_u32("vocab_size")
        .unwrap_or(32)
        .max(1);
    let window = 4u32;
    let zeros = std::iter::repeat_n("0.0", vocab as usize)
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        r#"// dyninfer bridge module for {arch}
func.func @prefill(%tokens: tensor<{window}xi64>) -> tensor<{vocab}xf32> {{
  %c = arith.constant dense<[{zeros}]> : tensor<{vocab}xf32>
  return %c : tensor<{vocab}xf32>
}}

func.func @decode(%token: tensor<i64>) -> tensor<{vocab}xf32> {{
  %c = arith.constant dense<[{zeros}]> : tensor<{vocab}xf32>
  return %c : tensor<{vocab}xf32>
}}

func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {{
  %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}}
"#,
        arch = package.id,
        vocab = vocab,
        window = window,
        zeros = zeros,
    )
}

/// Minimal add-only smoke module.
pub fn emit_add_smoke_module() -> &'static str {
    r#"func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
"#
}
