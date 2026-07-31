//! Generic IREE MLIR scaffolding (architecture-agnostic).

use dyninfer_architecture::ArchitecturePackage;

/// Constant-zero logits bridge when an architecture cannot emit a real module.
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
func.func @prefill(%tokens: tensor<{window}xi64>, %last: tensor<i64>) -> tensor<{vocab}xf32> {{
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
