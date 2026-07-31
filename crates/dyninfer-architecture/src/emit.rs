//! Architecture-produced executable IR.

use serde::{Deserialize, Serialize};

/// Result of architecture-specific executable emission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmitOutput {
    pub mlir_text: String,
    pub prefill_window: u32,
    /// Mutable KV capacity compiled into the executable (`>= prefill_window`).
    #[serde(default)]
    pub max_kv: u32,
}
