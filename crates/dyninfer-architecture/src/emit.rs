//! Architecture-produced executable IR.
//!
//! Emit returns both the specialized MLIR and the shape parameters baked into
//! that IR. Callers (compiler / runtime) must keep them paired: the VMFB is
//! only valid for the recorded prefill window and KV capacity.

use serde::{Deserialize, Serialize};

/// Result of architecture-specific executable emission.
///
/// `prefill_window` and `max_kv` are not free-floating metadata — they are the
/// static shapes the architecture chose when generating `mlir_text`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmitOutput {
    pub mlir_text: String,
    /// Prefill sequence length specialized into the executable.
    pub prefill_window: u32,
    /// Mutable KV capacity specialized into the executable (`>= prefill_window`).
    pub max_kv: u32,
}
