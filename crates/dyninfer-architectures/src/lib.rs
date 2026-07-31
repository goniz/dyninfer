//! Concrete model architectures (mlx-lm-style: one file per arch under `models/`).
//!
//! Shared MLIR helpers live in [`ops`]. Add a new architecture by:
//! 1. Creating `models/<name>.rs` implementing [`ArchitectureDefinition`]
//! 2. Registering it in [`register_all`]
//! 3. Adding aliases in [`remapping`]

#![forbid(unsafe_code)]

mod models;
mod naming;
mod ops;
mod remapping;
mod slots;

pub use models::{LlamaArchitecture, Qwen3Architecture};
pub use ops::{
    DenseDecoderConfig, COMPUTE_DTYPE, LARGE_PREFILL_WINDOW, PREFILL_WINDOW, TINY_PREFILL_WINDOW,
};
pub use remapping::{remap_model_type, MODEL_REMAPPING};

use dyninfer_architecture::ArchitectureRegistry;
use dyninfer_core::{ArchitectureId, MetadataMap};
use dyninfer_error::{ArchitectureMismatchError, DynInferError, Result};

/// Register every built-in architecture.
pub fn register_all(registry: &mut ArchitectureRegistry) {
    registry.register(LlamaArchitecture);
    registry.register(Qwen3Architecture);
}

/// Resolve architecture id from an optional CLI override and/or checkpoint metadata.
pub fn resolve_architecture(
    registry: &ArchitectureRegistry,
    override_id: Option<&str>,
    metadata: &MetadataMap,
) -> Result<ArchitectureId> {
    if let Some(id) = override_id {
        let id = id.trim();
        if !id.is_empty() && !id.eq_ignore_ascii_case("auto") {
            let aid = ArchitectureId::new(id);
            if registry.get(&aid).is_some() {
                return Ok(aid);
            }
            return Err(DynInferError::ArchitectureMismatch(
                ArchitectureMismatchError {
                    message: format!("unknown architecture `{id}`"),
                    architecture_id: Some(id.to_string()),
                    detail: Some(format!(
                        "known: {}",
                        registry.ids().collect::<Vec<_>>().join(", ")
                    )),
                },
            ));
        }
    }

    // Prefer remapping table (handles aliases), then registry model_types.
    let candidates = model_type_candidates(metadata);
    for c in &candidates {
        if let Some(id) = remap_model_type(c) {
            let aid = ArchitectureId::new(id);
            if registry.get(&aid).is_some() {
                return Ok(aid);
            }
        }
    }
    registry.resolve_from_metadata(metadata)
}

fn model_type_candidates(meta: &MetadataMap) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(v) = meta.get("model_type").and_then(|v| v.as_str()) {
        out.push(v.to_string());
    }
    if let Some(v) = meta.get("hf_architecture").and_then(|v| v.as_str()) {
        out.push(v.to_string());
    }
    out
}
