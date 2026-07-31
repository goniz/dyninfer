use crate::config::{ConfigSchema, ResolvedModelConfig};
use crate::emit::EmitOutput;
use crate::package::ArchitecturePackage;
use dyninfer_checkpoint::{CheckpointCatalog, ParameterCatalog};
use dyninfer_core::{ArchitectureId, ParameterSlot};
use dyninfer_error::{DynInferError, Result};
use dyninfer_mlir::{ModuleBuilder, VerifiedModule};
use serde::{Deserialize, Serialize};

/// In-memory MLIR module produced by an architecture builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelModule {
    pub architecture_id: ArchitectureId,
    pub mlir_text: String,
    pub parameter_slots: Vec<ParameterSlot>,
}

/// Narrow builder API used by architecture definitions.
///
/// Backed by [`dyninfer_mlir::ModuleBuilder`] (MLIR C API / melior-style).
#[derive(Debug, Default)]
pub struct ModelBuilder {
    architecture_id: Option<ArchitectureId>,
    slots: Vec<ParameterSlot>,
    /// Accumulated MLIR source fragments (joined before verify).
    mlir_parts: Vec<String>,
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_architecture_id(&mut self, id: ArchitectureId) {
        self.architecture_id = Some(id);
    }

    pub fn declare_parameter(&mut self, slot: ParameterSlot) -> Result<()> {
        self.slots.push(slot);
        Ok(())
    }

    /// Append a chunk of MLIR text (ops, globals, functions).
    pub fn append_mlir(&mut self, text: impl Into<String>) {
        self.mlir_parts.push(text.into());
    }

    /// Note a high-level op name for debugging (does not emit IR).
    pub fn note_op(&mut self, op: impl Into<String>) {
        self.mlir_parts
            .push(format!("// op {}\n", op.into()));
    }

    pub fn finish(&mut self) -> Result<ModelModule> {
        let architecture_id = self.architecture_id.clone().ok_or_else(|| {
            DynInferError::internal("ModelBuilder missing architecture_id")
        })?;
        let slots = std::mem::take(&mut self.slots);
        let parts = std::mem::take(&mut self.mlir_parts);

        let mut source = format!(
            "// dyninfer architecture={architecture_id}\n"
        );
        for slot in &slots {
            source.push_str(&format!(
                "// parameter {} role={}\n",
                slot.canonical_name,
                slot.role.as_str()
            ));
        }
        for part in parts {
            source.push_str(&part);
            if !part.ends_with('\n') {
                source.push('\n');
            }
        }

        // Empty / comment-only builders still produce a trivial verified module.
        // Keep architecture_id on the Rust ModelModule only — do not emit a
        // custom `dyninfer.*` dialect attribute (not registered with IREE).
        let body = source.lines().any(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with("//")
        });
        let to_verify = if body {
            if source.contains("module ") {
                source
            } else {
                format!("module {{\n{source}}}\n")
            }
        } else {
            "module {\n}\n".to_string()
        };

        let verified = verify_mlir(&to_verify)?;
        Ok(ModelModule {
            architecture_id,
            mlir_text: verified.mlir_text,
            parameter_slots: slots,
        })
    }
}

/// Parse + verify MLIR text through the melior-style builder.
pub fn verify_mlir(source: &str) -> Result<VerifiedModule> {
    let mut builder = ModuleBuilder::new()?;
    builder.parse_source(source)?;
    builder.finish()
}

/// Architecture plugin: slots, naming, and executable emission.
pub trait ArchitectureDefinition: Send + Sync {
    fn id(&self) -> ArchitectureId;
    fn revision(&self) -> &str;
    fn config_schema(&self) -> &ConfigSchema;

    /// HF `model_type` / architecture class stems this definition accepts.
    fn model_types(&self) -> &[&str];

    fn build(
        &self,
        config: &ResolvedModelConfig,
        builder: &mut ModelBuilder,
    ) -> Result<ModelModule>;

    /// Map a checkpoint tensor key to a canonical parameter name.
    ///
    /// Return `None` to skip the tensor (e.g. cached RoPE freqs).
    /// Default keeps the key unchanged.
    fn canonicalize_param(&self, key: &str) -> Option<String> {
        Some(key.to_string())
    }

    /// Post-process the parameter catalog (tied embeddings, drop unused keys, …).
    fn sanitize_catalog(&self, _catalog: &mut ParameterCatalog) {}

    /// Emit the IREE-facing MLIR executable for this architecture.
    fn emit_executable(
        &self,
        package: &ArchitecturePackage,
        catalog: &CheckpointCatalog,
    ) -> Result<EmitOutput>;
}
