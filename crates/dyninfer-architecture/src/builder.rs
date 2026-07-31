use crate::config::{ConfigSchema, ResolvedModelConfig};
use crate::emit::EmitOutput;
use crate::package::ArchitecturePackage;
use dyninfer_checkpoint::{CheckpointCatalog, ParameterCatalog};
use dyninfer_core::{ArchitectureId, ParameterSlot};
use dyninfer_error::{DynInferError, Result};
use serde::{Deserialize, Serialize};

/// Placeholder for an in-memory MLIR module until melior/FFI is wired.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelModule {
    pub architecture_id: ArchitectureId,
    pub mlir_text: String,
    pub parameter_slots: Vec<ParameterSlot>,
}

/// Narrow builder API used by architecture definitions.
#[derive(Debug, Default)]
pub struct ModelBuilder {
    architecture_id: Option<ArchitectureId>,
    slots: Vec<ParameterSlot>,
    ops: Vec<String>,
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

    pub fn note_op(&mut self, op: impl Into<String>) {
        self.ops.push(op.into());
    }

    pub fn finish(&mut self) -> Result<ModelModule> {
        let architecture_id = self.architecture_id.clone().ok_or_else(|| {
            DynInferError::internal("ModelBuilder missing architecture_id")
        })?;
        let slots = std::mem::take(&mut self.slots);
        let ops = std::mem::take(&mut self.ops);
        let mut mlir = format!(
            "module attributes {{dyninfer.architecture_id = \"{architecture_id}\"}} {{\n"
        );
        for slot in &slots {
            mlir.push_str(&format!(
                "  // parameter {} role={}\n",
                slot.canonical_name,
                slot.role.as_str()
            ));
        }
        for op in &ops {
            mlir.push_str(&format!("  // op {op}\n"));
        }
        mlir.push_str("}\n");
        Ok(ModelModule {
            architecture_id,
            mlir_text: mlir,
            parameter_slots: slots,
        })
    }
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
