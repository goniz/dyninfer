use crate::config::{ConfigSchema, ResolvedModelConfig};
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

pub trait ArchitectureDefinition: Send + Sync {
    fn id(&self) -> ArchitectureId;
    fn revision(&self) -> &str;
    fn config_schema(&self) -> &ConfigSchema;
    fn build(
        &self,
        config: &ResolvedModelConfig,
        builder: &mut ModelBuilder,
    ) -> Result<ModelModule>;
}
