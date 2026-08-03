use crate::builder::ModelModule;
use crate::config::ResolvedModelConfig;
use dyninfer_core::{ArchitectureGraph, ArchitectureId, ParameterSlot};
use dyninfer_error::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturePackage {
    pub id: ArchitectureId,
    pub revision: String,
    pub graph: ArchitectureGraph,
    pub resolved_config: ResolvedModelConfig,
}

impl ArchitecturePackage {
    pub fn from_module(
        module: ModelModule,
        resolved_config: ResolvedModelConfig,
        revision: String,
    ) -> Self {
        Self {
            id: module.graph.architecture_id.clone(),
            revision,
            graph: module.graph,
            resolved_config,
        }
    }

    pub fn parameter_slots(&self) -> &[ParameterSlot] {
        &self.graph.parameter_slots
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn read_json(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}
