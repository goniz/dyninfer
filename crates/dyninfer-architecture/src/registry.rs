use crate::builder::{ArchitectureDefinition, ModelBuilder};
use crate::config::ResolvedModelConfig;
use crate::package::ArchitecturePackage;
use dyninfer_core::{ArchitectureId, MetadataMap};
use dyninfer_error::{ArchitectureMismatchError, DynInferError, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info_span;

#[derive(Default)]
pub struct ArchitectureRegistry {
    defs: HashMap<String, Arc<dyn ArchitectureDefinition>>,
}

impl ArchitectureRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, def: impl ArchitectureDefinition + 'static) {
        let id = def.id().to_string();
        self.defs.insert(id, Arc::new(def));
    }

    pub fn get(&self, id: &ArchitectureId) -> Option<Arc<dyn ArchitectureDefinition>> {
        self.defs.get(id.as_str()).cloned()
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.defs.keys().map(|s| s.as_str())
    }

    pub fn build_package(
        &self,
        id: &ArchitectureId,
        overrides: &MetadataMap,
        checkpoint_meta: &MetadataMap,
    ) -> Result<ArchitecturePackage> {
        let _span = info_span!("architecture.load", architecture = %id).entered();
        let def = self.get(id).ok_or_else(|| {
            DynInferError::ArchitectureMismatch(ArchitectureMismatchError {
                message: format!("unknown architecture `{id}`"),
                architecture_id: Some(id.to_string()),
                detail: None,
            })
        })?;
        let defaults = MetadataMap::new();
        let config = def
            .config_schema()
            .resolve(overrides, checkpoint_meta, &defaults)?;
        let mut builder = ModelBuilder::new();
        builder.set_architecture_id(def.id());
        let module = def.build(&config, &mut builder)?;
        Ok(ArchitecturePackage::from_module(
            module,
            config,
            def.revision().to_string(),
        ))
    }

    pub fn build_with_config(
        &self,
        id: &ArchitectureId,
        config: &ResolvedModelConfig,
    ) -> Result<ArchitecturePackage> {
        let def = self.get(id).ok_or_else(|| {
            DynInferError::ArchitectureMismatch(ArchitectureMismatchError {
                message: format!("unknown architecture `{id}`"),
                architecture_id: Some(id.to_string()),
                detail: None,
            })
        })?;
        let mut builder = ModelBuilder::new();
        builder.set_architecture_id(def.id());
        let module = def.build(config, &mut builder)?;
        Ok(ArchitecturePackage::from_module(
            module,
            config.clone(),
            def.revision().to_string(),
        ))
    }
}
