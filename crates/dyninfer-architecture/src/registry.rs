use crate::builder::{ArchitectureDefinition, ModelBuilder};
use crate::config::ResolvedModelConfig;
use crate::package::ArchitecturePackage;
use dyninfer_checkpoint::{
    schema_fingerprint_from_parameters, CheckpointCatalog, LogicalParameter, ParameterCatalog,
};
use dyninfer_core::{
    ArchitectureId, CanonicalParameterName, MetadataMap,
};
use dyninfer_checkpoint::infer_role;
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

    /// Find an architecture that lists `model_type` in [`ArchitectureDefinition::model_types`].
    pub fn find_by_model_type(&self, model_type: &str) -> Option<ArchitectureId> {
        let needle = model_type.trim();
        if needle.is_empty() {
            return None;
        }
        for def in self.defs.values() {
            if def
                .model_types()
                .iter()
                .any(|t| t.eq_ignore_ascii_case(needle))
            {
                return Some(def.id());
            }
        }
        None
    }

    /// Resolve architecture from checkpoint metadata (`model_type` / `hf_architecture`).
    pub fn resolve_from_metadata(&self, meta: &MetadataMap) -> Result<ArchitectureId> {
        let candidates = meta_model_type_candidates(meta);
        for c in &candidates {
            if let Some(id) = self.find_by_model_type(c) {
                return Ok(id);
            }
        }
        Err(DynInferError::ArchitectureMismatch(
            ArchitectureMismatchError {
                message: format!(
                    "could not resolve architecture from checkpoint metadata (tried {candidates:?})"
                ),
                architecture_id: None,
                detail: Some(
                    "pass --architecture explicitly or add a model file for this model_type".into(),
                ),
            },
        ))
    }

    /// Apply architecture naming (canonicalize + sanitize) and refresh schema fingerprint.
    pub fn apply_naming(
        &self,
        id: &ArchitectureId,
        catalog: &mut CheckpointCatalog,
    ) -> Result<()> {
        let def = self.get(id).ok_or_else(|| {
            DynInferError::ArchitectureMismatch(ArchitectureMismatchError {
                message: format!("unknown architecture `{id}`"),
                architecture_id: Some(id.to_string()),
                detail: None,
            })
        })?;

        let mut remapped = Vec::with_capacity(catalog.parameters.len());
        for param in &catalog.parameters {
            // Prefer original storage key for remapping when present.
            let source_key = param
                .aliases
                .first()
                .map(|s| s.as_str())
                .unwrap_or(param.canonical_name.as_str());
            let Some(canonical) = def.canonicalize_param(source_key) else {
                continue;
            };
            // Also try canonical_name itself when aliases empty / already remapped.
            let canonical = if canonical == source_key
                && param.canonical_name.as_str() != source_key
            {
                def.canonicalize_param(param.canonical_name.as_str())
                    .unwrap_or(canonical)
            } else {
                canonical
            };
            let mut p = param.clone();
            p.role = infer_role(&canonical);
            p.canonical_name = CanonicalParameterName::new(canonical.clone());
            if !p.aliases.iter().any(|a| a == &canonical) {
                p.aliases.push(canonical);
            }
            remapped.push(p);
        }
        catalog.parameters = remapped;

        let mut pc = ParameterCatalog {
            convention_id: catalog.convention_id.clone(),
            parameters: catalog.parameters.clone(),
            metadata: catalog.metadata.clone(),
        };
        def.sanitize_catalog(&mut pc);
        catalog.parameters = pc.parameters;
        for (k, v) in pc.metadata {
            catalog.metadata.entry(k).or_insert(v);
        }
        catalog.schema_fingerprint = schema_fingerprint_from_parameters(&catalog.parameters)?;
        Ok(())
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

    pub fn emit_executable(
        &self,
        id: &ArchitectureId,
        package: &ArchitecturePackage,
        catalog: &CheckpointCatalog,
    ) -> Result<crate::emit::EmitOutput> {
        let def = self.get(id).ok_or_else(|| {
            DynInferError::ArchitectureMismatch(ArchitectureMismatchError {
                message: format!("unknown architecture `{id}`"),
                architecture_id: Some(id.to_string()),
                detail: None,
            })
        })?;
        def.emit_executable(package, catalog)
    }
}

fn meta_model_type_candidates(meta: &MetadataMap) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(v) = meta.get("model_type").and_then(|v| v.as_str()) {
        out.push(v.to_string());
    }
    if let Some(v) = meta.get("hf_architecture").and_then(|v| v.as_str()) {
        out.push(v.to_string());
        if let Some(stem) = v.strip_suffix("ForCausalLM") {
            out.push(stem.to_ascii_lowercase());
        }
    }
    out
}

/// Deduplicate parameters by canonical name (last wins).
pub fn dedupe_parameters(parameters: Vec<LogicalParameter>) -> Vec<LogicalParameter> {
    let mut map = std::collections::BTreeMap::new();
    for p in parameters {
        map.insert(p.canonical_name.to_string(), p);
    }
    map.into_values().collect()
}
