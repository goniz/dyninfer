use crate::builder::{ArchitectureDefinition, ModelBuilder};
use crate::config::ResolvedModelConfig;
use crate::package::ArchitecturePackage;
use dyninfer_checkpoint::infer_role;
use dyninfer_checkpoint::{
    CheckpointCatalog, ParameterCatalog, schema_fingerprint_from_parameters,
};
use dyninfer_core::{ArchitectureId, CanonicalParameterName, MetadataMap};
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

    fn require(&self, id: &ArchitectureId) -> Result<Arc<dyn ArchitectureDefinition>> {
        self.get(id).ok_or_else(|| {
            DynInferError::ArchitectureMismatch(ArchitectureMismatchError {
                message: format!("unknown architecture `{id}`"),
                architecture_id: Some(id.to_string()),
                detail: None,
            })
        })
    }

    /// Find an architecture that lists `model_type` in [`ArchitectureDefinition::model_types`].
    pub fn find_by_model_type(&self, model_type: &str) -> Option<ArchitectureId> {
        let needle = model_type.trim();
        if needle.is_empty() {
            return None;
        }
        self.defs.values().find_map(|def| {
            def.model_types()
                .iter()
                .any(|t| t.eq_ignore_ascii_case(needle))
                .then(|| def.id())
        })
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
    pub fn apply_naming(&self, id: &ArchitectureId, catalog: &mut CheckpointCatalog) -> Result<()> {
        let def = self.require(id)?;

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
            let canonical =
                if canonical == source_key && param.canonical_name.as_str() != source_key {
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
        let def = self.require(id)?;
        let defaults = MetadataMap::new();
        let config = def
            .config_schema()
            .resolve(overrides, checkpoint_meta, &defaults)?;
        let mut builder = ModelBuilder::new()?;
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
        let def = self.require(id)?;
        let mut builder = ModelBuilder::new()?;
        builder.set_architecture_id(def.id());
        let module = def.build(config, &mut builder)?;
        Ok(ArchitecturePackage::from_module(
            module,
            config.clone(),
            def.revision().to_string(),
        ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{register_all, verify_architecture_catalog_conformance};
    use dyninfer_checkpoint::{LogicalParameter, ParameterCatalog};
    use dyninfer_core::{
        ConventionId, LogicalTensorType, OperationKind, PhysicalEncoding, ScalarType, Shape,
    };

    fn fixture_for(package: &ArchitecturePackage) -> ParameterCatalog {
        ParameterCatalog {
            convention_id: ConventionId::new("test.canonical"),
            parameters: package
                .parameter_slots()
                .iter()
                .map(|slot| LogicalParameter {
                    canonical_name: slot.canonical_name.clone(),
                    role: slot.role.clone(),
                    logical_type: LogicalTensorType {
                        shape: Shape::new(vec![1; slot.expected_type.rank.unwrap_or(1)]),
                        element_type: ScalarType::F32,
                    },
                    encoding: PhysicalEncoding::plain(ScalarType::F32),
                    components: vec![],
                    aliases: vec![slot.canonical_name.to_string()],
                })
                .collect(),
            metadata: MetadataMap::new(),
        }
    }

    #[test]
    fn built_in_architectures_pass_typed_graph_conformance() {
        let mut registry = ArchitectureRegistry::new();
        register_all(&mut registry);
        for id in ["llama.decoder", "qwen3.decoder"] {
            let package = registry
                .build_package(
                    &ArchitectureId::new(id),
                    &MetadataMap::new(),
                    &MetadataMap::new(),
                )
                .unwrap();
            verify_architecture_catalog_conformance(&package.graph, &fixture_for(&package))
                .unwrap();
            assert!(
                package
                    .graph
                    .operations
                    .iter()
                    .any(|operation| matches!(operation.kind, OperationKind::Attention { .. }))
            );
            assert!(
                package
                    .graph
                    .exports
                    .iter()
                    .any(|export| export.name == "prefill")
            );
            assert!(
                package
                    .graph
                    .exports
                    .iter()
                    .any(|export| export.name == "decode")
            );
        }
    }

    #[test]
    fn qwen_graph_contains_per_head_qk_normalization() {
        let mut registry = ArchitectureRegistry::new();
        register_all(&mut registry);
        let package = registry
            .build_package(
                &ArchitectureId::new("qwen3.decoder"),
                &MetadataMap::new(),
                &MetadataMap::new(),
            )
            .unwrap();
        let count = package
            .graph
            .operations
            .iter()
            .filter(|operation| matches!(operation.kind, OperationKind::PerHeadRmsNorm { .. }))
            .count();
        assert_eq!(count, 2 * 28);
    }
}
