//! Hugging Face sharded SafeTensors index reader.

use crate::SafeTensorsContainer;
use dyninfer_checkpoint::{
    CheckpointContainerReader, ContainerIdentity, FileSource, InspectionLimits, ProbeScore,
    RandomAccessSource, RawCheckpointIndex, RuntimeProviderPlan,
};
use dyninfer_core::{ContainerFormatId, MetadataMap};
use dyninfer_error::{CheckpointValidationError, DynInferError, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct ShardedSafeTensorsContainer;

#[derive(Debug, Deserialize)]
struct IndexFile {
    #[serde(default)]
    metadata: BTreeMap<String, Value>,
    weight_map: BTreeMap<String, String>,
}

impl CheckpointContainerReader for ShardedSafeTensorsContainer {
    fn format_id(&self) -> ContainerFormatId {
        ContainerFormatId::new("safetensors.sharded")
    }

    fn probe(&self, source: &dyn RandomAccessSource) -> Result<ProbeScore> {
        let is_index = source.path().is_some_and(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".safetensors.index.json"))
        });
        Ok(if is_index {
            ProbeScore::STRONG
        } else {
            ProbeScore::NONE
        })
    }

    fn index(
        &self,
        source: Arc<dyn RandomAccessSource>,
        limits: &InspectionLimits,
    ) -> Result<RawCheckpointIndex> {
        if source.len() == 0 || source.len() > limits.max_header_bytes {
            return Err(invalid(format!(
                "invalid sharded SafeTensors index length {}",
                source.len()
            )));
        }
        let index_path = source
            .path()
            .ok_or_else(|| invalid("sharded SafeTensors index requires a filesystem path"))?;
        let parent = index_path
            .parent()
            .ok_or_else(|| invalid("sharded SafeTensors index has no parent directory"))?;
        let parsed: IndexFile = serde_json::from_slice(&source.read_range(0, source.len())?)
            .map_err(|error| invalid(format!("invalid sharded SafeTensors index JSON: {error}")))?;
        if parsed.weight_map.is_empty() {
            return Err(invalid("sharded SafeTensors weight_map is empty"));
        }
        if parsed.weight_map.len() as u64 > limits.max_tensor_count {
            return Err(invalid(format!(
                "sharded tensor count {} exceeds limit {}",
                parsed.weight_map.len(),
                limits.max_tensor_count
            )));
        }

        let mut expected_by_shard: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (key, shard) in &parsed.weight_map {
            limits.validate_key(key)?;
            validate_shard_name(shard)?;
            expected_by_shard
                .entry(shard.clone())
                .or_default()
                .insert(key.clone());
        }

        let reader = SafeTensorsContainer;
        let mut entries = Vec::with_capacity(parsed.weight_map.len());
        let mut source_files = Vec::with_capacity(expected_by_shard.len());
        let mut metadata: MetadataMap = parsed.metadata;
        for (shard_name, expected) in expected_by_shard {
            let shard_path = parent.join(&shard_name);
            let shard_source = FileSource::open(&shard_path)?.into_arc();
            let shard_index = reader.index(shard_source, limits)?;
            let actual: BTreeSet<_> = shard_index
                .entries
                .iter()
                .map(|entry| entry.key.clone())
                .collect();
            if actual != expected {
                let missing: Vec<_> = expected.difference(&actual).cloned().collect();
                let unlisted: Vec<_> = actual.difference(&expected).cloned().collect();
                return Err(invalid(format!(
                    "shard `{shard_name}` disagrees with weight_map; missing={missing:?}, unlisted={unlisted:?}"
                )));
            }
            let source_file_index = u32::try_from(source_files.len())
                .map_err(|_| invalid("too many SafeTensors shards"))?;
            let Some(source_file) = shard_index.source_files.into_iter().next() else {
                return Err(invalid(format!("shard `{shard_name}` has no source file")));
            };
            source_files.push(source_file);
            for (key, value) in shard_index.metadata {
                metadata.entry(key).or_insert(value);
            }
            entries.extend(shard_index.entries.into_iter().map(|mut entry| {
                entry.source_file_index = source_file_index;
                entry
            }));
        }
        entries.sort_by(|left, right| left.key.cmp(&right.key));

        Ok(RawCheckpointIndex {
            container: ContainerIdentity {
                format_id: self.format_id(),
                version: Some(1),
                magic: Some("safetensors.index.json".into()),
            },
            source_files,
            metadata,
            entries,
            data_offset: 0,
        })
    }

    fn runtime_provider_plan(&self, index: &RawCheckpointIndex) -> Result<RuntimeProviderPlan> {
        Ok(RuntimeProviderPlan {
            kind: "file-mapped-external-parameters".into(),
            scope: "weights".into(),
            file_paths: index
                .source_files
                .iter()
                .map(|source| source.path.clone())
                .collect(),
            parameters: vec![],
            notes: vec![
                "Sharded SafeTensors components retain their source_file_index and direct byte ranges"
                    .into(),
            ],
        })
    }
}

fn validate_shard_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(invalid(format!("unsafe SafeTensors shard path `{name}`")));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> DynInferError {
    DynInferError::InvalidCheckpoint(CheckpointValidationError {
        message: message.into(),
        key: None,
        detail: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write_safetensors;
    use dyninfer_checkpoint::{BuiltinCheckpointSupport, DecodeContext};
    use std::collections::BTreeMap;

    #[test]
    fn indexes_all_shards_and_preserves_file_identity() {
        let temp = tempfile::tempdir().unwrap();
        let mut first = BTreeMap::new();
        first.insert("a.weight".into(), (vec![2], vec![1.0, 2.0]));
        let mut second = BTreeMap::new();
        second.insert("b.weight".into(), (vec![2], vec![3.0, 4.0]));
        std::fs::write(
            temp.path().join("model-00001-of-00002.safetensors"),
            write_safetensors(&first, Value::Null),
        )
        .unwrap();
        std::fs::write(
            temp.path().join("model-00002-of-00002.safetensors"),
            write_safetensors(&second, Value::Null),
        )
        .unwrap();
        let index_path = temp.path().join("model.safetensors.index.json");
        std::fs::write(
            &index_path,
            serde_json::to_vec(&serde_json::json!({
                "metadata": {"total_size": 16},
                "weight_map": {
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let mut support = BuiltinCheckpointSupport::new();
        crate::register(&mut support);
        let catalog = support
            .inspect_path(
                &index_path,
                &InspectionLimits::default(),
                &DecodeContext::default(),
            )
            .unwrap();
        assert_eq!(catalog.source_files.len(), 2);
        assert_eq!(catalog.parameters.len(), 2);
        assert_eq!(catalog.parameters[0].components[0].source_file_index, 0);
        assert_eq!(catalog.parameters[1].components[0].source_file_index, 1);
    }

    #[test]
    fn rejects_parent_traversal_in_shard_name() {
        let error = validate_shard_name("../weights.safetensors").unwrap_err();
        assert_eq!(error.code(), "E_INVALID_CHECKPOINT");
    }
}
