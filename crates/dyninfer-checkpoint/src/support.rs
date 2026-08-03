//! Builtin statically-selected checkpoint support registry.

use crate::catalog::{CheckpointCatalog, DecodeContext, ParameterCatalog, RawCheckpointIndex};
use crate::fingerprint::schema_fingerprint_from_parameters;
use crate::limits::InspectionLimits;
use crate::source::{FileSource, RandomAccessSource};
use crate::traits::{
    CheckpointContainerReader, CheckpointConventionDecoder, ParameterMaterializer,
};
use dyninfer_error::{CheckpointValidationError, DynInferError, Result, UnsupportedContainerError};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info_span};

/// Fixed set of statically linked checkpoint implementations.
#[derive(Default)]
pub struct BuiltinCheckpointSupport {
    container_readers: Vec<Arc<dyn CheckpointContainerReader>>,
    convention_decoders: Vec<Arc<dyn CheckpointConventionDecoder>>,
    materializers: Vec<Arc<dyn ParameterMaterializer>>,
}

impl BuiltinCheckpointSupport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_container(&mut self, reader: impl CheckpointContainerReader + 'static) {
        self.container_readers.push(Arc::new(reader));
    }

    pub fn register_convention(&mut self, decoder: impl CheckpointConventionDecoder + 'static) {
        self.convention_decoders.push(Arc::new(decoder));
    }

    pub fn register_materializer(&mut self, materializer: impl ParameterMaterializer + 'static) {
        self.materializers.push(Arc::new(materializer));
    }

    pub fn containers(&self) -> &[Arc<dyn CheckpointContainerReader>] {
        &self.container_readers
    }

    pub fn conventions(&self) -> &[Arc<dyn CheckpointConventionDecoder>] {
        &self.convention_decoders
    }

    /// Probe and index a checkpoint path, then decode the best matching convention.
    pub fn inspect_path(
        &self,
        path: impl AsRef<Path>,
        limits: &InspectionLimits,
        context: &DecodeContext,
    ) -> Result<CheckpointCatalog> {
        let path = path.as_ref();
        let _span = info_span!("checkpoint.probe", path = %path.display()).entered();
        let source = FileSource::open(path)?.into_arc();
        self.inspect_source(source, limits, context)
    }

    pub fn inspect_source(
        &self,
        source: Arc<dyn RandomAccessSource>,
        limits: &InspectionLimits,
        context: &DecodeContext,
    ) -> Result<CheckpointCatalog> {
        let (reader, index) = self.index_source(source, limits)?;
        let catalog = self.decode_index(&index, context)?;
        let _ = reader;
        Ok(catalog)
    }

    pub fn index_source(
        &self,
        source: Arc<dyn RandomAccessSource>,
        limits: &InspectionLimits,
    ) -> Result<(Arc<dyn CheckpointContainerReader>, RawCheckpointIndex)> {
        let mut probed: Vec<(u32, Arc<dyn CheckpointContainerReader>)> = Vec::new();
        let mut probed_formats = Vec::new();

        for reader in &self.container_readers {
            probed_formats.push(reader.format_id().to_string());
            let score = reader.probe(source.as_ref())?;
            debug!(
                format = %reader.format_id(),
                score = score.score,
                "container probe"
            );
            if score.is_match() {
                probed.push((score.score, Arc::clone(reader)));
            }
        }

        probed.sort_by(|a, b| b.0.cmp(&a.0));
        let Some((_, reader)) = probed.into_iter().next() else {
            return Err(DynInferError::UnsupportedContainer(
                UnsupportedContainerError {
                    message: "no registered container matched the checkpoint".into(),
                    path: source.path().map(|p| p.display().to_string()),
                    probed_formats,
                },
            ));
        };

        let _span = info_span!("checkpoint.index", format = %reader.format_id()).entered();
        let index = reader.index(Arc::clone(&source), limits)?;
        Ok((reader, index))
    }

    pub fn decode_index(
        &self,
        index: &RawCheckpointIndex,
        context: &DecodeContext,
    ) -> Result<CheckpointCatalog> {
        let _span = info_span!("checkpoint.decode_convention").entered();
        let mut scored: Vec<(u32, Arc<dyn CheckpointConventionDecoder>)> = Vec::new();
        for decoder in &self.convention_decoders {
            let score = decoder.match_score(index, context)?;
            if score.is_match() {
                scored.push((score.score, Arc::clone(decoder)));
            }
        }
        scored.sort_by(|a, b| b.0.cmp(&a.0));

        let Some((_, decoder)) = scored.into_iter().next() else {
            return Err(DynInferError::InvalidCheckpoint(
                CheckpointValidationError {
                    message: "no convention decoder matched the checkpoint index".into(),
                    key: None,
                    detail: Some(format!("container={}", index.container.format_id)),
                },
            ));
        };

        let ParameterCatalog {
            convention_id,
            parameters,
            metadata,
        } = decoder.decode(index, context)?;

        let mut merged_metadata = index.metadata.clone();
        for (k, v) in metadata {
            merged_metadata.insert(k, v);
        }

        let schema_fingerprint = schema_fingerprint_from_parameters(&parameters)?;

        Ok(CheckpointCatalog {
            container: index.container.clone(),
            convention_id,
            source_files: index.source_files.clone(),
            metadata: merged_metadata,
            raw_entries: index.entries.clone(),
            parameters,
            schema_fingerprint,
        })
    }
}

/// Convenience: empty support with no registrations (tests fill it in).
pub fn empty_support() -> BuiltinCheckpointSupport {
    BuiltinCheckpointSupport::new()
}
