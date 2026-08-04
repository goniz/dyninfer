//! Checkpoint catalog and raw index types.

use dyninfer_core::{
    ByteRange, CanonicalParameterName, ContainerFormatId, ConventionId, Endianness,
    LogicalTensorType, MetadataMap, ParameterRole, PhysicalEncoding, SchemaFingerprint, SourceFile,
    StorageComponent, StorageElementType,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Identity of a probed container.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerIdentity {
    pub format_id: ContainerFormatId,
    pub version: Option<u32>,
    pub magic: Option<String>,
}

/// Score returned by container probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProbeScore {
    pub score: u32,
}

impl ProbeScore {
    pub const NONE: Self = Self { score: 0 };
    pub const WEAK: Self = Self { score: 10 };
    pub const STRONG: Self = Self { score: 100 };

    pub fn is_match(self) -> bool {
        self.score > 0
    }
}

/// Score returned by convention matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MatchScore {
    pub score: u32,
}

impl MatchScore {
    pub const NONE: Self = Self { score: 0 };
    pub fn is_match(self) -> bool {
        self.score > 0
    }
}

/// One raw tensor as stored in the container.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawTensorEntry {
    pub key: String,
    #[serde(default)]
    pub source_file_index: u32,
    pub shape: Vec<u64>,
    pub storage_type: StorageElementType,
    pub byte_ranges: Vec<ByteRange>,
    pub alignment: u64,
    pub endianness: Endianness,
    #[serde(default)]
    pub metadata: MetadataMap,
}

/// Raw container index before convention decoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawCheckpointIndex {
    pub container: ContainerIdentity,
    pub source_files: Vec<SourceFile>,
    pub metadata: MetadataMap,
    pub entries: Vec<RawTensorEntry>,
    pub data_offset: u64,
}

/// Logical parameter after convention decoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalParameter {
    pub canonical_name: CanonicalParameterName,
    pub role: ParameterRole,
    pub logical_type: LogicalTensorType,
    pub encoding: PhysicalEncoding,
    pub components: Vec<StorageComponent>,
    #[serde(default)]
    pub aliases: Vec<String>,
}

/// Decoded parameter catalog for a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterCatalog {
    pub convention_id: ConventionId,
    pub parameters: Vec<LogicalParameter>,
    pub metadata: MetadataMap,
}

/// Complete inspection result used by binder and CLI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointCatalog {
    pub container: ContainerIdentity,
    pub convention_id: ConventionId,
    pub source_files: Vec<SourceFile>,
    pub metadata: MetadataMap,
    pub raw_entries: Vec<RawTensorEntry>,
    pub parameters: Vec<LogicalParameter>,
    pub schema_fingerprint: SchemaFingerprint,
}

/// How the runtime should open checkpoint bytes for IREE parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeProviderPlan {
    pub kind: String,
    pub scope: String,
    pub file_paths: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parameters: Vec<ProviderParameterDescriptor>,
    pub notes: Vec<String>,
}

/// One stable IREE parameter key mapped directly onto an original checkpoint
/// file range. `aliases` are temporary compatibility keys for legacy emitters;
/// they reference the same range and never create derived bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderParameterDescriptor {
    pub external_key: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    pub source_file_index: u32,
    pub offset: u64,
    pub length: u64,
}

/// Context passed to convention decoders.
#[derive(Debug, Clone, Default)]
pub struct DecodeContext {
    pub architecture_hint: Option<String>,
    pub overrides: MetadataMap,
}
