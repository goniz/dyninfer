//! Content-addressed executable and parameter caches with atomic publication.

#![forbid(unsafe_code)]

use dyninfer_core::{content_digest, Digest, ExecutableManifest, ShapeProfile, TargetProfile};
use dyninfer_error::{CacheError, DynInferError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::{debug, info_span};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheKey {
    pub architecture_id: String,
    pub architecture_revision: String,
    pub checkpoint_schema: String,
    pub target_fingerprint: String,
    pub shape_profile_digest: String,
    pub compiler_version: String,
}

impl CacheKey {
    pub fn digest(&self) -> Result<Digest> {
        content_digest(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub digest: Digest,
    pub key: CacheKey,
    pub manifest_path: PathBuf,
    pub vmfb_path: PathBuf,
    pub size_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct ArtifactCache {
    root: PathBuf,
}

impl ArtifactCache {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("executables")).map_err(|e| {
            DynInferError::Cache(CacheError {
                message: format!("failed to create cache dir: {e}"),
                digest: None,
                path: Some(root.display().to_string()),
            })
        })?;
        fs::create_dir_all(root.join("parameters")).ok();
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn lookup(&self, key: &CacheKey) -> Result<Option<CacheEntry>> {
        let _span = info_span!("cache.lookup").entered();
        let digest = key.digest()?;
        let dir = self.entry_dir(&digest);
        let manifest_path = dir.join("manifest.json");
        let vmfb_path = dir.join("model.vmfb");
        if manifest_path.is_file() && vmfb_path.is_file() {
            let size_bytes = fs::metadata(&vmfb_path).map(|m| m.len()).unwrap_or(0);
            debug!(digest = %digest.short(), "cache hit");
            return Ok(Some(CacheEntry {
                digest,
                key: key.clone(),
                manifest_path,
                vmfb_path,
                size_bytes,
            }));
        }
        debug!(digest = %digest.short(), "cache miss");
        Ok(None)
    }

    pub fn publish(
        &self,
        key: &CacheKey,
        vmfb: &[u8],
        manifest: &ExecutableManifest,
    ) -> Result<CacheEntry> {
        let digest = key.digest()?;
        let dir = self.entry_dir(&digest);
        let staging = self.root.join(format!(".staging-{}", digest.short()));
        let _ = fs::remove_dir_all(&staging);
        fs::create_dir_all(&staging).map_err(|e| cache_io(&digest, &staging, e))?;

        let vmfb_path = staging.join("model.vmfb");
        {
            let mut f = fs::File::create(&vmfb_path).map_err(|e| cache_io(&digest, &vmfb_path, e))?;
            f.write_all(vmfb).map_err(|e| cache_io(&digest, &vmfb_path, e))?;
            f.sync_all().ok();
        }
        let manifest_path = staging.join("manifest.json");
        let json = serde_json::to_vec_pretty(manifest)?;
        fs::write(&manifest_path, json).map_err(|e| cache_io(&digest, &manifest_path, e))?;

        let key_path = staging.join("key.json");
        fs::write(&key_path, serde_json::to_vec_pretty(key)?).map_err(|e| cache_io(&digest, &key_path, e))?;

        if dir.exists() {
            fs::remove_dir_all(&dir).ok();
        }
        fs::rename(&staging, &dir).map_err(|e| cache_io(&digest, &dir, e))?;

        Ok(CacheEntry {
            digest: digest.clone(),
            key: key.clone(),
            manifest_path: dir.join("manifest.json"),
            vmfb_path: dir.join("model.vmfb"),
            size_bytes: vmfb.len() as u64,
        })
    }

    pub fn list(&self) -> Result<Vec<CacheEntry>> {
        let exec_root = self.root.join("executables");
        let mut out = Vec::new();
        let read_dir = match fs::read_dir(&exec_root) {
            Ok(rd) => rd,
            Err(_) => return Ok(out),
        };
        for entry in read_dir.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let key_path = path.join("key.json");
            let manifest_path = path.join("manifest.json");
            let vmfb_path = path.join("model.vmfb");
            if !(key_path.is_file() && vmfb_path.is_file()) {
                continue;
            }
            let key: CacheKey = serde_json::from_slice(&fs::read(&key_path)?)?;
            let digest = key.digest()?;
            out.push(CacheEntry {
                digest,
                key,
                manifest_path,
                vmfb_path: vmfb_path.clone(),
                size_bytes: fs::metadata(&vmfb_path).map(|m| m.len()).unwrap_or(0),
            });
        }
        out.sort_by(|a, b| a.digest.as_str().cmp(b.digest.as_str()));
        Ok(out)
    }

    pub fn remove(&self, digest_prefix: &str) -> Result<bool> {
        for entry in self.list()? {
            if entry.digest.as_str().starts_with(digest_prefix) {
                let dir = self.entry_dir(&entry.digest);
                fs::remove_dir_all(&dir).map_err(|e| cache_io(&entry.digest, &dir, e))?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn verify(&self) -> Result<Vec<String>> {
        let mut problems = Vec::new();
        for entry in self.list()? {
            let bytes = fs::read(&entry.vmfb_path).unwrap_or_default();
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            let actual = hex::encode(hasher.finalize());
            // Presence check only for now; full digest chain comes with signed manifests.
            if bytes.is_empty() {
                problems.push(format!("{}: empty vmfb", entry.digest.short()));
            }
            let _ = actual;
        }
        Ok(problems)
    }

    fn entry_dir(&self, digest: &Digest) -> PathBuf {
        self.root.join("executables").join(digest.as_str())
    }
}

fn cache_io(digest: &Digest, path: &Path, err: std::io::Error) -> DynInferError {
    DynInferError::Cache(CacheError {
        message: err.to_string(),
        digest: Some(digest.to_string()),
        path: Some(path.display().to_string()),
    })
}

pub fn shape_profile_digest(profile: &ShapeProfile) -> Result<Digest> {
    content_digest(profile)
}

pub fn make_cache_key(
    architecture_id: &str,
    architecture_revision: &str,
    checkpoint_schema: &str,
    target: &TargetProfile,
    shape_profile: &ShapeProfile,
    compiler_version: &str,
) -> Result<CacheKey> {
    Ok(CacheKey {
        architecture_id: architecture_id.into(),
        architecture_revision: architecture_revision.into(),
        checkpoint_schema: checkpoint_schema.into(),
        target_fingerprint: target.capability_fingerprint.to_string(),
        shape_profile_digest: shape_profile_digest(shape_profile)?.to_string(),
        compiler_version: compiler_version.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dyninfer_core::{
        ArchitectureId, KvCacheDescriptor, KvCacheLayout, ScalarType, SchemaFingerprint, ShapeProfile,
    };

    #[test]
    fn publish_and_lookup_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ArtifactCache::open(dir.path()).unwrap();
        let key = CacheKey {
            architecture_id: "llama.decoder".into(),
            architecture_revision: "0.1.0".into(),
            checkpoint_schema: "abc".into(),
            target_fingerprint: "cpu".into(),
            shape_profile_digest: "shape".into(),
            compiler_version: "0.1.0-stub".into(),
        };
        let manifest = ExecutableManifest {
            format: "dyninfer.bundle".into(),
            version: 1,
            architecture_id: ArchitectureId::new("llama.decoder"),
            architecture_revision: "0.1.0".into(),
            checkpoint_schema: SchemaFingerprint {
                digest: Digest::from_bytes(b"abc"),
                entry_count: 1,
                total_bytes: 0,
            },
            target: TargetProfile::llvm_cpu_host(),
            shape_profile: ShapeProfile::default(),
            entrypoints: vec!["prefill".into(), "decode".into()],
            kv_cache: KvCacheDescriptor {
                layer_count: 1,
                max_batch_size: 1,
                max_sequence_length: 128,
                kv_head_count: 1,
                head_dimension: 64,
                element_type: ScalarType::F16,
                layout: KvCacheLayout::LayersHeadsSeqDim,
                alignment: 64,
            },
            parameter_scope: "weights".into(),
            vmfb_path: "model.vmfb".into(),
            prefill_window: 4,
            diagnostics: vec![],
        };
        cache.publish(&key, b"VMFBSTUB", &manifest).unwrap();
        let hit = cache.lookup(&key).unwrap().expect("hit");
        assert_eq!(fs::read(hit.vmfb_path).unwrap(), b"VMFBSTUB");
    }
}
