//! Content-addressed executable cache with atomic publication.

#![forbid(unsafe_code)]

use dyninfer_core::{
    BindingPlan, Digest, ExecutableManifest, PrecisionPolicy, ShapeProfile, TargetProfile,
    content_digest,
};
use dyninfer_error::{CacheError, DynInferError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::{debug, info_span};

/// Inputs that specialize a VMFB (spec §19.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheKey {
    pub architecture_id: String,
    pub architecture_revision: String,
    /// Digest of architecture package content that affects codegen (slots + IR).
    pub architecture_digest: String,
    pub resolved_config_digest: String,
    pub binding_plan_digest: String,
    pub checkpoint_schema: String,
    pub target_fingerprint: String,
    pub precision_policy_digest: String,
    pub shape_profile_digest: String,
    pub kernel_registry_version: String,
    pub compiler_version: String,
    pub iree_revision: String,
    pub compile_options_digest: String,
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
        fs::create_dir_all(root.join("locks")).ok();
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn lookup(&self, key: &CacheKey) -> Result<Option<CacheEntry>> {
        let _span = info_span!("cache.lookup").entered();
        let digest = key.digest()?;
        Ok(self.entry_if_complete(&digest, key))
    }

    pub fn publish(
        &self,
        key: &CacheKey,
        vmfb: &[u8],
        manifest: &ExecutableManifest,
    ) -> Result<CacheEntry> {
        let digest = key.digest()?;
        let dir = self.entry_dir(&digest);

        // Coordinate concurrent publishers for the same digest (spec §19.3).
        let _lock = self.acquire_publish_lock(&digest)?;

        if let Some(existing) = self.entry_if_complete(&digest, key) {
            return Ok(existing);
        }

        // Unique staging path so concurrent publishers cannot clobber each other.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let staging = self.root.join(format!(
            ".staging-{}-{}-{}",
            digest.as_str(),
            std::process::id(),
            nanos
        ));
        if let Err(e) = fs::create_dir_all(&staging) {
            return Err(cache_io(&digest, &staging, e));
        }

        let publish_result = (|| -> Result<CacheEntry> {
            let vmfb_path = staging.join("model.vmfb");
            {
                let mut f =
                    fs::File::create(&vmfb_path).map_err(|e| cache_io(&digest, &vmfb_path, e))?;
                f.write_all(vmfb)
                    .map_err(|e| cache_io(&digest, &vmfb_path, e))?;
                f.sync_all().ok();
            }
            let manifest_path = staging.join("manifest.json");
            let json = serde_json::to_vec_pretty(manifest)?;
            fs::write(&manifest_path, json).map_err(|e| cache_io(&digest, &manifest_path, e))?;

            let key_path = staging.join("key.json");
            fs::write(&key_path, serde_json::to_vec_pretty(key)?)
                .map_err(|e| cache_io(&digest, &key_path, e))?;

            // Never delete a complete destination (avoids reader gaps). Only
            // rename into place when the final dir is absent.
            if let Some(existing) = self.entry_if_complete(&digest, key) {
                return Ok(existing);
            }
            if dir.exists() {
                // Incomplete leftover from a crash — safe to replace under lock.
                fs::remove_dir_all(&dir).map_err(|e| cache_io(&digest, &dir, e))?;
            }
            fs::rename(&staging, &dir).map_err(|e| cache_io(&digest, &dir, e))?;

            Ok(CacheEntry {
                digest: digest.clone(),
                key: key.clone(),
                manifest_path: dir.join("manifest.json"),
                vmfb_path: dir.join("model.vmfb"),
                size_bytes: vmfb.len() as u64,
            })
        })();

        if staging.exists() {
            let _ = fs::remove_dir_all(&staging);
        }
        publish_result
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

    fn entry_if_complete(&self, digest: &Digest, key: &CacheKey) -> Option<CacheEntry> {
        let dir = self.entry_dir(digest);
        let manifest_path = dir.join("manifest.json");
        let vmfb_path = dir.join("model.vmfb");
        if manifest_path.is_file() && vmfb_path.is_file() {
            let size_bytes = fs::metadata(&vmfb_path).map(|m| m.len()).unwrap_or(0);
            debug!(digest = %digest.short(), "cache hit");
            Some(CacheEntry {
                digest: digest.clone(),
                key: key.clone(),
                manifest_path,
                vmfb_path,
                size_bytes,
            })
        } else {
            debug!(digest = %digest.short(), "cache miss");
            None
        }
    }

    fn lock_path(&self, digest: &Digest) -> PathBuf {
        self.root
            .join("locks")
            .join(format!("{}.lock", digest.as_str()))
    }

    /// Exclusive create-new lock file with stale-PID recovery.
    fn acquire_publish_lock(&self, digest: &Digest) -> Result<PublishLockGuard> {
        let path = self.lock_path(digest);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| cache_io(digest, parent, e))?;
        }
        for _ in 0..500 {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut f) => {
                    let _ = writeln!(f, "{}", std::process::id());
                    let _ = f.sync_all();
                    return Ok(PublishLockGuard { path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if self.lock_is_stale(&path) {
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => return Err(cache_io(digest, &path, e)),
            }
        }
        Err(DynInferError::Cache(CacheError {
            message: "timed out waiting for cache publish lock".into(),
            digest: Some(digest.to_string()),
            path: Some(path.display().to_string()),
        }))
    }

    fn lock_is_stale(&self, path: &Path) -> bool {
        // Only reclaim an old lock after its recorded holder has exited. A
        // valid compile may legitimately take longer than this threshold.
        const STALE_SECS: u64 = 300;
        let Ok(meta) = fs::metadata(path) else {
            return true;
        };
        let Ok(modified) = meta.modified() else {
            return true;
        };
        let Ok(age) = modified.elapsed() else {
            return true;
        };
        if age.as_secs() < STALE_SECS {
            return false;
        }

        let holder_is_alive = fs::read_to_string(path)
            .ok()
            .and_then(|contents| contents.trim().parse::<u32>().ok())
            .is_some_and(process_is_running);
        !holder_is_alive
    }
}

#[cfg(target_os = "linux")]
fn process_is_running(pid: u32) -> bool {
    pid != 0 && Path::new("/proc").join(pid.to_string()).exists()
}

#[cfg(not(target_os = "linux"))]
fn process_is_running(_pid: u32) -> bool {
    false
}

struct PublishLockGuard {
    path: PathBuf,
}

impl Drop for PublishLockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
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

/// Inputs required to build a §19.1-compliant executable cache key.
pub struct CacheKeyInputs<'a> {
    pub architecture_id: &'a str,
    pub architecture_revision: &'a str,
    pub architecture_digest: Digest,
    pub resolved_config_digest: Digest,
    pub binding: &'a BindingPlan,
    pub checkpoint_schema: &'a str,
    pub target: &'a TargetProfile,
    pub precision_policy: &'a PrecisionPolicy,
    pub shape_profile: &'a ShapeProfile,
    pub kernel_registry_version: &'a str,
    pub compiler_version: &'a str,
    pub iree_revision: &'a str,
    pub compile_options_digest: Digest,
}

pub fn make_cache_key(inputs: &CacheKeyInputs<'_>) -> Result<CacheKey> {
    Ok(CacheKey {
        architecture_id: inputs.architecture_id.into(),
        architecture_revision: inputs.architecture_revision.into(),
        architecture_digest: inputs.architecture_digest.to_string(),
        resolved_config_digest: inputs.resolved_config_digest.to_string(),
        binding_plan_digest: content_digest(inputs.binding)?.to_string(),
        checkpoint_schema: inputs.checkpoint_schema.into(),
        target_fingerprint: inputs.target.capability_fingerprint.to_string(),
        precision_policy_digest: content_digest(inputs.precision_policy)?.to_string(),
        shape_profile_digest: shape_profile_digest(inputs.shape_profile)?.to_string(),
        kernel_registry_version: inputs.kernel_registry_version.into(),
        compiler_version: inputs.compiler_version.into(),
        iree_revision: inputs.iree_revision.into(),
        compile_options_digest: inputs.compile_options_digest.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dyninfer_core::{
        ArchitectureId, KvCacheDescriptor, KvCacheLayout, ScalarType, SchemaFingerprint,
        ShapeProfile,
    };

    fn sample_key() -> CacheKey {
        CacheKey {
            architecture_id: "llama.decoder".into(),
            architecture_revision: "0.1.0".into(),
            architecture_digest: "arch".into(),
            resolved_config_digest: "cfg".into(),
            binding_plan_digest: "bind".into(),
            checkpoint_schema: "abc".into(),
            target_fingerprint: "cpu".into(),
            precision_policy_digest: "precision".into(),
            shape_profile_digest: "shape".into(),
            kernel_registry_version: "1".into(),
            compiler_version: "0.1.0-stub".into(),
            iree_revision: "3.11.0".into(),
            compile_options_digest: "opts".into(),
        }
    }

    #[test]
    fn publish_and_lookup_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ArtifactCache::open(dir.path()).unwrap();
        let key = sample_key();
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
            precision_policy: PrecisionPolicy::default(),
            selected_kernels: vec![],
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
                storage: dyninfer_core::KvCacheStorage::StaticGlobals,
            },
            parameter_scope: "weights".into(),
            parameter_components: vec![],
            derived_parameters_required: false,
            vmfb_path: "model.vmfb".into(),
            prefill_window: 4,
            diagnostics: vec![],
        };
        cache.publish(&key, b"VMFBSTUB", &manifest).unwrap();
        let hit = cache.lookup(&key).unwrap().expect("hit");
        assert_eq!(fs::read(hit.vmfb_path).unwrap(), b"VMFBSTUB");
        assert!(!dir.path().join("parameters").exists());
    }

    #[test]
    fn config_change_changes_cache_key() {
        let mut a = sample_key();
        let mut b = sample_key();
        a.resolved_config_digest = "rope_theta=10000".into();
        b.resolved_config_digest = "rope_theta=500000".into();
        assert_ne!(a.digest().unwrap().as_str(), b.digest().unwrap().as_str());
    }

    #[test]
    fn precision_policy_change_changes_cache_key() {
        let mut a = sample_key();
        let mut b = sample_key();
        a.precision_policy_digest = "conservative-v1".into();
        b.precision_policy_digest = "conservative-v2".into();
        assert_ne!(a.digest().unwrap().as_str(), b.digest().unwrap().as_str());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn detects_live_lock_holder() {
        assert!(process_is_running(std::process::id()));
    }
}
