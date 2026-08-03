//! Resolve Hugging Face Hub model IDs against the local HF cache.
//!
//! Layout (huggingface_hub):
//! `$HF_HUB_CACHE/models--org--name/refs/<rev>` → snapshot hash
//! `$HF_HUB_CACHE/models--org--name/snapshots/<hash>/`

use dyninfer_error::{DynInferError, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Default revision when none is specified (`main`).
pub const DEFAULT_HF_REVISION: &str = "main";

/// Locate the Hugging Face hub cache root.
///
/// Order: `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `$HF_HOME/hub`,
/// then `~/.cache/huggingface/hub`.
pub fn hf_hub_cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(p);
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    let mut p = dirs_home();
    p.push(".cache");
    p.push("huggingface");
    p.push("hub");
    p
}

fn dirs_home() -> PathBuf {
    if let Ok(h) = std::env::var("HOME") {
        return PathBuf::from(h);
    }
    PathBuf::from(".")
}

/// Encode `org/name` (or bare `name`) as the Hub cache folder `models--org--name`.
pub fn hf_repo_folder_name(repo_id: &str) -> String {
    let cleaned = repo_id.trim().trim_start_matches("https://huggingface.co/");
    let cleaned = cleaned.trim_end_matches('/');
    let parts: Vec<&str> = cleaned.split('/').filter(|s| !s.is_empty()).collect();
    match parts.as_slice() {
        [org, name, ..] => format!("models--{org}--{name}"),
        [name] => format!("models--{name}"),
        _ => format!("models--{cleaned}"),
    }
}

/// Resolve a Hub repo id to a local snapshot directory that contains weights.
///
/// `revision` defaults to [`DEFAULT_HF_REVISION`]. Does not download; the
/// snapshot must already exist in the local cache (e.g. via `hf download`).
pub fn resolve_hf_snapshot(repo_id: &str, revision: Option<&str>) -> Result<PathBuf> {
    let revision = revision.unwrap_or(DEFAULT_HF_REVISION);
    let cache = hf_hub_cache_dir();
    let repo_dir = cache.join(hf_repo_folder_name(repo_id));
    if !repo_dir.is_dir() {
        return Err(DynInferError::io(format!(
            "HF model `{repo_id}` not in local cache at {} (run: hf download {repo_id})",
            repo_dir.display()
        )));
    }

    let snap = resolve_revision_snapshot(&repo_dir, revision)?;
    ensure_model_files(&snap, repo_id)?;
    Ok(snap)
}

fn resolve_revision_snapshot(repo_dir: &Path, revision: &str) -> Result<PathBuf> {
    let refs_file = repo_dir.join("refs").join(revision);
    if refs_file.is_file() {
        let hash = fs::read_to_string(&refs_file)
            .map_err(|e| {
                DynInferError::io_path(refs_file.display().to_string(), format!("read ref: {e}"))
            })?
            .trim()
            .to_string();
        if hash.is_empty() {
            return Err(DynInferError::io(format!(
                "empty HF ref file {}",
                refs_file.display()
            )));
        }
        let snap = repo_dir.join("snapshots").join(&hash);
        if snap.is_dir() {
            return Ok(snap);
        }
        return Err(DynInferError::io(format!(
            "HF ref `{revision}` → `{hash}` but snapshot missing at {}",
            snap.display()
        )));
    }

    // Allow passing a raw snapshot hash / partial prefix.
    let snapshots = repo_dir.join("snapshots");
    let direct = snapshots.join(revision);
    if direct.is_dir() {
        return Ok(direct);
    }
    if snapshots.is_dir() {
        if let Ok(rd) = fs::read_dir(&snapshots) {
            let mut matches: Vec<PathBuf> = rd
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.is_dir()
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with(revision))
                })
                .collect();
            matches.sort();
            if matches.len() == 1 {
                return Ok(matches.remove(0));
            }
            // Prefer the newest snapshot mtime when revision is "main"-like missing.
            if revision == DEFAULT_HF_REVISION {
                if let Some(best) = newest_snapshot(&snapshots) {
                    return Ok(best);
                }
            }
        }
    }

    Err(DynInferError::io(format!(
        "HF revision `{revision}` not found under {} (no refs/{revision})",
        repo_dir.display()
    )))
}

fn newest_snapshot(snapshots: &Path) -> Option<PathBuf> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    let rd = fs::read_dir(snapshots).ok()?;
    for e in rd.flatten() {
        let p = e.path();
        if !p.is_dir() {
            continue;
        }
        let modified = e.metadata().ok().and_then(|m| m.modified().ok())?;
        match &best {
            None => best = Some((modified, p)),
            Some((t, _)) if modified > *t => best = Some((modified, p)),
            _ => {}
        }
    }
    best.map(|(_, p)| p)
}

fn ensure_model_files(snap: &Path, repo_id: &str) -> Result<()> {
    let st = snap.join("model.safetensors");
    let sharded = snap.join("model.safetensors.index.json");
    if st.is_file() || sharded.is_file() {
        return Ok(());
    }
    // Some repos use a single non-standard name; accept any *.safetensors.
    if let Ok(rd) = fs::read_dir(snap) {
        if rd
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        {
            return Ok(());
        }
    }
    Err(DynInferError::io(format!(
        "HF snapshot for `{repo_id}` at {} has no .safetensors weights",
        snap.display()
    )))
}

/// Find `model.safetensors` (or the first `*.safetensors`) under a model directory.
pub fn find_safetensors_checkpoint(model_dir: &Path) -> Result<PathBuf> {
    let preferred = model_dir.join("model.safetensors");
    if preferred.is_file() {
        return Ok(preferred);
    }
    let mut found: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = fs::read_dir(model_dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "safetensors")
                && !p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.contains("index"))
            {
                found.push(p);
            }
        }
    }
    found.sort();
    found.into_iter().next().ok_or_else(|| {
        DynInferError::io(format!(
            "no .safetensors checkpoint in {}",
            model_dir.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folder_name_encoding() {
        assert_eq!(
            hf_repo_folder_name("Maykeye/TinyLLama-v0"),
            "models--Maykeye--TinyLLama-v0"
        );
        assert_eq!(
            hf_repo_folder_name("https://huggingface.co/Maykeye/TinyLLama-v0"),
            "models--Maykeye--TinyLLama-v0"
        );
    }

    #[test]
    fn resolves_maykeye_if_cached() {
        let Ok(snap) = resolve_hf_snapshot("Maykeye/TinyLLama-v0", Some("main")) else {
            eprintln!("skip: Maykeye/TinyLLama-v0 not in HF cache");
            return;
        };
        assert!(
            snap.join("model.safetensors").is_file() || find_safetensors_checkpoint(&snap).is_ok()
        );
        assert!(snap.join("tokenizer.json").is_file());
    }
}
