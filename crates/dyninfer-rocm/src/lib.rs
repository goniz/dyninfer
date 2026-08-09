//! Discovery and child-process setup for the Bazel-pinned TheRock SDK.

#![forbid(unsafe_code)]

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;

const REPOSITORY: &str = "therock_rocm_gfx1151";
const MANIFEST: &str = "share/therock/therock_manifest.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmSdk {
    root: PathBuf,
}

impl RocmSdk {
    pub fn discover() -> Option<Self> {
        for key in ["DYNINFER_ROCM_HOME", "ROCM_HOME", "ROCM_PATH"] {
            if let Some(root) = std::env::var_os(key).map(PathBuf::from) {
                if Self::is_sdk(&root) {
                    return Some(Self { root });
                }
            }
        }

        for runfiles in runfiles_roots() {
            for prefix in [
                REPOSITORY.to_string(),
                format!("+http_archive+{REPOSITORY}"),
            ] {
                let root = runfiles.join(prefix);
                if Self::is_sdk(&root) {
                    return Some(Self { root });
                }
            }
        }

        let manifest_path = rlocation_manifest(MANIFEST)?;
        let root = manifest_path.ancestors().nth(3)?.to_path_buf();
        Self::is_sdk(&root).then_some(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn bitcode_dir(&self) -> PathBuf {
        self.root.join("lib/llvm/amdgcn/bitcode")
    }

    pub fn linker(&self) -> PathBuf {
        self.root.join("lib/llvm/bin/ld.lld")
    }

    pub fn hip_runtime(&self) -> PathBuf {
        self.root.join("lib/libamdhip64.so")
    }

    /// Configure only the child command; do not mutate the current process's
    /// environment, which may already be multi-threaded.
    pub fn configure_command(&self, command: &mut Command) {
        command
            .env("ROCM_HOME", &self.root)
            .env("ROCM_PATH", &self.root);
        prepend_env_path(
            command,
            "PATH",
            [self.root.join("bin"), self.root.join("lib/llvm/bin")],
        );
        prepend_env_path(
            command,
            "LD_LIBRARY_PATH",
            [self.root.join("lib"), self.root.join("lib64")],
        );
    }

    fn is_sdk(root: &Path) -> bool {
        root.join(MANIFEST).is_file()
            && root.join("lib/llvm/bin/ld.lld").is_file()
            && root.join("lib/llvm/amdgcn/bitcode/ocml.bc").is_file()
            && root.join("lib/llvm/amdgcn/bitcode/ockl.bc").is_file()
            && root.join("lib/libamdhip64.so").is_file()
    }
}

fn prepend_env_path<const N: usize>(command: &mut Command, key: &str, paths: [PathBuf; N]) {
    let configured = command
        .get_envs()
        .find_map(|(name, value)| (name == OsStr::new(key)).then(|| value.map(OsString::from)))
        .flatten();
    let inherited = configured
        .or_else(|| std::env::var_os(key))
        .unwrap_or_default();
    let mut values = paths.into_iter().collect::<Vec<_>>();
    values.extend(std::env::split_paths(&inherited));
    if let Ok(joined) = std::env::join_paths(values) {
        command.env(key, joined);
    }
}

fn runfiles_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for key in ["RUNFILES_DIR", "TEST_SRCDIR"] {
        if let Some(path) = std::env::var_os(key)
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
        {
            roots.push(path);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let sibling = PathBuf::from(format!("{}.runfiles", exe.display()));
        if sibling.is_dir() {
            roots.push(sibling);
        }
    }
    roots
}

fn rlocation_manifest(inner: &str) -> Option<PathBuf> {
    let manifest = std::env::var_os("RUNFILES_MANIFEST_FILE")?;
    let text = std::fs::read_to_string(manifest).ok()?;
    let suffixes = [
        format!("{REPOSITORY}/{inner}"),
        format!("+http_archive+{REPOSITORY}/{inner}"),
    ];
    for line in text.lines() {
        let Some((key, value)) = line.split_once(' ') else {
            continue;
        };
        if suffixes.iter().any(|suffix| key.ends_with(suffix)) {
            return Some(PathBuf::from(OsString::from(value)));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bazel_runfiles_contain_pinned_sdk() {
        let sdk = RocmSdk::discover().expect("TheRock SDK must be present in Bazel runfiles");
        assert!(sdk.bitcode_dir().join("ocml.bc").is_file());
        assert!(sdk.linker().is_file());
        assert!(sdk.hip_runtime().is_file());
    }
}
