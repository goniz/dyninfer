//! Drive pinned `iree-compile` / discover IREE tool installs.
//!
//! Primary discovery path is Bazel runfiles (`//bazel/iree:tools`). Optional
//! env/`PATH`/`third_party/iree-venv` fallbacks remain for cargo-only workflows.

use dyninfer_error::{CompilationError, DynInferError, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{debug, info};

/// Resolved paths to IREE CLI tools.
#[derive(Debug, Clone)]
pub struct IreeTools {
    pub root: PathBuf,
    pub compile: PathBuf,
    pub run_module: PathBuf,
}

impl IreeTools {
    /// Discover tools from Bazel runfiles, env overrides, `PATH`, or local venv.
    pub fn discover() -> Result<Self> {
        if let Ok(compile) = std::env::var("DYNINFER_IREE_COMPILE") {
            let compile = PathBuf::from(compile);
            let run = std::env::var("DYNINFER_IREE_RUN_MODULE")
                .map(PathBuf::from)
                .unwrap_or_else(|_| compile.with_file_name("iree-run-module"));
            if compile.is_file() {
                return Ok(Self {
                    root: compile
                        .parent()
                        .unwrap_or_else(|| Path::new("."))
                        .to_path_buf(),
                    compile,
                    run_module: run,
                });
            }
        }
        if let Ok(home) = std::env::var("DYNINFER_IREE_HOME") {
            let home = PathBuf::from(home);
            return Self::from_root(&home);
        }
        if let Some(tools) = from_runfiles() {
            return Ok(tools);
        }
        if let Ok(path) = which("iree-compile") {
            let run =
                which("iree-run-module").unwrap_or_else(|_| path.with_file_name("iree-run-module"));
            return Ok(Self {
                root: path.parent().unwrap_or(Path::new(".")).to_path_buf(),
                compile: path,
                run_module: run,
            });
        }
        for base in candidate_roots() {
            let venv = base.join("third_party/iree-venv");
            if venv.join("bin/iree-compile").is_file() {
                return Self::from_root(&venv);
            }
        }
        Err(DynInferError::Compilation(CompilationError {
            message: "IREE tools not found; build with Bazel (`//bazel/iree:tools`) or set DYNINFER_IREE_COMPILE"
                .into(),
            pass: Some("iree-discover".into()),
            diagnostics: vec![],
        }))
    }

    fn from_root(root: &Path) -> Result<Self> {
        let bin = if root.join("bin/iree-compile").is_file() {
            root.join("bin")
        } else {
            root.to_path_buf()
        };
        let compile = bin.join("iree-compile");
        let run_module = bin.join("iree-run-module");
        if !compile.is_file() {
            return Err(DynInferError::Compilation(CompilationError {
                message: format!("iree-compile not found under {}", root.display()),
                pass: Some("iree-discover".into()),
                diagnostics: vec![],
            }));
        }
        Ok(Self {
            root: root.to_path_buf(),
            compile,
            run_module,
        })
    }

    /// Compile MLIR text to VMFB bytes for the given HAL driver profile.
    pub fn compile_mlir(&self, mlir: &str, driver: &str) -> Result<Vec<u8>> {
        let dir = tempfile::tempdir().map_err(|e| {
            DynInferError::Compilation(CompilationError {
                message: format!("tempdir failed: {e}"),
                pass: Some("iree-compile".into()),
                diagnostics: vec![],
            })
        })?;
        let mlir_path = dir.path().join("input.mlir");
        let vmfb_path = dir.path().join("out.vmfb");
        std::fs::write(&mlir_path, mlir)?;

        let mut cmd = Command::new(&self.compile);
        cmd.arg(&mlir_path).arg("-o").arg(&vmfb_path);
        match driver {
            "vulkan" => {
                cmd.arg("--iree-hal-target-device=vulkan");
            }
            _ => {
                // local-task / llvm-cpu host
                cmd.arg("--iree-hal-target-device=local")
                    .arg("--iree-hal-local-target-device-backends=llvm-cpu")
                    .arg("--iree-llvmcpu-target-cpu=generic");
            }
        }

        info!(?cmd, "invoking iree-compile");
        let output = cmd.output().map_err(|e| {
            DynInferError::Compilation(CompilationError {
                message: format!("failed to spawn iree-compile: {e}"),
                pass: Some("iree-compile".into()),
                diagnostics: vec![],
            })
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            return Err(DynInferError::Compilation(CompilationError {
                message: format!("iree-compile failed: {stderr}\n{stdout}"),
                pass: Some("iree-compile".into()),
                diagnostics: vec![],
            }));
        }
        let bytes = std::fs::read(&vmfb_path)?;
        debug!(bytes = bytes.len(), "compiled VMFB");
        if bytes.is_empty() {
            return Err(DynInferError::Compilation(CompilationError {
                message: "iree-compile produced empty VMFB".into(),
                pass: Some("iree-compile".into()),
                diagnostics: vec![],
            }));
        }
        Ok(bytes)
    }
}

fn from_runfiles() -> Option<IreeTools> {
    let roots = runfiles_roots();
    if roots.is_empty() {
        return None;
    }
    for arch in ["x86_64", "aarch64"] {
        let compile_repo = format!("iree_compiler_linux_{arch}");
        let run_repo = format!("iree_runtime_linux_{arch}");
        let compile_inner = "iree/compiler/_mlir_libs/iree-compile";
        let run_inner = "iree/_runtime_libs/iree-run-module";
        for root in &roots {
            if let (Some(compile), Some(run_module)) = (
                rlocation(root, &compile_repo, compile_inner),
                rlocation(root, &run_repo, run_inner),
            ) {
                return Some(IreeTools {
                    root: compile.parent().unwrap_or(root).to_path_buf(),
                    compile,
                    run_module,
                });
            }
        }
    }
    None
}

/// Resolve a file under a Bazel external repo in runfiles.
///
/// Bzlmod uses `+http_archive+<name>/...`; workspace-style uses `<name>/...`.
fn rlocation(root: &Path, repo: &str, inner: &str) -> Option<PathBuf> {
    for prefix in [repo.to_string(), format!("+http_archive+{repo}")] {
        let candidate = root.join(&prefix).join(inner);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    let rel = format!("{repo}/{inner}");
    manifest_rlocation(&rel).or_else(|| manifest_rlocation(&format!("+http_archive+{rel}")))
}

fn runfiles_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for key in ["RUNFILES_DIR", "TEST_SRCDIR"] {
        if let Ok(dir) = std::env::var(key) {
            let p = PathBuf::from(dir);
            if p.is_dir() {
                roots.push(p);
            }
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

fn manifest_rlocation(rel: &str) -> Option<PathBuf> {
    let manifest = std::env::var_os("RUNFILES_MANIFEST_FILE")?;
    let text = std::fs::read_to_string(manifest).ok()?;
    for line in text.lines() {
        let mut parts = line.splitn(2, ' ');
        let key = parts.next()?;
        let path = parts.next()?;
        if key == rel || key.ends_with(&format!("/{rel}")) {
            return Some(PathBuf::from(path));
        }
    }
    None
}

fn which(bin: &str) -> std::result::Result<PathBuf, ()> {
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            let candidate = Path::new(dir).join(bin);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    Err(())
}

fn candidate_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let mut p = PathBuf::from(manifest);
        for _ in 0..6 {
            roots.push(p.clone());
            if !p.pop() {
                break;
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let mut p = cwd;
        for _ in 0..6 {
            roots.push(p.clone());
            if !p.pop() {
                break;
            }
        }
    }
    roots
}
