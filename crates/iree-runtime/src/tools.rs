//! Tool-backed IREE runtime using `iree-run-module`.
//!
//! Primary discovery path is Bazel runfiles (`//bazel/iree:tools`).

use dyninfer_error::{DynInferError, IreeRuntimeError, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{debug, info};

#[derive(Debug, Clone)]
pub struct IreeRunTools {
    pub run_module: PathBuf,
}

impl IreeRunTools {
    pub fn discover() -> Result<Self> {
        if let Ok(path) = std::env::var("DYNINFER_IREE_RUN_MODULE") {
            let path = PathBuf::from(path);
            if path.is_file() {
                return Ok(Self { run_module: path });
            }
        }
        if let Ok(home) = std::env::var("DYNINFER_IREE_HOME") {
            let candidate = PathBuf::from(home).join("bin/iree-run-module");
            if candidate.is_file() {
                return Ok(Self {
                    run_module: candidate,
                });
            }
        }
        if let Some(path) = from_runfiles() {
            return Ok(Self { run_module: path });
        }
        if let Ok(path) = which("iree-run-module") {
            return Ok(Self { run_module: path });
        }
        for base in candidate_roots() {
            let candidate = base.join("third_party/iree-venv/bin/iree-run-module");
            if candidate.is_file() {
                return Ok(Self {
                    run_module: candidate,
                });
            }
        }
        Err(DynInferError::IreeRuntime(IreeRuntimeError {
            message: "iree-run-module not found; build with Bazel (`//bazel/iree:tools`) or set DYNINFER_IREE_RUN_MODULE"
                .into(),
            status_code: None,
        }))
    }

    /// Enumerate HAL devices via `iree-run-module --dump_devices` (stdout text).
    ///
    /// Unavailable optional drivers (e.g. CUDA without libcuda) may print to
    /// stderr; we still return stdout so other drivers remain usable.
    pub fn dump_devices(&self) -> Result<String> {
        let mut cmd = Command::new(&self.run_module);
        if let Some(sdk) = dyninfer_rocm::RocmSdk::discover() {
            sdk.configure_command(&mut cmd);
        }
        cmd.arg("--dump_devices");
        debug!(?cmd, "invoking iree-run-module --dump_devices");
        let output = cmd.output().map_err(|e| {
            DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!("failed to spawn iree-run-module: {e}"),
                status_code: None,
            })
        })?;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if stdout.trim().is_empty() && !output.status.success() {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!("iree-run-module --dump_devices failed: {stderr}"),
                status_code: output.status.code(),
            }));
        }
        Ok(stdout)
    }

    pub fn run_add(
        &self,
        module: &Path,
        a: &[f32; 4],
        b: &[f32; 4],
        parameters: Option<&Path>,
        device: Option<&str>,
    ) -> Result<Vec<f32>> {
        let input_a = format!(
            "4xf32={}",
            a.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let input_b = format!(
            "4xf32={}",
            b.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let stdout = self.invoke(module, "add", &[&input_a, &input_b], parameters, device)?;
        parse_f32_buffer_view(&stdout)
    }

    pub fn run_prefill(
        &self,
        module: &Path,
        tokens: &[i64],
        last: i64,
        parameters: Option<&Path>,
        device: Option<&str>,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: "prefill requires a non-empty token window".into(),
                status_code: None,
            }));
        }
        let input = format!(
            "{}xi64={}",
            tokens.len(),
            tokens
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let last_input = format!("i64={last}");
        let stdout = self.invoke(
            module,
            "prefill",
            &[&input, &last_input],
            parameters,
            device,
        )?;
        parse_f32_buffer_view(&stdout)
    }

    pub fn run_decode(
        &self,
        module: &Path,
        token: i64,
        pos: i64,
        parameters: Option<&Path>,
        device: Option<&str>,
    ) -> Result<Vec<f32>> {
        let input = format!("i64={token}");
        let pos_input = format!("i64={pos}");
        let stdout = self.invoke(module, "decode", &[&input, &pos_input], parameters, device)?;
        parse_f32_buffer_view(&stdout)
    }

    fn invoke(
        &self,
        module: &Path,
        function: &str,
        inputs: &[&str],
        parameters: Option<&Path>,
        device: Option<&str>,
    ) -> Result<String> {
        let mut cmd = Command::new(&self.run_module);
        if let Some(sdk) = dyninfer_rocm::RocmSdk::discover() {
            sdk.configure_command(&mut cmd);
        }
        cmd.arg(format!("--module={}", module.display()))
            .arg(format!("--function={function}"))
            // Default (1024) elides large logits with `...`; we need full buffers.
            .arg("--output_max_element_count=1048576");
        if let Some(device) = device {
            cmd.arg(format!("--device={device}"));
        }
        if let Some(params) = parameters {
            cmd.arg(format!("--parameters=weights={}", params.display()));
        }
        for input in inputs {
            cmd.arg(format!("--input={input}"));
        }
        info!(?cmd, "invoking iree-run-module");
        let output = cmd.output().map_err(|e| {
            DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!("failed to spawn iree-run-module: {e}"),
                status_code: None,
            })
        })?;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if !output.status.success() {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!("iree-run-module failed: {stderr}\n{stdout}"),
                status_code: output.status.code(),
            }));
        }
        debug!(%stdout, "iree-run-module ok");
        Ok(stdout)
    }
}

pub fn discover_run_module() -> Result<PathBuf> {
    Ok(IreeRunTools::discover()?.run_module)
}

fn from_runfiles() -> Option<PathBuf> {
    let roots = runfiles_roots();
    for arch in ["x86_64", "aarch64"] {
        let repo = format!("iree_runtime_linux_{arch}");
        let inner = "iree/_runtime_libs/iree-run-module";
        for root in &roots {
            if let Some(path) = rlocation(root, &repo, inner) {
                return Some(path);
            }
        }
    }
    None
}

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

fn parse_f32_buffer_view(stdout: &str) -> Result<Vec<f32>> {
    // Prefer the first result buffer (logits); ignore later KV dumps if present.
    for line in stdout.lines() {
        let line = line.trim();
        if let Some(idx) = line.find("xf32=") {
            let values = &line[idx + "xf32=".len()..];
            let nums: Result<Vec<f32>, _> = values
                .replace('[', " ")
                .replace(']', " ")
                .split_whitespace()
                .map(|s| {
                    s.parse::<f32>().map_err(|e| {
                        DynInferError::IreeRuntime(IreeRuntimeError {
                            message: format!("failed to parse f32 `{s}`: {e}"),
                            status_code: None,
                        })
                    })
                })
                .collect();
            return nums;
        }
    }
    Err(DynInferError::IreeRuntime(IreeRuntimeError {
        message: format!("no xf32 buffer view in iree-run-module output:\n{stdout}"),
        status_code: None,
    }))
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
