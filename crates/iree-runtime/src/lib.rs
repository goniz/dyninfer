//! Safe wrappers over IREE runtime execution.
//!
//! Milestone 1 drives pinned `iree-run-module` with SafeTensors external
//! parameters (`--parameters=weights=...`). The public API never exposes raw
//! IREE handles.

#![forbid(unsafe_code)]

mod tools;

pub use tools::{discover_run_module, IreeRunTools};

use dyninfer_error::{DynInferError, IreeRuntimeError, Result};
use std::path::{Path, PathBuf};
use tracing::info_span;

/// Process-wide IREE "instance" placeholder (tool-backed).
pub struct Instance {
    tools: IreeRunTools,
}

impl Instance {
    pub fn new() -> Result<Self> {
        let _span = info_span!("runtime.create_device").entered();
        Ok(Self {
            tools: IreeRunTools::discover()?,
        })
    }

    pub fn tools(&self) -> &IreeRunTools {
        &self.tools
    }
}

/// Loaded VMFB module bytes plus an optional on-disk path for tool invocation.
pub struct Module {
    pub bytes: Vec<u8>,
    path: Option<PathBuf>,
    _temp_dir: Option<tempfile::TempDir>,
}

impl Module {
    pub fn from_vmfb(bytes: Vec<u8>) -> Result<Self> {
        let _span = info_span!("runtime.load_vmfb", bytes = bytes.len()).entered();
        if bytes.is_empty() {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: "empty VMFB".into(),
                status_code: None,
            }));
        }
        if bytes.starts_with(b"DYNINFER_VMFB_STUB") {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: "refusing to load stub VMFB; compile with real IREE tools".into(),
                status_code: None,
            }));
        }
        let temp_dir = tempfile::tempdir().map_err(|e| {
            DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!("tempdir failed: {e}"),
                status_code: None,
            })
        })?;
        let path = temp_dir.path().join("module.vmfb");
        std::fs::write(&path, &bytes)?;
        Ok(Self {
            bytes,
            path: Some(path),
            _temp_dir: Some(temp_dir),
        })
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let bytes = std::fs::read(&path)?;
        Ok(Self {
            bytes,
            path: Some(path),
            _temp_dir: None,
        })
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}

/// Execution context bound to a module (+ optional external parameter file).
pub struct Context {
    instance: Instance,
    module: Module,
    parameters: Option<PathBuf>,
}

impl Context {
    pub fn create(instance: Instance, module: Module) -> Result<Self> {
        Ok(Self {
            instance,
            module,
            parameters: None,
        })
    }

    pub fn with_parameters(mut self, path: impl AsRef<Path>) -> Self {
        self.parameters = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn module_bytes(&self) -> &[u8] {
        &self.module.bytes
    }

    pub fn module_path(&self) -> Result<&Path> {
        self.module.path.as_deref().ok_or_else(|| {
            DynInferError::IreeRuntime(IreeRuntimeError {
                message: "module has no filesystem path for tool invocation".into(),
                status_code: None,
            })
        })
    }

    pub fn invoke_add(&self, a: &[f32; 4], b: &[f32; 4]) -> Result<Vec<f32>> {
        let path = self.module_path()?;
        self.instance
            .tools
            .run_add(path, a, b, self.parameters.as_deref())
    }

    pub fn invoke_prefill(&self, tokens: &[i64]) -> Result<Vec<f32>> {
        let path = self.module_path()?;
        self.instance
            .tools
            .run_prefill(path, tokens, self.parameters.as_deref())
    }

    pub fn invoke_decode(&self, token: i64) -> Result<Vec<f32>> {
        let path = self.module_path()?;
        self.instance
            .tools
            .run_decode(path, token, self.parameters.as_deref())
    }
}
