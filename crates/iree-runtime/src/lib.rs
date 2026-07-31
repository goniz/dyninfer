//! Safe wrappers over IREE runtime execution.
//!
//! Prefill/decode/add invoke the in-process C session (`iree-runtime-sys` →
//! `@iree_core`). Device discovery still uses `iree-run-module --dump_devices`.

mod tools;

pub use tools::{discover_run_module, IreeRunTools};

use dyninfer_error::{DynInferError, IreeRuntimeError, Result};
use iree_runtime_sys as sys;
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Mutex;
use tracing::info_span;

/// Process-wide IREE instance (tool discovery + native session factory).
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

/// Loaded VMFB module bytes plus an optional on-disk path for native load.
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

struct NativeSession {
    ptr: *mut sys::dyninfer_iree_session_t,
}

// IREE sessions are thread-compatible (external sync required). We only use
// them from the generate loop on one thread; Send allows Arc<Context>.
unsafe impl Send for NativeSession {}
unsafe impl Sync for NativeSession {}

impl Drop for NativeSession {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::dyninfer_iree_session_destroy(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

/// Execution context bound to a module (+ optional external parameter file).
pub struct Context {
    _instance: Instance,
    module: Module,
    parameters: Option<PathBuf>,
    /// IREE HAL driver name (`hip`, `vulkan`, …). Empty → local-task.
    device: Option<String>,
    session: Mutex<Option<NativeSession>>,
}

impl Context {
    pub fn create(instance: Instance, module: Module) -> Result<Self> {
        Ok(Self {
            _instance: instance,
            module,
            parameters: None,
            device: None,
            session: Mutex::new(None),
        })
    }

    pub fn with_parameters(mut self, path: impl AsRef<Path>) -> Self {
        self.parameters = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set HAL device/driver (e.g. `hip`, `vulkan`). `rocm` aliases to `hip`.
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        let d = device.into();
        self.device = if d.is_empty()
            || d == "local-task"
            || d == "local-sync"
            || d == "local"
            || d == "cpu"
        {
            None
        } else if d == "rocm" {
            Some("hip".into())
        } else {
            Some(d)
        };
        self
    }

    pub fn module_bytes(&self) -> &[u8] {
        &self.module.bytes
    }

    pub fn module_path(&self) -> Result<&Path> {
        self.module.path.as_deref().ok_or_else(|| {
            DynInferError::IreeRuntime(IreeRuntimeError {
                message: "module has no filesystem path for native load".into(),
                status_code: None,
            })
        })
    }

    fn with_session<R>(&self, f: impl FnOnce(*mut sys::dyninfer_iree_session_t) -> Result<R>) -> Result<R> {
        let mut guard = self.session.lock().map_err(|_| {
            DynInferError::IreeRuntime(IreeRuntimeError {
                message: "IREE session mutex poisoned".into(),
                status_code: None,
            })
        })?;
        if guard.is_none() {
            let vmfb = self.module_path()?;
            let vmfb_c = path_cstring(vmfb)?;
            let device_c = self
                .device
                .as_deref()
                .map(CString::new)
                .transpose()
                .map_err(|e| {
                    DynInferError::IreeRuntime(IreeRuntimeError {
                        message: format!("invalid device string: {e}"),
                        status_code: None,
                    })
                })?;
            let params_c = self.parameters.as_deref().map(path_cstring).transpose()?;

            let mut ptr = ptr::null_mut();
            let rc = unsafe {
                sys::dyninfer_iree_session_create(
                    device_c
                        .as_ref()
                        .map(|c| c.as_ptr())
                        .unwrap_or(ptr::null()),
                    vmfb_c.as_ptr(),
                    params_c
                        .as_ref()
                        .map(|c| c.as_ptr())
                        .unwrap_or(ptr::null()),
                    &mut ptr,
                )
            };
            if rc != 0 || ptr.is_null() {
                return Err(native_error(rc));
            }
            *guard = Some(NativeSession { ptr });
        }
        let ptr = guard.as_ref().expect("session just created").ptr;
        f(ptr)
    }

    pub fn invoke_add(&self, a: &[f32; 4], b: &[f32; 4]) -> Result<Vec<f32>> {
        self.with_session(|session| {
            let mut out = ptr::null_mut();
            let mut count = 0usize;
            let rc = unsafe {
                sys::dyninfer_iree_session_invoke_add(
                    session,
                    a.as_ptr(),
                    b.as_ptr(),
                    &mut out,
                    &mut count,
                )
            };
            take_f32_buf(rc, out, count)
        })
    }

    pub fn invoke_prefill(&self, tokens: &[i64], last: i64) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: "prefill requires a non-empty token window".into(),
                status_code: None,
            }));
        }
        self.with_session(|session| {
            let mut out = ptr::null_mut();
            let mut count = 0usize;
            let rc = unsafe {
                sys::dyninfer_iree_session_invoke_prefill(
                    session,
                    tokens.as_ptr(),
                    tokens.len(),
                    last,
                    &mut out,
                    &mut count,
                )
            };
            take_f32_buf(rc, out, count)
        })
    }

    pub fn invoke_decode(&self, token: i64) -> Result<Vec<f32>> {
        self.with_session(|session| {
            let mut out = ptr::null_mut();
            let mut count = 0usize;
            let rc = unsafe {
                sys::dyninfer_iree_session_invoke_decode(session, token, &mut out, &mut count)
            };
            take_f32_buf(rc, out, count)
        })
    }
}

fn path_cstring(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes()).map_err(|e| {
        DynInferError::IreeRuntime(IreeRuntimeError {
            message: format!("path contains NUL: {e}"),
            status_code: None,
        })
    })
}

fn last_error_string() -> String {
    unsafe {
        let p = sys::dyninfer_iree_last_error();
        if p.is_null() {
            return "unknown IREE runtime error".into();
        }
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

fn native_error(code: i32) -> DynInferError {
    DynInferError::IreeRuntime(IreeRuntimeError {
        message: last_error_string(),
        status_code: Some(code),
    })
}

fn take_f32_buf(rc: i32, out: *mut f32, count: usize) -> Result<Vec<f32>> {
    if rc != 0 {
        if !out.is_null() {
            unsafe { sys::dyninfer_iree_free(out.cast()) };
        }
        return Err(native_error(rc));
    }
    if out.is_null() {
        return Err(DynInferError::IreeRuntime(IreeRuntimeError {
            message: "null logits pointer from IREE session".into(),
            status_code: None,
        }));
    }
    let values = unsafe { std::slice::from_raw_parts(out, count) }.to_vec();
    unsafe { sys::dyninfer_iree_free(out.cast()) };
    Ok(values)
}
