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
use std::sync::{Arc, Mutex};
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

/// Execution context bound to a module (+ optional external parameters).
///
/// Each [`Context`] owns one native IREE session (and therefore one mutable KV
/// cache). Do not share a single context across concurrent model sessions.
pub struct Context {
    _instance: Instance,
    module: Module,
    parameters: Option<PathBuf>,
    /// In-memory f32 parameter blobs (Vulkan bf16 promote). Kept alive for the
    /// session; mutually exclusive with [`Self::parameters`]. Shared via [`Arc`]
    /// so multiple sessions can reuse the same host weight staging.
    host_parameters: Option<Arc<HostParameterStorage>>,
    /// IREE HAL driver name or full device URI (`hip`, `vulkan://…`, …).
    /// Empty → local-task.
    device: Option<String>,
    session: Mutex<Option<NativeSession>>,
}

/// Owns CString keys + f32 LE bytes for `dyninfer_iree_session_create_with_host_params`.
pub struct HostParameterStorage {
    entries: Vec<(CString, Vec<u8>)>,
}

impl HostParameterStorage {
    pub fn from_f32_entries(entries: Vec<(String, Vec<u8>)>) -> Result<Self> {
        let mut out = Vec::with_capacity(entries.len());
        for (key, data) in entries {
            let ckey = CString::new(key.as_str()).map_err(|e| {
                DynInferError::IreeRuntime(IreeRuntimeError {
                    message: format!("invalid parameter key {key:?}: {e}"),
                    status_code: None,
                })
            })?;
            out.push((ckey, data));
        }
        Ok(Self { entries: out })
    }
}

impl Context {
    pub fn create(instance: Instance, module: Module) -> Result<Self> {
        Ok(Self {
            _instance: instance,
            module,
            parameters: None,
            host_parameters: None,
            device: None,
            session: Mutex::new(None),
        })
    }

    pub fn with_parameters(mut self, path: impl AsRef<Path>) -> Self {
        self.parameters = Some(path.as_ref().to_path_buf());
        self.host_parameters = None;
        self
    }

    /// Bind host-owned f32 parameter blobs (no parameter file). Used when the
    /// VMFB expects promoted f32 weights.
    pub fn with_host_parameters(mut self, storage: HostParameterStorage) -> Self {
        self.host_parameters = Some(Arc::new(storage));
        self.parameters = None;
        self
    }

    /// Share an existing host parameter staging across contexts.
    pub fn with_host_parameters_shared(mut self, storage: Arc<HostParameterStorage>) -> Self {
        self.host_parameters = Some(storage);
        self.parameters = None;
        self
    }

    /// Set HAL device/driver or full device URI (e.g. `hip`, `vulkan://GPU-…`).
    /// `rocm` aliases to `hip`. Bare driver names still select the default device.
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

            let mut ptr = ptr::null_mut();
            let rc = if let Some(host) = self.host_parameters.as_ref() {
                let c_params: Vec<sys::dyninfer_iree_host_param_t> = host
                    .entries
                    .iter()
                    .map(|(key, data)| sys::dyninfer_iree_host_param_t {
                        key: key.as_ptr(),
                        data: data.as_ptr().cast(),
                        length: data.len(),
                    })
                    .collect();
                unsafe {
                    sys::dyninfer_iree_session_create_with_host_params(
                        device_c
                            .as_ref()
                            .map(|c| c.as_ptr())
                            .unwrap_or(ptr::null()),
                        vmfb_c.as_ptr(),
                        c_params.as_ptr(),
                        c_params.len(),
                        &mut ptr,
                    )
                }
            } else {
                let params_c = self.parameters.as_deref().map(path_cstring).transpose()?;
                unsafe {
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
                }
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

    pub fn invoke_decode(&self, token: i64, pos: i64, attn_bias: &[f32]) -> Result<Vec<f32>> {
        if attn_bias.is_empty() {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: "decode requires a non-empty attn_bias".into(),
                status_code: None,
            }));
        }
        self.with_session(|session| {
            let mut out = ptr::null_mut();
            let mut count = 0usize;
            let rc = unsafe {
                sys::dyninfer_iree_session_invoke_decode(
                    session,
                    token,
                    pos,
                    attn_bias.as_ptr(),
                    attn_bias.len(),
                    &mut out,
                    &mut count,
                )
            };
            take_f32_buf(rc, out, count)
        })
    }

    pub fn invoke_decode_at(&self, token: i64, pos: i64, max_kv: usize) -> Result<Vec<f32>> {
        let mut bias = Vec::new();
        fill_causal_attn_bias(&mut bias, pos, max_kv);
        self.invoke_decode(token, pos, &bias)
    }
}

/// Fill `bias` in-place: `0` for `j <= pos`, else `-1e7`. Reuses capacity.
pub fn fill_causal_attn_bias(bias: &mut Vec<f32>, pos: i64, max_kv: usize) {
    let n = max_kv.max(1);
    bias.clear();
    bias.resize(n, 0.0);
    for (j, b) in bias.iter_mut().enumerate() {
        if (j as i64) > pos {
            *b = -1.0e7;
        }
    }
}

/// `0` for `j <= pos`, else `-1e7`.
pub fn causal_attn_bias(pos: i64, max_kv: usize) -> Vec<f32> {
    let mut bias = Vec::new();
    fill_causal_attn_bias(&mut bias, pos, max_kv);
    bias
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
        // Scratch pointer must not be freed (session-owned); ignore on error too.
        return Err(native_error(rc));
    }
    if out.is_null() {
        return Err(DynInferError::IreeRuntime(IreeRuntimeError {
            message: "null logits pointer from IREE session".into(),
            status_code: None,
        }));
    }
    // Copy immediately: pointer aliases session scratch invalidated by the next invoke.
    let values = unsafe { std::slice::from_raw_parts(out, count) }.to_vec();
    unsafe { sys::dyninfer_iree_free(out.cast()) };
    Ok(values)
}
