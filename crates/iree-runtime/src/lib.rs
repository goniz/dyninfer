//! Safe wrappers over IREE runtime execution.
//!
//! Prefill/decode/add invoke the in-process C session (`iree-runtime-sys` →
//! `@iree_core`). Device discovery still uses `iree-run-module --dump_devices`.

mod tools;

pub use tools::{IreeRunTools, discover_run_module};

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
    decode_module: Option<Module>,
    /// Explicit original-file/range descriptors, shared by independent
    /// sessions without staging checkpoint payloads in host memory.
    file_parameters: Option<Arc<FileParameterStorage>>,
    /// IREE HAL driver name or full device URI (`hip`, `hip://GPU-…`, …).
    /// Empty → local-task.
    device: Option<String>,
    session: Mutex<Option<NativeSession>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileParameterDescriptor {
    pub key: String,
    pub source_file_index: usize,
    pub offset: u64,
    pub length: u64,
}

/// Owns descriptor strings passed to the direct IREE file-range provider.
/// File payloads are never read by this type.
#[derive(Debug)]
pub struct FileParameterStorage {
    files: Vec<CString>,
    entries: Vec<(CString, usize, u64, u64)>,
}

impl FileParameterStorage {
    pub fn new(files: Vec<PathBuf>, entries: Vec<FileParameterDescriptor>) -> Result<Self> {
        if files.is_empty() || entries.is_empty() {
            return Err(runtime_error(
                "direct parameter files and entries must be non-empty",
            ));
        }
        let mut file_sizes = Vec::with_capacity(files.len());
        let mut c_files = Vec::with_capacity(files.len());
        for path in files {
            let metadata = std::fs::metadata(&path).map_err(|error| {
                runtime_error(format!(
                    "cannot stat direct parameter file {}: {error}",
                    path.display()
                ))
            })?;
            if !metadata.is_file() {
                return Err(runtime_error(format!(
                    "direct parameter source is not a file: {}",
                    path.display()
                )));
            }
            file_sizes.push(metadata.len());
            c_files.push(path_cstring(&path)?);
        }

        let mut keys = std::collections::BTreeMap::new();
        for entry in entries {
            if entry.length == 0 || entry.source_file_index >= c_files.len() {
                return Err(runtime_error(format!(
                    "invalid direct parameter descriptor `{}`",
                    entry.key
                )));
            }
            let end = entry.offset.checked_add(entry.length).ok_or_else(|| {
                runtime_error(format!("parameter range overflows for `{}`", entry.key))
            })?;
            if end > file_sizes[entry.source_file_index] {
                return Err(runtime_error(format!(
                    "parameter `{}` range [{}, {end}) exceeds file size {}",
                    entry.key, entry.offset, file_sizes[entry.source_file_index]
                )));
            }
            let location = (entry.source_file_index, entry.offset, entry.length);
            if let Some(existing) = keys.get(&entry.key) {
                if existing != &location {
                    return Err(runtime_error(format!(
                        "direct parameter key `{}` maps to conflicting ranges",
                        entry.key
                    )));
                }
                continue;
            }
            keys.insert(entry.key, location);
        }
        let entries = keys
            .into_iter()
            .map(|(key, (source_file_index, offset, length))| {
                CString::new(key)
                    .map(|key| (key, source_file_index, offset, length))
                    .map_err(|error| runtime_error(format!("invalid parameter key: {error}")))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            files: c_files,
            entries,
        })
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }
}

impl Context {
    pub fn create(instance: Instance, module: Module) -> Result<Self> {
        Ok(Self {
            _instance: instance,
            module,
            decode_module: None,
            file_parameters: None,
            device: None,
            session: Mutex::new(None),
        })
    }

    pub fn with_decode_module(mut self, module: Module) -> Self {
        self.decode_module = Some(module);
        self
    }

    pub fn with_file_parameters(mut self, storage: Arc<FileParameterStorage>) -> Self {
        self.file_parameters = Some(storage);
        self
    }

    /// Set HAL device/driver or full device URI (e.g. `hip`, `hip://GPU-…`).
    /// `rocm` aliases to `hip`. Bare driver names still select the default device.
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        let d = device.into();
        self.device =
            if d.is_empty() || d == "local-task" || d == "local-sync" || d == "local" || d == "cpu"
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

    fn with_session<R>(
        &self,
        f: impl FnOnce(*mut sys::dyninfer_iree_session_t) -> Result<R>,
    ) -> Result<R> {
        let mut guard = self.session.lock().map_err(|_| {
            DynInferError::IreeRuntime(IreeRuntimeError {
                message: "IREE session mutex poisoned".into(),
                status_code: None,
            })
        })?;
        if guard.is_none() {
            let vmfb = self.module_path()?;
            let vmfb_c = path_cstring(vmfb)?;
            let decode_vmfb_c = self
                .decode_module
                .as_ref()
                .map(|module| {
                    module.path.as_deref().ok_or_else(|| {
                        runtime_error("decode module has no filesystem path for native load")
                    })
                })
                .transpose()?
                .map(path_cstring)
                .transpose()?;
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
            let rc = if let Some(files) = self.file_parameters.as_ref() {
                let c_files: Vec<sys::dyninfer_iree_parameter_file_t> = files
                    .files
                    .iter()
                    .map(|path| sys::dyninfer_iree_parameter_file_t {
                        path: path.as_ptr(),
                    })
                    .collect();
                let c_params: Vec<sys::dyninfer_iree_file_param_t> = files
                    .entries
                    .iter()
                    .map(|(key, source_file_index, offset, length)| {
                        sys::dyninfer_iree_file_param_t {
                            key: key.as_ptr(),
                            source_file_index: *source_file_index,
                            offset: *offset,
                            length: *length,
                        }
                    })
                    .collect();
                unsafe {
                    if let Some(decode_vmfb) = decode_vmfb_c.as_ref() {
                        sys::dyninfer_iree_session_create_modules_with_file_params(
                            device_c.as_ref().map(|c| c.as_ptr()).unwrap_or(ptr::null()),
                            vmfb_c.as_ptr(),
                            decode_vmfb.as_ptr(),
                            c_files.as_ptr(),
                            c_files.len(),
                            c_params.as_ptr(),
                            c_params.len(),
                            &mut ptr,
                        )
                    } else {
                        sys::dyninfer_iree_session_create_with_file_params(
                            device_c.as_ref().map(|c| c.as_ptr()).unwrap_or(ptr::null()),
                            vmfb_c.as_ptr(),
                            c_files.as_ptr(),
                            c_files.len(),
                            c_params.as_ptr(),
                            c_params.len(),
                            &mut ptr,
                        )
                    }
                }
            } else {
                unsafe {
                    sys::dyninfer_iree_session_create(
                        device_c.as_ref().map(|c| c.as_ptr()).unwrap_or(ptr::null()),
                        vmfb_c.as_ptr(),
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

    pub fn configure_paged_kv(
        &self,
        layer_count: usize,
        page_size: usize,
        kv_head_count: usize,
        head_dim: usize,
        chunk_size: usize,
        vocab_size: usize,
    ) -> Result<()> {
        self.with_session(|session| {
            let rc = unsafe {
                sys::dyninfer_iree_session_configure_paged_kv(
                    session,
                    layer_count,
                    page_size,
                    kv_head_count,
                    head_dim,
                    chunk_size,
                    vocab_size,
                )
            };
            (rc == 0).then_some(()).ok_or_else(|| native_error(rc))
        })
    }

    pub fn ensure_kv_pages(&self, page_count: usize) -> Result<()> {
        self.with_session(|session| {
            let rc = unsafe { sys::dyninfer_iree_session_ensure_kv_pages(session, page_count) };
            (rc == 0).then_some(()).ok_or_else(|| native_error(rc))
        })
    }

    pub fn invoke_paged_chunk(
        &self,
        tokens: &[i64],
        last: i64,
        start_pos: i64,
    ) -> Result<Vec<f32>> {
        self.invoke_paged_chunk_ex(tokens, last, start_pos, true)
            .map(|(logits, _)| logits.expect("want_logits"))
    }

    /// Paged chunk invoke. When `want_logits` is false, only the device argmax
    /// token is copied to the host (avoids vocab-sized D2H).
    pub fn invoke_paged_chunk_ex(
        &self,
        tokens: &[i64],
        last: i64,
        start_pos: i64,
        want_logits: bool,
    ) -> Result<(Option<Vec<f32>>, i64)> {
        self.with_session(|session| {
            let mut out = ptr::null_mut();
            let mut count = 0usize;
            let mut token = -1i64;
            let rc = unsafe {
                sys::dyninfer_iree_session_invoke_paged_chunk(
                    session,
                    tokens.as_ptr(),
                    tokens.len(),
                    last,
                    start_pos,
                    if want_logits {
                        &mut out
                    } else {
                        ptr::null_mut()
                    },
                    if want_logits {
                        &mut count
                    } else {
                        ptr::null_mut()
                    },
                    &mut token,
                    if want_logits { 1 } else { 0 },
                )
            };
            if rc != 0 {
                return Err(native_error(rc));
            }
            let logits = if want_logits {
                Some(take_f32_buf(0, out, count)?)
            } else {
                None
            };
            Ok((logits, token))
        })
    }

    pub fn reset_paged_kv(&self) -> Result<()> {
        self.with_session(|session| {
            let rc = unsafe { sys::dyninfer_iree_session_reset_paged_kv(session) };
            (rc == 0).then_some(()).ok_or_else(|| native_error(rc))
        })
    }

    pub fn paged_kv_metrics(&self) -> Result<(usize, usize)> {
        self.with_session(|session| {
            Ok(unsafe {
                (
                    sys::dyninfer_iree_session_kv_page_count(session),
                    sys::dyninfer_iree_session_kv_allocated_bytes(session),
                )
            })
        })
    }

    /// Snapshot IREE HAL allocator statistics (host/device peak + live bytes).
    pub fn allocator_statistics(&self) -> Result<AllocatorStatistics> {
        self.with_session(|session| {
            let mut raw = unsafe {
                std::mem::MaybeUninit::<sys::dyninfer_iree_allocator_statistics_t>::zeroed()
                    .assume_init()
            };
            let rc = unsafe {
                sys::dyninfer_iree_session_allocator_statistics(session, &mut raw)
            };
            if rc != 0 {
                return Err(native_error(rc));
            }
            Ok(AllocatorStatistics {
                host_bytes_peak: raw.host_bytes_peak,
                host_bytes_allocated: raw.host_bytes_allocated,
                host_bytes_freed: raw.host_bytes_freed,
                device_bytes_peak: raw.device_bytes_peak,
                device_bytes_allocated: raw.device_bytes_allocated,
                device_bytes_freed: raw.device_bytes_freed,
            })
        })
    }
}

/// IREE HAL allocator statistics snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllocatorStatistics {
    pub host_bytes_peak: u64,
    pub host_bytes_allocated: u64,
    pub host_bytes_freed: u64,
    pub device_bytes_peak: u64,
    pub device_bytes_allocated: u64,
    pub device_bytes_freed: u64,
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
    #[cfg(unix)]
    let encoded = {
        use std::os::unix::ffi::OsStrExt;
        CString::new(path.as_os_str().as_bytes())
    };
    #[cfg(not(unix))]
    let encoded = CString::new(path.to_string_lossy().into_owned());
    encoded.map_err(|e| {
        DynInferError::IreeRuntime(IreeRuntimeError {
            message: format!("path contains NUL: {e}"),
            status_code: None,
        })
    })
}

fn runtime_error(message: impl Into<String>) -> DynInferError {
    DynInferError::IreeRuntime(IreeRuntimeError {
        message: message.into(),
        status_code: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_file_storage_coalesces_identical_aliases_and_rejects_conflicts() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        temp.as_file().set_len(64).unwrap();
        let path = temp.path().to_path_buf();
        let storage = FileParameterStorage::new(
            vec![path.clone()],
            vec![
                FileParameterDescriptor {
                    key: "shared".into(),
                    source_file_index: 0,
                    offset: 8,
                    length: 16,
                },
                FileParameterDescriptor {
                    key: "shared".into(),
                    source_file_index: 0,
                    offset: 8,
                    length: 16,
                },
            ],
        )
        .unwrap();
        assert_eq!(storage.file_count(), 1);
        assert_eq!(storage.entry_count(), 1);

        let conflict = FileParameterStorage::new(
            vec![path],
            vec![
                FileParameterDescriptor {
                    key: "shared".into(),
                    source_file_index: 0,
                    offset: 8,
                    length: 16,
                },
                FileParameterDescriptor {
                    key: "shared".into(),
                    source_file_index: 0,
                    offset: 16,
                    length: 16,
                },
            ],
        )
        .unwrap_err();
        assert!(conflict.to_string().contains("conflicting ranges"));
    }

    #[test]
    fn direct_file_storage_rejects_out_of_bounds_range() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        temp.as_file().set_len(16).unwrap();
        let error = FileParameterStorage::new(
            vec![temp.path().to_path_buf()],
            vec![FileParameterDescriptor {
                key: "too-large".into(),
                source_file_index: 0,
                offset: 8,
                length: 16,
            }],
        )
        .unwrap_err();
        assert!(error.to_string().contains("exceeds file size"));
    }
}
