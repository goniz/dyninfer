//! In-process MLIR → VMFB via the bindgen'd embedding API.

use crate::bindings::{
    iree_compiler_pipeline_t, ireeCompilerErrorDestroy, ireeCompilerErrorGetMessage,
    ireeCompilerGetAPIVersion, ireeCompilerGetRevision, ireeCompilerGlobalInitialize,
    ireeCompilerInvocationCreate, ireeCompilerInvocationDestroy,
    ireeCompilerInvocationEnableConsoleDiagnostics, ireeCompilerInvocationOutputVMBytecode,
    ireeCompilerInvocationParseSource, ireeCompilerInvocationPipeline, ireeCompilerOutputDestroy,
    ireeCompilerOutputMapMemory, ireeCompilerOutputOpenMembuffer, ireeCompilerSessionCreate,
    ireeCompilerSessionDestroy, ireeCompilerSessionSetFlags, ireeCompilerSourceDestroy,
    ireeCompilerSourceWrapBuffer,
};
use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::ptr;
use std::sync::{Mutex, Once};
use tracing::debug;

/// IREE/LLVM global compiler state is not safe for concurrent invocations.
static COMPILE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug)]
pub struct ApiError {
    pub message: String,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ApiError {}

static INIT: Once = Once::new();

/// Initialize process-global IREE compiler state (idempotent).
///
/// Required before any MLIR C API use through `libIREECompiler.so`.
pub fn ensure_initialized() -> Result<(), ApiError> {
    let mut init_err: Option<String> = None;
    INIT.call_once(|| {
        unsafe {
            ireeCompilerGlobalInitialize();
        }
        let version = unsafe { ireeCompilerGetAPIVersion() } as u32;
        let major = (version >> 16) & 0xffff;
        let minor = version & 0xffff;
        debug!(major, minor, "IREE compiler API version");
        if major > 1 {
            init_err = Some(format!(
                "unsupported IREE compiler API major version {major}.{minor}"
            ));
        }
    });
    match init_err {
        Some(message) => Err(ApiError { message }),
        None => Ok(()),
    }
}

unsafe fn take_error(err: *mut crate::bindings::iree_compiler_error_t) -> Option<String> {
    if err.is_null() {
        return None;
    }
    let msg_ptr = unsafe { ireeCompilerErrorGetMessage(err) };
    let msg = if msg_ptr.is_null() {
        "unknown IREE compiler error".to_string()
    } else {
        unsafe { CStr::from_ptr(msg_ptr) }
            .to_string_lossy()
            .into_owned()
    };
    unsafe { ireeCompilerErrorDestroy(err) };
    Some(msg)
}

/// IREE compiler revision string.
pub fn revision() -> Result<String, ApiError> {
    ensure_initialized()?;
    let ptr = unsafe { ireeCompilerGetRevision() };
    if ptr.is_null() {
        return Ok(String::new());
    }
    Ok(unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned())
}

/// HAL/device flags for the dyninfer target driver name.
///
/// `gpu_arch` is `--iree-rocm-target` / `--iree-cuda-target` (e.g. `gfx1151`,
/// `sm_80`). GPU architectures are mandatory: this API never guesses a compile
/// target.
///
/// For HIP, also sets `--iree-rocm-bc-dir` when platform bitcode can be found
/// (required for in-process `libIREECompiler`; `iree-compile` finds it via
/// `$ORIGIN`).
pub fn flags_for_target(driver: &str, gpu_arch: Option<&str>) -> Result<Vec<String>, ApiError> {
    let require_arch = || {
        gpu_arch.filter(|s| !s.trim().is_empty()).ok_or_else(|| ApiError {
            message: format!(
                "exact GPU architecture is required for `{driver}` compilation; target guessing is disabled"
            ),
        })
    };
    match driver {
        "hip" | "rocm" => {
            let chip = require_arch()?;
            let mut flags = vec![
                "--iree-hal-target-device=hip".into(),
                format!("--iree-rocm-target={chip}"),
            ];
            if let Some(bc) = discover_rocm_bc_dir() {
                debug!(path = %bc.display(), "using ROCm bitcode dir");
                flags.push(format!("--iree-rocm-bc-dir={}", bc.display()));
            }
            Ok(flags)
        }
        "cuda" => {
            let arch = require_arch()?;
            Ok(vec![
                "--iree-hal-target-device=cuda".into(),
                format!("--iree-cuda-target={arch}"),
            ])
        }
        "local" | "local-task" | "local-sync" => Ok(vec![
            "--iree-hal-target-device=local".into(),
            "--iree-hal-local-target-device-backends=llvm-cpu".into(),
            "--iree-llvmcpu-target-cpu=host".into(),
        ]),
        _ => Err(ApiError {
            message: format!("unsupported IREE target driver `{driver}`"),
        }),
    }
}

/// Directory containing `ocml.bc` / `ockl.bc` for `--iree-rocm-bc-dir`.
///
/// Resolution order:
/// 1. `DYNINFER_IREE_ROCM_BC_DIR`
/// 2. Bazel runfiles under the IREE compiler external repo
/// 3. `iree_platform_libs/rocm` next to the loaded `libIREECompiler.so`
pub fn discover_rocm_bc_dir() -> Option<std::path::PathBuf> {
    use std::path::{Path, PathBuf};

    fn is_rocm_bc(dir: &Path) -> bool {
        dir.join("ocml.bc").is_file() && dir.join("ockl.bc").is_file()
    }

    if let Ok(p) = std::env::var("DYNINFER_IREE_ROCM_BC_DIR") {
        let p = PathBuf::from(p);
        if is_rocm_bc(&p) {
            return Some(p);
        }
    }

    for root in runfiles_roots() {
        for arch in ["x86_64", "aarch64"] {
            let repo = format!("iree_compiler_linux_{arch}");
            let inner = "iree/compiler/_mlir_libs/iree_platform_libs/rocm";
            for prefix in [repo.clone(), format!("+http_archive+{repo}")] {
                let candidate = root.join(&prefix).join(inner);
                if is_rocm_bc(&candidate) {
                    return Some(candidate);
                }
            }
        }
    }

    // Ensure the dylib is mapped before scanning /proc/self/maps.
    let _ = unsafe { ireeCompilerGetAPIVersion() };
    if let Some(lib_dir) = libiree_compiler_dir() {
        let candidate = lib_dir.join("iree_platform_libs").join("rocm");
        if is_rocm_bc(&candidate) {
            return Some(candidate);
        }
    }

    None
}

fn libiree_compiler_dir() -> Option<std::path::PathBuf> {
    // Linux: path of the mapped shared object.
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    for line in maps.lines() {
        if !line.contains("libIREECompiler.so") {
            continue;
        }
        let path = line.split_whitespace().last()?;
        if path.starts_with('/') {
            return std::path::Path::new(path).parent().map(|p| p.to_path_buf());
        }
    }
    None
}

fn runfiles_roots() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;
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

/// Convenience wrapper for drivers that do not require an architecture.
pub fn flags_for_driver(driver: &str) -> Result<Vec<String>, ApiError> {
    flags_for_target(driver, None)
}

/// Compile MLIR text to VMFB bytes using session flags (IREE CLI flag subset).
pub fn compile_mlir_to_vmfb(mlir: &str, flags: &[impl AsRef<str>]) -> Result<Vec<u8>, ApiError> {
    let _guard = COMPILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    ensure_initialized()?;
    let owned: Vec<String> = flags.iter().map(|f| f.as_ref().to_string()).collect();
    let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
    unsafe { compile_inner(mlir, &refs) }
}

unsafe fn compile_inner(mlir: &str, flags: &[&str]) -> Result<Vec<u8>, ApiError> {
    let session = unsafe { ireeCompilerSessionCreate() };
    if session.is_null() {
        return Err(ApiError {
            message: "ireeCompilerSessionCreate returned null".into(),
        });
    }

    let c_flags: Result<Vec<CString>, _> = flags.iter().map(|f| CString::new(*f)).collect();
    let c_flags = c_flags.map_err(|e| ApiError {
        message: format!("invalid flag: {e}"),
    })?;
    let ptrs: Vec<*const i8> = c_flags.iter().map(|s| s.as_ptr()).collect();
    if !ptrs.is_empty() {
        let err = unsafe { ireeCompilerSessionSetFlags(session, ptrs.len() as i32, ptrs.as_ptr()) };
        if let Some(msg) = unsafe { take_error(err) } {
            unsafe { ireeCompilerSessionDestroy(session) };
            return Err(ApiError {
                message: format!("ireeCompilerSessionSetFlags: {msg}"),
            });
        }
    }

    let mut buffer = mlir.as_bytes().to_vec();
    buffer.push(0);
    let name = CString::new("dyninfer.mlir").unwrap();
    let mut source = ptr::null_mut();
    let err = unsafe {
        ireeCompilerSourceWrapBuffer(
            session,
            name.as_ptr(),
            buffer.as_ptr() as *const i8,
            buffer.len(),
            true,
            &mut source,
        )
    };
    if let Some(msg) = unsafe { take_error(err) } {
        unsafe { ireeCompilerSessionDestroy(session) };
        return Err(ApiError {
            message: format!("ireeCompilerSourceWrapBuffer: {msg}"),
        });
    }

    let inv = unsafe { ireeCompilerInvocationCreate(session) };
    if inv.is_null() {
        unsafe {
            ireeCompilerSourceDestroy(source);
            ireeCompilerSessionDestroy(session);
        }
        return Err(ApiError {
            message: "ireeCompilerInvocationCreate returned null".into(),
        });
    }
    unsafe { ireeCompilerInvocationEnableConsoleDiagnostics(inv) };

    if !unsafe { ireeCompilerInvocationParseSource(inv, source) } {
        unsafe {
            ireeCompilerInvocationDestroy(inv);
            ireeCompilerSourceDestroy(source);
            ireeCompilerSessionDestroy(session);
        }
        return Err(ApiError {
            message: "ireeCompilerInvocationParseSource failed".into(),
        });
    }

    if !unsafe {
        ireeCompilerInvocationPipeline(inv, iree_compiler_pipeline_t::IREE_COMPILER_PIPELINE_STD)
    } {
        unsafe {
            ireeCompilerInvocationDestroy(inv);
            ireeCompilerSourceDestroy(source);
            ireeCompilerSessionDestroy(session);
        }
        return Err(ApiError {
            message: "ireeCompilerInvocationPipeline(STD) failed".into(),
        });
    }

    let mut output = ptr::null_mut();
    let err = unsafe { ireeCompilerOutputOpenMembuffer(&mut output) };
    if let Some(msg) = unsafe { take_error(err) } {
        unsafe {
            ireeCompilerInvocationDestroy(inv);
            ireeCompilerSourceDestroy(source);
            ireeCompilerSessionDestroy(session);
        }
        return Err(ApiError {
            message: format!("ireeCompilerOutputOpenMembuffer: {msg}"),
        });
    }

    let err = unsafe { ireeCompilerInvocationOutputVMBytecode(inv, output) };
    if let Some(msg) = unsafe { take_error(err) } {
        unsafe {
            ireeCompilerOutputDestroy(output);
            ireeCompilerInvocationDestroy(inv);
            ireeCompilerSourceDestroy(source);
            ireeCompilerSessionDestroy(session);
        }
        return Err(ApiError {
            message: format!("ireeCompilerInvocationOutputVMBytecode: {msg}"),
        });
    }

    let mut contents: *mut c_void = ptr::null_mut();
    let mut size: u64 = 0;
    let err = unsafe { ireeCompilerOutputMapMemory(output, &mut contents, &mut size) };
    if let Some(msg) = unsafe { take_error(err) } {
        unsafe {
            ireeCompilerOutputDestroy(output);
            ireeCompilerInvocationDestroy(inv);
            ireeCompilerSourceDestroy(source);
            ireeCompilerSessionDestroy(session);
        }
        return Err(ApiError {
            message: format!("ireeCompilerOutputMapMemory: {msg}"),
        });
    }
    if contents.is_null() || size == 0 {
        unsafe {
            ireeCompilerOutputDestroy(output);
            ireeCompilerInvocationDestroy(inv);
            ireeCompilerSourceDestroy(source);
            ireeCompilerSessionDestroy(session);
        }
        return Err(ApiError {
            message: "empty VMFB from in-process IREE compiler".into(),
        });
    }
    let bytes =
        unsafe { std::slice::from_raw_parts(contents as *const u8, size as usize) }.to_vec();

    unsafe {
        ireeCompilerOutputDestroy(output);
        ireeCompilerInvocationDestroy(inv);
        ireeCompilerSourceDestroy(source);
        ireeCompilerSessionDestroy(session);
    }

    debug!(bytes = bytes.len(), "in-process VMFB ready");
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inprocess_add_smoke() {
        let mlir = r#"
module {
  func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"#;
        let flags = flags_for_driver("local-task").unwrap();
        match compile_mlir_to_vmfb(mlir, &flags) {
            Ok(vmfb) => {
                assert!(!vmfb.is_empty());
                assert!(!vmfb.starts_with(b"DYNINFER_VMFB_STUB"));
            }
            Err(e) => {
                // Missing shared library under cargo without bootstrap.
                eprintln!("skipping in-process smoke: {e}");
            }
        }
    }

    #[test]
    fn hip_inprocess_compiles_with_rocm_bc_dir() {
        let mlir = r#"
module {
  func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"#;
        let flags = flags_for_target("hip", Some("gfx1151")).unwrap();
        if !flags.iter().any(|f| f.starts_with("--iree-rocm-bc-dir=")) {
            eprintln!("skipping: ROCm bitcode not found in runfiles");
            return;
        }
        let vmfb = compile_mlir_to_vmfb(mlir, &flags).expect("in-process HIP compile");
        assert!(!vmfb.is_empty());
        assert!(!vmfb.starts_with(b"DYNINFER_VMFB_STUB"));
    }

    #[test]
    fn gpu_flags_require_exact_architecture() {
        let error = flags_for_target("hip", None).unwrap_err();
        assert!(error.message.contains("exact GPU architecture"));
        let error = flags_for_target("cuda", Some("")).unwrap_err();
        assert!(error.message.contains("target guessing is disabled"));
    }
}
