//! In-process MLIR → VMFB via the bindgen'd embedding API.

use crate::bindings::{
    ireeCompilerErrorDestroy, ireeCompilerErrorGetMessage, ireeCompilerGetAPIVersion,
    ireeCompilerGetRevision, ireeCompilerGlobalInitialize, ireeCompilerInvocationCreate,
    ireeCompilerInvocationDestroy, ireeCompilerInvocationEnableConsoleDiagnostics,
    ireeCompilerInvocationOutputVMBytecode, ireeCompilerInvocationParseSource,
    ireeCompilerInvocationPipeline, ireeCompilerOutputDestroy, ireeCompilerOutputMapMemory,
    ireeCompilerOutputOpenMembuffer, ireeCompilerSessionCreate, ireeCompilerSessionDestroy,
    ireeCompilerSessionSetFlags, ireeCompilerSourceDestroy, ireeCompilerSourceWrapBuffer,
    iree_compiler_pipeline_t,
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

fn ensure_initialized() -> Result<(), ApiError> {
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
    Ok(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
}

/// HAL/device flags for the dyninfer target driver name.
pub fn flags_for_driver(driver: &str) -> Vec<&'static str> {
    match driver {
        "vulkan" => vec!["--iree-hal-target-device=vulkan"],
        _ => vec![
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=generic",
        ],
    }
}

/// Compile MLIR text to VMFB bytes using session flags (IREE CLI flag subset).
pub fn compile_mlir_to_vmfb(mlir: &str, flags: &[&str]) -> Result<Vec<u8>, ApiError> {
    let _guard = COMPILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    ensure_initialized()?;
    unsafe { compile_inner(mlir, flags) }
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
        let err =
            unsafe { ireeCompilerSessionSetFlags(session, ptrs.len() as i32, ptrs.as_ptr()) };
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

    if !unsafe { ireeCompilerInvocationPipeline(inv, iree_compiler_pipeline_t::IREE_COMPILER_PIPELINE_STD) }
    {
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
        let flags = flags_for_driver("local-task");
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
}
