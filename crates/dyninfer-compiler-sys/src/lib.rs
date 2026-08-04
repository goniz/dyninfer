//! Raw FFI surface for `dyninfer_compiler.h`.
//!
//! In-process Rust stub mirroring the C ABI until a real `compiler/capi`
//! library is linked through Bazel. (There is no separate C++ stub translation
//! unit; this crate is the stub.)

#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

use std::os::raw::{c_int, c_void};
use std::ptr;

pub type dyninfer_compiler_t = c_void;

#[repr(C)]
pub struct dyninfer_bytes_t {
    pub data: *const u8,
    pub size: usize,
}

#[repr(C)]
pub struct dyninfer_owned_bytes_t {
    pub data: *mut u8,
    pub size: usize,
    pub release: Option<unsafe extern "C" fn(*mut u8, usize, *mut c_void)>,
    pub user_data: *mut c_void,
}

#[repr(C)]
pub struct dyninfer_compile_request_t {
    pub architecture_mlirbc: dyninfer_bytes_t,
    pub resolved_config_json: dyninfer_bytes_t,
    pub binding_plan_json: dyninfer_bytes_t,
    pub target_profile_json: dyninfer_bytes_t,
    pub shape_profile_json: dyninfer_bytes_t,
    pub compile_options_json: dyninfer_bytes_t,
}

#[repr(C)]
pub struct dyninfer_compile_result_t {
    pub vmfb: dyninfer_owned_bytes_t,
    pub metadata_json: dyninfer_owned_bytes_t,
    pub diagnostics_utf8: dyninfer_owned_bytes_t,
}

unsafe extern "C" fn stub_release(data: *mut u8, size: usize, _user: *mut c_void) {
    if !data.is_null() && size > 0 {
        drop(unsafe { Vec::from_raw_parts(data, size, size) });
    } else if !data.is_null() {
        drop(unsafe { Vec::from_raw_parts(data, 0, 0) });
    }
}

fn own_bytes(bytes: &[u8]) -> dyninfer_owned_bytes_t {
    let mut vec = bytes.to_vec();
    let data = vec.as_mut_ptr();
    let size = vec.len();
    std::mem::forget(vec);
    dyninfer_owned_bytes_t {
        data,
        size,
        release: Some(stub_release),
        user_data: ptr::null_mut(),
    }
}

/// Create a stub compiler instance.
pub unsafe fn dyninfer_compiler_create(
    _options_json: dyninfer_bytes_t,
    out_compiler: *mut *mut dyninfer_compiler_t,
) -> c_int {
    if out_compiler.is_null() {
        return 1;
    }
    let handle = Box::into_raw(Box::new(1u32)) as *mut dyninfer_compiler_t;
    unsafe { *out_compiler = handle };
    0
}

pub unsafe fn dyninfer_compiler_compile(
    compiler: *mut dyninfer_compiler_t,
    _request: *const dyninfer_compile_request_t,
    out_result: *mut dyninfer_compile_result_t,
) -> c_int {
    if compiler.is_null() || out_result.is_null() {
        return 1;
    }
    let result = dyninfer_compile_result_t {
        vmfb: own_bytes(b"DYNINFER_VMFB_STUB_v1"),
        metadata_json: own_bytes(br#"{"stub":true,"compiler":"dyninfer-stub"}"#),
        diagnostics_utf8: own_bytes(b"remark: stub compiler emitted placeholder VMFB\n"),
    };
    unsafe { *out_result = result };
    0
}

pub unsafe fn dyninfer_compiler_destroy(compiler: *mut dyninfer_compiler_t) {
    if !compiler.is_null() {
        drop(unsafe { Box::from_raw(compiler as *mut u32) });
    }
}

pub unsafe fn dyninfer_compile_result_destroy(result: *mut dyninfer_compile_result_t) {
    if result.is_null() {
        return;
    }
    let r = unsafe { &mut *result };
    if let Some(release) = r.vmfb.release {
        if !r.vmfb.data.is_null() {
            unsafe { release(r.vmfb.data, r.vmfb.size, r.vmfb.user_data) };
        }
    }
    if let Some(release) = r.metadata_json.release {
        if !r.metadata_json.data.is_null() {
            unsafe {
                release(
                    r.metadata_json.data,
                    r.metadata_json.size,
                    r.metadata_json.user_data,
                )
            };
        }
    }
    if let Some(release) = r.diagnostics_utf8.release {
        if !r.diagnostics_utf8.data.is_null() {
            unsafe {
                release(
                    r.diagnostics_utf8.data,
                    r.diagnostics_utf8.size,
                    r.diagnostics_utf8.user_data,
                )
            };
        }
    }
    unsafe {
        *result = dyninfer_compile_result_t {
            vmfb: dyninfer_owned_bytes_t {
                data: ptr::null_mut(),
                size: 0,
                release: None,
                user_data: ptr::null_mut(),
            },
            metadata_json: dyninfer_owned_bytes_t {
                data: ptr::null_mut(),
                size: 0,
                release: None,
                user_data: ptr::null_mut(),
            },
            diagnostics_utf8: dyninfer_owned_bytes_t {
                data: ptr::null_mut(),
                size: 0,
                release: None,
                user_data: ptr::null_mut(),
            },
        }
    };
}
