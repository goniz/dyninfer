//! Unsafe raw bindings to the IREE C runtime subset.
//!
//! Currently a stub until a pinned IREE revision is linked via Bazel.

#![allow(non_camel_case_types)]

use std::os::raw::c_int;

#[repr(C)]
pub struct iree_instance_t {
    _private: u8,
}

#[repr(C)]
pub struct iree_device_t {
    _private: u8,
}

#[repr(C)]
pub struct iree_vm_module_t {
    _private: u8,
}

#[repr(C)]
pub struct iree_vm_context_t {
    _private: u8,
}

pub type iree_status_t = c_int;

pub unsafe fn iree_runtime_stub_create_instance(out: *mut *mut iree_instance_t) -> iree_status_t {
    if out.is_null() {
        return 1;
    }
    let inst = Box::into_raw(Box::new(iree_instance_t { _private: 0 }));
    unsafe { *out = inst };
    0
}

pub unsafe fn iree_runtime_stub_destroy_instance(inst: *mut iree_instance_t) {
    if !inst.is_null() {
        drop(unsafe { Box::from_raw(inst) });
    }
}
