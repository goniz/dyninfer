//! Unsafe bindings to the dyninfer IREE runtime C wrapper.
//!
//! Bindings are generated with bindgen:
//! - Bazel: `rust_bindgen` → `IREE_RUNTIME_BINDINGS`
//! - Cargo: stub until a local IREE build is wired (prefer Bazel).

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(clippy::all)]
#![allow(improper_ctypes)]
#![allow(dead_code)]

#[cfg(bazel)]
pub mod bindings {
    include!(env!("IREE_RUNTIME_BINDINGS"));
}

#[cfg(not(bazel))]
pub mod bindings {
    use std::os::raw::{c_char, c_int, c_void};

    #[repr(C)]
    pub struct dyninfer_iree_session_t {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn dyninfer_iree_session_create(
            device_uri: *const c_char,
            vmfb_path: *const c_char,
            parameters_path: *const c_char,
            out_session: *mut *mut dyninfer_iree_session_t,
        ) -> c_int;
        pub fn dyninfer_iree_session_destroy(session: *mut dyninfer_iree_session_t);
        pub fn dyninfer_iree_last_error() -> *const c_char;
        pub fn dyninfer_iree_free(p: *mut c_void);
        pub fn dyninfer_iree_session_invoke_add(
            session: *mut dyninfer_iree_session_t,
            a: *const f32,
            b: *const f32,
            out_logits: *mut *mut f32,
            out_count: *mut usize,
        ) -> c_int;
        pub fn dyninfer_iree_session_invoke_prefill(
            session: *mut dyninfer_iree_session_t,
            tokens: *const i64,
            token_count: usize,
            last: i64,
            out_logits: *mut *mut f32,
            out_count: *mut usize,
        ) -> c_int;
        pub fn dyninfer_iree_session_invoke_decode(
            session: *mut dyninfer_iree_session_t,
            token: i64,
            pos: i64,
            out_logits: *mut *mut f32,
            out_count: *mut usize,
        ) -> c_int;
    }
}

pub use bindings::*;
