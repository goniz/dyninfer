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

    #[repr(C)]
    pub struct dyninfer_iree_parameter_file_t {
        pub path: *const c_char,
    }

    #[repr(C)]
    pub struct dyninfer_iree_file_param_t {
        pub key: *const c_char,
        pub source_file_index: usize,
        pub offset: u64,
        pub length: u64,
    }

    // Rust 2024 requires `unsafe extern` for FFI blocks.
    unsafe extern "C" {
        pub fn dyninfer_iree_session_create(
            device_uri: *const c_char,
            vmfb_path: *const c_char,
            out_session: *mut *mut dyninfer_iree_session_t,
        ) -> c_int;
        pub fn dyninfer_iree_session_create_with_file_params(
            device_uri: *const c_char,
            vmfb_path: *const c_char,
            files: *const dyninfer_iree_parameter_file_t,
            file_count: usize,
            params: *const dyninfer_iree_file_param_t,
            param_count: usize,
            out_session: *mut *mut dyninfer_iree_session_t,
        ) -> c_int;
        pub fn dyninfer_iree_session_create_modules_with_file_params(
            device_uri: *const c_char,
            prefill_vmfb_path: *const c_char,
            decode_vmfb_path: *const c_char,
            files: *const dyninfer_iree_parameter_file_t,
            file_count: usize,
            params: *const dyninfer_iree_file_param_t,
            param_count: usize,
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
            attn_bias: *const f32,
            bias_len: usize,
            out_logits: *mut *mut f32,
            out_count: *mut usize,
        ) -> c_int;
        pub fn dyninfer_iree_session_configure_paged_kv(
            session: *mut dyninfer_iree_session_t,
            layer_count: usize,
            page_size: usize,
            kv_head_count: usize,
            head_dim: usize,
            chunk_size: usize,
            vocab_size: usize,
        ) -> c_int;
        pub fn dyninfer_iree_session_ensure_kv_pages(
            session: *mut dyninfer_iree_session_t,
            page_count: usize,
        ) -> c_int;
        pub fn dyninfer_iree_session_invoke_paged_chunk(
            session: *mut dyninfer_iree_session_t,
            tokens: *const i64,
            token_count: usize,
            last: i64,
            start_pos: i64,
            out_logits: *mut *mut f32,
            out_count: *mut usize,
            out_token: *mut i64,
            want_logits: c_int,
        ) -> c_int;
        pub fn dyninfer_iree_session_reset_paged_kv(session: *mut dyninfer_iree_session_t)
        -> c_int;
        pub fn dyninfer_iree_session_kv_page_count(
            session: *const dyninfer_iree_session_t,
        ) -> usize;
        pub fn dyninfer_iree_session_kv_allocated_bytes(
            session: *const dyninfer_iree_session_t,
        ) -> usize;
    }
}

pub use bindings::*;
