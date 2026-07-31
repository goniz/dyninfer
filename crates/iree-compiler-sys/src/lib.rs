//! Bindings to IREE's ABI-stable compiler embedding API (`libIREECompiler.so`).
//!
//! Bindings are generated with bindgen:
//! - Bazel: `rust_bindgen` → `IREE_COMPILER_BINDINGS`
//! - Cargo: `build.rs` → same env var via `OUT_DIR`

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(clippy::all)]
#![allow(improper_ctypes)]
#![allow(dead_code)]

pub mod bindings {
    include!(env!("IREE_COMPILER_BINDINGS"));
}

mod api;

pub use api::{
    compile_mlir_to_vmfb, discover_rocm_bc_dir, ensure_initialized, flags_for_driver,
    flags_for_target, revision, ApiError, DEFAULT_CUDA_TARGET, DEFAULT_ROCM_TARGET,
};
