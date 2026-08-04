//! Raw MLIR C API bindings from `libIREECompiler.so`.
//!
//! Prefer [`dyninfer_mlir`] for the safe builder surface.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(clippy::all)]
#![allow(improper_ctypes)]
#![allow(dead_code)]

pub mod bindings {
    include!(env!("DYNINFER_MLIR_BINDINGS"));
}

pub use iree_compiler_sys::ensure_initialized;
