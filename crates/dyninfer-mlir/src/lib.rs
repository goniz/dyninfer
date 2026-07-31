//! In-memory MLIR builder backed by the MLIR C API in `libIREECompiler.so`.
//!
//! Spec §8.3.1: architecture Rust code builds an in-memory module through a
//! narrow C API wrapper (melior-style), verifies it, then lowers. This crate is
//! that wrapper — dialects known to IREE (including `util` / `stream`) are
//! registered via `ireeCompilerRegisterDialects`.

mod builder;
mod context;
mod module;
mod string_ref;

pub use builder::{ModuleBuilder, VerifiedModule};
pub use context::Context;
pub use module::Module;

use dyninfer_error::{CompilationError, DynInferError, Result};

/// Parse `source`, verify the module, and return canonical printed text.
pub fn parse_verify_print(source: &str) -> Result<String> {
    let mut builder = ModuleBuilder::new()?;
    builder.parse_source(source)?;
    builder.verify()?;
    Ok(builder.print())
}

pub(crate) fn mlir_err(message: impl Into<String>, pass: &'static str) -> DynInferError {
    DynInferError::Compilation(CompilationError {
        message: message.into(),
        pass: Some(pass.into()),
        diagnostics: vec![],
    })
}
