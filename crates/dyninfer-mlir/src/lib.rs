//! In-memory MLIR builder backed by the MLIR C API in `libIREECompiler.so`.
//!
//! Spec §8.3.1: architecture Rust code builds an in-memory module through a
//! narrow C API wrapper (melior-style), verifies it, then lowers. This crate is
//! that wrapper — dialects known to IREE (including `util` / `stream`) are
//! registered via `ireeCompilerRegisterDialects`.
//!
//! # Melior vs a tensor DSL
//!
//! [melior](https://github.com/mlir-rs/melior) (and this crate) are **MLIR C API
//! bindings**: Context / Module / OperationBuilder / dialect helpers. They do
//! **not** compile Rust (or Zig) into graphs the way [zml](https://github.com/zml/zml)
//! does (Zig `Tensor` API → StableHLO → PJRT). Graph volume for Llama/Qwen lives
//! in `dyninfer-architecture` emitters; shrinking that means higher-level
//! `ModelBuilder` / kernel helpers, not switching binding crates.

mod attribute;
mod builder;
mod context;
pub mod dialect;
mod func_builder;
mod location;
mod module;
mod operation;
mod string_ref;
mod r#type;
mod value;

pub use attribute::Attribute;
pub use builder::{ModuleBuilder, VerifiedModule};
pub use context::Context;
pub use dialect::{Arith, Func, Linalg, Tensor, Util};
pub use func_builder::FuncBuilder;
pub use location::Location;
pub use module::Module;
pub use operation::{Operation, OperationBuilder};
pub use r#type::Type;
pub use value::Value;

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
