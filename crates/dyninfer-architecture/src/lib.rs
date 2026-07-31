//! Architecture registry, config resolution, and package model.
//!
//! Concrete model graphs live in `dyninfer-architectures`. This crate owns the
//! plugin trait and registry only.

#![forbid(unsafe_code)]

mod builder;
mod config;
mod emit;
mod package;
mod registry;

pub use builder::{verify_mlir, ArchitectureDefinition, ModelBuilder, ModelModule, Value};
pub use config::{ConfigField, ConfigSchema, ResolvedModelConfig};
pub use emit::EmitOutput;
pub use package::ArchitecturePackage;
pub use registry::{dedupe_parameters, ArchitectureRegistry};
