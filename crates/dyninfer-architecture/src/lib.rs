//! Architecture registry, config resolution, and package model.
//!
//! MLIR emission is stubbed until the compiler FFI lands; architectures still
//! expose parameter slots for binding.

#![forbid(unsafe_code)]

mod builder;
mod config;
mod package;
mod registry;

pub use builder::{ArchitectureDefinition, ModelBuilder, ModelModule};
pub use config::{ConfigField, ConfigSchema, ResolvedModelConfig};
pub use package::ArchitecturePackage;
pub use registry::ArchitectureRegistry;
