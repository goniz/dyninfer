//! GGUF container reader and Q4_0 / dense convention decoders.

#![forbid(unsafe_code)]

mod container;
mod convention;
mod types;

pub use container::GgufContainer;
pub use convention::{GgufDenseConvention, GgufQ40Convention};
pub use types::GgufType;

use dyninfer_checkpoint::BuiltinCheckpointSupport;

/// Register GGUF support into a builtin registry.
pub fn register(support: &mut BuiltinCheckpointSupport) {
    support.register_container(GgufContainer::default());
    support.register_convention(GgufQ40Convention::default());
    support.register_convention(GgufDenseConvention::default());
}
