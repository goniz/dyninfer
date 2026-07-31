//! SafeTensors container reader and dense convention decoder.

#![forbid(unsafe_code)]

mod convention;
mod container;
mod fixture;

pub use convention::DenseSafetensorsConvention;
pub use container::SafeTensorsContainer;
pub use fixture::{tiny_llama_dense_f32, write_safetensors};

use dyninfer_checkpoint::BuiltinCheckpointSupport;

/// Register SafeTensors support into a builtin registry.
pub fn register(support: &mut BuiltinCheckpointSupport) {
    support.register_container(SafeTensorsContainer::default());
    support.register_convention(DenseSafetensorsConvention::default());
}
