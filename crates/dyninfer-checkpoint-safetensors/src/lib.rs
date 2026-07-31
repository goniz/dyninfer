//! SafeTensors container reader and dense convention decoder.

#![forbid(unsafe_code)]

mod config_json;
mod convention;
mod container;
mod fixture;
mod hf_names;
mod materialize;

pub use convention::DenseSafetensorsConvention;
pub use container::SafeTensorsContainer;
pub use fixture::{fill_f32, tiny_llama_dense_f32, write_safetensors};
pub use hf_names::{hf_to_canonical, looks_like_hf_llama};
pub use materialize::materialize_f32_safetensors;

use dyninfer_checkpoint::BuiltinCheckpointSupport;

/// Register SafeTensors support into a builtin registry.
pub fn register(support: &mut BuiltinCheckpointSupport) {
    support.register_container(SafeTensorsContainer::default());
    support.register_convention(DenseSafetensorsConvention::default());
}
