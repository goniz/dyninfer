//! SafeTensors container reader and dense convention decoder.

#![forbid(unsafe_code)]

mod config_json;
mod container;
mod convention;
mod fixture;
mod hf_names;
mod mlx;
mod sharded;

pub use container::SafeTensorsContainer;
pub use convention::DenseSafetensorsConvention;
pub use fixture::{
    fill_f32, tiny_gqa_plain_f32, tiny_gqa_rope_f32, tiny_llama_dense_f32,
    tiny_llama_mlx_affine_u4, tiny_mha_rope_f32, write_safetensors,
};
pub use hf_names::{hf_to_canonical, looks_like_hf_llama};
pub use mlx::MlxSafeTensorsConvention;
pub use sharded::ShardedSafeTensorsContainer;

use dyninfer_checkpoint::BuiltinCheckpointSupport;

/// Register SafeTensors support into a builtin registry.
pub fn register(support: &mut BuiltinCheckpointSupport) {
    support.register_container(ShardedSafeTensorsContainer);
    support.register_container(SafeTensorsContainer::default());
    support.register_convention(MlxSafeTensorsConvention);
    support.register_convention(DenseSafetensorsConvention::default());
}
