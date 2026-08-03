//! GGUF container reader and mixed per-tensor convention decoder.

#![forbid(unsafe_code)]

mod container;
mod convention;
mod q4;
mod types;

pub use container::GgufContainer;
pub use convention::GgufMixedConvention;
pub use q4::{
    MetaValue, Q4_0_BLOCK, Q4_0_TYPE_SIZE, fill_f32, pack_q4_0, q4_0_nbytes, tiny_llama_q4_0,
    write_gguf,
};
pub use types::GgufType;

use dyninfer_checkpoint::BuiltinCheckpointSupport;

/// Register GGUF support into a builtin registry.
pub fn register(support: &mut BuiltinCheckpointSupport) {
    support.register_container(GgufContainer::default());
    support.register_convention(GgufMixedConvention);
}
