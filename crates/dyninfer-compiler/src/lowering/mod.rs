//! Shared semantic transformer lowerings owned by the compiler.

mod dense_decoder;
mod kernels;
pub(super) mod parameter;

pub use dense_decoder::{
    DenseDecoderConfig, LARGE_MAX_KV, LARGE_PREFILL_WINDOW, PAGED_KV_PAGE_SIZE,
    PAGED_PREFILL_CHUNK_SIZE, PAGED_PREFILL_CHUNK_SIZE_VULKAN, PREFILL_MAX_KV, PREFILL_WINDOW,
    PagedProgram, TINY_MAX_KV, paged_prefill_chunk_size,
    TINY_PREFILL_WINDOW, emit_dense_decoder_cfg, emit_dense_decoder_cfg_program,
};
