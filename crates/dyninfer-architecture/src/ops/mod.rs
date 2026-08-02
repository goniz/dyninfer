//! Shared MLIR op / decoder emitters used by concrete model files.

mod dense_decoder;
mod kernels;

pub use dense_decoder::{
    emit_dense_decoder_cfg, DenseDecoderConfig, COMPUTE_DTYPE, LARGE_PREFILL_WINDOW,
    PREFILL_WINDOW, TINY_PREFILL_WINDOW,
};
