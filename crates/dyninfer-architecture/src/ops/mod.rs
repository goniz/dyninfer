//! Shared MLIR op / decoder emitters used by concrete model files.

mod dense_decoder;
mod kernels;

pub use dense_decoder::{
    COMPUTE_DTYPE, DenseDecoderConfig, LARGE_PREFILL_WINDOW, PREFILL_WINDOW, TINY_PREFILL_WINDOW,
    emit_dense_decoder_cfg,
};
