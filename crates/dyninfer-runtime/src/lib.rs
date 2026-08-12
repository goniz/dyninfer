//! High-level runtime: inspect, bind, compile, load, and run sessions.

#![forbid(unsafe_code)]

mod builtins;
mod e2e;
mod generate;
mod hf_hub;
mod model;
mod perf;
mod perf_sample;
mod qwen_e2e;
mod reference;
mod session;
mod stories_e2e;
mod tokenizer_bpe;

pub use builtins::{
    default_architecture_registry, default_checkpoint_support, default_kernel_registry,
    default_quantization_registry,
};
pub use dyninfer_compiler::{LARGE_PREFILL_WINDOW, PREFILL_WINDOW, TINY_PREFILL_WINDOW};
pub use dyninfer_quantization::{CoverageReport, OperationCoverage};
pub use generate::{
    GenerateConfig, GenerateOutput, GenerateStats, argmax, generate_greedy, load_tokenizer,
};
pub use hf_hub::{
    DEFAULT_HF_REVISION, find_checkpoint, find_gguf_checkpoint, find_safetensors_checkpoint,
    hf_hub_cache_dir, hf_repo_folder_name, resolve_hf_snapshot,
};
pub use model::{LoadedModel, ModelLoader};
pub use perf::{PerfConfig, PerfReport, PhaseMetrics, format_perf_report, run_perf};
pub use perf_sample::parse_token_count;
pub use reference::{
    max_abs_err, tiny_llama_gguf_q4_0_prefill_logits, tiny_llama_mlx_u4_prefill_logits,
    tiny_llama_prefill_logits,
};
pub use session::{IreeSession, Logits};
pub use tokenizer_bpe::BpeTokenizer;

use dyninfer_core::{ModelMetadata, ScalarType, SessionConfig, TokenId};
use dyninfer_error::Result;

pub trait CausalLanguageModel: Send + Sync {
    fn metadata(&self) -> &ModelMetadata;
    fn create_session(&self, config: SessionConfig) -> Result<Box<dyn ModelSession>>;
}

#[derive(Debug, Clone, Copy)]
pub struct KvCacheMetrics {
    pub page_count: usize,
    /// Runtime-backed footprint (paged pages, or static capacity when preallocated).
    pub allocated_bytes: usize,
    /// Full K+V capacity at compiled max sequence length.
    pub capacity_bytes: usize,
    /// Bytes covering currently filled positions (static) or allocated pages (paged).
    pub used_bytes: usize,
    pub filled_tokens: u64,
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub max_sequence_length: u32,
    pub key_dtype: ScalarType,
    pub value_dtype: ScalarType,
    pub paged: bool,
    /// Prefill chunk length for paged ABI (0 when static).
    pub chunk_size: u32,
}

impl Default for KvCacheMetrics {
    fn default() -> Self {
        Self {
            page_count: 0,
            allocated_bytes: 0,
            capacity_bytes: 0,
            used_bytes: 0,
            filled_tokens: 0,
            layers: 0,
            kv_heads: 0,
            head_dim: 0,
            max_sequence_length: 0,
            key_dtype: ScalarType::F32,
            value_dtype: ScalarType::F32,
            paged: false,
            chunk_size: 0,
        }
    }
}

/// IREE HAL allocator statistics (zeros when unavailable).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllocatorMetrics {
    pub host_bytes_peak: u64,
    pub host_bytes_allocated: u64,
    pub host_bytes_freed: u64,
    pub device_bytes_peak: u64,
    pub device_bytes_allocated: u64,
    pub device_bytes_freed: u64,
}

impl AllocatorMetrics {
    /// Currently live host bytes (`allocated - freed`).
    pub fn host_live_bytes(&self) -> u64 {
        self.host_bytes_allocated
            .saturating_sub(self.host_bytes_freed)
    }

    /// Currently live device bytes (`allocated - freed`). Includes weights + KV + temps.
    pub fn device_live_bytes(&self) -> u64 {
        self.device_bytes_allocated
            .saturating_sub(self.device_bytes_freed)
    }
}

pub trait ModelSession: Send {
    fn prefill(&mut self, tokens: &[TokenId]) -> Result<Logits>;
    fn decode(&mut self, token: TokenId) -> Result<Logits>;
    /// Prefill and return the device/host greedy token (skips vocab D2H when possible).
    fn prefill_argmax(&mut self, tokens: &[TokenId]) -> Result<TokenId> {
        Ok(crate::argmax(&self.prefill(tokens)?.values))
    }
    /// Decode and return the device/host greedy token (skips vocab D2H when possible).
    fn decode_argmax(&mut self, token: TokenId) -> Result<TokenId> {
        Ok(crate::argmax(&self.decode(token)?.values))
    }
    fn position(&self) -> u64;
    fn reset(&mut self) -> Result<()>;
    fn kv_cache_metrics(&self) -> Result<KvCacheMetrics> {
        Ok(KvCacheMetrics::default())
    }
    fn allocator_metrics(&self) -> Result<AllocatorMetrics> {
        Ok(AllocatorMetrics::default())
    }
}
