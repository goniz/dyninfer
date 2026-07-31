//! High-level runtime: inspect, bind, compile, load, and run sessions.

#![forbid(unsafe_code)]

mod builtins;
mod e2e;
mod generate;
mod hf_hub;
mod model;
mod reference;
mod session;
mod stories_e2e;
mod tokenizer_bpe;

pub use builtins::{default_architecture_registry, default_checkpoint_support};
pub use generate::{argmax, generate_greedy, load_tokenizer, GenerateConfig, GenerateOutput};
pub use hf_hub::{
    find_safetensors_checkpoint, hf_hub_cache_dir, hf_repo_folder_name, resolve_hf_snapshot,
    DEFAULT_HF_REVISION,
};
pub use model::{LoadedModel, ModelLoader};
pub use reference::{max_abs_err, tiny_llama_prefill_logits};
pub use session::{IreeSession, Logits};
pub use tokenizer_bpe::BpeTokenizer;

use dyninfer_core::{ModelMetadata, SessionConfig, TokenId};
use dyninfer_error::Result;

pub trait CausalLanguageModel: Send + Sync {
    fn metadata(&self) -> &ModelMetadata;
    fn create_session(&self, config: SessionConfig) -> Result<Box<dyn ModelSession>>;
}

pub trait ModelSession: Send {
    fn prefill(&mut self, tokens: &[TokenId]) -> Result<Logits>;
    fn decode(&mut self, token: TokenId) -> Result<Logits>;
    fn position(&self) -> u64;
    fn reset(&mut self) -> Result<()>;
}
