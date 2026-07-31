//! High-level runtime: inspect, bind, compile, load, and run sessions.

#![forbid(unsafe_code)]

mod builtins;
mod e2e;
mod model;
mod session;

pub use builtins::{default_architecture_registry, default_checkpoint_support};
pub use model::{LoadedModel, ModelLoader};
pub use session::{IreeSession, Logits};

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
