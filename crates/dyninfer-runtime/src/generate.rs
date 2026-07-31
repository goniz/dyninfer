//! Greedy text generation over a loaded causal LM session.

use crate::tokenizer_bpe::BpeTokenizer;
use crate::{CausalLanguageModel, Logits};
use dyninfer_core::{SessionConfig, TokenId};
use dyninfer_error::{DynInferError, Result};
use std::path::Path;

/// Load a HuggingFace `tokenizer.json` (or directory containing one).
pub fn load_tokenizer(path: impl AsRef<Path>) -> Result<BpeTokenizer> {
    BpeTokenizer::from_file(path)
}

pub fn argmax(logits: &[f32]) -> TokenId {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as TokenId
}

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub eos_token_id: Option<TokenId>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 48,
            eos_token_id: Some(2),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub prompt: String,
    pub text: String,
    pub token_ids: Vec<TokenId>,
}

/// Encode `prompt`, run greedy decode, return decoded text (prompt + continuation).
pub fn generate_greedy(
    model: &dyn CausalLanguageModel,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &GenerateConfig,
    session_cfg: SessionConfig,
) -> Result<GenerateOutput> {
    let mut ids: Vec<TokenId> = tokenizer
        .encode(prompt, true)?
        .into_iter()
        .map(|t| t as TokenId)
        .collect();
    if ids.is_empty() {
        return Err(DynInferError::io("prompt produced no tokens"));
    }

    let mut session = model.create_session(session_cfg)?;
    let mut logits: Logits = session.prefill(&ids)?;
    let mut generated = Vec::new();

    for _ in 0..config.max_new_tokens {
        let next = argmax(&logits.values);
        if config.eos_token_id == Some(next) {
            break;
        }
        generated.push(next);
        ids.push(next);
        logits = session.decode(next)?;
    }

    let all_ids: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&all_ids, true)?;

    Ok(GenerateOutput {
        prompt: prompt.to_string(),
        text,
        token_ids: ids,
    })
}
