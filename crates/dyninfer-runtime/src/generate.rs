//! Greedy text generation over a loaded causal LM session.

use crate::tokenizer_bpe::BpeTokenizer;
use crate::{CausalLanguageModel, Logits};
use dyninfer_core::{SessionConfig, TokenId};
use dyninfer_error::{DynInferError, Result};
use std::path::Path;
use std::time::Instant;

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
pub struct GenerateStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

impl GenerateStats {
    pub fn prefill_tps(&self) -> f64 {
        if self.prefill_secs > 0.0 {
            self.prompt_tokens as f64 / self.prefill_secs
        } else {
            0.0
        }
    }

    pub fn decode_tps(&self) -> f64 {
        if self.decode_secs > 0.0 {
            self.generated_tokens as f64 / self.decode_secs
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub prompt: String,
    pub text: String,
    pub token_ids: Vec<TokenId>,
    pub stats: GenerateStats,
}

/// Encode `prompt`, run greedy decode, return decoded text (prompt + continuation).
pub fn generate_greedy(
    model: &dyn CausalLanguageModel,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    config: &GenerateConfig,
    session_cfg: SessionConfig,
) -> Result<GenerateOutput> {
    // ByteLevel (Qwen) tokenizers typically have no BOS; SentencePiece (Llama) does.
    let add_special = !tokenizer.is_byte_level();
    let mut ids: Vec<TokenId> = tokenizer
        .encode(prompt, add_special)?
        .into_iter()
        .map(|t| t as TokenId)
        .collect();
    if ids.is_empty() {
        return Err(DynInferError::io("prompt produced no tokens"));
    }

    let eos = config.eos_token_id.or_else(|| tokenizer.eos_id());
    let prompt_tokens = ids.len();

    let mut session = model.create_session(session_cfg)?;
    let t0 = Instant::now();
    let mut logits: Logits = session.prefill(&ids)?;
    let prefill_secs = t0.elapsed().as_secs_f64();
    let mut generated = Vec::new();

    let t1 = Instant::now();
    for _ in 0..config.max_new_tokens {
        let next = argmax(&logits.values);
        if eos == Some(next) {
            break;
        }
        generated.push(next);
        ids.push(next);
        logits = session.decode(next)?;
    }
    let decode_secs = t1.elapsed().as_secs_f64();

    let all_ids: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&all_ids, true)?;

    Ok(GenerateOutput {
        prompt: prompt.to_string(),
        text,
        token_ids: ids,
        stats: GenerateStats {
            prompt_tokens,
            generated_tokens: generated.len(),
            prefill_secs,
            decode_secs,
        },
    })
}
