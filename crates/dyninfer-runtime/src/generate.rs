//! Greedy text generation over a loaded causal LM session.

use crate::tokenizer_bpe::BpeTokenizer;
use crate::{CausalLanguageModel, Logits};
use dyninfer_core::{SessionConfig, TokenId};
use dyninfer_error::{DynInferError, Result};
use std::collections::HashMap;
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

/// HuggingFace-style repetition penalty on tokens already present in `seen`.
pub fn apply_repetition_penalty(logits: &mut [f32], seen: &HashMap<TokenId, u32>, penalty: f32) {
    if (penalty - 1.0).abs() < f32::EPSILON || seen.is_empty() {
        return;
    }
    for (&token, _) in seen {
        let i = token as usize;
        if i >= logits.len() {
            continue;
        }
        let score = logits[i];
        logits[i] = if score < 0.0 {
            score * penalty
        } else {
            score / penalty
        };
    }
}

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub eos_token_id: Option<TokenId>,
    /// Extra stop ids (e.g. `<|endoftext|>` alongside `<|im_end|>`).
    pub stop_token_ids: Vec<TokenId>,
    /// HF-style repetition penalty (`1.0` = off). Prefer `> 1` for long greedy runs.
    pub repetition_penalty: f32,
    /// Apply the model's HF chat template to bare prompts.
    pub apply_chat_template: bool,
    /// Qwen3-style thinking switch passed into the chat template.
    pub enable_thinking: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 48,
            // Prefer tokenizer / model metadata; never hardcode Llama's 2.
            eos_token_id: None,
            stop_token_ids: Vec::new(),
            repetition_penalty: 1.1,
            apply_chat_template: true,
            enable_thinking: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub kv_page_count: usize,
    pub kv_allocated_bytes: usize,
    pub kv_capacity_bytes: usize,
    pub kv_used_bytes: usize,
    pub kv_key_dtype: String,
    pub kv_value_dtype: String,
    pub kv_paged: bool,
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
    let model_prompt = if config.apply_chat_template {
        tokenizer
            .apply_chat_template(prompt, config.enable_thinking)?
            .unwrap_or_else(|| prompt.to_string())
    } else {
        prompt.to_string()
    };

    // ByteLevel (Qwen) tokenizers typically have no BOS; SentencePiece (Llama) does.
    // Chat templates already emit specials — never double-add BOS.
    let add_special = !tokenizer.is_byte_level() && !config.apply_chat_template;
    let mut ids: Vec<TokenId> = tokenizer
        .encode(&model_prompt, add_special)?
        .into_iter()
        .map(|t| t as TokenId)
        .collect();
    if ids.is_empty() {
        return Err(DynInferError::io("prompt produced no tokens"));
    }

    let mut stop = config.stop_token_ids.clone();
    if let Some(eos) = config.eos_token_id.or_else(|| tokenizer.eos_id()) {
        if !stop.contains(&eos) {
            stop.push(eos);
        }
    }
    // Qwen generation_config also stops on <|endoftext|>.
    if let Some(eof) = tokenizer.token_id("<|endoftext|>") {
        let eof = eof as TokenId;
        if !stop.contains(&eof) {
            stop.push(eof);
        }
    }

    let prompt_tokens = ids.len();
    let mut seen: HashMap<TokenId, u32> = HashMap::new();
    for &id in &ids {
        *seen.entry(id).or_insert(0) += 1;
    }

    let mut session = model.create_session(session_cfg)?;
    let skip_logits_d2h = (config.repetition_penalty - 1.0).abs() < f32::EPSILON;
    let t0 = Instant::now();
    let mut generated = Vec::new();
    let prefill_secs;
    let decode_secs;
    if skip_logits_d2h {
        let mut next = session.prefill_argmax(&ids)?;
        prefill_secs = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for _ in 0..config.max_new_tokens {
            if stop.contains(&next) {
                break;
            }
            generated.push(next);
            ids.push(next);
            *seen.entry(next).or_insert(0) += 1;
            next = session.decode_argmax(next)?;
        }
        decode_secs = t1.elapsed().as_secs_f64();
    } else {
        let mut logits: Logits = session.prefill(&ids)?;
        prefill_secs = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for _ in 0..config.max_new_tokens {
            apply_repetition_penalty(
                &mut logits.values,
                &seen,
                config.repetition_penalty,
            );
            let next = argmax(&logits.values);
            if stop.contains(&next) {
                break;
            }
            generated.push(next);
            ids.push(next);
            *seen.entry(next).or_insert(0) += 1;
            logits = session.decode(next)?;
        }
        decode_secs = t1.elapsed().as_secs_f64();
    }
    let cache = session.kv_cache_metrics()?;

    // Decode only the assistant continuation so ChatML markup stays out of `text`.
    let continuation = tokenizer.decode(
        &generated.iter().map(|&t| t as u32).collect::<Vec<_>>(),
        true,
    )?;
    let text = format!("{prompt}{continuation}");

    Ok(GenerateOutput {
        prompt: prompt.to_string(),
        text,
        token_ids: ids,
        stats: GenerateStats {
            prompt_tokens,
            generated_tokens: generated.len(),
            prefill_secs,
            decode_secs,
            kv_page_count: cache.page_count,
            kv_allocated_bytes: cache.allocated_bytes,
            kv_capacity_bytes: cache.capacity_bytes,
            kv_used_bytes: cache.used_bytes,
            kv_key_dtype: cache.key_dtype.to_string(),
            kv_value_dtype: cache.value_dtype.to_string(),
            kv_paged: cache.paged,
        },
    })
}
