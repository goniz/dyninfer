use crate::{KvCacheMetrics, ModelSession};
use dyninfer_core::{KvCacheDescriptor, KvCacheStorage, ModelMetadata, SessionConfig, TokenId};
use dyninfer_error::{DynInferError, IreeRuntimeError, Result};
use iree_runtime::Context;
use std::sync::Arc;
use tracing::info_span;

#[derive(Debug, Clone)]
pub struct Logits {
    pub values: Vec<f32>,
}

/// Session that prefills once, then decodes with `@decode(token, pos)` against
/// mutable KV globals held in the persistent in-process IREE context.
pub struct IreeSession {
    metadata: ModelMetadata,
    config: SessionConfig,
    kv: KvCacheDescriptor,
    prefill_window: usize,
    pad_token_id: TokenId,
    position: u64,
    history: Vec<TokenId>,
    context: Arc<Context>,
    /// Reused causal attention bias across decode steps.
    attn_bias: Vec<f32>,
}

impl IreeSession {
    pub fn new(
        metadata: ModelMetadata,
        config: SessionConfig,
        kv: KvCacheDescriptor,
        prefill_window: u32,
        context: Arc<Context>,
    ) -> Self {
        let pad_token_id = metadata
            .extra
            .get("pad_token_id")
            .or_else(|| metadata.extra.get("bos_token_id"))
            .and_then(|v| v.as_u64())
            .map(|v| v as TokenId)
            .unwrap_or(0);
        Self {
            metadata,
            config,
            kv,
            prefill_window: prefill_window.max(1) as usize,
            pad_token_id,
            position: 0,
            history: Vec::new(),
            context,
            attn_bias: Vec::new(),
        }
    }

    /// Effective sequence cap: session config ∩ compiled KV capacity.
    fn max_seq(&self) -> u64 {
        let cfg = self.config.max_sequence_length.max(1) as u64;
        let kv = self.kv.max_sequence_length.max(1) as u64;
        cfg.min(kv)
    }

    /// Left-align tokens (right-pad). Returns `(window, last_real_index)`.
    fn window_from_tokens(&self, tokens: &[TokenId]) -> (Vec<i64>, i64) {
        let w = self.prefill_window;
        let mut window = vec![i64::from(self.pad_token_id); w];
        let n = tokens.len().min(w);
        if n == 0 {
            return (window, 0);
        }
        for i in 0..n {
            window[i] = i64::from(tokens[i]);
        }
        (window, (n as i64) - 1)
    }

    fn paged_geometry(&self) -> Option<(usize, usize)> {
        match self.kv.storage {
            KvCacheStorage::StaticGlobals => None,
            KvCacheStorage::Paged {
                page_size,
                chunk_size,
            } => Some((page_size as usize, chunk_size as usize)),
        }
    }
}

impl ModelSession for IreeSession {
    fn prefill(&mut self, tokens: &[TokenId]) -> Result<Logits> {
        let _span = info_span!("runtime.prefill", tokens = tokens.len()).entered();
        if self.config.batch_size != 1 {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!(
                    "batch_size {} is not supported (only batch_size=1)",
                    self.config.batch_size
                ),
                status_code: None,
            }));
        }
        if tokens.is_empty() {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: "prefill requires a non-empty token list".into(),
                status_code: None,
            }));
        }
        if self.paged_geometry().is_none() && tokens.len() > self.prefill_window {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!(
                    "prefill length {} exceeds compiled window {}",
                    tokens.len(),
                    self.prefill_window
                ),
                status_code: None,
            }));
        }
        let max_seq = self.max_seq() as usize;
        if tokens.len() > max_seq {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!(
                    "prefill length {} exceeds session/KV limit {}",
                    tokens.len(),
                    max_seq
                ),
                status_code: None,
            }));
        }
        self.history.clear();
        self.history.extend_from_slice(tokens);
        let values = if let Some((page_size, chunk_size)) = self.paged_geometry() {
            self.context.reset_paged_kv()?;
            let mut logits = Vec::new();
            for (chunk_index, chunk) in tokens.chunks(chunk_size).enumerate() {
                let start = chunk_index * chunk_size;
                let required_pages = (start + chunk.len()).div_ceil(page_size);
                self.context.ensure_kv_pages(required_pages)?;
                let (window, last) = self.window_from_tokens(chunk);
                logits = self
                    .context
                    .invoke_paged_chunk(&window, last, start as i64)?;
            }
            logits
        } else {
            let (window, last) = self.window_from_tokens(tokens);
            self.context.invoke_prefill(&window, last)?
        };
        if values.len() != self.metadata.vocabulary_size as usize {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!(
                    "prefill returned {} logits, expected vocab {}",
                    values.len(),
                    self.metadata.vocabulary_size
                ),
                status_code: None,
            }));
        }
        self.position = tokens.len() as u64;
        Ok(Logits { values })
    }

    fn decode(&mut self, token: TokenId) -> Result<Logits> {
        let _span = info_span!("runtime.decode", token, position = self.position).entered();
        if self.config.batch_size != 1 {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!(
                    "batch_size {} is not supported (only batch_size=1)",
                    self.config.batch_size
                ),
                status_code: None,
            }));
        }
        let max_seq = self.max_seq();
        // Position is the write index into KV; once it reaches the session or
        // compiled limit, refuse rather than re-decoding at a stale position.
        if self.position >= max_seq {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!(
                    "decode position {} exceeds session/KV limit {}",
                    self.position, max_seq
                ),
                status_code: None,
            }));
        }
        let values = if let Some((page_size, _)) = self.paged_geometry() {
            self.context
                .ensure_kv_pages((self.position as usize + 1).div_ceil(page_size))?;
            let decode_token = [i64::from(token)];
            self.context
                .invoke_paged_chunk(&decode_token, 0, self.position as i64)?
        } else {
            let max_kv = self.kv.max_sequence_length.max(1) as usize;
            iree_runtime::fill_causal_attn_bias(&mut self.attn_bias, self.position as i64, max_kv);
            self.context
                .invoke_decode(i64::from(token), self.position as i64, &self.attn_bias)?
        };
        if values.len() != self.metadata.vocabulary_size as usize {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!(
                    "decode returned {} logits, expected vocab {}",
                    values.len(),
                    self.metadata.vocabulary_size
                ),
                status_code: None,
            }));
        }
        self.history.push(token);
        self.position += 1;
        Ok(Logits { values })
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn reset(&mut self) -> Result<()> {
        if self.paged_geometry().is_some() {
            self.context.reset_paged_kv()?;
        }
        self.position = 0;
        self.history.clear();
        Ok(())
    }

    fn kv_cache_metrics(&self) -> Result<KvCacheMetrics> {
        let (page_count, allocated_bytes) = if self.paged_geometry().is_some() {
            self.context.paged_kv_metrics()?
        } else {
            (0, 0)
        };
        Ok(KvCacheMetrics {
            page_count,
            allocated_bytes,
        })
    }
}
