use crate::ModelSession;
use dyninfer_core::{KvCacheDescriptor, ModelMetadata, SessionConfig, TokenId};
use dyninfer_error::{DynInferError, IreeRuntimeError, Result};
use iree_runtime::Context;
use std::sync::Arc;
use tracing::info_span;

#[derive(Debug, Clone)]
pub struct Logits {
    pub values: Vec<f32>,
}

/// Session backed by a persistent in-process IREE context.
///
/// Prefill seeds the compiled static window. Decode currently reuses `@prefill`
/// on a host-managed rolling window (correct logits). Mutable KV globals are
/// emitted for `@decode(token,pos)` but are not yet used here — cross-invoke
/// KV persistence still needs validation against the prefill differential.
pub struct IreeSession {
    metadata: ModelMetadata,
    config: SessionConfig,
    kv: KvCacheDescriptor,
    prefill_window: usize,
    pad_token_id: TokenId,
    position: u64,
    history: Vec<TokenId>,
    context: Arc<Context>,
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
        }
    }

    /// Left-align tokens (right-pad). Returns `(window, last_real_index)`.
    fn window_from_history(&self) -> (Vec<i64>, i64) {
        let w = self.prefill_window;
        let mut window = vec![i64::from(self.pad_token_id); w];
        let n = self.history.len().min(w);
        if n == 0 {
            return (window, 0);
        }
        let hist_start = self.history.len() - n;
        for i in 0..n {
            window[i] = i64::from(self.history[hist_start + i]);
        }
        (window, (n as i64) - 1)
    }
}

impl ModelSession for IreeSession {
    fn prefill(&mut self, tokens: &[TokenId]) -> Result<Logits> {
        let _span = info_span!("runtime.prefill", tokens = tokens.len()).entered();
        self.history.clear();
        self.history.extend_from_slice(tokens);
        let (window, last) = self.window_from_history();
        let values = self.context.invoke_prefill(&window, last)?;
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
        let _ = &self.kv;
        Ok(Logits { values })
    }

    fn decode(&mut self, token: TokenId) -> Result<Logits> {
        let _span = info_span!("runtime.decode", token, position = self.position).entered();
        self.history.push(token);
        // Rolling-window re-prefill (in-process session — no subprocess).
        let (window, last) = self.window_from_history();
        let values = self.context.invoke_prefill(&window, last)?;
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
        if self.position < self.config.max_sequence_length as u64 {
            self.position += 1;
        }
        Ok(Logits { values })
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn reset(&mut self) -> Result<()> {
        self.position = 0;
        self.history.clear();
        Ok(())
    }
}
