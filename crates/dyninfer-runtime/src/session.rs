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
            // ABI v8 per-layer compact hist; capacity is ceil(max_kv / page_size).
            let max_pages = (self.max_seq() as usize).div_ceil(page_size).max(1);
            self.context.ensure_kv_pages(max_pages)?;
            if let Ok((_, bytes)) = self.context.paged_kv_metrics() {
                if bytes > 256 * 1024 * 1024 {
                    eprintln!(
                        "paged KV pool: {max_pages} pages × {page_size} tok ≈ {:.2} GiB",
                        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                    );
                }
            }
            let chunks: Vec<_> = tokens.chunks(chunk_size).collect();
            let mut logits = Vec::new();
            for (chunk_index, chunk) in chunks.iter().enumerate() {
                let start = chunk_index * chunk_size;
                let (window, last) = self.window_from_tokens(chunk);
                let want_logits = chunk_index + 1 == chunks.len();
                let (chunk_logits, _) = self.context.invoke_paged_chunk_ex(
                    &window,
                    last,
                    start as i64,
                    want_logits,
                )?;
                if want_logits {
                    logits = chunk_logits.expect("final prefill chunk returns logits");
                }
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
            let max_pages = (self.max_seq() as usize).div_ceil(page_size).max(1);
            self.context.ensure_kv_pages(max_pages)?;
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

    fn prefill_argmax(&mut self, tokens: &[TokenId]) -> Result<TokenId> {
        if self.paged_geometry().is_none() {
            return Ok(crate::argmax(&self.prefill(tokens)?.values));
        }
        let _span = info_span!("runtime.prefill_argmax", tokens = tokens.len()).entered();
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
        let (page_size, chunk_size) = self.paged_geometry().expect("paged");
        self.history.clear();
        self.history.extend_from_slice(tokens);
        self.context.reset_paged_kv()?;
        let max_pages = (self.max_seq() as usize).div_ceil(page_size).max(1);
        self.context.ensure_kv_pages(max_pages)?;
        let chunks: Vec<_> = tokens.chunks(chunk_size).collect();
        let mut token = -1i64;
        for (chunk_index, chunk) in chunks.iter().enumerate() {
            let start = chunk_index * chunk_size;
            let (window, last) = self.window_from_tokens(chunk);
            let (_, tok) = self.context.invoke_paged_chunk_ex(
                &window,
                last,
                start as i64,
                /*want_logits=*/ false,
            )?;
            token = tok;
        }
        if token < 0 {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!("prefill_argmax returned invalid token {token}"),
                status_code: None,
            }));
        }
        self.position = tokens.len() as u64;
        Ok(token as TokenId)
    }

    fn decode_argmax(&mut self, token: TokenId) -> Result<TokenId> {
        if self.paged_geometry().is_none() {
            return Ok(crate::argmax(&self.decode(token)?.values));
        }
        let _span = info_span!("runtime.decode_argmax", token, position = self.position).entered();
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
        if self.position >= max_seq {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!(
                    "decode position {} exceeds session/KV limit {}",
                    self.position, max_seq
                ),
                status_code: None,
            }));
        }
        let (page_size, _) = self.paged_geometry().expect("paged");
        let max_pages = (self.max_seq() as usize).div_ceil(page_size).max(1);
        self.context.ensure_kv_pages(max_pages)?;
        let decode_token = [i64::from(token)];
        let (_, next) = self.context.invoke_paged_chunk_ex(
            &decode_token,
            0,
            self.position as i64,
            /*want_logits=*/ false,
        )?;
        if next < 0 {
            return Err(DynInferError::IreeRuntime(IreeRuntimeError {
                message: format!("decode_argmax returned invalid token {next}"),
                status_code: None,
            }));
        }
        self.history.push(token);
        self.position += 1;
        Ok(next as TokenId)
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
        let capacity = self.kv.capacity_bytes() as usize;
        let geometry = self.paged_geometry();
        let paged = geometry.is_some();
        let chunk_size = geometry.map(|(_, chunk)| chunk as u32).unwrap_or(0);
        let (page_count, runtime_allocated) = if paged {
            self.context.paged_kv_metrics()?
        } else {
            // Static util.global KV is preallocated at compile-time max_kv.
            (0, capacity)
        };
        let used = if paged {
            runtime_allocated
        } else {
            self.kv.bytes_for_tokens(self.position) as usize
        };
        Ok(KvCacheMetrics {
            page_count,
            allocated_bytes: runtime_allocated,
            capacity_bytes: capacity,
            used_bytes: used,
            filled_tokens: self.position,
            layers: self.kv.layer_count,
            kv_heads: self.kv.kv_head_count,
            head_dim: self.kv.head_dimension,
            max_sequence_length: self.kv.max_sequence_length,
            key_dtype: self.kv.element_type,
            value_dtype: self.kv.element_type,
            paged,
            chunk_size,
        })
    }

    fn allocator_metrics(&self) -> Result<crate::AllocatorMetrics> {
        let s = self.context.allocator_statistics()?;
        Ok(crate::AllocatorMetrics {
            host_bytes_peak: s.host_bytes_peak,
            host_bytes_allocated: s.host_bytes_allocated,
            host_bytes_freed: s.host_bytes_freed,
            device_bytes_peak: s.device_bytes_peak,
            device_bytes_allocated: s.device_bytes_allocated,
            device_bytes_freed: s.device_bytes_freed,
        })
    }
}
