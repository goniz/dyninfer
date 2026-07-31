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

/// Session that invokes real IREE `@prefill` / `@decode` entrypoints.
pub struct IreeSession {
    metadata: ModelMetadata,
    config: SessionConfig,
    kv: KvCacheDescriptor,
    position: u64,
    context: Arc<Context>,
}

impl IreeSession {
    pub fn new(
        metadata: ModelMetadata,
        config: SessionConfig,
        kv: KvCacheDescriptor,
        context: Arc<Context>,
    ) -> Self {
        Self {
            metadata,
            config,
            kv,
            position: 0,
            context,
        }
    }
}

impl ModelSession for IreeSession {
    fn prefill(&mut self, tokens: &[TokenId]) -> Result<Logits> {
        let _span = info_span!("runtime.prefill", tokens = tokens.len()).entered();
        let mut window = [0i64; 4];
        for (i, t) in tokens.iter().take(4).enumerate() {
            window[i] = i64::from(*t);
        }
        let values = self.context.invoke_prefill(&window)?;
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
        let values = self.context.invoke_decode(i64::from(token))?;
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
        Ok(())
    }
}
