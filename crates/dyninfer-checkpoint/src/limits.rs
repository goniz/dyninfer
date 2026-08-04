//! Bounds for untrusted checkpoint inspection.

/// Hard limits applied while probing/indexing checkpoints.
#[derive(Debug, Clone)]
pub struct InspectionLimits {
    pub max_header_bytes: u64,
    pub max_tensor_count: u64,
    pub max_metadata_entries: u64,
    pub max_key_len: usize,
    pub max_shape_rank: usize,
    pub max_dim: u64,
    pub max_total_declared_bytes: u64,
}

impl Default for InspectionLimits {
    fn default() -> Self {
        Self {
            max_header_bytes: 64 * 1024 * 1024,
            max_tensor_count: 1_000_000,
            max_metadata_entries: 1_000_000,
            max_key_len: 16 * 1024,
            max_shape_rank: 16,
            max_dim: 1 << 40,
            max_total_declared_bytes: 1 << 50,
        }
    }
}

impl InspectionLimits {
    pub fn validate_key(&self, key: &str) -> dyninfer_error::Result<()> {
        if key.len() > self.max_key_len {
            return Err(dyninfer_error::DynInferError::InvalidCheckpoint(
                dyninfer_error::CheckpointValidationError {
                    message: format!(
                        "tensor key length {} exceeds limit {}",
                        key.len(),
                        self.max_key_len
                    ),
                    key: Some(key.chars().take(64).collect()),
                    detail: None,
                },
            ));
        }
        Ok(())
    }

    pub fn validate_shape(&self, shape: &[u64]) -> dyninfer_error::Result<()> {
        if shape.len() > self.max_shape_rank {
            return Err(dyninfer_error::DynInferError::InvalidCheckpoint(
                dyninfer_error::CheckpointValidationError {
                    message: format!(
                        "shape rank {} exceeds limit {}",
                        shape.len(),
                        self.max_shape_rank
                    ),
                    key: None,
                    detail: None,
                },
            ));
        }
        for &d in shape {
            if d > self.max_dim {
                return Err(dyninfer_error::DynInferError::InvalidCheckpoint(
                    dyninfer_error::CheckpointValidationError {
                        message: format!("dimension {d} exceeds limit {}", self.max_dim),
                        key: None,
                        detail: None,
                    },
                ));
            }
        }
        Ok(())
    }
}
