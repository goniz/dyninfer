//! GGUF tensor type codes.

use dyninfer_core::{ScalarType, StorageElementType};
use dyninfer_error::{CheckpointValidationError, DynInferError, Result};

/// GGUF ggml_type values used by version 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgufType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
}

impl GgufType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            other => {
                return Err(DynInferError::InvalidCheckpoint(
                    CheckpointValidationError {
                        message: format!("unknown GGUF tensor type code {other}"),
                        key: None,
                        detail: None,
                    },
                ));
            }
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2_K => "q2_k",
            Self::Q3_K => "q3_k",
            Self::Q4_K => "q4_k",
            Self::Q5_K => "q5_k",
            Self::Q6_K => "q6_k",
            Self::Q8_K => "q8_k",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F64 => "f64",
            Self::BF16 => "bf16",
        }
    }

    pub fn is_q4_0(self) -> bool {
        matches!(self, Self::Q4_0)
    }

    pub fn is_dense_v1(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }

    pub fn block_size(self) -> Option<u64> {
        match self {
            Self::Q4_0 => Some(32),
            _ => None,
        }
    }

    /// Bytes for one quantized block, if block-quantized.
    pub fn type_size(self) -> Option<u64> {
        match self {
            Self::Q4_0 => Some(18), // 2 byte scale + 16 packed nibbles
            Self::F32 => Some(4),
            Self::F16 | Self::BF16 => Some(2),
            Self::F64 | Self::I64 => Some(8),
            Self::I32 => Some(4),
            Self::I16 => Some(2),
            Self::I8 => Some(1),
            _ => None,
        }
    }

    pub fn storage_element_type(self) -> StorageElementType {
        match self {
            Self::F32 => StorageElementType::scalar(ScalarType::F32),
            Self::F16 => StorageElementType::scalar(ScalarType::F16),
            Self::BF16 => StorageElementType::scalar(ScalarType::Bf16),
            Self::F64 => StorageElementType::scalar(ScalarType::F64),
            Self::I8 => StorageElementType::scalar(ScalarType::I8),
            Self::I16 => StorageElementType::scalar(ScalarType::I16),
            Self::I32 => StorageElementType::scalar(ScalarType::I32),
            Self::I64 => StorageElementType::scalar(ScalarType::I64),
            Self::Q4_0 => StorageElementType::Opaque {
                codec: "gguf.q4_0".into(),
            },
            other => StorageElementType::Opaque {
                codec: format!("gguf.{}", other.name()),
            },
        }
    }

    pub fn nbytes_for_shape(self, shape: &[u64]) -> Result<u64> {
        let numel = shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                DynInferError::InvalidCheckpoint(CheckpointValidationError {
                    message: "shape numel overflow".into(),
                    key: None,
                    detail: None,
                })
            })?;
        match self {
            Self::Q4_0 => {
                let block = self.block_size().unwrap();
                let type_size = self.type_size().unwrap();
                if !numel.is_multiple_of(block) {
                    return Err(DynInferError::InvalidCheckpoint(
                        CheckpointValidationError {
                            message: format!("Q4_0 numel {numel} not divisible by block {block}"),
                            key: None,
                            detail: None,
                        },
                    ));
                }
                Ok((numel / block).saturating_mul(type_size))
            }
            _ => {
                let sz = self.type_size().ok_or_else(|| {
                    DynInferError::InvalidCheckpoint(CheckpointValidationError {
                        message: format!("cannot size GGUF type {}", self.name()),
                        key: None,
                        detail: None,
                    })
                })?;
                numel.checked_mul(sz).ok_or_else(|| {
                    DynInferError::InvalidCheckpoint(CheckpointValidationError {
                        message: "tensor byte size overflow".into(),
                        key: None,
                        detail: None,
                    })
                })
            }
        }
    }
}
