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
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
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
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::MXFP4,
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
            Self::IQ2_XXS => "iq2_xxs",
            Self::IQ2_XS => "iq2_xs",
            Self::IQ3_XXS => "iq3_xxs",
            Self::IQ1_S => "iq1_s",
            Self::IQ4_NL => "iq4_nl",
            Self::IQ3_S => "iq3_s",
            Self::IQ2_S => "iq2_s",
            Self::IQ4_XS => "iq4_xs",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F64 => "f64",
            Self::IQ1_M => "iq1_m",
            Self::BF16 => "bf16",
            Self::TQ1_0 => "tq1_0",
            Self::TQ2_0 => "tq2_0",
            Self::MXFP4 => "mxfp4",
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
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL
            | Self::MXFP4 => Some(32),
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ4_XS
            | Self::IQ1_M
            | Self::TQ1_0
            | Self::TQ2_0 => Some(256),
            _ => None,
        }
    }

    /// Bytes for one quantized block, if block-quantized.
    pub fn type_size(self) -> Option<u64> {
        match self {
            Self::Q4_0 => Some(18), // 2 byte scale + 16 packed nibbles
            Self::Q4_1 => Some(20),
            Self::Q5_0 => Some(22),
            Self::Q5_1 => Some(24),
            Self::Q8_0 => Some(34),
            Self::Q8_1 => Some(36),
            Self::Q2_K => Some(84),
            Self::Q3_K => Some(110),
            Self::Q4_K => Some(144),
            Self::Q5_K => Some(176),
            Self::Q6_K => Some(210),
            Self::Q8_K => Some(292),
            Self::IQ2_XXS => Some(66),
            Self::IQ2_XS => Some(74),
            Self::IQ3_XXS => Some(98),
            Self::IQ1_S => Some(50),
            Self::IQ4_NL => Some(18),
            Self::IQ3_S => Some(110),
            Self::IQ2_S => Some(82),
            Self::IQ4_XS => Some(136),
            Self::IQ1_M => Some(56),
            Self::TQ1_0 => Some(54),
            Self::TQ2_0 => Some(66),
            Self::MXFP4 => Some(17),
            Self::F32 => Some(4),
            Self::F16 | Self::BF16 => Some(2),
            Self::F64 | Self::I64 => Some(8),
            Self::I32 => Some(4),
            Self::I16 => Some(2),
            Self::I8 => Some(1),
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
        match self.block_size() {
            Some(block) => {
                let type_size = self.type_size().ok_or_else(|| {
                    DynInferError::InvalidCheckpoint(CheckpointValidationError {
                        message: format!("cannot size GGUF type {}", self.name()),
                        key: None,
                        detail: None,
                    })
                })?;
                if !numel.is_multiple_of(block) {
                    return Err(DynInferError::InvalidCheckpoint(
                        CheckpointValidationError {
                            message: format!(
                                "GGUF {} numel {numel} not divisible by block {block}",
                                self.name()
                            ),
                            key: None,
                            detail: None,
                        },
                    ));
                }
                (numel / block).checked_mul(type_size).ok_or_else(|| {
                    DynInferError::InvalidCheckpoint(CheckpointValidationError {
                        message: "tensor byte size overflow".into(),
                        key: None,
                        detail: None,
                    })
                })
            }
            None => {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_all_supported_block_formats() {
        for (ty, block, bytes) in [
            (GgufType::Q4_0, 32, 18),
            (GgufType::Q4_1, 32, 20),
            (GgufType::Q5_0, 32, 22),
            (GgufType::Q5_1, 32, 24),
            (GgufType::Q8_0, 32, 34),
            (GgufType::Q8_1, 32, 36),
            (GgufType::Q2_K, 256, 84),
            (GgufType::Q3_K, 256, 110),
            (GgufType::Q4_K, 256, 144),
            (GgufType::Q5_K, 256, 176),
            (GgufType::Q6_K, 256, 210),
            (GgufType::Q8_K, 256, 292),
            (GgufType::IQ2_XXS, 256, 66),
            (GgufType::IQ2_XS, 256, 74),
            (GgufType::IQ3_XXS, 256, 98),
            (GgufType::IQ1_S, 256, 50),
            (GgufType::IQ4_NL, 32, 18),
            (GgufType::IQ3_S, 256, 110),
            (GgufType::IQ2_S, 256, 82),
            (GgufType::IQ4_XS, 256, 136),
            (GgufType::IQ1_M, 256, 56),
            (GgufType::TQ1_0, 256, 54),
            (GgufType::TQ2_0, 256, 66),
            (GgufType::MXFP4, 32, 17),
        ] {
            assert_eq!(ty.block_size(), Some(block));
            assert_eq!(ty.type_size(), Some(bytes));
            assert_eq!(ty.nbytes_for_shape(&[block]).unwrap(), bytes);
        }
    }

    #[test]
    fn rejects_partial_quantization_blocks() {
        let error = GgufType::Q6_K.nbytes_for_shape(&[255]).unwrap_err();
        assert!(error.to_string().contains("q6_k"));
    }
}
