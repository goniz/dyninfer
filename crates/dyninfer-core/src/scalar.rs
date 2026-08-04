//! Scalar and storage element type models.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Logical or physical scalar element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScalarType {
    F32,
    F16,
    Bf16,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl ScalarType {
    pub fn size_bytes(self) -> Option<u32> {
        Some(match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::Bf16 | Self::I16 | Self::U16 => 2,
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::I8 | Self::U8 | Self::Bool => 1,
        })
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "float32" => Some(Self::F32),
            "f16" | "float16" => Some(Self::F16),
            "bf16" | "bfloat16" => Some(Self::Bf16),
            "f64" | "float64" => Some(Self::F64),
            "i8" | "int8" => Some(Self::I8),
            "i16" | "int16" => Some(Self::I16),
            "i32" | "int32" => Some(Self::I32),
            "i64" | "int64" => Some(Self::I64),
            "u8" | "uint8" => Some(Self::U8),
            "u16" | "uint16" => Some(Self::U16),
            "u32" | "uint32" => Some(Self::U32),
            "u64" | "uint64" => Some(Self::U64),
            "bool" | "boolean" => Some(Self::Bool),
            _ => None,
        }
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::Bool => "bool",
        };
        f.write_str(s)
    }
}

/// Storage element type as stored on disk (may be packed / sub-byte).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StorageElementType {
    Scalar { ty: ScalarType },
    PackedBits { bits: u8, signed: bool },
    Opaque { codec: String },
}

impl StorageElementType {
    pub fn scalar(ty: ScalarType) -> Self {
        Self::Scalar { ty }
    }
}

impl fmt::Display for StorageElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar { ty } => write!(f, "{ty}"),
            Self::PackedBits { bits, signed } => {
                write!(f, "{}i{bits}", if *signed { "s" } else { "u" })
            }
            Self::Opaque { codec } => write!(f, "opaque({codec})"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Endianness {
    Little,
    Big,
    Native,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorOrder {
    RowMajor,
    ColumnMajor,
}
