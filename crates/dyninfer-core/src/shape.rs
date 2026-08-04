//! Shape and byte-range helpers.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::Range;

/// Tensor shape as a list of dimension sizes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Shape(pub Vec<u64>);

impl Shape {
    pub fn new(dims: impl Into<Vec<u64>>) -> Self {
        Self(dims.into())
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[u64] {
        &self.0
    }

    pub fn numel(&self) -> Option<u64> {
        self.0.iter().try_fold(1u64, |acc, &d| acc.checked_mul(d))
    }

    pub fn is_compatible_with(&self, other: &Shape) -> bool {
        self.0 == other.0
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

impl From<Vec<u64>> for Shape {
    fn from(value: Vec<u64>) -> Self {
        Self(value)
    }
}

impl From<&[u64]> for Shape {
    fn from(value: &[u64]) -> Self {
        Self(value.to_vec())
    }
}

/// Inclusive-start exclusive-end byte range within a source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ByteRange {
    pub offset: u64,
    pub length: u64,
}

impl ByteRange {
    pub fn new(offset: u64, length: u64) -> Self {
        Self { offset, length }
    }

    pub fn end(&self) -> u64 {
        self.offset.saturating_add(self.length)
    }

    pub fn as_range(&self) -> Range<u64> {
        self.offset..self.end()
    }
}

/// Serializable half-open range used by binding transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Range64 {
    pub start: u64,
    pub end: u64,
}

impl From<Range<u64>> for Range64 {
    fn from(r: Range<u64>) -> Self {
        Self {
            start: r.start,
            end: r.end,
        }
    }
}

/// Shape specialization profile for compilation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeProfile {
    pub batch_sizes: Vec<u32>,
    pub sequence_buckets: Vec<u32>,
    pub max_sequence_length: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

impl Default for ShapeProfile {
    fn default() -> Self {
        Self {
            batch_sizes: vec![1],
            sequence_buckets: vec![128, 512, 2048],
            max_sequence_length: 2048,
            extra: None,
        }
    }
}

impl ShapeProfile {
    pub fn single_bucket(max_seq: u32) -> Self {
        Self {
            batch_sizes: vec![1],
            sequence_buckets: vec![max_seq],
            max_sequence_length: max_seq,
            extra: None,
        }
    }
}
