//! Content-addressed digests and schema fingerprints.

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fmt;

/// Hex-encoded SHA-256 digest.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Digest(String);

impl Digest {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let hash = Sha256::digest(bytes);
        Self(hex::encode(hash))
    }

    pub fn from_hex(hex_str: impl Into<String>) -> Self {
        Self(hex_str.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn short(&self) -> &str {
        let s = self.as_str();
        if s.len() >= 12 { &s[..12] } else { s }
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest({})", self.short())
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for Digest {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

/// Fingerprint of a checkpoint schema (names, shapes, encodings) independent of weight bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchemaFingerprint {
    pub digest: Digest,
    pub entry_count: u64,
    pub total_bytes: u64,
}

impl SchemaFingerprint {
    pub fn compute(canonical_json: &str, entry_count: u64, total_bytes: u64) -> Self {
        Self {
            digest: Digest::from_bytes(canonical_json.as_bytes()),
            entry_count,
            total_bytes,
        }
    }
}

/// Hash arbitrary serializable content into a digest.
pub fn content_digest<T: Serialize>(value: &T) -> dyninfer_error::Result<Digest> {
    let bytes = serde_json::to_vec(value)?;
    Ok(Digest::from_bytes(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_stable() {
        let a = Digest::from_bytes(b"hello");
        let b = Digest::from_bytes(b"hello");
        assert_eq!(a, b);
        assert_eq!(a.as_str().len(), 64);
    }
}
