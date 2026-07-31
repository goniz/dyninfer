//! Structured errors and diagnostics for Dynamic Inference Engine.
//!
//! All public fallible APIs return [`Result`] using [`DynInferError`].

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Crate-wide result alias.
pub type Result<T, E = DynInferError> = std::result::Result<T, E>;

/// Top-level structured error taxonomy from the specification.
#[derive(Debug, Error)]
pub enum DynInferError {
    #[error("I/O error: {0}")]
    Io(#[from] IoError),

    #[error("unsupported container: {0}")]
    UnsupportedContainer(UnsupportedContainerError),

    #[error("invalid checkpoint: {0}")]
    InvalidCheckpoint(CheckpointValidationError),

    #[error("unsupported encoding: {0}")]
    UnsupportedEncoding(UnsupportedEncodingError),

    #[error("architecture mismatch: {0}")]
    ArchitectureMismatch(ArchitectureMismatchError),

    #[error("binding error: {0}")]
    Binding(BindingError),

    #[error("compilation error: {0}")]
    Compilation(CompilationError),

    #[error("IREE runtime error: {0}")]
    IreeRuntime(IreeRuntimeError),

    #[error("device error: {0}")]
    Device(DeviceError),

    #[error("cache error: {0}")]
    Cache(CacheError),

    #[error("configuration error: {0}")]
    Config(ConfigError),

    #[error("internal error: {0}")]
    Internal(String),
}

impl DynInferError {
    /// Stable machine-readable error code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Io(_) => "E_IO",
            Self::UnsupportedContainer(_) => "E_UNSUPPORTED_CONTAINER",
            Self::InvalidCheckpoint(_) => "E_INVALID_CHECKPOINT",
            Self::UnsupportedEncoding(_) => "E_UNSUPPORTED_ENCODING",
            Self::ArchitectureMismatch(_) => "E_ARCHITECTURE_MISMATCH",
            Self::Binding(_) => "E_BINDING",
            Self::Compilation(_) => "E_COMPILATION",
            Self::IreeRuntime(_) => "E_IREE_RUNTIME",
            Self::Device(_) => "E_DEVICE",
            Self::Cache(_) => "E_CACHE",
            Self::Config(_) => "E_CONFIG",
            Self::Internal(_) => "E_INTERNAL",
        }
    }

    pub fn io(message: impl Into<String>) -> Self {
        Self::Io(IoError {
            message: message.into(),
            path: None,
            source: None,
        })
    }

    pub fn io_path(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Io(IoError {
            message: message.into(),
            path: Some(path.into()),
            source: None,
        })
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
}

impl From<std::io::Error> for DynInferError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(IoError {
            message: err.to_string(),
            path: None,
            source: Some(err),
        })
    }
}

impl From<serde_json::Error> for DynInferError {
    fn from(err: serde_json::Error) -> Self {
        Self::Config(ConfigError {
            message: format!("JSON error: {err}"),
        })
    }
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct IoError {
    pub message: String,
    pub path: Option<String>,
    #[source]
    pub source: Option<std::io::Error>,
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[error("{message}")]
pub struct UnsupportedContainerError {
    pub message: String,
    pub path: Option<String>,
    pub probed_formats: Vec<String>,
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[error("{message}")]
pub struct CheckpointValidationError {
    pub message: String,
    pub key: Option<String>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[error("{message}")]
pub struct UnsupportedEncodingError {
    pub message: String,
    pub key: Option<String>,
    pub codec: Option<String>,
    pub codec_version: Option<u32>,
    pub expected: Option<String>,
    pub actual: Option<String>,
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[error("{message}")]
pub struct ArchitectureMismatchError {
    pub message: String,
    pub architecture_id: Option<String>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[error("{message}")]
pub struct BindingError {
    pub message: String,
    pub slot: Option<String>,
    pub checkpoint_key: Option<String>,
    pub expected: Option<String>,
    pub actual: Option<String>,
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[error("{message}")]
pub struct CompilationError {
    pub message: String,
    pub pass: Option<String>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[error("{message}")]
pub struct IreeRuntimeError {
    pub message: String,
    pub status_code: Option<i32>,
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[error("{message}")]
pub struct DeviceError {
    pub message: String,
    pub driver: Option<String>,
    pub device: Option<String>,
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[error("{message}")]
pub struct CacheError {
    pub message: String,
    pub digest: Option<String>,
    pub path: Option<String>,
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[error("{message}")]
pub struct ConfigError {
    pub message: String,
}

/// Severity for compiler and inspection diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
    Note,
    Remark,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "error"),
            Self::Warning => write!(f, "warning"),
            Self::Note => write!(f, "note"),
            Self::Remark => write!(f, "remark"),
        }
    }
}

/// Structured diagnostic attached to compilation or inspection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: String,
    pub severity: Severity,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture_op: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter_slot: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pass_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

impl Diagnostic {
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Error,
            message: message.into(),
            architecture_op: None,
            parameter_slot: None,
            checkpoint_key: None,
            expected: None,
            actual: None,
            pass_name: None,
            suggestion: None,
        }
    }

    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Warning,
            message: message.into(),
            architecture_op: None,
            parameter_slot: None,
            checkpoint_key: None,
            expected: None,
            actual: None,
            pass_name: None,
            suggestion: None,
        }
    }

    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.checkpoint_key = Some(key.into());
        self
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    pub fn with_expected_actual(
        mut self,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        self.expected = Some(expected.into());
        self.actual = Some(actual.into());
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]: {}", self.severity, self.code, self.message)?;
        if let Some(key) = &self.checkpoint_key {
            write!(f, " (key={key})")?;
        }
        if let Some(suggestion) = &self.suggestion {
            write!(f, "; suggestion: {suggestion}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_codes_are_stable() {
        let err = DynInferError::internal("boom");
        assert_eq!(err.code(), "E_INTERNAL");
    }

    #[test]
    fn diagnostic_display_includes_code() {
        let d = Diagnostic::error("E_TEST", "something broke").with_key("w.weight");
        let s = d.to_string();
        assert!(s.contains("E_TEST"));
        assert!(s.contains("w.weight"));
    }
}
