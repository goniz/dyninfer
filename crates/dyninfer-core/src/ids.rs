//! Strongly-typed string identifiers.

use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! string_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::new(value)
            }
        }
    };
}

string_id!(
    /// Stable architecture identifier, e.g. `llama.decoder`.
    ArchitectureId
);
string_id!(
    /// Container format id, e.g. `safetensors`, `gguf`.
    ContainerFormatId
);
string_id!(
    /// Convention decoder id, e.g. `dense`, `gguf.q4_0`.
    ConventionId
);
string_id!(
    /// Canonical logical parameter name used by architecture slots.
    CanonicalParameterName
);
string_id!(
    /// Architecture parameter slot identifier.
    ParameterSlotId
);
string_id!(
    /// Stable identifier for a semantic operation in Architecture IR.
    OperationId
);
string_id!(
    /// Stable identifier for a tensor value in Architecture IR.
    GraphValueId
);
string_id!(
    /// Stable identifier for a selected production kernel.
    KernelId
);
string_id!(
    /// Stable identifier for a compiler lowering implementation.
    LoweringId
);
string_id!(
    /// Quantization / packing codec identifier.
    CodecId
);
string_id!(
    /// Stable physical encoding definition identifier.
    EncodingId
);
string_id!(
    /// Group of tied parameters that must share storage.
    TiedParameterGroup
);
