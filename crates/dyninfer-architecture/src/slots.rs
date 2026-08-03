//! Shared parameter-slot helpers for decoder architectures.

use dyninfer_core::{
    CanonicalParameterName, LogicalTensorConstraint, ParameterRole, ParameterSlot, ParameterSlotId,
    ScalarType,
};

#[allow(dead_code)] // kept for architecture plugins that declare slots explicitly
pub fn slot(name: &str, role: ParameterRole, rank: usize) -> ParameterSlot {
    slot_opt(name, role, rank, false)
}

#[allow(dead_code)]
pub fn slot_opt(name: &str, role: ParameterRole, rank: usize, optional: bool) -> ParameterSlot {
    ParameterSlot {
        id: ParameterSlotId::new(name),
        canonical_name: CanonicalParameterName::new(name),
        role,
        expected_type: LogicalTensorConstraint {
            rank: Some(rank),
            shape: None,
            element_types: vec![ScalarType::Bf16, ScalarType::F16, ScalarType::F32],
        },
        // Q4_0 stays listed for forward-looking slot docs; binder rejects it
        // until qkernel lowering lands (`PhysicalEncoding::is_supported_v1`).
        supported_encodings: vec!["plain".into(), "gguf.q4_0".into()],
        optional,
        tied_group: None,
    }
}

pub fn field(
    name: &str,
    ty: &str,
    required: bool,
    default: Option<serde_json::Value>,
) -> crate::ConfigField {
    crate::ConfigField {
        name: name.into(),
        ty: ty.into(),
        required,
        default,
        description: None,
    }
}
