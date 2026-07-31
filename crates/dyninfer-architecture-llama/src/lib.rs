//! Llama decoder architecture (Milestone 1 dense proof of concept).

#![forbid(unsafe_code)]

use dyninfer_architecture::{
    ArchitectureDefinition, ConfigField, ConfigSchema, ModelBuilder, ModelModule, ResolvedModelConfig,
};
use dyninfer_core::{
    ArchitectureId, CanonicalParameterName, LogicalTensorConstraint, ParameterRole, ParameterSlot,
    ParameterSlotId, ScalarType,
};
use dyninfer_error::Result;
use std::sync::LazyLock;

static LLAMA_CONFIG_SCHEMA: LazyLock<ConfigSchema> = LazyLock::new(|| ConfigSchema {
    fields: vec![
        field("num_layers", "u32", true, Some(serde_json::json!(2))),
        field("num_heads", "u32", true, Some(serde_json::json!(4))),
        field("num_kv_heads", "u32", true, Some(serde_json::json!(4))),
        field("head_dim", "u32", true, Some(serde_json::json!(64))),
        field("hidden_size", "u32", true, Some(serde_json::json!(256))),
        field("intermediate_size", "u32", true, Some(serde_json::json!(512))),
        field("vocab_size", "u32", true, Some(serde_json::json!(32000))),
        field("context_length", "u32", true, Some(serde_json::json!(2048))),
        field("rms_norm_eps", "f64", false, Some(serde_json::json!(1e-5))),
    ],
});

fn field(name: &str, ty: &str, required: bool, default: Option<serde_json::Value>) -> ConfigField {
    ConfigField {
        name: name.into(),
        ty: ty.into(),
        required,
        default,
        description: None,
    }
}

fn slot(name: &str, role: ParameterRole, rank: usize) -> ParameterSlot {
    ParameterSlot {
        id: ParameterSlotId::new(name),
        canonical_name: CanonicalParameterName::new(name),
        role,
        expected_type: LogicalTensorConstraint {
            rank: Some(rank),
            shape: None,
            element_types: vec![ScalarType::Bf16, ScalarType::F16, ScalarType::F32],
        },
        supported_encodings: vec![
            "plain".into(),
            "gguf.q4_0".into(),
        ],
        optional: false,
        tied_group: None,
    }
}

#[derive(Debug, Default)]
pub struct LlamaArchitecture;

impl ArchitectureDefinition for LlamaArchitecture {
    fn id(&self) -> ArchitectureId {
        ArchitectureId::new("llama.decoder")
    }

    fn revision(&self) -> &str {
        "0.1.0"
    }

    fn config_schema(&self) -> &ConfigSchema {
        &LLAMA_CONFIG_SCHEMA
    }

    fn build(
        &self,
        config: &ResolvedModelConfig,
        m: &mut ModelBuilder,
    ) -> Result<ModelModule> {
        let num_layers = config.num_layers()?;
        let hidden = config.get_u32("hidden_size")?;
        let vocab = config.get_u32("vocab_size")?;

        m.note_op(format!("embedding vocab={vocab} hidden={hidden}"));
        m.declare_parameter(slot(
            "token_embd.weight",
            ParameterRole::Embedding,
            2,
        ))?;

        for layer in 0..num_layers {
            let prefix = format!("blk.{layer}");
            m.note_op(format!("llama_block {layer}"));
            m.declare_parameter(slot(
                &format!("{prefix}.attn_norm.weight"),
                ParameterRole::Norm,
                1,
            ))?;
            m.declare_parameter(slot(
                &format!("{prefix}.attn_q.weight"),
                ParameterRole::AttentionQ,
                2,
            ))?;
            m.declare_parameter(slot(
                &format!("{prefix}.attn_k.weight"),
                ParameterRole::AttentionK,
                2,
            ))?;
            m.declare_parameter(slot(
                &format!("{prefix}.attn_v.weight"),
                ParameterRole::AttentionV,
                2,
            ))?;
            m.declare_parameter(slot(
                &format!("{prefix}.attn_output.weight"),
                ParameterRole::AttentionO,
                2,
            ))?;
            m.declare_parameter(slot(
                &format!("{prefix}.ffn_norm.weight"),
                ParameterRole::Norm,
                1,
            ))?;
            m.declare_parameter(slot(
                &format!("{prefix}.ffn_gate.weight"),
                ParameterRole::FfnGate,
                2,
            ))?;
            m.declare_parameter(slot(
                &format!("{prefix}.ffn_up.weight"),
                ParameterRole::FfnUp,
                2,
            ))?;
            m.declare_parameter(slot(
                &format!("{prefix}.ffn_down.weight"),
                ParameterRole::FfnDown,
                2,
            ))?;
        }

        m.declare_parameter(slot("output_norm.weight", ParameterRole::Norm, 1))?;
        m.declare_parameter(slot("output.weight", ParameterRole::Output, 2))?;
        m.note_op("export prefill,decode");
        m.finish()
    }
}

/// Register the Llama architecture into a registry.
pub fn register(registry: &mut dyninfer_architecture::ArchitectureRegistry) {
    registry.register(LlamaArchitecture);
}
