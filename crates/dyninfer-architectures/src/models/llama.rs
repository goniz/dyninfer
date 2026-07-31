//! Llama decoder architecture (`llama.decoder`).
//!
//! Covers Meta Llama / OpenLLaMA / Mistral-layout checkpoints and the synthetic
//! Milestone-1 fixture. Q/K norms are optional (absent for classic Llama).

use crate::naming::canonicalize_hf_family;
use crate::ops::{emit_dense_decoder_cfg, DenseDecoderConfig};
use crate::slots::{field, slot, slot_opt};
use dyninfer_architecture::{
    ArchitectureDefinition, ArchitecturePackage, ConfigSchema, EmitOutput, ModelBuilder,
    ModelModule, ResolvedModelConfig,
};
use dyninfer_checkpoint::{CheckpointCatalog, ParameterCatalog};
use dyninfer_core::{ArchitectureId, ParameterRole};
use dyninfer_error::{CompilationError, DynInferError, Result};
use std::sync::LazyLock;

static CONFIG_SCHEMA: LazyLock<ConfigSchema> = LazyLock::new(|| ConfigSchema {
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
        field("rope_theta", "f64", false, None),
    ],
});

#[derive(Debug, Default)]
pub struct LlamaArchitecture;

impl ArchitectureDefinition for LlamaArchitecture {
    fn id(&self) -> ArchitectureId {
        ArchitectureId::new("llama.decoder")
    }

    fn revision(&self) -> &str {
        "0.2.0"
    }

    fn config_schema(&self) -> &ConfigSchema {
        &CONFIG_SCHEMA
    }

    fn model_types(&self) -> &[&str] {
        &[
            "llama",
            "mistral",
            "LlamaForCausalLM",
            "MistralForCausalLM",
        ]
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
        m.declare_parameter(slot("token_embd.weight", ParameterRole::Embedding, 2))?;

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
            m.declare_parameter(slot_opt(
                &format!("{prefix}.attn_q_norm.weight"),
                ParameterRole::Norm,
                1,
                true,
            ))?;
            m.declare_parameter(slot_opt(
                &format!("{prefix}.attn_k_norm.weight"),
                ParameterRole::Norm,
                1,
                true,
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

    fn canonicalize_param(&self, key: &str) -> Option<String> {
        canonicalize_hf_family(key)
    }

    fn sanitize_catalog(&self, catalog: &mut ParameterCatalog) {
        crate::naming::tie_output_to_embed(catalog);
    }

    fn emit_executable(
        &self,
        package: &ArchitecturePackage,
        catalog: &CheckpointCatalog,
    ) -> Result<EmitOutput> {
        let cfg = DenseDecoderConfig::from_package(package, catalog);
        if !cfg.supports_dense_emit() {
            return Err(DynInferError::Compilation(CompilationError {
                message: format!("llama.decoder cannot emit dense executable for {cfg:?}"),
                pass: Some("emit".into()),
                diagnostics: vec![],
            }));
        }
        let mlir_text = emit_dense_decoder_cfg(package.id.as_str(), &cfg)?;
        Ok(EmitOutput {
            prefill_window: cfg.seq,
            max_kv: cfg.max_kv,
            mlir_text,
        })
    }
}
