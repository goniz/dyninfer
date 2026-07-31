//! Qwen3 decoder architecture (`qwen3.decoder`).
//!
//! GQA, independent `head_dim`, per-head Q/K RMSNorm, often tied embeddings.

use crate::naming::{canonicalize_hf_family, tie_output_to_embed};
use crate::ops::{emit_dense_decoder_cfg, DenseDecoderConfig};
use crate::slots::{field, slot};
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
        field("num_layers", "u32", true, Some(serde_json::json!(28))),
        field("num_heads", "u32", true, Some(serde_json::json!(16))),
        field("num_kv_heads", "u32", true, Some(serde_json::json!(8))),
        field("head_dim", "u32", true, Some(serde_json::json!(128))),
        field("hidden_size", "u32", true, Some(serde_json::json!(1024))),
        field("intermediate_size", "u32", true, Some(serde_json::json!(3072))),
        field("vocab_size", "u32", true, Some(serde_json::json!(151936))),
        field("context_length", "u32", true, Some(serde_json::json!(40960))),
        field("rms_norm_eps", "f64", false, Some(serde_json::json!(1e-6))),
        field("rope_theta", "f64", false, Some(serde_json::json!(1_000_000.0))),
    ],
});

#[derive(Debug, Default)]
pub struct Qwen3Architecture;

impl ArchitectureDefinition for Qwen3Architecture {
    fn id(&self) -> ArchitectureId {
        ArchitectureId::new("qwen3.decoder")
    }

    fn revision(&self) -> &str {
        "0.1.0"
    }

    fn config_schema(&self) -> &ConfigSchema {
        &CONFIG_SCHEMA
    }

    fn model_types(&self) -> &[&str] {
        &["qwen3", "Qwen3ForCausalLM"]
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
            m.note_op(format!("qwen3_block {layer}"));
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
                &format!("{prefix}.attn_q_norm.weight"),
                ParameterRole::Norm,
                1,
            ))?;
            m.declare_parameter(slot(
                &format!("{prefix}.attn_k_norm.weight"),
                ParameterRole::Norm,
                1,
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
        tie_output_to_embed(catalog);
    }

    fn emit_executable(
        &self,
        package: &ArchitecturePackage,
        catalog: &CheckpointCatalog,
    ) -> Result<EmitOutput> {
        let mut cfg = DenseDecoderConfig::from_package(package, catalog);
        cfg.has_qk_norm = true;
        if !cfg.supports_dense_emit() {
            return Err(DynInferError::Compilation(CompilationError {
                message: format!("qwen3.decoder cannot emit dense executable for {cfg:?}"),
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
