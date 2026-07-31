use crate::config::{ConfigSchema, ResolvedModelConfig};
use crate::emit::EmitOutput;
use crate::package::ArchitecturePackage;
use dyninfer_checkpoint::{CheckpointCatalog, ParameterCatalog};
use dyninfer_core::{
    ArchitectureId, CanonicalParameterName, LogicalTensorConstraint, ParameterRole, ParameterSlot,
    ParameterSlotId, ScalarType,
};
use dyninfer_error::{DynInferError, Result};
use dyninfer_mlir::{ModuleBuilder, VerifiedModule};
use serde::{Deserialize, Serialize};

/// Opaque SSA value handle produced by [`ModelBuilder`] ops.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Value {
    name: String,
}

impl Value {
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// In-memory MLIR module produced by an architecture builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelModule {
    pub architecture_id: ArchitectureId,
    pub mlir_text: String,
    pub parameter_slots: Vec<ParameterSlot>,
}

/// Narrow builder API used by architecture definitions (spec §8.3.1).
///
/// Backed by [`dyninfer_mlir::ModuleBuilder`]. High-level helpers record the
/// graph sketch and parameter slots; [`ArchitectureDefinition::emit_executable`]
/// produces the full IREE executable IR.
pub struct ModelBuilder {
    architecture_id: Option<ArchitectureId>,
    slots: Vec<ParameterSlot>,
    mlir: ModuleBuilder,
    notes: Vec<String>,
    next_ssa: u32,
    exports: Vec<String>,
}

impl std::fmt::Debug for ModelBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelBuilder")
            .field("architecture_id", &self.architecture_id)
            .field("slots", &self.slots.len())
            .field("exports", &self.exports)
            .finish_non_exhaustive()
    }
}

impl ModelBuilder {
    pub fn new() -> Result<Self> {
        Ok(Self {
            architecture_id: None,
            slots: Vec::new(),
            mlir: ModuleBuilder::new()?,
            notes: Vec::new(),
            next_ssa: 0,
            exports: Vec::new(),
        })
    }

    pub fn set_architecture_id(&mut self, id: ArchitectureId) {
        self.architecture_id = Some(id);
    }

    pub fn mlir_builder(&mut self) -> &mut ModuleBuilder {
        &mut self.mlir
    }

    pub fn declare_parameter(&mut self, slot: ParameterSlot) -> Result<()> {
        self.slots.push(slot);
        Ok(())
    }

    fn alloc(&mut self, prefix: &str) -> Value {
        let id = self.next_ssa;
        self.next_ssa += 1;
        Value {
            name: format!("{prefix}{id}"),
        }
    }

    fn push_slot(&mut self, name: &str, role: ParameterRole, rank: usize, optional: bool) -> Result<()> {
        self.declare_parameter(ParameterSlot {
            id: ParameterSlotId::new(name),
            canonical_name: CanonicalParameterName::new(name),
            role,
            expected_type: LogicalTensorConstraint {
                rank: Some(rank),
                shape: None,
                element_types: vec![ScalarType::Bf16, ScalarType::F16, ScalarType::F32],
            },
            supported_encodings: vec!["plain".into(), "gguf.q4_0".into()],
            optional,
            tied_group: None,
        })
    }

    /// Declare the token input value for the graph.
    pub fn input_tokens(&mut self, name: &str) -> Result<Value> {
        Ok(Value {
            name: name.to_string(),
        })
    }

    /// Embedding lookup: declares `weight` and returns an activation value.
    pub fn embedding(&mut self, tokens: Value, weight: &str) -> Result<Value> {
        self.push_slot(weight, ParameterRole::Embedding, 2, false)?;
        let out = self.alloc("emb");
        self.note_op(format!(
            "embedding {} = embed({}, weight={})",
            out.name,
            tokens.name(),
            weight
        ))?;
        Ok(out)
    }

    /// Linear projection: declares `weight` and returns an activation value.
    pub fn linear(&mut self, x: Value, weight: &str) -> Result<Value> {
        let role = if weight.contains("output") {
            ParameterRole::Output
        } else {
            ParameterRole::FfnDown
        };
        self.push_slot(weight, role, 2, false)?;
        let out = self.alloc("lin");
        self.note_op(format!(
            "linear {} = linear({}, weight={})",
            out.name,
            x.name(),
            weight
        ))?;
        Ok(out)
    }

    /// RMSNorm: declares `weight` and returns a normalized activation.
    pub fn rms_norm(&mut self, x: Value, weight: &str) -> Result<Value> {
        self.push_slot(weight, ParameterRole::Norm, 1, false)?;
        let out = self.alloc("rms");
        self.note_op(format!(
            "rms_norm {} = rms_norm({}, weight={})",
            out.name,
            x.name(),
            weight
        ))?;
        Ok(out)
    }

    /// Record a dense transformer block (declares layer parameter slots).
    pub fn dense_block(&mut self, x: Value, layer: u32, has_qk_norm: bool) -> Result<Value> {
        let prefix = format!("blk.{layer}");
        self.push_slot(&format!("{prefix}.attn_norm.weight"), ParameterRole::Norm, 1, false)?;
        self.push_slot(&format!("{prefix}.attn_q.weight"), ParameterRole::AttentionQ, 2, false)?;
        self.push_slot(&format!("{prefix}.attn_k.weight"), ParameterRole::AttentionK, 2, false)?;
        self.push_slot(&format!("{prefix}.attn_v.weight"), ParameterRole::AttentionV, 2, false)?;
        self.push_slot(
            &format!("{prefix}.attn_output.weight"),
            ParameterRole::AttentionO,
            2,
            false,
        )?;
        if has_qk_norm {
            self.push_slot(&format!("{prefix}.attn_q_norm.weight"), ParameterRole::Norm, 1, false)?;
            self.push_slot(&format!("{prefix}.attn_k_norm.weight"), ParameterRole::Norm, 1, false)?;
        } else {
            self.push_slot(&format!("{prefix}.attn_q_norm.weight"), ParameterRole::Norm, 1, true)?;
            self.push_slot(&format!("{prefix}.attn_k_norm.weight"), ParameterRole::Norm, 1, true)?;
        }
        self.push_slot(&format!("{prefix}.ffn_norm.weight"), ParameterRole::Norm, 1, false)?;
        self.push_slot(&format!("{prefix}.ffn_gate.weight"), ParameterRole::FfnGate, 2, false)?;
        self.push_slot(&format!("{prefix}.ffn_up.weight"), ParameterRole::FfnUp, 2, false)?;
        self.push_slot(&format!("{prefix}.ffn_down.weight"), ParameterRole::FfnDown, 2, false)?;
        let out = self.alloc("blk");
        self.note_op(format!(
            "dense_block {} = block({}, layer={layer})",
            out.name,
            x.name()
        ))?;
        Ok(out)
    }

    /// Mark prefill/decode exports for the executable ABI.
    pub fn export_prefill_and_decode(&mut self, logits: Value) -> Result<()> {
        self.exports.push("prefill".into());
        self.exports.push("decode".into());
        self.note_op(format!("export prefill,decode logits={}", logits.name()))?;
        Ok(())
    }

    /// Append a raw MLIR fragment into the underlying module builder.
    pub fn append_mlir(&mut self, text: impl Into<String>) -> Result<()> {
        self.mlir.append_toplevel_asm(&text.into())
    }

    /// Note a high-level op (graph sketch; slots come from helpers).
    pub fn note_op(&mut self, op: impl Into<String>) -> Result<()> {
        self.notes.push(op.into());
        Ok(())
    }

    pub fn finish(&mut self) -> Result<ModelModule> {
        let architecture_id = self.architecture_id.clone().ok_or_else(|| {
            DynInferError::internal("ModelBuilder missing architecture_id")
        })?;
        let slots = std::mem::take(&mut self.slots);
        let _notes = std::mem::take(&mut self.notes);

        // Graph sketch lives in `notes` / slots; executable IR comes from
        // `emit_executable`. Do not call verify_mlir here: that would try to
        // create another ModuleBuilder while `self.mlir` still holds the
        // process-wide MLIR lock.
        Ok(ModelModule {
            architecture_id,
            mlir_text: "module {\n}\n".into(),
            parameter_slots: slots,
        })
    }
}

/// Parse + verify MLIR text through the melior-style builder.
pub fn verify_mlir(source: &str) -> Result<VerifiedModule> {
    let mut builder = ModuleBuilder::new()?;
    builder.parse_source(source)?;
    builder.finish()
}

/// Architecture plugin: slots, naming, and executable emission.
pub trait ArchitectureDefinition: Send + Sync {
    fn id(&self) -> ArchitectureId;
    fn revision(&self) -> &str;
    fn config_schema(&self) -> &ConfigSchema;

    /// HF `model_type` / architecture class stems this definition accepts.
    fn model_types(&self) -> &[&str];

    fn build(
        &self,
        config: &ResolvedModelConfig,
        builder: &mut ModelBuilder,
    ) -> Result<ModelModule>;

    /// Map a checkpoint tensor key to a canonical parameter name.
    ///
    /// Return `None` to skip the tensor (e.g. cached RoPE freqs).
    /// Default keeps the key unchanged.
    fn canonicalize_param(&self, key: &str) -> Option<String> {
        Some(key.to_string())
    }

    /// Post-process the parameter catalog (tied embeddings, drop unused keys, …).
    fn sanitize_catalog(&self, _catalog: &mut ParameterCatalog) {}

    /// Emit the IREE-facing MLIR executable for this architecture.
    fn emit_executable(
        &self,
        package: &ArchitecturePackage,
        catalog: &CheckpointCatalog,
    ) -> Result<EmitOutput>;
}
