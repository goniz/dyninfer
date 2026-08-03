use crate::config::{ConfigSchema, ResolvedModelConfig};
use dyninfer_checkpoint::ParameterCatalog;
use dyninfer_core::{
    ArchitectureExport, ArchitectureGraph, ArchitectureId, ArchitectureOperation,
    CanonicalParameterName, ElementwiseFunction, ExecutionMode, GraphValue, GraphValueId,
    KvCacheComponent, LogicalTensorConstraint, ModelInputKind, OperationId, OperationKind,
    ParameterRole, ParameterSlot, ParameterSlotId, ScalarType, SemanticTensorType,
};
use dyninfer_error::{DynInferError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Typed graph value produced by [`ModelBuilder`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Value {
    id: GraphValueId,
    tensor_type: SemanticTensorType,
}

impl Value {
    pub fn id(&self) -> &GraphValueId {
        &self.id
    }

    pub fn name(&self) -> &str {
        self.id.as_str()
    }

    pub fn tensor_type(&self) -> &SemanticTensorType {
        &self.tensor_type
    }
}

/// Architecture module produced before checkpoint binding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelModule {
    pub graph: ArchitectureGraph,
}

/// Attributes needed to expand a shared causal decoder block into typed
/// semantic operations.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderBlockSpec {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rms_norm_epsilon: f64,
    pub rope_theta: Option<f64>,
}

impl DecoderBlockSpec {
    fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_heads == 0
            || self.num_kv_heads == 0
            || self.head_dim == 0
        {
            return Err(DynInferError::internal(
                "decoder block dimensions and head counts must be non-zero",
            ));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(DynInferError::internal(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            )));
        }
        Ok(())
    }
}

/// Typed semantic graph builder used by compiled-in architecture definitions.
///
/// Helpers construct checkpoint- and encoding-independent operations. Composite
/// decoder helpers expand immediately; no opaque block operation survives into
/// [`ArchitectureGraph`].
pub struct ModelBuilder {
    architecture_id: Option<ArchitectureId>,
    slots: Vec<ParameterSlot>,
    values: Vec<GraphValue>,
    operations: Vec<ArchitectureOperation>,
    exports: Vec<ArchitectureExport>,
    primary_input: Option<GraphValueId>,
}

impl std::fmt::Debug for ModelBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelBuilder")
            .field("architecture_id", &self.architecture_id)
            .field("slots", &self.slots.len())
            .field("values", &self.values.len())
            .field("operations", &self.operations.len())
            .field("exports", &self.exports.len())
            .finish()
    }
}

impl ModelBuilder {
    pub fn new() -> Result<Self> {
        Ok(Self {
            architecture_id: None,
            slots: Vec::new(),
            values: Vec::new(),
            operations: Vec::new(),
            exports: Vec::new(),
            primary_input: None,
        })
    }

    pub fn set_architecture_id(&mut self, id: ArchitectureId) {
        self.architecture_id = Some(id);
    }

    pub fn declare_parameter(&mut self, slot: ParameterSlot) -> Result<()> {
        if self.slots.iter().any(|existing| existing.id == slot.id) {
            return Err(DynInferError::internal(format!(
                "duplicate parameter slot `{}`",
                slot.id
            )));
        }
        self.slots.push(slot);
        Ok(())
    }

    fn push_slot(
        &mut self,
        name: &str,
        role: ParameterRole,
        rank: usize,
    ) -> Result<ParameterSlotId> {
        let id = ParameterSlotId::new(name);
        self.declare_parameter(ParameterSlot {
            id: id.clone(),
            canonical_name: CanonicalParameterName::new(name),
            role,
            expected_type: LogicalTensorConstraint {
                rank: Some(rank),
                shape: None,
                element_types: vec![ScalarType::Bf16, ScalarType::F16, ScalarType::F32],
            },
            optional: false,
            tied_group: None,
        })?;
        Ok(id)
    }

    fn value(&mut self, id: impl Into<String>, tensor_type: SemanticTensorType) -> Result<Value> {
        let id = GraphValueId::new(id);
        if self.values.iter().any(|value| value.id == id) {
            return Err(DynInferError::internal(format!(
                "duplicate graph value `{id}`"
            )));
        }
        self.values.push(GraphValue {
            id: id.clone(),
            tensor_type: tensor_type.clone(),
        });
        Ok(Value { id, tensor_type })
    }

    fn operation(
        &mut self,
        id: impl Into<String>,
        kind: OperationKind,
        inputs: &[Value],
        outputs: &[Value],
        parameters: Vec<ParameterSlotId>,
    ) -> Result<()> {
        let id = OperationId::new(id);
        if self.operations.iter().any(|operation| operation.id == id) {
            return Err(DynInferError::internal(format!(
                "duplicate architecture operation `{id}`"
            )));
        }
        self.operations.push(ArchitectureOperation {
            id,
            kind,
            inputs: inputs.iter().map(|value| value.id.clone()).collect(),
            outputs: outputs.iter().map(|value| value.id.clone()).collect(),
            parameters,
        });
        Ok(())
    }

    fn unary(
        &mut self,
        id: &str,
        kind: OperationKind,
        input: Value,
        output_type: SemanticTensorType,
        parameters: Vec<ParameterSlotId>,
    ) -> Result<Value> {
        let output = self.value(format!("{id}.output"), output_type)?;
        self.operation(
            id,
            kind,
            &[input],
            std::slice::from_ref(&output),
            parameters,
        )?;
        Ok(output)
    }

    /// Declare the token input value for the graph.
    pub fn input_tokens(&mut self, name: &str) -> Result<Value> {
        if self.primary_input.is_some() {
            return Err(DynInferError::internal(
                "an architecture graph may declare only one primary token input",
            ));
        }
        let value = self.value(name, SemanticTensorType::tokens())?;
        self.operation(
            format!("input.{name}"),
            OperationKind::Input {
                input: ModelInputKind::Tokens,
            },
            &[],
            std::slice::from_ref(&value),
            vec![],
        )?;
        self.primary_input = Some(value.id.clone());
        Ok(value)
    }

    /// Embedding lookup with an explicitly named semantic operation.
    pub fn embedding(
        &mut self,
        operation_id: &str,
        tokens: Value,
        weight: &str,
        hidden_size: u32,
    ) -> Result<Value> {
        let slot = self.push_slot(weight, ParameterRole::Embedding, 2)?;
        self.unary(
            operation_id,
            OperationKind::Embedding,
            tokens,
            SemanticTensorType::activations(hidden_size),
            vec![slot],
        )
    }

    fn linear(
        &mut self,
        operation_id: &str,
        input: Value,
        weight: &str,
        output_width: u32,
        role: ParameterRole,
    ) -> Result<Value> {
        let slot = self.push_slot(weight, role.clone(), 2)?;
        self.unary(
            operation_id,
            OperationKind::Linear { role },
            input,
            SemanticTensorType::activations(output_width),
            vec![slot],
        )
    }

    fn rms_norm(
        &mut self,
        operation_id: &str,
        input: Value,
        weight: &str,
        epsilon: f64,
    ) -> Result<Value> {
        let slot = self.push_slot(weight, ParameterRole::Norm, 1)?;
        let output_type = input.tensor_type.clone();
        self.unary(
            operation_id,
            OperationKind::RmsNorm { epsilon },
            input,
            output_type,
            vec![slot],
        )
    }

    fn per_head_rms_norm(
        &mut self,
        operation_id: &str,
        input: Value,
        weight: &str,
        epsilon: f64,
        head_count: u32,
        head_dim: u32,
    ) -> Result<Value> {
        let slot = self.push_slot(weight, ParameterRole::Norm, 1)?;
        let output_type = input.tensor_type.clone();
        self.unary(
            operation_id,
            OperationKind::PerHeadRmsNorm {
                epsilon,
                head_count,
                head_dim,
            },
            input,
            output_type,
            vec![slot],
        )
    }

    fn rope(
        &mut self,
        operation_id: &str,
        input: Value,
        head_count: u32,
        head_dim: u32,
        theta: f64,
    ) -> Result<Value> {
        let output_type = input.tensor_type.clone();
        self.unary(
            operation_id,
            OperationKind::Rope {
                head_count,
                head_dim,
                theta,
            },
            input,
            output_type,
            vec![],
        )
    }

    fn cache_write(
        &mut self,
        operation_id: &str,
        input: Value,
        layer: u32,
        component: KvCacheComponent,
    ) -> Result<()> {
        self.operation(
            operation_id,
            OperationKind::KvCacheWrite { layer, component },
            &[input],
            &[],
            vec![],
        )
    }

    fn cache_read(
        &mut self,
        operation_id: &str,
        current: Value,
        layer: u32,
        component: KvCacheComponent,
        head_count: u32,
        head_dim: u32,
    ) -> Result<Value> {
        self.unary(
            operation_id,
            OperationKind::KvCacheRead { layer, component },
            current,
            SemanticTensorType::kv_cache(head_count, head_dim),
            vec![],
        )
    }

    fn attention(
        &mut self,
        operation_id: &str,
        query: Value,
        key_cache: Value,
        value_cache: Value,
        spec: &DecoderBlockSpec,
    ) -> Result<Value> {
        let output = self.value(
            format!("{operation_id}.output"),
            SemanticTensorType::activations(spec.num_heads * spec.head_dim),
        )?;
        self.operation(
            operation_id,
            OperationKind::Attention {
                num_heads: spec.num_heads,
                num_kv_heads: spec.num_kv_heads,
                head_dim: spec.head_dim,
                causal: true,
            },
            &[query, key_cache, value_cache],
            std::slice::from_ref(&output),
            vec![],
        )?;
        Ok(output)
    }

    fn elementwise_unary(
        &mut self,
        operation_id: &str,
        input: Value,
        function: ElementwiseFunction,
    ) -> Result<Value> {
        let output_type = input.tensor_type.clone();
        self.unary(
            operation_id,
            OperationKind::Elementwise { function },
            input,
            output_type,
            vec![],
        )
    }

    fn binary(
        &mut self,
        operation_id: &str,
        kind: OperationKind,
        left: Value,
        right: Value,
    ) -> Result<Value> {
        if left.tensor_type != right.tensor_type {
            return Err(DynInferError::internal(format!(
                "operation `{operation_id}` requires equal input tensor types"
            )));
        }
        let output = self.value(format!("{operation_id}.output"), left.tensor_type.clone())?;
        self.operation(
            operation_id,
            kind,
            &[left, right],
            std::slice::from_ref(&output),
            vec![],
        )?;
        Ok(output)
    }

    /// Expand a causal decoder layer into its typed semantic operations.
    pub fn decoder_block(
        &mut self,
        input: Value,
        layer: u32,
        has_qk_norm: bool,
        spec: &DecoderBlockSpec,
    ) -> Result<Value> {
        spec.validate()?;
        let prefix = format!("blk.{layer}");
        let residual = input.clone();
        let normalized = self.rms_norm(
            &format!("{prefix}.attn_norm"),
            input,
            &format!("{prefix}.attn_norm.weight"),
            spec.rms_norm_epsilon,
        )?;
        let q_dim = spec.num_heads * spec.head_dim;
        let kv_dim = spec.num_kv_heads * spec.head_dim;
        let mut query = self.linear(
            &format!("{prefix}.attn_q"),
            normalized.clone(),
            &format!("{prefix}.attn_q.weight"),
            q_dim,
            ParameterRole::AttentionQ,
        )?;
        let mut key = self.linear(
            &format!("{prefix}.attn_k"),
            normalized.clone(),
            &format!("{prefix}.attn_k.weight"),
            kv_dim,
            ParameterRole::AttentionK,
        )?;
        let value = self.linear(
            &format!("{prefix}.attn_v"),
            normalized,
            &format!("{prefix}.attn_v.weight"),
            kv_dim,
            ParameterRole::AttentionV,
        )?;
        if has_qk_norm {
            query = self.per_head_rms_norm(
                &format!("{prefix}.attn_q_norm"),
                query,
                &format!("{prefix}.attn_q_norm.weight"),
                spec.rms_norm_epsilon,
                spec.num_heads,
                spec.head_dim,
            )?;
            key = self.per_head_rms_norm(
                &format!("{prefix}.attn_k_norm"),
                key,
                &format!("{prefix}.attn_k_norm.weight"),
                spec.rms_norm_epsilon,
                spec.num_kv_heads,
                spec.head_dim,
            )?;
        }
        if let Some(theta) = spec.rope_theta {
            query = self.rope(
                &format!("{prefix}.attn_q_rope"),
                query,
                spec.num_heads,
                spec.head_dim,
                theta,
            )?;
            key = self.rope(
                &format!("{prefix}.attn_k_rope"),
                key,
                spec.num_kv_heads,
                spec.head_dim,
                theta,
            )?;
        }
        self.cache_write(
            &format!("{prefix}.key_cache_write"),
            key.clone(),
            layer,
            KvCacheComponent::Key,
        )?;
        self.cache_write(
            &format!("{prefix}.value_cache_write"),
            value.clone(),
            layer,
            KvCacheComponent::Value,
        )?;
        let key_cache = self.cache_read(
            &format!("{prefix}.key_cache_read"),
            key,
            layer,
            KvCacheComponent::Key,
            spec.num_kv_heads,
            spec.head_dim,
        )?;
        let value_cache = self.cache_read(
            &format!("{prefix}.value_cache_read"),
            value,
            layer,
            KvCacheComponent::Value,
            spec.num_kv_heads,
            spec.head_dim,
        )?;
        let attended = self.attention(
            &format!("{prefix}.attention"),
            query,
            key_cache,
            value_cache,
            spec,
        )?;
        let projected = self.linear(
            &format!("{prefix}.attn_output"),
            attended,
            &format!("{prefix}.attn_output.weight"),
            spec.hidden_size,
            ParameterRole::AttentionO,
        )?;
        let after_attention = self.binary(
            &format!("{prefix}.attn_residual"),
            OperationKind::Residual,
            residual,
            projected,
        )?;

        let ffn_residual = after_attention.clone();
        let ffn_input = self.rms_norm(
            &format!("{prefix}.ffn_norm"),
            after_attention,
            &format!("{prefix}.ffn_norm.weight"),
            spec.rms_norm_epsilon,
        )?;
        let gate = self.linear(
            &format!("{prefix}.ffn_gate"),
            ffn_input.clone(),
            &format!("{prefix}.ffn_gate.weight"),
            spec.intermediate_size,
            ParameterRole::FfnGate,
        )?;
        let up = self.linear(
            &format!("{prefix}.ffn_up"),
            ffn_input,
            &format!("{prefix}.ffn_up.weight"),
            spec.intermediate_size,
            ParameterRole::FfnUp,
        )?;
        let activated = self.elementwise_unary(
            &format!("{prefix}.ffn_silu"),
            gate,
            ElementwiseFunction::Silu,
        )?;
        let gated = self.binary(
            &format!("{prefix}.ffn_multiply"),
            OperationKind::Elementwise {
                function: ElementwiseFunction::Multiply,
            },
            activated,
            up,
        )?;
        let down = self.linear(
            &format!("{prefix}.ffn_down"),
            gated,
            &format!("{prefix}.ffn_down.weight"),
            spec.hidden_size,
            ParameterRole::FfnDown,
        )?;
        self.binary(
            &format!("{prefix}.ffn_residual"),
            OperationKind::Residual,
            ffn_residual,
            down,
        )
    }

    pub fn final_rms_norm(
        &mut self,
        operation_id: &str,
        input: Value,
        weight: &str,
        epsilon: f64,
    ) -> Result<Value> {
        self.rms_norm(operation_id, input, weight, epsilon)
    }

    pub fn output_projection(
        &mut self,
        operation_id: &str,
        input: Value,
        weight: &str,
        vocab_size: u32,
    ) -> Result<Value> {
        let slot = self.push_slot(weight, ParameterRole::Output, 2)?;
        self.unary(
            operation_id,
            OperationKind::OutputProjection,
            input,
            SemanticTensorType::activations(vocab_size),
            vec![slot],
        )
    }

    /// Mark the standard prefill and decode semantic exports.
    pub fn export_prefill_and_decode(&mut self, logits: Value) -> Result<()> {
        let input = self
            .primary_input
            .clone()
            .ok_or_else(|| DynInferError::internal("model has no token input"))?;
        self.exports.extend([
            ArchitectureExport {
                name: "prefill".into(),
                mode: ExecutionMode::Prefill,
                inputs: vec![input.clone()],
                outputs: vec![logits.id.clone()],
            },
            ArchitectureExport {
                name: "decode".into(),
                mode: ExecutionMode::Decode,
                inputs: vec![input],
                outputs: vec![logits.id],
            },
        ]);
        Ok(())
    }

    pub fn finish(&mut self) -> Result<ModelModule> {
        let architecture_id = self
            .architecture_id
            .take()
            .ok_or_else(|| DynInferError::internal("ModelBuilder missing architecture_id"))?;
        let graph = ArchitectureGraph {
            version: 1,
            architecture_id,
            values: std::mem::take(&mut self.values),
            operations: std::mem::take(&mut self.operations),
            parameter_slots: std::mem::take(&mut self.slots),
            exports: std::mem::take(&mut self.exports),
        };
        verify_architecture_conformance(&graph)?;
        Ok(ModelModule { graph })
    }
}

/// Validate graph identity, dataflow, parameter references, and exports.
pub fn verify_architecture_graph(graph: &ArchitectureGraph) -> Result<()> {
    if graph.version != 1 {
        return Err(DynInferError::internal(format!(
            "unsupported Architecture IR version {}",
            graph.version
        )));
    }
    let mut value_ids = BTreeSet::new();
    let value_types: BTreeMap<_, _> = graph
        .values
        .iter()
        .map(|value| (value.id.clone(), value.tensor_type.clone()))
        .collect();
    for value in &graph.values {
        if !value_ids.insert(value.id.clone()) {
            return Err(DynInferError::internal(format!(
                "duplicate graph value `{}`",
                value.id
            )));
        }
    }
    let mut slot_ids = BTreeSet::new();
    for slot in &graph.parameter_slots {
        if !slot_ids.insert(slot.id.clone()) {
            return Err(DynInferError::internal(format!(
                "duplicate parameter slot `{}`",
                slot.id
            )));
        }
    }
    let mut operation_ids = BTreeSet::new();
    let mut produced_values = BTreeSet::new();
    let mut consumed_slots = BTreeSet::new();
    for operation in &graph.operations {
        if !operation_ids.insert(operation.id.clone()) {
            return Err(DynInferError::internal(format!(
                "duplicate architecture operation `{}`",
                operation.id
            )));
        }
        for input in &operation.inputs {
            if !value_ids.contains(input) {
                return Err(DynInferError::internal(format!(
                    "operation `{}` references unknown input `{input}`",
                    operation.id
                )));
            }
        }
        for output in &operation.outputs {
            if !value_ids.contains(output) {
                return Err(DynInferError::internal(format!(
                    "operation `{}` references unknown output `{output}`",
                    operation.id
                )));
            }
            if !produced_values.insert(output.clone()) {
                return Err(DynInferError::internal(format!(
                    "graph value `{output}` has multiple producers"
                )));
            }
        }
        for parameter in &operation.parameters {
            if !slot_ids.contains(parameter) {
                return Err(DynInferError::internal(format!(
                    "operation `{}` references unknown parameter slot `{parameter}`",
                    operation.id
                )));
            }
            consumed_slots.insert(parameter.clone());
        }
        if matches!(operation.kind, OperationKind::Residual)
            || matches!(
                operation.kind,
                OperationKind::Elementwise {
                    function: ElementwiseFunction::Multiply
                }
            )
        {
            if operation.inputs.len() != 2
                || value_types.get(&operation.inputs[0]) != value_types.get(&operation.inputs[1])
            {
                return Err(DynInferError::internal(format!(
                    "binary operation `{}` requires two equal tensor types",
                    operation.id
                )));
            }
        }
    }
    for value in &graph.values {
        if !produced_values.contains(&value.id) {
            return Err(DynInferError::internal(format!(
                "graph value `{}` has no producer",
                value.id
            )));
        }
    }
    for slot in &graph.parameter_slots {
        if !slot.optional && !consumed_slots.contains(&slot.id) {
            return Err(DynInferError::internal(format!(
                "required parameter slot `{}` has no consuming operation",
                slot.id
            )));
        }
    }
    let mut export_names = BTreeSet::new();
    for export in &graph.exports {
        if !export_names.insert(export.name.as_str()) {
            return Err(DynInferError::internal(format!(
                "duplicate architecture export `{}`",
                export.name
            )));
        }
        if export.outputs.is_empty() {
            return Err(DynInferError::internal(format!(
                "architecture export `{}` has no output",
                export.name
            )));
        }
        for value in export.inputs.iter().chain(&export.outputs) {
            if !value_ids.contains(value) {
                return Err(DynInferError::internal(format!(
                    "architecture export `{}` references unknown value `{value}`",
                    export.name
                )));
            }
        }
    }
    Ok(())
}

/// Conformance checks shared by every compiled-in architecture.
pub fn verify_architecture_conformance(graph: &ArchitectureGraph) -> Result<()> {
    verify_architecture_graph(graph)?;
    for (name, mode) in [
        ("prefill", ExecutionMode::Prefill),
        ("decode", ExecutionMode::Decode),
    ] {
        if !graph
            .exports
            .iter()
            .any(|export| export.name == name && export.mode == mode)
        {
            return Err(DynInferError::internal(format!(
                "architecture `{}` is missing its `{name}` export",
                graph.architecture_id
            )));
        }
    }
    Ok(())
}

/// Validate a typed graph against a small canonical parameter fixture.
///
/// Architecture tests use this before the binder exists in the dependency
/// graph. It proves that required slots have canonical fixture matches and
/// compatible ranks without introducing an architecture -> binding cycle.
pub fn verify_architecture_catalog_conformance(
    graph: &ArchitectureGraph,
    catalog: &ParameterCatalog,
) -> Result<()> {
    verify_architecture_conformance(graph)?;
    let parameters: BTreeMap<_, _> = catalog
        .parameters
        .iter()
        .map(|parameter| (parameter.canonical_name.as_str(), parameter))
        .collect();
    for slot in &graph.parameter_slots {
        let Some(parameter) = parameters.get(slot.canonical_name.as_str()) else {
            if slot.optional {
                continue;
            }
            return Err(DynInferError::internal(format!(
                "architecture fixture is missing canonical parameter `{}`",
                slot.canonical_name
            )));
        };
        if let Some(rank) = slot.expected_type.rank {
            if parameter.logical_type.shape.rank() != rank {
                return Err(DynInferError::internal(format!(
                    "architecture fixture parameter `{}` has rank {}, expected {rank}",
                    slot.canonical_name,
                    parameter.logical_type.shape.rank()
                )));
            }
        }
    }
    Ok(())
}

/// Compiled-in architecture definition.
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
    fn canonicalize_param(&self, key: &str) -> Option<String> {
        Some(key.to_string())
    }

    /// Post-process the parameter catalog (tied embeddings, drop unused keys, …).
    fn sanitize_catalog(&self, _catalog: &mut ParameterCatalog) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_block_expands_to_typed_operations() {
        let mut builder = ModelBuilder::new().unwrap();
        builder.set_architecture_id(ArchitectureId::new("test.decoder"));
        let tokens = builder.input_tokens("tokens").unwrap();
        let hidden = builder
            .embedding("token_embedding", tokens, "token_embd.weight", 64)
            .unwrap();
        let output = builder
            .decoder_block(
                hidden,
                0,
                true,
                &DecoderBlockSpec {
                    hidden_size: 64,
                    intermediate_size: 128,
                    num_heads: 4,
                    num_kv_heads: 2,
                    head_dim: 16,
                    rms_norm_epsilon: 1e-6,
                    rope_theta: Some(10_000.0),
                },
            )
            .unwrap();
        let output = builder
            .final_rms_norm("output_norm", output, "output_norm.weight", 1e-6)
            .unwrap();
        let logits = builder
            .output_projection("output_projection", output, "output.weight", 32)
            .unwrap();
        builder.export_prefill_and_decode(logits).unwrap();
        let module = builder.finish().unwrap();

        assert!(module.graph.operations.iter().any(|op| matches!(
            op.kind,
            OperationKind::Attention {
                num_heads: 4,
                num_kv_heads: 2,
                ..
            }
        )));
        assert!(module.graph.operations.iter().any(|op| matches!(
            op.kind,
            OperationKind::KvCacheWrite {
                component: KvCacheComponent::Key,
                ..
            }
        )));
    }
}
