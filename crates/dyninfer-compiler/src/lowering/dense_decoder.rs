//! Shared dense causal-decoder MLIR emitter (ops library).
//!
//! Architecture files (`models/*`) configure [`DenseDecoderConfig`] and call
//! [`emit_dense_decoder_cfg`]. Supports MHA/GQA, independent `head_dim`, optional
//! Q/K RMSNorm, and HuggingFace Llama/Qwen-style (`rotate_half`) RoPE.
//!
//! Weight globals always use the per-tensor dtype from the checkpoint catalog.
//! Activation and accumulator types come from the selected kernel records;
//! narrower float weights are converted explicitly at operation boundaries.
//!
//! # Prefill / KV sizes
//!
//! `seq` / `max_kv` are **compile-time specialization shapes** baked into the
//! MLIR (static `tensor<Sx…>` types). They are not runtime generate knobs —
//! changing them requires a recompile. Defaults below are PoC-sized so CPU
//! compile + e2e stays tractable; override via config
//! (`--set prefill_window=N --set max_kv=M` or CLI `--prefill-window` /
//! `--max-kv` on `generate`).
//!
//! # Emission style
//!
//! Large blocks still use `FuncBuilder::op_asm` for region-heavy `linalg`
//! ops. That is assembly buffered then verified — same abstraction level as
//! melior bindings, **not** a zml-like tensor DSL. Follow-up is higher-level
//! helpers (`attn`, `rope`, …), not switching IR frontends.

use super::{
    kernels,
    parameter::{ParameterLowerings, default_parameter_lowerings},
};
use dyninfer_core::{
    BoundModel, ExecutionMode, OperationKind, ParameterBinding, PhysicalEncoding, ScalarType,
    StorageElementType,
};
use dyninfer_error::Result;
use dyninfer_mlir::{FuncBuilder, ModuleBuilder};
use std::collections::BTreeMap;

/// Default static prefill window for mid-size dense models (e.g. Maykeye 64-d).
pub const PREFILL_WINDOW: u32 = 64;
/// Smaller window for large models (Qwen3-0.6B) to keep compile/runtime tractable.
pub const LARGE_PREFILL_WINDOW: u32 = 32;
/// Window used by the synthetic Milestone-1 fixture (fast differential tests).
pub const TINY_PREFILL_WINDOW: u32 = 4;

/// Default mutable KV capacity (decode can grow past the prefill window).
pub const PREFILL_MAX_KV: u32 = 128;
pub const LARGE_MAX_KV: u32 = 256;
/// Matches [`TINY_PREFILL_WINDOW`]: decode-at-last fills a dense softmax domain.
/// Larger models pass a host `attn_bias` for padded KV slots.
pub const TINY_MAX_KV: u32 = 4;
pub const PAGED_KV_PAGE_SIZE: u32 = 256;
/// Prefill specializes a static chunk width. Values >= 1024 currently corrupt
/// short-prompt greedy decode on HIP (flash and portable); 512 stays correct
/// and long prompts are covered by multiple chunks.
pub const PAGED_PREFILL_CHUNK_SIZE: u32 = 512;
/// Vulkan portable attention + SPIR-V promotion is far more sensitive to chunk
/// width than HIP/CPU. Keep a smaller specialized width so score/mask tiles
/// stay within the 32 KiB workgroup-memory budget and avoid the garbage-token
/// loops seen at 512.
pub const PAGED_PREFILL_CHUNK_SIZE_VULKAN: u32 = 64;

/// Prefill chunk width for the active target.
pub fn paged_prefill_chunk_size(driver: &str) -> u32 {
    if driver == "vulkan" {
        PAGED_PREFILL_CHUNK_SIZE_VULKAN
    } else {
        PAGED_PREFILL_CHUNK_SIZE
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseDecoderConfig {
    pub vocab: u32,
    pub hidden: u32,
    pub intermediate: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub num_layers: u32,
    /// Static prefill token window (prompt bucket).
    pub seq: u32,
    /// Mutable KV capacity (`>= seq`); decode positions use `[0, max_kv)`.
    pub max_kv: u32,
    /// Runtime-owned paged KV shared by split prefill/decode modules (ABI v6).
    pub paged_kv: bool,
    /// HIP uses IREE's fused online-attention op with WMMA flash configs;
    /// Vulkan/CPU keep the portable page-local linalg fallback (SPIR-V cannot
    /// lower `LLVMGPUVectorDistribute` / `WMMAR3_*`).
    pub iree_flash_attention: bool,
    pub rms_norm_eps: f32,
    pub rope_theta: Option<f32>,
    /// Qwen3-style RMSNorm on Q/K heads before RoPE.
    pub has_qk_norm: bool,
    /// Canonical slot name → tensor key in the parameter file (often HF names).
    pub param_keys: BTreeMap<String, String>,
    /// Canonical slot name → on-disk scalar dtype from the checkpoint catalog.
    pub param_dtypes: BTreeMap<String, ScalarType>,
    /// Canonical slot name → selected operation-local activation dtype.
    pub param_compute_dtypes: BTreeMap<String, ScalarType>,
    /// Complete bound parameter/component descriptors for encoding-specific
    /// compiler-owned lowerings.
    pub param_bindings: BTreeMap<String, ParameterBinding>,
    /// Canonical slot + mode → lowering selected by strict coverage.
    pub param_lowerings: BTreeMap<(String, ExecutionMode), String>,
    /// Vulkan SPIR-V in the pinned IREE revision cannot legalize BF16/F16
    /// widening when it is fused into a contraction. Keep that conversion in
    /// its own device dispatch; other backends retain the fused path.
    pub separate_storage_casts: bool,
    /// Vulkan fused paged modules cannot legalize `vector.step` from
    /// `linalg.index` in the causal mask; materialize positions as dense
    /// constants instead. HIP must not use dense constants (breaks codegen).
    pub dense_index_constants: bool,
}

impl DenseDecoderConfig {
    pub fn q_dim(&self) -> u32 {
        self.num_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }

    pub fn gqa_group(&self) -> u32 {
        self.num_heads / self.num_kv_heads.max(1)
    }

    pub fn from_bound_model(bound: &BoundModel) -> Self {
        let cfg = &bound.resolved_config;
        let u = |keys: &[&str], default: u32| -> u32 {
            for k in keys {
                if let Some(v) = cfg.get(*k).and_then(|v| v.as_u64()) {
                    return v as u32;
                }
            }
            default
        };
        let f = |keys: &[&str], default: f32| -> f32 {
            for k in keys {
                if let Some(v) = cfg.get(*k).and_then(|v| v.as_f64()) {
                    return v as f32;
                }
            }
            default
        };
        let hidden = u(
            // HF config.json keys, then GGUF metadata aliases (`llama.*` is the
            // GGUF namespace even when the arch file is Qwen — not Llama-specific).
            &["hidden_size", "llama.embedding_length"],
            64,
        );
        let num_heads = u(&["num_heads", "llama.attention.head_count"], 4);
        let num_kv_heads = u(
            &["num_kv_heads", "llama.attention.head_count_kv"],
            num_heads,
        );
        let vocab = u(&["vocab_size", "llama.vocab_size"], 32);
        let num_layers = u(
            &["num_layers", "n_layer", "block_count", "llama.block_count"],
            1,
        );
        let head_dim = u(&["head_dim"], hidden / num_heads.max(1));
        let is_synthetic = vocab == 32 && hidden == 64 && num_heads == 4;
        let seq = bound
            .execution_shapes
            .iter()
            .find(|shape| shape.mode == ExecutionMode::Prefill)
            .map(|shape| shape.sequence_length)
            .unwrap_or(1);
        let max_kv = bound
            .execution_shapes
            .iter()
            .map(|shape| shape.max_kv_length)
            .max()
            .unwrap_or(seq);
        let rope_theta = bound
            .architecture
            .operations
            .iter()
            .find_map(|operation| match operation.kind {
                OperationKind::Rope { theta, .. } => Some(theta as f32),
                _ => None,
            });
        let has_qk_norm = bound
            .architecture
            .operations
            .iter()
            .any(|operation| matches!(operation.kind, OperationKind::PerHeadRmsNorm { .. }));
        let paged_kv = bound.architecture.operations.iter().any(|operation| {
            matches!(operation.kind, OperationKind::Attention { .. })
                && bound.selected_kernels.iter().any(|selected| {
                    selected.operation_id == operation.id
                        && selected.mode == ExecutionMode::Prefill
                        && selected.lowering_id.as_str() == "attention.online_paged.generated"
                })
        });
        let mut param_keys = BTreeMap::new();
        let mut param_dtypes = BTreeMap::new();
        let mut param_compute_dtypes = BTreeMap::new();
        let mut param_bindings = BTreeMap::new();
        let mut param_lowerings = BTreeMap::new();
        for p in &bound.binding.bindings {
            let key = p
                .components
                .first()
                .map(|c| c.external_key.clone())
                .unwrap_or_else(|| p.canonical_name.to_string());
            param_keys.insert(p.canonical_name.to_string(), key);
            if let Some(ty) = bound_param_dtype(p) {
                param_dtypes.insert(p.canonical_name.to_string(), ty);
            }
            param_bindings.insert(p.canonical_name.to_string(), p.clone());
        }
        for selected in &bound.selected_kernels {
            for slot in &selected.parameter_slots {
                if let Some(binding) = bound
                    .binding
                    .bindings
                    .iter()
                    .find(|binding| &binding.slot_id == slot)
                {
                    param_compute_dtypes
                        .insert(binding.canonical_name.to_string(), selected.activation_type);
                    param_lowerings.insert(
                        (binding.canonical_name.to_string(), selected.mode),
                        selected.lowering_id.to_string(),
                    );
                }
            }
        }
        Self {
            vocab,
            hidden,
            intermediate: u(
                &["intermediate_size", "llama.feed_forward_length"],
                hidden * 2,
            ),
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            seq,
            max_kv: max_kv.max(seq),
            paged_kv,
            // IREE online attention (`iree_linalg_ext.online_attention`) is
            // emitted only for HIP: the flash configs hardcode
            // `LLVMGPUVectorDistribute` + `WMMAR3_*` MMA layouts that SPIR-V
            // / Vulkan cannot lower. Other backends use portable page-local
            // linalg attention (`emit_online_attention_page`).
            iree_flash_attention: bound.target.driver == "hip",
            rms_norm_eps: f(
                &["rms_norm_eps", "llama.attention.layer_norm_rms_epsilon"],
                1e-5,
            ),
            // The typed graph decides whether RoPE exists and carries theta.
            rope_theta: if is_synthetic {
                None
            } else if rope_theta.is_some_and(|t| t <= 0.0) {
                None
            } else {
                rope_theta.or(Some(10000.0))
            },
            has_qk_norm,
            param_keys,
            param_dtypes,
            param_compute_dtypes,
            param_bindings,
            param_lowerings,
            // Vulkan requires separate BF16/F16 casts and dense causal-mask
            // indices (`vector.step` / fused bitcast fail otherwise).
            separate_storage_casts: bound.target.driver == "vulkan",
            dense_index_constants: bound.target.driver == "vulkan",
        }
    }

    pub(super) fn param_key(&self, canonical: &str) -> String {
        self.param_keys
            .get(canonical)
            .cloned()
            .unwrap_or_else(|| canonical.to_string())
    }

    /// On-disk dtype for a parameter. Always sourced from the checkpoint catalog
    /// when present; synthetic fixtures without catalog entries default to f32.
    pub(super) fn param_dtype(&self, canonical: &str) -> ScalarType {
        self.param_dtypes
            .get(canonical)
            .copied()
            .unwrap_or(ScalarType::F32)
    }

    pub(super) fn param_compute_dtype(&self, canonical: &str) -> ScalarType {
        self.param_compute_dtypes
            .get(canonical)
            .copied()
            .unwrap_or(ScalarType::F32)
    }

    pub(super) fn param_binding(&self, canonical: &str) -> Option<&ParameterBinding> {
        self.param_bindings.get(canonical)
    }

    pub(super) fn param_lowering(&self, canonical: &str, mode: ExecutionMode) -> Option<&str> {
        self.param_lowerings
            .get(&(canonical.to_string(), mode))
            .map(String::as_str)
    }

    /// True for the synthetic Milestone-1 fixture (`tiny_llama_dense_f32`):
    /// vocab=32, hidden=64, 1 layer, seq=4, no RoPE — used by differential e2e.
    pub fn is_synthetic_fixture(&self) -> bool {
        self.vocab == 32
            && self.hidden == 64
            && self.intermediate == 128
            && self.num_heads == 4
            && self.head_dim == 16
            && self.num_layers == 1
            && self.seq == TINY_PREFILL_WINDOW
            && self.rope_theta.is_none()
    }

    pub fn supports_dense_emit(&self) -> bool {
        self.num_heads > 0
            && self.num_kv_heads > 0
            && self.num_heads % self.num_kv_heads == 0
            && self.head_dim > 0
            && self.head_dim % 2 == 0
            && self.q_dim() > 0
            && self.kv_dim() > 0
            && self.intermediate > 0
            && self.num_layers >= 1
            && self.num_layers <= 32
            && self.vocab >= 1
            && self.vocab <= 160_000
            && self.hidden <= 2048
            && self.head_dim <= 256
            && self.q_dim() <= 4096
            && self.seq >= 1
            && self.seq
                <= if self.paged_kv {
                    PAGED_PREFILL_CHUNK_SIZE
                } else {
                    256
                }
            && self.max_kv >= self.seq
            && self.max_kv <= if self.paged_kv { 1_048_576 } else { 512 }
    }
}

/// Emit using an explicit config (architecture files may override flags).
///
/// Builds an in-memory MLIR module via [`ModuleBuilder`], verifies, then prints
/// for the IREE compile boundary (spec §8.3.1).
pub fn emit_dense_decoder_cfg(arch_id: &str, c: &DenseDecoderConfig) -> Result<String> {
    emit_dense_decoder_cfg_program(arch_id, c, PagedProgram::Combined)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedProgram {
    Combined,
    Prefill,
    Decode,
}

pub fn emit_dense_decoder_cfg_program(
    arch_id: &str,
    c: &DenseDecoderConfig,
    program: PagedProgram,
) -> Result<String> {
    assert!(
        c.supports_dense_emit(),
        "unsupported dense decoder emit config: {c:?}"
    );
    let mut builder = match program {
        PagedProgram::Combined => ModuleBuilder::new()?,
        PagedProgram::Prefill => ModuleBuilder::new_named("prefill")?,
        PagedProgram::Decode => ModuleBuilder::new_named("decode")?,
    };
    build_dense_decoder_program(&mut builder, arch_id, c, program)?;
    Ok(builder.finish()?.mlir_text)
}

fn build_dense_decoder_program(
    builder: &mut ModuleBuilder,
    arch_id: &str,
    c: &DenseDecoderConfig,
    program: PagedProgram,
) -> Result<()> {
    let _ = arch_id; // retained for call-site tracing / future module attrs
    emit_globals(builder, c, program)?;
    emit_helpers(builder, c, program)?;
    if c.paged_kv {
        if matches!(program, PagedProgram::Combined | PagedProgram::Prefill) {
            emit_paged_decoder(builder, c, PagedVariant { decode: false })?;
        }
        if matches!(program, PagedProgram::Combined | PagedProgram::Decode) {
            let mut decode = c.clone();
            decode.seq = 1;
            emit_paged_decoder(builder, &decode, PagedVariant { decode: true })?;
        }
    } else {
        emit_prefill(builder, c)?;
        emit_decode(builder, c)?;
    }
    kernels::emit_add_smoke(builder)?;
    Ok(())
}

#[derive(Clone, Copy)]
struct PagedVariant {
    decode: bool,
}

impl PagedVariant {
    fn function(self, base: &str) -> String {
        if self.decode {
            format!("decode_{base}")
        } else {
            base.into()
        }
    }

    fn helper(self, base: &str) -> String {
        if self.decode {
            format!("{base}_tok")
        } else {
            base.into()
        }
    }

    fn global(self, base: &str) -> String {
        if self.decode {
            format!("paged_decode_{base}")
        } else {
            format!("paged_{base}")
        }
    }

    fn mode(self) -> ExecutionMode {
        if self.decode {
            ExecutionMode::Decode
        } else {
            ExecutionMode::Prefill
        }
    }
}

fn emit_paged_decoder(
    module: &mut ModuleBuilder,
    c: &DenseDecoderConfig,
    variant: PagedVariant,
) -> Result<()> {
    emit_paged_chunk_begin(module, c, variant)?;
    emit_paged_layer_page(module, c, variant)?;
    for layer in 0..c.num_layers {
        emit_paged_layer_prepare(module, c, layer, variant)?;
        emit_paged_layer_finish(module, c, layer, variant)?;
    }
    emit_paged_chunk_logits(module, c, variant)?;
    emit_paged_fused_chunk(module, c, variant)
}

fn emit_paged_fused_chunk(
    module: &mut ModuleBuilder,
    c: &DenseDecoderConfig,
    variant: PagedVariant,
) -> Result<()> {
    let (s, v, layers, page, nkv, g, d) = (
        c.seq,
        c.vocab,
        c.num_layers,
        PAGED_KV_PAGE_SIZE,
        c.num_kv_heads,
        c.gqa_group(),
        c.head_dim,
    );
    // Fixed page arity keeps HIP on the proven tensor ABI (list<!hal.buffer_view>
    // import/alias faults on current ROCm). Host always passes ceil(max_kv/page).
    let num_pages = c.max_kv.div_ceil(page).max(1);
    let page_ty = format!("tensor<{layers}x2x{page}x{nkv}x{d}xf32>");
    let qg_ty = format!("tensor<{nkv}x{g}x{s}x{d}xf32>");
    let kv3_ty = format!("tensor<{s}x{nkv}x{d}xf32>");
    let row_ty = format!("tensor<{nkv}x{g}x{s}xf32>");
    let entry = if variant.decode {
        "decode_chunk"
    } else {
        "prefill_chunk"
    };
    let begin = variant.function("chunk_begin");
    let layer_page = variant.function("layer_page");
    let logits_fn = variant.function("chunk_logits");
    let mut f = module.func(entry);
    f.arg("tokens", format!("tensor<{s}xi64>"));
    f.arg("last", "tensor<i64>");
    f.arg("start_pos", "tensor<i64>");
    for pi in 0..num_pages {
        f.arg(format!("page{pi}"), &page_ty);
    }
    f.result_ty(format!("tensor<{v}xf32>"));
    for _pi in 0..num_pages {
        f.result_ty(&page_ty);
    }
    f.op_asm(format!(
        "  func.call @{begin}(%tokens, %last, %start_pos) : (tensor<{s}xi64>, tensor<i64>, tensor<i64>) -> ()\n"
    ));
    let mut page_ssas: Vec<String> = (0..num_pages).map(|pi| format!("page{pi}")).collect();
    for layer in 0..layers {
        let prepare = variant.function(&format!("layer_prepare_{layer}"));
        let finish = variant.function(&format!("layer_finish_{layer}"));
        // Return query+chunk KV as SSA: util.global across outlined private
        // calls is unreliable for these tensors (same class as attn out/max/sum).
        f.op_asm(format!(
            r#"  %query{layer}, %chunk_k{layer}, %chunk_v{layer}, %out_l{layer}_0, %max_l{layer}_0, %sum_l{layer}_0 = func.call @{prepare}() : () -> ({qg_ty}, {kv3_ty}, {kv3_ty}, {qg_ty}, {row_ty}, {row_ty})
  %layer{layer}_i64 = arith.constant {layer} : i64
  %layer{layer}_e = tensor.empty() : tensor<i64>
  %layer{layer}_t = tensor.insert %layer{layer}_i64 into %layer{layer}_e[] : tensor<i64>
"#
        ));
        let mut out_ssa = format!("out_l{layer}_0");
        let mut max_ssa = format!("max_l{layer}_0");
        let mut sum_ssa = format!("sum_l{layer}_0");
        let mut next_pages = Vec::with_capacity(num_pages as usize);
        for pi in 0..num_pages {
            let page_in = &page_ssas[pi as usize];
            let page_out = format!("page_l{layer}_p{pi}");
            let out_next = format!("out_l{layer}_p{pi}");
            let max_next = format!("max_l{layer}_p{pi}");
            let sum_next = format!("sum_l{layer}_p{pi}");
            f.op_asm(format!(
                r#"  %pi{layer}_{pi}_i64 = arith.constant {pi} : i64
  %pi{layer}_{pi}_e = tensor.empty() : tensor<i64>
  %pi{layer}_{pi}_t = tensor.insert %pi{layer}_{pi}_i64 into %pi{layer}_{pi}_e[] : tensor<i64>
  %{page_out}, %{out_next}, %{max_next}, %{sum_next} = func.call @{layer_page}(%{page_in}, %pi{layer}_{pi}_t, %start_pos, %layer{layer}_t, %query{layer}, %chunk_k{layer}, %chunk_v{layer}, %{out_ssa}, %{max_ssa}, %{sum_ssa}) : ({page_ty}, tensor<i64>, tensor<i64>, tensor<i64>, {qg_ty}, {kv3_ty}, {kv3_ty}, {qg_ty}, {row_ty}, {row_ty}) -> ({page_ty}, {qg_ty}, {row_ty}, {row_ty})
"#
            ));
            next_pages.push(page_out);
            out_ssa = out_next;
            max_ssa = max_next;
            sum_ssa = sum_next;
        }
        page_ssas = next_pages;
        f.op_asm(format!(
            r#"  func.call @{finish}(%{out_ssa}, %{sum_ssa}) : ({qg_ty}, {row_ty}) -> ()
"#
        ));
    }
    let page_rets = page_ssas
        .iter()
        .map(|s| format!("%{s}"))
        .collect::<Vec<_>>()
        .join(", ");
    let page_tys = std::iter::repeat(page_ty.as_str())
        .take(num_pages as usize)
        .collect::<Vec<_>>()
        .join(", ");
    f.op_asm(format!(
        "  %logits = func.call @{logits_fn}(%last) : (tensor<i64>) -> tensor<{v}xf32>\n  return %logits, {page_rets} : tensor<{v}xf32>, {page_tys}"
    ));
    f.finish(module)
}

fn emit_paged_chunk_begin(
    module: &mut ModuleBuilder,
    c: &DenseDecoderConfig,
    variant: PagedVariant,
) -> Result<()> {
    let parameters = default_parameter_lowerings();
    let (s, h, v) = (c.seq, c.hidden, c.vocab);
    let hidden_ty = format!("tensor<{s}x{h}xf32>");
    let mut f = module.func_private(&variant.function("chunk_begin"));
    f.arg("tokens", format!("tensor<{s}xi64>"));
    f.arg("last", "tensor<i64>");
    f.arg("start_pos", "tensor<i64>");
    emit_load_compute(
        &mut f,
        c,
        "emb_t",
        "token_embd_weight",
        "token_embd.weight",
        &format!("{v}x{h}"),
    );
    f.op_asm(format!("  %h_acc0 = tensor.empty() : {hidden_ty}\n"));
    let plain_embedding = c
        .param_binding("token_embd.weight")
        .is_none_or(|binding| matches!(binding.encoding, PhysicalEncoding::Plain { .. }));
    // Vulkan cannot lower iree_linalg_ext.gather inside the fused chunk; HIP/CPU
    // keep the gather path. Quantized embeddings always use the loop.
    let use_gather = plain_embedding && !c.separate_storage_casts;
    if use_gather {
        f.op_asm(format!(
            "  %h_acc{s} = iree_linalg_ext.gather dimension_map = [0] ins(%emb_t, %tokens : tensor<{v}x{h}xf32>, tensor<{s}xi64>) outs(%h_acc0 : {hidden_ty}) -> {hidden_ty}\n"
        ));
    } else {
        f.op_asm(format!(
            "  %c0i = arith.constant 0 : index\n  %c1i = arith.constant 1 : index\n  %csi = arith.constant {s} : index\n  %h_acc{s} = scf.for %p = %c0i to %csi step %c1i iter_args(%acc = %h_acc0) -> ({hidden_ty}) {{\n  %t = tensor.extract %tokens[%p] : tensor<{s}xi64>\n  %i = arith.index_cast %t : i64 to index\n"
        ));
        if plain_embedding {
            f.op_asm(format!(
                "  %r = tensor.extract_slice %emb_t[%i, 0] [1, {h}] [1, 1] : tensor<{v}x{h}xf32> to tensor<1x{h}xf32>\n"
            ));
        } else if !parameters.emit_embedding_call(&mut f, c, "r", "i", "emb_t", variant.mode())?
        {
            unreachable!("plain embeddings use gather or extract_slice");
        }
        f.op_asm(format!(
            "  %next = tensor.insert_slice %r into %acc[%p, 0] [1, {h}] [1, 1] : tensor<1x{h}xf32> into {hidden_ty}\n  scf.yield %next : {hidden_ty}\n  }}\n"
        ));
    }
    f.op_asm(format!(
        "  %last64 = tensor.extract %last[] : tensor<i64>\n  %one64 = arith.constant 1 : i64\n  %valid64 = arith.addi %last64, %one64 : i64\n  %valid_e = tensor.empty() : tensor<i64>\n  %valid = tensor.insert %valid64 into %valid_e[] : tensor<i64>\n  util.global.store %h_acc{s}, @{hidden_global} : {hidden_ty}\n  util.global.store %start_pos, @{start_global} : tensor<i64>\n  util.global.store %valid, @{valid_global} : tensor<i64>\n  return",
        hidden_global = variant.global("hidden"),
        start_global = variant.global("start_pos"),
        valid_global = variant.global("valid_count"),
    ));
    f.finish(module)
}

fn emit_paged_layer_prepare(
    module: &mut ModuleBuilder,
    c: &DenseDecoderConfig,
    layer: u32,
    variant: PagedVariant,
) -> Result<()> {
    let (s, h, q, kv, nkv, g, d) = (
        c.seq,
        c.hidden,
        c.q_dim(),
        c.kv_dim(),
        c.num_kv_heads,
        c.gqa_group(),
        c.head_dim,
    );
    let p = format!("blk{layer}");
    let n = format!("blk.{layer}");
    let hidden_ty = format!("tensor<{s}x{h}xf32>");
    let qg_ty = format!("tensor<{nkv}x{g}x{s}x{d}xf32>");
    let kv3_ty = format!("tensor<{s}x{nkv}x{d}xf32>");
    let row_ty = format!("tensor<{nkv}x{g}x{s}xf32>");
    let rms_norm = variant.helper("rms_norm");
    let linear_hq = variant.helper("linear_hq");
    let linear_hkv = variant.helper("linear_hkv");
    let prepare_attention = variant.helper("prepare_paged_attention");
    let mut f = module.func_private(&variant.function(&format!("layer_prepare_{layer}")));
    f.result_ty(&qg_ty);
    f.result_ty(&kv3_ty);
    f.result_ty(&kv3_ty);
    f.result_ty(&qg_ty);
    f.result_ty(&row_ty);
    f.result_ty(&row_ty);
    emit_load_compute(
        &mut f,
        c,
        "attn_nw",
        &format!("{p}_attn_norm_weight"),
        &format!("{n}.attn_norm.weight"),
        &format!("{h}"),
    );
    for (ssa, sym, canonical, shape) in [
        (
            "wq",
            format!("{p}_attn_q_weight"),
            format!("{n}.attn_q.weight"),
            format!("{q}x{h}"),
        ),
        (
            "wk",
            format!("{p}_attn_k_weight"),
            format!("{n}.attn_k.weight"),
            format!("{kv}x{h}"),
        ),
        (
            "wv",
            format!("{p}_attn_v_weight"),
            format!("{n}.attn_v.weight"),
            format!("{kv}x{h}"),
        ),
    ] {
        emit_load_compute(&mut f, c, ssa, &sym, &canonical, &shape);
    }
    if c.has_qk_norm {
        emit_load_compute(
            &mut f,
            c,
            "qnw",
            &format!("{p}_attn_q_norm_weight"),
            &format!("{n}.attn_q_norm.weight"),
            &format!("{d}"),
        );
        emit_load_compute(
            &mut f,
            c,
            "knw",
            &format!("{p}_attn_k_norm_weight"),
            &format!("{n}.attn_k_norm.weight"),
            &format!("{d}"),
        );
    }
    f.op_asm(format!(
        "  %hidden = util.global.load @{hidden_global} : {hidden_ty}\n  %start_pos = util.global.load @{start_global} : tensor<i64>\n  %xn = func.call @{rms_norm}(%hidden, %attn_nw) : ({hidden_ty}, tensor<{h}xf32>) -> {hidden_ty}\n",
        hidden_global = variant.global("hidden"),
        start_global = variant.global("start_pos"),
    ));
    emit_linear_call(
        &mut f,
        c,
        "q",
        &linear_hq,
        "xn",
        "wq",
        &format!("{n}.attn_q.weight"),
        s,
        h,
        q,
    );
    emit_linear_call(
        &mut f,
        c,
        "k",
        &linear_hkv,
        "xn",
        "wk",
        &format!("{n}.attn_k.weight"),
        s,
        h,
        kv,
    );
    emit_linear_call(
        &mut f,
        c,
        "v",
        &linear_hkv,
        "xn",
        "wv",
        &format!("{n}.attn_v.weight"),
        s,
        h,
        kv,
    );
    let norm_args = if c.has_qk_norm { ", %qnw, %knw" } else { "" };
    let norm_types = if c.has_qk_norm {
        format!(", tensor<{d}xf32>, tensor<{d}xf32>")
    } else {
        String::new()
    };
    f.op_asm(format!(
        "  %qg, %kc, %vc = func.call @{prepare_attention}(%q, %k, %v, %start_pos{norm_args}) : (tensor<{s}x{q}xf32>, tensor<{s}x{kv}xf32>, tensor<{s}x{kv}xf32>, tensor<i64>{norm_types}) -> ({qg_ty}, {kv3_ty}, {kv3_ty})\n"
    ));
    f.op_asm(format!(
        r#"  %zero = arith.constant 0.0 : f32
  %neg = arith.constant -3.40282347E+38 : f32
  %out_e = tensor.empty() : {qg_ty}
  %row_e = tensor.empty() : {row_ty}
  %out = linalg.fill ins(%zero : f32) outs(%out_e : {qg_ty}) -> {qg_ty}
  %max = linalg.fill ins(%neg : f32) outs(%row_e : {row_ty}) -> {row_ty}
  %sum = linalg.fill ins(%zero : f32) outs(%row_e : {row_ty}) -> {row_ty}
  return %qg, %kc, %vc, %out, %max, %sum : {qg_ty}, {kv3_ty}, {kv3_ty}, {qg_ty}, {row_ty}, {row_ty}"#
    ));
    f.finish(module)
}

fn emit_paged_layer_page(
    module: &mut ModuleBuilder,
    c: &DenseDecoderConfig,
    variant: PagedVariant,
) -> Result<()> {
    let (s, nkv, g, d, layers, page) = (
        c.seq,
        c.num_kv_heads,
        c.gqa_group(),
        c.head_dim,
        c.num_layers,
        PAGED_KV_PAGE_SIZE,
    );
    let qg_ty = format!("tensor<{nkv}x{g}x{s}x{d}xf32>");
    let kv3_ty = format!("tensor<{s}x{nkv}x{d}xf32>");
    let row_ty = format!("tensor<{nkv}x{g}x{s}xf32>");
    let mask_ty = format!("tensor<{nkv}x{g}x{s}x{page}xf32>");
    let page_ty = format!("tensor<{layers}x2x{page}x{nkv}x{d}xf32>");
    let expanded_ty = format!("tensor<1x1x{page}x{nkv}x{d}xf32>");
    let page_slice_ty = format!("tensor<1x1x{page}x{nkv}x{d}xf32>");
    let page_seq_ty = format!("tensor<{page}x{nkv}x{d}xf32>");
    let page_head_ty = format!("tensor<{nkv}x{page}x{d}xf32>");
    let scale_ty = if c.iree_flash_attention { "f16" } else { "f32" };
    let mask_helper = variant.helper("paged_causal_mask");
    let attention_helper = variant.helper("online_attention_page");
    let mut f = module.func_private(&variant.function("layer_page"));
    f.arg("page", &page_ty);
    f.arg("page_index", "tensor<i64>");
    f.arg("start_pos", "tensor<i64>");
    f.arg("layer_index", "tensor<i64>");
    f.arg("query", &qg_ty);
    f.arg("chunk_k", &kv3_ty);
    f.arg("chunk_v", &kv3_ty);
    f.arg("out", &qg_ty);
    f.arg("max", &row_ty);
    f.arg("sum", &row_ty);
    f.result_ty(&page_ty);
    f.result_ty(&qg_ty);
    f.result_ty(&row_ty);
    f.result_ty(&row_ty);
    f.op_asm(format!(
        r#"  %pi64 = tensor.extract %page_index[] : tensor<i64>
  %start64 = tensor.extract %start_pos[] : tensor<i64>
  %layer64 = tensor.extract %layer_index[] : tensor<i64>
  %pi = arith.index_cast %pi64 : i64 to index
  %start = arith.index_cast %start64 : i64 to index
  %layer = arith.index_cast %layer64 : i64 to index
  %page_size = arith.constant {page} : index
  %page_start = arith.muli %pi, %page_size : index
  %page_end = arith.addi %page_start, %page_size : index
  %valid_t = util.global.load @{valid_global} : tensor<i64>
  %valid64 = tensor.extract %valid_t[] : tensor<i64>
  %valid = arith.index_cast %valid64 : i64 to index
  %chunk_end = arith.addi %start, %valid : index
  %starts_before_chunk_end = arith.cmpi ult, %page_start, %chunk_end : index
  %ends_after_chunk_start = arith.cmpi ugt, %page_end, %start : index
  %intersects = arith.andi %starts_before_chunk_end, %ends_after_chunk_start : i1
  %page_updated = scf.if %intersects -> ({page_ty}) {{
    %old_ks = tensor.extract_slice %page[%layer, 0, 0, 0, 0] [1, 1, {page}, {nkv}, {d}] [1, 1, 1, 1, 1] : {page_ty} to {page_slice_ty}
    %old_vs = tensor.extract_slice %page[%layer, 1, 0, 0, 0] [1, 1, {page}, {nkv}, {d}] [1, 1, 1, 1, 1] : {page_ty} to {page_slice_ty}
    %old_k = tensor.collapse_shape %old_ks [[0, 1, 2], [3], [4]] : {page_slice_ty} into {page_seq_ty}
    %old_v = tensor.collapse_shape %old_vs [[0, 1, 2], [3], [4]] : {page_slice_ty} into {page_seq_ty}
    %c0i = arith.constant 0 : index
    %new_k = linalg.generic {{
        indexing_maps = [affine_map<(p, h, d) -> (p, h, d)>],
        iterator_types = ["parallel", "parallel", "parallel"]}}
      outs(%old_k : {page_seq_ty}) {{
      ^bb0(%old: f32):
        %pidx = linalg.index 0 : index
        %hidx = linalg.index 1 : index
        %didx = linalg.index 2 : index
        %absolute = arith.addi %page_start, %pidx : index
        %after_start = arith.cmpi uge, %absolute, %start : index
        %before_end = arith.cmpi ult, %absolute, %chunk_end : index
        %replace = arith.andi %after_start, %before_end : i1
        %source_raw = arith.subi %absolute, %start : index
        %source = arith.select %replace, %source_raw, %c0i : index
        %value = tensor.extract %chunk_k[%source, %hidx, %didx] : {kv3_ty}
        %result = arith.select %replace, %value, %old : f32
        linalg.yield %result : f32
    }} -> {page_seq_ty}
    %new_v = linalg.generic {{
        indexing_maps = [affine_map<(p, h, d) -> (p, h, d)>],
        iterator_types = ["parallel", "parallel", "parallel"]}}
      outs(%old_v : {page_seq_ty}) {{
      ^bb0(%old: f32):
        %pidx = linalg.index 0 : index
        %hidx = linalg.index 1 : index
        %didx = linalg.index 2 : index
        %absolute = arith.addi %page_start, %pidx : index
        %after_start = arith.cmpi uge, %absolute, %start : index
        %before_end = arith.cmpi ult, %absolute, %chunk_end : index
        %replace = arith.andi %after_start, %before_end : i1
        %source_raw = arith.subi %absolute, %start : index
        %source = arith.select %replace, %source_raw, %c0i : index
        %value = tensor.extract %chunk_v[%source, %hidx, %didx] : {kv3_ty}
        %result = arith.select %replace, %value, %old : f32
        linalg.yield %result : f32
    }} -> {page_seq_ty}
    %ke = tensor.expand_shape %new_k [[0, 1, 2], [3], [4]] output_shape [1, 1, {page}, {nkv}, {d}] : {page_seq_ty} into {expanded_ty}
    %ve = tensor.expand_shape %new_v [[0, 1, 2], [3], [4]] output_shape [1, 1, {page}, {nkv}, {d}] : {page_seq_ty} into {expanded_ty}
    %pk = tensor.insert_slice %ke into %page[%layer, 0, 0, 0, 0] [1, 1, {page}, {nkv}, {d}] [1, 1, 1, 1, 1] : {expanded_ty} into {page_ty}
    %pv = tensor.insert_slice %ve into %pk[%layer, 1, 0, 0, 0] [1, 1, {page}, {nkv}, {d}] [1, 1, 1, 1, 1] : {expanded_ty} into {page_ty}
    scf.yield %pv : {page_ty}
  }} else {{
    scf.yield %page : {page_ty}
  }}
  %ks = tensor.extract_slice %page_updated[%layer, 0, 0, 0, 0] [1, 1, {page}, {nkv}, {d}] [1, 1, 1, 1, 1] : {page_ty} to {page_slice_ty}
  %vs = tensor.extract_slice %page_updated[%layer, 1, 0, 0, 0] [1, 1, {page}, {nkv}, {d}] [1, 1, 1, 1, 1] : {page_ty} to {page_slice_ty}
  %kseq = tensor.collapse_shape %ks [[0, 1, 2], [3], [4]] : {page_slice_ty} into {page_seq_ty}
  %vseq = tensor.collapse_shape %vs [[0, 1, 2], [3], [4]] : {page_slice_ty} into {page_seq_ty}
  %ke2 = tensor.empty() : {page_head_ty}
  %ve2 = tensor.empty() : {page_head_ty}
  %kh = linalg.transpose ins(%kseq : {page_seq_ty}) outs(%ke2 : {page_head_ty}) permutation = [1, 0, 2]
  %vh = linalg.transpose ins(%vseq : {page_seq_ty}) outs(%ve2 : {page_head_ty}) permutation = [1, 0, 2]
  %mask = func.call @{mask_helper}(%start_pos, %pi, %valid_t) : (tensor<i64>, index, tensor<i64>) -> {mask_ty}
  %scale = arith.constant {scale:.8e} : {scale_ty}
  %next:3 = func.call @{attention_helper}(%query, %kh, %vh, %scale, %mask, %out, %max, %sum) : ({qg_ty}, {page_head_ty}, {page_head_ty}, {scale_ty}, {mask_ty}, {qg_ty}, {row_ty}, {row_ty}) -> ({qg_ty}, {row_ty}, {row_ty})
  return %page_updated, %next#0, %next#1, %next#2 : {page_ty}, {qg_ty}, {row_ty}, {row_ty}"#,
        scale = 1.0 / (d as f32).sqrt(),
        valid_global = variant.global("valid_count"),
    ));
    f.finish(module)
}

fn emit_paged_layer_finish(
    module: &mut ModuleBuilder,
    c: &DenseDecoderConfig,
    layer: u32,
    variant: PagedVariant,
) -> Result<()> {
    let (s, h, q, nkv, g, d, i) = (
        c.seq,
        c.hidden,
        c.q_dim(),
        c.num_kv_heads,
        c.gqa_group(),
        c.head_dim,
        c.intermediate,
    );
    let p = format!("blk{layer}");
    let n = format!("blk.{layer}");
    let hidden_ty = format!("tensor<{s}x{h}xf32>");
    let qg_ty = format!("tensor<{nkv}x{g}x{s}x{d}xf32>");
    let row_ty = format!("tensor<{nkv}x{g}x{s}xf32>");
    let rms_norm = variant.helper("rms_norm");
    let linear_qh = variant.helper("linear_qh");
    let linear_hi = variant.helper("linear_hi");
    let linear_ih = variant.helper("linear_ih");
    let mut f = module.func_private(&variant.function(&format!("layer_finish_{layer}")));
    f.arg("out", &qg_ty);
    f.arg("sum", &row_ty);
    for (ssa, sym, canonical, shape) in [
        (
            "wo",
            format!("{p}_attn_output_weight"),
            format!("{n}.attn_output.weight"),
            format!("{h}x{q}"),
        ),
        (
            "ffn_nw",
            format!("{p}_ffn_norm_weight"),
            format!("{n}.ffn_norm.weight"),
            format!("{h}"),
        ),
        (
            "wgate",
            format!("{p}_ffn_gate_weight"),
            format!("{n}.ffn_gate.weight"),
            format!("{i}x{h}"),
        ),
        (
            "wup",
            format!("{p}_ffn_up_weight"),
            format!("{n}.ffn_up.weight"),
            format!("{i}x{h}"),
        ),
        (
            "wdown",
            format!("{p}_ffn_down_weight"),
            format!("{n}.ffn_down.weight"),
            format!("{h}x{i}"),
        ),
    ] {
        emit_load_compute(&mut f, c, ssa, &sym, &canonical, &shape);
    }
    f.op_asm(format!(
        r#"  %hidden = util.global.load @{hidden_global} : {hidden_ty}
  %norm_e = tensor.empty() : {qg_ty}
  %norm = linalg.generic {{
      indexing_maps = [
        affine_map<(kh, g, q, d) -> (kh, g, q, d)>,
        affine_map<(kh, g, q, d) -> (kh, g, q)>,
        affine_map<(kh, g, q, d) -> (kh, g, q, d)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}}
    ins(%out, %sum : {qg_ty}, {row_ty}) outs(%norm_e : {qg_ty}) {{
    ^bb0(%value: f32, %den: f32, %o: f32):
      %result = arith.divf %value, %den : f32
      linalg.yield %result : f32
  }} -> {qg_ty}
  %ctx4e = tensor.empty() : tensor<{s}x{nkv}x{g}x{d}xf32>
  %ctx4 = linalg.transpose ins(%norm : {qg_ty}) outs(%ctx4e : tensor<{s}x{nkv}x{g}x{d}xf32>) permutation = [2, 0, 1, 3]
  %ctx = tensor.collapse_shape %ctx4 [[0], [1, 2, 3]] : tensor<{s}x{nkv}x{g}x{d}xf32> into tensor<{s}x{q}xf32>
"#,
        hidden_global = variant.global("hidden"),
    ));
    emit_linear_call(
        &mut f,
        c,
        "o",
        &linear_qh,
        "ctx",
        "wo",
        &format!("{n}.attn_output.weight"),
        s,
        q,
        h,
    );
    f.op_asm(format!(
        "  %h2 = arith.addf %hidden, %o : {hidden_ty}\n  %fn = func.call @{rms_norm}(%h2, %ffn_nw) : ({hidden_ty}, tensor<{h}xf32>) -> {hidden_ty}\n"
    ));
    emit_linear_call(
        &mut f,
        c,
        "gate",
        &linear_hi,
        "fn",
        "wgate",
        &format!("{n}.ffn_gate.weight"),
        s,
        h,
        i,
    );
    emit_linear_call(
        &mut f,
        c,
        "up",
        &linear_hi,
        "fn",
        "wup",
        &format!("{n}.ffn_up.weight"),
        s,
        h,
        i,
    );
    f.op_asm(format!(
        r#"  %silu = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}}
    ins(%gate : tensor<{s}x{i}xf32>) outs(%gate : tensor<{s}x{i}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %neg = arith.negf %a : f32
      %exp = math.exp %neg : f32
      %one = arith.constant 1.0 : f32
      %den = arith.addf %one, %exp : f32
      %value = arith.divf %a, %den : f32
      linalg.yield %value : f32
  }} -> tensor<{s}x{i}xf32>
  %ff = arith.mulf %silu, %up : tensor<{s}x{i}xf32>
"#
    ));
    emit_linear_call(
        &mut f,
        c,
        "down",
        &linear_ih,
        "ff",
        "wdown",
        &format!("{n}.ffn_down.weight"),
        s,
        i,
        h,
    );
    f.op_asm(format!(
        "  %next_hidden = arith.addf %h2, %down : {hidden_ty}\n  util.global.store %next_hidden, @{hidden_global} : {hidden_ty}\n  return",
        hidden_global = variant.global("hidden"),
    ));
    f.finish(module)
}

fn emit_paged_chunk_logits(
    module: &mut ModuleBuilder,
    c: &DenseDecoderConfig,
    variant: PagedVariant,
) -> Result<()> {
    let (s, h, v) = (c.seq, c.hidden, c.vocab);
    let hidden_ty = format!("tensor<{s}x{h}xf32>");
    let rms_norm = variant.helper("rms_norm");
    let mut f = module.func_private(&variant.function("chunk_logits"));
    f.arg("last", "tensor<i64>");
    f.result_ty(format!("tensor<{v}xf32>"));
    emit_load_compute(
        &mut f,
        c,
        "out_nw",
        "output_norm_weight",
        "output_norm.weight",
        &format!("{h}"),
    );
    emit_load_compute(
        &mut f,
        c,
        "wout",
        "output_weight",
        "output.weight",
        &format!("{v}x{h}"),
    );
    f.op_asm(format!(
        r#"  %hidden = util.global.load @{hidden_global} : {hidden_ty}
  %last_i64 = tensor.extract %last[] : tensor<i64>
  %li = arith.index_cast %last_i64 : i64 to index
  %last_row = tensor.extract_slice %hidden[%li, 0] [1, {h}] [1, 1] : {hidden_ty} to tensor<1x{h}xf32>
  %last_tile = tensor.empty() : {hidden_ty}
  %last_s = tensor.insert_slice %last_row into %last_tile[0, 0] [1, {h}] [1, 1] : tensor<1x{h}xf32> into {hidden_ty}
  %ln = func.call @{rms_norm}(%last_s, %out_nw) : ({hidden_ty}, tensor<{h}xf32>) -> {hidden_ty}
  %ln1 = tensor.extract_slice %ln[0, 0] [1, {h}] [1, 1] : {hidden_ty} to tensor<1x{h}xf32>
"#,
        hidden_global = variant.global("hidden"),
    ));
    emit_output_proj(&mut f, c, "ln1", "wout", variant.mode());
    f.op_asm(format!("  return %logits : tensor<{v}xf32>"));
    f.finish(module)
}

fn bound_param_dtype(param: &dyninfer_core::ParameterBinding) -> Option<ScalarType> {
    let comp = param.components.first()?;
    match &comp.storage_type {
        StorageElementType::Scalar { ty } => Some(*ty),
        _ => None,
    }
}

fn emit_global(
    builder: &mut ModuleBuilder,
    c: &DenseDecoderConfig,
    _parameters: &ParameterLowerings,
    sym: &str,
    canonical: &str,
    shape: &str,
) -> Result<()> {
    default_parameter_lowerings().emit_global(builder, c, sym, canonical, shape)
}

/// Load a weight global and cast to the operation-local selected compute type.
fn emit_load_compute(
    f: &mut FuncBuilder,
    c: &DenseDecoderConfig,
    ssa: &str,
    sym: &str,
    canonical: &str,
    shape: &str,
) {
    default_parameter_lowerings()
        .emit_load(f, c, ssa, sym, canonical, shape)
        .expect("selected parameter load was validated before MLIR emission");
}

fn emit_linear_call(
    f: &mut FuncBuilder,
    c: &DenseDecoderConfig,
    result: &str,
    dense_fn: &str,
    x: &str,
    w: &str,
    weight_canonical: &str,
    s: u32,
    in_dim: u32,
    out_dim: u32,
) {
    let mode = if dense_fn.ends_with("_tok") {
        ExecutionMode::Decode
    } else {
        ExecutionMode::Prefill
    };
    default_parameter_lowerings()
        .emit_linear_call(
            f,
            c,
            result,
            dense_fn,
            x,
            w,
            weight_canonical,
            mode,
            s,
            in_dim,
            out_dim,
        )
        .expect("selected linear lowering was validated before MLIR emission");
}

/// Project hidden → vocab logits. Emits `%logits : tensor<{vocab}xf32>`.
fn emit_output_proj(
    f: &mut FuncBuilder,
    c: &DenseDecoderConfig,
    hidden_ssa: &str,
    weight_ssa: &str,
    mode: ExecutionMode,
) {
    let (h, v) = (c.hidden, c.vocab);
    if default_parameter_lowerings()
        .emit_output_projection(f, c, hidden_ssa, weight_ssa, mode)
        .expect("selected output lowering was validated before MLIR emission")
    {
        return;
    }
    f.op_asm("  %c0f = arith.constant 0.0 : f32\n");
    f.op_asm(format!("  %wt_i = tensor.empty() : tensor<{h}x{v}xf32>\n"));
    f.op_asm(format!(
        "  %wt = linalg.transpose ins(%{weight_ssa} : tensor<{v}x{h}xf32>) outs(%wt_i : tensor<{h}x{v}xf32>) permutation = [1, 0]\n"
    ));
    f.op_asm(format!("  %yi = tensor.empty() : tensor<1x{v}xf32>\n"));
    f.op_asm(format!(
        "  %yz = linalg.fill ins(%c0f : f32) outs(%yi : tensor<1x{v}xf32>) -> tensor<1x{v}xf32>\n"
    ));
    f.op_asm(format!(
        "  %y = linalg.matmul ins(%{hidden_ssa}, %wt : tensor<1x{h}xf32>, tensor<{h}x{v}xf32>) outs(%yz : tensor<1x{v}xf32>) -> tensor<1x{v}xf32>\n"
    ));
    f.op_asm(format!(
        "  %logits = tensor.collapse_shape %y [[0, 1]] : tensor<1x{v}xf32> into tensor<{v}xf32>\n"
    ));
}

fn emit_globals(
    builder: &mut ModuleBuilder,
    c: &DenseDecoderConfig,
    program: PagedProgram,
) -> Result<()> {
    let parameters = default_parameter_lowerings();
    let (v, h, i) = (c.vocab, c.hidden, c.intermediate);
    let (q, kv, d) = (c.q_dim(), c.kv_dim(), c.head_dim);
    emit_global(
        builder,
        c,
        parameters,
        "token_embd_weight",
        "token_embd.weight",
        &format!("{v}x{h}"),
    )?;
    for layer in 0..c.num_layers {
        let p = format!("blk{layer}");
        let n = format!("blk.{layer}");
        emit_global(
            builder,
            c,
            parameters,
            &format!("{p}_attn_norm_weight"),
            &format!("{n}.attn_norm.weight"),
            &format!("{h}"),
        )?;
        emit_global(
            builder,
            c,
            parameters,
            &format!("{p}_attn_q_weight"),
            &format!("{n}.attn_q.weight"),
            &format!("{q}x{h}"),
        )?;
        emit_global(
            builder,
            c,
            parameters,
            &format!("{p}_attn_k_weight"),
            &format!("{n}.attn_k.weight"),
            &format!("{kv}x{h}"),
        )?;
        emit_global(
            builder,
            c,
            parameters,
            &format!("{p}_attn_v_weight"),
            &format!("{n}.attn_v.weight"),
            &format!("{kv}x{h}"),
        )?;
        emit_global(
            builder,
            c,
            parameters,
            &format!("{p}_attn_output_weight"),
            &format!("{n}.attn_output.weight"),
            &format!("{h}x{q}"),
        )?;
        if c.has_qk_norm {
            emit_global(
                builder,
                c,
                parameters,
                &format!("{p}_attn_q_norm_weight"),
                &format!("{n}.attn_q_norm.weight"),
                &format!("{d}"),
            )?;
            emit_global(
                builder,
                c,
                parameters,
                &format!("{p}_attn_k_norm_weight"),
                &format!("{n}.attn_k_norm.weight"),
                &format!("{d}"),
            )?;
        }
        emit_global(
            builder,
            c,
            parameters,
            &format!("{p}_ffn_norm_weight"),
            &format!("{n}.ffn_norm.weight"),
            &format!("{h}"),
        )?;
        emit_global(
            builder,
            c,
            parameters,
            &format!("{p}_ffn_gate_weight"),
            &format!("{n}.ffn_gate.weight"),
            &format!("{i}x{h}"),
        )?;
        emit_global(
            builder,
            c,
            parameters,
            &format!("{p}_ffn_up_weight"),
            &format!("{n}.ffn_up.weight"),
            &format!("{i}x{h}"),
        )?;
        emit_global(
            builder,
            c,
            parameters,
            &format!("{p}_ffn_down_weight"),
            &format!("{n}.ffn_down.weight"),
            &format!("{h}x{i}"),
        )?;
    }
    emit_global(
        builder,
        c,
        parameters,
        "output_norm_weight",
        "output_norm.weight",
        &format!("{h}"),
    )?;
    emit_global(
        builder,
        c,
        parameters,
        "output_weight",
        "output.weight",
        &format!("{v}x{h}"),
    )?;
    if c.paged_kv {
        if matches!(program, PagedProgram::Combined | PagedProgram::Prefill) {
            emit_paged_state_globals(builder, c, PagedVariant { decode: false })?;
        }
        if matches!(program, PagedProgram::Combined | PagedProgram::Decode) {
            let mut decode = c.clone();
            decode.seq = 1;
            emit_paged_state_globals(builder, &decode, PagedVariant { decode: true })?;
        }
        return Ok(());
    }
    // Mutable KV cache (f32 compute). Prefill seeds [0, seq); decode grows to max_kv.
    let (mk, nkv, d) = (c.max_kv, c.num_kv_heads, c.head_dim);
    for layer in 0..c.num_layers {
        builder.util_global_mutable_zero(
            &format!("kv_k{layer}"),
            &format!("tensor<{mk}x{nkv}x{d}xf32>"),
        )?;
        builder.util_global_mutable_zero(
            &format!("kv_v{layer}"),
            &format!("tensor<{mk}x{nkv}x{d}xf32>"),
        )?;
    }
    Ok(())
}

fn emit_paged_state_globals(
    builder: &mut ModuleBuilder,
    c: &DenseDecoderConfig,
    variant: PagedVariant,
) -> Result<()> {
    let (s, h, nkv, g, d) = (c.seq, c.hidden, c.num_kv_heads, c.gqa_group(), c.head_dim);
    builder.util_global_mutable_zero(&variant.global("hidden"), &format!("tensor<{s}x{h}xf32>"))?;
    builder.util_global_mutable_zero(
        &variant.global("query"),
        &format!("tensor<{nkv}x{g}x{s}x{d}xf32>"),
    )?;
    builder.util_global_mutable_zero(
        &variant.global("chunk_k"),
        &format!("tensor<{s}x{nkv}x{d}xf32>"),
    )?;
    builder.util_global_mutable_zero(
        &variant.global("chunk_v"),
        &format!("tensor<{s}x{nkv}x{d}xf32>"),
    )?;
    builder.util_global_mutable_zero(
        &variant.global("attn_output"),
        &format!("tensor<{nkv}x{g}x{s}x{d}xf32>"),
    )?;
    builder.util_global_mutable_zero(
        &variant.global("attn_max"),
        &format!("tensor<{nkv}x{g}x{s}xf32>"),
    )?;
    builder.util_global_mutable_zero(
        &variant.global("attn_sum"),
        &format!("tensor<{nkv}x{g}x{s}xf32>"),
    )?;
    builder.append_toplevel_asm(&format!(
        "util.global private mutable @{} = dense<0> : tensor<i64>",
        variant.global("start_pos")
    ))?;
    builder.append_toplevel_asm(&format!(
        "util.global private mutable @{} = dense<0> : tensor<i64>",
        variant.global("valid_count")
    ))?;
    Ok(())
}

fn emit_helpers(
    module: &mut ModuleBuilder,
    c: &DenseDecoderConfig,
    program: PagedProgram,
) -> Result<()> {
    let parameters = default_parameter_lowerings();
    let (s, h, i, nh, nkv, d) = (
        c.seq,
        c.hidden,
        c.intermediate,
        c.num_heads,
        c.num_kv_heads,
        c.head_dim,
    );
    let (q, kv) = (c.q_dim(), c.kv_dim());
    let g = c.gqa_group();
    let emit_prefill_helpers =
        !c.paged_kv || matches!(program, PagedProgram::Combined | PagedProgram::Prefill);

    if emit_prefill_helpers {
        kernels::emit_rms_norm(module, "rms_norm", s, h, c.rms_norm_eps)?;
        kernels::emit_linear(module, "linear_hq", s, h, q, true)?;
        kernels::emit_linear(module, "linear_hkv", s, h, kv, true)?;
        kernels::emit_linear(module, "linear_qh", s, q, h, true)?;
        kernels::emit_linear(module, "linear_hi", s, h, i, true)?;
        kernels::emit_linear(module, "linear_ih", s, i, h, true)?;
    }

    parameters.emit_quantized_helpers(module, c)?;

    if emit_prefill_helpers && c.has_qk_norm {
        kernels::emit_rms_norm_heads(module, "rms_norm_q_heads", s, nh, d, c.rms_norm_eps)?;
        if nkv != nh {
            kernels::emit_rms_norm_heads(module, "rms_norm_kv_heads", s, nkv, d, c.rms_norm_eps)?;
        }
    }

    if emit_prefill_helpers && nkv != nh {
        kernels::emit_repeat_kv(module, "repeat_kv", s, nkv, nh, d, g)?;
    }

    if emit_prefill_helpers
        && !c.paged_kv
        && let Some(theta) = c.rope_theta
    {
        kernels::emit_rope(module, "apply_rope_q", s, nh, d, theta)?;
        if nkv != nh {
            kernels::emit_rope(module, "apply_rope_kv", s, nkv, d, theta)?;
        }
    }

    if c.paged_kv {
        if matches!(program, PagedProgram::Combined | PagedProgram::Prefill) {
            if c.iree_flash_attention {
                kernels::emit_iree_online_attention_page(
                    module,
                    "online_attention_page",
                    s,
                    PAGED_KV_PAGE_SIZE,
                    nkv,
                    g,
                    d,
                )?;
            } else {
                kernels::emit_online_attention_page(
                    module,
                    "online_attention_page",
                    s,
                    PAGED_KV_PAGE_SIZE,
                    nkv,
                    g,
                    d,
                )?;
            }
            kernels::emit_paged_causal_mask(
                module,
                "paged_causal_mask",
                s,
                PAGED_KV_PAGE_SIZE,
                nkv,
                g,
                c.dense_index_constants,
            )?;
            if let Some(theta) = c.rope_theta {
                kernels::emit_rope_chunk(module, "apply_rope_q_chunk", s, nh, d, theta)?;
                kernels::emit_rope_chunk(module, "apply_rope_kv_chunk", s, nkv, d, theta)?;
            }
            emit_prepare_paged_attention(module, c, PagedVariant { decode: false })?;
        }
        if matches!(program, PagedProgram::Combined | PagedProgram::Decode) {
            let mut decode = c.clone();
            decode.seq = 1;
            emit_paged_decode_helpers(module, &decode)?;
        }
    }

    if !c.paged_kv {
        emit_attn(module, c)?;
        emit_decode_helpers(module, c)?;
    }
    Ok(())
}

fn emit_paged_decode_helpers(module: &mut ModuleBuilder, c: &DenseDecoderConfig) -> Result<()> {
    let (s, h, i, nh, nkv, d, q, kv, g) = (
        c.seq,
        c.hidden,
        c.intermediate,
        c.num_heads,
        c.num_kv_heads,
        c.head_dim,
        c.q_dim(),
        c.kv_dim(),
        c.gqa_group(),
    );
    debug_assert_eq!(s, 1);
    kernels::emit_rms_norm(module, "rms_norm_tok", s, h, c.rms_norm_eps)?;
    kernels::emit_linear(module, "linear_hq_tok", s, h, q, true)?;
    kernels::emit_linear(module, "linear_hkv_tok", s, h, kv, true)?;
    kernels::emit_linear(module, "linear_qh_tok", s, q, h, true)?;
    kernels::emit_linear(module, "linear_hi_tok", s, h, i, true)?;
    kernels::emit_linear(module, "linear_ih_tok", s, i, h, true)?;
    if c.has_qk_norm {
        kernels::emit_rms_norm_heads(module, "rms_norm_q_heads_tok", s, nh, d, c.rms_norm_eps)?;
        if nkv != nh {
            kernels::emit_rms_norm_heads(
                module,
                "rms_norm_kv_heads_tok",
                s,
                nkv,
                d,
                c.rms_norm_eps,
            )?;
        }
    }
    if c.iree_flash_attention {
        kernels::emit_iree_online_attention_page(
            module,
            "online_attention_page_tok",
            s,
            PAGED_KV_PAGE_SIZE,
            nkv,
            g,
            d,
        )?;
    } else {
        kernels::emit_online_attention_page(
            module,
            "online_attention_page_tok",
            s,
            PAGED_KV_PAGE_SIZE,
            nkv,
            g,
            d,
        )?;
    }
    kernels::emit_paged_causal_mask(
        module,
        "paged_causal_mask_tok",
        s,
        PAGED_KV_PAGE_SIZE,
        nkv,
        g,
        c.dense_index_constants,
    )?;
    if let Some(theta) = c.rope_theta {
        kernels::emit_rope_chunk(module, "apply_rope_q_chunk_tok", s, nh, d, theta)?;
        kernels::emit_rope_chunk(module, "apply_rope_kv_chunk_tok", s, nkv, d, theta)?;
    }
    emit_prepare_paged_attention(module, c, PagedVariant { decode: true })
}

fn emit_prepare_paged_attention(
    module: &mut ModuleBuilder,
    c: &DenseDecoderConfig,
    variant: PagedVariant,
) -> Result<()> {
    let (s, nh, nkv, d, q, kv, g) = (
        c.seq,
        c.num_heads,
        c.num_kv_heads,
        c.head_dim,
        c.q_dim(),
        c.kv_dim(),
        c.gqa_group(),
    );
    let q_grouped_ty = format!("tensor<{nkv}x{g}x{s}x{d}xf32>");
    let kv_ty = format!("tensor<{s}x{nkv}x{d}xf32>");
    let q_norm_helper = variant.helper("rms_norm_q_heads");
    let kv_norm_helper = if nkv == nh {
        q_norm_helper.clone()
    } else {
        variant.helper("rms_norm_kv_heads")
    };
    let rope_q_helper = variant.helper("apply_rope_q_chunk");
    let rope_kv_helper = variant.helper("apply_rope_kv_chunk");
    let mut f = module.func_private(&variant.helper("prepare_paged_attention"));
    f.arg("q", format!("tensor<{s}x{q}xf32>"));
    f.arg("k", format!("tensor<{s}x{kv}xf32>"));
    f.arg("v", format!("tensor<{s}x{kv}xf32>"));
    f.arg("start_pos", "tensor<i64>");
    if c.has_qk_norm {
        f.arg("q_norm", format!("tensor<{d}xf32>"));
        f.arg("k_norm", format!("tensor<{d}xf32>"));
    }
    f.result_ty(&q_grouped_ty);
    f.result_ty(&kv_ty);
    f.result_ty(&kv_ty);
    f.op_asm(format!(
        r#"  %q3 = tensor.expand_shape %q [[0], [1, 2]] output_shape [{s}, {nh}, {d}] : tensor<{s}x{q}xf32> into tensor<{s}x{nh}x{d}xf32>
  %k3 = tensor.expand_shape %k [[0], [1, 2]] output_shape [{s}, {nkv}, {d}] : tensor<{s}x{kv}xf32> into {kv_ty}
  %v3 = tensor.expand_shape %v [[0], [1, 2]] output_shape [{s}, {nkv}, {d}] : tensor<{s}x{kv}xf32> into {kv_ty}
"#
    ));
    let (q_name, k_name) = if c.has_qk_norm {
        f.op_asm(format!(
            "  %qn = func.call @{q_norm_helper}(%q3, %q_norm) : (tensor<{s}x{nh}x{d}xf32>, tensor<{d}xf32>) -> tensor<{s}x{nh}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  %kn = func.call @{kv_norm_helper}(%k3, %k_norm) : ({kv_ty}, tensor<{d}xf32>) -> {kv_ty}\n"
        ));
        ("qn", "kn")
    } else {
        ("q3", "k3")
    };
    let (q_name, k_name) = if c.rope_theta.is_some() {
        f.op_asm(format!(
            "  %qr = func.call @{rope_q_helper}(%{q_name}, %start_pos) : (tensor<{s}x{nh}x{d}xf32>, tensor<i64>) -> tensor<{s}x{nh}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  %kr = func.call @{rope_kv_helper}(%{k_name}, %start_pos) : ({kv_ty}, tensor<i64>) -> {kv_ty}\n"
        ));
        ("qr", "kr")
    } else {
        (q_name, k_name)
    };
    f.op_asm(format!(
        r#"  %q4 = tensor.expand_shape %{q_name} [[0], [1, 2], [3]] output_shape [{s}, {nkv}, {g}, {d}] : tensor<{s}x{nh}x{d}xf32> into tensor<{s}x{nkv}x{g}x{d}xf32>
  %qe = tensor.empty() : {q_grouped_ty}
  %qg = linalg.transpose ins(%q4 : tensor<{s}x{nkv}x{g}x{d}xf32>) outs(%qe : {q_grouped_ty}) permutation = [1, 2, 0, 3]
  return %qg, %{k_name}, %v3 : {q_grouped_ty}, {kv_ty}, {kv_ty}"#
    ));
    f.finish(module)
}

fn emit_attn(module: &mut ModuleBuilder, c: &DenseDecoderConfig) -> Result<()> {
    let (s, nh, nkv, d) = (c.seq, c.num_heads, c.num_kv_heads, c.head_dim);
    let (q, kv) = (c.q_dim(), c.kv_dim());
    let scale = 1.0 / (d as f32).sqrt();
    // Returns (context, k_post_rope [S×nkv×d], v [S×nkv×d]) so callers can seed KV cache.
    let mut f = module.func_private("attn");
    f.arg("q", format!("tensor<{s}x{q}xf32>"));
    f.arg("k", format!("tensor<{s}x{kv}xf32>"));
    f.arg("v", format!("tensor<{s}x{kv}xf32>"));
    if c.has_qk_norm {
        f.arg("q_norm", format!("tensor<{d}xf32>"));
        f.arg("k_norm", format!("tensor<{d}xf32>"));
    }
    f.result_ty(format!("tensor<{s}x{q}xf32>"));
    f.result_ty(format!("tensor<{s}x{nkv}x{d}xf32>"));
    f.result_ty(format!("tensor<{s}x{nkv}x{d}xf32>"));
    f.op_asm(format!(
        r#"  %q3 = tensor.expand_shape %q [[0], [1, 2]] output_shape [{s}, {nh}, {d}] : tensor<{s}x{q}xf32> into tensor<{s}x{nh}x{d}xf32>
  %k3 = tensor.expand_shape %k [[0], [1, 2]] output_shape [{s}, {nkv}, {d}] : tensor<{s}x{kv}xf32> into tensor<{s}x{nkv}x{d}xf32>
  %v3 = tensor.expand_shape %v [[0], [1, 2]] output_shape [{s}, {nkv}, {d}] : tensor<{s}x{kv}xf32> into tensor<{s}x{nkv}x{d}xf32>
"#,
        s = s,
        q = q,
        kv = kv,
        nh = nh,
        nkv = nkv,
        d = d,
    ));

    let q_heads = if c.has_qk_norm {
        f.op_asm(format!(
            "  %qn = func.call @rms_norm_q_heads(%q3, %q_norm) : (tensor<{s}x{nh}x{d}xf32>, tensor<{d}xf32>) -> tensor<{s}x{nh}x{d}xf32>\n",
            s = s, nh = nh, d = d,
        ));
        "qn"
    } else {
        "q3"
    };
    let k_heads = if c.has_qk_norm {
        let kn_fn = if nkv != nh {
            "rms_norm_kv_heads"
        } else {
            "rms_norm_q_heads"
        };
        f.op_asm(format!(
            "  %kn = func.call @{kn_fn}(%k3, %k_norm) : (tensor<{s}x{nkv}x{d}xf32>, tensor<{d}xf32>) -> tensor<{s}x{nkv}x{d}xf32>\n",
            kn_fn = kn_fn,
            s = s,
            nkv = nkv,
            d = d,
        ));
        "kn"
    } else {
        "k3"
    };

    let (q_ssa, k_ssa) = if c.rope_theta.is_some() {
        f.op_asm(format!(
            r#"  %qr = func.call @apply_rope_q(%{q_heads}) : (tensor<{s}x{nh}x{d}xf32>) -> tensor<{s}x{nh}x{d}xf32>
"#,
            q_heads = q_heads,
            s = s,
            nh = nh,
            d = d,
        ));
        let rope_kv = if nkv != nh {
            "apply_rope_kv"
        } else {
            "apply_rope_q"
        };
        f.op_asm(format!(
            r#"  %kr = func.call @{rope_kv}(%{k_heads}) : (tensor<{s}x{nkv}x{d}xf32>) -> tensor<{s}x{nkv}x{d}xf32>
"#,
            rope_kv = rope_kv,
            k_heads = k_heads,
            s = s,
            nkv = nkv,
            d = d,
        ));
        ("qr", "kr")
    } else {
        (q_heads, k_heads)
    };

    let k_full = if nkv != nh {
        f.op_asm(format!(
            "  %krep = func.call @repeat_kv(%{k_ssa}) : (tensor<{s}x{nkv}x{d}xf32>) -> tensor<{s}x{nh}x{d}xf32>\n",
            k_ssa = k_ssa,
            s = s,
            nkv = nkv,
            nh = nh,
            d = d,
        ));
        f.op_asm(format!(
            "  %vrep = func.call @repeat_kv(%v3) : (tensor<{s}x{nkv}x{d}xf32>) -> tensor<{s}x{nh}x{d}xf32>\n",
            s = s,
            nkv = nkv,
            nh = nh,
            d = d,
        ));
        ("krep", "vrep")
    } else {
        (k_ssa, "v3")
    };

    f.op_asm(format!(
        // Region-heavy linalg still emitted as verified assembly snippets.
        // Typed OperationBuilder helpers would not shrink this meaningfully yet.
        r#"  %q_ti = tensor.empty() : tensor<{nh}x{s}x{d}xf32>
  %k_ti = tensor.empty() : tensor<{nh}x{s}x{d}xf32>
  %v_ti = tensor.empty() : tensor<{nh}x{s}x{d}xf32>
  %qb = linalg.transpose ins(%{q_ssa} : tensor<{s}x{nh}x{d}xf32>) outs(%q_ti : tensor<{nh}x{s}x{d}xf32>) permutation = [1, 0, 2]
  %kb = linalg.transpose ins(%{k_full} : tensor<{s}x{nh}x{d}xf32>) outs(%k_ti : tensor<{nh}x{s}x{d}xf32>) permutation = [1, 0, 2]
  %vb = linalg.transpose ins(%{v_full} : tensor<{s}x{nh}x{d}xf32>) outs(%v_ti : tensor<{nh}x{s}x{d}xf32>) permutation = [1, 0, 2]
  %kt_i = tensor.empty() : tensor<{nh}x{d}x{s}xf32>
  %kt = linalg.transpose ins(%kb : tensor<{nh}x{s}x{d}xf32>) outs(%kt_i : tensor<{nh}x{d}x{s}xf32>) permutation = [0, 2, 1]
  %c0 = arith.constant 0.0 : f32
  %neg = arith.constant -1.0e+7 : f32
  %scale = arith.constant {scale:.8e} : f32
  %sc_i = tensor.empty() : tensor<{nh}x{s}x{s}xf32>
  %sc_z = linalg.fill ins(%c0 : f32) outs(%sc_i : tensor<{nh}x{s}x{s}xf32>) -> tensor<{nh}x{s}x{s}xf32>
  %scores = linalg.batch_matmul ins(%qb, %kt : tensor<{nh}x{s}x{d}xf32>, tensor<{nh}x{d}x{s}xf32>) outs(%sc_z : tensor<{nh}x{s}x{s}xf32>) -> tensor<{nh}x{s}x{s}xf32>
  %scores_s = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%scores : tensor<{nh}x{s}x{s}xf32>) outs(%scores : tensor<{nh}x{s}x{s}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %m = arith.mulf %a, %scale : f32
      linalg.yield %m : f32
  }} -> tensor<{nh}x{s}x{s}xf32>
  %masked = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%scores_s : tensor<{nh}x{s}x{s}xf32>) outs(%scores_s : tensor<{nh}x{s}x{s}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %ii = linalg.index 1 : index
      %jj = linalg.index 2 : index
      %cmp = arith.cmpi sgt, %jj, %ii : index
      %sel = arith.select %cmp, %neg, %a : f32
      linalg.yield %sel : f32
  }} -> tensor<{nh}x{s}x{s}xf32>
  %sm_i = tensor.empty() : tensor<{nh}x{s}x{s}xf32>
  %attn = linalg.softmax dimension(2) ins(%masked : tensor<{nh}x{s}x{s}xf32>) outs(%sm_i : tensor<{nh}x{s}x{s}xf32>) -> tensor<{nh}x{s}x{s}xf32>
  %ctx_i = tensor.empty() : tensor<{nh}x{s}x{d}xf32>
  %ctx_z = linalg.fill ins(%c0 : f32) outs(%ctx_i : tensor<{nh}x{s}x{d}xf32>) -> tensor<{nh}x{s}x{d}xf32>
  %ctx_b = linalg.batch_matmul ins(%attn, %vb : tensor<{nh}x{s}x{s}xf32>, tensor<{nh}x{s}x{d}xf32>) outs(%ctx_z : tensor<{nh}x{s}x{d}xf32>) -> tensor<{nh}x{s}x{d}xf32>
  %ctx_t_i = tensor.empty() : tensor<{s}x{nh}x{d}xf32>
  %ctx_t = linalg.transpose ins(%ctx_b : tensor<{nh}x{s}x{d}xf32>) outs(%ctx_t_i : tensor<{s}x{nh}x{d}xf32>) permutation = [1, 0, 2]
  %ctx = tensor.collapse_shape %ctx_t [[0], [1, 2]] : tensor<{s}x{nh}x{d}xf32> into tensor<{s}x{q}xf32>
  return %ctx, %{k_ssa}, %v3 : tensor<{s}x{q}xf32>, tensor<{s}x{nkv}x{d}xf32>, tensor<{s}x{nkv}x{d}xf32>
"#,
        s = s,
        q = q,
        nh = nh,
        nkv = nkv,
        d = d,
        scale = scale,
        q_ssa = q_ssa,
        k_ssa = k_ssa,
        k_full = k_full.0,
        v_full = k_full.1,
    ));

    f.finish(module)
}

fn emit_decode_helpers(module: &mut ModuleBuilder, c: &DenseDecoderConfig) -> Result<()> {
    let (mk, h, i, nh, nkv, d) = (
        c.max_kv,
        c.hidden,
        c.intermediate,
        c.num_heads,
        c.num_kv_heads,
        c.head_dim,
    );
    let (q, kv) = (c.q_dim(), c.kv_dim());
    let g = c.gqa_group();
    let scale = 1.0 / (d as f32).sqrt();

    kernels::emit_rms_norm(module, "rms_norm_tok", 1, h, c.rms_norm_eps)?;
    kernels::emit_linear(module, "linear_hq_tok", 1, h, q, true)?;
    kernels::emit_linear(module, "linear_hkv_tok", 1, h, kv, true)?;
    kernels::emit_linear(module, "linear_qh_tok", 1, q, h, true)?;
    kernels::emit_linear(module, "linear_hi_tok", 1, h, i, true)?;
    kernels::emit_linear(module, "linear_ih_tok", 1, i, h, true)?;

    if c.has_qk_norm {
        kernels::emit_rms_norm_heads(module, "rms_norm_q_heads_tok", 1, nh, d, c.rms_norm_eps)?;
        if nkv != nh {
            kernels::emit_rms_norm_heads(
                module,
                "rms_norm_kv_heads_tok",
                1,
                nkv,
                d,
                c.rms_norm_eps,
            )?;
        }
    }

    if nkv != nh {
        kernels::emit_repeat_kv(module, "repeat_kv_mk", mk, nkv, nh, d, g)?;
    }

    if let Some(theta) = c.rope_theta {
        kernels::emit_rope_at(module, "apply_rope_q_at", mk, nh, d, theta)?;
        if nkv != nh {
            kernels::emit_rope_at(module, "apply_rope_kv_at", mk, nkv, d, theta)?;
        }
    }

    {
        let mut f = module.func_private("kv_update_row");
        f.arg("cache", format!("tensor<{mk}x{nkv}x{d}xf32>"));
        f.arg("row", format!("tensor<1x{nkv}x{d}xf32>"));
        f.arg("pos_t", "tensor<i64>");
        f.result_ty(format!("tensor<{mk}x{nkv}x{d}xf32>"));
        f.op_asm(format!(
            r#"  %pos_i64 = tensor.extract %pos_t[] : tensor<i64>
  %pos = arith.index_cast %pos_i64 : i64 to index
  %c0 = arith.constant 0 : index
  %y = tensor.insert_slice %row into %cache[%pos, %c0, %c0] [1, {nkv}, {d}] [1, 1, 1] : tensor<1x{nkv}x{d}xf32> into tensor<{mk}x{nkv}x{d}xf32>
  return %y : tensor<{mk}x{nkv}x{d}xf32>"#,
            mk = mk,
            nkv = nkv,
            d = d,
        ));
        f.finish(module)?;
    }

    let mut f = module.func_private("attn_decode");
    f.arg("q", format!("tensor<1x{q}xf32>"));
    f.arg("k_cache", format!("tensor<{mk}x{nkv}x{d}xf32>"));
    f.arg("v_cache", format!("tensor<{mk}x{nkv}x{d}xf32>"));
    f.arg("k_new", format!("tensor<1x{nkv}x{d}xf32>"));
    f.arg("v_new", format!("tensor<1x{nkv}x{d}xf32>"));
    f.arg("pos_t", "tensor<i64>");
    f.arg("attn_bias", format!("tensor<{mk}xf32>"));
    if c.has_qk_norm {
        f.arg("q_norm", format!("tensor<{d}xf32>"));
        f.arg("k_norm", format!("tensor<{d}xf32>"));
    }
    f.result_ty(format!("tensor<1x{q}xf32>"));
    f.result_ty(format!("tensor<{mk}x{nkv}x{d}xf32>"));
    f.result_ty(format!("tensor<{mk}x{nkv}x{d}xf32>"));
    f.op_asm(format!(
        r#"  %pos_i64 = tensor.extract %pos_t[] : tensor<i64>
  %pos = arith.index_cast %pos_i64 : i64 to index
  %q3 = tensor.expand_shape %q [[0], [1, 2]] output_shape [1, {nh}, {d}] : tensor<1x{q}xf32> into tensor<1x{nh}x{d}xf32>
"#,
        q = q,
        d = d,
        nh = nh,
    ));

    let q_heads = if c.has_qk_norm {
        f.op_asm(format!(
            "  %qn = func.call @rms_norm_q_heads_tok(%q3, %q_norm) : (tensor<1x{nh}x{d}xf32>, tensor<{d}xf32>) -> tensor<1x{nh}x{d}xf32>\n",
            nh = nh,
            d = d,
        ));
        "qn"
    } else {
        "q3"
    };
    let k_heads = if c.has_qk_norm {
        let kn_fn = if nkv != nh {
            "rms_norm_kv_heads_tok"
        } else {
            "rms_norm_q_heads_tok"
        };
        f.op_asm(format!(
            "  %kn = func.call @{kn_fn}(%k_new, %k_norm) : (tensor<1x{nkv}x{d}xf32>, tensor<{d}xf32>) -> tensor<1x{nkv}x{d}xf32>\n",
            kn_fn = kn_fn,
            nkv = nkv,
            d = d,
        ));
        "kn"
    } else {
        "k_new"
    };

    let (q_ssa, k_ssa) = if c.rope_theta.is_some() {
        f.op_asm(format!(
            "  %qr = func.call @apply_rope_q_at(%{q_heads}, %pos_t) : (tensor<1x{nh}x{d}xf32>, tensor<i64>) -> tensor<1x{nh}x{d}xf32>\n",
            q_heads = q_heads,
            nh = nh,
            d = d,
        ));
        let rope_kv = if nkv != nh {
            "apply_rope_kv_at"
        } else {
            "apply_rope_q_at"
        };
        f.op_asm(format!(
            "  %kr = func.call @{rope_kv}(%{k_heads}, %pos_t) : (tensor<1x{nkv}x{d}xf32>, tensor<i64>) -> tensor<1x{nkv}x{d}xf32>\n",
            rope_kv = rope_kv,
            k_heads = k_heads,
            nkv = nkv,
            d = d,
        ));
        ("qr", "kr")
    } else {
        (q_heads, k_heads)
    };

    f.op_asm(format!(
        r#"  %k_upd = func.call @kv_update_row(%k_cache, %{k_ssa}, %pos_t) : (tensor<{mk}x{nkv}x{d}xf32>, tensor<1x{nkv}x{d}xf32>, tensor<i64>) -> tensor<{mk}x{nkv}x{d}xf32>
  %v_upd = func.call @kv_update_row(%v_cache, %v_new, %pos_t) : (tensor<{mk}x{nkv}x{d}xf32>, tensor<1x{nkv}x{d}xf32>, tensor<i64>) -> tensor<{mk}x{nkv}x{d}xf32>
"#,
        k_ssa = k_ssa,
        nkv = nkv,
        d = d,
        mk = mk,
    ));

    let (k_full, v_full) = if nkv != nh {
        f.op_asm(format!(
            "  %krep = func.call @repeat_kv_mk(%k_upd) : (tensor<{mk}x{nkv}x{d}xf32>) -> tensor<{mk}x{nh}x{d}xf32>\n",
            mk = mk,
            nkv = nkv,
            nh = nh,
            d = d,
        ));
        f.op_asm(format!(
            "  %vrep = func.call @repeat_kv_mk(%v_upd) : (tensor<{mk}x{nkv}x{d}xf32>) -> tensor<{mk}x{nh}x{d}xf32>\n",
            mk = mk,
            nkv = nkv,
            nh = nh,
            d = d,
        ));
        ("krep", "vrep")
    } else {
        ("k_upd", "v_upd")
    };

    f.op_asm(format!(
        r#"  %q_ti = tensor.empty() : tensor<{nh}x1x{d}xf32>
  %k_ti = tensor.empty() : tensor<{nh}x{mk}x{d}xf32>
  %v_ti = tensor.empty() : tensor<{nh}x{mk}x{d}xf32>
  %qb = linalg.transpose ins(%{q_ssa} : tensor<1x{nh}x{d}xf32>) outs(%q_ti : tensor<{nh}x1x{d}xf32>) permutation = [1, 0, 2]
  %kb = linalg.transpose ins(%{k_full} : tensor<{mk}x{nh}x{d}xf32>) outs(%k_ti : tensor<{nh}x{mk}x{d}xf32>) permutation = [1, 0, 2]
  %vb = linalg.transpose ins(%{v_full} : tensor<{mk}x{nh}x{d}xf32>) outs(%v_ti : tensor<{nh}x{mk}x{d}xf32>) permutation = [1, 0, 2]
  %kt_i = tensor.empty() : tensor<{nh}x{d}x{mk}xf32>
  %kt = linalg.transpose ins(%kb : tensor<{nh}x{mk}x{d}xf32>) outs(%kt_i : tensor<{nh}x{d}x{mk}xf32>) permutation = [0, 2, 1]
  %c0 = arith.constant 0.0 : f32
  %scale = arith.constant {scale:.8e} : f32
  %sc_i = tensor.empty() : tensor<{nh}x1x{mk}xf32>
  %sc_z = linalg.fill ins(%c0 : f32) outs(%sc_i : tensor<{nh}x1x{mk}xf32>) -> tensor<{nh}x1x{mk}xf32>
  %scores = linalg.batch_matmul ins(%qb, %kt : tensor<{nh}x1x{d}xf32>, tensor<{nh}x{d}x{mk}xf32>) outs(%sc_z : tensor<{nh}x1x{mk}xf32>) -> tensor<{nh}x1x{mk}xf32>
  %scores_s = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%scores : tensor<{nh}x1x{mk}xf32>) outs(%scores : tensor<{nh}x1x{mk}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %m = arith.mulf %a, %scale : f32
      linalg.yield %m : f32
  }} -> tensor<{nh}x1x{mk}xf32>
  %bias_i = tensor.empty() : tensor<{nh}x1x{mk}xf32>
  %bias_3d = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%attn_bias : tensor<{mk}xf32>) outs(%bias_i : tensor<{nh}x1x{mk}xf32>) {{
    ^bb0(%b: f32, %o: f32):
      linalg.yield %b : f32
  }} -> tensor<{nh}x1x{mk}xf32>
  %masked_i = tensor.empty() : tensor<{nh}x1x{mk}xf32>
  %masked = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]}}
    ins(%scores_s, %bias_3d : tensor<{nh}x1x{mk}xf32>, tensor<{nh}x1x{mk}xf32>) outs(%masked_i : tensor<{nh}x1x{mk}xf32>) {{
    ^bb0(%a: f32, %b: f32, %o: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  }} -> tensor<{nh}x1x{mk}xf32>
  %sm_i = tensor.empty() : tensor<{nh}x1x{mk}xf32>
  %attn = linalg.softmax dimension(2) ins(%masked : tensor<{nh}x1x{mk}xf32>) outs(%sm_i : tensor<{nh}x1x{mk}xf32>) -> tensor<{nh}x1x{mk}xf32>
  %ctx_i = tensor.empty() : tensor<{nh}x1x{d}xf32>
  %ctx_z = linalg.fill ins(%c0 : f32) outs(%ctx_i : tensor<{nh}x1x{d}xf32>) -> tensor<{nh}x1x{d}xf32>
  %ctx_b = linalg.batch_matmul ins(%attn, %vb : tensor<{nh}x1x{mk}xf32>, tensor<{nh}x{mk}x{d}xf32>) outs(%ctx_z : tensor<{nh}x1x{d}xf32>) -> tensor<{nh}x1x{d}xf32>
  %ctx_t_i = tensor.empty() : tensor<1x{nh}x{d}xf32>
  %ctx_t = linalg.transpose ins(%ctx_b : tensor<{nh}x1x{d}xf32>) outs(%ctx_t_i : tensor<1x{nh}x{d}xf32>) permutation = [1, 0, 2]
  %ctx = tensor.collapse_shape %ctx_t [[0], [1, 2]] : tensor<1x{nh}x{d}xf32> into tensor<1x{q}xf32>
  return %ctx, %k_upd, %v_upd : tensor<1x{q}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<{mk}x{nkv}x{d}xf32>
"#,
        q = q,
        nh = nh,
        nkv = nkv,
        d = d,
        mk = mk,
        scale = scale,
        q_ssa = q_ssa,
        k_full = k_full,
        v_full = v_full,
    ));

    f.finish(module)
}

fn emit_prefill(module: &mut ModuleBuilder, c: &DenseDecoderConfig) -> Result<()> {
    let parameters = default_parameter_lowerings();
    // Prefill is still largely `op_asm` for region-bearing linalg. Typed
    // melior/IREE dialect helpers cover single ops; they do not remove the need
    // to author the graph. See crate-level docs in `dyninfer-mlir`.

    let (s, h, v) = (c.seq, c.hidden, c.vocab);
    let (q, kv) = (c.q_dim(), c.kv_dim());
    let d = c.head_dim;
    // Tokens are left-aligned (right-padded). `%last` is the index of the
    // newest real token so RoPE/causal attention ignore trailing pads.
    let mut f = module.func(if c.paged_kv {
        "prefill_chunk"
    } else {
        "prefill"
    });
    f.arg("tokens", format!("tensor<{s}xi64>"));
    f.arg("last", "tensor<i64>");
    if c.paged_kv {
        f.arg("start_pos", "tensor<i64>");
        f.op_asm(format!(
            "  %pages = util.global.load @kv_pages : !util.list<tensor<{}x2x{}x{}x{}xf32>>\n",
            c.num_layers, PAGED_KV_PAGE_SIZE, c.num_kv_heads, c.head_dim
        ));
    }
    f.result_ty(format!("tensor<{v}xf32>"));
    emit_load_compute(
        &mut f,
        c,
        "emb_t",
        "token_embd_weight",
        "token_embd.weight",
        &format!("{v}x{h}"),
    );

    // Embedding gather (unrolled).
    f.op_asm(format!(
        "  %h_acc0 = tensor.empty() : tensor<{s}x{h}xf32>\n"
    ));
    for pos in 0..s {
        f.op_asm(format!("  %c{pos}i = arith.constant {pos} : index\n"));
        f.op_asm(format!(
            "  %t{pos} = tensor.extract %tokens[%c{pos}i] : tensor<{s}xi64>\n"
        ));
        f.op_asm(format!(
            "  %i{pos} = arith.index_cast %t{pos} : i64 to index\n"
        ));
        if !parameters.emit_embedding_call(
            &mut f,
            c,
            &format!("r{pos}"),
            &format!("i{pos}"),
            "emb_t",
            ExecutionMode::Prefill,
        )? {
            f.op_asm(format!(
                "  %r{pos} = tensor.extract_slice %emb_t[%i{pos}, 0] [1, {h}] [1, 1] : tensor<{v}x{h}xf32> to tensor<1x{h}xf32>\n"
            ));
        }
        let prev = if pos == 0 {
            "h_acc0".to_string()
        } else {
            format!("h_acc{pos}")
        };
        let next = format!("h_acc{}", pos + 1);
        f.op_asm(format!(
            "  %{next} = tensor.insert_slice %r{pos} into %{prev}[{pos}, 0] [1, {h}] [1, 1] : tensor<1x{h}xf32> into tensor<{s}x{h}xf32>\n"
        ));
    }
    let mut h_name = format!("h_acc{s}");

    for layer in 0..c.num_layers {
        let p = format!("blk{layer}");
        let n = format!("blk.{layer}");
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_attn_nw"),
            &format!("{p}_attn_norm_weight"),
            &format!("{n}.attn_norm.weight"),
            &format!("{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wq"),
            &format!("{p}_attn_q_weight"),
            &format!("{n}.attn_q.weight"),
            &format!("{q}x{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wk"),
            &format!("{p}_attn_k_weight"),
            &format!("{n}.attn_k.weight"),
            &format!("{kv}x{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wv"),
            &format!("{p}_attn_v_weight"),
            &format!("{n}.attn_v.weight"),
            &format!("{kv}x{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wo"),
            &format!("{p}_attn_output_weight"),
            &format!("{n}.attn_output.weight"),
            &format!("{h}x{q}"),
        );
        if c.has_qk_norm {
            emit_load_compute(
                &mut f,
                c,
                &format!("{p}_qnw"),
                &format!("{p}_attn_q_norm_weight"),
                &format!("{n}.attn_q_norm.weight"),
                &format!("{d}"),
            );
            emit_load_compute(
                &mut f,
                c,
                &format!("{p}_knw"),
                &format!("{p}_attn_k_norm_weight"),
                &format!("{n}.attn_k_norm.weight"),
                &format!("{d}"),
            );
        }
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_ffn_nw"),
            &format!("{p}_ffn_norm_weight"),
            &format!("{n}.ffn_norm.weight"),
            &format!("{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wgate"),
            &format!("{p}_ffn_gate_weight"),
            &format!("{n}.ffn_gate.weight"),
            &format!("{}x{h}", c.intermediate),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wup"),
            &format!("{p}_ffn_up_weight"),
            &format!("{n}.ffn_up.weight"),
            &format!("{}x{h}", c.intermediate),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wdown"),
            &format!("{p}_ffn_down_weight"),
            &format!("{n}.ffn_down.weight"),
            &format!("{h}x{}", c.intermediate),
        );

        let xin = h_name.clone();
        f.op_asm(format!(
            "  %{p}_xn = func.call @rms_norm(%{xin}, %{p}_attn_nw) : (tensor<{s}x{h}xf32>, tensor<{h}xf32>) -> tensor<{s}x{h}xf32>\n"
        ));
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_q"),
            "linear_hq",
            &format!("{p}_xn"),
            &format!("{p}_wq"),
            &format!("{n}.attn_q.weight"),
            s,
            h,
            q,
        );
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_k"),
            "linear_hkv",
            &format!("{p}_xn"),
            &format!("{p}_wk"),
            &format!("{n}.attn_k.weight"),
            s,
            h,
            kv,
        );
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_v"),
            "linear_hkv",
            &format!("{p}_xn"),
            &format!("{p}_wv"),
            &format!("{n}.attn_v.weight"),
            s,
            h,
            kv,
        );
        let nkv = c.num_kv_heads;
        let mk = c.max_kv;
        if c.paged_kv {
            let g = c.gqa_group();
            let qg_ty = format!("tensor<{nkv}x{g}x{s}x{d}xf32>");
            let kv3_ty = format!("tensor<{s}x{nkv}x{d}xf32>");
            let list_ty = format!(
                "!util.list<tensor<{}x2x{}x{}x{}xf32>>",
                c.num_layers, PAGED_KV_PAGE_SIZE, nkv, d
            );
            let norm_args = if c.has_qk_norm {
                format!(", %{p}_qnw, %{p}_knw")
            } else {
                String::new()
            };
            let norm_types = if c.has_qk_norm {
                format!(", tensor<{d}xf32>, tensor<{d}xf32>")
            } else {
                String::new()
            };
            f.op_asm(format!(
                "  %{p}_qg, %{p}_kc, %{p}_vc = func.call @prepare_paged_attention(%{p}_q, %{p}_k, %{p}_v, %start_pos{norm_args}) : (tensor<{s}x{q}xf32>, tensor<{s}x{kv}xf32>, tensor<{s}x{kv}xf32>, tensor<i64>{norm_types}) -> ({qg_ty}, {kv3_ty}, {kv3_ty})\n"
            ));
            f.op_asm(format!(
                "  func.call @store_paged_kv_{layer}(%{p}_kc, %{p}_vc, %start_pos, %pages) : ({kv3_ty}, {kv3_ty}, tensor<i64>, {list_ty}) -> ()\n"
            ));
            f.op_asm(format!(
                "  %{p}_ctxg = func.call @attend_pages_{layer}(%{p}_qg, %start_pos, %pages) : ({qg_ty}, tensor<i64>, {list_ty}) -> {qg_ty}\n"
            ));
            f.op_asm(format!(
                "  %{p}_ctx4e = tensor.empty() : tensor<{s}x{nkv}x{g}x{d}xf32>\n"
            ));
            f.op_asm(format!(
                "  %{p}_ctx4 = linalg.transpose ins(%{p}_ctxg : {qg_ty}) outs(%{p}_ctx4e : tensor<{s}x{nkv}x{g}x{d}xf32>) permutation = [2, 0, 1, 3]\n"
            ));
            f.op_asm(format!(
                "  %{p}_ctx = tensor.collapse_shape %{p}_ctx4 [[0], [1, 2, 3]] : tensor<{s}x{nkv}x{g}x{d}xf32> into tensor<{s}x{q}xf32>\n"
            ));
        } else {
            if c.has_qk_norm {
                f.op_asm(format!(
                    "  %{p}_ctx, %{p}_kc, %{p}_vc = func.call @attn(%{p}_q, %{p}_k, %{p}_v, %{p}_qnw, %{p}_knw) : (tensor<{s}x{q}xf32>, tensor<{s}x{kv}xf32>, tensor<{s}x{kv}xf32>, tensor<{d}xf32>, tensor<{d}xf32>) -> (tensor<{s}x{q}xf32>, tensor<{s}x{nkv}x{d}xf32>, tensor<{s}x{nkv}x{d}xf32>)\n"
                ));
            } else {
                f.op_asm(format!(
                    "  %{p}_ctx, %{p}_kc, %{p}_vc = func.call @attn(%{p}_q, %{p}_k, %{p}_v) : (tensor<{s}x{q}xf32>, tensor<{s}x{kv}xf32>, tensor<{s}x{kv}xf32>) -> (tensor<{s}x{q}xf32>, tensor<{s}x{nkv}x{d}xf32>, tensor<{s}x{nkv}x{d}xf32>)\n"
                ));
            }
            // Seed mutable KV at positions [0, seq).
            f.op_asm(format!(
                "  %{p}_k_z = tensor.empty() : tensor<{mk}x{nkv}x{d}xf32>\n  %{p}_v_z = tensor.empty() : tensor<{mk}x{nkv}x{d}xf32>\n  %{p}_k_f0 = arith.constant 0.0 : f32\n"
            ));
            f.op_asm(format!(
                "  %{p}_k_old = linalg.fill ins(%{p}_k_f0 : f32) outs(%{p}_k_z : tensor<{mk}x{nkv}x{d}xf32>) -> tensor<{mk}x{nkv}x{d}xf32>\n  %{p}_v_old = linalg.fill ins(%{p}_k_f0 : f32) outs(%{p}_v_z : tensor<{mk}x{nkv}x{d}xf32>) -> tensor<{mk}x{nkv}x{d}xf32>\n"
            ));
            f.op_asm(format!(
                "  %{p}_k_new = tensor.insert_slice %{p}_kc into %{p}_k_old[0, 0, 0] [{s}, {nkv}, {d}] [1, 1, 1] : tensor<{s}x{nkv}x{d}xf32> into tensor<{mk}x{nkv}x{d}xf32>\n  %{p}_v_new = tensor.insert_slice %{p}_vc into %{p}_v_old[0, 0, 0] [{s}, {nkv}, {d}] [1, 1, 1] : tensor<{s}x{nkv}x{d}xf32> into tensor<{mk}x{nkv}x{d}xf32>\n"
            ));
            f.op_asm(format!(
                "  util.global.store %{p}_k_new, @kv_k{layer} : tensor<{mk}x{nkv}x{d}xf32>\n  util.global.store %{p}_v_new, @kv_v{layer} : tensor<{mk}x{nkv}x{d}xf32>\n"
            ));
        }
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_o"),
            "linear_qh",
            &format!("{p}_ctx"),
            &format!("{p}_wo"),
            &format!("{n}.attn_output.weight"),
            s,
            q,
            h,
        );
        f.op_asm(format!(
            "  %{p}_h2 = arith.addf %{xin}, %{p}_o : tensor<{s}x{h}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_fn = func.call @rms_norm(%{p}_h2, %{p}_ffn_nw) : (tensor<{s}x{h}xf32>, tensor<{h}xf32>) -> tensor<{s}x{h}xf32>\n"
        ));
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_gate"),
            "linear_hi",
            &format!("{p}_fn"),
            &format!("{p}_wgate"),
            &format!("{n}.ffn_gate.weight"),
            s,
            h,
            c.intermediate,
        );
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_up"),
            "linear_hi",
            &format!("{p}_fn"),
            &format!("{p}_wup"),
            &format!("{n}.ffn_up.weight"),
            s,
            h,
            c.intermediate,
        );
        f.op_asm(format!(
            r#"  %{p}_silu = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}}
    ins(%{p}_gate : tensor<{s}x{i}xf32>) outs(%{p}_gate : tensor<{s}x{i}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %n = arith.negf %a : f32
      %e = math.exp %n : f32
      %one = arith.constant 1.0 : f32
      %den = arith.addf %one, %e : f32
      %sg = arith.divf %a, %den : f32
      linalg.yield %sg : f32
  }} -> tensor<{s}x{i}xf32>
"#,
            i = c.intermediate
        ));
        f.op_asm(format!(
            "  %{p}_ff = arith.mulf %{p}_silu, %{p}_up : tensor<{s}x{i}xf32>\n",
            i = c.intermediate
        ));
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_down"),
            "linear_ih",
            &format!("{p}_ff"),
            &format!("{p}_wdown"),
            &format!("{n}.ffn_down.weight"),
            s,
            c.intermediate,
            h,
        );
        f.op_asm(format!(
            "  %{p}_hout = arith.addf %{p}_h2, %{p}_down : tensor<{s}x{h}xf32>\n"
        ));
        h_name = format!("{p}_hout");
    }

    emit_load_compute(
        &mut f,
        c,
        "out_nw",
        "output_norm_weight",
        "output_norm.weight",
        &format!("{h}"),
    );
    emit_load_compute(
        &mut f,
        c,
        "wout",
        "output_weight",
        "output.weight",
        &format!("{v}x{h}"),
    );
    f.op_asm("  %last_i64 = tensor.extract %last[] : tensor<i64>\n");
    f.op_asm("  %li = arith.index_cast %last_i64 : i64 to index\n");
    f.op_asm(format!(
        "  %last_row = tensor.extract_slice %{h_name}[%li, 0] [1, {h}] [1, 1] : tensor<{s}x{h}xf32> to tensor<1x{h}xf32>\n"
    ));
    // Tile to S for rms_norm helper, take row 0.
    f.op_asm(format!(
        "  %last_tile = tensor.empty() : tensor<{s}x{h}xf32>\n"
    ));
    f.op_asm(format!(
        "  %last_s = tensor.insert_slice %last_row into %last_tile[0, 0] [1, {h}] [1, 1] : tensor<1x{h}xf32> into tensor<{s}x{h}xf32>\n"
    ));
    f.op_asm(format!(
        "  %ln = func.call @rms_norm(%last_s, %out_nw) : (tensor<{s}x{h}xf32>, tensor<{h}xf32>) -> tensor<{s}x{h}xf32>\n"
    ));
    f.op_asm(format!(
        "  %ln1 = tensor.extract_slice %ln[0, 0] [1, {h}] [1, 1] : tensor<{s}x{h}xf32> to tensor<1x{h}xf32>\n"
    ));
    emit_output_proj(&mut f, c, "ln1", "wout", ExecutionMode::Prefill);
    f.op_asm(format!("  return %logits : tensor<{v}xf32>"));

    f.finish(module)
}

fn emit_decode(module: &mut ModuleBuilder, c: &DenseDecoderConfig) -> Result<()> {
    let parameters = default_parameter_lowerings();
    let (h, v, mk) = (c.hidden, c.vocab, c.max_kv);
    let (q, kv) = (c.q_dim(), c.kv_dim());
    let d = c.head_dim;
    let nkv = c.num_kv_heads;
    let mut f = module.func("decode");
    f.arg("token", "tensor<i64>");
    f.arg("pos", "tensor<i64>");
    f.arg("attn_bias", format!("tensor<{mk}xf32>"));
    f.result_ty(format!("tensor<{v}xf32>"));
    emit_load_compute(
        &mut f,
        c,
        "emb_t",
        "token_embd_weight",
        "token_embd.weight",
        &format!("{v}x{h}"),
    );
    f.op_asm("  %tok = tensor.extract %token[] : tensor<i64>\n");
    f.op_asm("  %ti = arith.index_cast %tok : i64 to index\n");
    if !parameters.emit_embedding_call(&mut f, c, "row", "ti", "emb_t", ExecutionMode::Decode)? {
        f.op_asm(format!(
            "  %row = tensor.extract_slice %emb_t[%ti, 0] [1, {h}] [1, 1] : tensor<{v}x{h}xf32> to tensor<1x{h}xf32>\n"
        ));
    }
    let mut h_name = "row".to_string();

    for layer in 0..c.num_layers {
        let p = format!("blk{layer}");
        let n = format!("blk.{layer}");
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_attn_nw"),
            &format!("{p}_attn_norm_weight"),
            &format!("{n}.attn_norm.weight"),
            &format!("{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wq"),
            &format!("{p}_attn_q_weight"),
            &format!("{n}.attn_q.weight"),
            &format!("{q}x{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wk"),
            &format!("{p}_attn_k_weight"),
            &format!("{n}.attn_k.weight"),
            &format!("{kv}x{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wv"),
            &format!("{p}_attn_v_weight"),
            &format!("{n}.attn_v.weight"),
            &format!("{kv}x{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wo"),
            &format!("{p}_attn_output_weight"),
            &format!("{n}.attn_output.weight"),
            &format!("{h}x{q}"),
        );
        if c.has_qk_norm {
            emit_load_compute(
                &mut f,
                c,
                &format!("{p}_qnw"),
                &format!("{p}_attn_q_norm_weight"),
                &format!("{n}.attn_q_norm.weight"),
                &format!("{d}"),
            );
            emit_load_compute(
                &mut f,
                c,
                &format!("{p}_knw"),
                &format!("{p}_attn_k_norm_weight"),
                &format!("{n}.attn_k_norm.weight"),
                &format!("{d}"),
            );
        }
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_ffn_nw"),
            &format!("{p}_ffn_norm_weight"),
            &format!("{n}.ffn_norm.weight"),
            &format!("{h}"),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wgate"),
            &format!("{p}_ffn_gate_weight"),
            &format!("{n}.ffn_gate.weight"),
            &format!("{}x{h}", c.intermediate),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wup"),
            &format!("{p}_ffn_up_weight"),
            &format!("{n}.ffn_up.weight"),
            &format!("{}x{h}", c.intermediate),
        );
        emit_load_compute(
            &mut f,
            c,
            &format!("{p}_wdown"),
            &format!("{p}_ffn_down_weight"),
            &format!("{n}.ffn_down.weight"),
            &format!("{h}x{}", c.intermediate),
        );

        let xin = h_name.clone();
        f.op_asm(format!(
            "  %{p}_xn = func.call @rms_norm_tok(%{xin}, %{p}_attn_nw) : (tensor<1x{h}xf32>, tensor<{h}xf32>) -> tensor<1x{h}xf32>\n"
        ));
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_q"),
            "linear_hq_tok",
            &format!("{p}_xn"),
            &format!("{p}_wq"),
            &format!("{n}.attn_q.weight"),
            1,
            h,
            q,
        );
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_k"),
            "linear_hkv_tok",
            &format!("{p}_xn"),
            &format!("{p}_wk"),
            &format!("{n}.attn_k.weight"),
            1,
            h,
            kv,
        );
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_v"),
            "linear_hkv_tok",
            &format!("{p}_xn"),
            &format!("{p}_wv"),
            &format!("{n}.attn_v.weight"),
            1,
            h,
            kv,
        );
        f.op_asm(format!(
            "  %{p}_k3 = tensor.expand_shape %{p}_k [[0], [1, 2]] output_shape [1, {nkv}, {d}] : tensor<1x{kv}xf32> into tensor<1x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_v3 = tensor.expand_shape %{p}_v [[0], [1, 2]] output_shape [1, {nkv}, {d}] : tensor<1x{kv}xf32> into tensor<1x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_k_old = util.global.load @kv_k{layer} : tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_v_old = util.global.load @kv_v{layer} : tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        if c.has_qk_norm {
            f.op_asm(format!(
                "  %{p}_ctx, %{p}_k_new, %{p}_v_new = func.call @attn_decode(%{p}_q, %{p}_k_old, %{p}_v_old, %{p}_k3, %{p}_v3, %pos, %attn_bias, %{p}_qnw, %{p}_knw) : (tensor<1x{q}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<1x{nkv}x{d}xf32>, tensor<1x{nkv}x{d}xf32>, tensor<i64>, tensor<{mk}xf32>, tensor<{d}xf32>, tensor<{d}xf32>) -> (tensor<1x{q}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<{mk}x{nkv}x{d}xf32>)\n"
            ));
        } else {
            f.op_asm(format!(
                "  %{p}_ctx, %{p}_k_new, %{p}_v_new = func.call @attn_decode(%{p}_q, %{p}_k_old, %{p}_v_old, %{p}_k3, %{p}_v3, %pos, %attn_bias) : (tensor<1x{q}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<1x{nkv}x{d}xf32>, tensor<1x{nkv}x{d}xf32>, tensor<i64>, tensor<{mk}xf32>) -> (tensor<1x{q}xf32>, tensor<{mk}x{nkv}x{d}xf32>, tensor<{mk}x{nkv}x{d}xf32>)\n"
            ));
        }
        f.op_asm(format!(
            "  util.global.store %{p}_k_new, @kv_k{layer} : tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        f.op_asm(format!(
            "  util.global.store %{p}_v_new, @kv_v{layer} : tensor<{mk}x{nkv}x{d}xf32>\n"
        ));
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_o"),
            "linear_qh_tok",
            &format!("{p}_ctx"),
            &format!("{p}_wo"),
            &format!("{n}.attn_output.weight"),
            1,
            q,
            h,
        );
        f.op_asm(format!(
            "  %{p}_h2 = arith.addf %{xin}, %{p}_o : tensor<1x{h}xf32>\n"
        ));
        f.op_asm(format!(
            "  %{p}_fn = func.call @rms_norm_tok(%{p}_h2, %{p}_ffn_nw) : (tensor<1x{h}xf32>, tensor<{h}xf32>) -> tensor<1x{h}xf32>\n"
        ));
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_gate"),
            "linear_hi_tok",
            &format!("{p}_fn"),
            &format!("{p}_wgate"),
            &format!("{n}.ffn_gate.weight"),
            1,
            h,
            c.intermediate,
        );
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_up"),
            "linear_hi_tok",
            &format!("{p}_fn"),
            &format!("{p}_wup"),
            &format!("{n}.ffn_up.weight"),
            1,
            h,
            c.intermediate,
        );
        f.op_asm(format!(
            r#"  %{p}_silu = linalg.generic {{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}}
    ins(%{p}_gate : tensor<1x{i}xf32>) outs(%{p}_gate : tensor<1x{i}xf32>) {{
    ^bb0(%a: f32, %b: f32):
      %n = arith.negf %a : f32
      %e = math.exp %n : f32
      %one = arith.constant 1.0 : f32
      %den = arith.addf %one, %e : f32
      %sg = arith.divf %a, %den : f32
      linalg.yield %sg : f32
  }} -> tensor<1x{i}xf32>
"#,
            i = c.intermediate
        ));
        f.op_asm(format!(
            "  %{p}_ff = arith.mulf %{p}_silu, %{p}_up : tensor<1x{i}xf32>\n",
            i = c.intermediate
        ));
        emit_linear_call(
            &mut f,
            c,
            &format!("{p}_down"),
            "linear_ih_tok",
            &format!("{p}_ff"),
            &format!("{p}_wdown"),
            &format!("{n}.ffn_down.weight"),
            1,
            c.intermediate,
            h,
        );
        f.op_asm(format!(
            "  %{p}_hout = arith.addf %{p}_h2, %{p}_down : tensor<1x{h}xf32>\n"
        ));
        h_name = format!("{p}_hout");
    }

    emit_load_compute(
        &mut f,
        c,
        "out_nw",
        "output_norm_weight",
        "output_norm.weight",
        &format!("{h}"),
    );
    emit_load_compute(
        &mut f,
        c,
        "wout",
        "output_weight",
        "output.weight",
        &format!("{v}x{h}"),
    );
    f.op_asm(format!(
        "  %ln = func.call @rms_norm_tok(%{h_name}, %out_nw) : (tensor<1x{h}xf32>, tensor<{h}xf32>) -> tensor<1x{h}xf32>\n"
    ));
    emit_output_proj(&mut f, c, "ln", "wout", ExecutionMode::Decode);
    f.op_asm(format!("  return %logits : tensor<{v}xf32>"));

    f.finish(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_0_6b_shape_supported() {
        let c = DenseDecoderConfig {
            vocab: 151_936,
            hidden: 1024,
            intermediate: 3072,
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 128,
            num_layers: 28,
            seq: LARGE_PREFILL_WINDOW,
            max_kv: LARGE_MAX_KV,
            paged_kv: false,
            iree_flash_attention: false,
            rms_norm_eps: 1e-6,
            rope_theta: Some(1_000_000.0),
            has_qk_norm: true,
            param_keys: BTreeMap::new(),
            param_dtypes: BTreeMap::from([("token_embd.weight".into(), ScalarType::Bf16)]),
            param_compute_dtypes: BTreeMap::new(),
            param_bindings: BTreeMap::new(),
            param_lowerings: BTreeMap::new(),
            separate_storage_casts: false,
            dense_index_constants: false,
        };
        assert!(c.supports_dense_emit());
        assert_eq!(c.q_dim(), 2048);
        assert_eq!(c.kv_dim(), 1024);
        assert_eq!(c.gqa_group(), 2);
        assert_eq!(c.param_dtype("token_embd.weight"), ScalarType::Bf16);
        assert_eq!(c.param_dtype("missing.weight"), ScalarType::F32);
    }

    #[test]
    fn tiny_mha_still_supported() {
        let mut c = DenseDecoderConfig {
            vocab: 32,
            hidden: 64,
            intermediate: 128,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            num_layers: 1,
            seq: TINY_PREFILL_WINDOW,
            max_kv: TINY_MAX_KV,
            paged_kv: false,
            iree_flash_attention: false,
            rms_norm_eps: 1e-5,
            rope_theta: None,
            has_qk_norm: false,
            param_keys: BTreeMap::new(),
            param_dtypes: BTreeMap::new(),
            param_compute_dtypes: BTreeMap::new(),
            param_bindings: BTreeMap::new(),
            param_lowerings: BTreeMap::new(),
            separate_storage_casts: false,
            dense_index_constants: false,
        };
        assert!(c.supports_dense_emit());
        assert!(c.is_synthetic_fixture());
        c.paged_kv = true;
        c.max_kv = 1024;
        let mlir = emit_dense_decoder_cfg("test.paged", &c).expect("paged MLIR verifies");
        assert!(mlir.contains("@prefill_chunk"));
        assert!(mlir.contains("@decode_chunk"));
        assert!(mlir.contains("func.func private @layer_page"));
        assert!(!mlir.contains("!util.list<!hal.buffer_view>"));
        // Chunk KV + attn state returned as SSA from prepare (not via util.global).
        assert!(mlir.contains("@paged_attn_max") || mlir.contains("@paged_decode_attn_max"));
        assert!(mlir.contains("@chunk_begin"));
        assert!(!mlir.contains("tensor<4x4096x4096"));
        crate::compile_mlir_prefer_inprocess(
            &mlir,
            &dyninfer_core::TargetProfile::llvm_cpu_host(),
            false,
            None,
        )
        .expect("paged MLIR compiles through IREE");
    }

    #[test]
    fn emit_uses_per_tensor_checkpoint_dtypes() {
        let mut param_dtypes = BTreeMap::new();
        param_dtypes.insert("token_embd.weight".into(), ScalarType::Bf16);
        param_dtypes.insert("blk.0.attn_norm.weight".into(), ScalarType::F16);
        param_dtypes.insert("blk.0.attn_q.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.attn_k.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.attn_v.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.attn_output.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.ffn_norm.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.ffn_gate.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.ffn_up.weight".into(), ScalarType::F32);
        param_dtypes.insert("blk.0.ffn_down.weight".into(), ScalarType::F32);
        param_dtypes.insert("output_norm.weight".into(), ScalarType::F32);
        param_dtypes.insert("output.weight".into(), ScalarType::Bf16);
        let param_keys = BTreeMap::from([(
            "token_embd.weight".into(),
            "weights::token_embd.weight::data".into(),
        )]);
        let c = DenseDecoderConfig {
            vocab: 32,
            hidden: 64,
            intermediate: 128,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            num_layers: 1,
            seq: TINY_PREFILL_WINDOW,
            max_kv: TINY_MAX_KV,
            paged_kv: false,
            iree_flash_attention: false,
            rms_norm_eps: 1e-5,
            rope_theta: None,
            has_qk_norm: false,
            param_keys,
            param_dtypes,
            param_compute_dtypes: BTreeMap::new(),
            param_bindings: BTreeMap::new(),
            param_lowerings: BTreeMap::new(),
            separate_storage_casts: false,
            dense_index_constants: false,
        };
        let mlir = emit_dense_decoder_cfg("test.decoder", &c).expect("mlir verify");
        assert!(
            mlir.contains("@token_embd_weight") && mlir.contains("tensor<32x64xbf16>"),
            "missing bf16 token embd global: {mlir}"
        );
        assert!(mlir.contains(
            "#stream.parameter.named<\"weights\"::\"weights::token_embd.weight::data\">"
        ));
        assert!(!mlir.contains("q4_linear"));
        assert!(
            mlir.contains("@blk0_attn_norm_weight") && mlir.contains("tensor<64xf16>"),
            "missing f16 attn norm global: {mlir}"
        );
        assert!(
            mlir.contains("@blk0_attn_q_weight") && mlir.contains("tensor<64x64xf32>"),
            "missing f32 q weight global: {mlir}"
        );
        assert!(
            mlir.contains("arith.extf") && mlir.contains("bf16") && mlir.contains("f32"),
            "missing bf16→f32 cast: {mlir}"
        );
        assert!(
            mlir.contains("@prefill") || mlir.contains("func.func @prefill"),
            "verified module missing prefill: {mlir}"
        );
    }

    #[test]
    fn tiny_gqa_rope_decode_mlir_indexes_pos() {
        let c = DenseDecoderConfig {
            vocab: 32,
            hidden: 64,
            intermediate: 128,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            num_layers: 1,
            seq: 4,
            max_kv: 8,
            paged_kv: false,
            iree_flash_attention: false,
            rms_norm_eps: 1e-5,
            rope_theta: Some(10000.0),
            has_qk_norm: true,
            param_keys: BTreeMap::new(),
            param_dtypes: BTreeMap::new(),
            param_compute_dtypes: BTreeMap::new(),
            param_bindings: BTreeMap::new(),
            param_lowerings: BTreeMap::new(),
            separate_storage_casts: false,
            dense_index_constants: false,
        };
        let mlir = emit_dense_decoder_cfg("test.gqa", &c).expect("emit");
        assert!(mlir.contains("func.func private @attn_decode"));
        assert!(mlir.contains("func.func private @apply_rope_kv_at"));
        let rope = mlir
            .split("func.func private @apply_rope_kv_at")
            .nth(1)
            .expect("apply_rope_kv_at");
        let rope = rope.split("func.func").next().unwrap();
        assert!(
            rope.contains("pos_t") || rope.contains("%arg1"),
            "rope_at must take tensor<i64> pos"
        );
        let dec = mlir
            .split("func.func private @attn_decode")
            .nth(1)
            .expect("attn_decode");
        let dec = dec.split("func.func").next().unwrap();
        assert!(dec.contains("repeat_kv_mk") || dec.contains("@repeat_kv_mk"));
        // Prefill must zero-fill KV before insert (avoids garbage when max_kv > seq).
        assert!(
            mlir.contains("linalg.fill") && mlir.contains("kv_k0"),
            "prefill should zero-fill KV globals before seeding"
        );
    }
}
