# Dynamic Inference Engine

**Technical Architecture and Implementation Specification**  
**Status:** Draft 0.1  
**Last updated:** 2026-07-25  
**Primary implementation language:** Rust 2024 edition  
**Build system:** Bazel with `rules_rs`  
**Compiler backend:** MLIR + IREE  
**Initial deployment targets:** Linux x86-64 and AArch64; LLVM CPU and Vulkan GPU

---

## 1. Executive summary

Dynamic Inference Engine is a checkpoint-specializing compiler and local inference runtime. It accepts:

1. A model architecture package that describes model semantics without embedding learned parameters.
2. An unmodified checkpoint, initially SafeTensors, GGUF, or MLX-compatible NPZ/SafeTensors.
3. A target hardware and execution profile.

It produces an IREE VM FlatBuffer (`.vmfb`) executable that resolves model parameters from the original checkpoint or from an optional derived parameter cache. The executable exports inference entrypoints such as `prefill` and `decode` and is invoked by a Rust runtime through IREE's C runtime API.

The core compilation equation is:

```text
Architecture IR
  + Checkpoint catalog and physical tensor encodings
  + Target and shape profile
  = Specialized IREE executable
```

The system is designed around one important boundary: the **Bound Model IR**. This is the point where logical model operations are associated with actual checkpoint tensors, physical encodings, packing schemes, layouts, and auxiliary scale or zero-point tensors.

Most of the product is implemented in Rust. A small C++ library implements custom MLIR dialects and compiler passes, because MLIR and IREE compiler extension APIs are most complete in C++. Rust communicates with that compiler library through a project-owned C ABI. Runtime execution uses IREE's native C API directly from Rust.

---

## 2. Normative terminology

The keywords **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are normative.

- **Architecture**: Logical model computation and parameter roles, independent of a checkpoint's byte representation.
- **Checkpoint container**: A file structure such as SafeTensors, GGUF, NPZ, or legacy GGML.
- **Checkpoint convention**: The semantic interpretation of tensor names, metadata, packing, auxiliary tensors, and quantization layout.
- **Logical tensor**: The mathematical tensor visible to the model architecture.
- **Physical tensor**: The stored bytes and metadata used to represent a logical tensor.
- **Binding**: Association of an architecture parameter slot with one or more physical checkpoint entries.
- **Materialization**: Creation of derived parameter bytes, such as a transpose, concatenation, alignment copy, or hardware-specific repack.
- **Architecture package**: Portable artifact containing canonical architecture MLIR and metadata.
- **Executable bundle**: VMFB executable, manifests, and optional derived parameters.
- **AOT**: Compilation before deployment to the target host.
- **First-load JIT**: Compilation performed when a checkpoint is first opened on a local target and cached for subsequent runs.

---

## 3. Goals

### 3.1 Functional goals

The engine MUST:

- Load an architecture independently from learned parameters.
- Inspect checkpoint metadata without eagerly loading all tensor bytes.
- Preserve the original checkpoint file unchanged.
- Bind architecture parameters to checkpoint entries with explicit validation.
- Specialize model operations according to actual tensor encodings and layouts.
- Generate separate optimized entrypoints for prefill and decode.
- Compile through IREE for local CPU and GPU execution.
- Support both AOT and first-load JIT compilation.
- Cache compiled executables independently from checkpoint values when schemas are compatible.
- Support optional content-dependent derived parameter caches.
- Expose a safe, model-oriented Rust runtime API.

### 3.2 Performance goals

The engine SHOULD:

- Avoid full dequantization of low-bit weights when a fused consuming kernel is available.
- Avoid unnecessary host-device synchronization.
- Reuse buffers and KV-cache allocations across decode steps.
- Produce a correct generic generated implementation before requiring a handwritten fast path.
- Allow hardware-specific generated and external kernels without changing architecture definitions.
- Make compilation and parameter materialization observable and cacheable.

### 3.3 Engineering goals

The project MUST:

- Use Bazel and Bzlmod.
- Use `rules_rs` for Rust toolchains, Rust targets, and Cargo dependency import.
- Pin IREE, LLVM/MLIR, `rules_rs`, Rust, and target toolchains.
- Keep unsafe Rust isolated in FFI crates.
- Keep IREE-specific types out of the public model runtime API.
- Make all generated artifacts reproducible from declared inputs.

---

## 4. Non-goals for version 1

Version 1 does not attempt to:

- Compile arbitrary Python at inference runtime.
- Correctly import every arbitrary Transformers model implementation.
- Support dynamically loaded Rust plugins with an unstable Rust ABI.
- Match the best handwritten kernel for every model, quantization, and GPU.
- Implement distributed inference or multi-node execution.
- Implement training, autograd, or fine-tuning.
- Implement arbitrary dynamic control flow inside model architectures.
- Guarantee a stable public MLIR dialect across all future major versions.
- Reimplement IREE's HAL, VM, scheduling, or device runtime.
- Reimplement tokenizer ecosystems inside the compiler.

---

## 5. Design principles

### 5.1 Checkpoint formats are storage, not model semantics

A SafeTensors dtype and shape do not uniquely identify a quantization algorithm. A convention decoder must interpret names, metadata, packing, auxiliary tensors, and model configuration.

### 5.2 Architecture and checkpoint ingestion are independent

The architecture frontend MUST NOT depend directly on SafeTensors, GGUF, or MLX APIs. Checkpoint plugins MUST NOT contain model execution code.

### 5.3 Quantization is an encoding contract, not merely a dtype

A physical encoding includes:

- Logical element type.
- Storage element type.
- Packing and bit order.
- Quantization axis.
- Group or block shape.
- Scale representation.
- Zero-point behavior.
- Auxiliary tensor references.
- Logical orientation and storage layout.
- Alignment and byte-range constraints.

### 5.4 Correctness path before fast path

Every supported operation and encoding combination MUST have a portable correctness implementation or be rejected at compile time. Optimized kernels are optional candidates selected by target constraints and cost models.

### 5.5 Original checkpoints remain immutable

The engine MUST never rewrite the input checkpoint in place. Derived parameters are stored separately in a content-addressed cache or explicit output bundle.

### 5.6 Runtime does not know model architecture

The runtime invokes a stable exported inference ABI. Architecture-specific logic belongs in compiled code and manifests, not in the token generation loop.

### 5.7 High-level specialization, low-level delegation

The engine owns architecture semantics, physical encodings, binding, and kernel candidate selection. IREE owns dispatch formation, device lowering, executable packaging, and runtime execution.

---

## 6. High-level architecture

```text
                         Compilation side

 Architecture source                       Checkpoint file(s)
 +----------------------+                  +----------------------+
 | Rust architecture DSL|                  | SafeTensors          |
 | Canonical Model MLIR |                  | GGUF                 |
 | Imported graph       |                  | MLX NPZ/SafeTensors  |
 +----------+-----------+                  +----------+-----------+
            |                                         |
            v                                         v
   Architecture package                       Raw checkpoint index
            |                                         |
            |                              Convention decoder
            |                                         |
            +----------------+------------------------+
                             v
                    Binding and validation
                             v
                       Bound Model IR
                             v
              Encoding and shape specialization
                             v
                 Kernel candidate selection
                             v
         Standard MLIR + IREE-compatible extensions
                             v
                      IREE compilation
                             v
                 VMFB + execution manifest

                         Runtime side

 VMFB + original checkpoint + optional parameter cache
                             v
             Rust IREE runtime and parameter providers
                             v
                  prefill / decode / sampling
```

### 6.1 Process boundaries

The reference implementation has two optional processes:

- `dyninfer` runtime and CLI, implemented in Rust.
- `dyninfer-compiler-worker`, an optional isolated compiler process containing the C++ MLIR/IREE compiler extension.

For development and low-latency first-load JIT, the compiler MAY be linked in-process. For untrusted architecture packages or daemon deployments, compilation SHOULD run in a restricted worker process.

---

## 7. Artifact model

### 7.1 Architecture package

An architecture package is a directory or archive:

```text
llama.arch/
|-- manifest.json
|-- graph.mlirbc
|-- config.schema.json
|-- parameter-schema.json
`-- provenance.json
```

`manifest.json` example:

```json
{
  "format": "dyninfer.architecture",
  "version": 1,
  "architecture_id": "llama.decoder",
  "architecture_revision": "1.0.0",
  "model_dialect_version": 1,
  "entrypoint_templates": ["prefill", "decode"],
  "config_schema": "config.schema.json",
  "graph": "graph.mlirbc",
  "parameter_schema": "parameter-schema.json"
}
```

The package MUST NOT contain executable native code. It MAY contain MLIR bytecode using the engine's versioned input dialects.

### 7.2 Checkpoint catalog

A checkpoint catalog is produced by metadata-only inspection:

```rust
pub struct CheckpointCatalog {
    pub container: ContainerIdentity,
    pub source_files: Vec<SourceFile>,
    pub metadata: MetadataMap,
    pub raw_entries: Vec<RawTensorEntry>,
    pub parameters: Vec<LogicalParameter>,
    pub schema_fingerprint: SchemaFingerprint,
}
```

The catalog MUST include enough information to bind and compile without reading all weight payloads.

### 7.3 Binding plan

```rust
pub struct BindingPlan {
    pub architecture_id: ArchitectureId,
    pub checkpoint_schema: SchemaFingerprint,
    pub bindings: Vec<ParameterBinding>,
    pub unresolved_optional_slots: Vec<ParameterSlotId>,
    pub materializations: Vec<MaterializationRequest>,
}
```

### 7.4 Executable bundle

```text
model.bundle/
|-- manifest.json
|-- bindings.json
|-- checkpoint-schema.json
|-- executables/
|   |-- vulkan-generic.vmfb
|   |-- llvm-cpu-x86_64.vmfb
|   `-- rocm-gfx1151.vmfb
|-- tuning/
|   `-- tuning-spec.mlirbc
`-- parameters/
    `-- optional-derived.irpa
```

A bundle MAY reference an external original checkpoint path at invocation time. The path is never embedded as a required absolute path in the VMFB; parameters are resolved by scope and key.

---

## 8. Canonical architecture representation

### 8.1 Canonical form

The canonical architecture representation is MLIR bytecode using engine-owned input dialects:

- `dyninfer.model`: Logical model operations.
- `dyninfer.checkpoint`: Symbolic parameter slots and physical bindings.
- `dyninfer.qkernel`: Encoding-aware operations before generic lowering.

The engine SHOULD keep these dialects small. Operations SHOULD lower to standard MLIR and IREE-supported dialects as early as practical.

### 8.2 Model dialect operations

Initial operations:

```text
dyninfer.model.embedding
dyninfer.model.linear
dyninfer.model.rms_norm
dyninfer.model.rope
dyninfer.model.attention
dyninfer.model.activation
dyninfer.model.kv_cache_read
dyninfer.model.kv_cache_write
dyninfer.model.reshape_heads
dyninfer.model.logits
```

Conceptual example:

```mlir
module attributes {
  dyninfer.architecture_id = "llama.decoder"
} {
  dyninfer.model.parameter @tok_embeddings
    : !dyninfer.model.parameter<tensor<?x?xf16>, "embedding">

  dyninfer.model.parameter @layer0_q
    : !dyninfer.model.parameter<tensor<?x?xf16>, "attention.q">

  func.func @decode(
      %token: tensor<?xi64>,
      %position: tensor<?xi64>,
      %key_cache: tensor<?x?x?x?xf16>,
      %value_cache: tensor<?x?x?x?xf16>
  ) -> tensor<?x?xf32> {
    %x = dyninfer.model.embedding %token, @tok_embeddings
    %q = dyninfer.model.linear %x, @layer0_q
    // ...
    return %logits : tensor<?x?xf32>
  }
}
```

### 8.3 Architecture authoring frontends

The engine supports three frontend classes.

#### 8.3.1 Rust architecture builder

This is the primary native frontend.

```rust
pub trait ArchitectureDefinition: Send + Sync {
    fn id(&self) -> ArchitectureId;
    fn config_schema(&self) -> &'static ConfigSchema;
    fn build(
        &self,
        config: &ResolvedModelConfig,
        builder: &mut ModelBuilder,
    ) -> Result<ModelModule>;
}
```

Example:

```rust
pub struct LlamaArchitecture;

impl ArchitectureDefinition for LlamaArchitecture {
    fn id(&self) -> ArchitectureId {
        ArchitectureId::new("llama.decoder")
    }

    fn config_schema(&self) -> &'static ConfigSchema {
        &LLAMA_CONFIG_SCHEMA
    }

    fn build(
        &self,
        config: &ResolvedModelConfig,
        m: &mut ModelBuilder,
    ) -> Result<ModelModule> {
        let tokens = m.input_tokens("tokens")?;
        let mut x = m.embedding(tokens, "token_embeddings.weight")?;

        for layer in 0..config.num_layers() {
            x = build_llama_block(m, x, layer, config)?;
        }

        let logits = m.linear(x, "output.weight")?;
        m.export_prefill_and_decode(logits)?;
        m.finish()
    }
}
```

The Rust builder MAY directly construct MLIR through a narrow C API, but the recommended MVP is to construct an engine-owned serializable graph in Rust and pass it to the C++ compiler bridge for MLIR creation.

#### 8.3.2 Direct MLIR frontend

Expert users MAY author architecture MLIR text or bytecode directly. The compiler MUST verify dialect versions and reject unsupported operations.

#### 8.3.3 External importer

A separate tool MAY import Transformers, Torch Export, ONNX, or another graph source and produce an architecture package. Python is permitted in import tooling but MUST NOT be required by the inference runtime.

### 8.4 Configuration resolution

Model configuration is resolved from ordered sources:

1. Explicit CLI or API overrides.
2. Architecture package defaults.
3. Checkpoint metadata.
4. Adjacent configuration files such as `config.json`.
5. Shape inference from checkpoint entries.

Conflicts MUST be errors unless an explicit override policy is provided.

---

## 9. Checkpoint plugin architecture

### 9.1 Plugin levels

Checkpoint support is divided into:

1. Container readers.
2. Convention decoders.
3. Runtime parameter providers.
4. Optional materializers.

These are separate because a single container can store many conventions.

### 9.2 Static plugin registry for version 1

Version 1 uses statically linked Rust crates and explicit registration:

```rust
pub struct PluginRegistryBuilder {
    container_readers: Vec<Arc<dyn CheckpointContainerReader>>,
    convention_decoders: Vec<Arc<dyn CheckpointConventionDecoder>>,
    materializers: Vec<Arc<dyn ParameterMaterializer>>,
}

impl PluginRegistryBuilder {
    pub fn with_builtin_plugins(mut self) -> Self {
        self.register_container(SafeTensorsContainer::default());
        self.register_container(GgufContainer::default());
        self.register_container(NpzContainer::default());
        self.register_convention(DenseConvention::default());
        self.register_convention(GgufConvention::default());
        self.register_convention(MlxGroupwiseConvention::default());
        self
    }
}
```

Dynamic Rust shared-library plugins are explicitly deferred because Rust does not provide a stable native ABI. A future plugin protocol SHOULD use a versioned C ABI, subprocess RPC, or WebAssembly component boundary.

### 9.3 Container reader trait

```rust
pub trait CheckpointContainerReader: Send + Sync {
    fn format_id(&self) -> ContainerFormatId;

    fn probe(&self, source: &dyn RandomAccessSource) -> Result<ProbeScore>;

    fn index(
        &self,
        source: Arc<dyn RandomAccessSource>,
        limits: &InspectionLimits,
    ) -> Result<RawCheckpointIndex>;

    fn runtime_provider_plan(
        &self,
        index: &RawCheckpointIndex,
    ) -> Result<RuntimeProviderPlan>;
}
```

### 9.4 Raw tensor entry

```rust
pub struct RawTensorEntry {
    pub key: String,
    pub shape: Vec<u64>,
    pub storage_type: StorageElementType,
    pub byte_ranges: Vec<ByteRange>,
    pub alignment: u64,
    pub endianness: Endianness,
    pub metadata: MetadataMap,
}
```

### 9.5 Convention decoder trait

```rust
pub trait CheckpointConventionDecoder: Send + Sync {
    fn convention_id(&self) -> ConventionId;

    fn match_score(
        &self,
        index: &RawCheckpointIndex,
        context: &DecodeContext,
    ) -> Result<MatchScore>;

    fn decode(
        &self,
        index: &RawCheckpointIndex,
        context: &DecodeContext,
    ) -> Result<ParameterCatalog>;
}
```

### 9.6 Logical parameter model

```rust
pub struct LogicalParameter {
    pub canonical_name: CanonicalParameterName,
    pub role: ParameterRole,
    pub logical_type: LogicalTensorType,
    pub encoding: PhysicalEncoding,
    pub components: Vec<StorageComponent>,
    pub aliases: Vec<String>,
}
```

### 9.7 Physical encoding model

```rust
pub enum PhysicalEncoding {
    Plain {
        storage_type: ScalarType,
        order: TensorOrder,
    },
    GroupQuantized {
        logical_type: ScalarType,
        storage_bits: u8,
        signed: bool,
        axis: i32,
        group_size: u32,
        scale_type: ScalarType,
        zero_point: ZeroPointMode,
        packing: PackingFormat,
    },
    BlockQuantized {
        logical_type: ScalarType,
        block_shape: Vec<u32>,
        codec: CodecId,
        codec_version: u32,
        components: Vec<EncodingComponent>,
    },
    Sparse {
        logical_type: ScalarType,
        format: SparseFormat,
        block_shape: Vec<u32>,
    },
    Opaque {
        codec: CodecId,
        codec_version: u32,
        descriptor: serde_json::Value,
    },
}
```

### 9.8 Initial checkpoint support

| Container | Convention | Version 1 policy |
|---|---|---|
| SafeTensors | Dense FP16/BF16/FP32 | Direct IREE parameter access or aligned copy |
| GGUF | Dense FP16/BF16 | Direct parameter access |
| GGUF | Q4_0 | Fused generated matmul correctness path |
| GGUF | Q4_K | Phase 2 |
| NPZ | MLX dense arrays | Custom reader; staged copy or IRPA cache |
| SafeTensors | MLX groupwise quantization | Phase 2 convention decoder |
| Legacy GGML | Legacy model files | Deferred; recommend conversion to GGUF |

---

## 10. Parameter binding

### 10.1 Parameter slots

Architecture packages declare logical slots:

```rust
pub struct ParameterSlot {
    pub id: ParameterSlotId,
    pub canonical_name: CanonicalParameterName,
    pub role: ParameterRole,
    pub expected_type: LogicalTensorConstraint,
    pub supported_encodings: Vec<EncodingConstraint>,
    pub optional: bool,
    pub tied_group: Option<TiedParameterGroup>,
}
```

### 10.2 Binding transforms

```rust
pub enum BindingTransform {
    Identity,
    Rename,
    Reshape { shape: Vec<u64> },
    LogicalTranspose { permutation: Vec<u32> },
    Slice { ranges: Vec<Range<u64>> },
    Concatenate { axis: u32 },
    Split { axis: u32, segments: Vec<u64> },
    Permute { permutation: Vec<u32> },
    Alias,
    Repack { target_encoding: PhysicalEncoding },
}
```

### 10.3 Materialization policies

```rust
pub enum MaterializationPolicy {
    DirectView,
    CopyAligned,
    DecodeOnTheFly,
    PrepackToCache,
}
```

- `DirectView` consumes original bytes directly.
- `CopyAligned` copies unchanged bytes to suitable memory.
- `DecodeOnTheFly` decodes blocks inside the consuming kernel.
- `PrepackToCache` creates a derived parameter artifact.

### 10.4 Binding rules

Binding MUST validate:

- Parameter presence.
- Logical shape.
- Physical storage size.
- Encoding and codec version.
- Auxiliary component relationships.
- Orientation and dimension conventions.
- Tied-weight requirements.
- Shard completeness.
- Alignment assumptions.

The binder MUST NOT silently cast or reinterpret an unknown encoding.

---

## 11. Bound Model IR

The Bound Model IR is the canonical compilation input after binding.

Conceptual example:

```mlir
dyninfer.checkpoint.binding @layer0_q {
  scope = "weights",
  key = "blk.0.attn_q.weight",
  logical_shape = [4096, 4096],
  encoding = #dyninfer.checkpoint.gguf_q4_0<block_size = 32>,
  storage_bytes = 2359296 : i64,
  alignment = 32 : i64
}

%q = dyninfer.model.linear %x, @layer0_q
  : tensor<?x4096xf16> -> tensor<?x4096xf16>
```

The bound IR SHOULD reference logical parameter symbols and external scope/key identifiers. It MUST NOT embed host-specific absolute paths.

---

## 12. Compilation pipeline

### 12.1 Compiler inputs

The compiler receives:

- Architecture MLIR bytecode.
- Resolved model configuration.
- Binding plan.
- Target profile.
- Shape profile.
- Compilation options.
- Optional tuning specification.

### 12.2 Pass pipeline

```text
1. Parse and verify architecture IR
2. Resolve configuration symbols
3. Infer architecture shapes
4. Apply parameter bindings
5. Validate bound model
6. Canonicalize semantic model operations
7. Specialize parameter encodings
8. Select materialization strategies
9. Split and specialize inference entrypoints
10. Select kernel candidates
11. Lower model operations to qkernel, linalg, and IREE extensions
12. Lower checkpoint accesses to IREE parameters
13. Lower qkernel operations to standard/vector/target-specific MLIR
14. Run IREE compilation pipeline
15. Emit VMFB and compilation metadata
```

### 12.3 Encoding specialization

Dense weight:

```text
dyninfer.model.linear
  -> external parameter load
  -> linalg.matmul or equivalent
```

Q4_0 weight:

```text
dyninfer.model.linear
  -> dyninfer.qkernel.quantized_matmul
  -> block loads + scale loads + unpack + dot/reduction
  -> IREE-generated device dispatch
```

### 12.4 Entry point specialization

The compiler MUST produce separate entrypoints for at least:

- `model.prefill`
- `model.decode`

It MAY additionally produce:

- `model.prefill_chunk`
- `model.decode_batch`
- `model.verify_speculative`
- `model.project_logits`
- `model.sample`

Prefill and decode MAY have different graph rewrites, kernel candidates, and shape buckets.

---

## 13. Kernel architecture

### 13.1 Kernel ownership

The engine defines:

- Operation semantics.
- Physical encoding semantics.
- Valid lowering strategies.
- Kernel candidate constraints.
- Optional target-specific transformations.

IREE defines:

- Dispatch formation.
- Fusion and canonical optimization.
- Workgroup/thread distribution.
- Vector and backend lowering.
- SPIR-V, LLVM CPU, CUDA, or ROCm executable generation.
- VMFB packaging and HAL invocation.

### 13.2 Kernel levels

Every kernel candidate belongs to one of three levels:

1. Portable generated correctness implementation.
2. Specialized compiler-generated implementation.
3. Handwritten external implementation.

### 13.3 Kernel registry

```rust
pub struct KernelCandidateDescriptor {
    pub id: KernelCandidateId,
    pub operation: OperationKind,
    pub input_constraints: Vec<TypeConstraint>,
    pub weight_encoding: EncodingConstraint,
    pub shape_constraint: ShapeConstraint,
    pub target_constraint: TargetConstraint,
    pub priority: i32,
    pub lowering: KernelLoweringId,
}

pub trait KernelCostModel: Send + Sync {
    fn estimate(
        &self,
        candidate: &KernelCandidateDescriptor,
        request: &KernelRequest,
        target: &TargetProfile,
    ) -> Result<EstimatedCost>;
}
```

The compiler extension owns actual MLIR lowering registrations. Rust owns policy, target description, cache keys, and candidate configuration.

### 13.4 Initial kernel plan

| Operation | Initial implementation |
|---|---|
| Dense linear | Standard MLIR/IREE matmul lowering |
| RMSNorm | Generated reduction + elementwise MLIR |
| RoPE | Generated elementwise/indexing MLIR |
| SwiGLU | Dense matmuls plus generated elementwise fusion |
| Prefill attention | IREE attention/LinalgExt path where suitable |
| Decode attention | Custom high-level op with generic generated lowering |
| Q4_0 linear | Engine-owned qkernel lowering, IREE-generated device code |
| KV-cache write | Generated indexed store |

---

## 14. IREE integration

### 14.1 Runtime ABI

Rust invokes IREE through its C runtime API. The runtime wrapper crate owns opaque handles for:

- VM instance.
- HAL driver and device.
- VM module.
- VM context/session.
- Function handle.
- HAL buffer and buffer view.
- Fence/semaphore.
- Parameter provider.

The high-level Rust API MUST not expose raw IREE handles.

### 14.2 Compiler ABI

The Rust compiler orchestrator invokes a project-owned C ABI implemented by a C++ compiler library.

```c
#ifndef DYNINFER_COMPILER_H_
#define DYNINFER_COMPILER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dyninfer_compiler_t dyninfer_compiler_t;

typedef struct dyninfer_bytes_t {
  const uint8_t* data;
  size_t size;
} dyninfer_bytes_t;

typedef struct dyninfer_owned_bytes_t {
  uint8_t* data;
  size_t size;
  void (*release)(uint8_t* data, size_t size, void* user_data);
  void* user_data;
} dyninfer_owned_bytes_t;

typedef struct dyninfer_compile_request_t {
  dyninfer_bytes_t architecture_mlirbc;
  dyninfer_bytes_t resolved_config_json;
  dyninfer_bytes_t binding_plan_json;
  dyninfer_bytes_t target_profile_json;
  dyninfer_bytes_t compile_options_json;
} dyninfer_compile_request_t;

typedef struct dyninfer_compile_result_t {
  dyninfer_owned_bytes_t vmfb;
  dyninfer_owned_bytes_t metadata_json;
  dyninfer_owned_bytes_t diagnostics_utf8;
} dyninfer_compile_result_t;

int32_t dyninfer_compiler_create(
    dyninfer_bytes_t options_json,
    dyninfer_compiler_t** out_compiler);

int32_t dyninfer_compiler_compile(
    dyninfer_compiler_t* compiler,
    const dyninfer_compile_request_t* request,
    dyninfer_compile_result_t* out_result);

void dyninfer_compiler_destroy(dyninfer_compiler_t* compiler);
void dyninfer_compile_result_destroy(dyninfer_compile_result_t* result);

#ifdef __cplusplus
}
#endif

#endif
```

The ABI MUST:

- Use fixed-width integer types.
- Avoid C++ standard library types.
- Define ownership for every returned allocation.
- Never unwind exceptions across the C boundary.
- Return structured diagnostics in addition to status codes.
- Carry an ABI version in compiler creation options.

### 14.3 Compiler extension strategy

The compiler library links:

- IREE compiler API.
- Engine custom dialect definitions.
- Engine verification and conversion passes.
- Kernel lowering registrations.

The extension SHOULD lower custom operations to standard MLIR and IREE-supported input dialects before the main IREE pipeline whenever possible.

### 14.4 Parameter scopes

The generated VMFB references external parameters by scope and key:

```text
weights::blk.0.attn_q.weight
adapters::layer0.lora_a
runtime_constants::rope_table
```

Version 1 MAY bind actual source checkpoint keys directly. A later aliasing provider MAY let VMFBs use canonical parameter keys independent of checkpoint naming.

### 14.5 IREE build policy

IREE is pinned to an exact source revision.

Because upstream IREE describes its Bazel build as primarily internal and Linux-focused, version 1 officially supports building IREE from source through Bazel on Linux only. Other host platforms MAY import prebuilt IREE libraries produced by the upstream CMake build while preserving Bazel as the top-level build system.

---

## 15. Rust workspace and crates

### 15.1 Workspace layout

```text
crates/
|-- dyninfer-core/
|-- dyninfer-error/
|-- dyninfer-checkpoint/
|-- dyninfer-checkpoint-safetensors/
|-- dyninfer-checkpoint-gguf/
|-- dyninfer-checkpoint-npz/
|-- dyninfer-architecture/
|-- dyninfer-architecture-llama/
|-- dyninfer-binding/
|-- dyninfer-kernel-registry/
|-- dyninfer-target/
|-- dyninfer-cache/
|-- dyninfer-compiler-sys/
|-- dyninfer-compiler/
|-- iree-runtime-sys/
|-- iree-runtime/
|-- dyninfer-runtime/
|-- dyninfer-tokenizer/
`-- dyninfer-cli/
```

### 15.2 Crate responsibilities

#### `dyninfer-core`

Stable data types shared across compiler orchestration and runtime manifests. No IREE FFI.

#### `dyninfer-checkpoint`

Container and convention traits, catalog model, random-access source abstraction, inspection limits.

#### Format crates

Each container and convention is isolated so dependency-heavy support can be optional.

#### `dyninfer-architecture`

Architecture registry, graph builder, architecture package reader/writer, config schema.

#### `dyninfer-binding`

Name matching, role matching, shape validation, transforms, materialization planning.

#### `dyninfer-kernel-registry`

Kernel candidate descriptors and cost-model policy. It does not generate device code directly.

#### `dyninfer-target`

Hardware discovery, IREE target mapping, target capability fingerprinting.

#### `dyninfer-cache`

Content-addressed executable and parameter caches with atomic publication and eviction.

#### `dyninfer-compiler-sys`

Unsafe raw bindings to `dyninfer_compiler.h` only.

#### `dyninfer-compiler`

Safe Rust compiler wrapper and optional compiler worker client.

#### `iree-runtime-sys`

Unsafe raw bindings to the subset of IREE C runtime headers used by the project.

#### `iree-runtime`

Safe RAII wrappers over IREE instance, device, module, context, function, buffer, buffer view, and parameter provider.

#### `dyninfer-runtime`

Model loading, bundle selection, schema validation, session and KV-cache management, prefill/decode invocation.

#### `dyninfer-cli`

User-facing commands and diagnostics.

### 15.3 Unsafe code policy

Only the following crates MAY contain unrestricted unsafe blocks:

- `iree-runtime-sys`
- `iree-runtime`
- `dyninfer-compiler-sys`

Other crates MUST use `#![forbid(unsafe_code)]` unless an exception is documented in an architecture decision record.

---

## 16. Public Rust APIs

### 16.1 Compilation API

```rust
pub struct CompileRequest<'a> {
    pub architecture: &'a ArchitecturePackage,
    pub checkpoint: &'a CheckpointCatalog,
    pub binding: &'a BindingPlan,
    pub target: &'a TargetProfile,
    pub shape_profile: &'a ShapeProfile,
    pub options: &'a CompileOptions,
}

pub struct CompileOutput {
    pub executable: VmfbArtifact,
    pub manifest: ExecutableManifest,
    pub diagnostics: Vec<Diagnostic>,
}

pub trait ModelCompiler: Send + Sync {
    fn compile(&self, request: &CompileRequest<'_>) -> Result<CompileOutput>;
}
```

### 16.2 Runtime API

```rust
pub trait CausalLanguageModel: Send + Sync {
    fn metadata(&self) -> &ModelMetadata;

    fn create_session(&self, config: SessionConfig) -> Result<Box<dyn ModelSession>>;
}

pub trait ModelSession: Send {
    fn prefill(&mut self, tokens: &[TokenId]) -> Result<Logits>;
    fn decode(&mut self, token: TokenId) -> Result<Logits>;
    fn position(&self) -> u64;
    fn reset(&mut self) -> Result<()>;
}
```

### 16.3 Asynchronous API

A later runtime revision SHOULD expose asynchronous submission:

```rust
pub trait AsyncModelSession: Send {
    fn prefill<'a>(
        &'a mut self,
        tokens: &'a [TokenId],
    ) -> Pin<Box<dyn Future<Output = Result<Logits>> + Send + 'a>>;
}
```

Version 1 MAY internally use asynchronous IREE fences while presenting a synchronous API.

---

## 17. Runtime design

### 17.1 Load flow

```text
1. Open executable bundle.
2. Discover local IREE drivers and devices.
3. Select the best compatible executable.
4. Open and inspect checkpoint metadata.
5. Validate schema compatibility.
6. Register original and derived parameter providers.
7. Create IREE instance, device, modules, and context.
8. Resolve exported prefill and decode functions.
9. Allocate session KV cache and reusable input/output buffers.
10. Invoke prefill and decode.
```

### 17.2 Inference ABI

Version 1 uses explicit cache buffers rather than hidden model globals.

Conceptual functions:

```text
model.prefill(
    token_ids,
    token_count,
    start_position,
    key_cache,
    value_cache
) -> logits

model.decode(
    token_id,
    position,
    key_cache,
    value_cache
) -> logits
```

Actual VM function signatures MAY use buffer views and fixed shape buckets. The bundle manifest maps user-level arguments to exported functions.

### 17.3 KV-cache descriptor

```rust
pub struct KvCacheDescriptor {
    pub layer_count: u32,
    pub max_batch_size: u32,
    pub max_sequence_length: u32,
    pub kv_head_count: u32,
    pub head_dimension: u32,
    pub element_type: ScalarType,
    pub layout: KvCacheLayout,
    pub alignment: u64,
}
```

### 17.4 Device selection

The runtime selects a VMFB using:

- Required IREE driver.
- Target architecture and capability predicates.
- Required extensions or subgroup behavior.
- Memory requirements.
- User preference.
- Expected performance rank.

The runtime MUST reject a VMFB whose capability requirements are not met.

---

## 18. AOT, first-load JIT, and hybrid modes

### 18.1 Fully AOT

Compilation occurs before deployment:

```text
architecture + checkpoint schema + target profile -> bundle
```

The target machine receives:

- Runtime binary.
- VMFB bundle.
- Original checkpoint.
- Optional derived parameters.

### 18.2 First-load JIT

At first model load:

```text
architecture package + checkpoint catalog + local target -> VMFB cache
```

JIT compilation occurs at model-load granularity, not per operation in the decode loop.

### 18.3 Recommended hybrid mode

1. Architecture author exports a verified architecture package.
2. Model installation binds the checkpoint and creates a target-independent bound model artifact.
3. Local installation or first use compiles target-specific VMFBs.

This mode avoids executing arbitrary model Python locally while retaining hardware specialization.

---

## 19. Caching and fingerprints

### 19.1 Executable cache key

```text
hash(
  architecture package digest,
  resolved model config,
  binding plan,
  checkpoint schema fingerprint,
  target profile fingerprint,
  shape profile,
  kernel registry version,
  compiler extension version,
  IREE revision,
  compilation options
)
```

Raw learned parameter bytes SHOULD NOT be included unless a value-dependent optimization is enabled.

### 19.2 Parameter cache key

```text
hash(
  checkpoint content identity,
  materialization plan,
  target profile,
  materializer and codec versions
)
```

### 19.3 Cache layout

```text
~/.cache/dyninfer/
|-- executables/<digest>/model.vmfb
|-- executables/<digest>/manifest.json
|-- parameters/<digest>/derived.irpa
|-- catalogs/<digest>/catalog.json
`-- locks/<digest>.lock
```

Cache publication MUST be atomic. Concurrent processes compiling the same key SHOULD coordinate through advisory locks and verify the final digest.

---

## 20. Bazel and `rules_rs` build specification

### 20.1 Build assumptions

- Bzlmod is mandatory.
- The repository contains `Cargo.toml` and `Cargo.lock` for third-party Rust dependency resolution.
- First-party build targets are declared in `BUILD.bazel` files.
- `rules_rs` imports external crates through `crate.from_cargo`.
- The Rust and LLVM toolchains are hermetic.
- IREE is pinned separately and not resolved as an unconstrained floating dependency.

### 20.2 Baseline `MODULE.bazel`

The versions below are an illustrative pinned baseline assembled from current documentation at the specification date. They have not been validated together by this document and MUST be verified in project CI.

```starlark
module(
    name = "dyninfer",
    version = "0.1.0",
)

bazel_dep(name = "rules_rs", version = "0.0.86")
bazel_dep(name = "llvm", version = "0.8.14")
bazel_dep(name = "platforms", version = "1.1.0")
bazel_dep(name = "rules_cc", version = "0.2.22")
bazel_dep(name = "rules_rust_bindgen", version = "0.71.3")

toolchains = use_extension(
    "@rules_rs//rs/toolchains:module_extension.bzl",
    "toolchains",
)
toolchains.toolchain(
    edition = "2024",
    version = "1.92.0",
)
use_repo(toolchains, "default_rust_toolchains")

rules_rust = use_extension(
    "@rules_rs//rs:rules_rust.bzl",
    "rules_rust",
)
use_repo(rules_rust, "rules_rust")

register_toolchains(
    "@default_rust_toolchains//:all",
    "@llvm//toolchain:all",
    "@rules_rust_bindgen//:default_bindgen_toolchain",
)

crate = use_extension("@rules_rs//rs:extensions.bzl", "crate")
crate.from_cargo(
    name = "crates",
    cargo_lock = "//:Cargo.lock",
    cargo_toml = "//:Cargo.toml",
    platform_triples = [
        "aarch64-apple-darwin",
        "aarch64-unknown-linux-gnu",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
    ],
)
use_repo(crate, "crates")

# IREE integration option A, Linux source build:
# Vendor a pinned IREE checkout under third_party/iree and expose it through
# a repository rule or local_path_override compatible with the pinned commit.
#
# IREE integration option B, non-Linux host:
# Import pinned CMake-built libraries through project-owned cc_import targets.
```

The exact `rules_cc`, Rust, LLVM, and IREE versions MUST be validated together in CI. The repository SHOULD use Renovate or a similar tool to propose upgrades, but upgrades MUST include compiler and inference regression tests.

### 20.3 Root `Cargo.toml`

```toml
[workspace]
resolver = "3"
members = [
  "crates/dyninfer-core",
  "crates/dyninfer-error",
  "crates/dyninfer-checkpoint",
  "crates/dyninfer-checkpoint-safetensors",
  "crates/dyninfer-checkpoint-gguf",
  "crates/dyninfer-checkpoint-npz",
  "crates/dyninfer-architecture",
  "crates/dyninfer-architecture-llama",
  "crates/dyninfer-binding",
  "crates/dyninfer-kernel-registry",
  "crates/dyninfer-target",
  "crates/dyninfer-cache",
  "crates/dyninfer-compiler-sys",
  "crates/dyninfer-compiler",
  "crates/iree-runtime-sys",
  "crates/iree-runtime",
  "crates/dyninfer-runtime",
  "crates/dyninfer-cli",
]

[workspace.package]
edition = "2024"
rust-version = "1.92"
license = "Apache-2.0"

[workspace.dependencies]
anyhow = "1"
bytes = "1"
clap = { version = "4", features = ["derive"] }
memmap2 = "0.9"
parking_lot = "0.12"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = "0.3"
uuid = { version = "1", features = ["v4", "serde"] }
```

Cargo is the dependency declaration and lockfile source. Bazel remains the authoritative build and test entrypoint.

### 20.4 Rust target example

```starlark
load("@crates//:defs.bzl", "aliases", "all_crate_deps")
load("@rules_rs//rs:rust_library.bzl", "rust_library")
load("@rules_rs//rs:rust_test.bzl", "rust_test")

rust_library(
    name = "dyninfer_checkpoint",
    srcs = glob(["src/**/*.rs"]),
    crate_name = "dyninfer_checkpoint",
    edition = "2024",
    aliases = aliases(
        normal = True,
        normal_dev = True,
    ),
    deps = all_crate_deps(normal = True) + [
        "//crates/dyninfer-core",
        "//crates/dyninfer-error",
    ],
    visibility = ["//visibility:public"],
)

rust_test(
    name = "dyninfer_checkpoint_tests",
    crate = ":dyninfer_checkpoint",
    deps = all_crate_deps(normal_dev = True),
)
```

Projects SHOULD avoid attaching every Cargo dependency to every first-party crate in the final implementation. The broad `all_crate_deps` example is suitable for bootstrap; production BUILD files SHOULD use generated per-package aliases or explicit dependency labels.

### 20.5 CLI target example

```starlark
load("@crates//:defs.bzl", "aliases", "all_crate_deps")
load("@rules_rs//rs:rust_binary.bzl", "rust_binary")

rust_binary(
    name = "dyninfer",
    srcs = glob(["src/**/*.rs"]),
    crate_name = "dyninfer",
    edition = "2024",
    aliases = aliases(),
    deps = all_crate_deps(normal = True) + [
        "//crates/dyninfer-compiler",
        "//crates/dyninfer-runtime",
        "//crates/dyninfer-checkpoint-safetensors",
        "//crates/dyninfer-checkpoint-gguf",
        "//crates/dyninfer-architecture-llama",
    ],
    visibility = ["//visibility:public"],
)
```

### 20.6 C++ compiler bridge target

```starlark
cc_library(
    name = "dyninfer_compiler_capi",
    srcs = [
        "compiler_capi.cc",
        "compiler_session.cc",
    ],
    hdrs = ["include/dyninfer/compiler.h"],
    includes = ["include"],
    deps = [
        "//compiler/dialects:model",
        "//compiler/dialects:checkpoint",
        "//compiler/dialects:qkernel",
        "//compiler/passes:all_passes",
        "@iree//compiler/src/iree/compiler/API:CAPI",
    ],
    visibility = ["//visibility:public"],
)
```

The actual IREE target labels MUST follow the pinned IREE revision. Project-owned wrapper targets SHOULD isolate the rest of the repository from upstream target label churn.

### 20.7 Rust bindgen target

```starlark
load("@rules_rust_bindgen//:defs.bzl", "rust_bindgen_library")

rust_bindgen_library(
    name = "dyninfer_compiler_bindings",
    header = "//compiler/capi:include/dyninfer/compiler.h",
    cc_lib = "//compiler/capi:dyninfer_compiler_capi",
    bindgen_flags = [
        "--allowlist-function=dyninfer_.*",
        "--allowlist-type=dyninfer_.*",
        "--allowlist-var=DYNINFER_.*",
        "--no-layout-tests",
    ],
    visibility = ["//visibility:public"],
)
```

IREE runtime bindings SHOULD be generated from a project-owned umbrella header that includes only the required IREE C APIs. This limits build time and unsafe API surface.

### 20.8 Build configurations

Suggested `.bazelrc`:

```text
common --enable_platform_specific_config
common --announce_rc
common --incompatible_strict_action_env
common --experimental_convenience_symlinks=ignore

build --keep_going
build --show_timestamps
build --verbose_failures

build:linux --host_platform=//platforms:local_linux_gnu
build:release --compilation_mode=opt
build:release --strip=always
build:asan --config=dbg
build:asan --features=asan

test --test_output=errors
test --flaky_test_attempts=2
```

A real ASan/TSan configuration must be aligned with the selected C++ and Rust toolchains. Sanitized FFI tests SHOULD run in dedicated CI configurations.

### 20.9 Recommended build commands

```bash
bazel build //crates/dyninfer-cli:dyninfer
bazel test //...
bazel run //crates/dyninfer-cli:dyninfer -- inspect model.gguf
bazel run @rules_rs//tools/rust_analyzer:gen_rust_project
```

---

## 21. Repository layout

```text
.
|-- MODULE.bazel
|-- MODULE.bazel.lock
|-- Cargo.toml
|-- Cargo.lock
|-- BUILD.bazel
|-- .bazelrc
|-- .bazelversion
|-- platforms/
|   `-- BUILD.bazel
|-- crates/
|   `-- ...
|-- compiler/
|   |-- capi/
|   |-- dialects/
|   |   |-- Model/
|   |   |-- Checkpoint/
|   |   `-- QKernel/
|   |-- passes/
|   |-- conversion/
|   |-- kernels/
|   `-- BUILD.bazel
|-- architectures/
|   |-- llama/
|   `-- testdata/
|-- schemas/
|   |-- architecture-manifest.schema.json
|   |-- binding-plan.schema.json
|   `-- executable-manifest.schema.json
|-- tools/
|   |-- architecture-import/
|   `-- golden-update/
|-- tests/
|   |-- integration/
|   |-- differential/
|   |-- conformance/
|   `-- performance/
|-- third_party/
|   `-- iree/
`-- docs/
    |-- architecture/
    |-- adr/
    `-- formats/
```

---

## 22. CLI specification

### 22.1 Inspect checkpoint

```bash
dyninfer checkpoint inspect model.gguf --json
```

Outputs:

- Container identity.
- Convention match.
- Model metadata.
- Parameter count and total bytes.
- Encodings and logical roles.
- Schema fingerprint.
- Warnings and unsupported entries.

### 22.2 Validate architecture and checkpoint

```bash
dyninfer bind \
  --architecture llama.arch \
  --checkpoint model.gguf \
  --output binding.json
```

### 22.3 Compile

```bash
dyninfer compile \
  --architecture llama.arch \
  --checkpoint model.gguf \
  --target vulkan://local \
  --mode local-jit \
  --output model.bundle
```

### 22.4 Run

```bash
dyninfer run \
  --bundle model.bundle \
  --checkpoint model.gguf \
  --prompt "Hello"
```

### 22.5 Install model

```bash
dyninfer model install \
  --architecture llama.arch \
  --checkpoint model.gguf \
  --target auto
```

This performs inspection, binding, optional parameter materialization, local compilation, and atomic cache publication.

### 22.6 Cache management

```bash
dyninfer cache list
dyninfer cache verify
dyninfer cache prune --max-size 100GB
dyninfer cache remove <digest>
```

---

## 23. Error and diagnostics model

### 23.1 Structured errors

```rust
pub enum DynInferError {
    Io(IoError),
    UnsupportedContainer(UnsupportedContainerError),
    InvalidCheckpoint(CheckpointValidationError),
    UnsupportedEncoding(UnsupportedEncodingError),
    ArchitectureMismatch(ArchitectureMismatchError),
    Binding(BindingError),
    Compilation(CompilationError),
    IreeRuntime(IreeRuntimeError),
    Device(DeviceError),
    Cache(CacheError),
}
```

### 23.2 Diagnostics

Compiler diagnostics SHOULD include:

- Stable error code.
- Severity.
- Architecture operation or parameter slot.
- Checkpoint source key.
- Expected and actual descriptor.
- Compiler pass name.
- Suggested corrective action.

Example:

```text
E_BIND_ENCODING_MISMATCH
parameter slot: layers.0.attention.q.weight
checkpoint key: blk.0.attn_q.weight
expected: one of [dense_bf16, dense_f16, gguf_q4_0]
actual: gguf_iq2_xxs codec version 1
suggestion: install the iq2 codec plugin or convert the checkpoint
```

---

## 24. Security and trust model

- Checkpoints are untrusted data.
- Architecture packages are untrusted declarative compiler inputs.
- Metadata sizes, tensor counts, dimensions, and byte ranges MUST be bounded.
- Integer arithmetic for offsets and tensor byte sizes MUST be overflow checked.
- Memory mapping MUST validate ranges before exposure to IREE providers.
- Compiler execution SHOULD be isolated for downloaded architecture packages.
- Runtime MUST NOT load arbitrary native libraries referenced by a model package.
- Dynamic plugin installation MUST require explicit user or administrator action.
- Cache manifests MUST be validated before loading artifacts.
- VMFB files SHOULD be treated as executable code and loaded only from trusted or locally compiled sources.

---

## 25. Testing strategy

### 25.1 Unit tests

- Container parsers with malformed input corpora.
- Convention decoder matching.
- Shape and byte-size arithmetic.
- Binding transforms.
- Schema fingerprint stability.
- Cache atomicity.
- FFI ownership and status conversion.

### 25.2 MLIR pass tests

Every custom pass MUST have textual MLIR tests using FileCheck-style expectations:

- Valid conversion.
- Invalid operation verifier behavior.
- Dense parameter lowering.
- Q4_0 lowering.
- Prefill/decode specialization.
- Target constraint selection.

### 25.3 Differential correctness tests

For small deterministic models:

```text
Reference implementation
  versus
CPU IREE executable
  versus
Vulkan IREE executable
```

Compare:

- Layer outputs for debug builds.
- Final logits.
- KV-cache contents.
- Multi-step decode sequences.

Quantized comparisons MUST use encoding-appropriate tolerances and deterministic reference decoding.

### 25.4 Checkpoint conformance tests

Maintain tiny synthetic fixtures for:

- SafeTensors dense.
- GGUF dense.
- GGUF Q4_0.
- NPZ dense.
- Sharded checkpoints.
- Tied weights.
- Truncated and malicious inputs.

### 25.5 Performance tests

Track separately:

- Checkpoint inspection time.
- Binding time.
- Compilation time.
- Parameter materialization time.
- Model load time.
- Prefill tokens per second.
- Decode tokens per second.
- Peak host memory.
- Peak device memory.
- Cache hit rate.

Performance regressions MUST NOT override correctness failures.

---

## 26. Observability

The Rust runtime uses `tracing` spans:

```text
checkpoint.probe
checkpoint.index
checkpoint.decode_convention
architecture.load
binding.resolve
compile.specialize
compile.iree
cache.lookup
parameters.open
runtime.create_device
runtime.load_vmfb
runtime.prefill
runtime.decode
```

The compiler SHOULD optionally emit:

- Intermediate MLIR after named passes.
- Kernel selection report.
- Dispatch and executable statistics.
- IREE compilation timing.
- Generated backend intermediate files.

Sensitive checkpoint contents MUST NOT be logged.

---

## 27. Milestones

### Milestone 0: Build and FFI skeleton

- Bazel/Bzlmod workspace using `rules_rs`.
- Pinned IREE source build on Linux.
- Rust IREE runtime wrapper.
- Rust-to-C++ compiler bridge.
- Compile and invoke a trivial dense MLIR module.

**Exit criterion:** Rust CLI compiles MLIR to VMFB and invokes it on CPU and Vulkan.

### Milestone 1: Dense Llama proof of concept

- Rust architecture builder for a small Llama decoder.
- SafeTensors dense reader.
- Parameter binding.
- External IREE parameter loading.
- FP16/BF16 prefill and single-token decode.
- Static KV cache.

**Exit criterion:** Correct logits against a reference implementation for a tiny model.

### Milestone 2: GGUF and Q4_0

- GGUF container and convention decoder.
- Q4_0 physical encoding.
- Portable generated Q4_0 linear kernel.
- Direct original-checkpoint parameter provider.
- CPU and Vulkan differential tests.

**Exit criterion:** Run an unmodified Q4_0 GGUF checkpoint through the engine.

### Milestone 3: Local specialization and caching

- Target capability fingerprint.
- First-load JIT.
- Executable cache.
- Derived parameter cache.
- Shape buckets.
- Benchmark and kernel selection reports.

### Milestone 4: Optimized decode

- Specialized decode attention.
- GQA-aware cache access.
- Vulkan subgroup-aware Q4_0 lowering.
- Buffer reuse and asynchronous execution.

### Milestone 5: Additional formats and targets

- MLX NPZ and MLX groupwise quantization.
- GGUF Q4_K.
- ROCm and CUDA deployment profiles.
- Optional architecture importer.

---

## 28. Acceptance criteria for version 1

Version 1 is complete when:

1. The repository builds with one Bazel command on the supported Linux hosts.
2. Rust is the implementation language for all checkpoint, orchestration, cache, and runtime components.
3. C++ is limited to MLIR/IREE compiler extension and C ABI code.
4. An unmodified SafeTensors dense checkpoint runs locally.
5. An unmodified GGUF Q4_0 checkpoint runs locally.
6. Both CPU and Vulkan backends are supported.
7. Separate prefill and decode VM functions are generated.
8. Executable caching reuses code across checkpoints with compatible schemas.
9. Derived parameter caches never modify the original checkpoint.
10. Differential tests validate logits and multi-token decode behavior.
11. Unsupported encodings fail with explicit structured diagnostics.
12. The public Rust API contains no raw IREE pointer types.

---

## 29. Open design questions

The following decisions should be captured in architecture decision records during implementation:

1. Whether the Rust architecture builder creates MLIR through the MLIR C API or emits an intermediate serializable graph first.
2. Whether checkpoint canonical names are embedded into VMFBs or resolved through an aliasing parameter provider.
3. Whether the first quantized Vulkan lowering uses pure standard/vector MLIR or a target-specific IREE compiler extension.
4. Whether KV cache is represented as explicit tensors, HAL buffers, or a custom runtime resource in later versions.
5. Whether the compiler is linked in-process by default or always invoked as a worker.
6. How architecture MLIR dialect version migration is handled.
7. How tuning data is represented and incorporated into executable cache keys.
8. Whether sampling is compiled into the model executable or remains a Rust runtime operation.
9. Whether tokenizer artifacts belong inside the executable bundle or in a sibling model package.
10. How adapter overlays such as LoRA are represented without invalidating the base executable cache.

---

## 30. Recommended initial decisions

To minimize risk, the implementation should begin with these choices:

- Linux-only source builds for IREE.
- Rust static plugin registry.
- Rust-generated engine graph serialized as canonical JSON, converted to MLIR by C++.
- Actual checkpoint keys embedded in the first VMFB binding implementation.
- SafeTensors dense followed by GGUF Q4_0.
- Explicit static KV-cache tensors.
- CPU reference backend before Vulkan optimization.
- Generic generated quantized kernels before handwritten kernels.
- In-process compiler for development, worker process option before accepting untrusted packages.
- Sampling in Rust for version 1.

These choices preserve the long-term architecture while keeping the first end-to-end slice implementable.

---

## 31. References

1. [IREE C API bindings](https://iree.dev/reference/bindings/c-api/)
2. [IREE parameter system](https://iree.dev/guides/parameters/)
3. [IREE extension guidance](https://iree.dev/reference/extensions/)
4. [IREE developer overview](https://iree.dev/developers/general/developer-overview/)
5. [IREE Bazel build documentation](https://github.com/iree-org/iree/blob/main/docs/website/docs/developers/building/bazel.md)
6. [`rules_rs` repository and setup](https://github.com/hermeticbuild/rules_rs)
7. [`rules_rs` in the Bazel Central Registry](https://registry.bazel.build/modules/rules_rs)
8. [`rules_rust` bindgen extension](https://bazelbuild.github.io/rules_rust/rust_bindgen.html)
