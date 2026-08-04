# Schema- and Local-Target-Specialized Inference

**Status:** implemented baseline; remaining encoding/performance work is listed below
**Audience:** repository maintainers and coding agents
**Scope:** replace the current architecture-specific dense emitter and host-materialized weight paths with a strict, schema-specialized compiler that directly addresses original checkpoint storage.

## 1. Outcome

`dyninfer` accepts a compiled-in Rust model architecture, a supported checkpoint, and the automatically detected local device. It produces and caches an IREE VMFB specialized for:

- the architecture's semantic graph;
- the checkpoint schema, including every parameter's physical encoding, layout, shape, and component relationships;
- static execution shapes such as prefill and KV-cache buckets;
- the exact detected local target and its relevant capabilities; and
- an explicit precision policy.

The compiler selects a legal kernel independently for every bound operation. A checkpoint may mix dense F32/F16/BF16 tensors and multiple quantized encodings in the same layer or model.

Compilation **must fail** if any operation/encoding/shape/target combination has no approved kernel. It must never make an unsupported checkpoint appear functional by silently dequantizing, promoting, repacking, changing device, or selecting a deliberately slow reference implementation.

## 2. Accepted product decisions

These are requirements, not open design questions.

1. Specialization is based on checkpoint schema, not weight values.
2. Two checkpoints with identical code-relevant schemas may reuse one VMFB.
3. Weight values remain external to the VMFB.
4. The original checkpoint is the only persistent weight storage.
5. No derived weight cache, dense copy, quantized repack, or rewritten checkpoint is allowed.
6. Runtime parameter providers address original files by file, offset, and length.
7. Device upload and ordinary runtime buffers are allowed; persistent duplicate weight artifacts and whole-model host expansion are not.
8. Supported architectures are compiled into the executable and implemented in Rust.
9. A new architecture normally requires one implementation file and one explicit registration line.
10. Quantization support is compiled into the executable.
11. A new quantization normally requires one implementation file and one explicit registration line.
12. The initial checkpoint containers/conventions are Hugging Face SafeTensors, MLX SafeTensors (dense and quantized), and GGUF.
13. Mixed encodings are resolved per logical parameter, never by assigning one quantization convention to the whole model.
14. The engine automatically detects and compiles for the selected local device. Cross-compilation is not an initial requirement.
15. The engine selects activation and accumulator types per operation from target-compatible kernel candidates.
16. Softmax, RMSNorm, RoPE, attention-score reductions, and other accuracy-sensitive operations initially use conservative precision, normally F32 internally.
17. Kernel selection and precision choices are deterministic and visible in diagnostics and the executable manifest.

## 3. Explicit non-goals

- Runtime-loaded architecture plugins or arbitrary Python/Transformers imports.
- Content-dependent kernel selection or inspection of weight values during compilation.
- Runtime autotuning in the first implementation.
- Automatic fallback to CPU or another device after selecting the preferred local device.
- A portable VMFB intended to run across materially different devices.
- Claiming every parsed GGUF or MLX encoding as executable support.
- Using host dequantization as a correctness path.
- Embedding a reference decoder in production kernel selection.

## 4. Current-state findings

The repository specification describes the desired Bound Model IR, but the implementation currently bypasses it.

- `dyninfer-architecture::ModelBuilder` records parameter slots and string notes rather than a typed executable semantic graph.
- `ArchitectureDefinition::emit_executable` lets each architecture produce executable MLIR directly.
- Llama and Qwen3 both delegate to a large `dense_decoder` emitter which reads checkpoint encodings itself.
- `COMPUTE_DTYPE` is globally fixed to F32.
- GGUF type codes are recognized broadly, but executable decoding is limited primarily to dense tensors and Q4_0.
- `GgufQ40Convention` treats Q4_0 as a model-level convention even though GGUF types are per tensor.
- Q4_0 knowledge is spread among `dyninfer-core`, the GGUF crate, the binder, architecture emission, compiler validation, runtime materialization, and the skeletal kernel registry.
- `dyninfer-kernel-registry` is not part of executable emission and lacks operation, shape, precision, and capability constraints needed for real selection.
- Target discovery occurs before compilation but the target is not passed to architecture emission; target-specific work mostly consists of IREE command-line flags.
- Target detection contains guessed/default GPU architectures. Local specialization must not compile for an assumed chip.
- Runtime loading can construct host-expanded parameter blobs. This violates direct checkpoint addressing and can disagree with MLIR expecting packed bytes.
- IREE's generic file-index parser is sufficient only for file formats it understands. GGUF and compound MLX bindings require a programmatically constructed file-backed parameter index.
- The cache already includes the checkpoint schema and target fingerprint, but both fingerprints need stronger, explicitly code-relevant definitions.

## 5. Target data flow

```text
local device discovery
        |
        v
exact LocalTargetProfile + capability fingerprint

checkpoint file(s)                      compiled-in architecture
        |                                         |
container indexing                               v
        |                                typed Architecture IR
per-parameter convention decoding                 |
        |                                         |
        +-------------- binding ------------------+
                               |
                               v
                        typed Bound Model IR
                               |
                    shape + precision specialization
                               |
              strict per-operation kernel selection
                               |
                  standard/quantized MLIR lowering
                               |
                         IREE compilation
                               |
                 VMFB + specialization manifest

runtime: VMFB + original checkpoint file-backed parameter index
```

The architecture graph must be checkpoint- and quantization-independent. The Bound Model IR is the first representation that may associate a semantic operation with a physical checkpoint encoding.

## 6. Core data model

### 6.1 Physical encodings

Keep physical encoding descriptors serializable and data-only. Do not put trait objects or lowering callbacks in catalogs or manifests.

Each `PhysicalEncoding` must fully describe the code-relevant storage contract:

- stable encoding ID and version;
- logical element type;
- stored element/container type;
- block or group shape;
- signedness and bit packing;
- quantization axis;
- scale type and placement;
- zero-point behavior;
- orientation/order;
- named storage components;
- component shape and length rules;
- endianness and alignment requirements.

Remove special helpers such as `PhysicalEncoding::gguf_q4_0`, `is_supported_v1`, and codec-specific byte validation from `dyninfer-core`. Core types describe encodings; registered definitions validate and lower them.

### 6.2 Architecture IR

Replace graph-sketch strings with a typed, serializable semantic graph. At minimum it must represent:

- graph values and tensor shapes;
- parameter slots and logical parameter references;
- embedding/gather;
- linear;
- RMSNorm and per-head Q/K normalization;
- RoPE;
- attention, including MHA/GQA and KV-cache reads/writes;
- elementwise activation/gating operations;
- residual operations;
- output projection;
- prefill and decode exports.

Architecture IR operations describe semantics, not MLIR implementation. `Linear` does not know whether its weight is Q4, Q8, BF16, SafeTensors, or GGUF.

Use shared composite builder helpers for common decoder blocks, but expand them into typed semantic operations before binding. Do not leave an opaque `dense_block` that requires a single architecture-specific executable emitter.

### 6.3 Bound Model IR

Introduce an explicit `BoundModel` containing:

- the resolved architecture graph and configuration;
- every parameter binding;
- logical shapes and transforms;
- physical encoding descriptors;
- one or more named storage-component bindings;
- source-independent external parameter keys;
- specialized prefill/decode shapes;
- the selected local target;
- the precision policy;
- selected kernel IDs and lowering metadata after selection.

Keep host paths and checkpoint byte offsets out of compiler IR. MLIR references stable external scope/key names. The runtime reconstructs those keys as file-backed ranges from the catalog associated with the supplied checkpoint.

### 6.4 Storage component keys

Define a deterministic external key for every physical component, for example:

```text
weights::<canonical parameter>::data
weights::<canonical parameter>::scales
weights::<canonical parameter>::biases
weights::<canonical parameter>::zero_points
```

Do not rely on a format-specific raw tensor name as the sole identity. A logical MLX quantized parameter may bind several SafeTensors entries, while a GGUF block encoding may expose one interleaved byte range.

## 7. Extension registries

### 7.1 Architecture registry

Retain explicit static registration in `dyninfer-architecture`.

Each `models/<architecture>.rs` file owns:

- architecture ID and revision;
- recognized model-type aliases;
- configuration schema and validation;
- raw-name to canonical-name mapping and tied-parameter rules;
- construction of typed semantic Architecture IR.

Remove `ArchitectureDefinition::emit_executable`. Model files must not import quantization modules or select MLIR kernels. Enforce this structurally by keeping `dyninfer-architecture` independent of the quantization crate.

Adding an architecture should require:

1. `models/new_arch.rs` implementing `ArchitectureDefinition`;
2. `pub mod new_arch` plus one `register_all` entry.

Provide an architecture conformance test helper that validates graph integrity, required slots, prefill/decode exports, and canonical naming against a small catalog fixture.

### 7.2 Quantization crate

Add `crates/dyninfer-quantization`. Keep the generic selection machinery in `dyninfer-kernel-registry`, but replace its string-only candidates with typed requests and constraints.

Each `quantizations/<encoding>.rs` file owns:

- stable encoding IDs/versions and external format tags it recognizes;
- schema construction and physical-layout validation;
- exact byte-size/component rules;
- supported semantic consuming operations;
- candidate activation/output/accumulator types;
- target capability and shape constraints;
- production MLIR lowering implementations;
- test-only reference unpack/dequantization and fixture helpers;
- unit and differential tests for that encoding.

Adding a quantization should require:

1. `quantizations/new_encoding.rs` implementing `QuantizationDefinition`;
2. `pub mod new_encoding` plus one `register_all` entry.

An illustrative interface—not a required final spelling—is:

```rust
trait QuantizationDefinition: Send + Sync {
    fn id(&self) -> EncodingId;
    fn version(&self) -> u32;
    fn external_tags(&self) -> &[ExternalEncodingTag];
    fn validate(&self, parameter: &LogicalParameter) -> Result<()>;
    fn candidates(
        &self,
        request: &KernelRequest,
        target: &LocalTargetProfile,
        precision: &PrecisionPolicy,
    ) -> Result<Vec<KernelCandidate>>;
    fn lower(
        &self,
        selected: &SelectedKernel,
        context: &mut LoweringContext,
    ) -> Result<()>;
}
```

The actual API should separate serializable descriptors from implementation objects and avoid cyclic crate dependencies.

### 7.3 Kernel registry

Replace `op: String`, `encoding: String`, and static priority with typed fields:

- operation kind;
- encoding ID/version;
- supported input, output, and accumulator dtypes;
- shape/divisibility constraints;
- parameter orientation constraints;
- backend and exact capability constraints;
- prefill/decode applicability;
- lowering ID;
- deterministic cost score;
- production-readiness status.

Only production-ready candidates participate in compilation. Scalar reference loops and host decoders remain test-only. A generic dense `linalg` lowering is a valid production candidate when IREE has an appropriate target path; an intentionally slow unpack/dequantize fallback is not.

Selection operates on every consuming operation, not once per tensor or layer. The same encoded parameter used by two operation shapes may select different prefill and decode kernels.

## 8. Checkpoint ingestion

### 8.1 Container versus convention

Container readers only produce raw entries, metadata, source files, and byte ranges. Convention decoding groups raw entries into logical parameters and attaches a complete physical encoding descriptor.

Do not select conventions based on a model-wide quantization type. A GGUF convention iterates every tensor and resolves each GGUF type independently through registered encoding definitions.

### 8.2 GGUF

- Keep `dyninfer-checkpoint-gguf` responsible for GGUF parsing, metadata, reversed dimension handling, and absolute data ranges.
- Replace `GgufQ40Convention` and `GgufDenseConvention` with one mixed GGUF convention.
- Resolve the `ggml_type` of every tensor separately.
- Preserve packed bytes exactly as stored.
- Reject unknown sizing/layout metadata during inspection.
- Recognizing a type code does not imply an executable kernel exists; executable support is decided after binding against the local target.

### 8.3 Hugging Face SafeTensors

- Keep one generic SafeTensors container indexer.
- Decode ordinary F32/F16/BF16 tensors as plain physical encodings.
- Support sharded SafeTensors using the index JSON and multiple `SourceFile` entries.
- Keep parameter bytes in their original shard and expose exact ranges.

### 8.4 MLX SafeTensors

- Add a distinct MLX convention decoder over the same raw SafeTensors index.
- Detect MLX convention metadata and naming without misclassifying ordinary dense SafeTensors.
- Group a quantized logical weight with its scale, bias, and/or zero-point tensors according to the documented MLX convention.
- Emit one `LogicalParameter` with multiple named `StorageComponent`s.
- Read group size, bit width, packing, and logical orientation from checkpoint/config metadata; do not infer missing critical properties silently.
- Add fixtures captured from real MLX exports for every supported layout.

NPZ support is outside this initial scope. Update documentation so it is not implied by the MLX SafeTensors requirement.

## 9. Binding and strict support validation

Binding must perform only semantic association and schema validation. It must not choose host materialization.

Change `ParameterBinding` and `BindingPlan` as follows:

- retain all component bindings, not only the first component/key;
- remove `MaterializationPolicy`, `MaterializationRequest`, and `BindingTransform::Repack` from executable paths;
- allow only logical transforms that a selected kernel can consume without rewriting bytes;
- defer final transform legality to kernel selection;
- remove architecture-declared lists such as `supported_encodings = [plain, q4_0]`; support belongs to registered operation/encoding/target candidates;
- validate exact rank, logical shape, tied weights, component completeness, component lengths, and alignment.

After binding, run a coverage pass over all parameter-consuming operations. For each operation, either select a production kernel or emit a structured error containing:

- architecture operation ID and kind;
- parameter slot and checkpoint component keys;
- physical encoding ID/version;
- logical and storage shapes;
- selected target and relevant missing capability;
- rejected candidate IDs and reasons;
- suggested actionable next step, such as implementing a particular encoding/operation kernel.

There is no host-dequantize branch and no device fallback branch.

## 10. Local target discovery

Replace broad/guessed target profiles with a verified `LocalTargetProfile` created before specialization.

The profile should record as available:

- exact runtime device URI and stable device identity;
- IREE driver/backend;
- CPU triple and detected ISA features, or GPU architecture/chip;
- native F16/BF16/integer/dot-product capabilities;
- subgroup/warp properties relevant to candidates;
- alignment, allocation, and workgroup limits needed by lowerings;
- IREE executable target flags derived from detected facts;
- a canonical capability fingerprint.

Requirements:

- Remove default guesses such as compiling for `gfx1151` when detection did not report it.
- Select the preferred local device deterministically and report it before compilation.
- Do not silently retry another device if kernel coverage fails.
- Fail early when the exact compile target cannot be derived from the selected runtime device.
- Feed the target into precision policy, kernel selection, MLIR generation, IREE flags, manifest generation, and cache keys.

Tool-backed discovery may remain temporarily, but add or extend the IREE C wrapper for capability queries that cannot be obtained reliably from `iree-run-module --dump_devices`.

## 11. Precision policy

Add an explicit, versioned `PrecisionPolicy`. The initial policy is `ConservativeSensitiveOps`.

Rules:

- Weight storage type/encoding is immutable and comes from the checkpoint.
- Linear and quantized-matmul candidates may propose native activation and accumulator types supported by the local target.
- Candidate ranking balances throughput and precision deterministically.
- Sensitive operations use F32 internal computation initially: softmax, normalization reductions, RoPE angle/application math, attention-score scaling/reduction, and similar reductions.
- Conversions between operation boundaries are explicit typed IR/MLIR operations.
- Reject when no candidate satisfies both target constraints and the precision floor.
- Record input, output, activation, and accumulator dtypes for every selected kernel.

Do not replace the global F32 constant with a different global type. Remove the global model entirely and make precision an operation-level decision.

## 12. Shared specialization and lowering pipeline

Move executable production out of architecture definitions and into `dyninfer-compiler` (with reusable MLIR builders in `dyninfer-mlir`). Implement this ordered pipeline:

1. Verify typed Architecture IR.
2. Resolve configuration and infer logical shapes.
3. Apply the binding plan to produce Bound Model IR.
4. Validate physical encoding descriptors and component completeness.
5. Specialize prefill and decode shapes.
6. Apply the precision policy per operation.
7. Enumerate production kernel candidates.
8. Reject incomplete coverage with structured diagnostics.
9. Select kernels deterministically.
10. Lower external component loads to stable IREE parameter keys.
11. Lower semantic operations through selected dense or quantized implementations.
12. Verify emitted MLIR.
13. Invoke IREE with exact local-target flags.
14. Emit VMFB and a specialization manifest.

Prefill and decode are distinct selection requests. They may use different kernels and activation types.

Delete the direct `DenseDecoderConfig::from_package(package, catalog)` specialization path after the shared pipeline passes parity tests. Break up `dense_decoder.rs` into semantic-op lowerings and reusable entrypoint/KV helpers; quantized branching must leave that crate.

## 13. Direct checkpoint parameter provider

Implement a file-backed parameter-index ABI in `third_party/iree_runtime_c_api`, `iree-runtime-sys`, and `iree-runtime`.

The Rust side passes a provider plan containing:

```rust
struct FileBackedParameter {
    scope: String,
    key: String,
    source_file_index: u32,
    offset: u64,
    length: u64,
}

struct FileBackedParameterPlan {
    source_files: Vec<PathBuf>,
    parameters: Vec<FileBackedParameter>,
}
```

The C wrapper must:

1. open each original checkpoint/shard read-only;
2. create an IREE parameter index;
3. add each external key as a file-backed entry using the original handle, offset, and length;
4. create the `weights` provider/module;
5. retain file handles/index/provider for the required session lifetime;
6. append the VMFB module;
7. release all resources correctly on success and every failure path.

The plan is rebuilt when loading a bundle by re-inspecting the supplied checkpoint and validating its schema. Do not persist absolute paths in the VMFB or cache key.

This provider is used uniformly for SafeTensors, sharded SafeTensors, MLX components, and GGUF. Do not ask IREE to parse GGUF as an archive. Do not construct whole-model host blobs.

Delete production uses of:

- `HostParameterStorage` for checkpoint weights;
- GGUF `decode_parameters_as_f32_host` variants;
- SafeTensors F32 promotion/materialization;
- the cache `parameters/` directory and derived-parameter APIs.

Reference decoding remains behind test-only APIs for differential validation.

## 14. Manifest and cache correctness

### 14.1 Schema fingerprint

Redefine the checkpoint schema fingerprint to include all code-relevant, value-independent facts:

- canonical logical name and role;
- logical shape and element type;
- complete physical encoding descriptor and version;
- component names, storage types, shapes, byte lengths, endianness, and required alignment;
- logical orientation and transform requirements;
- tied/alias relationships.

Exclude:

- weight bytes and content digest;
- absolute paths;
- incidental physical offsets when the provider can remap stable component keys;
- timestamps and source repository identities.

Add a test proving that checkpoints with different values and offsets but identical code-relevant schemas share a VMFB cache key and load through distinct runtime provider plans.

### 14.2 Specialization manifest

Extend `ExecutableManifest` with:

- exact local target profile and fingerprint;
- precision policy ID/version;
- kernel registry and quantization registry versions;
- one record per selected operation containing operation ID, parameter slot, encoding, kernel ID, prefill/decode mode, and all compute dtypes;
- direct parameter scopes/keys and expected byte lengths, without paths;
- explicit statement that no derived parameters are required;
- compiler and IREE revisions.

At load time, validate the manifest target against the currently opened device. Reject rather than running a VMFB compiled for a different capability fingerprint.

### 14.3 Cache contents

Executable cache entries contain only:

- VMFB;
- manifest;
- cache-key metadata;
- optional MLIR/diagnostic dumps when explicitly requested.

They never contain weight or repacked parameter bytes.

## 15. Initial encoding rollout

Build the new infrastructure without carrying the legacy fallback forward.

### Stage A: dense parity

- Implement plain F32, F16, and BF16 external parameters through the direct provider.
- Implement target-aware dense linear candidates and conservative sensitive ops.
- Bring Llama and Qwen3 prefill/decode through the shared Bound Model IR pipeline.
- Remove any BF16/F16-to-F32 whole-weight promotion. If a target cannot consume the stored dense type through an approved kernel, reject it.

### Stage B: Q4_0 through the production interface

- Move Q4_0 layout, validation, candidate selection, and lowering into `quantizations/q4_0.rs`.
- Treat the current scalar nested-loop qkernel as reference/prototype code until it meets the production performance bar for an explicitly supported target family.
- Support direct packed GGUF bytes for every operation claimed by the definition.
- If embeddings or output projections use Q4_0, require corresponding quantized gather/linear support; never host-dequantize them.

### Stage C: mixed GGUF

- Add production definitions in priority order driven by real target checkpoints, beginning with Q8_0 and the required K-quant variants.
- Add a mixed fixture containing at least dense BF16, Q8, and Q4 tensors across and within layers.
- Do not register a format/operation/target combination until correctness and performance qualification pass.

### Stage D: MLX quantized SafeTensors

- Implement compound component binding for actual MLX fixtures.
- Add the corresponding groupwise quantization definition(s).
- Address packed weight, scale, bias, and zero-point components directly from SafeTensors ranges.
- Reuse the same operation-level kernel selection path used by GGUF.

Each stage ends with removal of any legacy branch that could silently make its tests pass through host F32 weights.

## 16. Performance qualification

Because a very slow fallback is considered unsupported, correctness alone is insufficient to register a production quantized kernel.

For each `(kernel ID, target family, mode)` registration:

- maintain differential correctness tests against a test-only reference decoder;
- benchmark representative prefill and decode shapes;
- demonstrate that weights remain packed until consumed by the device kernel;
- record the supported architecture/chip capability predicate;
- document the evidence and threshold used to mark the candidate production-ready;
- keep unqualified candidates out of the production registry.

Benchmarks should report latency, effective weight bandwidth, and temporary memory. CI may run correctness and structural checks while hardware-specific performance qualification runs on designated machines.

## 17. Implementation sequence

Work in dependency order. Keep the tree runnable at the end of each phase, but prefer an explicit unsupported error over routing through legacy materialization.

### Phase 1: contracts and typed IR

- Add this design's core types, `PrecisionPolicy`, operation IDs/kinds, Architecture IR, and Bound Model IR.
- Convert `ModelBuilder` from string notes to typed operations.
- Convert Llama and Qwen3 model files to the typed builder.
- Add architecture conformance tests.
- Do not change production execution yet beyond what is needed to serialize and inspect the new IR.

### Phase 2: registries and strict coverage

- Add `dyninfer-quantization` to Cargo and Bazel.
- Replace the skeletal kernel registry with typed candidates and rejection reasons.
- Add explicit registry construction in `dyninfer-runtime::builtins`.
- Implement plain dense candidates first.
- Add a dry-run coverage report command/test that selects kernels without compiling.

### Phase 3: per-parameter checkpoint conventions

- Replace model-wide GGUF quantization convention selection with mixed per-tensor decoding.
- Strengthen physical encoding and schema fingerprints.
- Implement MLX SafeTensors convention detection and compound logical parameters.
- Add sharded SafeTensors indexing/provider metadata.
- Add inspection and binding fixtures before kernel implementation.

### Phase 4: exact local target and precision

- Remove guessed target architectures.
- Populate exact capability profiles and fingerprints.
- Select the local device before specialization.
- Implement conservative per-operation precision selection.
- Add target/precision information to diagnostics and cache keys.

### Phase 5: direct file-backed IREE parameters

- Add the descriptor-array C ABI and safe Rust wrapper.
- Support multiple files and multiple components.
- Switch dense SafeTensors e2e tests to the direct provider.
- Add direct GGUF packed-byte provider tests independent of full inference.
- Verify no derived files or full-model host buffers are created.

### Phase 6: shared compiler pipeline

- Produce Bound Model IR from architecture, binding, shapes, target, and policy.
- Select kernels before MLIR emission.
- Move common transformer lowering out of architecture definitions.
- Generate separate prefill/decode lowering decisions.
- Compile and run dense Llama/Qwen3 parity tests.
- Remove `ArchitectureDefinition::emit_executable` and the monolithic specialization path.

### Phase 7: quantized production kernels

- Port and qualify Q4_0 through the new definition interface.
- Add Q8_0 and required K/groupwise definitions using real mixed checkpoints.
- Add MLX quantized definitions and compound component loads.
- TODO: Add NVFP4 as a distinct mixed per-tensor encoding: packed E2M1 values,
  one FP8 E4M3 block scale per 16 values, and the format's global F32 scale
  component where present. Model NVIDIA NVFP4 and MLX `mode=nvfp4` as separate
  versioned layouts if their serialized scale conventions differ; do not infer
  equivalence from the shared name.
- TODO: Gate NVIDIA NVFP4 kernels on an exact locally detected Blackwell-class
  capability (currently CUDA `sm_100` or newer), keep them unavailable on a
  generic/guessed CUDA profile, and qualify linear, embedding, and output
  projection independently for prefill and decode. Never alias NVFP4 to GGUF
  MXFP4, affine U4, or another 4-bit encoding.
- TODO: Add explicitly separate direct packed NVFP4 software candidates for
  HIP (and CPU only if it meets the production bar). These backends
  must implement the same per-tensor mixed-schema contract without claiming
  native NVFP4 tensor-core acceleration; keep them out of production selection
  until differential correctness, temporary-memory, and throughput gates pass.
- Add structured rejection tests for every known-but-unimplemented combination.

Implemented baseline (2026-08-03): mixed GGUF Q4_0/Q4_1/Q8_0/Q6_K and MLX affine
U4 have strict per-operation implementations for embedding, linear, and output
projection in prefill and decode. Real Qwen3-0.6B checkpoints compile and run
directly on CPU and HIP. Unsloth UD-Q4_K_XL and UD-IQ1_S remain mixed
schema test cases and reject only their still-unimplemented IQ/K encodings.
GPU MLX currently uses an on-device dequantization dispatch boundary before the
F32 contraction; fused sub-byte GPU contraction and full GPU performance
qualification remain follow-up work.

### Phase 8: delete fallback/materialization paths

- Remove host weight decoding and promotion from runtime loading.
- Remove derived parameter cache APIs/directories.
- Remove materialization policies and repack transforms from binding/compiler paths.
- Remove legacy Q4 branches from architecture and compiler crates.
- Update README/spec/CLI help to distinguish inspectable formats from executable support.

### Phase 9: extensibility proof

- Add one small second architecture or architecture variant using only a new model file and registration line.
- Add one additional quantization using only a new quantization file and registration line.
- Treat any required unrelated edits as a failure of the extension interfaces and fix the abstraction before declaring completion.

## 18. Test matrix

### Unit tests

- Encoding byte-size, block/group divisibility, component, orientation, and alignment validation.
- Schema fingerprint stability and sensitivity.
- Architecture graph shape inference and slot validation.
- Kernel candidate filtering with explicit rejection reasons.
- Precision decisions for sensitive and linear operations.
- Local-target fingerprint determinism.
- Parameter-plan construction for single-file, sharded, GGUF, and compound MLX storage.

### MLIR structural tests

- External globals use stable component keys and original packed/storage types.
- Mixed layers call different selected lowerings.
- No whole-weight dequantization function or dense shadow global appears.
- Sensitive operations contain the required F32 conversions/reductions.
- Prefill and decode may contain different selected kernels.
- Emitted MLIR verifies before IREE compilation.

### Runtime integration tests

- Dense F32/F16/BF16 SafeTensors run directly from original files.
- Sharded SafeTensors bind across multiple original files.
- Mixed GGUF runs with packed parameters and no host expansion.
- Quantized MLX SafeTensors bind all auxiliary components directly.
- Supplying a schema-compatible checkpoint with different values reuses the VMFB and produces different expected outputs.
- Supplying a schema-incompatible checkpoint is rejected before invocation.
- Loading on a different target fingerprint is rejected.
- Missing embedding/gather support rejects the whole compile rather than dequantizing that tensor.

### Failure tests

- Known encoding with no local-target kernel.
- Kernel exists for linear but not embedding.
- Unsupported block divisibility or orientation.
- Missing MLX scale/bias component.
- Unknown/ambiguous MLX quantization metadata.
- Target detected but exact compile architecture unavailable.
- Candidate violates conservative precision floor.
- Runtime component range differs from manifest length.

### Resource-behavior tests

- Bundle and cache contain no weight-like artifact.
- Loading does not allocate a persistent host buffer proportional to total model weight size.
- Parameter descriptors reference original source files and ranges.
- Repeated sessions reuse file-backed parameter definitions while retaining independent execution/KV state.

## 19. Completion criteria

This effort is complete only when all of the following are true:

1. Architecture files contain no quantization-specific branches and no executable MLIR emitter.
2. Adding an architecture requires one model file plus one registry line.
3. Adding a quantization requires one definition/lowering file plus one registry line.
4. The compiler consumes a real typed Bound Model IR.
5. Kernel selection receives operation, encoding, shape, mode, precision policy, and exact local target.
6. Every selected kernel and dtype is present in the manifest.
7. Mixed-encoding GGUF and quantized MLX SafeTensors pass direct-addressing e2e tests.
8. No runtime path dequantizes or promotes whole checkpoint weights.
9. No cache or bundle contains derived weight bytes.
10. Unsupported operation/encoding/target combinations fail with actionable diagnostics.
11. VMFB reuse is proven across different checkpoint values with identical schemas.
12. The VMFB is rejected on a mismatched local-target capability fingerprint.
13. Documentation distinguishes container recognition, schema decoding, and executable kernel support.

## 20. Guardrails for coding agents

- Do not preserve a legacy fallback merely to keep an e2e test green; change the test to expect an explicit unsupported error until the production kernel exists.
- Do not add codec-specific `match` branches to architecture files, the generic compiler driver, or runtime loading.
- Do not make `PhysicalEncoding` claim global support with methods such as `is_supported_v1`; support is a property of an operation/encoding/shape/target/policy request.
- Do not read weight payloads during compilation or cache-key construction.
- Do not include absolute checkpoint paths or weight content hashes in VMFB cache keys.
- Do not infer missing critical quantization metadata from a filename.
- Do not use a guessed GPU chip or ISA feature set.
- Do not call a scalar reference implementation a production quantized kernel.
- Preserve existing unrelated worktree changes; implement this effort in reviewable phases with focused commits.
