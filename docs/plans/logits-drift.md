# Differential Logit Drift Against llama.cpp

**Status:** proposed
**Audience:** repository maintainers and coding agents
**Scope:** a development-only `dyninfer logits drift` command and a small
libllama companion used to compare final vocabulary logits from the same GGUF
checkpoint and token trajectory

## 1. Outcome

Add a differential utility that answers three questions without changing the
production inference ABI:

1. Do dyninfer and llama.cpp produce the same vocabulary-sized raw logits after
   the prompt?
2. How does the difference evolve over forced, identical decode steps?
3. Does the difference change the greedy token choice?

The default reference is llama.cpp on CPU. Both engines open the exact same
GGUF path. llama.cpp tokenizes the prompt from the tokenizer embedded in that
GGUF, and dyninfer replays the returned token IDs. After prefill, llama.cpp's
raw argmax tokens drive both decode sessions. The command therefore continues
to compare like-for-like states even if dyninfer's own argmax diverges.

The utility is diagnostic. It MUST NOT become a production fallback, a kernel
selection input, or a reason to read checkpoint payloads during compilation.

## 2. Current code and upstream findings

### 2.1 dyninfer already has the correct public boundary

`dyninfer-runtime::ModelSession` returns one `Logits` value from both `prefill`
and `decode`. `Logits.values` is a vocabulary-sized `Vec<f32>`.

The current `IreeSession` semantics are:

- `prefill(prompt)` returns logits after the final real prompt token and sets
  the next write position to `prompt.len()`;
- `decode(token)` writes that token at the current position, advances the
  position by one, and returns logits predicting the following token;
- the static ABI right-pads to the compiled prefill window and selects the last
  real row;
- the paged ABI runs all chunks but requests host logits only for the final
  prompt chunk;
- `prefill_argmax` and `decode_argmax` can skip the vocabulary-sized device to
  host copy, so the drift command must intentionally use `prefill` and `decode`.

No IREE or compiler ABI change is needed for final-logit comparison.

The executable manifest already records the selected target, precision policy,
kernel choices, shape profile, prefill window, and KV-cache descriptor. Those
facts belong in every drift report because they are necessary to reproduce and
interpret a difference.

### 2.2 Stock llama.cpp executables are close but insufficient

The inspected llama.cpp package is build 10216 at commit `06be260`; it provides
`llama-cli`, `llama-perplexity`, `llama-results`, and `llama-eval-callback`.

- `llama-cli` does not expose raw full-vocabulary logits.
- `llama-perplexity --save-all-logits` is organized around corpus windows and
  perplexity/KL workflows, not a prompt followed by forced decode tokens.
- `llama-results` is the closest existing tool. It tokenizes a prompt, marks
  every prompt token for output, calls `llama_decode`, and writes token IDs and
  all logits to a GGUF result file. It cannot accept an exact token-ID stream
  or exercise a one-token-at-a-time KV-cache decode trajectory.
- `llama-eval-callback` observes internal graph tensors. Tensor names and graph
  topology are intentionally not a stable public model API, and copying every
  selected tensor is much more intrusive than reading final logits.

The public libllama API has the precise operations required by this utility:

- `llama_tokenize` for the GGUF's embedded tokenizer;
- `llama_decode` for prompt and one-token forced decode batches;
- `llama_get_logits_ith(ctx, -1)` for the last requested raw logit row;
- `llama_vocab_n_tokens` for the row width.

`llama_get_logits_ith` synchronizes pending backend work before returning the
host pointer. The helper must copy exactly `n_vocab * sizeof(float)` bytes
before the next `llama_decode` call.

## 3. Decision

Implement two processes joined by a versioned file protocol.

```text
one GGUF path
     |
     +--> dyninfer logits drift
     |      |-- ask companion to tokenize prompt
     |      |-- compile/load dyninfer bundle
     |      |-- ask companion for reference trace
     |      |-- replay identical IDs through ModelSession
     |      `-- stream metrics + write report
     |
     `--> dyninfer-llama-logits
            |-- public llama.h / libllama only
            |-- prompt decode, then one-token decodes
            `-- trace metadata + little-endian F32 rows
```

Do not link libllama into `dyninfer-runtime` or `dyninfer-cli`. Keeping it in a
companion process has four useful properties:

- the normal dyninfer build and runtime do not acquire a llama.cpp ABI or
  packaging dependency;
- a helper built against one llama.cpp installation cannot corrupt the Rust
  process if its ABI differs;
- the reference process can use any user-selected llama.cpp backend;
- the token/logit trace is independently inspectable and reusable.

The companion should be a small C++ program rather than a Python or Rust
binding. It uses only the installed public headers and libraries, follows
llama.cpp's own `simple` and `results` examples, and avoids a second wrapper's
batching, sampling, and score-retention behavior.

## 4. CLI

Add a nested command so future dump/compare utilities have a natural home:

```text
dyninfer logits drift [OPTIONS]
```

Typical invocation:

```bash
bazel run //crates/dyninfer-cli:dyninfer -- logits drift \
  --checkpoint /models/Qwen3-0.6B-Q4_K_M.gguf \
  --target cpu \
  --prompt 'Explain why the sky is blue.' \
  --decode-steps 8 \
  --llama-runner ./bazel-bin/tools/llama-logits/dyninfer-llama-logits \
  --json-out /tmp/qwen-drift.json
```

Core options:

| Option | Behavior |
|---|---|
| `--checkpoint PATH` | Required single GGUF path passed to both engines. |
| `--architecture ID` | Defaults to `auto`, using the existing dyninfer resolver. |
| `--bundle PATH` | Reuse an existing dyninfer bundle; otherwise compile a temporary one. |
| `--target TARGET` | Dyninfer compile target when `--bundle` is absent; defaults to `cpu`. |
| `--prompt TEXT` | Tokenized by libllama from the supplied GGUF. |
| `--tokens IDS` | Exact comma-separated prompt IDs; conflicts with `--prompt`. |
| `--tokens-file PATH` | Exact whitespace/JSON token IDs; conflicts with the other prompt inputs. |
| `--decode-steps N` | Number of reference-greedy tokens to force through both engines. |
| `--decode-tokens IDS` | Explicit forced decode inputs instead of reference-greedy inputs. |
| `--add-special auto\|yes\|no` | libllama prompt tokenization policy; defaults to GGUF metadata. |
| `--parse-special` | Let libllama recognize textual special-token spellings. |
| `--llama-runner PATH` | Companion path; defaults to `dyninfer-llama-logits` on `PATH`. |
| `--llama-device DEVICE` | Defaults to CPU/no offload. |
| `--llama-gpu-layers N` | Defaults to `0`; explicit when using a GPU reference. |
| `--llama-threads N` | Optional explicit reference thread count. |
| `--llama-flash-attn MODE` | Defaults to `off` for the CPU reference; always recorded. |
| `--top-k N` | Top-token diagnostics, default `10`. |
| `--json-out PATH` | Durable machine-readable report. |
| `--keep-traces DIR` | Retain the two trace artifacts for later inspection. |

Thresholds are opt-in:

```text
--max-abs F
--max-rmse F
--max-relative-l2 F
--min-cosine F
--require-argmax-match
```

Without thresholds, the command exits nonzero only for execution/protocol
errors, incompatible rows, or non-finite logits. There is no universal safe
tolerance across dense and quantized checkpoints, targets, KV dtypes, and
kernel choices, so the utility must not pretend that one exists.

## 5. Exact comparison semantics

Let the prompt token IDs be `p[0..P)` and the forced decode inputs be
`d[0..D)`.

Row zero is:

```text
prefill(p) -> logits predicting the token after p[P - 1]
```

Each subsequent row is:

```text
decode(d[i]) at position P + i -> logits predicting the following token
```

The trace therefore contains exactly `1 + D` rows.

Each row records:

- phase (`prefill` or `decode`);
- input token and its absolute position;
- raw reference argmax;
- forced token used for the next row, when one exists;
- a contiguous F32 logit vector in vocabulary token-ID order.

For reference-greedy mode, `d[0]` is the raw argmax of the llama.cpp prefill
row, and `d[i + 1]` is the raw argmax of llama.cpp decode row `i`. No sampler,
temperature, repetition penalty, grammar, logit bias, or softmax is applied.
dyninfer replays those IDs even when its argmax differs. This prevents the
comparison from becoming invalid after the first top-1 disagreement.

An optional later `--free-run` mode may let both engines feed back their own
greedy choices and report the first sequence divergence. That is a generation
coherency check, not a logit-drift metric, and should remain separate in the
report.

## 6. Matching execution facts

The command must make semantic mismatches visible rather than silently choosing
defaults.

### 6.1 Checkpoint and vocabulary

- Accept exactly one checkpoint argument and pass its canonical path to both
  processes.
- Reject non-GGUF checkpoints for this command.
- Record file size, schema fingerprint, and SHA-256 of the GGUF in the report.
- Require identical vocabulary sizes and exact token-ID arrays before comparing
  any row.
- Do not compare token strings; token ID is the model input contract.

### 6.2 KV cache

After loading or compiling the dyninfer bundle, configure llama.cpp K and V
cache types to match `manifest.kv_cache.element_type` when libllama supports
that type. Reject an unsupported match unless the user explicitly supplies an
override. An override is recorded as a comparison caveat.

The llama.cpp context capacity and dyninfer `SessionConfig` must both be at
least `prompt_tokens + decode_steps`. Neither process may shift, truncate, or
reuse an older context.

### 6.3 Batching and backends

The two engines do not need identical graph or padding geometry; the purpose is
to compare their implementation of the same causal model state. They do need
the same token IDs, positions, checkpoint bytes, and model metadata.

Record the following because they can change floating-point reduction order:

- dyninfer target profile, capability fingerprint, precision policy, selected
  kernels, prefill window/chunk size, and KV dtype;
- llama.cpp build number/commit when discoverable, loaded libllama path, device,
  GPU layers, thread counts, batch/microbatch sizes, flash-attention mode, and
  KV dtypes.

The default llama.cpp CPU/no-offload configuration is a reference, not a claim
that its numerical order matches dyninfer CPU or GPU. Users can select a
llama.cpp device explicitly when backend-to-backend drift is the question.

## 7. Trace protocol

Use a directory with one JSON header and one raw data file:

```text
trace/
|-- trace.json
`-- logits.f32le
```

`trace.json` has a stable format identifier and version:

```json
{
  "format": "dyninfer.logit-trace",
  "version": 1,
  "engine": "llama.cpp",
  "checkpoint_sha256": "...",
  "vocab_size": 151936,
  "prompt_tokens": [9707, 13],
  "decode_inputs": [198],
  "logits_dtype": "f32",
  "logits_byte_order": "little",
  "rows": [
    {"phase": "prefill", "position": 1, "input_token": 13, "argmax": 198},
    {"phase": "decode", "position": 2, "input_token": 198, "argmax": 785}
  ],
  "logits_file": "logits.f32le"
}
```

The binary file is row-major, little-endian IEEE-754 F32 with exactly
`rows.len() * vocab_size * 4` bytes. Reject a truncated file, trailing bytes,
unknown version, non-finite value, invalid token ID, non-contiguous position,
or inconsistent row count.

The companion writes a temporary sibling directory, closes both files, then
renames the directory into place so the Rust side never consumes a partial
trace. The comparator reads one reference row at a time and compares it
immediately with the corresponding dyninfer result, keeping memory
proportional to one vocabulary row. When `--keep-traces` is used, dyninfer also
writes its replayed rows in the same format with `engine = "dyninfer"`.

JSON is metadata only. Full logits must not be serialized as decimal JSON.

## 8. Metrics and output

For each row, compute in F64 accumulation:

- maximum and mean absolute error;
- RMSE;
- mean signed delta;
- relative L2 error, `||dyn - ref||2 / max(||ref||2, epsilon)`;
- cosine similarity;
- centered RMSE after removing each row's mean, to expose harmless constant
  shifts separately from shape changes;
- raw argmax agreement and reference/dyninfer top-k overlap;
- stable softmax KL(`reference || dyninfer`) and maximum probability delta;
- the token IDs with the largest absolute deltas and the top reference and
  dyninfer logits.

The human output is one compact row per inference step followed by aggregate
worst cases:

```text
step phase    pos  argmax(ref/dyn)  max_abs   rmse      rel_l2    cosine
0    prefill   11  198/198          0.03125   0.00412   8.2e-4    0.9999997
1    decode    12  785/785          0.04688   0.00501   9.7e-4    0.9999995
```

The JSON report includes all provenance, per-row metrics, aggregate extrema,
thresholds, the first argmax disagreement, and the first free-run token
disagreement when that optional check is requested. It does not include every
logit unless traces are explicitly retained.

## 9. llama.cpp companion implementation

`tools/llama-logits/main.cc` should use only the public headers from the pinned
llama.cpp release. Do not depend on llama.cpp's `common` library or internal
graph classes.

The evaluation loop is deliberately small:

1. Load backends, model, vocabulary, and a single-sequence context.
2. Tokenize only when prompt text was supplied.
3. Create a prompt batch with explicit positions `0..P)`, sequence ID zero, and
   `logits=true` only for the final prompt token.
4. Call `llama_decode` and copy `llama_get_logits_ith(ctx, -1)`.
5. Select the next forced/reference-greedy token.
6. For each decode input, submit one token at the next explicit position with
   `logits=true`, then copy the last logit row.
7. Atomically finish the trace and release context/model/backend resources.

Explicit positions and output flags make the intended row mapping reviewable.
The helper must check every libllama return value and null logit pointer.

Expose a cheap `tokenize` mode using `model_params.vocab_only = true`. The Rust
command uses this first when `--prompt` was supplied, so it can size dyninfer's
KV/session before the full llama.cpp evaluation. Exact token inputs skip this
extra process.

Provide a manual Bazel `cc_binary` target linked to pinned official llama.cpp
release archives. The binary archive supplies libllama and its CPU/Vulkan
plugins; because release binaries omit headers, the matching tagged source
archive supplies public headers only. The target is not a dependency of
`//crates/dyninfer-cli` and is excluded from ordinary wildcard builds.

## 10. dyninfer implementation placement

Keep llama.cpp orchestration and the trace protocol out of the production
runtime crates.

Suggested files:

```text
crates/dyninfer-cli/src/logits.rs       command orchestration and report output
crates/dyninfer-cli/src/logit_trace.rs trace reader/writer and validation
crates/dyninfer-cli/src/drift.rs       streaming metric calculations
tools/llama-logits/main.cc             public-libllama companion
tools/llama-logits/BUILD.bazel         manual pinned-libllama target
```

The dyninfer side uses only existing public APIs:

```rust
let model = loader.load_bundle(bundle, checkpoint)?;
let mut session = model.create_session(session_config)?;

compare(reference.row(0)?, session.prefill(&prompt_tokens)?.values)?;
for token in reference.decode_inputs() {
    compare(reference.next_row()?, session.decode(token)?.values)?;
}
```

If no bundle is supplied, reuse `ModelLoader::compile_to_bundle_with_overrides`
with `max_kv` sized for the trace. Compilation and target diagnostics should be
identical to `generate` rather than introducing a separate compiler path.

## 11. Failure behavior

Hard failures include:

- the companion is missing or its protocol version is unsupported;
- checkpoint is not GGUF or cannot be opened by either engine;
- prompt tokenization produces no tokens;
- token IDs, positions, vocabulary sizes, checkpoint digest, or row counts do
  not match;
- either engine returns a non-finite logit;
- either engine cannot allocate the requested context/KV cache;
- llama.cpp cannot use the required matching KV dtype;
- an explicitly requested threshold is exceeded.

Subprocess stderr is captured to a diagnostic file and summarized on failure.
Never parse ordinary llama.cpp log text as data. Never continue by comparing a
different token sequence, truncated vocabulary, top-N probabilities, or
post-sampler scores.

## 12. Validation plan

### Unit tests

- Trace JSON and F32LE round trips.
- Rejection of truncation, trailing bytes, wrong endianness/version, invalid
  token IDs, discontinuous positions, and non-finite logits.
- Metric goldens for identical vectors, constant offsets, scaled vectors,
  argmax changes, and very large logits requiring stable log-sum-exp.
- Threshold exit behavior and first-worst-row selection.

### Bazel integration tests

- A fake companion emits a tiny deterministic trace; the CLI validates and
  compares it without requiring libllama.
- The real companion builds and runs against the pinned CI llama.cpp package.
- A tiny GGUF prompt-final row matches `llama-results` as an independent check
  of the companion's `llama_decode`/`llama_get_logits_ith` mapping.
- A tiny dense GGUF and a supported quantized GGUF run prefill plus at least
  four forced decode steps through both engines.
- CPU and one qualified GPU dyninfer target are compared separately against the
  same CPU llama.cpp trace.
- The retained report contains checkpoint identity, dyninfer manifest facts,
  and llama.cpp build/backend facts.

### Generation coherency

Every real-model validation ends with a short deterministic greedy generation
check in addition to numeric drift. Run the existing dyninfer generation path
and llama.cpp at temperature zero, record the token sequences, and report the
first divergence. Numeric thresholds must not replace this coherency check, and
a coherent-looking string must not replace numeric logit comparison.

Use Bazel for all dyninfer builds and tests.

## 13. Rollout

### Phase 1: final logits

- Add trace types, metrics, and fake-runner tests.
- Add the public-libllama companion and pinned-release Bazel target.
- Implement prompt-final plus forced decode comparison and JSON report.
- Qualify on the existing Qwen3 BF16 GGUF, then one supported quantized GGUF.

### Phase 2: easier bisection

If final drift is too large, add dyninfer debug exports for a small stable set
of semantic boundaries such as post-embedding and post-layer residual. A
matching llama.cpp eval-callback mode may then be used behind an explicitly
versioned, debug-only adapter.

Do not start with arbitrary tensor callbacks. First establish a reliable final
logit trace and identify a real need for intermediate bisection. Intermediate
activation names and layouts are not a stable cross-engine contract.

## 14. Acceptance criteria

The initial utility is complete when:

1. One GGUF path and one exact token trajectory are demonstrably used by both
   engines.
2. Prompt-final and at least four sequential decode rows are compared before
   sampling.
3. llama.cpp is accessed only through its public API in a separate process.
4. Normal dyninfer binaries do not link to or require libllama.
5. Reports contain enough checkpoint, target, kernel, precision, KV, llama.cpp,
   and batching provenance to reproduce a result.
6. Full logits use the bounded-memory F32LE trace rather than JSON or log
   scraping.
7. Shape/token/protocol mismatches and non-finite values fail explicitly.
8. Tolerances are user- or test-specified, never silently universal.
9. Bazel tests cover the protocol and metrics, and a pinned real-model job
   covers dense/quantized prefill, decode, and generation coherency.
10. No production compile, load, or kernel-selection path uses llama.cpp as a
    fallback or oracle.

## 15. Relevant llama.cpp sources

- `include/llama.h`: batch output flags, `llama_decode`, and logit accessors.
- `examples/simple/simple.cpp`: minimal public-API model/context/decode loop.
- `tools/results/results.cpp`: full-vocabulary logit copying and GGUF result
  output.
- `tools/perplexity/perplexity.cpp`: all-logit corpus evaluation.
- `examples/eval-callback/eval-callback.cpp` and `common/debug.cpp`: graph tensor
  callback behavior reserved for the later debug-bisection phase.
