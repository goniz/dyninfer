# Qwen3-0.6B BF16 llama.cpp baseline and DynInfer comparison on Strix Halo

This document records a llama.cpp reference baseline and a comparable DynInfer
measurement for Qwen3-0.6B BF16 on the local AMD Strix Halo system. llama.cpp
used Vulkan; DynInfer used IREE's ROCm/HIP target. The comparison therefore
answers the practical engine-on-current-hardware question, but it is not a
controlled comparison of the same GPU backend.

The observed prompt-processing throughput was 504.68 tokens/s at 4K,
412.61 tokens/s at 16K, and 325.20 tokens/s at 32K. Generation of 128 tokens
measured 127.92 tokens/s with an empty KV cache, falling to 87.40 tokens/s at
4K context, 48.64 tokens/s at 16K context, and 30.41 tokens/s at 32K context.

DynInfer prefill was faster at all three tested sizes: 826.30, 536.84, and
362.82 tokens/s. Its decode path was substantially slower: 17.17, 7.65, and
4.32 tokens/s after 4K, 16K, and 32K prefill respectively. The result points
to single-token decode, especially its long-context attention path, as the
highest-value performance work.

## Date and hardware

- Date: 2026-08-14
- Host: `halo`, Linux x86-64, kernel `7.1.5-arch1-2`
- Processor: AMD Ryzen AI MAX+ 395 with Radeon 8060S
- GPU: AMD Radeon 8060S Graphics, Strix Halo (`gfx1151`)
- Memory: 124 GiB unified system memory
- llama.cpp package: `llama.cpp-vulkan b10216-1`
- llama.cpp build: build 10216, commit `06be260`
- Backend selected for measurements: Vulkan, device `Vulkan0`
- Device capabilities reported by llama.cpp: FP16 enabled, native BF16 not
  reported, warp size 64, integer dot product and KHR cooperative matrices
  available

The prefill run began at a system load average of 0.37, 0.26, 0.26, with
110 GiB of memory available. The context-depth decode run began at a load
average of 0.30, 0.30, 0.23.

## Model

- Hugging Face repository:
  [`unsloth/Qwen3-0.6B-GGUF`](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF)
- Artifact: `Qwen3-0.6B-BF16.gguf`
- llama.cpp model description: `qwen3 0.6B BF16`
- Parameters: 596,049,920
- Model payload reported by llama-bench: 1,192,230,912 bytes (1.11 GiB)
- Repository revision: `50968a4468ef4233ed78cd7c3de230dd1d61a56b`
- SHA-256:
  `f9c9f1d3c1e21755b82d4e165f88dbbbd4355646d632fb5d6cef7c66ed4ee04e`
- Cached path:
  `/home/goniz/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-BF16.gguf`

Only the requested BF16 artifact and several previously cached files from the
repository were present locally. `hf cache verify` found no mismatch among the
five cached files and reported the expected warning that the remaining 27
remote files were not downloaded.

## Model acquisition and inspection commands

The cache was checked first, the repository siblings were inspected to resolve
the exact filename, and only the BF16 GGUF was downloaded:

```bash
hf models info unsloth/Qwen3-0.6B-GGUF --expand siblings,sha
hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-BF16.gguf --dry-run
hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-BF16.gguf
hf cache verify unsloth/Qwen3-0.6B-GGUF
sha256sum /home/goniz/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-BF16.gguf
```

Backend discovery used:

```bash
/usr/bin/llama-bench --list-devices
```

llama.cpp reported `Vulkan0: AMD Radeon 8060S Graphics`. A system HIP build was
not installed, while a separate locally modified ROCmFPX tree did contain HIP
support. The packaged Vulkan binary was selected to keep this baseline tied to
a clean, identifiable llama.cpp package and commit.

## Common benchmark parameters

| Parameter | Value |
|---|---:|
| Backend | Vulkan |
| Device | `Vulkan0` |
| Requested GPU layers | 99 (full model offload) |
| Split mode | layer |
| Main GPU | 0 |
| Batch size | 2048 |
| Micro-batch size | 512 |
| CPU threads | 16 |
| K cache | F16, GPU-offloaded |
| V cache | F16, GPU-offloaded |
| Flash Attention | auto (`-1` in JSON output) |
| Model loading | mmap |
| Repetitions | 5 measured runs per test |
| Warmup | enabled, llama-bench default |
| Output | JSON |

Tokenization and sampling are excluded from llama-bench timing. Throughput is
the mean of the five per-repetition token rates; `±` is their sample standard
deviation. Total latency is the mean timed duration.

## Prompt-processing and depth-zero generation

Command:

```bash
MODEL=/home/goniz/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-BF16.gguf

/usr/bin/llama-bench \
  -m "$MODEL" \
  -dev Vulkan0 \
  -ngl 99 \
  -p 4096,16384,32768 \
  -n 128 \
  -r 5 \
  --progress \
  -o json
```

With separate `-p` and `-n` arguments, llama-bench emits three independent
prompt-processing tests and one independent generation test. TG128 here starts
at depth zero; it is not generation following any of the measured prompts.

| Test | Context depth | Throughput (tokens/s) | Total latency | Raw samples (tokens/s) |
|---|---:|---:|---:|---|
| PP4096 | 0 | 504.68 ± 0.23 | 8.116 s | 505.005, 504.572, 504.753, 504.687, 504.397 |
| PP16384 | 0 | 412.61 ± 0.21 | 39.708 s | 412.910, 412.746, 412.515, 412.376, 412.512 |
| PP32768 | 0 | 325.20 ± 0.33 | 100.763 s | 325.285, 324.922, 325.326, 325.642, 324.826 |
| TG128 | 0 | 127.92 ± 3.76 | 1.001 s | 123.696, 123.917, 131.010, 130.565, 130.403 |

Prompt-processing throughput decreases by about 35.6% between the 4K and 32K
tests as attention work grows with context length.

## Context-dependent decode

Command:

```bash
MODEL=/home/goniz/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-BF16.gguf

/usr/bin/llama-bench \
  -m "$MODEL" \
  -dev Vulkan0 \
  -ngl 99 \
  -p 0 \
  -n 128 \
  -d 4096,16384,32768 \
  -r 5 \
  --progress \
  -o json
```

`-d` asks llama-bench to prefill the KV cache to the requested depth before
timing TG128. The depth prefill is outside the timed region. llama-bench caches
and restores that context state for the five decode repetitions.

| Test | KV-cache depth | Throughput (tokens/s) | Mean per-token latency | Total TG128 latency | Raw samples (tokens/s) |
|---|---:|---:|---:|---:|---|
| TG128 | 0 | 127.92 ± 3.76 | 7.82 ms | 1.001 s | 123.696, 123.917, 131.010, 130.565, 130.403 |
| TG128 @ d4096 | 4,096 | 87.40 ± 0.47 | 11.44 ms | 1.465 s | 87.020, 86.885, 87.541, 87.475, 88.056 |
| TG128 @ d16384 | 16,384 | 48.64 ± 0.12 | 20.56 ms | 2.632 s | 48.821, 48.510, 48.679, 48.558, 48.640 |
| TG128 @ d32768 | 32,768 | 30.41 ± 0.07 | 32.89 ms | 4.209 s | 30.449, 30.417, 30.489, 30.321, 30.365 |

Decode throughput at 32K is about 23.8% of the depth-zero result. This is the
more useful decode baseline for long-context comparisons: the earlier TG128
row measures model and backend decode overhead with no substantial KV history,
while the depth-qualified rows include attention over the existing cache.

## Generation coherency check

A short deterministic generation was run after the performance sweep:

```bash
MODEL=/home/goniz/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-BF16.gguf

/usr/bin/llama-cli \
  -m "$MODEL" \
  -dev Vulkan0 \
  -ngl 99 \
  -p 'Reply with exactly this sentence: The benchmark completed successfully.' \
  -n 24 \
  -st \
  -rea off \
  --temp 0 \
  --no-display-prompt \
  --no-show-timings \
  --simple-io
```

The model responded exactly:

```text
The benchmark completed successfully.
```

This is a basic load/generate coherency check only; it is not a quality or
numerical-parity evaluation.

## DynInfer comparison

### DynInfer configuration

The exact same GGUF file was passed unmodified to DynInfer through
`--checkpoint`. Bazel built and launched `//crates/dyninfer-cli:dyninfer`.
Compilation happened before the timed loops and is excluded from the reported
latencies.

| Parameter | Value |
|---|---:|
| Execution target | `rocm` (IREE HIP, detected `gfx1151`) |
| IREE compiler | `3.11.0rc20260316`, revision `e4a3b0405d7d23554da26403658d0e8c3c5ecf25` |
| Model | Exact `Qwen3-0.6B-BF16.gguf` above |
| Prefill sizes | 4,096; 16,384; 32,768 tokens |
| Decode size | 128 tokens immediately after each prefill |
| Prefill chunk | 512 tokens for the paged executable |
| KV cache | Paged, K=F16 and V=F16 for the 4K/16K/32K runs |
| Warmup | 1 iteration |
| Measured iterations | 5 |
| Fill token | token ID 1 |
| Output | JSON |

The commands used for the three context sizes were equivalent to:

```bash
BAZELISK=/home/goniz/Work/ollama-strix-halo/rust_bazel/bazelisk
MODEL_SNAP=/home/goniz/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b
MODEL="$MODEL_SNAP/Qwen3-0.6B-BF16.gguf"

for PREFILL in 4k 16k 32k; do
  "$BAZELISK" run //crates/dyninfer-cli:dyninfer -- perf \
    --model-dir "$MODEL_SNAP" \
    --checkpoint "$MODEL" \
    --target rocm \
    --prefill "$PREFILL" \
    --tg 128 \
    --warmup 1 \
    --iters 5 \
    --json
done
```

Each size was run as a separate invocation so that it received an executable
and KV allocation specialized for `prefill + tg`. DynInfer resets the session
between iterations, measures the entire prefill, and then measures TG128 from
the resulting context. Its reported throughput is token count divided by mean
elapsed time across the five iterations.

### Prefill comparison

| Prompt | llama.cpp Vulkan | DynInfer HIP | DynInfer / llama.cpp | Difference | llama.cpp latency | DynInfer latency |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 504.68 tok/s | 826.30 tok/s | 1.64x | +63.7% | 8.116 s | 4.957 s |
| 16,384 | 412.61 tok/s | 536.84 tok/s | 1.30x | +30.1% | 39.708 s | 30.519 s |
| 32,768 | 325.20 tok/s | 362.82 tok/s | 1.12x | +11.6% | 100.763 s | 90.315 s |

DynInfer is already ahead of this llama.cpp baseline for prefill, so prefill is
not the first optimization target. The lead narrows as context grows, however:
DynInfer throughput falls 56.1% from 4K to 32K, compared with 35.6% for
llama.cpp. Long-context prefill attention and temporary-memory scaling remain
secondary opportunities.

### Decode comparison

The context-qualified rows are the most comparable decode measurements.
llama.cpp's `-d` cache preparation is untimed; DynInfer's decode timer begins
after its separately timed prefill. Both then time 128 sequential generated
tokens.

| Starting context | llama.cpp Vulkan | DynInfer HIP | DynInfer / llama.cpp | Relative deficit | llama.cpp ms/token | DynInfer ms/token |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 87.40 tok/s | 17.17 tok/s | 19.6% | 5.09x slower | 11.44 | 58.26 |
| 16,384 | 48.64 tok/s | 7.65 tok/s | 15.7% | 6.36x slower | 20.56 | 130.74 |
| 32,768 | 30.41 tok/s | 4.32 tok/s | 14.2% | 7.03x slower | 32.89 | 231.22 |

DynInfer decode throughput drops 74.8% from 4K to 32K, versus 65.2% for
llama.cpp. The widening relative gap indicates that both fixed one-token work
and context-dependent attention need improvement.

For a closest available check against llama.cpp's independent depth-zero
TG128, DynInfer was also run with one prefill token. This uses the small static
KV path (F32 K/V), rather than the paged F16 path used above, so it is a useful
diagnostic rather than a strict like-for-like row.

```bash
"$BAZELISK" run //crates/dyninfer-cli:dyninfer -- perf \
  --model-dir "$MODEL_SNAP" \
  --checkpoint "$MODEL" \
  --target rocm \
  --prefill 1 \
  --tg 128 \
  --warmup 1 \
  --iters 5 \
  --json
```

| Decode case | llama.cpp | DynInfer | Relative deficit | DynInfer effective weight bandwidth | DynInfer GPU busy |
|---|---:|---:|---:|---:|---:|
| TG128 near depth zero | 127.92 tok/s | 45.27 tok/s | 2.83x slower | 68.06 GB/s | 87.2% |
| TG128 after 4K | 87.40 tok/s | 17.17 tok/s | 5.09x slower | 25.81 GB/s | 95.4% |
| TG128 after 16K | 48.64 tok/s | 7.65 tok/s | 6.36x slower | 11.50 GB/s | 97.4% |
| TG128 after 32K | 30.41 tok/s | 4.32 tok/s | 7.03x slower | 6.50 GB/s | 97.9% |

The near-depth-zero result exposes a roughly 2.8x fixed decode gap before
long-context attention becomes dominant. At long context, GPU busy remains
approximately 95–98%, making CPU scheduling an unlikely primary explanation.
The effective-weight-bandwidth metric declines as attention consumes more of
each token's elapsed time; it should not be interpreted as a direct DRAM
bandwidth counter.

### DynInfer memory observations

These are DynInfer telemetry values only; the installed llama-bench output did
not expose matching memory counters, so they are not an engine comparison.

| Prompt | Paged KV allocated | IREE device peak | Peak GTT | Prefill GPU busy |
|---:|---:|---:|---:|---:|
| 4K | 0.50 GB | 4.80 GB | 5.71 GB | 97.1% |
| 16K | 1.91 GB | 7.79 GB | 10.03 GB | 97.9% |
| 32K | 3.79 GB | 11.50 GB | 15.81 GB | 97.7% |

The paged KV figures include allocation overhead and are slightly above their
reported logical capacities of 0.48, 1.89, and 3.77 GB. The rapidly growing
IREE peak indicates material temporary-buffer pressure at long context even
though the 32K run fits comfortably on this 128 GB unified-memory machine.

## Performance priorities

The following ordering combines the benchmark shape with the current lowering
code. It identifies profiling targets, not proven root causes.

1. **Specialize one-token GQA decode attention.** Qwen3-0.6B has head dimension
   128, so the current paged HIP decode selects `iree_linalg_ext.attention` over
   the full padded KV capacity and emits a full causal-mask tensor for every
   token. The repository already contains an experimental page-wise online
   attention emitter, but its comment records a gfx1151 VectorDistribute issue
   and says production HIP uses the generic IREE attention path. A dedicated
   `s=1` GQA kernel or a corrected generated lowering that streams paged F16 K/V,
   applies the length bound internally, and avoids materializing the full mask
   is the highest-leverage long-context experiment.

2. **Profile and tune the fixed single-token path.** The 2.83x near-zero-context
   deficit cannot be explained by long KV scans. Collect per-dispatch GPU
   timestamps for one token and rank the Q/K/V/O projections, MLP GEMV kernels,
   RMSNorm/RoPE, vocabulary projection, and dispatch gaps. Prioritize single-row
   projection/GEMV lowering and graph-level dispatch fusion based on the trace.

3. **Fuse or eliminate KV page update transformations.** The paged lowering
   extracts and copies the current page, transposes it, writes the new K/V, then
   inserts the page into the tied packed cache before attention. Confirm buffer
   aliasing in the compiled executable and measure these helpers separately.
   Folding the write into the decode-attention kernel would remove a likely
   per-layer fixed cost and reduce cache traffic.

4. **Reduce long-context temporary memory after decode is under control.** At
   32K, IREE device peak is 11.50 GB versus 3.79 GB of paged KV allocation.
   Inspect bufferization/liveness for full masks and attention intermediates.
   This is important for concurrency and larger models, but the current prefill
   result makes it lower priority than decode throughput.

5. **Keep prefill as a regression guard.** DynInfer is 12–64% faster in these
   tests. Optimization work should preserve that advantage while improving its
   steeper 4K-to-32K scaling.

This direction matches `docs/plans/spec.md`: prefill and decode are separate
specialization decisions, the target kernel matrix calls for a generated
custom decode-attention lowering, and Milestone 4 explicitly lists specialized
decode attention. It also follows `docs/plans/schema-and-target-specialization.md`
by treating prefill and decode as separate operation-shape selections and by
tracking latency, effective weight bandwidth, and temporary memory.

Suggested performance acceptance gates for the next iteration are:

- preserve generation coherency and numerical checks before accepting any
  kernel change;
- report TG128 at near-zero, 4K, 16K, and 32K depth, not only one context;
- report prefill at 4K, 16K, and 32K to catch regressions;
- add per-kernel GPU timing and peak temporary memory;
- initially target at least 2x DynInfer decode speedup at every tested depth,
  then drive toward within 20% of the llama.cpp baseline: 69.9, 38.9, and
  24.3 tokens/s at 4K, 16K, and 32K.

## DynInfer generation coherency

After the performance runs, the exact GGUF was loaded again through the
DynInfer ROCm target and decoded greedily using the official Qwen tokenizer
snapshot:

```bash
TOKENIZER_SNAP=/home/goniz/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca

"$BAZELISK" run //crates/dyninfer-cli:dyninfer -- generate \
  --model-dir "$TOKENIZER_SNAP" \
  --checkpoint "$MODEL" \
  --target rocm \
  --prompt 'The capital of France is' \
  --max-new-tokens 12 \
  --repetition-penalty 1.0 \
  --raw-prompt
```

Observed continuation:

```text
The capital of France is Paris. The capital of Italy is Rome. The capital of
```

This passed the repository's required generate-coherency validation. It is a
functional smoke test, not numerical parity with llama.cpp.

## Comparison limitations

- The engines used different backends: llama.cpp Vulkan and DynInfer ROCm/HIP.
  The figures are valid current-engine baselines, not an isolated engine-code
  comparison.
- Both engines used five measured repetitions, but llama-bench reports the mean
  of per-run rates while DynInfer computes tokens divided by mean latency. The
  difference is negligible for stable samples but is methodologically distinct.
- DynInfer prefill and its following TG128 share a purpose-sized session;
  llama-bench's `-d` path creates and restores a prefilled context. Both exclude
  the context-fill time from decode, but their state-management mechanics differ.
- Only one model, one host, one date, and one run set were measured. Treat small
  differences as provisional; the multi-fold decode gaps are large enough to
  guide prioritization despite this limitation.
