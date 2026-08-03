# Dynamic Inference Engine (`dyninfer`)

Checkpoint-specializing compiler and local inference runtime.

The engine accepts a model architecture package, an unmodified checkpoint
(SafeTensors / GGUF initially), and a target profile, then produces an IREE
VM FlatBuffer (`.vmfb`) that resolves parameters from the original checkpoint
or an optional derived parameter cache.

```text
Architecture IR
  + Checkpoint catalog and physical tensor encodings
  + Target and shape profile
  = Specialized IREE executable
```

## Status

Implementation is tracking [`spec.md`](spec.md). **Milestone 1 (dense Llama PoC) is in:**

- Full Cargo workspace (`dyninfer` CLI + 18 crates)
- SafeTensors + GGUF metadata indexing and convention decoding
- Llama architecture slots, binder, target discovery, artifact cache
- Real tiny-Llama MLIR (external IREE params from SafeTensors) + Rust reference e2e
- In-process IREE compile via bindgen + `libIREECompiler` (`//crates/iree-compiler-sys`)
- Tool-backed run via Bazel-fetched `iree-run-module` (`//bazel/iree:tools`)
- CLI: `smoke`, `checkpoint inspect`, `bind`, `compile`, `run`, `model install`, `cache`

Differential check: `bazel test //crates/dyninfer-runtime:dyninfer-runtime_tests`

### Real TinyStories Llama example

Uses ungated [Maykeye/TinyLLama-v0](https://huggingface.co/Maykeye/TinyLLama-v0)
(~9MB BF16 Llama) from the local Hugging Face Hub cache
(`HF_HUB_CACHE` / `~/.cache/huggingface/hub`).
### Qwen3-0.6B (GQA + Q/K norm)

Built-in model graphs live in [`crates/dyninfer-architecture`](crates/dyninfer-architecture)
(`models/llama.rs`, `models/qwen3.rs`, …). CLI defaults to `--architecture auto`
(from `config.json` `model_type`).

Qwen3-0.6B ([Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)): GQA 16/8,
`head_dim=128`, Q/K norm, tied embeddings, ByteLevel BPE, window 32.

```bash
hf download Qwen/Qwen3-0.6B
# or: mlx-community/Qwen3-0.6B-bf16

bazel run //crates/dyninfer-cli:dyninfer -- generate \
  --hf Qwen/Qwen3-0.6B \
  --prompt "Hello" \
  --max-new-tokens 16
```

### Qwen3-0.6B Q4_0 GGUF (device-fused qkernel)

Prefer a pure Q4_0 file. Good source:
[`ggml-org/Qwen3-0.6B-GGUF`](https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF)
(`Qwen3-0.6B-Q4_0.gguf`). Pair with tokenizer/config from `Qwen/Qwen3-0.6B`.

```bash
hf download ggml-org/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_0.gguf --local-dir /tmp/qwen3-q4
hf download Qwen/Qwen3-0.6B config.json tokenizer.json tokenizer_config.json \
  vocab.json merges.txt --local-dir /tmp/qwen3-q4

bazel run //crates/dyninfer-cli:dyninfer -- generate \
  --model-dir /tmp/qwen3-q4 \
  --checkpoint /tmp/qwen3-q4/Qwen3-0.6B-Q4_0.gguf \
  --architecture qwen3.decoder \
  --prefill-window 8 --max-kv 16 \
  --prompt "Hello" --max-new-tokens 8 --target cpu
```

(`unsloth/Qwen3-0.6B-GGUF` Q4_0 mixes Q4_1 / Q6_K and is not supported yet.)

### TinyStories Llama example

Uses ungated [Maykeye/TinyLLama-v0](https://huggingface.co/Maykeye/TinyLLama-v0)
(~9MB BF16 Llama) from the local Hugging Face Hub cache
(`HF_HUB_CACHE` / `~/.cache/huggingface/hub`).

```bash
# one-time: populate the Hub cache
hf download Maykeye/TinyLLama-v0

bazel run //crates/dyninfer-cli:dyninfer -- generate \
  --hf Maykeye/TinyLLama-v0 \
  --prompt "Once upon a time" \
  --max-new-tokens 48
```

## Build

Bazel is the primary build: it fetches pinned IREE tools via `MODULE.bazel`
(`//bazel/iree:tools`) and builds the Rust workspace with `rules_rs`.

```bash
bazel build //crates/dyninfer-cli:dyninfer
bazel test //crates/... //bazel/iree:iree_tools_smoke
bazel run //crates/dyninfer-cli:dyninfer -- smoke
# after: hf download Maykeye/TinyLLama-v0
bazel run //crates/dyninfer-cli:dyninfer -- generate --hf Maykeye/TinyLLama-v0 --prompt "Hello"
```

Cargo remains usable for crate-level iteration. For IREE under cargo only,
optionally run `./scripts/bootstrap_iree.sh` (venv fallback); prefer Bazel.

## CLI sketch

```bash
dyninfer checkpoint inspect model.gguf --json
dyninfer bind --checkpoint model.safetensors --output binding.json   # auto arch
dyninfer compile --checkpoint model.safetensors --target auto --output model.bundle
dyninfer smoke                        # --target auto probes IREE (cuda/hip ≻ vulkan ≻ cpu)
dyninfer smoke --target rocm          # force HIP; chip from probe default gfx1151 / DYNINFER_ROCM_TARGET
dyninfer generate --hf ORG/NAME --prompt "Hello"
dyninfer run --bundle model.bundle --checkpoint model.safetensors --prompt "Hello"
dyninfer cache list
```

## License

MIT — see [LICENSE](LICENSE).
