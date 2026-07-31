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
(~9MB BF16 Llama). Prefer the local Hugging Face Hub cache
(`HF_HUB_CACHE` / `~/.cache/huggingface/hub`); testdata is only a fallback.
Meta Llama 3.2-1B is gated/heavier (GQA) and left for a later milestone.

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
bazel run //crates/dyninfer-cli:dyninfer -- checkpoint inspect architectures/testdata/tiny-llama.safetensors
```

Cargo remains usable for crate-level iteration. For IREE under cargo only,
optionally run `./scripts/bootstrap_iree.sh` (venv fallback); prefer Bazel.

## CLI sketch

```bash
dyninfer checkpoint inspect model.gguf --json
dyninfer bind --architecture llama.decoder --checkpoint model.safetensors --output binding.json
dyninfer compile --architecture llama.decoder --checkpoint model.safetensors --target cpu --output model.bundle
dyninfer run --bundle model.bundle --checkpoint model.safetensors --prompt "Hello"
dyninfer cache list
```

## License

MIT — see [LICENSE](LICENSE).
