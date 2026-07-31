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

Implementation is tracking [`spec.md`](spec.md). Milestone 0/1 skeleton is in place:

- Full Cargo workspace (`dyninfer` CLI + 18 crates)
- SafeTensors + GGUF metadata indexing and convention decoding
- Llama architecture slots, binder, target discovery, artifact cache
- In-process IREE compile via bindgen + `libIREECompiler` (`//crates/iree-compiler-sys`)
- Tool-backed run via Bazel-fetched `iree-run-module` (`//bazel/iree:tools`)
- CLI: `smoke`, `checkpoint inspect`, `bind`, `compile`, `run`, `model install`, `cache`

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
