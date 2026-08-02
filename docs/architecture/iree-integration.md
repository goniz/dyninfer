# IREE integration

## Pin

See [`third_party/IREE_VERSION`](../../third_party/IREE_VERSION) and
[`MODULE.bazel`](../../MODULE.bazel).

IREE is **fetched by Bazel** as pinned manylinux wheels
(`iree-base-compiler` / `iree-base-runtime` 3.11.0) and exposed under
[`bazel/iree`](../../bazel/iree):

| Target | Role |
|---|---|
| `//bazel/iree:libIREECompiler` | Shared library for in-process embedding API |
| `//bazel/iree:iree-compile` | CLI fallback |
| `//bazel/iree:iree-run-module` | Tool-backed runtime invoke |
| `//bazel/iree:tools` | Aggregate runfiles |

Vendored C headers: [`third_party/iree_compiler_c_api`](../../third_party/iree_compiler_c_api).
Bindings: bindgen via `rules_rs` (`//crates/iree-compiler-sys`).

Optional full source build notes: [`bazel/iree/SOURCE.md`](../../bazel/iree/SOURCE.md).

```bash
bazel test //crates/iree-compiler-sys:iree-compiler-sys_tests
bazel run //crates/dyninfer-cli:dyninfer -- smoke
```

## Compiler path

1. **Default:** in-process `libIREECompiler` (`ireeCompilerSession*` / `Invocation*` API).
2. **Fallback:** subprocess `iree-compile` if the SO cannot be loaded/linked.
3. **Override:** `CompileOptions.force_subprocess` or cargo-only venv via
   `scripts/bootstrap_iree.sh`.

## Parameters (Milestone 1)

Dense Llama VMFBs load weights via `#stream.parameter.named<"weights"::"...">`.
At invoke time the runtime passes:

```text
--parameters=weights=<checkpoint.safetensors>
```

Any exported function on a parameterized module (including `@add` smoke) needs
that flag when creating the VM context.

Dense MHA Llama shapes (including Maykeye TinyLLama-v0: `vocab=32000`,
`hidden=64`, 8 layers, seq=64 + RoPE) are emitted programmatically. GQA /
Meta-1B-scale models still fall back to the constant-logits bridge.

## Next steps

1. Grow high-level `ModelBuilder` / kernel helpers so architecture emitters stop
   hand-authoring large `linalg` asm blobs (melior bindings ≠ a zml-like DSL).
2. Optional bf16/fp16 compute dtype gated on target HAL (M1 uses f32).
3. Drop wheel `iree-run-module` once device discovery is fully in-process.
