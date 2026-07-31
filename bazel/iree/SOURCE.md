# Building IREE from source (optional)

Default builds use pinned manylinux wheels (`libIREECompiler.so` + tools) via
`MODULE.bazel`. That matches the ABI-stable embedding API and keeps CI light.

A full `@iree_core` source build is possible but heavy: IREE’s Bazel module
expects you (as the root module) to provide `llvm-raw` / StableHLO / etc., and
the `llvm-project` submodule alone is multi‑GB. Prefer wheels unless you need
to patch the compiler.

## Opt-in local checkout

```bash
git clone --recursive https://github.com/iree-org/iree.git third_party/iree
cd third_party/iree && git checkout e4a3b0405d7d23554da26403658d0e8c3c5ecf25
# submodule sync as required by that commit
```

Then follow upstream
[`BZLMOD_LLVM.md`](https://github.com/iree-org/iree/blob/main/build_tools/bazel/BZLMOD_LLVM.md)
to `local_path_override` `iree_core` and supply `llvm-raw`. Point
`//bazel/iree:libIREECompiler` at the source-built shared library when ready.

Until that wiring lands, in-process compilation uses the wheel SO through
`//bazel/iree:libIREECompiler` + bindgen (`//crates/iree-compiler-sys`).
