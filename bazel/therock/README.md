# TheRock ROCm SDK

`MODULE.bazel` pins the TheRock 7.14 `gfx1151` distribution used for Strix
Halo. Bazel downloads and verifies the archive; no system ROCm installation is
required.

The upstream stable source release is
[`therock-7.14`](https://github.com/ROCm/TheRock/releases/tag/therock-7.14).
TheRock does not attach prebuilt SDKs to that GitHub release; its
[`RELEASES.md`](https://github.com/ROCm/TheRock/blob/therock-7.14/RELEASES.md)
directs tarball users to AMD's CD artifact index. The pinned
`7.14.0a20260624` archive is the newest `gfx1151` artifact in that 7.14 release
line and is fixed by SHA-256 rather than tracking a moving nightly URL.

The archive is intentionally exposed as the public `//bazel/therock:sdk`
target. Rust consumers use `dyninfer-rocm` to locate that target in Bazel
runfiles and configure:

- `lib/llvm/bin/ld.lld` for IREE's embedded LLVM CPU linker;
- `lib/llvm/amdgcn/bitcode` for HIP compilation;
- `lib/libamdhip64.so` and sibling libraries for HIP execution; and
- `bin`, `.kpack`, and the remaining SDK files needed by ROCm at runtime.

For Cargo-only development, `DYNINFER_ROCM_HOME`, `ROCM_HOME`, or `ROCM_PATH`
may point at an extracted TheRock distribution. There is no implicit
`/opt/rocm` fallback.
