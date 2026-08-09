# IREE source build (bzlmod)

dyninfer depends on upstream [`iree_core`](https://github.com/iree-org/iree)
via **`bazel_dep` + `git_override`** (see root [`MODULE.bazel`](../../MODULE.bazel)).

| Pin | Value |
|---|---|
| Commit | `e4a3b0405d7d23554da26403658d0e8c3c5ecf25` (IREE 3.11.0) |
| Submodules | `recursive_init_submodules = True` |
| LLVM overlay | `archive_override` of `llvm-project-overlay` at submodule `66395ad94…` |
| Patches | [`patches/`](patches/) — third_party paths + flatcc `-I` for non-root |

## Build

```bash
bazel build //bazel/iree:runtime_cc
# == @iree_core//runtime/src/iree/runtime:runtime
```

HAL drivers enabled by default in [`.bazelrc`](../../.bazelrc):
`hip`, `local-sync`, `local-task`, `null`
(AMDGPU device-bitcode / CUDA host drivers off — override the
`enabled_drivers` flag if you need them).

## Bazel 9 notes

IREE 3.11.0 targets Bazel 7; dyninfer uses Bazel 9.1.1. Required flags
(already in `.bazelrc`):

- `--incompatible_autoload_externally=+@rules_cc`
- `--noincompatible_disallow_empty_glob`
- `--features=-layering_check`

## Patches (why)

1. **`iree_downstream_extension.patch`** — upstream
   `local_repository(path = "third_party/…")` is relative to the **root**
   workspace. With `git_override`, those trees live under `@iree_core`; the
   patch resolves them via `Label("@iree_core//:MODULE.bazel")`.
2. **`iree_flatcc_downstream.patch`** — flatcc used `-I runtime/src`, which
   only works when IREE is the root. Derive `-I` from the `.fbs` `$(location)`.

## Prebuilt wheels

`//bazel/iree:iree-compile` / `libIREECompiler` remain for compile; invoke path
uses `//third_party/iree_runtime_c_api` → `//bazel/iree:runtime_cc` (in-process).
`iree-run-module` is still used for `--dump_devices` discovery.

## ROCm SDK

The HIP userspace is the Bazel-pinned TheRock 7.14 `gfx1151` distribution in
[`//bazel/therock`](../therock/README.md), not a system `/opt/rocm` install.
`dyninfer-rocm` locates it in runfiles. The compiler uses its device bitcode and
`ld.lld`; CLI subprocesses receive its `PATH`/`LD_LIBRARY_PATH`; and the native
runtime loads its `libamdhip64.so` by absolute path before IREE creates a HIP
device.
