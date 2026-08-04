#!/usr/bin/env bash
# Thin Bazel sh_binary entrypoint for the pinned `iree-compile` wheel.
#
# Why this exists: `rules_rs` / `bazel run` tests need a stable label
# (`//bazel/iree:iree_compile_bin`) whose runfiles contain the platform-selected
# IREE compiler archive. The real binary lives under a repo-select / bzlmod
# prefix that differs between:
#   - directory runfiles (`RUNFILES_DIR/.../iree_compiler_linux_$arch/...`)
#   - bzlmod canonical names (`+http_archive+iree_compiler_linux_$arch/...`)
#   - manifest-only layouts (`RUNFILES_MANIFEST_FILE`)
# This script locates that binary and exec's it with the caller's argv so Rust
# crates can shell out without hard-coding runfiles paths.
set -euo pipefail

RUNFILES_DIR="${RUNFILES_DIR:-}"
if [[ -z "${RUNFILES_DIR}" && -n "${RUNFILES_MANIFEST_FILE:-}" ]]; then
  # Manifest mode: look up by key.
  for key in \
      iree_compiler_linux_x86_64/iree/compiler/_mlir_libs/iree-compile \
      iree_compiler_linux_aarch64/iree/compiler/_mlir_libs/iree-compile; do
    target="$(
      awk -v key="${key}" \
        '$1 == key || $1 == "+http_archive+" key {
          print substr($0, index($0, " ") + 1)
          exit
        }' \
        "${RUNFILES_MANIFEST_FILE}"
    )"
    if [[ -n "${target}" ]]; then
      exec "${target}" "$@"
    fi
  done
  echo "iree-compile not found in runfiles manifest" >&2
  exit 1
fi

if [[ -z "${RUNFILES_DIR}" ]]; then
  # Fallback for `bazel run` when only the .runfiles sibling exists.
  self="${BASH_SOURCE[0]}"
  if [[ -d "${self}.runfiles" ]]; then
    RUNFILES_DIR="${self}.runfiles"
  elif [[ -d "$(dirname "${self}")/../" ]]; then
    # Common layout: .../<bin>.runfiles/_main/...
    candidate="$(cd "$(dirname "${self}")" && pwd)"
    if [[ -d "${candidate}.runfiles" ]]; then
      RUNFILES_DIR="${candidate}.runfiles"
    fi
  fi
fi

if [[ -n "${RUNFILES_DIR}" ]]; then
  for arch in x86_64 aarch64; do
    for prefix in \
        "iree_compiler_linux_${arch}" \
        "+http_archive+iree_compiler_linux_${arch}"; do
      candidate="${RUNFILES_DIR}/${prefix}/iree/compiler/_mlir_libs/iree-compile"
      if [[ -x "${candidate}" ]]; then
        exec "${candidate}" "$@"
      fi
    done
  done
fi

echo "iree-compile not found in runfiles (RUNFILES_DIR=${RUNFILES_DIR:-unset})" >&2
exit 1
