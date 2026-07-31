#!/usr/bin/env bash
set -euo pipefail

# Resolve the real iree-compile from Bazel runfiles (repo-select layout).
RUNFILES_DIR="${RUNFILES_DIR:-}"
if [[ -z "${RUNFILES_DIR}" && -n "${RUNFILES_MANIFEST_FILE:-}" ]]; then
  # Manifest mode: look up by key.
  for key in \
      iree_compiler_linux_x86_64/iree/compiler/_mlir_libs/iree-compile \
      iree_compiler_linux_aarch64/iree/compiler/_mlir_libs/iree-compile; do
    line="$(grep -F " ${key}" "${RUNFILES_MANIFEST_FILE}" | head -1 || true)"
    if [[ -n "${line}" ]]; then
      exec "${line##* }" "$@"
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
