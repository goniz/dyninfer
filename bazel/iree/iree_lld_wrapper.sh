#!/usr/bin/env bash
# Run the pinned wheel's linker from its original directory so its $ORIGIN
# RPATH resolves libIREECompiler.so. This exposes a stable `iree-lld` name to
# Bazel tests without relying on a bzlmod-specific external-repository path.
set -euo pipefail

RUNFILES_DIR="${RUNFILES_DIR:-}"
if [[ -z "${RUNFILES_DIR}" && -n "${RUNFILES_MANIFEST_FILE:-}" ]]; then
  for key in \
      iree_compiler_linux_x86_64/iree/compiler/_mlir_libs/iree-lld \
      iree_compiler_linux_aarch64/iree/compiler/_mlir_libs/iree-lld; do
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
  echo "iree-lld not found in runfiles manifest" >&2
  exit 1
fi

if [[ -z "${RUNFILES_DIR}" ]]; then
  self="${BASH_SOURCE[0]}"
  if [[ -d "${self}.runfiles" ]]; then
    RUNFILES_DIR="${self}.runfiles"
  else
    # IREE runs the linker with only LLD_VERSION in its child environment.
    # In a Bazel test, this script is reached through
    # <runfiles>/_main/bazel/iree/iree-lld, so recover the runfiles root from
    # that stable location.
    candidate="$(cd "$(dirname "${self}")/../../.." && pwd)"
    if [[ -d "${candidate}" ]]; then
      RUNFILES_DIR="${candidate}"
    fi
  fi
fi

if [[ -n "${RUNFILES_DIR}" ]]; then
  for arch in x86_64 aarch64; do
    for prefix in \
        "iree_compiler_linux_${arch}" \
        "+http_archive+iree_compiler_linux_${arch}"; do
      candidate="${RUNFILES_DIR}/${prefix}/iree/compiler/_mlir_libs/iree-lld"
      if [[ -x "${candidate}" ]]; then
        exec "${candidate}" "$@"
      fi
    done
  done
fi

echo "iree-lld not found in runfiles (RUNFILES_DIR=${RUNFILES_DIR:-unset})" >&2
exit 1
