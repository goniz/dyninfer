#!/usr/bin/env bash
set -euo pipefail

runfiles_root() {
  if [[ -n "${RUNFILES_DIR:-}" && -d "${RUNFILES_DIR}" ]]; then
    echo "${RUNFILES_DIR}"
    return
  fi
  if [[ -n "${TEST_SRCDIR:-}" && -d "${TEST_SRCDIR}" ]]; then
    echo "${TEST_SRCDIR}"
    return
  fi
  return 1
}

# Bzlmod stores repos as +http_archive+<name>; also accept the apparent name.
find_in_repo() {
  local root="$1"
  local repo="$2"
  local rel="$3"
  for prefix in "${repo}" "+http_archive+${repo}"; do
    local candidate="${root}/${prefix}/${rel}"
    if [[ -x "${candidate}" || -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

ROOT="$(runfiles_root)" || {
  echo "no runfiles root" >&2
  exit 1
}

compile=""
run=""
for arch in x86_64 aarch64; do
  if c="$(find_in_repo "${ROOT}" "iree_compiler_linux_${arch}" "iree/compiler/_mlir_libs/iree-compile")"; then
    compile="$c"
  fi
  if r="$(find_in_repo "${ROOT}" "iree_runtime_linux_${arch}" "iree/_runtime_libs/iree-run-module")"; then
    run="$r"
  fi
done

if [[ -z "${compile}" || -z "${run}" ]]; then
  echo "missing IREE tools in runfiles under ${ROOT}" >&2
  ls -la "${ROOT}" >&2 || true
  exit 1
fi

"${compile}" --version | head -3
"${run}" --help >/dev/null
echo "IREE Bazel tools smoke OK"
