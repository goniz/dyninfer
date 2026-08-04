#!/usr/bin/env bash
set -euo pipefail

compiler="$1"
input="$2"
workdir="${TEST_TMPDIR:-/tmp}/paged-kv-abi"
mkdir -p "${workdir}"

expect_typed_list_rejection() {
  local target="$1"
  shift
  local log="${workdir}/${target}.log"
  if "${compiler}" "${input}" "$@" -o "${workdir}/${target}.vmfb" \
      >"${log}" 2>&1; then
    echo "${target}: unexpectedly compiled !util.list<tensor<*xf32>>" >&2
    return 1
  fi

  local diagnostic
  diagnostic="$(<"${log}")"
  if [[ "${diagnostic}" != *"failed to legalize operation 'util.func' that was explicitly marked illegal"* ]] ||
     [[ "${diagnostic}" != *'iree.abi.declaration = "sync func @sum_page_heads(%input0: !util.list<tensor<*xf32>>) -> (%output0: f32)"'* ]]; then
    echo "${target}: compiler failed for an unexpected reason" >&2
    echo "${diagnostic}" >&2
    return 1
  fi
  echo "${target}: confirmed typed tensor-list ABI rejection"
}

expect_typed_list_rejection local-task \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host

expect_typed_list_rejection hip \
  --iree-hal-target-device=hip \
  --iree-rocm-target="${DYNINFER_TEST_ROCM_TARGET:-gfx1151}"
