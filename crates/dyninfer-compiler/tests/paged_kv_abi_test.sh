#!/usr/bin/env bash
set -euo pipefail

compiler="$1"
input="$2"
workdir="${TEST_TMPDIR:-/tmp}/paged-kv-abi"
mkdir -p "${workdir}"

# Extract one public func into a standalone module for compile checks.
extract_func() {
  local name="$1"
  local out="$2"
  awk -v name="${name}" '
    BEGIN { printing = 0 }
    $0 ~ "util.func public @" name "\\(" { printing = 1; print "module @paged_kv_abi {" }
    printing { print }
    printing && $0 ~ /^  }$/ { print "}"; exit }
  ' "${input}" >"${out}"
}

expect_typed_list_rejection() {
  local target="$1"
  shift
  local mlir="${workdir}/tensor_list.mlir"
  local log="${workdir}/${target}_tensor_list.log"
  extract_func sum_page_heads_tensor_list "${mlir}"
  if "${compiler}" "${mlir}" "$@" -o "${workdir}/${target}_tensor_list.vmfb" \
      >"${log}" 2>&1; then
    echo "${target}: unexpectedly compiled !util.list<tensor<*xf32>>" >&2
    return 1
  fi

  local diagnostic
  diagnostic="$(<"${log}")"
  if [[ "${diagnostic}" != *"failed to legalize operation 'util.func' that was explicitly marked illegal"* ]] ||
     [[ "${diagnostic}" != *'iree.abi.declaration = "sync func @sum_page_heads_tensor_list(%input0: !util.list<tensor<*xf32>>) -> (%output0: f32)"'* ]]; then
    echo "${target}: compiler failed for an unexpected reason" >&2
    echo "${diagnostic}" >&2
    return 1
  fi
  echo "${target}: confirmed typed tensor-list ABI rejection"
}

expect_buffer_view_list_ok() {
  local target="$1"
  shift
  local mlir="${workdir}/bv_list.mlir"
  local log="${workdir}/${target}_bv_list.log"
  extract_func touch_pages "${mlir}"
  if ! "${compiler}" "${mlir}" "$@" -o "${workdir}/${target}_bv_list.vmfb" \
      >"${log}" 2>&1; then
    echo "${target}: failed to compile !util.list<!hal.buffer_view>" >&2
    cat "${log}" >&2
    return 1
  fi
  echo "${target}: confirmed buffer-view list ABI compiles"
}

expect_typed_list_rejection local-task \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host

expect_typed_list_rejection hip \
  --iree-hal-target-device=hip \
  --iree-rocm-target="${DYNINFER_TEST_ROCM_TARGET:-gfx1151}"

expect_buffer_view_list_ok local-task \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host

expect_buffer_view_list_ok hip \
  --iree-hal-target-device=hip \
  --iree-rocm-target="${DYNINFER_TEST_ROCM_TARGET:-gfx1151}"
