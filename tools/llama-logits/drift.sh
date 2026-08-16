#!/usr/bin/env bash
set -euo pipefail

find_runfile() {
  local relative="$1"
  local root="${RUNFILES_DIR:-${TEST_SRCDIR:-}}"
  local manifest="${RUNFILES_MANIFEST_FILE:-}"

  if [[ -z "${root}" && -d "${BASH_SOURCE[0]}.runfiles" ]]; then
    root="${BASH_SOURCE[0]}.runfiles"
  fi
  if [[ -z "${manifest}" && -f "${BASH_SOURCE[0]}.runfiles_manifest" ]]; then
    manifest="${BASH_SOURCE[0]}.runfiles_manifest"
  fi

  if [[ -n "${root}" ]]; then
    local prefix
    for prefix in _main dyninfer; do
      local candidate="${root}/${prefix}/${relative}"
      if [[ -x "${candidate}" ]]; then
        echo "${candidate}"
        return 0
      fi
    done
  fi

  if [[ -n "${manifest}" ]]; then
    local key
    for key in "_main/${relative}" "dyninfer/${relative}"; do
      local candidate
      candidate="$(
        awk -v key="${key}" \
          '$1 == key { print substr($0, index($0, " ") + 1); exit }' \
          "${manifest}"
      )"
      if [[ -n "${candidate}" && -x "${candidate}" ]]; then
        echo "${candidate}"
        return 0
      fi
    done
  fi

  return 1
}

dyninfer="$(find_runfile "crates/dyninfer-cli/dyninfer")" || {
  echo "dyninfer CLI not found in Bazel runfiles" >&2
  exit 1
}
llama_runner="$(find_runfile "tools/llama-logits/dyninfer-llama-logits")" || {
  echo "dyninfer-llama-logits not found in Bazel runfiles" >&2
  exit 1
}

has_decode_steps=false
has_decode_tokens=false
has_generate_coherency=false
has_llama_runner=false
for argument in "$@"; do
  case "${argument}" in
    --decode-steps | --decode-steps=*) has_decode_steps=true ;;
    --decode-tokens | --decode-tokens=*) has_decode_tokens=true ;;
    --generate-coherency) has_generate_coherency=true ;;
    --llama-runner | --llama-runner=*) has_llama_runner=true ;;
  esac
done

defaults=()
if [[ "${has_llama_runner}" == false ]]; then
  defaults+=(--llama-runner "${llama_runner}")
fi
if [[ "${has_decode_steps}" == false && "${has_decode_tokens}" == false ]]; then
  defaults+=(--decode-steps 4)
fi
if [[ "${has_decode_tokens}" == false && "${has_generate_coherency}" == false ]]; then
  defaults+=(--generate-coherency)
fi

exec "${dyninfer}" logits drift "${defaults[@]}" "$@"
