#!/usr/bin/env bash
# Optional cargo-only fallback: install pinned IREE wheels into third_party/iree-venv.
#
# Prefer Bazel — IREE is fetched hermetically via MODULE.bazel:
#   bazel test //bazel/iree:iree_tools_smoke
#   bazel run //crates/dyninfer-cli:dyninfer -- smoke
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/third_party/IREE_VERSION"
VENV="${ROOT}/third_party/iree-venv"

echo "NOTE: Prefer Bazel IREE targets (//bazel/iree:tools). This venv is only for cargo-only workflows." >&2

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi

"${VENV}/bin/pip" install -U pip
"${VENV}/bin/pip" install \
  "iree-base-compiler==${IREE_PIP_VERSION}" \
  "iree-base-runtime==${IREE_PIP_VERSION}"

echo "IREE tools ready (cargo fallback):"
echo "  ${VENV}/bin/iree-compile"
echo "  ${VENV}/bin/iree-run-module"
"${VENV}/bin/iree-compile" --version | head -5 || true
