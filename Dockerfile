# syntax=docker/dockerfile:1
#
# dyninfer — Bazel build + CPU & HIP/ROCm inference container.
#
# Bazel fetches the pinned TheRock ROCm userspace for Strix Halo (gfx1151).
# Ubuntu 24.04 provides Python 3.12, matching the pinned cp312 IREE wheels in
# MODULE.bazel; the host supplies only the Linux amdgpu kernel driver/devices.
#
# Build:
#   docker build -t dyninfer .
#
# Run (HIP needs the KFD/render devices passed through):
#   docker run --rm -it \
#     --device=/dev/kfd --device=/dev/dri \
#     --group-add video --group-add "$(getent group render | cut -d: -f3)" \
#     --ipc=host --shm-size=1g \
#     -v "$HOME/.cache/huggingface/hub:/hf-cache" \
#     dyninfer
#
# Inside the container (models resolved from the HF cache, no downloads):
#   # CPU
#   bazel run //crates/dyninfer-cli:dyninfer -- generate \
#     --hf Qwen/Qwen3-0.6B --prompt "Hello" --max-new-tokens 16 --target cpu
#   # HIP/ROCm (device is probed via iree-run-module --dump_devices)
#   bazel run //crates/dyninfer-cli:dyninfer -- generate \
#     --hf Qwen/Qwen3-0.6B --prompt "Hello" --max-new-tokens 16 --target rocm

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Bazel + hermetic toolchain prerequisites:
# - git: MODULE.bazel git_override of iree_core (recursive submodules)
# - xz/zlib/zstd/tinfo: LLVM toolchain + Rust toolchain tarballs
# - build-essential: cc autoconfiguration and native linking
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates curl git patch pkg-config \
      unzip zip xz-utils \
      python3 libtinfo6 zlib1g libzstd1 \
    && rm -rf /var/lib/apt/lists/*

# Bazelisk; the repo's .bazelversion pins Bazel 9.1.1.
ARG TARGETARCH
RUN set -eux; \
    case "${TARGETARCH:-amd64}" in \
      amd64) BAZELISK_ARCH=amd64 ;; \
      arm64) BAZELISK_ARCH=arm64 ;; \
      *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /usr/local/bin/bazel \
      "https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-${BAZELISK_ARCH}"; \
    chmod +x /usr/local/bin/bazel; \
    bazel --version

WORKDIR /src
COPY . .

# Build the CLI. This compiles the in-process IREE runtime from @iree_core
# (HIP + local-task/local-sync drivers, see .bazelrc) and stages the pinned
# prebuilt iree-compile / iree-run-module wheels and TheRock SDK into runfiles.
#
# CPU `generate` works as-is; HIP `generate` additionally needs the ROCm
# devices at runtime (see docker run flags above). `libamdhip64.so` and its
# dependencies are loaded from the Bazel-pinned TheRock SDK.
RUN bazel build //crates/dyninfer-cli:dyninfer \
    && bazel test //bazel/iree:iree_tools_smoke \
    && bazel run //crates/dyninfer-cli:dyninfer -- --help >/dev/null

# `generate --hf` resolves models from a local Hugging Face Hub cache
# (never downloads). Mount the host cache at /hf-cache, or `pip install
# huggingface_hub` and `hf download ...` with HF_HUB_CACHE=/hf-cache.
ENV HF_HUB_CACHE=/hf-cache

CMD ["bash"]
