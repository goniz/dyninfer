"""Hermetic repository rule for official llama.cpp release archives."""

def _llama_release_repository_impl(repository_ctx):
    architecture = repository_ctx.os.arch
    if architecture in ["amd64", "x86_64"]:
        archive_arch = "x64"
        archive_sha256 = repository_ctx.attr.x86_64_sha256
    elif architecture in ["aarch64", "arm64"]:
        archive_arch = "arm64"
        archive_sha256 = repository_ctx.attr.arm64_sha256
    else:
        fail("llama.cpp release binaries are unavailable for host architecture %s" % architecture)

    version = repository_ctx.attr.version
    release_base = "https://github.com/ggml-org/llama.cpp/releases/download/%s" % version
    repository_ctx.download_and_extract(
        url = "%s/llama-%s-bin-ubuntu-vulkan-%s.tar.gz" % (release_base, version, archive_arch),
        sha256 = archive_sha256,
        stripPrefix = "llama-%s" % version,
    )

    # Binary release archives omit headers. Extract the same tagged source
    # archive into a subdirectory and expose only its public headers in BUILD.
    repository_ctx.download_and_extract(
        url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/%s.tar.gz" % version,
        output = "source",
        sha256 = repository_ctx.attr.source_sha256,
        stripPrefix = "llama.cpp-%s" % version,
    )
    repository_ctx.symlink(repository_ctx.path(repository_ctx.attr.build_file), "BUILD.bazel")

llama_release_repository = repository_rule(
    implementation = _llama_release_repository_impl,
    attrs = {
        "arm64_sha256": attr.string(mandatory = True),
        "build_file": attr.label(mandatory = True, allow_single_file = True),
        "source_sha256": attr.string(mandatory = True),
        "version": attr.string(mandatory = True),
        "x86_64_sha256": attr.string(mandatory = True),
    },
)
