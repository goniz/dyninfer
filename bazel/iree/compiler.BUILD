load("@rules_cc//cc:defs.bzl", "cc_import")

package(default_visibility = ["//visibility:public"])

# Native binaries and shared libs live next to each other ($ORIGIN RPATH).
exports_files([
    "iree/compiler/_mlir_libs/iree-compile",
    "iree/compiler/_mlir_libs/libIREECompiler.so",
])

filegroup(
    name = "compiler_libs",
    srcs = glob(["iree/compiler/_mlir_libs/**"]),
)

filegroup(
    name = "iree-compile",
    srcs = ["iree/compiler/_mlir_libs/iree-compile"],
)

cc_import(
    name = "libIREECompiler",
    shared_library = "iree/compiler/_mlir_libs/libIREECompiler.so",
)
