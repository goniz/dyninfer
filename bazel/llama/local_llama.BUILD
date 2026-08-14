package(default_visibility = ["//visibility:public"])

cc_import(
    name = "ggml_shared",
    shared_library = "lib/libggml.so",
)

cc_import(
    name = "ggml_base_shared",
    shared_library = "lib/libggml-base.so",
)

cc_import(
    name = "llama_shared",
    shared_library = "lib/libllama.so",
)

cc_library(
    name = "llama",
    hdrs = glob([
        "include/ggml*.h",
        "include/gguf.h",
        "include/llama.h",
    ]),
    includes = ["include"],
    deps = [
        ":ggml_base_shared",
        ":ggml_shared",
        ":llama_shared",
    ],
)
