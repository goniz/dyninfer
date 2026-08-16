package(default_visibility = ["//visibility:public"])

cc_import(
    name = "ggml_shared",
    shared_library = "libggml.so.0",
)

cc_import(
    name = "ggml_base_shared",
    shared_library = "libggml-base.so.0",
)

cc_import(
    name = "llama_shared",
    shared_library = "libllama.so.0",
)

cc_library(
    name = "llama",
    hdrs = glob([
        "source/ggml/include/ggml*.h",
        "source/ggml/include/gguf.h",
        "source/include/llama.h",
    ]),
    includes = [
        "source/ggml/include",
        "source/include",
    ],
    data = glob(["lib*.so*"]),
    deps = [
        ":ggml_base_shared",
        ":ggml_shared",
        ":llama_shared",
    ],
)
