// Stub compiler C ABI used until IREE is pinned and linked.
#include "dyninfer/compiler.h"

#include <cstdlib>
#include <cstring>
#include <string>

struct dyninfer_compiler_t {
  int abi_version;
};

static void release_bytes(uint8_t* data, size_t, void*) {
  std::free(data);
}

static dyninfer_owned_bytes_t own_string(const std::string& s) {
  auto* buf = static_cast<uint8_t*>(std::malloc(s.size()));
  if (buf && !s.empty()) {
    std::memcpy(buf, s.data(), s.size());
  }
  return dyninfer_owned_bytes_t{buf, s.size(), release_bytes, nullptr};
}

extern "C" int32_t dyninfer_compiler_create(
    dyninfer_bytes_t, dyninfer_compiler_t** out_compiler) {
  if (!out_compiler) return 1;
  *out_compiler = new dyninfer_compiler_t{1};
  return 0;
}

extern "C" int32_t dyninfer_compiler_compile(
    dyninfer_compiler_t* compiler,
    const dyninfer_compile_request_t*,
    dyninfer_compile_result_t* out_result) {
  if (!compiler || !out_result) return 1;
  // Placeholder VMFB magic used by the Rust stub runtime.
  static const char kStub[] = "DYNINFER_VMFB_STUB_v1";
  out_result->vmfb = own_string(std::string(kStub));
  out_result->metadata_json = own_string(R"({"stub":true,"compiler":"dyninfer-stub"})");
  out_result->diagnostics_utf8 = own_string("remark: stub compiler emitted placeholder VMFB\n");
  return 0;
}

extern "C" void dyninfer_compiler_destroy(dyninfer_compiler_t* compiler) {
  delete compiler;
}

extern "C" void dyninfer_compile_result_destroy(dyninfer_compile_result_t* result) {
  if (!result) return;
  if (result->vmfb.release && result->vmfb.data) {
    result->vmfb.release(result->vmfb.data, result->vmfb.size, result->vmfb.user_data);
  }
  if (result->metadata_json.release && result->metadata_json.data) {
    result->metadata_json.release(result->metadata_json.data, result->metadata_json.size,
                                  result->metadata_json.user_data);
  }
  if (result->diagnostics_utf8.release && result->diagnostics_utf8.data) {
    result->diagnostics_utf8.release(result->diagnostics_utf8.data,
                                     result->diagnostics_utf8.size,
                                     result->diagnostics_utf8.user_data);
  }
  *result = dyninfer_compile_result_t{};
}
