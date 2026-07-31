#ifndef DYNINFER_COMPILER_H_
#define DYNINFER_COMPILER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dyninfer_compiler_t dyninfer_compiler_t;

typedef struct dyninfer_bytes_t {
  const uint8_t* data;
  size_t size;
} dyninfer_bytes_t;

typedef struct dyninfer_owned_bytes_t {
  uint8_t* data;
  size_t size;
  void (*release)(uint8_t* data, size_t size, void* user_data);
  void* user_data;
} dyninfer_owned_bytes_t;

typedef struct dyninfer_compile_request_t {
  dyninfer_bytes_t architecture_mlirbc;
  dyninfer_bytes_t resolved_config_json;
  dyninfer_bytes_t binding_plan_json;
  dyninfer_bytes_t target_profile_json;
  dyninfer_bytes_t shape_profile_json;
  dyninfer_bytes_t compile_options_json;
} dyninfer_compile_request_t;

typedef struct dyninfer_compile_result_t {
  dyninfer_owned_bytes_t vmfb;
  dyninfer_owned_bytes_t metadata_json;
  dyninfer_owned_bytes_t diagnostics_utf8;
} dyninfer_compile_result_t;

int32_t dyninfer_compiler_create(
    dyninfer_bytes_t options_json,
    dyninfer_compiler_t** out_compiler);

int32_t dyninfer_compiler_compile(
    dyninfer_compiler_t* compiler,
    const dyninfer_compile_request_t* request,
    dyninfer_compile_result_t* out_result);

void dyninfer_compiler_destroy(dyninfer_compiler_t* compiler);
void dyninfer_compile_result_destroy(dyninfer_compile_result_t* result);

#ifdef __cplusplus
}
#endif

#endif
