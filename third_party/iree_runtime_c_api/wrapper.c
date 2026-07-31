#include "wrapper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/formats/parser_registry.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/modules/io/parameters/module.h"
#include "iree/runtime/api.h"

struct dyninfer_iree_session_t {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_runtime_session_t* session;
};

static char g_last_error[2048] = {0};

static void set_error_status(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    g_last_error[0] = '\0';
    return;
  }
  iree_host_size_t written = 0;
  (void)iree_status_format(status, sizeof(g_last_error), g_last_error, &written);
  if (written == 0 || g_last_error[0] == '\0') {
    snprintf(g_last_error, sizeof(g_last_error), "IREE status code %d",
             (int)iree_status_code(status));
  }
  iree_status_ignore(status);
}

static void set_error_msg(const char* msg) {
  snprintf(g_last_error, sizeof(g_last_error), "%s", msg ? msg : "unknown error");
}

const char* dyninfer_iree_last_error(void) { return g_last_error; }

void dyninfer_iree_free(void* p) { free(p); }

static iree_status_t append_parameters_module(iree_runtime_session_t* session,
                                             const char* parameters_path) {
  if (!parameters_path || parameters_path[0] == '\0') {
    return iree_ok_status();
  }

  iree_allocator_t host_allocator =
      iree_runtime_session_host_allocator(session);
  iree_vm_instance_t* vm_instance =
      iree_runtime_instance_vm_instance(iree_runtime_session_instance(session));

  iree_io_parameter_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(iree_io_parameter_index_create(host_allocator, &index));

  iree_io_file_handle_t* file_handle = NULL;
  iree_status_t status = iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ, iree_make_cstring_view(parameters_path),
      host_allocator, &file_handle);
  if (iree_status_is_ok(status)) {
    status = iree_io_parse_file_index(iree_make_cstring_view(parameters_path),
                                      file_handle, index, host_allocator);
  }
  iree_io_file_handle_release(file_handle);

  iree_io_parameter_provider_t* provider = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(
        iree_make_cstring_view("weights"), index,
        IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
        host_allocator, &provider);
  }
  iree_io_parameter_index_release(index);

  iree_vm_module_t* module = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameters_module_create(vm_instance, /*provider_count=*/1,
                                             &provider, host_allocator, &module);
  }
  iree_io_parameter_provider_release(provider);

  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_module(session, module);
  }
  iree_vm_module_release(module);
  return status;
}

static iree_status_t allocate_i64_tensor(iree_runtime_session_t* session,
                                         iree_host_size_t rank,
                                         const iree_hal_dim_t* shape,
                                         const int64_t* data,
                                         iree_host_size_t element_count,
                                         iree_hal_buffer_view_t** out_view) {
  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  return iree_hal_buffer_view_allocate_buffer_copy(
      device, device_allocator, rank, shape, IREE_HAL_ELEMENT_TYPE_INT_64,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(data, element_count * sizeof(int64_t)),
      out_view);
}

static iree_status_t allocate_f32_tensor(iree_runtime_session_t* session,
                                         iree_host_size_t rank,
                                         const iree_hal_dim_t* shape,
                                         const float* data,
                                         iree_host_size_t element_count,
                                         iree_hal_buffer_view_t** out_view) {
  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  return iree_hal_buffer_view_allocate_buffer_copy(
      device, device_allocator, rank, shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(data, element_count * sizeof(float)), out_view);
}

static iree_status_t copy_f32_view_to_host(iree_runtime_session_t* session,
                                           iree_hal_buffer_view_t* view,
                                           float** out_logits,
                                           size_t* out_count) {
  iree_device_size_t byte_length = iree_hal_buffer_view_byte_length(view);
  if (byte_length % sizeof(float) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected f32 buffer view");
  }
  size_t count = (size_t)(byte_length / sizeof(float));
  float* host = (float*)malloc(byte_length);
  if (!host) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "malloc logits");
  }
  iree_status_t status = iree_hal_device_transfer_d2h(
      iree_runtime_session_device(session), iree_hal_buffer_view_buffer(view),
      0, host, byte_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout());
  if (!iree_status_is_ok(status)) {
    free(host);
    return status;
  }
  *out_logits = host;
  *out_count = count;
  return iree_ok_status();
}

static iree_status_t invoke_named(iree_runtime_session_t* session,
                                  const char* full_name,
                                  iree_hal_buffer_view_t** inputs,
                                  iree_host_size_t input_count,
                                  float** out_logits, size_t* out_count) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view(full_name), &call));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < input_count; ++i) {
    if (!iree_status_is_ok(status)) break;
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, inputs[i]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  iree_hal_buffer_view_t* ret = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret);
  }
  if (iree_status_is_ok(status)) {
    status = copy_f32_view_to_host(session, ret, out_logits, out_count);
  }
  iree_hal_buffer_view_release(ret);
  iree_runtime_call_deinitialize(&call);
  return status;
}

int dyninfer_iree_session_create(const char* device_uri, const char* vmfb_path,
                                 const char* parameters_path,
                                 dyninfer_iree_session_t** out_session) {
  *out_session = NULL;
  g_last_error[0] = '\0';
  if (!vmfb_path || !out_session) {
    set_error_msg("vmfb_path and out_session are required");
    return 1;
  }

  const char* driver =
      (device_uri && device_uri[0] != '\0') ? device_uri : "local-task";

  dyninfer_iree_session_t* s =
      (dyninfer_iree_session_t*)calloc(1, sizeof(*s));
  if (!s) {
    set_error_msg("calloc session");
    return 1;
  }

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_status_t status = iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &s->instance);

  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_try_create_default_device(
        s->instance, iree_make_cstring_view(driver), &s->device);
  }

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        s->instance, &session_options, s->device,
        iree_runtime_instance_host_allocator(s->instance), &s->session);
  }

  if (iree_status_is_ok(status)) {
    status = append_parameters_module(s->session, parameters_path);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_bytecode_module_from_file(s->session,
                                                                   vmfb_path);
  }

  if (!iree_status_is_ok(status)) {
    set_error_status(status);
    dyninfer_iree_session_destroy(s);
    return 1;
  }
  *out_session = s;
  return 0;
}

void dyninfer_iree_session_destroy(dyninfer_iree_session_t* session) {
  if (!session) return;
  iree_runtime_session_release(session->session);
  iree_hal_device_release(session->device);
  iree_runtime_instance_release(session->instance);
  free(session);
}

int dyninfer_iree_session_invoke_add(dyninfer_iree_session_t* session,
                                     const float a[4], const float b[4],
                                     float** out_logits, size_t* out_count) {
  *out_logits = NULL;
  *out_count = 0;
  if (!session) {
    set_error_msg("null session");
    return 1;
  }
  static const iree_hal_dim_t shape[1] = {4};
  iree_hal_buffer_view_t* va = NULL;
  iree_hal_buffer_view_t* vb = NULL;
  iree_status_t status =
      allocate_f32_tensor(session->session, 1, shape, a, 4, &va);
  if (iree_status_is_ok(status)) {
    status = allocate_f32_tensor(session->session, 1, shape, b, 4, &vb);
  }
  iree_hal_buffer_view_t* inputs[2] = {va, vb};
  if (iree_status_is_ok(status)) {
    status = invoke_named(session->session, "module.add", inputs, 2, out_logits,
                          out_count);
  }
  iree_hal_buffer_view_release(va);
  iree_hal_buffer_view_release(vb);
  if (!iree_status_is_ok(status)) {
    set_error_status(status);
    return 1;
  }
  return 0;
}

int dyninfer_iree_session_invoke_prefill(dyninfer_iree_session_t* session,
                                         const int64_t* tokens,
                                         size_t token_count, int64_t last,
                                         float** out_logits, size_t* out_count) {
  *out_logits = NULL;
  *out_count = 0;
  if (!session || !tokens || token_count == 0) {
    set_error_msg("prefill requires session + non-empty tokens");
    return 1;
  }
  iree_hal_dim_t tokens_shape[1] = {(iree_hal_dim_t)token_count};
  iree_hal_buffer_view_t* v_tokens = NULL;
  iree_hal_buffer_view_t* v_last = NULL;
  iree_status_t status = allocate_i64_tensor(
      session->session, 1, tokens_shape, tokens, token_count, &v_tokens);
  if (iree_status_is_ok(status)) {
    // 0-d tensor<i64>
    status = allocate_i64_tensor(session->session, 0, NULL, &last, 1, &v_last);
  }
  iree_hal_buffer_view_t* inputs[2] = {v_tokens, v_last};
  if (iree_status_is_ok(status)) {
    status = invoke_named(session->session, "module.prefill", inputs, 2,
                          out_logits, out_count);
  }
  iree_hal_buffer_view_release(v_tokens);
  iree_hal_buffer_view_release(v_last);
  if (!iree_status_is_ok(status)) {
    set_error_status(status);
    return 1;
  }
  return 0;
}

int dyninfer_iree_session_invoke_decode(dyninfer_iree_session_t* session,
                                        int64_t token, float** out_logits,
                                        size_t* out_count) {
  *out_logits = NULL;
  *out_count = 0;
  if (!session) {
    set_error_msg("null session");
    return 1;
  }
  iree_hal_buffer_view_t* v_token = NULL;
  iree_status_t status =
      allocate_i64_tensor(session->session, 0, NULL, &token, 1, &v_token);
  iree_hal_buffer_view_t* inputs[1] = {v_token};
  if (iree_status_is_ok(status)) {
    status = invoke_named(session->session, "module.decode", inputs, 1,
                          out_logits, out_count);
  }
  iree_hal_buffer_view_release(v_token);
  if (!iree_status_is_ok(status)) {
    set_error_status(status);
    return 1;
  }
  return 0;
}
