#include "wrapper.h"

#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/modules/hal/types.h"
#include "iree/modules/io/parameters/module.h"
#include "iree/runtime/api.h"
#include "iree/vm/api.h"

// hipDeviceScheduleBlockingSync — sleep instead of spinning on GPU waits.
#define DYNINFER_HIP_DEVICE_SCHEDULE_BLOCKING_SYNC 0x04u
#define DYNINFER_MAX_PAGED_LAYERS 32

typedef struct dyninfer_cached_call_t {
  iree_runtime_call_t call;
  bool ready;
  char name[96];
} dyninfer_cached_call_t;

struct dyninfer_iree_session_t {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_runtime_session_t* session;
  // Cached entrypoints — avoid initialize_by_name + list alloc each step.
  dyninfer_cached_call_t call_add;
  dyninfer_cached_call_t call_prefill;
  dyninfer_cached_call_t call_decode;
  dyninfer_cached_call_t call_paged_prefill;
  dyninfer_cached_call_t call_paged_decode;
  // Dense decode scratch (mutated in place via H2D).
  iree_hal_buffer_view_t* decode_token;
  iree_hal_buffer_view_t* decode_pos;
  iree_hal_buffer_view_t* decode_bias;
  size_t decode_bias_len;
  // Dense prefill scratch.
  iree_hal_buffer_view_t* prefill_tokens;
  size_t prefill_tokens_len;
  iree_hal_buffer_view_t* prefill_last;
  // Paged chunk scratch.
  iree_hal_buffer_view_t* paged_tokens;
  size_t paged_tokens_len;
  iree_hal_buffer_view_t* paged_last;
  iree_hal_buffer_view_t* paged_start_pos;
  iree_hal_buffer_view_t* paged_logits;
  iree_hal_buffer_view_t* paged_write_pages;
  iree_hal_buffer_view_t* paged_write_flats;
  size_t paged_write_n;
  float* logits_host;
  size_t logits_capacity;
  // ABI v11: typed packed K and V per layer [nkv, kv_len, d].
  // fused_attn insert_slices the written page and attends in-place.
  iree_hal_buffer_view_t* kv_k[DYNINFER_MAX_PAGED_LAYERS];
  iree_hal_buffer_view_t* kv_v[DYNINFER_MAX_PAGED_LAYERS];
  size_t kv_layer_n;
  size_t kv_page_count;
  size_t kv_layer_count;
  size_t kv_page_size;
  size_t kv_head_count;
  size_t kv_head_dim;
  size_t kv_element_byte_count;
  size_t kv_chunk_size;
  size_t kv_vocab_size;
  size_t kv_allocated_bytes;
  bool split_modules;
  uint64_t hist_alias_n;
  uint64_t hist_clone_n;
};

static char g_last_error[2048] = {0};

static const char* g_paged_step = "";

static void set_error_status(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    g_last_error[0] = '\0';
    return;
  }
  // Skip iree_status_format: libbacktrace can SIGSEGV while walking a
  // dispatch-fault stack, hiding the original IREE error.
  const iree_status_code_t code = iree_status_code(status);
  if (g_paged_step[0] != '\0') {
    snprintf(g_last_error, sizeof(g_last_error), "%s (%d) at %s",
             iree_status_code_string(code), (int)code, g_paged_step);
  } else {
    snprintf(g_last_error, sizeof(g_last_error), "%s (%d)",
             iree_status_code_string(code), (int)code);
  }
  g_paged_step = "";
  iree_status_ignore(status);
}

static void set_error_msg(const char* msg) {
  snprintf(g_last_error, sizeof(g_last_error), "%s", msg ? msg : "unknown error");
}

// Prefer BlockingSync so HIP/HSA sleeps on GPU waits instead of spinning a
// host core (~100% CPU). Must run before IREE creates the HIP primary context.
// Opt out: DYNINFER_HIP_ALLOW_BUSY_WAIT=1.
static void configure_hip_blocking_sync(const char* device_uri) {
  if (!device_uri || device_uri[0] == '\0') return;
  if (strncmp(device_uri, "hip", 3) != 0 && strncmp(device_uri, "rocm", 4) != 0) {
    return;
  }
  const char* allow_busy = getenv("DYNINFER_HIP_ALLOW_BUSY_WAIT");
  if (allow_busy && allow_busy[0] == '1') return;

  void* lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_NOLOAD);
  if (!lib) {
    lib = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_LOCAL);
  }
  if (!lib) return;

  typedef int (*hip_init_fn)(unsigned int);
  typedef int (*hip_get_device_count_fn)(int*);
  typedef int (*hip_set_device_fn)(int);
  typedef int (*hip_set_device_flags_fn)(unsigned int);

  hip_init_fn hipInit =
      (hip_init_fn)dlsym(lib, "hipInit");
  hip_get_device_count_fn hipGetDeviceCount =
      (hip_get_device_count_fn)dlsym(lib, "hipGetDeviceCount");
  hip_set_device_fn hipSetDevice =
      (hip_set_device_fn)dlsym(lib, "hipSetDevice");
  hip_set_device_flags_fn hipSetDeviceFlags =
      (hip_set_device_flags_fn)dlsym(lib, "hipSetDeviceFlags");
  if (!hipSetDeviceFlags) return;

  if (hipInit) (void)hipInit(0);
  int count = 1;
  if (hipGetDeviceCount) {
    if (hipGetDeviceCount(&count) != 0 || count < 1) count = 1;
  }
  for (int i = 0; i < count; ++i) {
    if (hipSetDevice) (void)hipSetDevice(i);
    (void)hipSetDeviceFlags(DYNINFER_HIP_DEVICE_SCHEDULE_BLOCKING_SYNC);
  }
}

static iree_status_t cached_call_prepare(dyninfer_iree_session_t* session,
                                         dyninfer_cached_call_t* cached,
                                         const char* full_name) {
  if (cached->ready && strcmp(cached->name, full_name) == 0) {
    iree_runtime_call_reset(&cached->call);
    return iree_ok_status();
  }
  if (cached->ready) {
    iree_runtime_call_deinitialize(&cached->call);
    cached->ready = false;
    cached->name[0] = '\0';
  }
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session->session, iree_make_cstring_view(full_name), &cached->call));
  snprintf(cached->name, sizeof(cached->name), "%s", full_name);
  cached->ready = true;
  return iree_ok_status();
}

static void cached_call_release(dyninfer_cached_call_t* cached) {
  if (!cached || !cached->ready) return;
  iree_runtime_call_deinitialize(&cached->call);
  cached->ready = false;
  cached->name[0] = '\0';
}

const char* dyninfer_iree_last_error(void) { return g_last_error; }

void dyninfer_iree_free(void* p) {
  // Logits are returned from session-owned scratch and must not be freed here.
  // Retained for ABI stability; callers still copy into owned Vecs immediately.
  (void)p;
}

// Builds a container-independent parameter index over exact byte ranges in the
// original checkpoint files. File handles are opened once here and retained by
// index entries/provider/module for the session lifetime.
static iree_status_t append_file_parameters_module(
    iree_runtime_session_t* session,
    const dyninfer_iree_parameter_file_t* files, size_t file_count,
    const dyninfer_iree_file_param_t* params, size_t param_count) {
  if (!files || file_count == 0 || !params || param_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file parameter descriptors are empty");
  }

  iree_allocator_t host_allocator =
      iree_runtime_session_host_allocator(session);
  iree_vm_instance_t* vm_instance =
      iree_runtime_instance_vm_instance(iree_runtime_session_instance(session));
  iree_io_file_handle_t** handles =
      (iree_io_file_handle_t**)calloc(file_count, sizeof(*handles));
  if (!handles) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "allocating %zu checkpoint file handles", file_count);
  }

  iree_status_t status = iree_ok_status();
  for (size_t i = 0; iree_status_is_ok(status) && i < file_count; ++i) {
    if (!files[i].path || files[i].path[0] == '\0') {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "checkpoint file %zu has no path", i);
      break;
    }
    status = iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS,
        iree_make_cstring_view(files[i].path), host_allocator, &handles[i]);
  }

  iree_io_parameter_index_t* index = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_create(host_allocator, &index);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_reserve(index, param_count);
  }
  for (size_t i = 0; iree_status_is_ok(status) && i < param_count; ++i) {
    if (!params[i].key || params[i].key[0] == '\0' || params[i].length == 0 ||
        params[i].source_file_index >= file_count ||
        params[i].offset > UINT64_MAX - params[i].length) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid file parameter descriptor %zu", i);
      break;
    }
    iree_io_parameter_index_entry_t entry = {
        .key = iree_make_cstring_view(params[i].key),
        .metadata = iree_const_byte_span_empty(),
        .length = params[i].length,
        .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE,
        .storage =
            {
                .file =
                    {
                        .handle = handles[params[i].source_file_index],
                        .offset = params[i].offset,
                    },
            },
    };
    status = iree_io_parameter_index_add(index, &entry);
  }

  for (size_t i = 0; i < file_count; ++i) {
    if (handles[i]) iree_io_file_handle_release(handles[i]);
  }
  free(handles);

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

static int session_create_common(const char* device_uri, const char* vmfb_path,
                                 const char* decode_vmfb_path,
                                 const dyninfer_iree_parameter_file_t* files,
                                 size_t file_count,
                                 const dyninfer_iree_file_param_t* file_params,
                                 size_t file_param_count,
                                 dyninfer_iree_session_t** out_session) {
  *out_session = NULL;
  g_last_error[0] = '\0';
  if (!vmfb_path || !out_session) {
    set_error_msg("vmfb_path and out_session are required");
    return 1;
  }

  const char* driver_or_uri =
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
    // Sleep on GPU completion instead of busy-waiting a host core.
    configure_hip_blocking_sync(driver_or_uri);
    // Full HAL URIs (contain "://") select a specific device; bare driver
    // names fall back to the driver's default device.
    if (strstr(driver_or_uri, "://") != NULL) {
      iree_hal_driver_registry_t* registry =
          iree_runtime_instance_driver_registry(s->instance);
      status = iree_hal_create_device(
          registry, iree_make_cstring_view(driver_or_uri),
          iree_runtime_instance_host_allocator(s->instance), &s->device);
    } else {
      status = iree_runtime_instance_try_create_default_device(
          s->instance, iree_make_cstring_view(driver_or_uri), &s->device);
    }
  }

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        s->instance, &session_options, s->device,
        iree_runtime_instance_host_allocator(s->instance), &s->session);
  }

  if (iree_status_is_ok(status)) {
    if (files && file_count > 0 && file_params && file_param_count > 0) {
      status = append_file_parameters_module(s->session, files, file_count,
                                             file_params, file_param_count);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_bytecode_module_from_file(s->session,
                                                                   vmfb_path);
  }
  if (iree_status_is_ok(status) && decode_vmfb_path &&
      decode_vmfb_path[0] != '\0') {
    status = iree_runtime_session_append_bytecode_module_from_file(
        s->session, decode_vmfb_path);
    s->split_modules = iree_status_is_ok(status);
  }

  if (!iree_status_is_ok(status)) {
    set_error_status(status);
    dyninfer_iree_session_destroy(s);
    return 1;
  }
  *out_session = s;
  return 0;
}

int dyninfer_iree_session_create(const char* device_uri, const char* vmfb_path,
                                 dyninfer_iree_session_t** out_session) {
  return session_create_common(device_uri, vmfb_path, /*decode_vmfb_path=*/NULL,
                               /*files=*/NULL, /*file_count=*/0,
                               /*file_params=*/NULL, /*file_param_count=*/0,
                               out_session);
}

int dyninfer_iree_session_create_with_file_params(
    const char* device_uri, const char* vmfb_path,
    const dyninfer_iree_parameter_file_t* files, size_t file_count,
    const dyninfer_iree_file_param_t* params, size_t param_count,
    dyninfer_iree_session_t** out_session) {
  return session_create_common(device_uri, vmfb_path, /*decode_vmfb_path=*/NULL,
                               files, file_count, params, param_count,
                               out_session);
}

int dyninfer_iree_session_create_modules_with_file_params(
    const char* device_uri, const char* prefill_vmfb_path,
    const char* decode_vmfb_path, const dyninfer_iree_parameter_file_t* files,
    size_t file_count, const dyninfer_iree_file_param_t* params,
    size_t param_count, dyninfer_iree_session_t** out_session) {
  return session_create_common(device_uri, prefill_vmfb_path, decode_vmfb_path,
                               files, file_count, params, param_count,
                               out_session);
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

static iree_status_t allocate_float_tensor(
    iree_runtime_session_t* session, iree_host_size_t rank,
    const iree_hal_dim_t* shape, const void* data,
    iree_host_size_t element_count, size_t element_byte_count,
    iree_hal_buffer_view_t** out_view) {
  if (element_byte_count == sizeof(float)) {
    return allocate_f32_tensor(session, rank, shape, (const float*)data,
                               element_count, out_view);
  }
  const iree_hal_element_type_t element_type = element_byte_count == 2
      ? IREE_HAL_ELEMENT_TYPE_FLOAT_16
      : IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  return iree_hal_buffer_view_allocate_buffer_copy(
      device, device_allocator, rank, shape, element_type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(data, element_count * element_byte_count),
      out_view);
}

// Zero a device buffer. Prefer a GPU fill (no multi-GB host staging); HIP
// DEVICE_LOCAL buffers are not host-mappable (`map_zero` fails) and a chunked
// H2D of a packed 32k pool has aborted the HIP semaphore.
static iree_status_t zero_device_buffer(iree_hal_device_t* device,
                                        iree_hal_buffer_t* buffer,
                                        iree_device_size_t byte_length) {
  if (byte_length == 0) return iree_ok_status();
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull, IREE_HAL_SEMAPHORE_FLAG_NONE,
      &semaphore));
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal = {
      .count = 1,
      .semaphores = &semaphore,
      .payload_values = &signal_value,
  };
  uint32_t pattern = 0;
  iree_status_t status = iree_hal_device_queue_fill(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signal, buffer, 0, byte_length, &pattern, sizeof(pattern),
      IREE_HAL_FILL_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_HAL_WAIT_FLAG_DEFAULT);
  }
  iree_hal_semaphore_release(semaphore);
  if (iree_status_is_ok(status)) return status;
  iree_status_ignore(status);

  enum { kChunkBytes = 4u * 1024u * 1024u };  // 4 MiB host staging fallback
  void* zeros = calloc(1, kChunkBytes);
  if (!zeros) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "allocating KV zero staging chunk");
  }
  status = iree_ok_status();
  for (iree_device_size_t offset = 0; offset < byte_length;) {
    iree_device_size_t n = byte_length - offset;
    if (n > kChunkBytes) n = kChunkBytes;
    status = iree_hal_device_transfer_h2d(
        device, zeros, buffer, offset, n, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
    if (!iree_status_is_ok(status)) break;
    offset += n;
  }
  free(zeros);
  return status;
}

// Device-local f32 tensor without a full-size host staging copy. Used for
// multi-GB KV pools. Contents are GPU-filled to zero so flash-attn never
// reads NaNs from uninitialized DEVICE_LOCAL memory.
static iree_status_t allocate_empty_f32_tensor(
    iree_runtime_session_t* session, iree_host_size_t rank,
    const iree_hal_dim_t* shape, iree_host_size_t element_count,
    iree_hal_buffer_view_t** out_view) {
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  iree_device_size_t byte_length =
      (iree_device_size_t)element_count * sizeof(float);
  if (element_count != 0 && byte_length / sizeof(float) != element_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "f32 tensor byte length overflow");
  }
  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      device_allocator, params, byte_length, &buffer));
  iree_status_t status = zero_device_buffer(
      iree_runtime_session_device(session), buffer, byte_length);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(buffer);
    return status;
  }
  status = iree_hal_buffer_view_create(
      buffer, rank, shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      iree_runtime_session_host_allocator(session), out_view);
  iree_hal_buffer_release(buffer);  // view retains its own reference
  return status;
}

static iree_status_t allocate_empty_float_tensor(
    iree_runtime_session_t* session, iree_host_size_t rank,
    const iree_hal_dim_t* shape, iree_host_size_t element_count,
    size_t element_byte_count, iree_hal_buffer_view_t** out_view) {
  if (element_byte_count == sizeof(float)) {
    return allocate_empty_f32_tensor(session, rank, shape, element_count,
                                     out_view);
  }
  const iree_hal_element_type_t element_type = element_byte_count == 2
      ? IREE_HAL_ELEMENT_TYPE_FLOAT_16
      : IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  iree_device_size_t byte_length =
      (iree_device_size_t)element_count * element_byte_count;
  if (element_count != 0 && byte_length / element_byte_count != element_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "float tensor byte length overflow");
  }
  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      device_allocator, params, byte_length, &buffer));
  iree_status_t status = zero_device_buffer(
      iree_runtime_session_device(session), buffer, byte_length);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(buffer);
    return status;
  }
  status = iree_hal_buffer_view_create(
      buffer, rank, shape, element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      iree_runtime_session_host_allocator(session), out_view);
  iree_hal_buffer_release(buffer);
  return status;
}

static iree_status_t copy_f32_view_to_host(iree_runtime_session_t* session,
                                           iree_hal_buffer_view_t* view,
                                           float** logits_scratch,
                                           size_t* logits_capacity,
                                           float** out_logits,
                                           size_t* out_count) {
  iree_device_size_t byte_length = iree_hal_buffer_view_byte_length(view);
  if (byte_length % sizeof(float) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected f32 buffer view");
  }
  size_t count = (size_t)(byte_length / sizeof(float));
  if (*logits_capacity < count) {
    float* grown = (float*)realloc(*logits_scratch, byte_length);
    if (!grown) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "realloc logits");
    }
    *logits_scratch = grown;
    *logits_capacity = count;
  }
  float* host = *logits_scratch;
  iree_status_t status = iree_hal_device_transfer_d2h(
      iree_runtime_session_device(session), iree_hal_buffer_view_buffer(view),
      0, host, byte_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout());
  if (!iree_status_is_ok(status)) {
    return status;
  }
  // Borrowed session scratch: valid until the next invoke on this session.
  // Callers must copy immediately; dyninfer_iree_free is a no-op for these.
  *out_logits = host;
  *out_count = count;
  return iree_ok_status();
}

static iree_status_t invoke_with_cached_call(
    dyninfer_iree_session_t* wrapper, dyninfer_cached_call_t* cached,
    const char* full_name, iree_hal_buffer_view_t** inputs,
    iree_host_size_t input_count, float** out_logits, size_t* out_count) {
  IREE_RETURN_IF_ERROR(cached_call_prepare(wrapper, cached, full_name));
  iree_runtime_call_t* call = &cached->call;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < input_count; ++i) {
    if (!iree_status_is_ok(status)) break;
    status = iree_runtime_call_inputs_push_back_buffer_view(call, inputs[i]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(call, /*flags=*/0);
  }

  iree_hal_buffer_view_t* ret = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_outputs_pop_front_buffer_view(call, &ret);
  }
  if (iree_status_is_ok(status)) {
    status = copy_f32_view_to_host(wrapper->session, ret, &wrapper->logits_host,
                                   &wrapper->logits_capacity, out_logits,
                                   out_count);
  }
  iree_hal_buffer_view_release(ret);
  // Drop retained input refs so the next prepare/reset starts clean.
  iree_runtime_call_reset(call);
  return status;
}

void dyninfer_iree_session_destroy(dyninfer_iree_session_t* session) {
  if (!session) return;
  if (getenv("DYNINFER_PAGED_HIST_DEBUG") &&
      (session->hist_alias_n != 0 || session->hist_clone_n != 0)) {
    fprintf(stderr, "paged hist alias=%llu clone=%llu\n",
            (unsigned long long)session->hist_alias_n,
            (unsigned long long)session->hist_clone_n);
  }
  cached_call_release(&session->call_add);
  cached_call_release(&session->call_prefill);
  cached_call_release(&session->call_decode);
  cached_call_release(&session->call_paged_prefill);
  cached_call_release(&session->call_paged_decode);
  iree_hal_buffer_view_release(session->decode_token);
  iree_hal_buffer_view_release(session->decode_pos);
  iree_hal_buffer_view_release(session->decode_bias);
  iree_hal_buffer_view_release(session->prefill_tokens);
  iree_hal_buffer_view_release(session->prefill_last);
  iree_hal_buffer_view_release(session->paged_tokens);
  iree_hal_buffer_view_release(session->paged_last);
  iree_hal_buffer_view_release(session->paged_start_pos);
  iree_hal_buffer_view_release(session->paged_logits);
  iree_hal_buffer_view_release(session->paged_write_pages);
  iree_hal_buffer_view_release(session->paged_write_flats);
  for (size_t i = 0; i < session->kv_layer_n; ++i) {
    iree_hal_buffer_view_release(session->kv_k[i]);
    iree_hal_buffer_view_release(session->kv_v[i]);
  }
  free(session->logits_host);
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
    status = invoke_with_cached_call(session, &session->call_add, "module.add",
                                     inputs, 2, out_logits, out_count);
  }
  iree_hal_buffer_view_release(va);
  iree_hal_buffer_view_release(vb);
  if (!iree_status_is_ok(status)) {
    set_error_status(status);
    return 1;
  }
  return 0;
}

static iree_status_t ensure_i64_scalar_view(iree_runtime_session_t* session,
                                            iree_hal_buffer_view_t** slot,
                                            int64_t value) {
  if (*slot == NULL) {
    return allocate_i64_tensor(session, 0, NULL, &value, 1, slot);
  }
  return iree_hal_device_transfer_h2d(
      iree_runtime_session_device(session), &value,
      iree_hal_buffer_view_buffer(*slot), 0, sizeof(value),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
}

static iree_status_t ensure_i64_vector_view(iree_runtime_session_t* session,
                                            iree_hal_buffer_view_t** slot,
                                            size_t* slot_len,
                                            const int64_t* data,
                                            size_t count) {
  if (*slot == NULL || *slot_len != count) {
    iree_hal_buffer_view_release(*slot);
    *slot = NULL;
    *slot_len = 0;
    iree_hal_dim_t shape[1] = {(iree_hal_dim_t)count};
    iree_status_t status =
        allocate_i64_tensor(session, 1, shape, data, count, slot);
    if (iree_status_is_ok(status)) {
      *slot_len = count;
    }
    return status;
  }
  return iree_hal_device_transfer_h2d(
      iree_runtime_session_device(session), data,
      iree_hal_buffer_view_buffer(*slot), 0, count * sizeof(int64_t),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
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
  iree_status_t status = ensure_i64_vector_view(
      session->session, &session->prefill_tokens, &session->prefill_tokens_len,
      tokens, token_count);
  if (iree_status_is_ok(status)) {
    status = ensure_i64_scalar_view(session->session, &session->prefill_last,
                                    last);
  }
  iree_hal_buffer_view_t* inputs[2] = {session->prefill_tokens,
                                       session->prefill_last};
  if (iree_status_is_ok(status)) {
    status = invoke_with_cached_call(session, &session->call_prefill,
                                     "module.prefill", inputs, 2, out_logits,
                                     out_count);
  }
  if (!iree_status_is_ok(status)) {
    set_error_status(status);
    return 1;
  }
  return 0;
}

static iree_status_t ensure_f32_bias_view(iree_runtime_session_t* session,
                                          iree_hal_buffer_view_t** slot,
                                          size_t* slot_len,
                                          const float* attn_bias,
                                          size_t bias_len) {
  if (*slot == NULL || *slot_len != bias_len) {
    iree_hal_buffer_view_release(*slot);
    *slot = NULL;
    *slot_len = 0;
    iree_hal_dim_t bias_shape[1] = {(iree_hal_dim_t)bias_len};
    iree_status_t status =
        allocate_f32_tensor(session, 1, bias_shape, attn_bias, bias_len, slot);
    if (iree_status_is_ok(status)) {
      *slot_len = bias_len;
    }
    return status;
  }
  return iree_hal_device_transfer_h2d(
      iree_runtime_session_device(session), attn_bias,
      iree_hal_buffer_view_buffer(*slot), 0, bias_len * sizeof(float),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
}

int dyninfer_iree_session_invoke_decode(dyninfer_iree_session_t* session,
                                        int64_t token, int64_t pos,
                                        const float* attn_bias, size_t bias_len,
                                        float** out_logits, size_t* out_count) {
  *out_logits = NULL;
  *out_count = 0;
  if (!session || !attn_bias || bias_len == 0) {
    set_error_msg("decode requires session + non-empty attn_bias");
    return 1;
  }
  iree_status_t status =
      ensure_i64_scalar_view(session->session, &session->decode_token, token);
  if (iree_status_is_ok(status)) {
    status = ensure_i64_scalar_view(session->session, &session->decode_pos, pos);
  }
  if (iree_status_is_ok(status)) {
    status = ensure_f32_bias_view(session->session, &session->decode_bias,
                                  &session->decode_bias_len, attn_bias,
                                  bias_len);
  }
  iree_hal_buffer_view_t* inputs[3] = {session->decode_token, session->decode_pos,
                                       session->decode_bias};
  if (iree_status_is_ok(status)) {
    status = invoke_with_cached_call(session, &session->call_decode,
                                     "module.decode", inputs, 3, out_logits,
                                     out_count);
  }
  if (!iree_status_is_ok(status)) {
    set_error_status(status);
    return 1;
  }
  return 0;
}

int dyninfer_iree_session_configure_paged_kv(
    dyninfer_iree_session_t* session, size_t layer_count, size_t page_size,
    size_t kv_head_count, size_t head_dim, size_t kv_element_byte_count,
    size_t chunk_size,
    size_t vocab_size) {
  if (!session || layer_count == 0 || page_size == 0 || kv_head_count == 0 ||
      head_dim == 0 ||
      (kv_element_byte_count != 2 && kv_element_byte_count != 4) ||
      chunk_size == 0 || vocab_size == 0 ||
      (page_size % chunk_size != 0 && chunk_size % page_size != 0)) {
    set_error_msg("invalid paged KV configuration");
    return 1;
  }
  if (session->kv_layer_n != 0) {
    set_error_msg("cannot reconfigure paged KV after allocation");
    return 1;
  }
  if (layer_count > DYNINFER_MAX_PAGED_LAYERS) {
    set_error_msg("paged KV layer_count exceeds DYNINFER_MAX_PAGED_LAYERS");
    return 1;
  }
  session->kv_layer_count = layer_count;
  session->kv_page_size = page_size;
  session->kv_head_count = kv_head_count;
  session->kv_head_dim = head_dim;
  session->kv_element_byte_count = kv_element_byte_count;
  session->kv_chunk_size = chunk_size;
  session->kv_vocab_size = vocab_size;
  return 0;
}

int dyninfer_iree_session_ensure_kv_pages(dyninfer_iree_session_t* session,
                                          size_t page_count) {
  if (!session || session->kv_page_size == 0) {
    set_error_msg("paged KV is not configured");
    return 1;
  }
  if (page_count == 0) {
    set_error_msg("paged KV pool requires page_count > 0");
    return 1;
  }
  // Already have per-layer hists large enough for the compiled arity.
  if (session->kv_layer_n == session->kv_layer_count &&
      session->kv_page_count >= page_count) {
    return 0;
  }
  for (size_t i = 0; i < session->kv_layer_n; ++i) {
    iree_hal_buffer_view_release(session->kv_k[i]);
    session->kv_k[i] = NULL;
    iree_hal_buffer_view_release(session->kv_v[i]);
    session->kv_v[i] = NULL;
  }
  session->kv_layer_n = 0;
  session->kv_page_count = 0;
  session->kv_allocated_bytes = 0;

  if (session->kv_layer_count == 0 ||
      session->kv_layer_count > DYNINFER_MAX_PAGED_LAYERS) {
    set_error_msg("KV layer count overflow");
    return 1;
  }
  if (session->kv_page_size == 0 ||
      page_count > SIZE_MAX / session->kv_page_size) {
    set_error_msg("KV packed length overflow");
    return 1;
  }
  const size_t kv_len = page_count * session->kv_page_size;
  size_t plane_elems = session->kv_head_count;
  const size_t factors[2] = {kv_len, session->kv_head_dim};
  for (size_t i = 0; i < 2; ++i) {
    if (factors[i] == 0 || plane_elems > SIZE_MAX / factors[i]) {
      set_error_msg("KV packed size overflow");
      return 1;
    }
    plane_elems *= factors[i];
  }
  if (plane_elems > SIZE_MAX / session->kv_element_byte_count) {
    set_error_msg("KV packed byte size overflow");
    return 1;
  }
  const size_t plane_bytes = plane_elems * session->kv_element_byte_count;
  if (session->kv_layer_count > SIZE_MAX / 2 ||
      session->kv_layer_count * 2 > SIZE_MAX / plane_bytes) {
    set_error_msg("KV pool byte size overflow");
    return 1;
  }
  iree_hal_dim_t shape[3] = {
      (iree_hal_dim_t)session->kv_head_count,
      (iree_hal_dim_t)kv_len,
      (iree_hal_dim_t)session->kv_head_dim};
  void* zeros = calloc(plane_elems, session->kv_element_byte_count);
  iree_status_t status = iree_ok_status();
  for (size_t layer = 0; layer < session->kv_layer_count; ++layer) {
    iree_hal_buffer_view_t* k = NULL;
    iree_hal_buffer_view_t* v = NULL;
    if (zeros) {
      status = allocate_float_tensor(
          session->session, 3, shape, zeros, plane_elems,
          session->kv_element_byte_count, &k);
      if (iree_status_is_ok(status)) {
        status = allocate_float_tensor(
            session->session, 3, shape, zeros, plane_elems,
            session->kv_element_byte_count, &v);
      }
    } else {
      status = allocate_empty_float_tensor(
          session->session, 3, shape, plane_elems,
          session->kv_element_byte_count, &k);
      if (iree_status_is_ok(status)) {
        status = allocate_empty_float_tensor(
            session->session, 3, shape, plane_elems,
            session->kv_element_byte_count, &v);
      }
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_buffer_view_release(k);
      iree_hal_buffer_view_release(v);
      for (size_t i = 0; i < layer; ++i) {
        iree_hal_buffer_view_release(session->kv_k[i]);
        session->kv_k[i] = NULL;
        iree_hal_buffer_view_release(session->kv_v[i]);
        session->kv_v[i] = NULL;
      }
      free(zeros);
      char msg[256];
      snprintf(msg, sizeof(msg),
               "allocating packed KV layer %zu (%zu tok, %.2f MiB): see IREE status",
               layer, kv_len,
               (double)(plane_bytes * 2) / (1024.0 * 1024.0));
      set_error_status(status);
      if (g_last_error[0] == '\0') {
        set_error_msg(msg);
      }
      return 1;
    }
    session->kv_k[layer] = k;
    session->kv_v[layer] = v;
  }
  free(zeros);
  session->kv_layer_n = session->kv_layer_count;
  session->kv_page_count = page_count;
  session->kv_allocated_bytes = plane_bytes * 2 * session->kv_layer_count;
  return 0;
}

static iree_status_t ensure_paged_logits(dyninfer_iree_session_t* session) {
  if (session->paged_logits) return iree_ok_status();
  if (session->kv_vocab_size == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "paged logits buffer requires vocab_size");
  }
  float* zeros = (float*)calloc(session->kv_vocab_size, sizeof(float));
  if (!zeros) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "allocating zero-filled logits staging");
  }
  iree_hal_dim_t shape[1] = {(iree_hal_dim_t)session->kv_vocab_size};
  iree_status_t status =
      allocate_f32_tensor(session->session, 1, shape, zeros,
                          session->kv_vocab_size, &session->paged_logits);
  free(zeros);
  return status;
}

static iree_status_t copy_i64_scalar_view_to_host(iree_runtime_session_t* session,
                                                  iree_hal_buffer_view_t* view,
                                                  int64_t* out_value) {
  iree_device_size_t byte_length = iree_hal_buffer_view_byte_length(view);
  if (byte_length != sizeof(int64_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected scalar i64 buffer view");
  }
  return iree_hal_device_transfer_d2h(
      iree_runtime_session_device(session), iree_hal_buffer_view_buffer(view),
      0, out_value, sizeof(int64_t), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout());
}

static void adopt_tied_kv_view(iree_hal_buffer_view_t** stored,
                               iree_hal_buffer_view_t* out,
                               uint64_t* alias_n, uint64_t* clone_n) {
  iree_hal_buffer_t* in_buf = iree_hal_buffer_view_buffer(*stored);
  iree_hal_buffer_t* out_buf = iree_hal_buffer_view_buffer(out);
  if (in_buf != out_buf) {
    (*clone_n)++;
    iree_hal_buffer_view_release(*stored);
    *stored = out;
  } else {
    (*alias_n)++;
    iree_hal_buffer_view_release(out);
  }
}

static iree_status_t invoke_paged_chunk_once(
    dyninfer_iree_session_t* wrapper, dyninfer_cached_call_t* cached,
    const char* full_name, iree_hal_buffer_view_t* v_tokens,
    iree_hal_buffer_view_t* v_last, iree_hal_buffer_view_t* v_start,
    float** out_logits, size_t* out_count, int64_t* out_token,
    bool want_logits) {
  if (wrapper->kv_layer_n == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "paged KV hist is not allocated");
  }
  IREE_RETURN_IF_ERROR(cached_call_prepare(wrapper, cached, full_name));
  IREE_RETURN_IF_ERROR(ensure_paged_logits(wrapper));
  iree_runtime_call_t* call = &cached->call;

  g_paged_step = "push_inputs";
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(call, v_tokens);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(call, v_last);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(call, v_start);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(call,
                                                            wrapper->paged_logits);
  }
  for (size_t layer = 0; iree_status_is_ok(status) && layer < wrapper->kv_layer_n;
       ++layer) {
    status = iree_runtime_call_inputs_push_back_buffer_view(
        call, wrapper->kv_k[layer]);
    if (iree_status_is_ok(status)) {
      status = iree_runtime_call_inputs_push_back_buffer_view(
          call, wrapper->kv_v[layer]);
    }
  }
  if (iree_status_is_ok(status)) {
    g_paged_step = "invoke";
    status = iree_runtime_call_invoke(call, /*flags=*/0);
  }

  // Results: logits (abi.output alias), tied packed K/V per layer, argmax token.
  iree_hal_buffer_view_t* logits_view = NULL;
  if (iree_status_is_ok(status)) {
    g_paged_step = "pop_logits";
    status =
        iree_runtime_call_outputs_pop_front_buffer_view(call, &logits_view);
  }
  if (iree_status_is_ok(status) && want_logits) {
    g_paged_step = "copy_logits";
    if (!out_logits || !out_count) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "want_logits requires out_logits/out_count");
    } else {
      status = copy_f32_view_to_host(wrapper->session, logits_view,
                                     &wrapper->logits_host,
                                     &wrapper->logits_capacity, out_logits,
                                     out_count);
    }
  }
  iree_hal_buffer_view_release(logits_view);

  for (size_t layer = 0; iree_status_is_ok(status) && layer < wrapper->kv_layer_n;
       ++layer) {
    iree_hal_buffer_view_t* k_out = NULL;
    iree_hal_buffer_view_t* v_out = NULL;
    g_paged_step = "pop_k";
    status = iree_runtime_call_outputs_pop_front_buffer_view(call, &k_out);
    if (iree_status_is_ok(status)) {
      g_paged_step = "pop_v";
      status = iree_runtime_call_outputs_pop_front_buffer_view(call, &v_out);
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_buffer_view_release(k_out);
      iree_hal_buffer_view_release(v_out);
      break;
    }
    adopt_tied_kv_view(&wrapper->kv_k[layer], k_out, &wrapper->hist_alias_n,
                       &wrapper->hist_clone_n);
    adopt_tied_kv_view(&wrapper->kv_v[layer], v_out, &wrapper->hist_alias_n,
                       &wrapper->hist_clone_n);
  }

  iree_hal_buffer_view_t* token_view = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_runtime_call_outputs_pop_front_buffer_view(call, &token_view);
  }
  if (iree_status_is_ok(status) && out_token) {
    status = copy_i64_scalar_view_to_host(wrapper->session, token_view,
                                          out_token);
  }
  iree_hal_buffer_view_release(token_view);

  iree_runtime_call_reset(call);
  if (iree_status_is_ok(status)) g_paged_step = "";
  return status;
}

int dyninfer_iree_session_invoke_paged_chunk(
    dyninfer_iree_session_t* session, const int64_t* tokens,
    size_t token_count, int64_t last, int64_t start_pos, float** out_logits,
    size_t* out_count, int64_t* out_token, int want_logits) {
  if (out_logits) *out_logits = NULL;
  if (out_count) *out_count = 0;
  if (out_token) *out_token = -1;
  if (!session || !tokens ||
      (token_count != session->kv_chunk_size && token_count != 1) ||
      last < 0 || (size_t)last >= token_count || start_pos < 0) {
    set_error_msg("invalid paged chunk invocation");
    return 1;
  }
  if (want_logits && (!out_logits || !out_count)) {
    set_error_msg("want_logits requires out_logits/out_count");
    return 1;
  }
  const bool is_decode = token_count == 1;
  const char* module_name =
      session->split_modules ? (is_decode ? "decode" : "prefill") : "module";
  const char* entry = is_decode ? "decode_chunk" : "prefill_chunk";
  char function_name[96];
  snprintf(function_name, sizeof(function_name), "%s.%s", module_name, entry);

  iree_status_t status = ensure_i64_vector_view(
      session->session, &session->paged_tokens, &session->paged_tokens_len,
      tokens, token_count);
  if (iree_status_is_ok(status)) {
    status =
        ensure_i64_scalar_view(session->session, &session->paged_last, last);
  }
  if (iree_status_is_ok(status)) {
    status = ensure_i64_scalar_view(session->session, &session->paged_start_pos,
                                    start_pos);
  }
  dyninfer_cached_call_t* cached =
      is_decode ? &session->call_paged_decode : &session->call_paged_prefill;
  if (iree_status_is_ok(status)) {
    status = invoke_paged_chunk_once(
        session, cached, function_name, session->paged_tokens,
        session->paged_last, session->paged_start_pos, out_logits, out_count,
        out_token, want_logits != 0);
  }
  if (!iree_status_is_ok(status)) {
    set_error_status(status);
    return 1;
  }
  return 0;
}

int dyninfer_iree_session_reset_paged_kv(dyninfer_iree_session_t* session) {
  if (!session) return 1;
  if (session->kv_layer_n == 0) {
    session->kv_page_count = 0;
    session->kv_allocated_bytes = 0;
    return 0;
  }
  // Zero in place — avoid free/realloc of multi-GB hists between iters.
  for (size_t i = 0; i < session->kv_layer_n; ++i) {
    iree_hal_buffer_view_t* views[2] = {session->kv_k[i], session->kv_v[i]};
    for (int p = 0; p < 2; ++p) {
      iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(views[p]);
      iree_device_size_t bytes = iree_hal_buffer_view_byte_length(views[p]);
      iree_status_t status = zero_device_buffer(session->device, buffer, bytes);
      if (!iree_status_is_ok(status)) {
        for (size_t j = 0; j < session->kv_layer_n; ++j) {
          iree_hal_buffer_view_release(session->kv_k[j]);
          session->kv_k[j] = NULL;
          iree_hal_buffer_view_release(session->kv_v[j]);
          session->kv_v[j] = NULL;
        }
        session->kv_layer_n = 0;
        session->kv_page_count = 0;
        session->kv_allocated_bytes = 0;
        set_error_status(status);
        return 1;
      }
    }
  }
  return 0;
}

size_t dyninfer_iree_session_kv_page_count(
    const dyninfer_iree_session_t* session) {
  return session ? session->kv_page_count : 0;
}

size_t dyninfer_iree_session_kv_allocated_bytes(
    const dyninfer_iree_session_t* session) {
  return session ? session->kv_allocated_bytes : 0;
}

int dyninfer_iree_session_allocator_statistics(
    const dyninfer_iree_session_t* session,
    dyninfer_iree_allocator_statistics_t* out_stats) {
  if (!session || !out_stats || !session->session) {
    return 1;
  }
  memset(out_stats, 0, sizeof(*out_stats));
  iree_hal_allocator_t* allocator =
      iree_runtime_session_device_allocator(session->session);
  if (!allocator) {
    return 1;
  }
  iree_hal_allocator_statistics_t stats;
  memset(&stats, 0, sizeof(stats));
  iree_hal_allocator_query_statistics(allocator, &stats);
#if IREE_STATISTICS_ENABLE
  out_stats->host_bytes_peak = (uint64_t)stats.host_bytes_peak;
  out_stats->host_bytes_allocated = (uint64_t)stats.host_bytes_allocated;
  out_stats->host_bytes_freed = (uint64_t)stats.host_bytes_freed;
  out_stats->device_bytes_peak = (uint64_t)stats.device_bytes_peak;
  out_stats->device_bytes_allocated = (uint64_t)stats.device_bytes_allocated;
  out_stats->device_bytes_freed = (uint64_t)stats.device_bytes_freed;
#else
  (void)stats;
#endif
  return 0;
}
