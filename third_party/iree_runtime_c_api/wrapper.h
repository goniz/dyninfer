// Thin dyninfer-facing wrapper over IREE's C runtime (session + parameters).
// Bindgen target; implementation links //bazel/iree:runtime_cc (+ HAL/IO).

#ifndef DYNINFER_IREE_RUNTIME_WRAPPER_H_
#define DYNINFER_IREE_RUNTIME_WRAPPER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dyninfer_iree_session_t dyninfer_iree_session_t;

// One original checkpoint file opened once and retained by the IREE parameter
// provider for the session lifetime.
typedef struct dyninfer_iree_parameter_file_t {
  const char* path;
} dyninfer_iree_parameter_file_t;

// One stable external parameter key backed by a contiguous range in an
// original checkpoint file. No container parser or host staging is involved.
typedef struct dyninfer_iree_file_param_t {
  const char* key;
  size_t source_file_index;
  uint64_t offset;
  uint64_t length;
} dyninfer_iree_file_param_t;

// Create a persistent session without external parameters (smoke modules).
// |device_uri| is an IREE driver name: "hip", "local-task", …
// Pass NULL / "" for local-task.
// For HIP/ROCm, sets hipDeviceScheduleBlockingSync (unless
// DYNINFER_HIP_ALLOW_BUSY_WAIT=1) so host waits sleep instead of spinning.
// Returns 0 on success; non-zero on failure (see dyninfer_iree_last_error).
int dyninfer_iree_session_create(const char* device_uri, const char* vmfb_path,
                                 dyninfer_iree_session_t** out_session);

// Creates a session with an explicit descriptor index over original checkpoint
// files. Each file is opened once; entries may reference different files and
// arbitrary container-independent byte ranges under scope "weights".
int dyninfer_iree_session_create_with_file_params(
    const char* device_uri, const char* vmfb_path,
    const dyninfer_iree_parameter_file_t* files, size_t file_count,
    const dyninfer_iree_file_param_t* params, size_t param_count,
    dyninfer_iree_session_t** out_session);

// ABI v7: appends independently compiled prefill and decode modules to one
// session so both share the same device and runtime-owned packed KV pool.
int dyninfer_iree_session_create_modules_with_file_params(
    const char* device_uri, const char* prefill_vmfb_path,
    const char* decode_vmfb_path, const dyninfer_iree_parameter_file_t* files,
    size_t file_count, const dyninfer_iree_file_param_t* params,
    size_t param_count, dyninfer_iree_session_t** out_session);

void dyninfer_iree_session_destroy(dyninfer_iree_session_t* session);

// Human-readable error from the most recent failed call (process-wide).
const char* dyninfer_iree_last_error(void);

// Free a buffer returned by invoke helpers (logits).
// No-op when the pointer is session scratch (current ABI); safe to call always.
void dyninfer_iree_free(void* p);

// module.add — smoke: two 4xf32 vectors → 4xf32.
int dyninfer_iree_session_invoke_add(dyninfer_iree_session_t* session,
                                     const float a[4], const float b[4],
                                     float** out_logits, size_t* out_count);

// module.prefill — tokens[token_count] + last index → vocab xf32.
int dyninfer_iree_session_invoke_prefill(dyninfer_iree_session_t* session,
                                         const int64_t* tokens,
                                         size_t token_count, int64_t last,
                                         float** out_logits, size_t* out_count);

// module.decode — token + pos + attn_bias[max_kv] → vocab xf32 (updates KV).
int dyninfer_iree_session_invoke_decode(dyninfer_iree_session_t* session,
                                        int64_t token, int64_t pos,
                                        const float* attn_bias, size_t bias_len,
                                        float** out_logits, size_t* out_count);

// Paged KV ABI v7: one packed pool tensor
// [num_pages * layers, 2, page_size, kv_heads, head_dim]. Prefill/decode chunk
// entrypoints take caller-owned logits via iree.abi.output. The packed
// kv_pool is a read-only input; page writes come back as a small delta
// tensor that the runtime scatters with D2D copies (HIP cannot clone the
// imported pool).
int dyninfer_iree_session_configure_paged_kv(
    dyninfer_iree_session_t* session, size_t layer_count, size_t page_size,
    size_t kv_head_count, size_t head_dim, size_t chunk_size,
    size_t vocab_size);
int dyninfer_iree_session_ensure_kv_pages(dyninfer_iree_session_t* session,
                                          size_t page_count);
int dyninfer_iree_session_invoke_paged_chunk(
    dyninfer_iree_session_t* session, const int64_t* tokens,
    size_t token_count, int64_t last, int64_t start_pos, float** out_logits,
    size_t* out_count, int64_t* out_token, int want_logits);
int dyninfer_iree_session_reset_paged_kv(dyninfer_iree_session_t* session);
size_t dyninfer_iree_session_kv_page_count(
    const dyninfer_iree_session_t* session);
size_t dyninfer_iree_session_kv_allocated_bytes(
    const dyninfer_iree_session_t* session);

// Snapshot of IREE HAL allocator statistics (IREE_STATISTICS_ENABLE).
// When statistics are compiled out, all fields are zeroed.
typedef struct dyninfer_iree_allocator_statistics_t {
  uint64_t host_bytes_peak;
  uint64_t host_bytes_allocated;
  uint64_t host_bytes_freed;
  uint64_t device_bytes_peak;
  uint64_t device_bytes_allocated;
  uint64_t device_bytes_freed;
} dyninfer_iree_allocator_statistics_t;

// Query aggregate HAL allocator statistics for the session device.
// Returns 0 on success; non-zero if session is null or has no allocator.
int dyninfer_iree_session_allocator_statistics(
    const dyninfer_iree_session_t* session,
    dyninfer_iree_allocator_statistics_t* out_stats);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DYNINFER_IREE_RUNTIME_WRAPPER_H_
