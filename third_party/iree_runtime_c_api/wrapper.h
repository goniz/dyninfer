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

// Create a persistent session: load VMFB (+ optional SafeTensors scope "weights").
// |device_uri| is an IREE driver name: "hip", "local-task", "vulkan", …
// Pass NULL / "" for local-task. |parameters_path| may be NULL.
// Returns 0 on success; non-zero on failure (see dyninfer_iree_last_error).
int dyninfer_iree_session_create(const char* device_uri, const char* vmfb_path,
                                 const char* parameters_path,
                                 dyninfer_iree_session_t** out_session);

void dyninfer_iree_session_destroy(dyninfer_iree_session_t* session);

// Human-readable error from the most recent failed call (process-wide).
const char* dyninfer_iree_last_error(void);

// Free a buffer returned by invoke helpers (logits).
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

// module.decode — token + absolute position → vocab xf32 (updates KV cache).
int dyninfer_iree_session_invoke_decode(dyninfer_iree_session_t* session,
                                        int64_t token, int64_t pos,
                                        float** out_logits, size_t* out_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DYNINFER_IREE_RUNTIME_WRAPPER_H_
