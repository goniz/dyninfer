#include <stddef.h>

#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"
#include "iree/vm/api.h"

// Builds the single nested-list argument required by an exported
// !util.list<tensor<*xf32>> parameter. The outer list belongs to |call|;
// ownership of the newly-created inner list transfers into it.
static iree_status_t push_page_list(
    iree_runtime_call_t* call, iree_hal_buffer_view_t* const* page_views,
    iree_host_size_t page_count) {
  iree_vm_list_t* pages = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(
      iree_vm_make_ref_type_def(iree_hal_buffer_view_type()), page_count,
      iree_allocator_system(), &pages));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < page_count; ++i) {
    iree_vm_ref_t page_ref =
        iree_hal_buffer_view_retain_ref(page_views[i]);
    status = iree_vm_list_push_ref_move(pages, &page_ref);
    if (!iree_status_is_ok(status)) {
      iree_vm_ref_release(&page_ref);
      break;
    }
  }

  if (iree_status_is_ok(status)) {
    iree_vm_ref_t pages_ref = iree_vm_list_move_ref(pages);
    pages = NULL;
    status = iree_vm_list_push_ref_move(iree_runtime_call_inputs(call),
                                        &pages_ref);
    if (!iree_status_is_ok(status)) iree_vm_ref_release(&pages_ref);
  }
  iree_vm_list_release(pages);
  return status;
}

int main(void) {
  // This target verifies the pinned runtime headers and linker expose the
  // nested VM-list API above; invocation requires a compiled ABI, which the
  // companion MLIR regression test proves is currently unavailable.
  (void)&push_page_list;
  return 0;
}
