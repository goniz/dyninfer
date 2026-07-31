/* Umbrella header for bindgen — MLIR C API + IREE dialect registration.
 *
 * Dialect-specific headers that pull generated Passes.capi.h.inc are omitted;
 * dialects are registered via ireeCompilerRegisterDialects.
 */
#include "mlir-c/IR.h"
#include "mlir-c/AffineExpr.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IntegerSet.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"
#include "iree/compiler/mlir_interop.h"
