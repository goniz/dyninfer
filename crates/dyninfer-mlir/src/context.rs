use crate::mlir_err;
use dyninfer_error::Result;
use dyninfer_mlir_sys::bindings::{
    MlirContext, MlirDiagnostic, MlirDiagnosticHandlerID, MlirLogicalResult,
    ireeCompilerInitializeContext, ireeCompilerRegisterDialects,
    mlirContextAttachDiagnosticHandler, mlirContextCreateWithRegistry, mlirContextDestroy,
    mlirContextDetachDiagnosticHandler, mlirContextLoadAllAvailableDialects,
    mlirContextSetAllowUnregisteredDialects, mlirDiagnosticGetSeverity, mlirDiagnosticPrint,
    mlirDialectRegistryCreate, mlirDialectRegistryDestroy,
};
use dyninfer_mlir_sys::ensure_initialized;
use std::cell::RefCell;
use std::ffi::c_void;
use std::rc::Rc;
use std::sync::Mutex;

/// Process-wide lock: MLIR/IREE global state is not concurrency-safe.
static MLIR_LOCK: Mutex<()> = Mutex::new(());

/// Owned MLIR context with IREE dialects registered.
pub struct Context {
    raw: MlirContext,
    _guard: std::sync::MutexGuard<'static, ()>,
    diagnostics: Rc<RefCell<Vec<String>>>,
    diag_id: MlirDiagnosticHandlerID,
}

impl Context {
    /// Create a context with all IREE-known dialects (util, stream, linalg, …).
    pub fn new() -> Result<Self> {
        ensure_initialized().map_err(|e| mlir_err(e.to_string(), "mlir-init"))?;
        let guard = MLIR_LOCK
            .lock()
            .map_err(|_| mlir_err("MLIR lock poisoned", "mlir-init"))?;

        let diagnostics = Rc::new(RefCell::new(Vec::new()));
        let raw = unsafe {
            let registry = mlirDialectRegistryCreate();
            ireeCompilerRegisterDialects(registry);
            let ctx = mlirContextCreateWithRegistry(registry, /*threadingEnabled=*/ false);
            mlirDialectRegistryDestroy(registry);
            if ctx.ptr.is_null() {
                return Err(mlir_err(
                    "mlirContextCreateWithRegistry returned null",
                    "mlir-init",
                ));
            }
            ireeCompilerInitializeContext(ctx);
            // IREE util/stream ops may appear before full dialect packaging.
            mlirContextSetAllowUnregisteredDialects(ctx, true);
            mlirContextLoadAllAvailableDialects(ctx);
            ctx
        };

        let diag_box = Box::new(Rc::clone(&diagnostics));
        let user_data = Box::into_raw(diag_box) as *mut c_void;
        let diag_id = unsafe {
            mlirContextAttachDiagnosticHandler(
                raw,
                Some(diagnostic_handler),
                user_data,
                Some(drop_diagnostic_user_data),
            )
        };

        Ok(Self {
            raw,
            _guard: guard,
            diagnostics,
            diag_id,
        })
    }

    pub(crate) fn raw(&self) -> MlirContext {
        self.raw
    }

    pub(crate) fn take_diagnostics(&self) -> Vec<String> {
        self.diagnostics.borrow_mut().drain(..).collect()
    }

    pub(crate) fn clear_diagnostics(&self) {
        self.diagnostics.borrow_mut().clear();
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            mlirContextDetachDiagnosticHandler(self.raw, self.diag_id);
            mlirContextDestroy(self.raw);
        }
    }
}

unsafe extern "C" fn diagnostic_handler(
    diagnostic: MlirDiagnostic,
    user_data: *mut c_void,
) -> MlirLogicalResult {
    unsafe {
        let sink = &*(user_data as *const Rc<RefCell<Vec<String>>>);
        let mut buf = String::new();
        let severity = mlirDiagnosticGetSeverity(diagnostic) as i32;
        buf.push_str(&format!("severity={severity}: "));
        let print_data = &mut buf as *mut String as *mut c_void;
        mlirDiagnosticPrint(diagnostic, Some(append_string_callback), print_data);
        sink.borrow_mut().push(buf);
        MlirLogicalResult { value: 1 }
    }
}

unsafe extern "C" fn append_string_callback(
    s: dyninfer_mlir_sys::bindings::MlirStringRef,
    user_data: *mut c_void,
) {
    unsafe {
        if s.data.is_null() || s.length == 0 {
            return;
        }
        let slice = std::slice::from_raw_parts(s.data.cast::<u8>(), s.length);
        if let Ok(text) = std::str::from_utf8(slice) {
            let out = &mut *(user_data as *mut String);
            out.push_str(text);
        }
    }
}

unsafe extern "C" fn drop_diagnostic_user_data(user_data: *mut c_void) {
    unsafe {
        if !user_data.is_null() {
            drop(Box::from_raw(user_data as *mut Rc<RefCell<Vec<String>>>));
        }
    }
}
