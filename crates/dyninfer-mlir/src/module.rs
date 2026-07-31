use crate::context::Context;
use crate::mlir_err;
use crate::string_ref;
use dyninfer_error::Result;
use dyninfer_mlir_sys::bindings::{
    mlirModuleCreateParse, mlirModuleDestroy, mlirModuleGetOperation, mlirOperationPrint,
    mlirOperationVerify, MlirModule, MlirStringRef,
};
use std::ffi::c_void;

/// Owned parsed MLIR module.
pub struct Module<'c> {
    raw: MlirModule,
    _ctx: &'c Context,
}

impl<'c> Module<'c> {
    pub fn parse(ctx: &'c Context, source: &str) -> Result<Self> {
        ctx.clear_diagnostics();
        let raw = unsafe {
            mlirModuleCreateParse(ctx.raw(), string_ref::from_str(source))
        };
        if raw.ptr.is_null() {
            let diags = ctx.take_diagnostics();
            let detail = if diags.is_empty() {
                "mlirModuleCreateParse failed".into()
            } else {
                diags.join("\n")
            };
            return Err(mlir_err(detail, "mlir-parse"));
        }
        Ok(Self { raw, _ctx: ctx })
    }

    pub fn verify(&self) -> Result<()> {
        self._ctx.clear_diagnostics();
        let op = unsafe { mlirModuleGetOperation(self.raw) };
        let ok = unsafe { mlirOperationVerify(op) };
        if !ok {
            let diags = self._ctx.take_diagnostics();
            let detail = if diags.is_empty() {
                "mlirOperationVerify failed".into()
            } else {
                diags.join("\n")
            };
            return Err(mlir_err(detail, "mlir-verify"));
        }
        Ok(())
    }

    pub fn print(&self) -> String {
        let mut out = String::new();
        let op = unsafe { mlirModuleGetOperation(self.raw) };
        let user_data = &mut out as *mut String as *mut c_void;
        unsafe {
            mlirOperationPrint(op, Some(append_string_callback), user_data);
        }
        out
    }

}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        unsafe {
            mlirModuleDestroy(self.raw);
        }
    }
}

unsafe extern "C" fn append_string_callback(s: MlirStringRef, user_data: *mut c_void) {
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
