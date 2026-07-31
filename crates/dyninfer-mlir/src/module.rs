use crate::context::Context;
use crate::mlir_err;
use crate::operation::Operation;
use crate::string_ref;
use dyninfer_error::Result;
use dyninfer_mlir_sys::bindings::{
    mlirBlockAppendOwnedOperation, mlirBlockGetFirstOperation, mlirModuleCreateParse,
    mlirModuleDestroy, mlirModuleGetBody, mlirModuleGetOperation, mlirOperationGetNextInBlock,
    mlirOperationPrint, mlirOperationRemoveFromParent, mlirOperationVerify, MlirModule,
    MlirOperation, MlirStringRef,
};
use std::ffi::c_void;

/// Owned MLIR module handle (lifetime tied to a [`Context`] that outlives it).
pub struct Module {
    raw: MlirModule,
}

impl Module {
    pub fn empty(ctx: &Context) -> Result<Self> {
        // Prefer parse over CreateEmpty: some IREE builds are happier destroying
        // modules that went through the parser/registration path.
        Self::parse(ctx, "module {\n}\n")
    }

    pub fn parse(ctx: &Context, source: &str) -> Result<Self> {
        ctx.clear_diagnostics();
        let raw = unsafe { mlirModuleCreateParse(ctx.raw(), string_ref::from_str(source)) };
        if raw.ptr.is_null() {
            let diags = ctx.take_diagnostics();
            let detail = if diags.is_empty() {
                "mlirModuleCreateParse failed".into()
            } else {
                diags.join("\n")
            };
            return Err(mlir_err(detail, "mlir-parse"));
        }
        Ok(Self { raw })
    }

    pub fn append_operation(&mut self, op: Operation) -> Result<()> {
        let body = unsafe { mlirModuleGetBody(self.raw) };
        if body.ptr.is_null() {
            return Err(mlir_err("module body is null", "mlir-append"));
        }
        unsafe {
            mlirBlockAppendOwnedOperation(body, op.into_raw());
        }
        Ok(())
    }

    /// Parse `asm` as `module { ... }` and move its body ops into this module.
    pub fn append_asm_ops(&mut self, ctx: &Context, asm: &str) -> Result<()> {
        let trimmed = asm.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        let wrapped = format!("module {{\n{trimmed}\n}}");
        let tmp = Module::parse(ctx, &wrapped)?;
        let tmp_body = unsafe { mlirModuleGetBody(tmp.raw) };
        let dst_body = unsafe { mlirModuleGetBody(self.raw) };
        let mut op = unsafe { mlirBlockGetFirstOperation(tmp_body) };
        while !op.ptr.is_null() {
            let next = unsafe { mlirOperationGetNextInBlock(op) };
            unsafe {
                mlirOperationRemoveFromParent(op);
                mlirBlockAppendOwnedOperation(dst_body, op);
            }
            op = next;
        }
        Ok(())
    }

    pub fn verify(&self, ctx: &Context) -> Result<()> {
        ctx.clear_diagnostics();
        let op = unsafe { mlirModuleGetOperation(self.raw) };
        let ok = unsafe { mlirOperationVerify(op) };
        if !ok {
            let diags = ctx.take_diagnostics();
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

impl Drop for Module {
    fn drop(&mut self) {
        if !self.raw.ptr.is_null() {
            unsafe {
                mlirModuleDestroy(self.raw);
            }
            self.raw = MlirModule {
                ptr: std::ptr::null_mut(),
            };
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

#[allow(dead_code)]
pub(crate) type RawOp = MlirOperation;
