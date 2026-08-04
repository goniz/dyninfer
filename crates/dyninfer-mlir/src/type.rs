use crate::context::Context;
use crate::mlir_err;
use crate::string_ref;
use dyninfer_error::Result;
use dyninfer_mlir_sys::bindings::{MlirType, mlirTypeParseGet};

/// Parsed MLIR type.
#[derive(Clone, Copy)]
pub struct Type {
    raw: MlirType,
}

impl Type {
    pub fn parse(ctx: &Context, asm: &str) -> Result<Self> {
        ctx.clear_diagnostics();
        let raw = unsafe { mlirTypeParseGet(ctx.raw(), string_ref::from_str(asm)) };
        if raw.ptr.is_null() {
            let diags = ctx.take_diagnostics();
            let detail = if diags.is_empty() {
                format!("failed to parse type: {asm}")
            } else {
                diags.join("\n")
            };
            return Err(mlir_err(detail, "mlir-type"));
        }
        Ok(Self { raw })
    }

    pub(crate) fn raw(self) -> MlirType {
        self.raw
    }
}
