use crate::context::Context;
use crate::mlir_err;
use crate::string_ref;
use dyninfer_error::Result;
use dyninfer_mlir_sys::bindings::{mlirAttributeParseGet, MlirAttribute};

/// Parsed MLIR attribute.
#[derive(Clone, Copy)]
pub struct Attribute {
    raw: MlirAttribute,
}

impl Attribute {
    pub fn parse(ctx: &Context, asm: &str) -> Result<Self> {
        ctx.clear_diagnostics();
        let raw = unsafe { mlirAttributeParseGet(ctx.raw(), string_ref::from_str(asm)) };
        if raw.ptr.is_null() {
            let diags = ctx.take_diagnostics();
            let detail = if diags.is_empty() {
                format!("failed to parse attribute: {asm}")
            } else {
                diags.join("\n")
            };
            return Err(mlir_err(detail, "mlir-attr"));
        }
        Ok(Self { raw })
    }

    pub(crate) fn raw(self) -> MlirAttribute {
        self.raw
    }
}
