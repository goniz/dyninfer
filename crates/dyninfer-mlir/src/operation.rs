use crate::attribute::Attribute;
use crate::context::Context;
use crate::location::Location;
use crate::mlir_err;
use crate::string_ref;
use crate::r#type::Type;
use dyninfer_error::Result;
use dyninfer_mlir_sys::bindings::{
    MlirNamedAttribute, MlirOperation, MlirOperationState, MlirType, mlirIdentifierGet,
    mlirNamedAttributeGet, mlirOperationCreate, mlirOperationCreateParse, mlirOperationDestroy,
    mlirOperationStateAddAttributes, mlirOperationStateAddResults,
    mlirOperationStateEnableResultTypeInference, mlirOperationStateGet,
};

/// Owned MLIR operation (not yet inserted, or detached).
pub struct Operation {
    raw: MlirOperation,
}

impl Operation {
    /// Parse a single top-level operation from assembly.
    pub fn parse(ctx: &Context, asm: &str, source_name: &str) -> Result<Self> {
        ctx.clear_diagnostics();
        let raw = unsafe {
            mlirOperationCreateParse(
                ctx.raw(),
                string_ref::from_str(asm),
                string_ref::from_str(source_name),
            )
        };
        if raw.ptr.is_null() {
            let diags = ctx.take_diagnostics();
            let detail = if diags.is_empty() {
                format!("failed to parse operation:\n{asm}")
            } else {
                diags.join("\n")
            };
            return Err(mlir_err(detail, "mlir-op-parse"));
        }
        Ok(Self { raw })
    }

    pub(crate) fn into_raw(self) -> MlirOperation {
        let raw = self.raw;
        std::mem::forget(self);
        raw
    }
}

impl Drop for Operation {
    fn drop(&mut self) {
        if !self.raw.ptr.is_null() {
            unsafe {
                mlirOperationDestroy(self.raw);
            }
        }
    }
}

/// Melior-style operation state builder.
pub struct OperationBuilder<'c> {
    state: MlirOperationState,
    ctx: &'c Context,
    /// Keep op / attribute name strings alive until [`build`](Self::build).
    name_bufs: Vec<String>,
}

impl<'c> OperationBuilder<'c> {
    pub fn new(name: &str, location: Location, ctx: &'c Context) -> Self {
        let name_owned = name.to_string();
        let mut b = Self {
            state: unsafe {
                mlirOperationStateGet(string_ref::from_str(name_owned.as_str()), location.raw())
            },
            ctx,
            name_bufs: vec![name_owned],
        };
        // Ensure name pointer refers to the owned buffer.
        b.state.name = string_ref::from_str(b.name_bufs[0].as_str());
        b
    }

    pub fn add_results(mut self, results: &[Type]) -> Self {
        let raws: Vec<MlirType> = results.iter().map(|t| t.raw()).collect();
        unsafe {
            mlirOperationStateAddResults(&mut self.state, raws.len() as isize, raws.as_ptr());
        }
        self
    }

    pub fn add_attribute(mut self, name: &str, attr: Attribute) -> Self {
        let name_owned = name.to_string();
        let id =
            unsafe { mlirIdentifierGet(self.ctx.raw(), string_ref::from_str(name_owned.as_str())) };
        let named: MlirNamedAttribute = unsafe { mlirNamedAttributeGet(id, attr.raw()) };
        self.name_bufs.push(name_owned);
        unsafe {
            mlirOperationStateAddAttributes(&mut self.state, 1, &named);
        }
        self
    }

    pub fn enable_result_type_inference(mut self) -> Self {
        unsafe {
            mlirOperationStateEnableResultTypeInference(&mut self.state);
        }
        self
    }

    pub fn build(mut self) -> Result<Operation> {
        self.ctx.clear_diagnostics();
        let raw = unsafe { mlirOperationCreate(&mut self.state) };
        if raw.ptr.is_null() {
            let diags = self.ctx.take_diagnostics();
            let detail = if diags.is_empty() {
                "mlirOperationCreate failed".into()
            } else {
                diags.join("\n")
            };
            return Err(mlir_err(detail, "mlir-op-build"));
        }
        Ok(Operation { raw })
    }
}
