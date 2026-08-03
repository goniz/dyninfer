use crate::context::Context;
use dyninfer_mlir_sys::bindings::{MlirLocation, mlirLocationUnknownGet};

/// MLIR source location.
#[derive(Clone, Copy)]
pub struct Location {
    raw: MlirLocation,
}

impl Location {
    pub fn unknown(ctx: &Context) -> Self {
        Self {
            raw: unsafe { mlirLocationUnknownGet(ctx.raw()) },
        }
    }

    pub(crate) fn raw(self) -> MlirLocation {
        self.raw
    }
}
