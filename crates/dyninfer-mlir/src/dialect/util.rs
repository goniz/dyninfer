//! IREE `util` / `stream` helpers (often unregistered at the C API layer).

use crate::builder::ModuleBuilder;
use dyninfer_error::Result;

pub struct Util;

impl Util {
    /// `util.global private @sym = #stream.parameter.named<"weights"::"key"> : ty`
    pub fn global_parameter(
        builder: &mut ModuleBuilder,
        sym: &str,
        key: &str,
        ty_asm: &str,
    ) -> Result<()> {
        let asm = format!(
            "util.global private @{sym} = #stream.parameter.named<\"weights\"::\"{key}\"> : {ty_asm}"
        );
        builder.append_toplevel_asm(&asm)
    }

    /// Mutable zero-init global: `util.global private mutable @sym = dense<0.0> : ty`
    pub fn global_mutable_zero(builder: &mut ModuleBuilder, sym: &str, ty_asm: &str) -> Result<()> {
        let asm = format!("util.global private mutable @{sym} = dense<0.0> : {ty_asm}");
        builder.append_toplevel_asm(&asm)
    }
}
