//! `func` dialect helpers.

use crate::builder::ModuleBuilder;
use dyninfer_error::Result;

pub struct Func;

impl Func {
    /// Append a complete `func.func` from assembly (body included).
    pub fn append_asm(builder: &mut ModuleBuilder, asm: &str) -> Result<()> {
        builder.append_toplevel_asm(asm)
    }
}
