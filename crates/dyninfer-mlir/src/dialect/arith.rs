//! `arith` dialect helpers.

use crate::builder::ModuleBuilder;
use dyninfer_error::Result;

pub struct Arith;

impl Arith {
    /// Append ops that use `arith.*` by parsing a (possibly multi-op) fragment
    /// wrapped as top-level operations into the module.
    pub fn append_asm(builder: &mut ModuleBuilder, asm: &str) -> Result<()> {
        builder.append_toplevel_asm(asm)
    }
}
