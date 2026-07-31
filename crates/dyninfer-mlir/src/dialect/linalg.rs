//! `linalg` dialect helpers.

use crate::builder::ModuleBuilder;
use dyninfer_error::Result;

pub struct Linalg;

impl Linalg {
    pub fn append_asm(builder: &mut ModuleBuilder, asm: &str) -> Result<()> {
        builder.append_toplevel_asm(asm)
    }
}
