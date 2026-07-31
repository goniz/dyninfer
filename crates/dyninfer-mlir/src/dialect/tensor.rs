//! `tensor` dialect helpers.

use crate::builder::ModuleBuilder;
use dyninfer_error::Result;

pub struct Tensor;

impl Tensor {
    pub fn append_asm(builder: &mut ModuleBuilder, asm: &str) -> Result<()> {
        builder.append_toplevel_asm(asm)
    }
}
