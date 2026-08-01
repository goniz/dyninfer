use crate::attribute::Attribute;
use crate::context::Context;
use crate::dialect::{Arith, Func, Linalg, Tensor, Util};
use crate::location::Location;
use crate::mlir_err;
use crate::module::Module;
use crate::func_builder::FuncBuilder;
use crate::operation::{Operation, OperationBuilder};
use crate::r#type::Type;
use dyninfer_error::Result;

/// Melior-style module builder: construct IR in-memory, verify, then print.
///
/// Top-level ops are retained as verified assembly fragments and materialized
/// into a live [`Module`] for verify/print (avoids co-owning Module+Context).
pub struct ModuleBuilder {
    ctx: Context,
    /// Top-level op assembly fragments (each already parse-checked on append).
    toplevel: Vec<String>,
    verified_text: Option<String>,
}

/// Verified MLIR module serialized for the IREE compile boundary.
#[derive(Debug, Clone)]
pub struct VerifiedModule {
    pub mlir_text: String,
}

impl ModuleBuilder {
    pub fn new() -> Result<Self> {
        Ok(Self {
            ctx: Context::new()?,
            toplevel: Vec::new(),
            verified_text: None,
        })
    }

    pub fn context(&self) -> &Context {
        &self.ctx
    }

    pub fn unknown_loc(&self) -> Location {
        Location::unknown(&self.ctx)
    }

    pub fn parse_type(&self, asm: &str) -> Result<Type> {
        Type::parse(&self.ctx, asm)
    }

    pub fn parse_attr(&self, asm: &str) -> Result<Attribute> {
        Attribute::parse(&self.ctx, asm)
    }

    pub fn op(&self, name: &str) -> OperationBuilder<'_> {
        OperationBuilder::new(name, self.unknown_loc(), &self.ctx)
    }

    /// Replace module contents by parsing a full module assembly.
    pub fn parse_source(&mut self, source: impl Into<String>) -> Result<()> {
        let source = source.into();
        // Validate, then store body ops as fragments.
        let module = Module::parse(&self.ctx, &source)?;
        let printed = module.print();
        drop(module);
        self.toplevel = extract_module_body(&printed);
        self.verified_text = None;
        Ok(())
    }

    /// Append one or more top-level ops (globals, funcs, …).
    pub fn append_toplevel_asm(&mut self, asm: &str) -> Result<()> {
        let trimmed = asm.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        // Parse-check against the full module so far (symbols may reference
        // earlier globals / funcs).
        self.toplevel.push(trimmed.to_string());
        if let Err(err) = self.materialize() {
            self.toplevel.pop();
            return Err(err);
        }
        self.verified_text = None;
        Ok(())
    }

    pub fn append_operation(&mut self, op: Operation) -> Result<()> {
        // Print the op and append as asm (operations are easier to round-trip as text).
        let mut tmp = Module::empty(&self.ctx)?;
        tmp.append_operation(op)?;
        let printed = tmp.print();
        drop(tmp);
        for frag in extract_module_body(&printed) {
            self.toplevel.push(frag);
        }
        self.verified_text = None;
        Ok(())
    }

    fn materialize(&self) -> Result<Module> {
        let mut src = String::from("module {\n");
        for op in &self.toplevel {
            src.push_str(op);
            if !op.ends_with('\n') {
                src.push('\n');
            }
        }
        src.push_str("}\n");
        Module::parse(&self.ctx, &src)
    }

    // --- dialect facades -------------------------------------------------

    pub fn util_global_parameter(&mut self, sym: &str, key: &str, ty_asm: &str) -> Result<()> {
        Util::global_parameter(self, sym, key, ty_asm)
    }

    pub fn util_global_mutable_zero(&mut self, sym: &str, ty_asm: &str) -> Result<()> {
        Util::global_mutable_zero(self, sym, ty_asm)
    }

    pub fn func_asm(&mut self, asm: &str) -> Result<()> {
        Func::append_asm(self, asm)
    }

    /// Start a typed [`FuncBuilder`] (melior-style).
    pub fn func(&self, name: impl Into<String>) -> FuncBuilder {
        FuncBuilder::new(name)
    }

    pub fn func_private(&self, name: impl Into<String>) -> FuncBuilder {
        FuncBuilder::new(name).private()
    }

    pub fn arith_asm(&mut self, asm: &str) -> Result<()> {
        Arith::append_asm(self, asm)
    }

    pub fn linalg_asm(&mut self, asm: &str) -> Result<()> {
        Linalg::append_asm(self, asm)
    }

    pub fn tensor_asm(&mut self, asm: &str) -> Result<()> {
        Tensor::append_asm(self, asm)
    }

    /// Structural verification via the MLIR verifier.
    pub fn verify(&mut self) -> Result<()> {
        let module = self.materialize()?;
        module.verify(&self.ctx)?;
        self.verified_text = Some(module.print());
        Ok(())
    }

    /// Printed module text (after [`verify`], returns the verified form).
    pub fn print(&self) -> String {
        if let Some(text) = &self.verified_text {
            return text.clone();
        }
        self.materialize()
            .map(|m| m.print())
            .unwrap_or_default()
    }

    /// Verify (if needed) and return the serialized module.
    pub fn finish(mut self) -> Result<VerifiedModule> {
        if self.verified_text.is_none() {
            self.verify()?;
        }
        Ok(VerifiedModule {
            mlir_text: self
                .verified_text
                .take()
                .ok_or_else(|| mlir_err("verify produced no text", "mlir-finish"))?,
        })
    }
}

/// Pull top-level ops out of a printed `module { ... }` (best-effort).
fn extract_module_body(printed: &str) -> Vec<String> {
    let mut body = printed.trim();
    if let Some(rest) = body.strip_prefix("module") {
        body = rest.trim_start();
        // optional attributes then {
        if let Some(idx) = body.find('{') {
            body = &body[idx + 1..];
        }
        if let Some(idx) = body.rfind('}') {
            body = &body[..idx];
        }
    }
    let trimmed = body.trim();
    if trimmed.is_empty() {
        Vec::new()
    } else {
        vec![trimmed.to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_verify_simple_func() {
        let src = r#"
module {
  func.func @add(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.addf %a, %b : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"#;
        let mut b = ModuleBuilder::new().expect("context");
        b.parse_source(src).expect("parse");
        b.verify().expect("verify");
        let out = b.print();
        assert!(out.contains("func.func @add") || out.contains("func @add"));
    }

    #[test]
    fn append_util_and_func() {
        let mut b = ModuleBuilder::new().expect("context");
        b.util_global_parameter("w", "token_embd.weight", "tensor<32x64xf32>")
            .expect("global");
        b.func_asm(
            r#"
func.func @add(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}
"#,
        )
        .expect("func");
        b.verify().expect("verify");
        let out = b.print();
        assert!(out.contains("@w") || out.contains("token_embd"));
        assert!(out.contains("@add"));
    }

    #[test]
    fn reject_invalid_ir() {
        let mut b = ModuleBuilder::new().expect("context");
        let err = b.parse_source("module { not_a_real_op }").unwrap_err();
        let msg = err.to_string();
        assert!(!msg.is_empty());
    }
}
