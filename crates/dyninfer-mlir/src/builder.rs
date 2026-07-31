use crate::context::Context;
use crate::module::Module;
use crate::mlir_err;
use dyninfer_error::Result;

/// Melior-style module builder: parse or construct IR, verify, then print.
pub struct ModuleBuilder {
    ctx: Context,
    source: Option<String>,
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
            source: None,
            verified_text: None,
        })
    }

    /// Load MLIR source into the builder (replaces any previous module).
    pub fn parse_source(&mut self, source: impl Into<String>) -> Result<()> {
        let source = source.into();
        // Eager parse to surface syntax errors immediately.
        let module = Module::parse(&self.ctx, &source)?;
        drop(module);
        self.source = Some(source);
        self.verified_text = None;
        Ok(())
    }

    /// Structural verification via the MLIR verifier.
    pub fn verify(&mut self) -> Result<()> {
        let source = self
            .source
            .as_deref()
            .ok_or_else(|| mlir_err("ModuleBuilder has no source", "mlir-verify"))?;
        let module = Module::parse(&self.ctx, source)?;
        module.verify()?;
        self.verified_text = Some(module.print());
        Ok(())
    }

    /// Printed module text (after [`verify`], returns the verified form).
    pub fn print(&self) -> String {
        if let Some(text) = &self.verified_text {
            return text.clone();
        }
        self.source.clone().unwrap_or_default()
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
    fn reject_invalid_ir() {
        let mut b = ModuleBuilder::new().expect("context");
        let err = b.parse_source("module { not_a_real_op }").unwrap_err();
        let msg = err.to_string();
        assert!(!msg.is_empty());
    }
}
