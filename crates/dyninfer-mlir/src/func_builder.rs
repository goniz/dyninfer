//! Value-based function body builder (melior-style).
//!
//! Call sites build SSA values through helpers; the finished `func.func` is
//! appended to a [`crate::ModuleBuilder`] (parse/verify). Complex regions
//! (e.g. `linalg.generic`) can still use [`FuncBuilder::op_asm`].

use crate::builder::ModuleBuilder;
use crate::value::Value;
use dyninfer_error::Result;

/// Builds one `func.func` with named SSA values.
pub struct FuncBuilder {
    name: String,
    private: bool,
    args: Vec<(String, String)>,
    results: Vec<String>,
    body: Vec<String>,
    next_id: u32,
    attrs: Option<String>,
}

impl FuncBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            private: false,
            args: Vec::new(),
            results: Vec::new(),
            body: Vec::new(),
            next_id: 0,
            attrs: None,
        }
    }

    pub fn private(mut self) -> Self {
        self.private = true;
        self
    }

    pub fn attrs(&mut self, attrs: impl Into<String>) -> &mut Self {
        self.attrs = Some(attrs.into());
        self
    }

    pub fn arg(&mut self, name: impl Into<String>, ty: impl Into<String>) -> Value {
        self.arg_attrs(name, ty, "")
    }

    /// Like [`Self::arg`], with optional MLIR attribute dict (e.g. `{iree.abi.output = 1 : index}`).
    pub fn arg_attrs(
        &mut self,
        name: impl Into<String>,
        ty: impl Into<String>,
        attrs: impl AsRef<str>,
    ) -> Value {
        let name = name.into();
        let attrs = attrs.as_ref().trim();
        let ty = if attrs.is_empty() {
            ty.into()
        } else {
            format!("{} {attrs}", ty.into())
        };
        self.args.push((name.clone(), ty));
        Value::new(name)
    }

    pub fn returns(mut self, tys: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.results.extend(tys.into_iter().map(Into::into));
        self
    }

    pub fn result_ty(&mut self, ty: impl Into<String>) -> &mut Self {
        self.results.push(ty.into());
        self
    }

    fn alloc(&mut self, hint: &str) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        // Avoid clashing with explicit binds like %c0 / %x.
        Value::new(format!("_{hint}{id}"))
    }

    /// Fixed SSA name (must be unique in this function).
    pub fn bind(&mut self, name: impl Into<String>) -> Value {
        Value::new(name.into())
    }

    fn push(&mut self, line: String) {
        self.body.push(line);
    }

    /// Multi-line / region-bearing op escape hatch.
    pub fn op_asm(&mut self, asm: impl AsRef<str>) -> &mut Self {
        let trimmed = asm.as_ref().trim_end();
        if !trimmed.is_empty() {
            self.body.push(trimmed.to_string());
        }
        self
    }

    // --- util ------------------------------------------------------------

    pub fn global_load(&mut self, sym: &str, ty: &str) -> Value {
        let v = self.alloc("ld");
        self.push(format!("  {v} = util.global.load @{sym} : {ty}"));
        v
    }

    pub fn global_load_as(&mut self, ssa: &str, sym: &str, ty: &str) -> Value {
        let v = Value::new(ssa);
        self.push(format!("  {v} = util.global.load @{sym} : {ty}"));
        v
    }

    pub fn global_store(&mut self, sym: &str, val: &Value, ty: &str) {
        self.push(format!("  util.global.store {val}, @{sym} : {ty}"));
    }

    // --- arith / math ----------------------------------------------------

    pub fn constant_index(&mut self, value: i64) -> Value {
        let v = self.alloc("c");
        self.push(format!("  {v} = arith.constant {value} : index"));
        v
    }

    pub fn constant_index_as(&mut self, ssa: &str, value: i64) -> Value {
        let v = Value::new(ssa);
        self.push(format!("  {v} = arith.constant {value} : index"));
        v
    }

    pub fn constant_f32(&mut self, lit: &str) -> Value {
        let v = self.alloc("cst");
        self.push(format!("  {v} = arith.constant {lit} : f32"));
        v
    }

    pub fn constant_f32_as(&mut self, ssa: &str, lit: &str) -> Value {
        let v = Value::new(ssa);
        self.push(format!("  {v} = arith.constant {lit} : f32"));
        v
    }

    pub fn constant_typed(&mut self, lit: &str, ty: &str) -> Value {
        let v = self.alloc("cst");
        self.push(format!("  {v} = arith.constant {lit} : {ty}"));
        v
    }

    pub fn extf(&mut self, src: &Value, from_ty: &str, to_ty: &str) -> Value {
        let v = self.alloc("ext");
        self.push(format!("  {v} = arith.extf {src} : {from_ty} to {to_ty}"));
        v
    }

    pub fn extf_as(&mut self, ssa: &str, src: &Value, from_ty: &str, to_ty: &str) -> Value {
        let v = Value::new(ssa);
        self.push(format!("  {v} = arith.extf {src} : {from_ty} to {to_ty}"));
        v
    }

    pub fn index_cast(&mut self, src: &Value, from_ty: &str, to_ty: &str) -> Value {
        let v = self.alloc("ic");
        self.push(format!(
            "  {v} = arith.index_cast {src} : {from_ty} to {to_ty}"
        ));
        v
    }

    pub fn index_cast_as(&mut self, ssa: &str, src: &Value, from_ty: &str, to_ty: &str) -> Value {
        let v = Value::new(ssa);
        self.push(format!(
            "  {v} = arith.index_cast {src} : {from_ty} to {to_ty}"
        ));
        v
    }

    pub fn addf(&mut self, a: &Value, b: &Value, ty: &str) -> Value {
        let v = self.alloc("add");
        self.push(format!("  {v} = arith.addf {a}, {b} : {ty}"));
        v
    }

    pub fn mulf(&mut self, a: &Value, b: &Value, ty: &str) -> Value {
        let v = self.alloc("mul");
        self.push(format!("  {v} = arith.mulf {a}, {b} : {ty}"));
        v
    }

    pub fn divf(&mut self, a: &Value, b: &Value, ty: &str) -> Value {
        let v = self.alloc("div");
        self.push(format!("  {v} = arith.divf {a}, {b} : {ty}"));
        v
    }

    pub fn subf(&mut self, a: &Value, b: &Value, ty: &str) -> Value {
        let v = self.alloc("sub");
        self.push(format!("  {v} = arith.subf {a}, {b} : {ty}"));
        v
    }

    pub fn negf(&mut self, a: &Value, ty: &str) -> Value {
        let v = self.alloc("neg");
        self.push(format!("  {v} = arith.negf {a} : {ty}"));
        v
    }

    pub fn addi(&mut self, a: &Value, b: &Value, ty: &str) -> Value {
        let v = self.alloc("addi");
        self.push(format!("  {v} = arith.addi {a}, {b} : {ty}"));
        v
    }

    pub fn remui(&mut self, a: &Value, b: &Value, ty: &str) -> Value {
        let v = self.alloc("rem");
        self.push(format!("  {v} = arith.remui {a}, {b} : {ty}"));
        v
    }

    pub fn cmpi(&mut self, pred: &str, a: &Value, b: &Value, ty: &str) -> Value {
        let v = self.alloc("cmp");
        self.push(format!("  {v} = arith.cmpi {pred}, {a}, {b} : {ty}"));
        v
    }

    pub fn select(&mut self, cond: &Value, a: &Value, b: &Value, ty: &str) -> Value {
        let v = self.alloc("sel");
        self.push(format!("  {v} = arith.select {cond}, {a}, {b} : {ty}"));
        v
    }

    pub fn math_sqrt(&mut self, a: &Value, ty: &str) -> Value {
        let v = self.alloc("sqrt");
        self.push(format!("  {v} = math.sqrt {a} : {ty}"));
        v
    }

    pub fn math_exp(&mut self, a: &Value, ty: &str) -> Value {
        let v = self.alloc("exp");
        self.push(format!("  {v} = math.exp {a} : {ty}"));
        v
    }

    // --- tensor ----------------------------------------------------------

    pub fn tensor_empty(&mut self, ty: &str) -> Value {
        let v = self.alloc("empty");
        self.push(format!("  {v} = tensor.empty() : {ty}"));
        v
    }

    pub fn tensor_empty_as(&mut self, ssa: &str, ty: &str) -> Value {
        let v = Value::new(ssa);
        self.push(format!("  {v} = tensor.empty() : {ty}"));
        v
    }

    pub fn tensor_extract(&mut self, src: &Value, indices: &str, ty: &str) -> Value {
        let v = self.alloc("ex");
        self.push(format!("  {v} = tensor.extract {src}[{indices}] : {ty}"));
        v
    }

    pub fn tensor_extract_as(&mut self, ssa: &str, src: &Value, indices: &str, ty: &str) -> Value {
        let v = Value::new(ssa);
        self.push(format!("  {v} = tensor.extract {src}[{indices}] : {ty}"));
        v
    }

    pub fn tensor_extract_slice(
        &mut self,
        src: &Value,
        offsets: &str,
        sizes: &str,
        strides: &str,
        from_ty: &str,
        to_ty: &str,
    ) -> Value {
        let v = self.alloc("slice");
        self.push(format!(
            "  {v} = tensor.extract_slice {src}[{offsets}] [{sizes}] [{strides}] : {from_ty} to {to_ty}"
        ));
        v
    }

    pub fn tensor_insert_slice(
        &mut self,
        src: &Value,
        dst: &Value,
        offsets: &str,
        sizes: &str,
        strides: &str,
        src_ty: &str,
        dst_ty: &str,
    ) -> Value {
        let v = self.alloc("ins");
        self.push(format!(
            "  {v} = tensor.insert_slice {src} into {dst}[{offsets}] [{sizes}] [{strides}] : {src_ty} into {dst_ty}"
        ));
        v
    }

    pub fn expand_shape(
        &mut self,
        src: &Value,
        reassoc: &str,
        from_ty: &str,
        to_ty: &str,
    ) -> Value {
        let v = self.alloc("exp");
        self.push(format!(
            "  {v} = tensor.expand_shape {src} {reassoc} : {from_ty} into {to_ty}"
        ));
        v
    }

    pub fn collapse_shape(
        &mut self,
        src: &Value,
        reassoc: &str,
        from_ty: &str,
        to_ty: &str,
    ) -> Value {
        let v = self.alloc("col");
        self.push(format!(
            "  {v} = tensor.collapse_shape {src} {reassoc} : {from_ty} into {to_ty}"
        ));
        v
    }

    // --- linalg ----------------------------------------------------------

    pub fn linalg_fill(&mut self, scalar: &Value, init: &Value, ty: &str) -> Value {
        let v = self.alloc("fill");
        self.push(format!(
            "  {v} = linalg.fill ins({scalar} : f32) outs({init} : {ty}) -> {ty}"
        ));
        v
    }

    pub fn linalg_matmul(
        &mut self,
        a: &Value,
        b: &Value,
        c: &Value,
        a_ty: &str,
        b_ty: &str,
        c_ty: &str,
    ) -> Value {
        let v = self.alloc("mm");
        self.push(format!(
            "  {v} = linalg.matmul ins({a}, {b} : {a_ty}, {b_ty}) outs({c} : {c_ty}) -> {c_ty}"
        ));
        v
    }

    pub fn linalg_batch_matmul(
        &mut self,
        a: &Value,
        b: &Value,
        c: &Value,
        a_ty: &str,
        b_ty: &str,
        c_ty: &str,
    ) -> Value {
        let v = self.alloc("bmm");
        self.push(format!(
            "  {v} = linalg.batch_matmul ins({a}, {b} : {a_ty}, {b_ty}) outs({c} : {c_ty}) -> {c_ty}"
        ));
        v
    }

    pub fn linalg_transpose(
        &mut self,
        src: &Value,
        init: &Value,
        perm: &str,
        from_ty: &str,
        to_ty: &str,
    ) -> Value {
        let v = self.alloc("tr");
        self.push(format!(
            "  {v} = linalg.transpose ins({src} : {from_ty}) outs({init} : {to_ty}) permutation = {perm}"
        ));
        v
    }

    pub fn linalg_softmax(&mut self, src: &Value, init: &Value, dimension: u32, ty: &str) -> Value {
        let v = self.alloc("sm");
        self.push(format!(
            "  {v} = linalg.softmax dimension({dimension}) ins({src} : {ty}) outs({init} : {ty}) -> {ty}"
        ));
        v
    }

    // --- func ------------------------------------------------------------

    pub fn call_ty(&mut self, callee: &str, args: &[&Value], func_ty: &str) -> Value {
        let v = self.alloc("call");
        let args_s = args.iter().map(|a| a.ssa()).collect::<Vec<_>>().join(", ");
        self.push(format!("  {v} = func.call @{callee}({args_s}) : {func_ty}"));
        v
    }

    pub fn call_ty_as(&mut self, ssa: &str, callee: &str, args: &[&Value], func_ty: &str) -> Value {
        let v = Value::new(ssa);
        let args_s = args.iter().map(|a| a.ssa()).collect::<Vec<_>>().join(", ");
        self.push(format!("  {v} = func.call @{callee}({args_s}) : {func_ty}"));
        v
    }

    pub fn ret_ty(&mut self, vals: &[&Value], tys: &str) {
        let vs = vals.iter().map(|v| v.ssa()).collect::<Vec<_>>().join(", ");
        self.push(format!("  return {vs} : {tys}"));
    }

    /// Render complete `func.func` assembly.
    pub fn to_asm(&self) -> String {
        let priv_kw = if self.private { " private" } else { "" };
        let args = self
            .args
            .iter()
            .map(|(n, t)| format!("%{n}: {t}"))
            .collect::<Vec<_>>()
            .join(", ");
        let rets = if self.results.is_empty() {
            String::new()
        } else if self.results.len() == 1 {
            format!(" -> {}", self.results[0])
        } else {
            format!(" -> ({})", self.results.join(", "))
        };
        let attr = self
            .attrs
            .as_ref()
            .map(|a| format!(" attributes {a}"))
            .unwrap_or_default();
        let mut out = format!("func.func{priv_kw} @{}({}){rets}{attr} {{\n", self.name, args);
        for line in &self.body {
            out.push_str(line);
            if !line.ends_with('\n') {
                out.push('\n');
            }
        }
        out.push_str("}\n");
        out
    }

    /// Append this function into `module` (parse-checked).
    pub fn finish(self, module: &mut ModuleBuilder) -> Result<()> {
        module.append_toplevel_asm(&self.to_asm())
    }
}
