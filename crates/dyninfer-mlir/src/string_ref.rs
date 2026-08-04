use dyninfer_mlir_sys::bindings::MlirStringRef;

pub(crate) fn from_str(s: &str) -> MlirStringRef {
    MlirStringRef {
        data: s.as_ptr().cast(),
        length: s.len(),
    }
}
