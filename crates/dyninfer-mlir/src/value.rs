/// SSA value handle produced by [`crate::FuncBuilder`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Value {
    name: String,
}

impl Value {
    pub(crate) fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Bare SSA name without `%`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `%name` printable form.
    pub fn ssa(&self) -> String {
        format!("%{}", self.name)
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.name)
    }
}
