use dyninfer_architecture::ArchitectureRegistry;
use dyninfer_checkpoint::BuiltinCheckpointSupport;
use dyninfer_error::Result;
use dyninfer_kernel_registry::{KernelRegistry, register_builtin_semantic_candidates};
use dyninfer_quantization::QuantizationRegistry;

pub fn default_checkpoint_support() -> BuiltinCheckpointSupport {
    let mut support = BuiltinCheckpointSupport::new();
    dyninfer_checkpoint_safetensors::register(&mut support);
    dyninfer_checkpoint_gguf::register(&mut support);
    support
}

pub fn default_architecture_registry() -> ArchitectureRegistry {
    let mut reg = ArchitectureRegistry::new();
    dyninfer_architecture::register_all(&mut reg);
    reg
}

pub fn default_quantization_registry() -> Result<QuantizationRegistry> {
    let mut registry = QuantizationRegistry::new();
    dyninfer_quantization::register_all(&mut registry)?;
    Ok(registry)
}

pub fn default_kernel_registry(encodings: &QuantizationRegistry) -> Result<KernelRegistry> {
    let mut registry = KernelRegistry::new();
    register_builtin_semantic_candidates(&mut registry)?;
    encodings.register_kernel_candidates(&mut registry)?;
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_registry_construction_is_explicit_and_unique() {
        let encodings = default_quantization_registry().unwrap();
        let kernels = default_kernel_registry(&encodings).unwrap();
        assert_eq!(encodings.definitions().len(), 32);
        assert!(kernels.candidates().len() >= 20);
    }
}
