use dyninfer_architecture::ArchitectureRegistry;
use dyninfer_architecture_llama::LlamaArchitecture;
use dyninfer_checkpoint::BuiltinCheckpointSupport;

pub fn default_checkpoint_support() -> BuiltinCheckpointSupport {
    let mut support = BuiltinCheckpointSupport::new();
    dyninfer_checkpoint_safetensors::register(&mut support);
    dyninfer_checkpoint_gguf::register(&mut support);
    support
}

pub fn default_architecture_registry() -> ArchitectureRegistry {
    let mut reg = ArchitectureRegistry::new();
    reg.register(LlamaArchitecture);
    reg
}
