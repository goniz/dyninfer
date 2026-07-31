//! End-to-end bootstrap path tests (inspect → bind → compile → run via IREE).

#[cfg(test)]
mod tests {
    use crate::{CausalLanguageModel, ModelLoader};
    use dyninfer_checkpoint_safetensors::tiny_llama_dense_f32;
    use dyninfer_compiler::{compile_add_smoke, CompileOptions, IreeTools};
    use dyninfer_core::{ArchitectureId, SessionConfig};
    use iree_runtime::{Context, Instance, Module};
    use std::fs;

    fn iree_available() -> bool {
        IreeTools::discover().is_ok()
    }

    #[test]
    fn iree_add_smoke_compiles_and_runs() {
        if !iree_available() {
            eprintln!("skipping: IREE tools not found (need //bazel/iree:tools runfiles)");
            return;
        }
        let vmfb = compile_add_smoke("local-task").expect("iree-compile available");
        assert!(!vmfb.is_empty());
        assert!(!vmfb.starts_with(b"DYNINFER_VMFB_STUB"));

        let instance = Instance::new().unwrap();
        let module = Module::from_vmfb(vmfb).unwrap();
        let ctx = Context::create(instance, module).unwrap();
        let out = ctx
            .invoke_add(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn inspect_bind_compile_run_tiny_llama() {
        if !iree_available() {
            eprintln!("skipping: IREE tools not found (need //bazel/iree:tools runfiles)");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("tiny-llama.safetensors");
        fs::write(&ckpt, tiny_llama_dense_f32()).unwrap();

        let loader = ModelLoader::default();
        let catalog = loader.inspect(&ckpt).unwrap();
        assert_eq!(catalog.container.format_id.as_str(), "safetensors");
        assert!(catalog.parameters.len() >= 12);

        let id = ArchitectureId::new("llama.decoder");
        let (package, _catalog, plan) = loader.bind(&id, &ckpt, &Default::default()).unwrap();
        assert_eq!(package.resolved_config.num_layers().unwrap(), 1);
        assert_eq!(plan.bindings.len(), package.parameter_slots.len());

        let bundle = dir.path().join("model.bundle");
        let paths = loader
            .compile_to_bundle(
                &id,
                &ckpt,
                "cpu",
                &bundle,
                &CompileOptions {
                    mode: "local-jit".into(),
                    ..Default::default()
                },
            )
            .unwrap();
        let vmfb = fs::read(&paths.vmfb).unwrap();
        assert!(!vmfb.starts_with(b"DYNINFER_VMFB_STUB"));
        assert!(vmfb.len() > 64);

        let model = loader.load_bundle(&bundle, &ckpt).unwrap();
        assert_eq!(model.metadata().vocabulary_size, 32);
        let mut session = model.create_session(SessionConfig::default()).unwrap();
        let logits = session.prefill(&[1, 2, 3]).unwrap();
        assert_eq!(logits.values.len(), 32);
        assert_eq!(session.position(), 3);
        let logits = session.decode(1).unwrap();
        assert_eq!(logits.values.len(), 32);
        assert_eq!(session.position(), 4);

        let sum = model
            .context
            .invoke_add(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .unwrap();
        assert_eq!(sum, vec![11.0, 22.0, 33.0, 44.0]);
    }
}
