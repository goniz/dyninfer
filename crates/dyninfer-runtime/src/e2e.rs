//! End-to-end Milestone 1 path (inspect → bind → compile → run vs reference).

#[cfg(test)]
mod tests {
    use crate::{max_abs_err, tiny_llama_prefill_logits, CausalLanguageModel, ModelLoader};
    use dyninfer_checkpoint_safetensors::tiny_llama_dense_f32;
    use dyninfer_compiler::{compile_add_smoke, CompileOptions, IreeTools};
    use dyninfer_core::{ArchitectureId, SessionConfig, TargetProfile};
    use iree_runtime::{Context, Instance, Module};
    use std::fs;

    fn iree_available() -> bool {
        IreeTools::discover().is_ok()
            || std::env::var_os("RUNFILES_DIR").is_some()
            || std::env::var_os("TEST_SRCDIR").is_some()
    }

    #[test]
    fn iree_add_smoke_compiles_and_runs() {
        if !iree_available() {
            eprintln!("skipping: IREE not available");
            return;
        }
        let target = TargetProfile::llvm_cpu_host();
        let vmfb = compile_add_smoke(&target).expect("iree compile");
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
    fn tiny_llama_matches_reference_logits() {
        if !iree_available() {
            eprintln!("skipping: IREE not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("tiny-llama.safetensors");
        fs::write(&ckpt, tiny_llama_dense_f32()).unwrap();

        let loader = ModelLoader::default();
        let id = ArchitectureId::new("llama.decoder");
        let (package, _catalog, plan) = loader.bind(&id, &ckpt, &Default::default()).unwrap();
        assert_eq!(package.resolved_config.num_layers().unwrap(), 1);
        assert_eq!(
            plan.bindings.len() + plan.unresolved_optional_slots.len(),
            package.parameter_slots.len()
        );

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
        assert!(vmfb.len() > 1024, "expected a real specialized VMFB");

        let model = loader.load_bundle(&bundle, &ckpt).unwrap();
        assert_eq!(model.metadata().vocabulary_size, 32);
        let mut session = model.create_session(SessionConfig::default()).unwrap();

        let tokens = [1u32, 2, 3, 0];
        let logits = session.prefill(&tokens).unwrap();
        assert_eq!(logits.values.len(), 32);
        assert_eq!(session.position(), 4);

        let reference = tiny_llama_prefill_logits(&tokens).unwrap();
        let err = max_abs_err(&logits.values, &reference).unwrap();
        assert!(
            err < 1e-3,
            "logits diverged from reference: max_abs_err={err}\ngot={:?}\nref={:?}",
            &logits.values[..8],
            &reference[..8]
        );

        let decode_logits = session.decode(1).unwrap();
        assert_eq!(decode_logits.values.len(), 32);
        assert_eq!(session.position(), 5);

        let sum = model
            .context
            .invoke_add(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .unwrap();
        assert_eq!(sum, vec![11.0, 22.0, 33.0, 44.0]);
    }
}
