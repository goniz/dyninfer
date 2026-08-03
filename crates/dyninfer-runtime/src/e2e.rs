//! End-to-end Milestone 1 path (inspect → bind → compile → run vs reference).

#[cfg(test)]
mod tests {
    use crate::{CausalLanguageModel, ModelLoader, max_abs_err, tiny_llama_prefill_logits};
    use dyninfer_checkpoint_safetensors::tiny_llama_dense_f32;
    use dyninfer_compiler::{CompileOptions, IreeTools, compile_add_smoke};
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
        assert_eq!(model.manifest.kv_cache.max_sequence_length, 4);
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

        // Prefill([a,b,c,d]) must match prefill([a,b,c]) + decode(d) at pos=3.
        let model_full = loader.load_bundle(&bundle, &ckpt).unwrap();
        let mut s_full = model_full.create_session(SessionConfig::default()).unwrap();
        let full = s_full.prefill(&[1, 2, 3, 0]).unwrap();
        let model_step = loader.load_bundle(&bundle, &ckpt).unwrap();
        let mut s_step = model_step.create_session(SessionConfig::default()).unwrap();
        let _ = s_step.prefill(&[1, 2, 3]).unwrap();
        assert_eq!(s_step.position(), 3);
        let stepped = s_step.decode(0).unwrap();
        let step_err = max_abs_err(&full.values, &stepped.values).unwrap();
        assert!(
            step_err < 1e-3,
            "KV decode diverged from full prefill: max_abs_err={step_err}\nfull={:?}\nstep={:?}",
            &full.values[..8],
            &stepped.values[..8]
        );

        // Non-pad decode: prefill([1,2,3])+decode(7) vs prefill([1,2,3,7]).
        let mut s_full2 = model_full.create_session(SessionConfig::default()).unwrap();
        let full2 = s_full2.prefill(&[1, 2, 3, 7]).unwrap();
        let mut s_step2 = model_step.create_session(SessionConfig::default()).unwrap();
        let _ = s_step2.prefill(&[1, 2, 3]).unwrap();
        let stepped2 = s_step2.decode(7).unwrap();
        let err2 = max_abs_err(&full2.values, &stepped2.values).unwrap();
        eprintln!("tiny_llama non-pad decode max_abs_err={err2}");
        assert!(
            err2 < 1e-3,
            "non-pad KV decode diverged: max_abs_err={err2}\nfull={:?}\nstep={:?}",
            &full2.values[..8],
            &stepped2.values[..8]
        );

        let sum = model
            .open_context()
            .unwrap()
            .invoke_add(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .unwrap();
        assert_eq!(sum, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn tiny_gqa_rope_prefill_decode_parity() {
        if !iree_available() {
            eprintln!("skipping: IREE not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("tiny-gqa-rope.safetensors");
        fs::write(&ckpt, dyninfer_checkpoint_safetensors::tiny_gqa_rope_f32()).unwrap();

        let loader = ModelLoader::default();
        let id = ArchitectureId::new("qwen3.decoder");
        let bundle = dir.path().join("model.bundle");
        loader
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
            .expect("compile tiny GQA+RoPE");

        let model = loader.load_bundle(&bundle, &ckpt).unwrap();
        assert_eq!(model.metadata().vocabulary_size, 48);
        assert_eq!(model.manifest.prefill_window, 4);
        assert_eq!(model.manifest.kv_cache.max_sequence_length, 8);
        assert_eq!(model.manifest.kv_cache.kv_head_count, 2);

        // Full prefill of 4 tokens vs prefill(3)+decode(4th).
        // Use separate bundles/contexts (same as tiny_llama) to avoid shared KV globals.
        let tokens = [1u32, 2, 3, 7];
        let model_full = loader.load_bundle(&bundle, &ckpt).unwrap();
        let mut s_full = model_full.create_session(SessionConfig::default()).unwrap();
        let full = s_full.prefill(&tokens).unwrap();

        let model_step = loader.load_bundle(&bundle, &ckpt).unwrap();
        let mut s_step = model_step.create_session(SessionConfig::default()).unwrap();
        let _ = s_step.prefill(&tokens[..3]).unwrap();
        assert_eq!(s_step.position(), 3);
        let stepped = s_step.decode(tokens[3]).unwrap();

        let err = max_abs_err(&full.values, &stepped.values).unwrap();
        eprintln!("tiny_gqa_rope parity max_abs_err={err}");
        assert!(
            err < 1e-2,
            "GQA+RoPE KV decode diverged from full prefill: max_abs_err={err}\nfull={:?}\nstep={:?}",
            &full.values[..8],
            &stepped.values[..8]
        );
    }

    fn parity_err_with(
        arch: &str,
        bytes: &[u8],
        overrides: &dyninfer_core::MetadataMap,
        label: &str,
    ) -> f32 {
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("m.safetensors");
        fs::write(&ckpt, bytes).unwrap();
        let loader = ModelLoader::default();
        let id = ArchitectureId::new(arch);
        let bundle = dir.path().join("b");
        loader
            .compile_to_bundle_with_overrides(
                &id,
                &ckpt,
                "cpu",
                &bundle,
                &CompileOptions {
                    mode: "local-jit".into(),
                    ..Default::default()
                },
                overrides,
            )
            .unwrap_or_else(|e| panic!("compile {label}: {e}"));
        let model_full = loader.load_bundle(&bundle, &ckpt).unwrap();
        let model_step = loader.load_bundle(&bundle, &ckpt).unwrap();
        let tokens = [1u32, 2, 3, 7];
        let mut s_full = model_full.create_session(SessionConfig::default()).unwrap();
        let full = s_full.prefill(&tokens).unwrap();
        let mut s_step = model_step.create_session(SessionConfig::default()).unwrap();
        let _ = s_step.prefill(&tokens[..3]).unwrap();
        let stepped = s_step.decode(tokens[3]).unwrap();
        let err = max_abs_err(&full.values, &stepped.values).unwrap();
        eprintln!("{label} max_abs_err={err}");
        err
    }

    /// Regression: decode parity must hold when `max_kv > prefill_window` (the
    /// Qwen shape) for MHA and GQA, with and without RoPE/qk_norm.
    ///
    /// Compiles several tiny models — enable with `DYNINFER_KV_PARITY=1`.
    #[test]
    fn decode_parity_max_kv_gt_window() {
        if std::env::var_os("DYNINFER_KV_PARITY").is_none() {
            eprintln!("skipping: set DYNINFER_KV_PARITY=1 for multi-config KV parity");
            return;
        }
        if !iree_available() {
            return;
        }
        let empty = dyninfer_core::MetadataMap::new();
        let mut maxkv8 = dyninfer_core::MetadataMap::new();
        maxkv8.insert("max_kv".into(), serde_json::json!(8));
        maxkv8.insert("prefill_window".into(), serde_json::json!(4));

        let cases: [(&str, &[u8], &dyninfer_core::MetadataMap, &str); 4] = [
            (
                "llama.decoder",
                &tiny_llama_dense_f32(),
                &maxkv8,
                "mha+norope+maxkv8",
            ),
            (
                "llama.decoder",
                &dyninfer_checkpoint_safetensors::tiny_mha_rope_f32(),
                &empty,
                "mha+rope",
            ),
            (
                "llama.decoder",
                &dyninfer_checkpoint_safetensors::tiny_gqa_plain_f32(),
                &empty,
                "gqa+plain",
            ),
            (
                "qwen3.decoder",
                &dyninfer_checkpoint_safetensors::tiny_gqa_rope_f32(),
                &empty,
                "gqa+rope+qknorm",
            ),
        ];
        for (arch, bytes, overrides, label) in cases {
            let err = parity_err_with(arch, bytes, overrides, label);
            assert!(err < 1e-2, "{label} KV decode diverged: max_abs_err={err}");
        }
    }
}
