//! End-to-end Milestone 1 path (inspect → bind → compile → run vs reference).

#[cfg(test)]
mod tests {
    use crate::{
        CausalLanguageModel, ModelLoader, max_abs_err, tiny_llama_gguf_q4_0_prefill_logits,
        tiny_llama_mlx_u4_prefill_logits, tiny_llama_prefill_logits,
    };
    use dyninfer_cache::ArtifactCache;
    use dyninfer_checkpoint_safetensors::{tiny_llama_dense_f32, tiny_llama_mlx_affine_u4};
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
            package.parameter_slots().len()
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
        assert!(!model.manifest.derived_parameters_required);
        assert_eq!(
            model.manifest.parameter_components.len(),
            model.binding.bindings.len()
        );
        assert!(!bundle.join("parameters").exists());
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
    fn schema_identical_values_reuse_vmfb_with_distinct_direct_parameters() {
        if !iree_available() {
            eprintln!("skipping: IREE not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let checkpoint_a = dir.path().join("values-a.safetensors");
        let checkpoint_b = dir.path().join("values-b.safetensors");
        let values_a = tiny_llama_dense_f32();
        let mut values_b = values_a.clone();
        let header_len = u64::from_le_bytes(values_b[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&values_b[8..8 + header_len]).unwrap();
        let offsets = header["output.weight"]["data_offsets"].as_array().unwrap();
        let start = 8 + header_len + offsets[0].as_u64().unwrap() as usize;
        let end = 8 + header_len + offsets[1].as_u64().unwrap() as usize;
        for value in values_b[start..end].chunks_exact_mut(4) {
            value.copy_from_slice(&0.0f32.to_le_bytes());
        }
        fs::write(&checkpoint_a, values_a).unwrap();
        fs::write(&checkpoint_b, values_b).unwrap();

        let cache = ArtifactCache::open(dir.path().join("cache")).unwrap();
        let loader = ModelLoader::default().with_cache(cache.clone());
        let architecture = ArchitectureId::new("llama.decoder");
        let bundle_a = dir.path().join("bundle-a");
        let bundle_b = dir.path().join("bundle-b");
        let options = CompileOptions {
            mode: "local-jit".into(),
            ..Default::default()
        };
        loader
            .compile_to_bundle(&architecture, &checkpoint_a, "cpu", &bundle_a, &options)
            .unwrap();
        loader
            .compile_to_bundle(&architecture, &checkpoint_b, "cpu", &bundle_b, &options)
            .unwrap();
        assert_eq!(cache.list().unwrap().len(), 1);
        assert_eq!(
            fs::read(bundle_a.join("executables/model.vmfb")).unwrap(),
            fs::read(bundle_b.join("executables/model.vmfb")).unwrap()
        );

        let tokens = [1u32, 2, 3, 0];
        let model_a = loader.load_bundle(&bundle_a, &checkpoint_a).unwrap();
        let model_b = loader.load_bundle(&bundle_b, &checkpoint_b).unwrap();
        let logits_a = model_a
            .create_session(SessionConfig::default())
            .unwrap()
            .prefill(&tokens)
            .unwrap();
        let logits_b = model_b
            .create_session(SessionConfig::default())
            .unwrap()
            .prefill(&tokens)
            .unwrap();
        assert_ne!(logits_a.values, logits_b.values);
    }

    #[test]
    fn bundle_with_different_target_fingerprint_is_rejected_before_invocation() {
        if !iree_available() {
            eprintln!("skipping: IREE not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("tiny.safetensors");
        fs::write(&checkpoint, tiny_llama_dense_f32()).unwrap();
        let bundle = dir.path().join("bundle");
        let loader = ModelLoader::default();
        loader
            .compile_to_bundle(
                &ArchitectureId::new("llama.decoder"),
                &checkpoint,
                "cpu",
                &bundle,
                &CompileOptions {
                    mode: "local-jit".into(),
                    ..Default::default()
                },
            )
            .unwrap();
        let manifest_path = bundle.join("manifest.json");
        let mut manifest: dyninfer_core::ExecutableManifest =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest.target.capability_fingerprint =
            dyninfer_core::Digest::from_bytes(b"different-local-target");
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let error = match loader.load_bundle(&bundle, &checkpoint) {
            Ok(_) => panic!("mismatched target fingerprint was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.code(), "E_CACHE");
        assert!(error.to_string().contains("target fingerprint"));
    }

    #[test]
    fn tiny_mlx_u4_matches_independent_dequantized_reference() {
        if !iree_available() {
            eprintln!("skipping: IREE not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("tiny-mlx-u4.safetensors");
        fs::write(&checkpoint, tiny_llama_mlx_affine_u4()).unwrap();

        let loader = ModelLoader::default();
        let architecture = ArchitectureId::new("llama.decoder");
        let coverage = loader
            .kernel_coverage(&architecture, &checkpoint, "cpu", &Default::default())
            .unwrap();
        coverage.require_complete().unwrap();
        let quantized: Vec<_> = coverage
            .operations
            .iter()
            .filter(|operation| {
                operation
                    .encoding
                    .as_ref()
                    .is_some_and(|encoding| encoding.id.as_str() == "mlx.affine.u4")
            })
            .collect();
        assert_eq!(quantized.len(), 18);
        assert!(quantized.iter().all(|operation| {
            operation.selected.is_some()
                && operation.validation_error.is_none()
                && operation.checkpoint_component_keys.len() == 3
        }));

        let bundle = dir.path().join("model.bundle");
        loader
            .compile_to_bundle(
                &architecture,
                &checkpoint,
                "cpu",
                &bundle,
                &CompileOptions {
                    mode: "local-jit".into(),
                    ..Default::default()
                },
            )
            .expect("compile direct MLX affine U4 fixture");
        let model = loader.load_bundle(&bundle, &checkpoint).unwrap();
        assert!(!model.manifest.derived_parameters_required);
        assert_eq!(model.manifest.parameter_components.len(), 30);
        assert!(!bundle.join("parameters").exists());

        let tokens = [1u32, 2, 3, 0];
        let mut session = model.create_session(SessionConfig::default()).unwrap();
        let logits = session.prefill(&tokens).unwrap();
        let reference = tiny_llama_mlx_u4_prefill_logits(&tokens).unwrap();
        let error = max_abs_err(&logits.values, &reference).unwrap();
        assert!(
            error < 1.0e-3,
            "MLX affine U4 logits diverged: max_abs_err={error}\ngot={:?}\nref={:?}",
            &logits.values[..8],
            &reference[..8]
        );
    }

    #[test]
    fn gguf_q4_provider_keeps_original_packed_ranges() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("tiny-q4.gguf");
        fs::write(
            &checkpoint,
            dyninfer_checkpoint_gguf::tiny_llama_q4_0().unwrap(),
        )
        .unwrap();
        let loader = ModelLoader::default();
        let id = ArchitectureId::new("llama.decoder");
        let (_package, catalog, binding) =
            loader.bind(&id, &checkpoint, &Default::default()).unwrap();
        let provider =
            dyninfer_checkpoint::build_runtime_provider_plan(&catalog, &binding).unwrap();

        let q4_parameter = catalog
            .parameters
            .iter()
            .find(|parameter| {
                matches!(
                    &parameter.encoding,
                    dyninfer_core::PhysicalEncoding::BlockQuantized { codec, .. }
                        if codec.as_str() == "gguf.q4_0"
                )
            })
            .unwrap();
        let component = &q4_parameter.components[0];
        let range = &component.byte_ranges[0];
        let descriptor = provider
            .parameters
            .iter()
            .find(|descriptor| descriptor.aliases.contains(&component.key))
            .unwrap();
        assert_eq!(descriptor.offset, range.offset);
        assert_eq!(descriptor.length, range.length);
        assert_eq!(descriptor.source_file_index, 0);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);

        let entries = provider
            .parameters
            .iter()
            .map(|descriptor| iree_runtime::FileParameterDescriptor {
                key: descriptor.external_key.clone(),
                source_file_index: descriptor.source_file_index as usize,
                offset: descriptor.offset,
                length: descriptor.length,
            })
            .collect();
        let storage =
            iree_runtime::FileParameterStorage::new(provider.file_paths.clone(), entries).unwrap();
        assert_eq!(storage.file_count(), 1);
        assert_eq!(storage.entry_count(), provider.parameters.len());
    }

    #[test]
    fn tiny_gguf_q4_0_matches_independent_dequantized_reference() {
        if !iree_available() {
            eprintln!("skipping: IREE not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("tiny-q4.gguf");
        fs::write(
            &checkpoint,
            dyninfer_checkpoint_gguf::tiny_llama_q4_0().unwrap(),
        )
        .unwrap();
        let loader = ModelLoader::default();
        let architecture = ArchitectureId::new("llama.decoder");
        let target = std::env::var("DYNINFER_TINY_GGUF_TARGET").unwrap_or_else(|_| "cpu".into());
        let bundle = dir.path().join("bundle");
        loader
            .compile_to_bundle(
                &architecture,
                &checkpoint,
                &target,
                &bundle,
                &CompileOptions {
                    mode: "local-jit".into(),
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| {
                panic!("compile direct mixed GGUF fixture on {target}: {error}")
            });
        let model = loader.load_bundle(&bundle, &checkpoint).unwrap();
        assert!(!model.manifest.derived_parameters_required);
        assert!(!bundle.join("parameters").exists());

        let tokens = [1u32, 2, 3, 0];
        let mut session = model.create_session(SessionConfig::default()).unwrap();
        let logits = session.prefill(&tokens).unwrap();
        let reference = tiny_llama_gguf_q4_0_prefill_logits(&tokens).unwrap();
        let error = max_abs_err(&logits.values, &reference).unwrap();
        assert!(
            error < 1.0e-3,
            "GGUF Q4_0 logits diverged: max_abs_err={error}\ngot={:?}\nref={:?}",
            &logits.values[..8],
            &reference[..8]
        );
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

    #[test]
    fn paged_kv_prefill_decode_and_reset() {
        if std::env::var_os("DYNINFER_PAGED_KV_E2E").is_none() {
            eprintln!("skipping: set DYNINFER_PAGED_KV_E2E=1 for paged KV E2E");
            return;
        }
        if !iree_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("m.safetensors");
        fs::write(&ckpt, tiny_llama_dense_f32()).unwrap();
        let loader = ModelLoader::default();
        let id = ArchitectureId::new("llama.decoder");
        let bundle = dir.path().join("paged");
        let overrides = dyninfer_core::MetadataMap::from([
            ("max_kv".into(), serde_json::json!(1024)),
            ("prefill_window".into(), serde_json::json!(256)),
        ]);
        let target = std::env::var("DYNINFER_PAGED_KV_TARGET").unwrap_or_else(|_| "cpu".into());
        loader
            .compile_to_bundle_with_overrides(
                &id,
                &ckpt,
                &target,
                &bundle,
                &CompileOptions {
                    mode: "local-jit".into(),
                    ..Default::default()
                },
                &overrides,
            )
            .unwrap();
        let model = loader.load_bundle(&bundle, &ckpt).unwrap();
        assert_eq!(model.manifest.version, 3);
        assert_eq!(model.manifest.prefill_window, 256);

        let mut session = model.create_session(SessionConfig::default()).unwrap();
        let tokens: Vec<u32> = (0..257).map(|index| 1 + index % 30).collect();
        let _ = session.prefill(&tokens).unwrap();
        let logits = session.decode(7).unwrap();
        assert!(logits.values.iter().all(|value| value.is_finite()));
        let metrics = session.kv_cache_metrics().unwrap();
        assert_eq!(metrics.page_count, 2);
        assert!(metrics.allocated_bytes > 0);
        session.reset().unwrap();
        assert_eq!(session.kv_cache_metrics().unwrap().page_count, 0);
    }
}
