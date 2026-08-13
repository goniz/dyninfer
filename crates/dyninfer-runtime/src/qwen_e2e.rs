//! Qwen3-0.6B BF16 bind / compile / short generate (GQA + q_norm/k_norm).

#[cfg(test)]
mod tests {
    use crate::{
        CausalLanguageModel, GenerateConfig, ModelLoader, find_gguf_checkpoint,
        find_safetensors_checkpoint, generate_greedy, load_tokenizer, resolve_hf_snapshot,
    };
    use dyninfer_compiler::LARGE_PREFILL_WINDOW;
    use dyninfer_compiler::{CompileOptions, IreeTools};
    use dyninfer_core::{ArchitectureId, SessionConfig};
    use std::path::{Path, PathBuf};

    fn iree_available() -> bool {
        IreeTools::discover().is_ok()
            || std::env::var_os("RUNFILES_DIR").is_some()
            || std::env::var_os("TEST_SRCDIR").is_some()
    }

    fn qwen3_dir() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("DYNINFER_QWEN3_DIR") {
            let p = PathBuf::from(p);
            if find_safetensors_checkpoint(&p).is_ok() {
                return Some(p);
            }
        }
        for repo in [
            "Qwen/Qwen3-0.6B",
            "mlx-community/Qwen3-0.6B-bf16",
            "mlx-community/Qwen3-0.6B-BF16",
        ] {
            if let Ok(snap) = resolve_hf_snapshot(repo, Some("main")) {
                if find_safetensors_checkpoint(&snap).is_ok() {
                    return Some(snap);
                }
            }
        }
        None
    }

    fn qwen3_gguf(filename: &str) -> Option<PathBuf> {
        if let Ok(path) = std::env::var("DYNINFER_QWEN3_GGUF") {
            let path = PathBuf::from(path);
            if path.is_file()
                && path
                    .file_name()
                    .is_some_and(|candidate| candidate == filename)
            {
                return Some(path);
            }
        }
        let snapshot = resolve_hf_snapshot("unsloth/Qwen3-0.6B-GGUF", Some("main")).ok()?;
        let preferred = snapshot.join(filename);
        preferred.is_file().then_some(preferred)
    }

    fn qwen3_q4_0() -> Option<PathBuf> {
        qwen3_gguf("Qwen3-0.6B-Q4_0.gguf").or_else(|| {
            let snapshot = resolve_hf_snapshot("unsloth/Qwen3-0.6B-GGUF", Some("main")).ok()?;
            find_gguf_checkpoint(&snapshot).ok()
        })
    }

    fn qwen3_mlx_4bit() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("DYNINFER_QWEN3_MLX") {
            let path = PathBuf::from(path);
            if path.is_file() {
                return Some(path);
            }
        }
        let snapshot = resolve_hf_snapshot("mlx-community/Qwen3-0.6B-4bit", Some("main")).ok()?;
        let index = snapshot.join("model.safetensors.index.json");
        index.is_file().then_some(index)
    }

    #[test]
    fn qwen3_0_6b_binds_gqa_and_qk_norm() {
        let Some(model_dir) = qwen3_dir() else {
            eprintln!("skipping: Qwen3-0.6B not in HF cache (set DYNINFER_QWEN3_DIR)");
            return;
        };
        let ckpt = find_safetensors_checkpoint(&model_dir).unwrap();
        let loader = ModelLoader::default();
        let id = loader
            .resolve_architecture(None, &ckpt)
            .unwrap_or_else(|_| ArchitectureId::new("qwen3.decoder"));
        assert_eq!(id.as_str(), "qwen3.decoder");
        let (package, catalog, plan) = loader.bind(&id, &ckpt, &Default::default()).unwrap();

        assert_eq!(package.resolved_config.num_layers().unwrap(), 28);
        assert_eq!(
            package.resolved_config.get_u32("hidden_size").unwrap(),
            1024
        );
        assert_eq!(package.resolved_config.get_u32("num_heads").unwrap(), 16);
        assert_eq!(package.resolved_config.get_u32("num_kv_heads").unwrap(), 8);
        assert_eq!(package.resolved_config.get_u32("head_dim").unwrap(), 128);
        assert_eq!(
            package.resolved_config.get_u32("vocab_size").unwrap(),
            151_936
        );

        assert!(
            catalog
                .parameters
                .iter()
                .any(|p| p.canonical_name.as_str() == "blk.0.attn_q_norm.weight"),
            "expected q_norm remap"
        );
        assert!(
            catalog
                .parameters
                .iter()
                .any(|p| p.canonical_name.as_str() == "output.weight"),
            "expected output.weight (tied or explicit)"
        );
        assert!(plan.unresolved_optional_slots.is_empty());
        assert_eq!(
            plan.bindings.len() + plan.unresolved_optional_slots.len(),
            package.parameter_slots().len()
        );

        let tok = load_tokenizer(&model_dir).unwrap();
        assert!(tok.is_byte_level());
        assert_eq!(
            tok.encode("Once upon a time", false).unwrap(),
            vec![12522, 5193, 264, 882]
        );
        let _ = LARGE_PREFILL_WINDOW;
    }

    #[test]
    fn qwen3_0_6b_dense_has_complete_kernel_coverage() {
        let Some(model_dir) = qwen3_dir() else {
            eprintln!("skipping: Qwen3-0.6B not in HF cache");
            return;
        };
        let checkpoint = find_safetensors_checkpoint(&model_dir).unwrap();
        let loader = ModelLoader::default();
        let id = ArchitectureId::new("qwen3.decoder");
        let report = loader
            .kernel_coverage(&id, checkpoint, "cpu", &Default::default())
            .unwrap();
        report.require_complete().unwrap();
        assert!(report.operations.len() > 1_000);
    }

    #[test]
    fn unsloth_qwen3_q4_0_has_complete_mixed_direct_cpu_coverage() {
        let Some(checkpoint) = qwen3_q4_0() else {
            eprintln!("skipping: unsloth Qwen3-0.6B Q4_0 GGUF not in HF cache");
            return;
        };
        let loader = ModelLoader::default();
        let id = ArchitectureId::new("qwen3.decoder");
        let report = loader
            .kernel_coverage(&id, checkpoint, "cpu", &Default::default())
            .unwrap();
        assert_eq!(report.operations.len(), 1_240);
        report.require_complete().unwrap();
        let encodings: std::collections::BTreeSet<_> = report
            .operations
            .iter()
            .filter_map(|operation| operation.encoding.as_ref())
            .map(|encoding| encoding.id.as_str())
            .collect();
        assert!(encodings.contains("gguf.q4_0"));
        assert!(encodings.contains("gguf.q4_1"));
        assert!(encodings.contains("gguf.q6_k"));
    }

    /// Full mixed Q4_0/Q4_1/Q6_K compile + execution. Enable with
    /// `DYNINFER_QWEN3_GGUF_E2E=1`; select the backend with
    /// `DYNINFER_QWEN3_GGUF_TARGET` (cpu or hip).
    #[test]
    fn unsloth_qwen3_q4_0_compiles_and_executes_directly() {
        if std::env::var_os("DYNINFER_QWEN3_GGUF_E2E").is_none() {
            eprintln!("skipping: set DYNINFER_QWEN3_GGUF_E2E=1 to run full GGUF Qwen3");
            return;
        }
        if !iree_available() {
            eprintln!("skipping: IREE not available");
            return;
        }
        let Some(checkpoint) = qwen3_q4_0() else {
            eprintln!("skipping: unsloth Qwen3-0.6B Q4_0 GGUF not available");
            return;
        };
        let target = std::env::var("DYNINFER_QWEN3_GGUF_TARGET").unwrap_or_else(|_| "cpu".into());
        let dir = tempfile::tempdir().unwrap();
        let bundle = dir.path().join("model.bundle");
        let loader = ModelLoader::default();
        let id = ArchitectureId::new("qwen3.decoder");
        loader
            .compile_to_bundle(
                &id,
                &checkpoint,
                &target,
                &bundle,
                &CompileOptions {
                    mode: "local-jit".into(),
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| panic!("compile direct GGUF Qwen3 on {target}: {error}"));

        let model = loader.load_bundle(&bundle, &checkpoint).unwrap();
        assert!(!model.manifest.derived_parameters_required);
        assert!(!bundle.join("parameters").exists());
        let mut session = model.create_session(SessionConfig::default()).unwrap();
        let logits = session.prefill(&[9707]).unwrap();
        assert_eq!(logits.values.len(), 151_936);
        assert!(logits.values.iter().all(|value| value.is_finite()));
        let next = logits
            .values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .unwrap()
            .0 as u32;
        let decoded = session.decode(next).unwrap();
        assert!(decoded.values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn unsloth_ud_variants_are_validated_as_mixed_per_tensor_schemas() {
        let cases: &[(&str, &[&str])] = &[
            (
                "Qwen3-0.6B-UD-IQ1_S.gguf",
                &[
                    "gguf.iq1_m",
                    "gguf.iq1_s",
                    "gguf.iq2_s",
                    "gguf.iq2_xxs",
                    "gguf.iq3_s",
                    "gguf.iq3_xxs",
                    "gguf.q2_k",
                    "gguf.q5_k",
                ],
            ),
            (
                "Qwen3-0.6B-UD-Q4_K_XL.gguf",
                &["gguf.iq4_xs", "gguf.q4_k", "gguf.q5_k"],
            ),
        ];
        let loader = ModelLoader::default();
        let id = ArchitectureId::new("qwen3.decoder");
        let mut exercised = 0;
        for (filename, expected_encodings) in cases {
            let Some(checkpoint) = qwen3_gguf(filename) else {
                continue;
            };
            exercised += 1;
            let report = loader
                .kernel_coverage(&id, checkpoint, "cpu", &Default::default())
                .unwrap();
            assert!(!report.is_complete());
            assert!(
                report
                    .operations
                    .iter()
                    .all(|operation| operation.validation_error.is_none()),
                "{filename} contains an unregistered or malformed encoding"
            );
            let actual: std::collections::BTreeSet<_> = report
                .operations
                .iter()
                .filter(|operation| operation.selected.is_none())
                .filter_map(|operation| operation.encoding.as_ref())
                .map(|encoding| encoding.id.as_str())
                .collect();
            let expected: std::collections::BTreeSet<_> =
                expected_encodings.iter().copied().collect();
            assert_eq!(actual, expected, "unexpected mixed schema for {filename}");
        }
        if exercised == 0 {
            eprintln!("skipping: Unsloth Qwen3 UD GGUFs not in HF cache");
        }
    }

    #[test]
    fn mlx_qwen3_4bit_has_complete_direct_grouped_cpu_coverage() {
        let Some(checkpoint) = qwen3_mlx_4bit() else {
            eprintln!("skipping: mlx-community Qwen3-0.6B-4bit not in HF cache");
            return;
        };
        let loader = ModelLoader::default();
        let id = ArchitectureId::new("qwen3.decoder");
        let report = loader
            .kernel_coverage(&id, checkpoint, "cpu", &Default::default())
            .unwrap();
        assert_eq!(report.operations.len(), 1_240);
        report.require_complete().unwrap();
        let mlx_requests: Vec<_> = report
            .operations
            .iter()
            .filter(|operation| {
                operation
                    .encoding
                    .as_ref()
                    .is_some_and(|encoding| encoding.id.as_str() == "mlx.affine.u4")
            })
            .collect();
        assert_eq!(mlx_requests.len(), 396);
        assert!(mlx_requests.iter().all(|operation| {
            operation.selected.as_ref().is_some_and(|selected| {
                selected
                    .descriptor
                    .lowering
                    .as_str()
                    .starts_with("mlx.affine.u4.")
            }) && operation.validation_error.is_none()
                && operation.checkpoint_component_keys.len() == 3
        }));
    }

    /// Full direct packed-U4 compile + short greedy decode. Enable with
    /// `DYNINFER_QWEN3_MLX_E2E=1`; pass the checkpoint through Bazel with
    /// `--test_env=DYNINFER_QWEN3_MLX=/absolute/path/to/index.json`. Select
    /// cpu or hip with `DYNINFER_QWEN3_MLX_TARGET`.
    #[test]
    fn mlx_qwen3_0_6b_compiles_and_generates_without_materialization() {
        if std::env::var_os("DYNINFER_QWEN3_MLX_E2E").is_none() {
            eprintln!("skipping: set DYNINFER_QWEN3_MLX_E2E=1 to run full MLX Qwen3 compile");
            return;
        }
        if !iree_available() {
            eprintln!("skipping: IREE not available");
            return;
        }
        let Some(checkpoint) = qwen3_mlx_4bit() else {
            eprintln!("skipping: mlx-community Qwen3-0.6B-4bit not available");
            return;
        };
        let model_dir = checkpoint.parent().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let bundle = dir.path().join("model.bundle");
        let loader = ModelLoader::default();
        let id = ArchitectureId::new("qwen3.decoder");
        let target = std::env::var("DYNINFER_QWEN3_MLX_TARGET").unwrap_or_else(|_| "cpu".into());
        loader
            .compile_to_bundle(
                &id,
                &checkpoint,
                &target,
                &bundle,
                &CompileOptions {
                    mode: "local-jit".into(),
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| panic!("compile direct MLX Qwen3-0.6B on {target}: {error}"));

        let model = loader.load_bundle(&bundle, &checkpoint).unwrap();
        assert!(!model.manifest.derived_parameters_required);
        assert!(model.manifest.parameter_components.len() > model.binding.bindings.len());
        assert!(!bundle.join("parameters").exists());
        let tokenizer = load_tokenizer(model_dir).unwrap();
        let output = generate_greedy(
            &model,
            &tokenizer,
            "Hello",
            &GenerateConfig {
                max_new_tokens: 2,
                eos_token_id: tokenizer.eos_id(),
                apply_chat_template: false,
                ..Default::default()
            },
            SessionConfig::default(),
        )
        .expect("generate with direct MLX Qwen3-0.6B");
        assert!(output.text.starts_with("Hello"));
        assert!(output.token_ids.len() > 1);
    }

    /// Full compile + short greedy decode. Enable with `DYNINFER_QWEN3_E2E=1`;
    /// select cpu or hip with `DYNINFER_QWEN3_TARGET`.
    #[test]
    fn qwen3_0_6b_compiles_and_generates() {
        if std::env::var_os("DYNINFER_QWEN3_E2E").is_none() {
            eprintln!("skipping: set DYNINFER_QWEN3_E2E=1 to run full Qwen3 compile");
            return;
        }
        if !iree_available() {
            eprintln!("skipping: IREE not available");
            return;
        }
        let Some(model_dir) = qwen3_dir() else {
            eprintln!("skipping: Qwen3-0.6B not available");
            return;
        };
        eprintln!("qwen3 model_dir={}", model_dir.display());
        let ckpt = find_safetensors_checkpoint(&model_dir).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let bundle = dir.path().join("model.bundle");

        let loader = ModelLoader::default();
        let id = loader.resolve_architecture(None, &ckpt).unwrap();
        assert_eq!(id.as_str(), "qwen3.decoder");
        let target = std::env::var("DYNINFER_QWEN3_TARGET").unwrap_or_else(|_| "cpu".into());
        loader
            .compile_to_bundle(
                &id,
                &ckpt,
                &target,
                &bundle,
                &CompileOptions {
                    mode: "local-jit".into(),
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| panic!("compile dense Qwen3-0.6B on {target}: {error}"));

        let model = loader.load_bundle(&bundle, &ckpt).unwrap();
        assert_eq!(model.metadata().vocabulary_size, 151_936);
        assert_eq!(model.manifest.prefill_window, LARGE_PREFILL_WINDOW);

        let tokenizer = load_tokenizer(&model_dir).unwrap();
        let eos = model
            .metadata()
            .extra
            .get("eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .or_else(|| tokenizer.eos_id());
        let out = generate_greedy(
            &model,
            &tokenizer,
            "Hello",
            &GenerateConfig {
                max_new_tokens: 8,
                eos_token_id: eos,
                apply_chat_template: false,
                ..Default::default()
            },
            SessionConfig::default(),
        )
        .expect("generate");
        eprintln!("qwen3 text={}", out.text);
        assert!(out.text.starts_with("Hello"));
        assert!(out.token_ids.len() > 1);
        let _ = Path::new(".");
    }

    #[test]
    fn qwen3_4k_paged_prompt_on_hip() {
        if std::env::var_os("DYNINFER_QWEN3_PAGED_E2E").is_none() {
            eprintln!("skipping: set DYNINFER_QWEN3_PAGED_E2E=1 for 4K HIP qualification");
            return;
        }
        let Some(model_dir) = qwen3_dir() else {
            eprintln!("skipping: Qwen3-0.6B not available");
            return;
        };
        let ckpt = find_safetensors_checkpoint(&model_dir).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let bundle = dir.path().join("paged.bundle");
        let loader = ModelLoader::default();
        let id = ArchitectureId::new("qwen3.decoder");
        let overrides = dyninfer_core::MetadataMap::from([
            ("prefill_window".into(), serde_json::json!(4096)),
            ("max_kv".into(), serde_json::json!(4224)),
        ]);
        loader
            .compile_to_bundle_with_overrides(
                &id,
                &ckpt,
                "hip",
                &bundle,
                &CompileOptions {
                    mode: "local-jit".into(),
                    ..Default::default()
                },
                &overrides,
            )
            .unwrap();
        let model = loader.load_bundle(&bundle, &ckpt).unwrap();
        assert_eq!(model.manifest.version, 8);
        let mut session = model
            .create_session(SessionConfig {
                max_sequence_length: 4224,
                ..SessionConfig::default()
            })
            .unwrap();
        let tokens: Vec<u32> = (0..4096).map(|index| 100 + index % 97).collect();
        let prefill = session.prefill(&tokens).unwrap();
        assert!(prefill.values.iter().all(|value| value.is_finite()));
        let decode = session.decode(42).unwrap();
        assert!(decode.values.iter().all(|value| value.is_finite()));
        let metrics = session.kv_cache_metrics().unwrap();
        assert_eq!(metrics.page_count, 17);
        assert!(metrics.allocated_bytes < 1024 * 1024 * 1024);
    }

    /// Prefill logit parity: CPU paged vs a GPU target. Catches GPU
    /// correctness bugs that still produce finite logits. Enable with
    /// `DYNINFER_QWEN3_PAGED_PARITY=1`; optional
    /// `DYNINFER_QWEN3_PAGED_PARITY_TARGET` (default hip).
    #[test]
    fn qwen3_paged_cpu_vs_gpu_prefill_parity() {
        if std::env::var_os("DYNINFER_QWEN3_PAGED_PARITY").is_none() {
            eprintln!("skipping: set DYNINFER_QWEN3_PAGED_PARITY=1 for CPU vs GPU logit parity");
            return;
        }
        if !iree_available() {
            eprintln!("skipping: IREE not available");
            return;
        }
        let Some(model_dir) = qwen3_dir() else {
            eprintln!("skipping: Qwen3-0.6B not available");
            return;
        };
        let ckpt = find_safetensors_checkpoint(&model_dir).unwrap();
        let gpu =
            std::env::var("DYNINFER_QWEN3_PAGED_PARITY_TARGET").unwrap_or_else(|_| "hip".into());
        let loader = ModelLoader::default();
        let id = ArchitectureId::new("qwen3.decoder");
        let overrides = dyninfer_core::MetadataMap::from([
            ("max_kv".into(), serde_json::json!(1024)),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let mut logits_by_target = Vec::new();
        for target in ["cpu", gpu.as_str()] {
            let bundle = dir.path().join(format!("{target}.bundle"));
            loader
                .compile_to_bundle_with_overrides(
                    &id,
                    &ckpt,
                    target,
                    &bundle,
                    &CompileOptions {
                        mode: "local-jit".into(),
                        ..Default::default()
                    },
                    &overrides,
                )
                .unwrap_or_else(|error| panic!("compile paged on {target}: {error}"));
            let model = loader.load_bundle(&bundle, &ckpt).unwrap();
            let mut session = model
                .create_session(SessionConfig {
                    max_sequence_length: 1024,
                    ..SessionConfig::default()
                })
                .unwrap();
            let tokenizer = load_tokenizer(&model_dir).unwrap();
            let tokens = tokenizer.encode("tell me a story", true).unwrap();
            let logits = session.prefill(&tokens).unwrap();
            assert!(
                logits.values.iter().all(|v| v.is_finite()),
                "{target} produced non-finite logits"
            );
            let mut top = crate::generate::argmax(&logits.values);
            let max_v = logits.values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "{target}: prefill argmax={top} max={max_v} tokens={}",
                tokens.len()
            );
            let mut greedy = vec![top];
            let mut cur = logits.values;
            for step in 0..8 {
                let next_logits = session.decode(top).unwrap();
                assert!(
                    next_logits.values.iter().all(|v| v.is_finite()),
                    "{target} decode step {step} non-finite"
                );
                top = crate::generate::argmax(&next_logits.values);
                greedy.push(top);
                cur = next_logits.values;
            }
            eprintln!("{target}: greedy8={greedy:?}");
            logits_by_target.push((target.to_string(), cur, greedy));
        }
        let (cpu_name, cpu, cpu_greedy) = &logits_by_target[0];
        let (gpu_name, gpu_logits, gpu_greedy) = &logits_by_target[1];
        let err = crate::reference::max_abs_err(cpu, gpu_logits).unwrap();
        eprintln!("{cpu_name} vs {gpu_name}: final max_abs_err={err}");
        eprintln!("{cpu_name} greedy={cpu_greedy:?}");
        eprintln!("{gpu_name} greedy={gpu_greedy:?}");
        assert_eq!(
            cpu_greedy, gpu_greedy,
            "paged greedy tokens diverged after prefill+8 decode"
        );
        assert!(
            err < 2.0,
            "paged logits diverged: max_abs_err={err}"
        );
    }

    /// Paged KV short generate (catches shared-memory / garbage-token
    /// regressions that finite-logit checks miss). Enable with
    /// `DYNINFER_QWEN3_PAGED_GENERATE_E2E=1`; select backend with
    /// `DYNINFER_QWEN3_PAGED_GENERATE_TARGET` (cpu or hip).
    #[test]
    fn qwen3_paged_short_generate() {
        if std::env::var_os("DYNINFER_QWEN3_PAGED_GENERATE_E2E").is_none() {
            eprintln!(
                "skipping: set DYNINFER_QWEN3_PAGED_GENERATE_E2E=1 for paged generate qualification"
            );
            return;
        }
        if !iree_available() {
            eprintln!("skipping: IREE not available");
            return;
        }
        let Some(model_dir) = qwen3_dir() else {
            eprintln!("skipping: Qwen3-0.6B not available");
            return;
        };
        let ckpt = find_safetensors_checkpoint(&model_dir).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let bundle = dir.path().join("paged-gen.bundle");
        let loader = ModelLoader::default();
        let id = ArchitectureId::new("qwen3.decoder");
        let target =
            std::env::var("DYNINFER_QWEN3_PAGED_GENERATE_TARGET").unwrap_or_else(|_| "hip".into());
        let overrides = dyninfer_core::MetadataMap::from([
            ("max_kv".into(), serde_json::json!(1024)),
        ]);
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
            .unwrap_or_else(|error| panic!("compile paged Qwen3 on {target}: {error}"));
        let model = loader.load_bundle(&bundle, &ckpt).unwrap();
        assert_eq!(model.manifest.version, 8);
        assert!(model.manifest.prefill_window >= 256);

        let tokenizer = load_tokenizer(&model_dir).unwrap();
        let eos = model
            .metadata()
            .extra
            .get("eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .or_else(|| tokenizer.eos_id());
        let out = generate_greedy(
            &model,
            &tokenizer,
            "tell me a story",
            &GenerateConfig {
                max_new_tokens: 48,
                eos_token_id: eos,
                apply_chat_template: false,
                ..Default::default()
            },
            SessionConfig {
                max_sequence_length: 1024,
                ..SessionConfig::default()
            },
        )
        .unwrap_or_else(|error| panic!("paged generate on {target}: {error}"));
        eprintln!("qwen3 paged/{target} text={}", out.text);
        assert!(
            out.token_ids.len() > 8,
            "expected several generated tokens, got {}",
            out.token_ids.len()
        );
        let unique: std::collections::BTreeSet<_> = out.token_ids.iter().copied().collect();
        assert!(
            unique.len() >= 8,
            "paged generate collapsed to {:?}; text={}",
            out.token_ids,
            out.text
        );
        // Refuse the known garbage loop pattern.
        assert!(
            !out.text.contains("odable"),
            "paged generate produced garbage text on {target}: {}",
            out.text
        );
    }
}
