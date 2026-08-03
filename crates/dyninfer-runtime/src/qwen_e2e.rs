//! Qwen3-0.6B BF16 bind / compile / short generate (GQA + q_norm/k_norm).

#[cfg(test)]
mod tests {
    use crate::{
        CausalLanguageModel, GenerateConfig, ModelLoader, find_safetensors_checkpoint,
        generate_greedy, load_tokenizer, resolve_hf_snapshot,
    };
    use dyninfer_architecture::LARGE_PREFILL_WINDOW;
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
            package.parameter_slots.len()
        );

        let tok = load_tokenizer(&model_dir).unwrap();
        assert!(tok.is_byte_level());
        assert_eq!(
            tok.encode("Once upon a time", false).unwrap(),
            vec![12522, 5193, 264, 882]
        );
        let _ = LARGE_PREFILL_WINDOW;
    }

    /// Full compile + short greedy decode. Slow (~minutes); enable with
    /// `DYNINFER_QWEN3_E2E=1`.
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
            .expect("compile Qwen3-0.6B");

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
            },
            SessionConfig::default(),
        )
        .expect("generate");
        eprintln!("qwen3 text={}", out.text);
        assert!(out.text.starts_with("Hello"));
        assert!(out.token_ids.len() > 1);
        let _ = Path::new(".");
    }
}
