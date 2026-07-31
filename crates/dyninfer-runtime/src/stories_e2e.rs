//! Real Maykeye/TinyLLama-v0 generation (coherent TinyStories English).

#[cfg(test)]
mod tests {
    use crate::{
        generate_greedy, load_tokenizer, resolve_hf_snapshot, CausalLanguageModel, GenerateConfig,
        ModelLoader,
    };
    use crate::find_safetensors_checkpoint;
    use dyninfer_compiler::{CompileOptions, IreeTools};
    use dyninfer_core::{ArchitectureId, SessionConfig};
    use std::path::{Path, PathBuf};

    fn iree_available() -> bool {
        IreeTools::discover().is_ok()
            || std::env::var_os("RUNFILES_DIR").is_some()
            || std::env::var_os("TEST_SRCDIR").is_some()
    }

    fn maykeye_dir() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("DYNINFER_TINYLLAMA_DIR") {
            let p = PathBuf::from(p);
            if find_safetensors_checkpoint(&p).is_ok() {
                return Some(p);
            }
        }
        // Prefer the user's Hugging Face Hub cache (no vendored copy required).
        if let Ok(snap) = resolve_hf_snapshot("Maykeye/TinyLLama-v0", Some("main")) {
            return Some(snap);
        }
        // Bazel runfiles / vendored testdata fallback.
        let candidates = [
            "architectures/testdata/maykeye-tinyllama-v0",
            "maykeye-tinyllama-v0",
            "../architectures/testdata/maykeye-tinyllama-v0",
            "../../architectures/testdata/maykeye-tinyllama-v0",
        ];
        for c in candidates {
            let p = PathBuf::from(c);
            if find_safetensors_checkpoint(&p).is_ok() {
                return Some(p);
            }
        }
        if let Ok(src) = std::env::var("TEST_SRCDIR") {
            let p = Path::new(&src)
                .join("_main")
                .join("architectures/testdata/maykeye-tinyllama-v0");
            if find_safetensors_checkpoint(&p).is_ok() {
                return Some(p);
            }
        }
        if let Ok(rf) = std::env::var("RUNFILES_DIR") {
            let p = Path::new(&rf)
                .join("_main")
                .join("architectures/testdata/maykeye-tinyllama-v0");
            if find_safetensors_checkpoint(&p).is_ok() {
                return Some(p);
            }
        }
        None
    }

    #[test]
    fn maykeye_tinyllama_generates_story_text() {
        if !iree_available() {
            eprintln!("skipping: IREE not available");
            return;
        }
        let Some(model_dir) = maykeye_dir() else {
            eprintln!("skipping: Maykeye/TinyLLama-v0 not in HF cache or testdata");
            return;
        };
        eprintln!("maykeye model_dir={}", model_dir.display());
        let ckpt = find_safetensors_checkpoint(&model_dir).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let bundle = dir.path().join("model.bundle");

        let loader = ModelLoader::default();
        let id = ArchitectureId::new("llama.decoder");
        let (package, catalog, plan) = loader.bind(&id, &ckpt, &Default::default()).unwrap();
        assert_eq!(package.resolved_config.num_layers().unwrap(), 8);
        assert_eq!(package.resolved_config.get_u32("hidden_size").unwrap(), 64);
        assert_eq!(package.resolved_config.get_u32("vocab_size").unwrap(), 32000);
        assert_eq!(
            plan.bindings.len() + plan.unresolved_optional_slots.len(),
            package.parameter_slots.len()
        );
        assert!(
            catalog
                .parameters
                .iter()
                .any(|p| p.canonical_name.as_str() == "token_embd.weight"),
            "expected HF→GGUF remapped embedding"
        );

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
            .expect("compile Maykeye TinyLLama");

        let model = loader.load_bundle(&bundle, &ckpt).unwrap();
        assert_eq!(model.metadata().vocabulary_size, 32000);
        assert_eq!(model.manifest.prefill_window, 64);

        let tokenizer = load_tokenizer(&model_dir).unwrap();
        let prompt = "Once upon a time";
        let out = generate_greedy(
            &model,
            &tokenizer,
            prompt,
            &GenerateConfig {
                max_new_tokens: 40,
                eos_token_id: Some(2),
            },
            SessionConfig::default(),
        )
        .expect("generate");

        println!("GENERATED: {}", out.text);
        assert!(
            out.text.len() > prompt.len(),
            "expected continuation beyond prompt, got {:?}",
            out.text
        );
        let lower = out.text.to_ascii_lowercase();
        let storyish = ["once", "there", "was", "little", "girl", "boy", "said", "day"]
            .iter()
            .any(|w| lower.contains(w));
        assert!(
            storyish,
            "expected TinyStories-like English, got {:?}",
            out.text
        );
        let printable = out
            .text
            .chars()
            .filter(|c| c.is_ascii_graphic() || c.is_ascii_whitespace())
            .count();
        assert!(
            printable * 100 / out.text.len().max(1) > 85,
            "output not mostly printable ASCII: {:?}",
            out.text
        );
    }
}
