//! dyninfer CLI.

#![forbid(unsafe_code)]

use clap::{Parser, Subcommand};
use dyninfer_cache::ArtifactCache;
use dyninfer_compiler::{CompileOptions, compile_add_smoke};
use dyninfer_core::SessionConfig;
use dyninfer_runtime::{
    CausalLanguageModel, GenerateConfig, GenerateOutput, ModelLoader, find_checkpoint,
    generate_greedy, load_tokenizer, resolve_hf_snapshot,
};
use dyninfer_target::TargetDiscovery;
use iree_runtime::{Context, Instance, Module};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "dyninfer", version, about = "Dynamic Inference Engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Checkpoint inspection and helpers.
    Checkpoint {
        #[command(subcommand)]
        command: CheckpointCommands,
    },
    /// Bind an architecture to a checkpoint.
    Bind {
        /// Architecture id, or `auto` to detect from config.json.
        #[arg(long, default_value = "auto")]
        architecture: String,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Config override, e.g. `--set num_layers=1`.
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
    },
    /// Validate per-operation production kernel coverage without compiling.
    Coverage {
        /// Architecture id, or `auto` to detect from checkpoint metadata.
        #[arg(long, default_value = "auto")]
        architecture: String,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value = "auto")]
        target: String,
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// Compile architecture + checkpoint into a bundle.
    Compile {
        /// Architecture id, or `auto` to detect from config.json.
        #[arg(long, default_value = "auto")]
        architecture: String,
        #[arg(long)]
        checkpoint: PathBuf,
        /// Execution target: `auto` (probe IREE HAL), `cpu`, `vulkan`, `rocm`/`hip`/`cuda`, or `rocm:gfxXXXX` / `cuda:sm_XX`.
        #[arg(long, default_value = "auto")]
        target: String,
        #[arg(long, default_value = "local-jit")]
        mode: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Config override, e.g. `--set num_layers=1`.
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
    },
    /// Run greedy generation against a compiled bundle (tokenizer + IREE).
    Run {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value = "Once upon a time")]
        prompt: String,
        #[arg(long, default_value_t = 48)]
        max_new_tokens: usize,
    },
    /// Compile + generate from a local HF snapshot or Hub cache entry.
    Generate {
        /// Architecture id, or `auto` to detect from config.json.
        #[arg(long, default_value = "auto")]
        architecture: String,
        /// Local model directory (config.json + weights + tokenizer.json).
        #[arg(long, conflicts_with = "hf")]
        model_dir: Option<PathBuf>,
        /// Hugging Face repo id resolved against the local Hub cache
        /// (`HF_HUB_CACHE` / `~/.cache/huggingface/hub`). Does not download.
        #[arg(long, value_name = "ORG/NAME")]
        hf: Option<String>,
        /// Hub revision / branch / snapshot hash (default: main).
        #[arg(long, default_value = "main")]
        revision: String,
        /// Explicit checkpoint path (`.safetensors` or `.gguf`). Overrides discovery.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Execution target: `auto` (probe IREE HAL), `cpu`, `vulkan`, `rocm`/`hip`/`cuda`, or `rocm:gfxXXXX` / `cuda:sm_XX`.
        #[arg(long, default_value = "auto")]
        target: String,
        #[arg(long, default_value = "Once upon a time")]
        prompt: String,
        #[arg(long, default_value_t = 48)]
        max_new_tokens: usize,
        /// HF-style repetition penalty (1.0 = off). Helps long greedy runs.
        #[arg(long, default_value_t = 1.1)]
        repetition_penalty: f32,
        /// Do not apply the model's HF chat template.
        #[arg(long)]
        raw_prompt: bool,
        /// Disable Qwen3-style thinking in the chat template (`enable_thinking=false`).
        #[arg(long)]
        no_thinking: bool,
        #[arg(long)]
        output_bundle: Option<PathBuf>,
        /// Legacy prefill window. Paged executables use chunks up to 512 tokens.
        #[arg(long)]
        prefill_window: Option<u32>,
        /// Session/model context limit. Defaults to prompt tokens + --max-new-tokens.
        /// Values above 512 select paged KV ABI v6.
        #[arg(long)]
        max_kv: Option<u32>,
    },
    /// Compile and run the trivial `@add` smoke module through real IREE.
    Smoke {
        /// Execution target: `auto` (probe IREE HAL), `cpu`, `vulkan`, `rocm`/`hip`/`cuda`, or `rocm:gfxXXXX` / `cuda:sm_XX`.
        #[arg(long, default_value = "auto")]
        target: String,
    },
    /// Install model into local cache (inspect+bind+compile+publish).
    Model {
        #[command(subcommand)]
        command: ModelCommands,
    },
    /// Executable cache management.
    Cache {
        #[command(subcommand)]
        command: CacheCommands,
    },
}

#[derive(Subcommand, Debug)]
enum CheckpointCommands {
    Inspect {
        path: PathBuf,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum ModelCommands {
    Install {
        /// Architecture id, or `auto` to detect from config.json.
        #[arg(long, default_value = "auto")]
        architecture: String,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value = "auto")]
        target: String,
        #[arg(long)]
        cache_dir: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum CacheCommands {
    List {
        #[arg(long)]
        cache_dir: PathBuf,
    },
    Verify {
        #[arg(long)]
        cache_dir: PathBuf,
    },
    Remove {
        #[arg(long)]
        cache_dir: PathBuf,
        digest: String,
    },
    Prune {
        #[arg(long)]
        cache_dir: PathBuf,
        #[arg(long)]
        max_size: String,
    },
}

fn parse_sets(sets: &[String]) -> anyhow::Result<dyninfer_core::MetadataMap> {
    let mut map = dyninfer_core::MetadataMap::new();
    for item in sets {
        let (k, v) = item
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("expected --set key=value, got {item}"))?;
        let value = if let Ok(n) = v.parse::<u64>() {
            serde_json::Value::Number(n.into())
        } else if let Ok(f) = v.parse::<f64>() {
            serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                .unwrap_or_else(|| serde_json::Value::String(v.to_string()))
        } else if v == "true" || v == "false" {
            serde_json::Value::Bool(v == "true")
        } else {
            serde_json::Value::String(v.to_string())
        };
        map.insert(k.to_string(), value);
    }
    Ok(map)
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Checkpoint { command } => match command {
            CheckpointCommands::Inspect { path, json } => {
                let loader = ModelLoader::default();
                let catalog = loader.inspect(&path)?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&catalog)?);
                } else {
                    println!("container: {}", catalog.container.format_id);
                    if let Some(v) = catalog.container.version {
                        println!("version: {v}");
                    }
                    println!("convention: {}", catalog.convention_id);
                    println!(
                        "schema: {} ({} entries, {} bytes)",
                        catalog.schema_fingerprint.digest.short(),
                        catalog.schema_fingerprint.entry_count,
                        catalog.schema_fingerprint.total_bytes
                    );
                    println!("parameters: {}", catalog.parameters.len());
                    for p in catalog.parameters.iter().take(20) {
                        println!(
                            "  {}  {}  {}  {:?}",
                            p.canonical_name,
                            p.role.as_str(),
                            p.logical_type.shape,
                            p.encoding
                        );
                    }
                    if catalog.parameters.len() > 20 {
                        println!("  ... {} more", catalog.parameters.len() - 20);
                    }
                }
            }
        },
        Commands::Bind {
            architecture,
            checkpoint,
            output,
            set,
        } => {
            let loader = ModelLoader::default();
            let id = loader.resolve_architecture(Some(&architecture), &checkpoint)?;
            eprintln!("architecture {}", id);
            let overrides = parse_sets(&set)?;
            let (_pkg, _catalog, plan) = loader.bind(&id, &checkpoint, &overrides)?;
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&output, serde_json::to_vec_pretty(&plan)?)?;
            println!("wrote binding plan to {}", output.display());
        }
        Commands::Coverage {
            architecture,
            checkpoint,
            target,
            set,
            json,
        } => {
            let loader = ModelLoader::default();
            let id = loader.resolve_architecture(Some(&architecture), &checkpoint)?;
            let overrides = parse_sets(&set)?;
            let report = loader.kernel_coverage(&id, &checkpoint, &target, &overrides)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                let selected = report
                    .operations
                    .iter()
                    .filter(|operation| operation.selected.is_some())
                    .count();
                println!(
                    "coverage: {selected}/{} operation-mode requests",
                    report.operations.len()
                );
                for operation in report.operations.iter().filter(|operation| {
                    operation.selected.is_none() || operation.validation_error.is_some()
                }) {
                    println!(
                        "  unsupported {} {:?} encoding={:?}: {}",
                        operation.operation_id,
                        operation.mode,
                        operation.encoding,
                        operation
                            .validation_error
                            .as_deref()
                            .unwrap_or("no production candidate")
                    );
                    for rejected in &operation.rejected {
                        println!(
                            "    rejected {}: {:?}",
                            rejected.candidate_id, rejected.reasons
                        );
                    }
                }
            }
            report.require_complete()?;
        }
        Commands::Compile {
            architecture,
            checkpoint,
            target,
            mode,
            output,
            cache_dir,
            set,
        } => {
            let mut loader = ModelLoader::default();
            if let Some(dir) = cache_dir {
                loader = loader.with_cache(ArtifactCache::open(dir)?);
            }
            let id = loader.resolve_architecture(Some(&architecture), &checkpoint)?;
            eprintln!("architecture {}", id);
            let options = CompileOptions {
                mode,
                ..Default::default()
            };
            let overrides = parse_sets(&set)?;
            let paths = loader.compile_to_bundle_with_overrides(
                &id,
                &checkpoint,
                &target,
                &output,
                &options,
                &overrides,
            )?;
            println!("wrote bundle to {}", paths.root.display());
            println!("vmfb: {}", paths.vmfb.display());
        }
        Commands::Run {
            bundle,
            checkpoint,
            tokenizer,
            prompt,
            max_new_tokens,
        } => {
            let loader = ModelLoader::default();
            let model = loader.load_bundle(&bundle, &checkpoint)?;
            let tok_path = tokenizer.unwrap_or_else(|| {
                checkpoint
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."))
            });
            let tokenizer = load_tokenizer(&tok_path)?;
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
                &prompt,
                &GenerateConfig {
                    max_new_tokens,
                    eos_token_id: eos,
                    apply_chat_template: false,
                    ..Default::default()
                },
                SessionConfig::default(),
            )?;
            print_generate_result(&out);
        }
        Commands::Generate {
            architecture,
            model_dir,
            hf,
            revision,
            checkpoint,
            target,
            prompt,
            max_new_tokens,
            repetition_penalty,
            raw_prompt,
            no_thinking,
            output_bundle,
            prefill_window,
            max_kv,
        } => {
            let model_dir = match (hf, model_dir) {
                (Some(repo), None) => {
                    let snap = resolve_hf_snapshot(&repo, Some(&revision))?;
                    eprintln!("using HF cache snapshot {}", snap.display());
                    snap
                }
                (None, Some(dir)) => dir,
                (None, None) => anyhow::bail!("provide --hf ORG/NAME or --model-dir PATH"),
                (Some(_), Some(_)) => unreachable!("clap conflicts_with"),
            };
            let ckpt = match checkpoint {
                Some(p) => p,
                None => find_checkpoint(&model_dir)?,
            };
            eprintln!("checkpoint {}", ckpt.display());
            let tokenizer = load_tokenizer(&model_dir)?;
            let apply_chat_template = !raw_prompt;
            let enable_thinking = !no_thinking;
            let model_prompt = if apply_chat_template {
                tokenizer
                    .apply_chat_template(&prompt, enable_thinking)?
                    .unwrap_or_else(|| prompt.clone())
            } else {
                prompt.clone()
            };
            if apply_chat_template && tokenizer.has_chat_template() {
                eprintln!(
                    "chat_template=tokenizer_config.json enable_thinking={enable_thinking}"
                );
            }
            // Size KV for the full request; Qwen/large defaults are only 256.
            let add_special = !tokenizer.is_byte_level() && !apply_chat_template;
            let prompt_tokens = tokenizer.encode(&model_prompt, add_special)?.len();
            let needed_kv = (prompt_tokens + max_new_tokens)
                .try_into()
                .unwrap_or(u32::MAX)
                .max(1);
            let effective_max_kv = match max_kv {
                Some(k) if k < needed_kv => anyhow::bail!(
                    "--max-kv {k} cannot fit prompt ({prompt_tokens} tokens) + --max-new-tokens {max_new_tokens} (need >= {needed_kv})"
                ),
                Some(k) => k,
                None => needed_kv,
            };
            if max_kv.is_none() {
                eprintln!(
                    "max_kv={effective_max_kv} (prompt={prompt_tokens} + max_new_tokens={max_new_tokens})"
                );
            }
            let default_bundle = std::env::temp_dir().join(format!(
                "dyninfer-generate-{}/model.bundle",
                std::process::id()
            ));
            let bundle = output_bundle.unwrap_or(default_bundle);
            let loader = ModelLoader::default();
            let id = loader.resolve_architecture(Some(&architecture), &ckpt)?;
            eprintln!("architecture {}", id);
            let mut overrides = dyninfer_core::MetadataMap::new();
            if let Some(w) = prefill_window {
                overrides.insert("prefill_window".into(), serde_json::json!(w));
            }
            overrides.insert("max_kv".into(), serde_json::json!(effective_max_kv));
            let paths = loader.compile_to_bundle_with_overrides(
                &id,
                &ckpt,
                &target,
                &bundle,
                &CompileOptions {
                    mode: "local-jit".into(),
                    ..Default::default()
                },
                &overrides,
            )?;
            let model = loader.load_bundle(&paths.root, &ckpt)?;
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
                &prompt,
                &GenerateConfig {
                    max_new_tokens,
                    eos_token_id: eos,
                    repetition_penalty,
                    apply_chat_template,
                    enable_thinking,
                    ..Default::default()
                },
                SessionConfig {
                    max_sequence_length: effective_max_kv,
                    ..SessionConfig::default()
                },
            )?;
            print_generate_result(&out);
        }
        Commands::Smoke { target } => {
            let profile = TargetDiscovery::resolve(&target)?;
            eprintln!(
                "target driver={} chip={:?}",
                profile.driver,
                profile.rocm_target()
            );
            let vmfb = compile_add_smoke(&profile)?;
            println!("compiled smoke VMFB ({} bytes)", vmfb.len());
            let instance = Instance::new()?;
            let module = Module::from_vmfb(vmfb)?;
            let ctx = Context::create(instance, module)?.with_device(profile.runtime_device());
            let out = ctx.invoke_add(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])?;
            println!("add([1,2,3,4],[10,20,30,40]) = {out:?}");
            if out == [11.0, 22.0, 33.0, 44.0] {
                println!("IREE smoke OK");
            } else {
                anyhow::bail!("unexpected smoke result: {out:?}");
            }
        }
        Commands::Model { command } => match command {
            ModelCommands::Install {
                architecture,
                checkpoint,
                target,
                cache_dir,
                output,
            } => {
                let cache = ArtifactCache::open(&cache_dir)?;
                let loader = ModelLoader::default().with_cache(cache);
                let id = loader.resolve_architecture(Some(&architecture), &checkpoint)?;
                eprintln!("architecture {}", id);
                let out = output.unwrap_or_else(|| cache_dir.join("bundles").join("model.bundle"));
                let paths = loader.compile_to_bundle(
                    &id,
                    &checkpoint,
                    &target,
                    &out,
                    &CompileOptions {
                        mode: "local-jit".into(),
                        ..Default::default()
                    },
                )?;
                println!("installed bundle at {}", paths.root.display());
            }
        },
        Commands::Cache { command } => match command {
            CacheCommands::List { cache_dir } => {
                let cache = ArtifactCache::open(cache_dir)?;
                for entry in cache.list()? {
                    println!(
                        "{}  {}  {} bytes  {}",
                        entry.digest.short(),
                        entry.key.architecture_id,
                        entry.size_bytes,
                        entry.key.target_fingerprint
                    );
                }
            }
            CacheCommands::Verify { cache_dir } => {
                let cache = ArtifactCache::open(cache_dir)?;
                let problems = cache.verify()?;
                if problems.is_empty() {
                    println!("cache ok");
                } else {
                    for p in problems {
                        println!("problem: {p}");
                    }
                    std::process::exit(1);
                }
            }
            CacheCommands::Remove { cache_dir, digest } => {
                let cache = ArtifactCache::open(cache_dir)?;
                if cache.remove(&digest)? {
                    println!("removed {digest}");
                } else {
                    println!("no entry matched {digest}");
                    std::process::exit(1);
                }
            }
            CacheCommands::Prune {
                cache_dir,
                max_size,
            } => {
                let _ = ArtifactCache::open(cache_dir)?;
                println!("prune is a stub; requested max_size={max_size} (no eviction yet)");
            }
        },
    }
    Ok(())
}

fn print_generate_result(out: &GenerateOutput) {
    println!("{}", out.text);
    let s = &out.stats;
    eprintln!(
        "prefill: {} tok in {:.3}s ({:.1} tok/s)  decode: {} tok in {:.3}s ({:.1} tok/s)  KV: {} pages ({:.1} MiB)",
        s.prompt_tokens,
        s.prefill_secs,
        s.prefill_tps(),
        s.generated_tokens,
        s.decode_secs,
        s.decode_tps(),
        s.kv_page_count,
        s.kv_allocated_bytes as f64 / (1024.0 * 1024.0),
    );
}
