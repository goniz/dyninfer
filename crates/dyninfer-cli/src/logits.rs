//! `dyninfer logits drift` orchestration and reporting.

use crate::drift::{RowMetrics, compare};
use crate::logit_trace::{AtomicTraceWriter, TokenizedPrompt, TracePhase, TraceReader};
use anyhow::{Context, Result, bail, ensure};
use clap::{ArgGroup, Args, ValueEnum};
use dyninfer_compiler::{CompileOptions, PAGED_PREFILL_CHUNK_SIZE};
use dyninfer_core::{
    ExecutableManifest, KvCacheStorage, ScalarType, SchemaFingerprint, SessionConfig,
};
use dyninfer_runtime::{CausalLanguageModel, ModelLoader};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const REPORT_FORMAT: &str = "dyninfer.logit-drift-report";
const REPORT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum AddSpecial {
    Auto,
    Yes,
    No,
}

impl AddSpecial {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Yes => "yes",
            Self::No => "no",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum FlashAttention {
    Off,
    On,
    Auto,
}

impl FlashAttention {
    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
            Self::Auto => "auto",
        }
    }
}

#[derive(Args, Debug)]
#[command(group(
    ArgGroup::new("prompt_input")
        .required(true)
        .args(["prompt", "tokens", "tokens_file"])
))]
pub struct DriftArgs {
    /// Exact GGUF checkpoint passed to both engines.
    #[arg(long)]
    checkpoint: PathBuf,
    /// Architecture id, or `auto` for checkpoint metadata resolution.
    #[arg(long, default_value = "auto")]
    architecture: String,
    /// Reuse an existing dyninfer bundle instead of compiling one.
    #[arg(long)]
    bundle: Option<PathBuf>,
    /// Dyninfer compile target when --bundle is absent.
    #[arg(long, default_value = "cpu")]
    target: String,
    /// Prompt text tokenized by the llama.cpp companion.
    #[arg(long)]
    prompt: Option<String>,
    /// Exact comma-separated prompt token IDs.
    #[arg(long)]
    tokens: Option<String>,
    /// Exact whitespace- or JSON-array prompt token IDs.
    #[arg(long)]
    tokens_file: Option<PathBuf>,
    /// Reference-greedy decode steps (defaults to 8).
    #[arg(long, conflicts_with = "decode_tokens")]
    decode_steps: Option<usize>,
    /// Explicit comma-separated forced decode inputs.
    #[arg(long, conflicts_with = "decode_steps")]
    decode_tokens: Option<String>,
    /// llama.cpp special-token insertion policy for text prompts.
    #[arg(long, value_enum, default_value_t = AddSpecial::Auto)]
    add_special: AddSpecial,
    /// Recognize textual special-token spellings during llama.cpp tokenization.
    #[arg(long)]
    parse_special: bool,
    /// Path or executable name of the public-libllama companion.
    #[arg(long, default_value = "dyninfer-llama-logits")]
    llama_runner: PathBuf,
    /// llama.cpp device name/description substring, or `cpu`.
    #[arg(long, default_value = "cpu")]
    llama_device: String,
    /// Number of llama.cpp model layers to offload (0 for the CPU reference).
    #[arg(long, default_value_t = 0)]
    llama_gpu_layers: i32,
    /// Explicit llama.cpp generation and batch thread count.
    #[arg(long)]
    llama_threads: Option<u32>,
    /// llama.cpp flash-attention mode.
    #[arg(long, value_enum, default_value_t = FlashAttention::Off)]
    llama_flash_attn: FlashAttention,
    /// Explicit llama.cpp KV type override (f32, f16, or bf16).
    #[arg(long)]
    llama_kv_type: Option<String>,
    /// Number of diagnostic top tokens/deltas to retain per row.
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    /// Write a durable JSON report.
    #[arg(long)]
    json_out: Option<PathBuf>,
    /// Retain `llama.cpp/` and `dyninfer/` trace directories here.
    #[arg(long)]
    keep_traces: Option<PathBuf>,
    /// Independently free-run dyninfer greedily and compare token sequences.
    #[arg(long)]
    generate_coherency: bool,
    /// Fail if any row's maximum absolute error exceeds this value.
    #[arg(long)]
    max_abs: Option<f64>,
    /// Fail if any row's RMSE exceeds this value.
    #[arg(long)]
    max_rmse: Option<f64>,
    /// Fail if any row's relative L2 error exceeds this value.
    #[arg(long)]
    max_relative_l2: Option<f64>,
    /// Fail if any row's cosine similarity is below this value.
    #[arg(long)]
    min_cosine: Option<f64>,
    /// Fail on the first raw-argmax disagreement.
    #[arg(long)]
    require_argmax_match: bool,
}

#[derive(Debug, Clone, Serialize)]
struct CheckpointIdentity {
    canonical_path: String,
    size_bytes: u64,
    sha256: String,
    schema_fingerprint: SchemaFingerprint,
}

#[derive(Debug, Clone, Serialize)]
struct DyninferProvenance {
    bundle_path: String,
    reused_bundle: bool,
    manifest: ExecutableManifest,
}

#[derive(Debug, Clone, Serialize)]
struct ReferenceProvenance {
    runner: String,
    requested_device: String,
    requested_gpu_layers: i32,
    requested_threads: Option<u32>,
    requested_flash_attention: String,
    requested_kv_type: String,
    trace_metadata: Value,
}

#[derive(Debug, Clone, Serialize)]
struct PromptProvenance {
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    add_special: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parse_special: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokenization_metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
struct ComparedRow {
    step: usize,
    phase: TracePhase,
    position: u64,
    input_token: u32,
    forced_token: Option<u32>,
    metrics: RowMetrics,
}

#[derive(Debug, Clone, Serialize)]
struct LocatedValue {
    row: usize,
    value: f64,
}

#[derive(Debug, Clone, Serialize)]
struct AggregateMetrics {
    worst_max_absolute_error: LocatedValue,
    worst_rmse: LocatedValue,
    worst_relative_l2_error: LocatedValue,
    minimum_cosine_similarity: LocatedValue,
    worst_softmax_kl: LocatedValue,
    worst_probability_delta: LocatedValue,
    first_argmax_disagreement: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct Thresholds {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_absolute_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_rmse: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_relative_l2_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_cosine_similarity: Option<f64>,
    require_argmax_match: bool,
}

impl Thresholds {
    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("--max-abs", self.max_absolute_error),
            ("--max-rmse", self.max_rmse),
            ("--max-relative-l2", self.max_relative_l2_error),
        ] {
            if let Some(value) = value {
                ensure!(
                    value.is_finite() && value >= 0.0,
                    "{name} must be finite and non-negative"
                );
            }
        }
        if let Some(value) = self.min_cosine_similarity {
            ensure!(
                value.is_finite() && (-1.0..=1.0).contains(&value),
                "--min-cosine must be finite and between -1 and 1"
            );
        }
        Ok(())
    }

    fn failures(&self, aggregate: &AggregateMetrics) -> Vec<String> {
        let mut failures = Vec::new();
        if let Some(limit) = self.max_absolute_error {
            if aggregate.worst_max_absolute_error.value > limit {
                failures.push(format!(
                    "max_abs {} > {} at row {}",
                    aggregate.worst_max_absolute_error.value,
                    limit,
                    aggregate.worst_max_absolute_error.row
                ));
            }
        }
        if let Some(limit) = self.max_rmse {
            if aggregate.worst_rmse.value > limit {
                failures.push(format!(
                    "rmse {} > {} at row {}",
                    aggregate.worst_rmse.value, limit, aggregate.worst_rmse.row
                ));
            }
        }
        if let Some(limit) = self.max_relative_l2_error {
            if aggregate.worst_relative_l2_error.value > limit {
                failures.push(format!(
                    "relative_l2 {} > {} at row {}",
                    aggregate.worst_relative_l2_error.value,
                    limit,
                    aggregate.worst_relative_l2_error.row
                ));
            }
        }
        if let Some(limit) = self.min_cosine_similarity {
            if aggregate.minimum_cosine_similarity.value < limit {
                failures.push(format!(
                    "cosine {} < {} at row {}",
                    aggregate.minimum_cosine_similarity.value,
                    limit,
                    aggregate.minimum_cosine_similarity.row
                ));
            }
        }
        if self.require_argmax_match {
            if let Some(row) = aggregate.first_argmax_disagreement {
                failures.push(format!("argmax disagreement at row {row}"));
            }
        }
        failures
    }
}

#[derive(Debug, Clone, Serialize)]
struct TokenDisagreement {
    step: usize,
    reference_token: u32,
    dyninfer_token: u32,
}

#[derive(Debug, Clone, Serialize)]
struct GenerationCoherency {
    reference_tokens: Vec<u32>,
    dyninfer_tokens: Vec<u32>,
    first_disagreement: Option<TokenDisagreement>,
    coherent: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DriftReport {
    format: &'static str,
    version: u32,
    checkpoint: CheckpointIdentity,
    prompt: PromptProvenance,
    prompt_tokens: Vec<u32>,
    decode_inputs: Vec<u32>,
    dyninfer: DyninferProvenance,
    reference: ReferenceProvenance,
    comparison_caveats: Vec<String>,
    thresholds: Thresholds,
    rows: Vec<ComparedRow>,
    aggregate: AggregateMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_coherency: Option<GenerationCoherency>,
}

pub fn run(args: DriftArgs) -> Result<()> {
    ensure!(args.top_k > 0, "--top-k must be greater than zero");
    if args.llama_device.eq_ignore_ascii_case("cpu") {
        ensure!(
            args.llama_gpu_layers == 0,
            "--llama-device cpu requires --llama-gpu-layers 0"
        );
    }
    let thresholds = Thresholds {
        max_absolute_error: args.max_abs,
        max_rmse: args.max_rmse,
        max_relative_l2_error: args.max_relative_l2,
        min_cosine_similarity: args.min_cosine,
        require_argmax_match: args.require_argmax_match,
    };
    thresholds.validate()?;

    let checkpoint = args
        .checkpoint
        .canonicalize()
        .with_context(|| format!("canonicalize checkpoint {}", args.checkpoint.display()))?;
    ensure!(
        checkpoint.is_file(),
        "checkpoint {} is not a file",
        checkpoint.display()
    );
    let checkpoint_size = fs::metadata(&checkpoint)?.len();
    let checkpoint_sha256 = sha256_file(&checkpoint)?;
    let loader = ModelLoader::default();
    let inspected = loader.inspect(&checkpoint)?;
    ensure!(
        inspected.container.format_id.as_str() == "gguf",
        "logit drift requires exactly one GGUF checkpoint (found {})",
        inspected.container.format_id
    );

    let scratch = tempfile::Builder::new()
        .prefix("dyninfer-logit-drift-")
        .tempdir()
        .context("create drift scratch directory")?;
    let artifacts_root = if let Some(directory) = &args.keep_traces {
        fs::create_dir_all(directory)
            .with_context(|| format!("create retained trace directory {}", directory.display()))?;
        directory.clone()
    } else {
        scratch.path().to_path_buf()
    };

    let (prompt_tokens, prompt_provenance) = if let Some(prompt) = args.prompt.as_deref() {
        let output = scratch.path().join("tokenized-prompt.json");
        let stderr = artifacts_root.join("llama-tokenize.stderr.log");
        let mut runner_args = vec![
            OsString::from("tokenize"),
            OsString::from("--checkpoint"),
            checkpoint.as_os_str().to_owned(),
            OsString::from("--prompt"),
            OsString::from(prompt),
            OsString::from("--add-special"),
            OsString::from(args.add_special.as_str()),
            OsString::from("--output"),
            output.as_os_str().to_owned(),
        ];
        if args.parse_special {
            runner_args.push(OsString::from("--parse-special"));
        }
        run_companion(&args.llama_runner, &runner_args, &stderr, "tokenize prompt")?;
        let tokenized = TokenizedPrompt::read(&output)?;
        ensure!(
            tokenized
                .checkpoint_sha256
                .eq_ignore_ascii_case(&checkpoint_sha256),
            "tokenizer checkpoint digest {} does not match {}",
            tokenized.checkpoint_sha256,
            checkpoint_sha256
        );
        (
            tokenized.prompt_tokens,
            PromptProvenance {
                source: "llama.cpp_tokenized_text".into(),
                tokens_file: None,
                add_special: Some(args.add_special.as_str().into()),
                parse_special: Some(args.parse_special),
                tokenization_metadata: Some(tokenized.engine_metadata),
            },
        )
    } else if let Some(tokens) = args.tokens.as_deref() {
        (
            parse_comma_tokens(tokens, "--tokens")?,
            PromptProvenance {
                source: "exact_cli_tokens".into(),
                tokens_file: None,
                add_special: None,
                parse_special: None,
                tokenization_metadata: None,
            },
        )
    } else if let Some(path) = args.tokens_file.as_deref() {
        (
            parse_token_file(path)?,
            PromptProvenance {
                source: "exact_tokens_file".into(),
                tokens_file: Some(path.display().to_string()),
                add_special: None,
                parse_special: None,
                tokenization_metadata: None,
            },
        )
    } else {
        unreachable!("clap prompt_input group")
    };
    ensure!(!prompt_tokens.is_empty(), "prompt token list is empty");

    let explicit_decode = args
        .decode_tokens
        .as_deref()
        .map(|tokens| parse_comma_tokens(tokens, "--decode-tokens"))
        .transpose()?;
    let decode_steps = explicit_decode
        .as_ref()
        .map(Vec::len)
        .unwrap_or(args.decode_steps.unwrap_or(8));
    if args.generate_coherency {
        ensure!(
            explicit_decode.is_none(),
            "--generate-coherency requires reference-greedy --decode-steps"
        );
        ensure!(
            decode_steps > 0,
            "--generate-coherency requires at least one decode step"
        );
    }
    let required_context = prompt_tokens
        .len()
        .checked_add(decode_steps)
        .context("prompt + decode step count overflow")?;
    let required_context_u32 = u32::try_from(required_context)
        .context("prompt + decode steps exceed the u32 context limit")?
        .max(1);
    let compilation_prefill_window =
        compilation_prefill_window(prompt_tokens.len(), required_context_u32)?;

    let (bundle_path, reused_bundle) = if let Some(bundle) = &args.bundle {
        (bundle.clone(), true)
    } else {
        let bundle = scratch.path().join("model.bundle");
        let architecture = loader.resolve_architecture(Some(&args.architecture), &checkpoint)?;
        eprintln!("architecture {architecture}");
        let mut overrides = dyninfer_core::MetadataMap::new();
        overrides.insert("max_kv".into(), json!(required_context_u32));
        overrides.insert("prefill_window".into(), json!(compilation_prefill_window));
        let paths = loader.compile_to_bundle_with_overrides(
            &architecture,
            &checkpoint,
            &args.target,
            &bundle,
            &CompileOptions {
                mode: "local-jit".into(),
                ..Default::default()
            },
            &overrides,
        )?;
        (paths.root, false)
    };
    let model = loader.load_bundle(&bundle_path, &checkpoint)?;
    ensure!(
        model.manifest.kv_cache.max_sequence_length >= required_context_u32,
        "bundle KV capacity {} cannot fit prompt ({}) + decode steps ({})",
        model.manifest.kv_cache.max_sequence_length,
        prompt_tokens.len(),
        decode_steps
    );
    if matches!(
        model.manifest.kv_cache.storage,
        KvCacheStorage::StaticGlobals
    ) {
        ensure!(
            prompt_tokens.len() <= model.manifest.prefill_window as usize,
            "bundle prefill window {} cannot fit {} prompt tokens",
            model.manifest.prefill_window,
            prompt_tokens.len()
        );
    }
    let vocab_size = model.metadata().vocabulary_size;
    validate_token_range(&prompt_tokens, vocab_size, "prompt")?;
    if let Some(tokens) = &explicit_decode {
        validate_token_range(tokens, vocab_size, "decode")?;
    }

    let matching_kv_type = llama_kv_type(model.manifest.kv_cache.element_type)?;
    let selected_kv_type = args
        .llama_kv_type
        .as_deref()
        .unwrap_or(matching_kv_type)
        .to_ascii_lowercase();
    ensure!(
        matches!(selected_kv_type.as_str(), "f32" | "f16" | "bf16"),
        "--llama-kv-type must be f32, f16, or bf16"
    );
    let mut caveats = Vec::new();
    if selected_kv_type != matching_kv_type {
        caveats.push(format!(
            "llama.cpp KV type override {selected_kv_type} differs from dyninfer {}",
            model.manifest.kv_cache.element_type
        ));
    }

    let reference_trace_path = artifacts_root.join("llama.cpp");
    let reference_stderr = artifacts_root.join("llama-reference.stderr.log");
    let mut runner_args = vec![
        OsString::from("run"),
        OsString::from("--checkpoint"),
        checkpoint.as_os_str().to_owned(),
        OsString::from("--output-dir"),
        reference_trace_path.as_os_str().to_owned(),
        OsString::from("--prompt-tokens"),
        OsString::from(join_tokens(&prompt_tokens)),
        OsString::from("--n-ctx"),
        OsString::from(required_context_u32.to_string()),
        OsString::from("--kv-type"),
        OsString::from(&selected_kv_type),
        OsString::from("--device"),
        OsString::from(&args.llama_device),
        OsString::from("--gpu-layers"),
        OsString::from(args.llama_gpu_layers.to_string()),
        OsString::from("--flash-attn"),
        OsString::from(args.llama_flash_attn.as_str()),
    ];
    if let Some(threads) = args.llama_threads {
        ensure!(threads > 0, "--llama-threads must be greater than zero");
        runner_args.push(OsString::from("--threads"));
        runner_args.push(OsString::from(threads.to_string()));
    }
    if let Some(tokens) = &explicit_decode {
        runner_args.push(OsString::from("--decode-tokens"));
        runner_args.push(OsString::from(join_tokens(tokens)));
    } else {
        runner_args.push(OsString::from("--decode-steps"));
        runner_args.push(OsString::from(decode_steps.to_string()));
    }
    run_companion(
        &args.llama_runner,
        &runner_args,
        &reference_stderr,
        "evaluate reference trajectory",
    )?;

    let mut reference = TraceReader::open(&reference_trace_path)?;
    ensure!(
        reference.header.engine == "llama.cpp",
        "reference trace engine must be llama.cpp"
    );
    ensure!(
        reference
            .header
            .checkpoint_sha256
            .eq_ignore_ascii_case(&checkpoint_sha256),
        "reference checkpoint digest {} does not match {}",
        reference.header.checkpoint_sha256,
        checkpoint_sha256
    );
    ensure!(
        reference.header.vocab_size == vocab_size,
        "vocabulary mismatch: llama.cpp {}, dyninfer {}",
        reference.header.vocab_size,
        vocab_size
    );
    ensure!(
        reference.header.prompt_tokens == prompt_tokens,
        "reference prompt token IDs differ from requested IDs"
    );
    ensure!(
        reference.header.decode_inputs.len() == decode_steps,
        "reference returned {} decode inputs, expected {decode_steps}",
        reference.header.decode_inputs.len()
    );
    if let Some(tokens) = &explicit_decode {
        ensure!(
            reference.header.decode_inputs == *tokens,
            "reference forced decode token IDs differ from requested IDs"
        );
    } else {
        for (index, token) in reference.header.decode_inputs.iter().copied().enumerate() {
            ensure!(
                reference.header.rows[index].argmax == token,
                "reference greedy token {index} does not equal preceding row argmax"
            );
        }
    }

    let decode_inputs = reference.header.decode_inputs.clone();
    let trace_rows = reference.header.rows.clone();
    let reference_metadata = reference.header.engine_metadata.clone();
    let mut session = model.create_session(SessionConfig {
        max_sequence_length: required_context_u32,
        ..SessionConfig::default()
    })?;
    let mut dyninfer_writer = if args.keep_traces.is_some() {
        Some(AtomicTraceWriter::create(
            &artifacts_root.join("dyninfer"),
            "dyninfer",
            &checkpoint_sha256,
            vocab_size,
            prompt_tokens.clone(),
            decode_inputs.clone(),
            json!({ "manifest": &model.manifest }),
        )?)
    } else {
        None
    };

    println!("step phase    pos  argmax(ref/dyn)  max_abs    rmse       rel_l2     cosine");
    let mut rows = Vec::with_capacity(trace_rows.len());
    for (step, trace_row) in trace_rows.iter().enumerate() {
        let dyninfer_logits = if step == 0 {
            session.prefill(&prompt_tokens)?.values
        } else {
            session.decode(decode_inputs[step - 1])?.values
        };
        let reference_logits = reference.next_row()?;
        let metrics = compare(&reference_logits, &dyninfer_logits, args.top_k)?;
        ensure!(
            metrics.reference_argmax == trace_row.argmax,
            "reference row {step} metadata/logit argmax mismatch"
        );
        if let Some(writer) = &mut dyninfer_writer {
            writer.write_row(
                trace_row.phase,
                trace_row.position,
                trace_row.input_token,
                &dyninfer_logits,
            )?;
        }
        println!(
            "{step:<4} {:<8} {:>4}  {}/{}  {:>10.4e} {:>10.4e} {:>10.4e} {:>11.8}",
            phase_name(trace_row.phase),
            trace_row.position,
            metrics.reference_argmax,
            metrics.dyninfer_argmax,
            metrics.max_absolute_error,
            metrics.rmse,
            metrics.relative_l2_error,
            metrics.cosine_similarity,
        );
        rows.push(ComparedRow {
            step,
            phase: trace_row.phase,
            position: trace_row.position,
            input_token: trace_row.input_token,
            forced_token: trace_row.forced_token,
            metrics,
        });
    }
    reference.finish()?;
    if let Some(writer) = dyninfer_writer {
        writer.finish()?;
    }

    let aggregate = aggregate(&rows)?;
    println!(
        "worst: max_abs={:.6e} (row {}) rmse={:.6e} (row {}) rel_l2={:.6e} (row {}) min_cosine={:.9} (row {})",
        aggregate.worst_max_absolute_error.value,
        aggregate.worst_max_absolute_error.row,
        aggregate.worst_rmse.value,
        aggregate.worst_rmse.row,
        aggregate.worst_relative_l2_error.value,
        aggregate.worst_relative_l2_error.row,
        aggregate.minimum_cosine_similarity.value,
        aggregate.minimum_cosine_similarity.row,
    );

    let generation_coherency = if args.generate_coherency {
        let coherency =
            generation_coherency(&model, &prompt_tokens, &decode_inputs, required_context_u32)?;
        if let Some(first) = &coherency.first_disagreement {
            println!(
                "generate coherency: first disagreement at step {} (ref={}, dyn={})",
                first.step, first.reference_token, first.dyninfer_token
            );
        } else {
            println!(
                "generate coherency: {} greedy tokens match",
                decode_inputs.len()
            );
        }
        Some(coherency)
    } else {
        None
    };

    let report = DriftReport {
        format: REPORT_FORMAT,
        version: REPORT_VERSION,
        checkpoint: CheckpointIdentity {
            canonical_path: checkpoint.display().to_string(),
            size_bytes: checkpoint_size,
            sha256: checkpoint_sha256,
            schema_fingerprint: model.catalog.schema_fingerprint.clone(),
        },
        prompt: prompt_provenance,
        prompt_tokens,
        decode_inputs,
        dyninfer: DyninferProvenance {
            bundle_path: bundle_path.display().to_string(),
            reused_bundle,
            manifest: model.manifest.clone(),
        },
        reference: ReferenceProvenance {
            runner: args.llama_runner.display().to_string(),
            requested_device: args.llama_device,
            requested_gpu_layers: args.llama_gpu_layers,
            requested_threads: args.llama_threads,
            requested_flash_attention: args.llama_flash_attn.as_str().into(),
            requested_kv_type: selected_kv_type,
            trace_metadata: reference_metadata,
        },
        comparison_caveats: caveats,
        thresholds,
        rows,
        aggregate,
        generation_coherency,
    };

    if let Some(path) = &args.json_out {
        write_json_atomic(path, &report)?;
        println!("report: {}", path.display());
    }
    let failures = report.thresholds.failures(&report.aggregate);
    if !failures.is_empty() {
        bail!("drift thresholds failed: {}", failures.join("; "));
    }
    Ok(())
}

fn generation_coherency(
    model: &dyninfer_runtime::LoadedModel,
    prompt_tokens: &[u32],
    reference_tokens: &[u32],
    max_sequence_length: u32,
) -> Result<GenerationCoherency> {
    let mut session = model.create_session(SessionConfig {
        max_sequence_length,
        ..SessionConfig::default()
    })?;
    let mut dyninfer_tokens = Vec::with_capacity(reference_tokens.len());
    let mut token = session.prefill_argmax(prompt_tokens)?;
    for step in 0..reference_tokens.len() {
        dyninfer_tokens.push(token);
        if step + 1 < reference_tokens.len() {
            token = session.decode_argmax(token)?;
        }
    }
    let first_disagreement = reference_tokens
        .iter()
        .copied()
        .zip(dyninfer_tokens.iter().copied())
        .enumerate()
        .find(|(_, (reference, dyninfer))| reference != dyninfer)
        .map(
            |(step, (reference_token, dyninfer_token))| TokenDisagreement {
                step,
                reference_token,
                dyninfer_token,
            },
        );
    Ok(GenerationCoherency {
        reference_tokens: reference_tokens.to_vec(),
        dyninfer_tokens,
        coherent: first_disagreement.is_none(),
        first_disagreement,
    })
}

fn aggregate(rows: &[ComparedRow]) -> Result<AggregateMetrics> {
    ensure!(!rows.is_empty(), "cannot aggregate an empty comparison");
    Ok(AggregateMetrics {
        worst_max_absolute_error: locate_max(rows, |row| row.metrics.max_absolute_error),
        worst_rmse: locate_max(rows, |row| row.metrics.rmse),
        worst_relative_l2_error: locate_max(rows, |row| row.metrics.relative_l2_error),
        minimum_cosine_similarity: locate_min(rows, |row| row.metrics.cosine_similarity),
        worst_softmax_kl: locate_max(rows, |row| row.metrics.softmax_kl_reference_dyninfer),
        worst_probability_delta: locate_max(rows, |row| row.metrics.max_probability_delta),
        first_argmax_disagreement: rows.iter().position(|row| !row.metrics.argmax_match),
    })
}

fn locate_max(rows: &[ComparedRow], value: impl Fn(&ComparedRow) -> f64) -> LocatedValue {
    rows.iter()
        .enumerate()
        .map(|(row, metrics)| LocatedValue {
            row,
            value: value(metrics),
        })
        .max_by(|left, right| {
            left.value
                .total_cmp(&right.value)
                .then_with(|| right.row.cmp(&left.row))
        })
        .expect("non-empty rows")
}

fn locate_min(rows: &[ComparedRow], value: impl Fn(&ComparedRow) -> f64) -> LocatedValue {
    rows.iter()
        .enumerate()
        .map(|(row, metrics)| LocatedValue {
            row,
            value: value(metrics),
        })
        .min_by(|left, right| {
            left.value
                .total_cmp(&right.value)
                .then_with(|| left.row.cmp(&right.row))
        })
        .expect("non-empty rows")
}

fn phase_name(phase: TracePhase) -> &'static str {
    match phase {
        TracePhase::Prefill => "prefill",
        TracePhase::Decode => "decode",
    }
}

fn llama_kv_type(element_type: ScalarType) -> Result<&'static str> {
    match element_type {
        ScalarType::F32 => Ok("f32"),
        ScalarType::F16 => Ok("f16"),
        ScalarType::Bf16 => Ok("bf16"),
        other => bail!("llama.cpp companion cannot match dyninfer KV type {other}"),
    }
}

fn compilation_prefill_window(prompt_tokens: usize, required_context: u32) -> Result<u32> {
    if required_context > PAGED_PREFILL_CHUNK_SIZE {
        Ok(PAGED_PREFILL_CHUNK_SIZE)
    } else {
        u32::try_from(prompt_tokens).context("prompt length exceeds the u32 limit")
    }
}

fn parse_comma_tokens(input: &str, option: &str) -> Result<Vec<u32>> {
    let input = input.trim();
    ensure!(!input.is_empty(), "{option} is empty");
    input
        .split(',')
        .enumerate()
        .map(|(index, token)| {
            token
                .trim()
                .parse::<u32>()
                .with_context(|| format!("{option} token {index} is not a u32: {token:?}"))
        })
        .collect()
}

fn parse_token_file(path: &Path) -> Result<Vec<u32>> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("read token file {}", path.display()))?;
    let trimmed = contents.trim();
    ensure!(
        !trimmed.is_empty(),
        "token file {} is empty",
        path.display()
    );
    if trimmed.starts_with('[') {
        let tokens: Vec<u32> = serde_json::from_str(trimmed)
            .with_context(|| format!("parse JSON token array {}", path.display()))?;
        ensure!(
            !tokens.is_empty(),
            "token file {} contains an empty array",
            path.display()
        );
        Ok(tokens)
    } else {
        trimmed
            .split_whitespace()
            .enumerate()
            .map(|(index, token)| {
                token.parse::<u32>().with_context(|| {
                    format!(
                        "token file {} token {index} is not a u32: {token:?}",
                        path.display()
                    )
                })
            })
            .collect()
    }
}

fn validate_token_range(tokens: &[u32], vocab_size: u32, kind: &str) -> Result<()> {
    if let Some((index, token)) = tokens
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token)| *token >= vocab_size)
    {
        bail!("{kind} token {index} ({token}) is outside vocabulary {vocab_size}");
    }
    Ok(())
}

fn join_tokens(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn run_companion(runner: &Path, args: &[OsString], stderr_path: &Path, stage: &str) -> Result<()> {
    if let Some(parent) = stderr_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let stderr = File::create(stderr_path)
        .with_context(|| format!("create companion diagnostic {}", stderr_path.display()))?;
    let status = Command::new(runner)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr))
        .status()
        .with_context(|| format!("launch llama.cpp companion {} to {stage}", runner.display()))?;
    if !status.success() {
        let summary = stderr_tail(stderr_path, 16 * 1024)
            .unwrap_or_else(|error| format!("<could not read stderr: {error}>"));
        bail!(
            "llama.cpp companion failed to {stage} ({status}); diagnostics: {}\n{}",
            stderr_path.display(),
            summary.trim()
        );
    }
    Ok(())
}

fn stderr_tail(path: &Path, limit: u64) -> Result<String> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    if length > limit {
        file.seek(SeekFrom::Start(length - limit))?;
    }
    let mut output = Vec::new();
    file.read_to_end(&mut output)?;
    Ok(String::from_utf8_lossy(&output).into_owned())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create report directory {}", parent.display()))?;
    let name = path
        .file_name()
        .and_then(OsStr::to_str)
        .context("report path must have a UTF-8 file name")?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    let result = (|| -> Result<()> {
        let mut file = File::create(&temporary)?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.with_context(|| format!("write JSON report {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compared_row(
        step: usize,
        max: f64,
        rmse: f64,
        relative: f64,
        cosine: f64,
        matches: bool,
    ) -> ComparedRow {
        ComparedRow {
            step,
            phase: if step == 0 {
                TracePhase::Prefill
            } else {
                TracePhase::Decode
            },
            position: step as u64,
            input_token: 0,
            forced_token: None,
            metrics: RowMetrics {
                max_absolute_error: max,
                mean_absolute_error: 0.0,
                rmse,
                mean_signed_delta: 0.0,
                relative_l2_error: relative,
                cosine_similarity: cosine,
                centered_rmse: 0.0,
                reference_argmax: 0,
                dyninfer_argmax: u32::from(!matches),
                argmax_match: matches,
                top_k: 1,
                top_k_overlap: usize::from(matches),
                top_k_overlap_fraction: f64::from(matches),
                softmax_kl_reference_dyninfer: rmse,
                max_probability_delta: relative,
                largest_deltas: Vec::new(),
                reference_top: Vec::new(),
                dyninfer_top: Vec::new(),
            },
        }
    }

    #[test]
    fn thresholds_report_first_and_worst_rows() {
        let rows = vec![
            compared_row(0, 1.0, 0.2, 0.1, 0.99, true),
            compared_row(1, 3.0, 0.1, 0.4, 0.90, false),
            compared_row(2, 2.0, 0.8, 0.2, 0.95, false),
        ];
        let aggregate = aggregate(&rows).unwrap();
        assert_eq!(aggregate.worst_max_absolute_error.row, 1);
        assert_eq!(aggregate.worst_rmse.row, 2);
        assert_eq!(aggregate.worst_relative_l2_error.row, 1);
        assert_eq!(aggregate.minimum_cosine_similarity.row, 1);
        assert_eq!(aggregate.first_argmax_disagreement, Some(1));
        let failures = Thresholds {
            max_absolute_error: Some(2.5),
            max_rmse: Some(1.0),
            max_relative_l2_error: None,
            min_cosine_similarity: Some(0.92),
            require_argmax_match: true,
        }
        .failures(&aggregate);
        assert_eq!(failures.len(), 3);
    }

    #[test]
    fn parses_exact_token_inputs() {
        assert_eq!(
            parse_comma_tokens("1, 2,3", "--tokens").unwrap(),
            vec![1, 2, 3]
        );
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("tokens.json");
        fs::write(&path, "[4, 5, 6]").unwrap();
        assert_eq!(parse_token_file(&path).unwrap(), vec![4, 5, 6]);
    }

    #[test]
    fn sizes_compiled_prefill_window_for_prompt_or_paged_geometry() {
        assert_eq!(compilation_prefill_window(33, 37).unwrap(), 33);
        assert_eq!(compilation_prefill_window(512, 512).unwrap(), 512);
        assert_eq!(compilation_prefill_window(513, 517).unwrap(), 512);
    }
}
