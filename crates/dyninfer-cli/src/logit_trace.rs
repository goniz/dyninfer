//! Versioned metadata + streaming little-endian F32 logit traces.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};

pub const TRACE_FORMAT: &str = "dyninfer.logit-trace";
pub const TRACE_VERSION: u32 = 1;
pub const TOKEN_TRACE_FORMAT: &str = "dyninfer.tokenized-prompt";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TracePhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceRow {
    pub phase: TracePhase,
    pub position: u64,
    pub input_token: u32,
    pub argmax: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forced_token: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHeader {
    pub format: String,
    pub version: u32,
    pub engine: String,
    pub checkpoint_sha256: String,
    pub vocab_size: u32,
    pub prompt_tokens: Vec<u32>,
    pub decode_inputs: Vec<u32>,
    pub logits_dtype: String,
    pub logits_byte_order: String,
    pub rows: Vec<TraceRow>,
    pub logits_file: String,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub engine_metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizedPrompt {
    pub format: String,
    pub version: u32,
    pub checkpoint_sha256: String,
    pub prompt_tokens: Vec<u32>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub engine_metadata: Value,
}

impl TokenizedPrompt {
    pub fn read(path: &Path) -> Result<Self> {
        let tokenized: Self = serde_json::from_slice(
            &fs::read(path).with_context(|| format!("read {}", path.display()))?,
        )
        .with_context(|| format!("parse {}", path.display()))?;
        ensure!(
            tokenized.format == TOKEN_TRACE_FORMAT,
            "unsupported tokenization format {:?}",
            tokenized.format
        );
        ensure!(
            tokenized.version == TRACE_VERSION,
            "unsupported tokenization version {}",
            tokenized.version
        );
        ensure!(
            !tokenized.prompt_tokens.is_empty(),
            "llama.cpp tokenization produced no tokens"
        );
        ensure_sha256(&tokenized.checkpoint_sha256)?;
        Ok(tokenized)
    }
}

pub struct TraceReader {
    pub header: TraceHeader,
    logits: BufReader<File>,
    next_row: usize,
}

impl TraceReader {
    pub fn open(directory: &Path) -> Result<Self> {
        let header_path = directory.join("trace.json");
        let header: TraceHeader = serde_json::from_slice(
            &fs::read(&header_path)
                .with_context(|| format!("read trace header {}", header_path.display()))?,
        )
        .with_context(|| format!("parse trace header {}", header_path.display()))?;
        validate_header(&header)?;

        let logits_path = directory.join(&header.logits_file);
        let expected_bytes = (header.rows.len() as u64)
            .checked_mul(u64::from(header.vocab_size))
            .and_then(|value| value.checked_mul(4))
            .context("trace byte size overflows u64")?;
        let actual_bytes = fs::metadata(&logits_path)
            .with_context(|| format!("stat trace logits {}", logits_path.display()))?
            .len();
        ensure!(
            actual_bytes == expected_bytes,
            "trace logits size mismatch: expected {expected_bytes} bytes, found {actual_bytes} (truncated or trailing data)"
        );
        let logits = BufReader::new(
            File::open(&logits_path)
                .with_context(|| format!("open trace logits {}", logits_path.display()))?,
        );
        Ok(Self {
            header,
            logits,
            next_row: 0,
        })
    }

    pub fn next_row(&mut self) -> Result<Vec<f32>> {
        ensure!(
            self.next_row < self.header.rows.len(),
            "trace has no row {}",
            self.next_row
        );
        let mut bytes = vec![0u8; self.header.vocab_size as usize * 4];
        self.logits
            .read_exact(&mut bytes)
            .with_context(|| format!("read trace row {}", self.next_row))?;
        let mut values = Vec::with_capacity(self.header.vocab_size as usize);
        for (token, bytes) in bytes.chunks_exact(4).enumerate() {
            let value = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
            ensure!(
                value.is_finite(),
                "trace row {} token {} is non-finite: {}",
                self.next_row,
                token,
                value
            );
            values.push(value);
        }
        let actual_argmax = argmax(&values);
        let recorded_argmax = self.header.rows[self.next_row].argmax;
        ensure!(
            actual_argmax == recorded_argmax,
            "trace row {} argmax mismatch: metadata {}, logits {}",
            self.next_row,
            recorded_argmax,
            actual_argmax
        );
        self.next_row += 1;
        Ok(values)
    }

    pub fn finish(&mut self) -> Result<()> {
        ensure!(
            self.next_row == self.header.rows.len(),
            "trace has {} unread rows",
            self.header.rows.len() - self.next_row
        );
        let mut trailing = [0u8; 1];
        ensure!(
            self.logits.read(&mut trailing)? == 0,
            "trace contains trailing logit bytes"
        );
        Ok(())
    }
}

pub struct AtomicTraceWriter {
    destination: PathBuf,
    temporary: PathBuf,
    header: TraceHeader,
    logits: Option<BufWriter<File>>,
    finished: bool,
}

impl AtomicTraceWriter {
    pub fn create(
        destination: &Path,
        engine: impl Into<String>,
        checkpoint_sha256: impl Into<String>,
        vocab_size: u32,
        prompt_tokens: Vec<u32>,
        decode_inputs: Vec<u32>,
        engine_metadata: Value,
    ) -> Result<Self> {
        ensure!(
            !destination.exists(),
            "trace output {} already exists",
            destination.display()
        );
        let parent = destination.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)
            .with_context(|| format!("create trace parent {}", parent.display()))?;
        let name = destination
            .file_name()
            .and_then(|name| name.to_str())
            .context("trace output must have a UTF-8 file name")?;
        let temporary = (0u32..1000)
            .map(|counter| parent.join(format!(".{name}.tmp-{}-{counter}", std::process::id())))
            .find(|candidate| !candidate.exists())
            .context("could not allocate a temporary trace directory")?;
        fs::create_dir(&temporary)
            .with_context(|| format!("create temporary trace {}", temporary.display()))?;
        let logits_path = temporary.join("logits.f32le");
        let logits = BufWriter::new(
            File::create(&logits_path)
                .with_context(|| format!("create trace logits {}", logits_path.display()))?,
        );
        let header = TraceHeader {
            format: TRACE_FORMAT.into(),
            version: TRACE_VERSION,
            engine: engine.into(),
            checkpoint_sha256: checkpoint_sha256.into(),
            vocab_size,
            prompt_tokens,
            decode_inputs,
            logits_dtype: "f32".into(),
            logits_byte_order: "little".into(),
            rows: Vec::new(),
            logits_file: "logits.f32le".into(),
            engine_metadata,
        };
        validate_trajectory(&header.prompt_tokens, &header.decode_inputs, vocab_size)?;
        ensure_sha256(&header.checkpoint_sha256)?;
        Ok(Self {
            destination: destination.to_path_buf(),
            temporary,
            header,
            logits: Some(logits),
            finished: false,
        })
    }

    pub fn write_row(
        &mut self,
        phase: TracePhase,
        position: u64,
        input_token: u32,
        values: &[f32],
    ) -> Result<()> {
        ensure!(
            values.len() == self.header.vocab_size as usize,
            "trace row width mismatch: expected {}, found {}",
            self.header.vocab_size,
            values.len()
        );
        let row_index = self.header.rows.len();
        let (expected_phase, expected_position, expected_input) =
            expected_row(&self.header, row_index)?;
        ensure!(
            phase == expected_phase,
            "trace row {row_index} phase mismatch"
        );
        ensure!(
            position == expected_position,
            "trace row {row_index} position mismatch: expected {expected_position}, found {position}"
        );
        ensure!(
            input_token == expected_input,
            "trace row {row_index} input mismatch: expected {expected_input}, found {input_token}"
        );
        let logits = self
            .logits
            .as_mut()
            .context("trace writer is already finished")?;
        for (token, value) in values.iter().copied().enumerate() {
            ensure!(
                value.is_finite(),
                "trace row {row_index} token {token} is non-finite: {value}"
            );
            logits.write_all(&value.to_le_bytes())?;
        }
        self.header.rows.push(TraceRow {
            phase,
            position,
            input_token,
            argmax: argmax(values),
            forced_token: self.header.decode_inputs.get(row_index).copied(),
        });
        Ok(())
    }

    pub fn finish(mut self) -> Result<PathBuf> {
        ensure!(
            self.header.rows.len() == self.header.decode_inputs.len() + 1,
            "trace row count mismatch: expected {}, found {}",
            self.header.decode_inputs.len() + 1,
            self.header.rows.len()
        );
        validate_header(&self.header)?;
        let mut logits = self
            .logits
            .take()
            .context("trace writer is already finished")?;
        logits.flush()?;
        logits.get_ref().sync_all()?;
        drop(logits);

        let header_path = self.temporary.join("trace.json");
        let mut header_file = BufWriter::new(File::create(&header_path)?);
        serde_json::to_writer_pretty(&mut header_file, &self.header)?;
        header_file.write_all(b"\n")?;
        header_file.flush()?;
        header_file.get_ref().sync_all()?;
        drop(header_file);
        fs::rename(&self.temporary, &self.destination).with_context(|| {
            format!(
                "publish trace {} -> {}",
                self.temporary.display(),
                self.destination.display()
            )
        })?;
        self.finished = true;
        Ok(self.destination.clone())
    }
}

impl Drop for AtomicTraceWriter {
    fn drop(&mut self) {
        if !self.finished && self.temporary.exists() {
            let _ = fs::remove_dir_all(&self.temporary);
        }
    }
}

fn validate_header(header: &TraceHeader) -> Result<()> {
    ensure!(
        header.format == TRACE_FORMAT,
        "unsupported trace format {:?}",
        header.format
    );
    ensure!(
        header.version == TRACE_VERSION,
        "unsupported trace version {}",
        header.version
    );
    ensure!(!header.engine.trim().is_empty(), "trace engine is empty");
    ensure_sha256(&header.checkpoint_sha256)?;
    ensure!(header.vocab_size > 0, "trace vocabulary is empty");
    ensure!(
        header.logits_dtype == "f32",
        "unsupported logits dtype {:?}",
        header.logits_dtype
    );
    ensure!(
        header.logits_byte_order == "little",
        "unsupported logits byte order {:?}",
        header.logits_byte_order
    );
    let logits_path = Path::new(&header.logits_file);
    ensure!(
        logits_path.components().count() == 1
            && matches!(logits_path.components().next(), Some(Component::Normal(_))),
        "logits_file must be a plain relative file name"
    );
    validate_trajectory(
        &header.prompt_tokens,
        &header.decode_inputs,
        header.vocab_size,
    )?;
    ensure!(
        header.rows.len() == header.decode_inputs.len() + 1,
        "trace row count mismatch: expected {}, found {}",
        header.decode_inputs.len() + 1,
        header.rows.len()
    );
    for (index, row) in header.rows.iter().enumerate() {
        let (phase, position, input) = expected_row(header, index)?;
        ensure!(
            row.phase == phase,
            "trace row {index} has inconsistent phase"
        );
        ensure!(
            row.position == position,
            "trace row {index} has discontinuous position {} (expected {position})",
            row.position
        );
        ensure!(
            row.input_token == input,
            "trace row {index} input token {} does not match trajectory token {input}",
            row.input_token
        );
        ensure!(
            row.argmax < header.vocab_size,
            "trace row {index} argmax {} is outside vocabulary {}",
            row.argmax,
            header.vocab_size
        );
        ensure!(
            row.forced_token == header.decode_inputs.get(index).copied(),
            "trace row {index} forced token is inconsistent with decode_inputs"
        );
    }
    Ok(())
}

fn validate_trajectory(prompt: &[u32], decode: &[u32], vocab_size: u32) -> Result<()> {
    ensure!(!prompt.is_empty(), "trace prompt token list is empty");
    for (kind, tokens) in [("prompt", prompt), ("decode", decode)] {
        if let Some((index, token)) = tokens
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| *token >= vocab_size)
        {
            bail!("{kind} token {index} ({token}) is outside vocabulary {vocab_size}");
        }
    }
    Ok(())
}

fn expected_row(header: &TraceHeader, index: usize) -> Result<(TracePhase, u64, u32)> {
    if index == 0 {
        let position = u64::try_from(header.prompt_tokens.len() - 1)?;
        return Ok((
            TracePhase::Prefill,
            position,
            *header.prompt_tokens.last().context("empty prompt")?,
        ));
    }
    let decode_index = index - 1;
    let position = u64::try_from(header.prompt_tokens.len())?
        .checked_add(u64::try_from(decode_index)?)
        .context("trace position overflows u64")?;
    let input = *header
        .decode_inputs
        .get(decode_index)
        .with_context(|| format!("trace has unexpected row {index}"))?;
    Ok((TracePhase::Decode, position, input))
}

fn ensure_sha256(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "checkpoint_sha256 must contain 64 hexadecimal characters"
    );
    Ok(())
}

fn argmax(values: &[f32]) -> u32 {
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().copied().enumerate() {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    best_index as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn write_valid(destination: &Path) {
        let mut writer = AtomicTraceWriter::create(
            destination,
            "test",
            SHA,
            3,
            vec![1, 2],
            vec![0],
            Value::Null,
        )
        .unwrap();
        writer
            .write_row(TracePhase::Prefill, 1, 2, &[0.0, 2.0, 1.0])
            .unwrap();
        writer
            .write_row(TracePhase::Decode, 2, 0, &[3.0, 2.0, 1.0])
            .unwrap();
        writer.finish().unwrap();
    }

    #[test]
    fn round_trip_streams_rows() {
        let temporary = tempfile::tempdir().unwrap();
        let trace = temporary.path().join("trace");
        write_valid(&trace);
        let mut reader = TraceReader::open(&trace).unwrap();
        assert_eq!(reader.next_row().unwrap(), vec![0.0, 2.0, 1.0]);
        assert_eq!(reader.next_row().unwrap(), vec![3.0, 2.0, 1.0]);
        reader.finish().unwrap();
    }

    #[test]
    fn rejects_truncated_and_trailing_binary_data() {
        for extra in [-1isize, 1] {
            let temporary = tempfile::tempdir().unwrap();
            let trace = temporary.path().join("trace");
            write_valid(&trace);
            let logits = trace.join("logits.f32le");
            let mut bytes = fs::read(&logits).unwrap();
            if extra < 0 {
                bytes.pop();
            } else {
                bytes.push(0);
            }
            fs::write(&logits, bytes).unwrap();
            assert!(TraceReader::open(&trace).is_err());
        }
    }

    #[test]
    fn rejects_protocol_and_trajectory_mismatches() {
        for mutation in ["version", "endianness", "token", "position"] {
            let temporary = tempfile::tempdir().unwrap();
            let trace = temporary.path().join("trace");
            write_valid(&trace);
            let header_path = trace.join("trace.json");
            let mut header: Value =
                serde_json::from_slice(&fs::read(&header_path).unwrap()).unwrap();
            match mutation {
                "version" => header["version"] = Value::from(999),
                "endianness" => header["logits_byte_order"] = Value::from("big"),
                "token" => header["prompt_tokens"][0] = Value::from(99),
                "position" => header["rows"][1]["position"] = Value::from(9),
                _ => unreachable!(),
            }
            fs::write(&header_path, serde_json::to_vec(&header).unwrap()).unwrap();
            assert!(TraceReader::open(&trace).is_err(), "mutation {mutation}");
        }
    }

    #[test]
    fn rejects_non_finite_logits_and_argmax_lies() {
        for mutation in ["nan", "argmax"] {
            let temporary = tempfile::tempdir().unwrap();
            let trace = temporary.path().join("trace");
            write_valid(&trace);
            if mutation == "nan" {
                let logits = trace.join("logits.f32le");
                let mut bytes = fs::read(&logits).unwrap();
                bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
                fs::write(logits, bytes).unwrap();
            } else {
                let header_path = trace.join("trace.json");
                let mut header: Value =
                    serde_json::from_slice(&fs::read(&header_path).unwrap()).unwrap();
                header["rows"][0]["argmax"] = Value::from(2);
                fs::write(header_path, serde_json::to_vec(&header).unwrap()).unwrap();
            }
            let mut reader = TraceReader::open(&trace).unwrap();
            assert!(reader.next_row().is_err(), "mutation {mutation}");
        }
    }
}
