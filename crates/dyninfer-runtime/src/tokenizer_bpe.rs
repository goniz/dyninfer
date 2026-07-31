//! Minimal HuggingFace `tokenizer.json` BPE loader.
//!
//! Supports:
//! - OpenLLaMA / Maykeye TinyLLama: SentencePiece-style BPE + `▁` + byte fallback
//! - Qwen2/Qwen3: GPT-2 ByteLevel BPE (`Ġ` space mark)
//!
//! Keeps the runtime free of the `tokenizers` crate (C++/esaxx).

use dyninfer_error::{DynInferError, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

const SPACE_MARK: char = '\u{2581}'; // ▁ (SentencePiece)
const GPT2_SPACE: char = '\u{0120}'; // Ġ (ByteLevel)

#[derive(Debug, Deserialize)]
struct TokenizerFile {
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    model: Model,
    #[serde(default)]
    decoder: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Debug, Deserialize)]
struct Model {
    vocab: HashMap<String, u32>,
    merges: Vec<MergeEntry>,
    #[serde(default)]
    byte_fallback: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MergeEntry {
    Str(String),
    Pair(Vec<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenizerKind {
    /// SentencePiece-style (`▁` prefix spaces).
    SentencePiece,
    /// GPT-2 / Qwen ByteLevel (`Ġ` / unicode byte map).
    ByteLevel,
}

/// Loaded BPE tokenizer.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    kind: TokenizerKind,
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
    /// merge pair -> rank (lower = earlier / higher priority)
    merges: HashMap<(String, String), u32>,
    byte_fallback: bool,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    special_ids: Vec<u32>,
    /// ByteLevel: unicode char -> byte
    byte_decoder: HashMap<char, u8>,
}

fn gpt2_bytes_to_unicode() -> &'static (HashMap<u8, char>, HashMap<char, u8>) {
    static MAPS: OnceLock<(HashMap<u8, char>, HashMap<char, u8>)> = OnceLock::new();
    MAPS.get_or_init(|| {
        let mut bs: Vec<u8> = (b'!'..=b'~').collect();
        bs.extend(0xA1u8..=0xACu8);
        bs.extend(0xAEu8..=0xFFu8);
        let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
        let mut n = 0u32;
        for b in 0u8..=255 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut enc = HashMap::with_capacity(256);
        let mut dec = HashMap::with_capacity(256);
        for (b, c) in bs.into_iter().zip(cs.into_iter()) {
            let ch = char::from_u32(c).unwrap();
            enc.insert(b, ch);
            dec.insert(ch, b);
        }
        (enc, dec)
    })
}

impl BpeTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };
        let bytes = fs::read(&file).map_err(|e| {
            DynInferError::io_path(file.display().to_string(), format!("read tokenizer: {e}"))
        })?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let tf: TokenizerFile = serde_json::from_slice(bytes).map_err(|e| {
            DynInferError::io(format!("tokenizer.json parse failed: {e}"))
        })?;
        if tf.model.vocab.is_empty() {
            return Err(DynInferError::io("tokenizer vocab is empty"));
        }
        let max_id = tf.model.vocab.values().copied().max().unwrap_or(0);
        let mut id_to_token = vec![String::new(); (max_id as usize) + 1];
        for (tok, id) in &tf.model.vocab {
            let i = *id as usize;
            if i >= id_to_token.len() {
                id_to_token.resize(i + 1, String::new());
            }
            id_to_token[i] = tok.clone();
        }
        for a in &tf.added_tokens {
            let i = a.id as usize;
            if i >= id_to_token.len() {
                id_to_token.resize(i + 1, String::new());
            }
            if id_to_token[i].is_empty() {
                id_to_token[i] = a.content.clone();
            }
        }

        let mut merges = HashMap::new();
        for (rank, entry) in tf.model.merges.iter().enumerate() {
            let (a, b) = match entry {
                MergeEntry::Str(s) => {
                    let mut parts = s.splitn(2, ' ');
                    let a = parts.next().unwrap_or("").to_string();
                    let b = parts.next().unwrap_or("").to_string();
                    (a, b)
                }
                MergeEntry::Pair(v) if v.len() >= 2 => (v[0].clone(), v[1].clone()),
                MergeEntry::Pair(_) => continue,
            };
            merges.insert((a, b), rank as u32);
        }

        let kind = detect_kind(&tf);
        let special_ids: Vec<u32> = tf
            .added_tokens
            .iter()
            .filter(|t| t.special)
            .map(|t| t.id)
            .collect();

        let bos_id = tf
            .added_tokens
            .iter()
            .find(|t| t.content == "<s>" || t.content == "<|begin_of_text|>")
            .map(|t| t.id);
        let eos_id = tf
            .added_tokens
            .iter()
            .find(|t| {
                t.content == "</s>"
                    || t.content == "<|endoftext|>"
                    || t.content == "<|im_end|>"
                    || t.content == "<|eot_id|>"
            })
            .map(|t| t.id)
            // Prefer chat end token when both exist: last matching above wins via find;
            // Qwen lists endoftext before im_end — prefer im_end / endoftext explicitly.
            .or_else(|| {
                tf.added_tokens
                    .iter()
                    .find(|t| t.content == "<|endoftext|>")
                    .map(|t| t.id)
            });

        // For Qwen chat tokenizers, prefer <|im_end|> over <|endoftext|> when both exist.
        let eos_id = tf
            .added_tokens
            .iter()
            .find(|t| t.content == "<|im_end|>")
            .map(|t| t.id)
            .or(eos_id);

        let byte_decoder = if kind == TokenizerKind::ByteLevel {
            gpt2_bytes_to_unicode().1.clone()
        } else {
            HashMap::new()
        };

        Ok(Self {
            kind,
            token_to_id: tf.model.vocab,
            id_to_token,
            merges,
            byte_fallback: tf.model.byte_fallback,
            bos_id,
            eos_id,
            special_ids,
            byte_decoder,
        })
    }

    pub fn is_byte_level(&self) -> bool {
        self.kind == TokenizerKind::ByteLevel
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    fn normalize_sp(&self, text: &str) -> String {
        let mut s = String::with_capacity(text.len() + 1);
        s.push(SPACE_MARK);
        for ch in text.chars() {
            if ch == ' ' {
                s.push(SPACE_MARK);
            } else {
                s.push(ch);
            }
        }
        s
    }

    fn piece_id(&self, piece: &str) -> Option<u32> {
        self.token_to_id.get(piece).copied()
    }

    fn initial_pieces_sp(&self, normalized: &str) -> Vec<String> {
        let mut pieces = Vec::new();
        for ch in normalized.chars() {
            let s = ch.to_string();
            if self.piece_id(&s).is_some() {
                pieces.push(s);
            } else if self.byte_fallback {
                for b in s.as_bytes() {
                    pieces.push(format!("<0x{b:02X}>"));
                }
            } else {
                pieces.push("<unk>".into());
            }
        }
        pieces
    }

    fn initial_pieces_byte_level(&self, text: &str) -> Vec<String> {
        let enc = &gpt2_bytes_to_unicode().0;
        text.as_bytes()
            .iter()
            .map(|b| enc[b].to_string())
            .collect()
    }

    fn bpe_merge(&self, mut pieces: Vec<String>) -> Vec<String> {
        if pieces.len() <= 1 {
            return pieces;
        }
        loop {
            let mut best_rank = u32::MAX;
            let mut best_i = None;
            for i in 0..pieces.len().saturating_sub(1) {
                if let Some(&rank) = self.merges.get(&(pieces[i].clone(), pieces[i + 1].clone())) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = Some(i);
                    }
                }
            }
            let Some(i) = best_i else {
                break;
            };
            let merged = format!("{}{}", pieces[i], pieces[i + 1]);
            pieces[i] = merged;
            pieces.remove(i + 1);
        }
        pieces
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let pieces = match self.kind {
            TokenizerKind::SentencePiece => {
                let normalized = self.normalize_sp(text);
                self.bpe_merge(self.initial_pieces_sp(&normalized))
            }
            TokenizerKind::ByteLevel => {
                self.bpe_merge(self.initial_pieces_byte_level(text))
            }
        };
        let mut ids = Vec::with_capacity(pieces.len() + 1);
        if add_special_tokens {
            if let Some(bos) = self.bos_id {
                ids.push(bos);
            }
        }
        for p in pieces {
            let id = self
                .piece_id(&p)
                .or_else(|| self.piece_id("<unk>"))
                .unwrap_or(0);
            ids.push(id);
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self.kind {
            TokenizerKind::SentencePiece => self.decode_sp(ids, skip_special_tokens),
            TokenizerKind::ByteLevel => self.decode_byte_level(ids, skip_special_tokens),
        }
    }

    fn is_special(&self, id: u32) -> bool {
        Some(id) == self.bos_id
            || Some(id) == self.eos_id
            || self.special_ids.contains(&id)
    }

    fn decode_sp(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut bytes: Vec<u8> = Vec::new();
        let mut text_acc = String::new();
        let flush_bytes = |bytes: &mut Vec<u8>, out: &mut String| {
            if !bytes.is_empty() {
                out.push_str(&String::from_utf8_lossy(bytes));
                bytes.clear();
            }
        };

        for &id in ids {
            if skip_special_tokens && self.is_special(id) {
                continue;
            }
            let tok = self
                .id_to_token
                .get(id as usize)
                .map(|s| s.as_str())
                .unwrap_or("");
            if tok.is_empty() {
                continue;
            }
            if let Some(hex) = tok.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    bytes.push(b);
                    continue;
                }
            }
            flush_bytes(&mut bytes, &mut text_acc);
            text_acc.push_str(tok);
        }
        flush_bytes(&mut bytes, &mut text_acc);

        let decoded = text_acc.replace(SPACE_MARK, " ");
        Ok(decoded.trim_start().to_string())
    }

    fn decode_byte_level(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut text = String::new();
        for &id in ids {
            if skip_special_tokens && self.is_special(id) {
                continue;
            }
            let tok = self
                .id_to_token
                .get(id as usize)
                .map(|s| s.as_str())
                .unwrap_or("");
            if tok.is_empty() {
                continue;
            }
            // Special tokens are literal unicode, not byte-encoded.
            if tok.starts_with("<|") && tok.ends_with("|>") {
                if !skip_special_tokens {
                    text.push_str(tok);
                }
                continue;
            }
            text.push_str(tok);
        }
        let mut bytes = Vec::with_capacity(text.len());
        for ch in text.chars() {
            if let Some(&b) = self.byte_decoder.get(&ch) {
                bytes.push(b);
            } else if ch == GPT2_SPACE {
                bytes.push(b' ');
            } else {
                // Fallback: encode char as UTF-8
                let mut buf = [0u8; 4];
                bytes.extend(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

fn detect_kind(tf: &TokenizerFile) -> TokenizerKind {
    if let Some(dec) = &tf.decoder {
        if decoder_is_byte_level(dec) {
            return TokenizerKind::ByteLevel;
        }
    }
    let has_gpt2 = tf.model.vocab.keys().any(|k| k.starts_with('Ġ'));
    let has_sp = tf.model.vocab.keys().any(|k| k.starts_with('▁'));
    if has_gpt2 && !has_sp {
        TokenizerKind::ByteLevel
    } else {
        TokenizerKind::SentencePiece
    }
}

fn decoder_is_byte_level(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Object(map) => {
            if map.get("type").and_then(|t| t.as_str()) == Some("ByteLevel") {
                return true;
            }
            if let Some(serde_json::Value::Array(arr)) = map.get("decoders") {
                return arr.iter().any(decoder_is_byte_level);
            }
            false
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_once_upon_a_time() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../architectures/testdata/maykeye-tinyllama-v0/tokenizer.json");
        if !path.is_file() {
            eprintln!("skip: tokenizer fixture missing");
            return;
        }
        let tok = BpeTokenizer::from_file(&path).unwrap();
        assert!(!tok.is_byte_level());
        let ids = tok.encode("Once upon a time", true).unwrap();
        assert_eq!(ids.first().copied(), Some(1)); // <s>
        // HF reference: [1, 4612, 2619, 260, 647]
        assert_eq!(ids, vec![1, 4612, 2619, 260, 647], "ids={ids:?}");
        let text = tok.decode(&ids, true).unwrap();
        assert_eq!(text, "Once upon a time");
    }

    #[test]
    fn encodes_qwen3_byte_level() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../architectures/testdata/qwen3-0.6b/tokenizer.json");
        let path = if path.is_file() {
            path
        } else if let Ok(home) = std::env::var("HOME") {
            // Prefer a cached HF snapshot when testdata is absent.
            let hub = Path::new(&home).join(".cache/huggingface/hub");
            let mut found = None;
            if let Ok(rd) = fs::read_dir(&hub) {
                for ent in rd.flatten() {
                    let name = ent.file_name();
                    let name = name.to_string_lossy();
                    if name.contains("Qwen3-0.6B") || name.contains("qwen3-0.6b") {
                        if let Ok(snaps) = fs::read_dir(ent.path().join("snapshots")) {
                            for snap in snaps.flatten() {
                                let tok = snap.path().join("tokenizer.json");
                                if tok.is_file() {
                                    found = Some(tok);
                                    break;
                                }
                            }
                        }
                    }
                    if found.is_some() {
                        break;
                    }
                }
            }
            match found {
                Some(p) => p,
                None => {
                    eprintln!("skip: Qwen3 tokenizer not in testdata or HF cache");
                    return;
                }
            }
        } else {
            eprintln!("skip: Qwen3 tokenizer not available");
            return;
        };

        let tok = BpeTokenizer::from_file(&path).unwrap();
        assert!(tok.is_byte_level());
        let ids = tok.encode("Once upon a time", false).unwrap();
        assert_eq!(ids, vec![12522, 5193, 264, 882], "ids={ids:?}");
        let text = tok.decode(&ids, true).unwrap();
        assert_eq!(text, "Once upon a time");
        assert_eq!(tok.encode("Hello", false).unwrap(), vec![9707]);
    }
}
