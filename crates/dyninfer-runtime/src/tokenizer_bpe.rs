//! Minimal HuggingFace `tokenizer.json` BPE (SentencePiece-style) loader.
//!
//! Supports OpenLLaMA / Maykeye TinyLLama: BPE + `▁` normalization + byte fallback.
//! Keeps the runtime free of the `tokenizers` crate (C++/esaxx).

use dyninfer_error::{DynInferError, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const SPACE_MARK: char = '\u{2581}'; // ▁

#[derive(Debug, Deserialize)]
struct TokenizerFile {
    added_tokens: Vec<AddedToken>,
    model: Model,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
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

/// Loaded BPE tokenizer.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
    /// merge pair -> rank (lower = earlier / higher priority)
    merges: HashMap<(String, String), u32>,
    byte_fallback: bool,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
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

        let bos_id = tf
            .added_tokens
            .iter()
            .find(|t| t.content == "<s>")
            .map(|t| t.id);
        let eos_id = tf
            .added_tokens
            .iter()
            .find(|t| t.content == "</s>")
            .map(|t| t.id);

        Ok(Self {
            token_to_id: tf.model.vocab,
            id_to_token,
            merges,
            byte_fallback: tf.model.byte_fallback,
            bos_id,
            eos_id,
        })
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    fn normalize(&self, text: &str) -> String {
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

    fn initial_pieces(&self, normalized: &str) -> Vec<String> {
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
        let normalized = self.normalize(text);
        let pieces = self.bpe_merge(self.initial_pieces(&normalized));
        let mut ids = Vec::with_capacity(pieces.len() + 1);
        if add_special_tokens {
            if let Some(bos) = self.bos_id {
                ids.push(bos);
            }
        }
        for p in pieces {
            let id = self.piece_id(&p).or_else(|| self.piece_id("<unk>")).unwrap_or(0);
            ids.push(id);
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut bytes: Vec<u8> = Vec::new();
        let mut text_acc = String::new();
        let flush_bytes = |bytes: &mut Vec<u8>, out: &mut String| {
            if !bytes.is_empty() {
                out.push_str(&String::from_utf8_lossy(bytes));
                bytes.clear();
            }
        };

        for &id in ids {
            if skip_special_tokens && (Some(id) == self.bos_id || Some(id) == self.eos_id) {
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
        let ids = tok.encode("Once upon a time", true).unwrap();
        assert_eq!(ids.first().copied(), Some(1)); // <s>
        // HF reference: [1, 4612, 2619, 260, 647]
        assert_eq!(ids, vec![1, 4612, 2619, 260, 647], "ids={ids:?}");
        let text = tok.decode(&ids, true).unwrap();
        assert_eq!(text, "Once upon a time");
    }
}
