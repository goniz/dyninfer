//! Minimal HuggingFace `tokenizer.json` BPE loader.
//!
//! Supports:
//! - OpenLLaMA / Maykeye TinyLLama: SentencePiece-style BPE + `▁` + byte fallback
//! - Qwen2/Qwen3: GPT-2 ByteLevel BPE (`Ġ` space mark)
//!
//! Keeps the runtime free of the `tokenizers` crate (C++/esaxx).

use dyninfer_error::{DynInferError, Result};
use hf_chat_template::{ChatTemplate, Message, RenderInput, TokenizerConfig as HfTokenizerConfig};
use serde::Deserialize;
use serde_json::{Map, Value as Json};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, OnceLock};

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
    /// Special/added pieces, longest-first for atomic encode splits.
    special_pieces: Vec<(String, u32)>,
    /// Official HF chat template from `tokenizer_config.json` / `chat_template.jinja`.
    chat_template: Option<Arc<ChatTemplate>>,
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

/// Load HF chat template from a model directory.
///
/// Precedence matches `transformers`: standalone `chat_template.jinja` wins over the
/// inline `chat_template` field in `tokenizer_config.json`.
fn load_chat_template(model_dir: &Path) -> Option<Arc<ChatTemplate>> {
    let config_path = model_dir.join("tokenizer_config.json");
    let jinja_path = model_dir.join("chat_template.jinja");
    let config = if config_path.is_file() {
        let bytes = fs::read(&config_path).ok()?;
        serde_json::from_slice::<HfTokenizerConfig>(&bytes).ok()
    } else {
        None
    };
    let tmpl = if jinja_path.is_file() {
        let source = fs::read_to_string(&jinja_path).ok()?;
        match &config {
            Some(cfg) => ChatTemplate::from_template_and_config(&source, cfg).ok(),
            None => ChatTemplate::from_str(&source).ok(),
        }
    } else if let Some(cfg) = config.as_ref() {
        ChatTemplate::from_tokenizer_config(cfg).ok()
    } else {
        None
    };
    tmpl.map(Arc::new)
}

impl BpeTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let (tokenizer_json, model_dir) = if path.is_dir() {
            (path.join("tokenizer.json"), Some(path.to_path_buf()))
        } else {
            let dir = path.parent().map(Path::to_path_buf);
            (path.to_path_buf(), dir)
        };
        let bytes = fs::read(&tokenizer_json).map_err(|e| {
            DynInferError::io_path(
                tokenizer_json.display().to_string(),
                format!("read tokenizer: {e}"),
            )
        })?;
        let mut tok = Self::from_bytes(&bytes)?;
        if let Some(dir) = model_dir {
            tok.chat_template = load_chat_template(&dir);
        }
        Ok(tok)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let tf: TokenizerFile = serde_json::from_slice(bytes)
            .map_err(|e| DynInferError::io(format!("tokenizer.json parse failed: {e}")))?;
        if tf.model.vocab.is_empty() {
            return Err(DynInferError::io("tokenizer vocab is empty"));
        }
        let max_id = tf
            .model
            .vocab
            .values()
            .copied()
            .chain(tf.added_tokens.iter().map(|t| t.id))
            .max()
            .unwrap_or(0);
        let mut token_to_id = tf.model.vocab.clone();
        let mut id_to_token = vec![String::new(); (max_id as usize) + 1];
        for (tok, id) in &tf.model.vocab {
            let i = *id as usize;
            if i >= id_to_token.len() {
                id_to_token.resize(i + 1, String::new());
            }
            id_to_token[i] = tok.clone();
        }
        let mut special_pieces = Vec::new();
        for a in &tf.added_tokens {
            let i = a.id as usize;
            if i >= id_to_token.len() {
                id_to_token.resize(i + 1, String::new());
            }
            id_to_token[i] = a.content.clone();
            token_to_id.insert(a.content.clone(), a.id);
            if a.special || a.content.starts_with("<|") {
                special_pieces.push((a.content.clone(), a.id));
            }
        }
        // Longest match first so `<|im_start|>` wins over shorter prefixes.
        special_pieces.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

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
            token_to_id,
            id_to_token,
            merges,
            byte_fallback: tf.model.byte_fallback,
            bos_id,
            eos_id,
            special_ids,
            special_pieces,
            chat_template: None,
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

    /// Token id for an exact vocab / added-token string, if present.
    pub fn token_id(&self, piece: &str) -> Option<u32> {
        self.token_to_id.get(piece).copied()
    }

    /// True when an official HF chat template was loaded from the model dir.
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    /// True when the tokenizer has ChatML markers (`<|im_start|>` / `<|im_end|>`).
    pub fn has_chatml_markers(&self) -> bool {
        self.token_id("<|im_start|>").is_some() && self.token_id("<|im_end|>").is_some()
    }

    /// Apply the model's HF chat template to a bare user prompt.
    ///
    /// Prefers `tokenizer_config.json` / `chat_template.jinja` via `hf-chat-template`.
    /// Falls back to a minimal ChatML wrap when markers exist but no template was loaded.
    /// Returns `None` if the prompt is already templated or no chat formatting is available.
    pub fn apply_chat_template(
        &self,
        user: &str,
        enable_thinking: bool,
    ) -> Result<Option<String>> {
        if user.contains("<|im_start|>") {
            return Ok(None);
        }
        if let Some(tmpl) = &self.chat_template {
            let mut extra = Map::new();
            extra.insert("enable_thinking".into(), Json::Bool(enable_thinking));
            let input = RenderInput {
                messages: vec![Message::user(user)],
                add_generation_prompt: true,
                extra,
                ..Default::default()
            };
            let rendered = tmpl.render(&input).map_err(|e| {
                DynInferError::io(format!("chat_template render failed: {e}"))
            })?;
            return Ok(Some(rendered));
        }
        if !self.has_chatml_markers() {
            return Ok(None);
        }
        let mut out = format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n");
        if !enable_thinking {
            out.push_str("<think>\n\n</think>\n\n");
        }
        Ok(Some(out))
    }

    /// Wrap a bare user string in ChatML turn markers when supported.
    #[deprecated(note = "use apply_chat_template")]
    pub fn chatml_user_prompt(&self, user: &str) -> Option<String> {
        self.apply_chat_template(user, true).ok().flatten()
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
        text.as_bytes().iter().map(|b| enc[b].to_string()).collect()
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
        let mut ids = Vec::new();
        if add_special_tokens {
            if let Some(bos) = self.bos_id {
                ids.push(bos);
            }
        }
        // Split out atomic special tokens, then BPE each ordinary span.
        let mut rest = text;
        while !rest.is_empty() {
            let mut special_at: Option<(usize, &str, u32)> = None;
            for (piece, id) in &self.special_pieces {
                if let Some(pos) = rest.find(piece.as_str()) {
                    if special_at.is_none_or(|(best, _, _)| pos < best) {
                        special_at = Some((pos, piece.as_str(), *id));
                    }
                }
            }
            let Some((pos, piece, id)) = special_at else {
                ids.extend(self.encode_ordinary(rest)?);
                break;
            };
            if pos > 0 {
                ids.extend(self.encode_ordinary(&rest[..pos])?);
            }
            ids.push(id);
            rest = &rest[pos + piece.len()..];
        }
        Ok(ids)
    }

    fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let pieces = match self.kind {
            TokenizerKind::SentencePiece => {
                let normalized = self.normalize_sp(text);
                self.bpe_merge(self.initial_pieces_sp(&normalized))
            }
            TokenizerKind::ByteLevel => self.bpe_merge(self.initial_pieces_byte_level(text)),
        };
        let mut ids = Vec::with_capacity(pieces.len());
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
        Some(id) == self.bos_id || Some(id) == self.eos_id || self.special_ids.contains(&id)
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
        // Prefer HF Hub cache (weights/tokenizer are not vendored in-repo).
        let path = if let Ok(home) = std::env::var("HOME") {
            let hub = Path::new(&home).join(".cache/huggingface/hub");
            let mut found = None;
            if let Ok(rd) = fs::read_dir(&hub) {
                for ent in rd.flatten() {
                    let name = ent.file_name();
                    let name = name.to_string_lossy();
                    if name.contains("TinyLLama-v0") || name.contains("tinyllama") {
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
            found
        } else {
            None
        };
        let Some(path) = path else {
            eprintln!("skip: Maykeye/TinyLLama-v0 tokenizer not in HF cache");
            return;
        };
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

        // Official template from tokenizer_config.json (or ChatML fallback).
        assert!(tok.has_chatml_markers());
        let chat = tok
            .apply_chat_template("tell me a story", true)
            .unwrap()
            .expect("chat template");
        assert_eq!(
            chat,
            "<|im_start|>user\ntell me a story<|im_end|>\n<|im_start|>assistant\n"
        );
        let chat_ids = tok.encode(&chat, false).unwrap();
        assert_eq!(
            chat_ids,
            vec![151644, 872, 198, 72357, 752, 264, 3364, 151645, 198, 151644, 77091, 198],
            "chat_ids={chat_ids:?}"
        );
        let no_think = tok
            .apply_chat_template("tell me a story", false)
            .unwrap()
            .expect("chat template");
        assert!(
            no_think.ends_with("<think>\n\n</think>\n\n"),
            "enable_thinking=false should prefill empty think block, got {no_think:?}"
        );
    }
}

#[cfg(test)]
mod lfm2_chat_template_tests {
    use super::*;
    use std::path::PathBuf;

    fn lfm2_dir() -> Option<PathBuf> {
        crate::resolve_hf_snapshot("LiquidAI/LFM2.5-2.6B", Some("main")).ok()
    }

    #[test]
    fn lfm2_chat_template_matches_transformers_shape() {
        let Some(dir) = lfm2_dir() else {
            eprintln!("skip: LFM2.5-2.6B not in HF cache");
            return;
        };
        let tok = BpeTokenizer::from_file(&dir).expect("load tokenizer");
        assert!(tok.has_chat_template());
        let rendered = tok
            .apply_chat_template("What is 2+2?", true)
            .unwrap()
            .expect("template should render");
        eprintln!("rendered={rendered:?}");
        let ids = tok.encode(&rendered, false).unwrap();
        eprintln!("n_ids={} ids={ids:?}", ids.len());
        // HF reference for this prompt: 17 tokens ending with <think>
        assert!(
            rendered.starts_with("<|startoftext|><|im_start|>user\n"),
            "missing bos+user turn: {rendered:?}"
        );
        assert!(
            rendered.ends_with("<|im_start|>assistant\n<think>"),
            "missing assistant generation prompt: {rendered:?}"
        );
        assert_eq!(
            rendered,
            "<|startoftext|><|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>"
        );
        assert_eq!(ids.len(), 17, "token count mismatch vs transformers");
        assert_eq!(
            ids,
            vec![
                124894, 124899, 5922, 207, 2992, 355, 229, 26, 19, 26, 39, 124900, 207,
                124899, 63514, 207, 124901,
            ],
            "token ids mismatch vs transformers"
        );
    }
}
