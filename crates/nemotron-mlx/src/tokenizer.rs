//! Minimal BPE decode path for the checkpoint's Metaspace tokenizer.

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("unsupported tokenizer model or decoder: {0}")]
    Unsupported(String),
    #[error("token id {0} is absent from tokenizer.json")]
    UnknownToken(u32),
}

pub type TokenizerResult<T> = std::result::Result<T, TokenizerError>;

/// Decode-only tokenizer; training and encoding tables are intentionally omitted at runtime.
#[derive(Debug)]
pub struct Tokenizer {
    tokens: HashMap<u32, String>,
    special: HashSet<u32>,
    replacement: String,
    pad_token_id: u32,
    blank_token_id: u32,
}

impl Tokenizer {
    pub fn from_json(
        path: impl AsRef<Path>,
        pad_token_id: u32,
        blank_token_id: u32,
    ) -> TokenizerResult<Self> {
        let document: TokenizerDocument = serde_json::from_slice(&fs::read(path)?)?;
        if document.model.kind != "BPE" || document.decoder.kind != "Metaspace" {
            return Err(TokenizerError::Unsupported(format!(
                "{} + {}",
                document.model.kind, document.decoder.kind
            )));
        }
        let mut tokens = document
            .model
            .vocab
            .into_iter()
            .map(|(token, id)| (id, token))
            .collect::<HashMap<_, _>>();
        let mut special = HashSet::new();
        for token in document.added_tokens {
            tokens.insert(token.id, token.content);
            if token.special {
                special.insert(token.id);
            }
        }
        Ok(Self {
            tokens,
            special,
            replacement: document.decoder.replacement,
            pad_token_id,
            blank_token_id,
        })
    }

    /// Maps token IDs to text and applies the checkpoint's Metaspace decoder.
    pub fn decode(&self, ids: &[u32], strip_language_tag: bool) -> TokenizerResult<String> {
        let mut pieces = String::new();
        for id in ids {
            if *id == self.pad_token_id || *id == self.blank_token_id {
                continue;
            }
            let token = self
                .tokens
                .get(id)
                .ok_or(TokenizerError::UnknownToken(*id))?;
            if strip_language_tag && self.special.contains(id) {
                continue;
            }
            pieces.push_str(token);
        }
        let decoded = pieces.replace(&self.replacement, " ");
        Ok(decoded.trim_start_matches(' ').to_string())
    }
}

#[derive(serde::Deserialize)]
struct TokenizerDocument {
    model: TokenizerModel,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    decoder: TokenizerDecoder,
}

#[derive(serde::Deserialize)]
struct TokenizerModel {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<String, u32>,
}

#[derive(serde::Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    special: bool,
}

#[derive(serde::Deserialize)]
struct TokenizerDecoder {
    #[serde(rename = "type")]
    kind: String,
    replacement: String,
}
