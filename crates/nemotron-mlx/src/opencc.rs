//! Simplified-to-Traditional (Taiwan) Chinese text conversion.
//!
//! All Chinese output produced by this workspace is normalized to Traditional
//! Chinese using OpenCC's `s2twp` configuration (Taiwan standard characters
//! plus Taiwan vocabulary substitutions, e.g. 软件→軟體, 信息→資訊).

use std::sync::OnceLock;

use ferrous_opencc::{OpenCC, config::BuiltinConfig};

static CONVERTER: OnceLock<OpenCC> = OnceLock::new();

fn converter() -> &'static OpenCC {
    CONVERTER.get_or_init(|| {
        // `s2twp`'s dictionaries are embedded in the `ferrous-opencc` binary,
        // so construction is deterministic and cannot fail at runtime; a
        // failure here would indicate a broken build of the dependency
        // itself, which is a programming error, not a recoverable condition.
        OpenCC::from_config(BuiltinConfig::S2twp)
            .expect("embedded s2twp OpenCC config must load successfully")
    })
}

/// Converts `text` to Traditional Chinese (Taiwan standard + phrases) using
/// OpenCC's `s2twp` configuration. Non-Chinese text (ASCII, punctuation,
/// whitespace) passes through unchanged, and Traditional Chinese input is
/// left unchanged (idempotent).
pub fn to_traditional(text: &str) -> String {
    converter().convert(text)
}
