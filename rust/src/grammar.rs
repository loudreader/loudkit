//! The shared grammar file, located once and parsed once.
//!
//! [`crate::numbers`], [`crate::dates`] and [`crate::letters`] each read a
//! different part of `numbers.json`. A parse per reader is a parse per reader
//! at startup, with the rest of the document thrown away each time, so this
//! module is the one locator and the one parse. It is a leaf: it knows nothing
//! about numbers, dates or letters, so the three readers depend on the data
//! and not on each other.
//!
//! Python reference: `loudkit.frontend.textconfig`.

use std::collections::HashMap;
use std::sync::LazyLock;

use serde_json::{Map, Value};

/// The grammar file this crate embeds, as the bytes every implementation
/// hashes into [`crate::numbers::grammar_digest`].
pub(crate) static SOURCE: &str = include_str!("numbers.json");

static LANGUAGES: LazyLock<Map<String, Value>> = LazyLock::new(|| {
    let doc: Value = serde_json::from_str(SOURCE).expect("numbers.json unreadable");
    let Value::Object(mut doc) = doc else {
        return Map::new();
    };
    match doc.remove("languages") {
        Some(Value::Object(langs)) => langs,
        _ => Map::new(),
    }
});

/// Every language block the grammar file names, keyed by language tag.
pub(crate) fn languages() -> &'static Map<String, Value> {
    &LANGUAGES
}

/// A JSON array of strings, or an empty list when the member is absent.
///
/// Absent is the same as empty throughout: a grammar that does not name, say,
/// `oblique_triggers` has none, and a missing member is how the file says so.
pub(crate) fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// A JSON string, or `""` when the member is absent.
pub(crate) fn text(v: &Value) -> String {
    v.as_str().unwrap_or_default().to_string()
}

/// A JSON object whose keys are written integers, as a map from the integer.
///
/// An entry whose key does not parse, or whose value is not a string, is
/// dropped rather than guessed at.
pub(crate) fn int_map(v: &Value) -> HashMap<i64, String> {
    v.as_object()
        .map(|m| {
            m.iter()
                .filter_map(|(k, val)| Some((k.parse().ok()?, val.as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default()
}
