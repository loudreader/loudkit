//! Acronyms, spelled in the language being read.
//!
//! `CIA` is *see-eye-ay* in an English render and *ce-i-a* in a Polish one, and
//! those are not two spellings of one thing — they are what the two languages
//! actually say. The engine is grapheme-based with a single language tag per
//! utterance, so the letter name has to be written in the target language's own
//! orthography: English `see` reads as /siː/ under English letter-to-sound
//! rules, Polish `ce` reads as /t͡sɛ/ under Polish ones, and putting either into
//! the other's render produces a word nobody says.
//!
//! Without this module, acronyms are spelled only in Polish, inside
//! [`crate::respell`], with a Polish letter table: `FBI` becomes *ef-be-i* in a
//! Polish render and reaches the model as the raw graphemes `FBI` in the other
//! eleven, where a grapheme engine reads them as a word-shaped thing rather
//! than as letters. The
//! tables are per language in the shared grammar file; this reads them for all
//! twelve, out of the same `numbers.json` every other implementation reads.
//!
//! What is not spelled: an acronym that is a word in its language stays a word —
//! `NASA` and `NATO` everywhere, `SIDA` and `OVNI` in the Romance three, `PESEL`
//! and `ZUS` in Polish, `TUTKA` in Finnish. Those lists are per language because
//! the fact is: `LOT` is an airline in Poland and a common noun in English, and
//! only one of them should be spelled out.
//! Python reference: `loudkit/frontend/letters.py`.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

const MIN_LETTERS: usize = 2;

/// Above five letters an all-caps run is far more often a shout, a product name
/// or a heading than an initialism, and spelling one out is a worse error than
/// leaving it — the listener can read `SIGGRAPH`; they cannot un-hear
/// *ess-eye-gee-gee-ar-ay-pee-aitch*.
const MAX_LETTERS: usize = 5;

struct Table {
    names: HashMap<String, String>,
    words: HashSet<String>,
}

static TABLES: LazyLock<HashMap<String, Table>> = LazyLock::new(|| {
    let doc: serde_json::Value =
        serde_json::from_str(include_str!("numbers.json")).expect("numbers.json unreadable");
    let Some(langs) = doc["languages"].as_object() else {
        return HashMap::new();
    };
    let mut out = HashMap::new();
    for (lang, entry) in langs {
        let Some(names) = entry["letter_names"].as_object() else {
            continue;
        };
        if names.is_empty() {
            continue;
        }
        let names = names
            .iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
            .collect();
        let words = entry["word_acronyms"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        out.insert(lang.clone(), Table { names, words });
    }
    out
});

/// Whether this language has a letter table at all.
pub fn spells_acronyms(language: &str) -> bool {
    TABLES.contains_key(language)
}

/// What `language` calls one letter, or `None` if it has no name for it.
///
/// `None` rather than a guess: a letter with no entry means the acronym is left
/// alone entirely, because half-spelling one (*ef-be-**q***) is worse than not
/// spelling it at all.
pub fn letter_name(letter: &str, language: &str) -> Option<String> {
    TABLES
        .get(language)?
        .names
        .get(&letter.to_lowercase())
        .cloned()
}

/// `word` as spelled-out letters, or `None` if it should be left alone.
///
/// `None` — "not an acronym, or not one I can spell" — for a word that is not
/// all-caps, is too short or too long, is a word in this language, or contains a
/// letter this language has no name for.
pub fn spell_acronym(word: &str, language: &str) -> Option<String> {
    let len = word.chars().count();
    if len < MIN_LETTERS || !is_all_caps_word(word) {
        return None;
    }
    let table = TABLES.get(language)?;
    let lowered = word.to_lowercase();
    if table.words.contains(&lowered) {
        // A word, not an initialism: read as itself, lowercased so no later pass
        // mistakes it for an acronym again.
        //
        // Checked *before* the length cap, and the order matters: the cap is
        // about how long a thing may be before spelling
        // it becomes worse than leaving it, and it has nothing to say about a
        // word. With the cap first, every entry over five letters is dead —
        // UNESCO, UNICEF and INTERPOL never reach this branch.
        return Some(lowered);
    }
    if len > MAX_LETTERS {
        return None;
    }
    let mut names = Vec::with_capacity(len);
    for ch in lowered.chars() {
        names.push(table.names.get(&ch.to_string())?.clone());
    }
    // Hyphens rather than spaces: they keep the letters one prosodic unit, so
    // the model reads a run of names instead of a list of tiny words.
    Some(names.join("-"))
}

/// Every lone acronym in `text`, spelled the way `language` spells it.
///
/// **Shouting is left alone**, and the rule for telling it from an initialism is
/// context rather than anything inside the word. An initialism appears as a
/// single capitalised island in ordinary text — "the CIA said" — while emphasis
/// comes in runs. That distinction is not available from the word itself: `IT`
/// is a word, an initialism and a shout depending only on what sits beside it,
/// and no table can separate those. So a capitalised word spells out only when
/// neither neighbour is also capitalised, and a text that is *entirely* capitals
/// is passed through whole, because someone pasted a headline and spelling all
/// of it would be the loudest possible wrong answer.
pub fn spell_acronyms(text: &str, language: &str) -> String {
    if !spells_acronyms(language) || !text.chars().any(char::is_uppercase) {
        return text.to_string();
    }

    let tokens = split_on_non_word(text);
    let words: Vec<&String> = tokens.iter().filter(|t| is_word_token(t)).collect();
    if words.len() > 1 && words.iter().all(|t| is_all_caps_word(t)) {
        // The whole text is capitals: someone pasted a shout, or a headline.
        //
        // More than one word, though. A text that is a single capitalised token
        // — `speech_text("GPT")` — is an acronym on its own, not a shout: there
        // is no run to read emphasis from, and refusing it would mean the one
        // call shaped exactly like "say this acronym" was the one that did not.
        return text.to_string();
    }

    let is_caps = |i: usize| -> bool {
        tokens
            .get(i)
            .is_some_and(|t| is_word_token(t) && is_all_caps_word(t))
    };

    let mut out = tokens.clone();
    for i in 0..tokens.len() {
        if !is_caps(i) {
            continue;
        }
        // Neighbours, skipping the separator token between words.
        let before = i >= 2 && is_caps(i - 2);
        let after = is_caps(i + 2);
        if before || after {
            continue; // part of a run: emphasis, not an initialism
        }
        if let Some(said) = spell_acronym(&tokens[i], language) {
            out[i] = said;
        }
    }
    out.concat()
}

/// Python's `token.isalpha() and token.isupper()`, over a whole token.
///
/// `isupper()` is true when there is at least one cased character and no
/// lowercase one, so a token has to be judged as a unit rather than char by
/// char.
fn is_all_caps_word(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    let mut saw_cased = false;
    for ch in token.chars() {
        if !ch.is_alphabetic() || ch.is_lowercase() {
            return false;
        }
        if ch.is_uppercase() {
            saw_cased = true;
        }
    }
    saw_cased
}

fn is_word_token(token: &str) -> bool {
    token.chars().count() > 1 && token.chars().all(char::is_alphabetic)
}

/// Python's `re.split(r"(\W+)", text)`: separators are kept, so the pieces
/// rejoin exactly. Word characters are letters, digits and underscore, which is
/// what `\w` means under Python's default Unicode rules.
fn split_on_non_word(text: &str) -> Vec<String> {
    let is_word = |ch: char| ch.is_alphanumeric() || ch == '_';
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_is_word: Option<bool> = None;
    for ch in text.chars() {
        let w = is_word(ch);
        match current_is_word {
            None => {
                // Python's split starts on a word field, even an empty one.
                if !w {
                    out.push(String::new());
                }
                current.push(ch);
                current_is_word = Some(w);
            }
            Some(prev) if prev == w => current.push(ch),
            Some(_) => {
                out.push(std::mem::take(&mut current));
                current.push(ch);
                current_is_word = Some(w);
            }
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}
