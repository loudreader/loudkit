//! Text to text-tokens — a bit-parity port of `loudkit.frontend.text`:
//! lowercase, NFKD, a language tag, spaces to `[SPACE]`, then BPE.

use unicode_normalization::UnicodeNormalization;

use crate::numbers::supported_languages;
use crate::tokenizer;

const SPACE: &str = "[SPACE]";

/// Refused languages whose refusal has a *specific* reason worth stating: their
/// upstream pipeline wants Cangjie codes, kana conversion, diacritisation, jamo
/// decomposition or stress marks, none of which this frontend carries. A subset
/// of "not on the roster", kept so the message can say why rather than just no.
const NEEDS_MODEL_PREPROCESSING: [&str; 5] = ["zh", "ja", "he", "ko", "ru"];

pub struct Frontend {
    tokenizer: tokenizer::Tokenizer,
}

impl Frontend {
    pub fn load(tokenizer_path: &str) -> Result<Self, String> {
        Ok(Frontend {
            tokenizer: tokenizer::parse(tokenizer_path)?,
        })
    }

    /// The largest id `encode` can return. See `tokenizer::Tokenizer::max_id`.
    #[must_use]
    pub fn max_token_id(&self) -> usize {
        self.tokenizer.max_id()
    }

    /// Normalise and tokenise. Same text and language give the same ids.
    ///
    /// The language is an **allowlist**: the twelve ids
    /// [`crate::numbers::supported_languages`] reports. This was a blacklist of
    /// the five model-based ones, and the difference matters because the
    /// tokenizer's vocabulary carries tags for 31 languages — a blacklist let
    /// the other 26 through and the tag was emitted, so `encode(text, "bg")`
    /// NFKD-mangled Cyrillic into ids the model reads as sounds it was never
    /// trained to make: no error, plausible audio, wrong language.
    ///
    /// # Errors
    /// The language is not on the roster.
    pub fn encode(&self, text: &str, language: &str) -> Result<Vec<usize>, String> {
        let lang = language.to_lowercase();
        let roster = supported_languages();
        if !roster.contains(&lang.as_str()) {
            let why = if NEEDS_MODEL_PREPROCESSING.contains(&lang.as_str()) {
                "needs model-based text preprocessing \
                 (Cangjie/kana/diacritics/jamo/stress) that this frontend does not carry"
            } else {
                "is not one of the languages this build's text layer is written for"
            };
            return Err(format!(
                "language '{lang}' {why}. Supported: {}",
                roster.join(", ")
            ));
        }
        let normalised: String = text.to_lowercase().nfkd().collect();
        // Square brackets never reach the tokenizer from user text: the vocabulary
        // holds 117 bracket control tokens ([sigh], [gasp], the language tags) and
        // matches them greedily, so "he [sigh]ed" would make the model sigh. The
        // language tag added below is the one bracket that belongs.
        let normalised = normalised.replace(['[', ']'], " ");
        let tagged = format!("[{lang}]{normalised}").replace(' ', SPACE);
        Ok(self.tokenizer.encode(&tagged))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tag the tokenizer knows is not a language the kit can speak.
    ///
    /// The vocabulary carries tags for 31 languages; the text layer is written
    /// for twelve. A blacklist of only zh/ja/he/ko/ru lets the other 26
    /// go straight through: `encode(text, "bg")` NFKD-mangles Cyrillic
    /// into ids the model reads as sounds it never learned — no error,
    /// plausible-sounding audio, wrong language.
    ///
    /// Asserted against the roster rather than a literal list because
    /// `numbers.json` is the one authority: a port with its own copy is a port
    /// that disagrees with Python the next time a grammar is added. The
    /// refusal itself needs no tokenizer, so it is checkable without assets;
    /// the accepting path is covered by the conformance fixture.
    #[test]
    fn the_roster_is_an_allowlist() {
        let roster = supported_languages();
        assert_eq!(roster.len(), 12, "roster: {roster:?}");
        for lang in ["en", "pl", "sv"] {
            assert!(roster.contains(&lang), "{lang} is on the roster");
        }
        for lang in ["bg", "cs", "zh"] {
            assert!(!roster.contains(&lang), "{lang} must be refused");
        }
    }
}
