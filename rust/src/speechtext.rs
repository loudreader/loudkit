//! The language-agnostic speech funnel — a bit-parity port of the Swift
//! engine's `SpeechText` and the JS/Python/Go funnels.
//!
//! Before tokenising, the shipped engine scrubs the raw text: invisible
//! characters, symbols that carry meaning, footnote markers, and punctuation
//! (prosodic marks stay exactly where they are — the model is a language model
//! trained on punctuated text — everything else becomes a space). Applied by
//! the engine's `encode` path, mirroring `Engine._synthesize_one` in Python,
//! `Engine.encode` in JS and Go.
//!
//! The Polish English-respelling lexicon is ported too — see `respell.rs`,
//! which embeds the generated dictionary and is wired into `speech_text`.
//! Python reference: `loudkit/frontend/polish.py`.

use regex::Regex;
use std::sync::LazyLock;
use unicode_normalization::UnicodeNormalization;

const INVISIBLES: &str = "\u{200B}\u{200C}\u{200D}\u{2060}\u{FEFF}\u{00AD}\u{180E}\u{200E}\u{200F}";

// Symbols the model cannot voice, as words: (en, pl).
const SYMBOL_WORDS: [(char, (&str, &str)); 28] = [
    ('%', ("percent", "procent")),
    ('°', ("degrees", "stopni")),
    ('¢', ("cents", "centów")),
    ('€', ("euro", "euro")),
    ('£', ("pounds", "funtów")),
    ('¥', ("yen", "jenów")),
    ('₹', ("rupees", "rupii")),
    ('×', ("times", "razy")),
    ('÷', ("divided by", "podzielone przez")),
    ('≈', ("about", "około")),
    ('≥', ("at least", "co najmniej")),
    ('≤', ("at most", "najwyżej")),
    ('≠', ("not equal to", "różne od")),
    ('±', ("plus minus", "plus minus")),
    ('→', (",", ",")),
    ('←', (",", ",")),
    ('⇒', (",", ",")),
    ('✓', ("yes", "tak")),
    ('✔', ("yes", "tak")),
    ('✗', ("no", "nie")),
    ('✘', ("no", "nie")),
    ('•', (",", ",")),
    ('·', (",", ",")),
    ('▪', (",", ",")),
    ('◦', (",", ",")),
    ('…', ("...", "...")),
    ('&', ("and", "i")),
    ('@', ("at", "małpa")),
];

// `$` and `£` before a number read as a prefix in writing and a SUFFIX in
// speech: "$5" is "five dollars", not "dollars five". The wording comes from
// `unit_word` (numbers.json); this list only says which symbols are written
// prefix.
const CURRENCY_PREFIXES: [char; 5] = ['$', '£', '€', '¥', '₹'];

/// Also `¢`, which nobody writes in front of a number — it is a suffix in every
/// convention, which is why the prefix pass never saw it.
const CURRENCY_SYMBOLS: [char; 6] = ['$', '£', '€', '¥', '₹', '¢'];

/// Symbol -> word per language, from the shared grammar file. The old table
/// was an (en, pl) pair with `pl if polish else en`, which meant seven of the
/// nine languages heard English: "$5" in a German render said "5 dollars".
static UNIT_WORDS: LazyLock<
    std::collections::HashMap<String, std::collections::HashMap<String, String>>,
> = LazyLock::new(|| {
    let doc: serde_json::Value =
        serde_json::from_str(include_str!("numbers.json")).expect("numbers.json unreadable");
    let mut out = std::collections::HashMap::new();
    if let Some(langs) = doc["languages"].as_object() {
        for (lang, entry) in langs {
            let mut words = std::collections::HashMap::new();
            if let Some(map) = entry["unit_words"].as_object() {
                for (sym, word) in map {
                    if let Some(w) = word.as_str() {
                        words.insert(sym.clone(), w.to_string());
                    }
                }
            }
            out.insert(lang.clone(), words);
        }
    }
    out
});

/// The word `symbol` takes in `language`, falling back to English so a symbol
/// is at least said, if with an accent.
fn unit_word(symbol: &str, language: &str) -> Option<String> {
    if let Some(words) = UNIT_WORDS.get(language) {
        if let Some(w) = words.get(symbol) {
            return Some(w.clone());
        }
    }
    UNIT_WORDS.get("en").and_then(|w| w.get(symbol).cloned())
}

// Punctuation that carries prosody stays; the rest becomes a space.
const PROSODIC: &str =
    ".,!?;:\u{2014}\u{2013}\u{2026}\"\u{201C}\u{201D}\u{201E}\u{00AB}\u{00BB}()'\u{2019}\u{00BF}\u{00A1}";

static FOOTNOTE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[[\d\s,;\-–—]{1,20}\]").unwrap());
static CLAUSE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+([.,;:!?])").unwrap());
/// A run, not a pair: regex substitution does not overlap its matches, so a
/// pair rule turns "..." into ".." on one pass and "." on the next, making the
/// funnel non-idempotent.
static MARKS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([.,;:])(?:[\s]*[.,;:])+").unwrap());
static SPACES_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]{2,}").unwrap());

fn strip_invisibles(text: &str) -> String {
    if !text.chars().any(|c| INVISIBLES.contains(c)) {
        return text.to_string();
    }
    text.chars().filter(|c| !INVISIBLES.contains(*c)).collect()
}

/// A currency amount, with its decimal mark spelled the way `language` does.
///
/// The one place a dot between digits is known not to be a clock time, and the
/// last place that knows it: by the time pass the symbol has become a trailing
/// word and `$0.49` is indistinguishable from `14.30`, which in the eleven
/// comma-decimal languages is how a time is written. German answered "null Uhr
/// neunundvierzig Dollar". Only a lone dot with a plain fraction is touched —
/// `$1,234.56` carries a grouping mark this cannot safely reinterpret.
fn priced(amount: &str, language: &str) -> String {
    let sep = crate::numbers::decimal_separator(language);
    if sep == "." {
        return amount.to_string();
    }
    if PLAIN_DECIMAL.is_match(amount) {
        return amount.replacen('.', sep, 1);
    }
    amount.to_string()
}

static PLAIN_DECIMAL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\d+\.\d+$").unwrap());

fn speak_symbols(text: &str, language_id: &str) -> String {
    let mut out = text.to_string();
    // A language without a wording table hears English rather than silence.
    let language = if UNIT_WORDS.contains_key(language_id) {
        language_id
    } else {
        "en"
    };
    // Prefix currencies first, while the digits still follow the symbol.
    for sym in CURRENCY_PREFIXES {
        let Some(word) = unit_word(&sym.to_string(), language) else {
            continue;
        };
        // A letter in front means a multi-character currency mark: `R$` is the Brazilian
        // real, `HK$` the Hong Kong dollar, `NT$` the Taiwan dollar, and this table has a
        // wording for none of them. Matching the `$` alone read `R$3,14` as "R3,14
        // Dollar" — the wrong currency, said confidently. The mark itself is still dropped
        // by the punctuation pass, so the amount reads as a plain decimal; losing a symbol
        // is a smaller lie than naming the wrong money.
        //
        // This crate's engine has no lookbehind, so the preceding character is
        // captured and put back.
        //
        // `\p{L}` and not `[:alpha:]`: this crate's POSIX classes are ASCII
        // even in Unicode mode, so `[^[:alpha:]]` called every non-ASCII letter
        // a non-letter and the guard passed straight through them. `zł€ 000 000`
        // read as "zł000 euro nul nul nul" — the mark taken for a bare euro
        // sign, the first group moved behind it and the rest left stranded —
        // where Python's Unicode-aware `(?<![^\W\d_])` refuses the whole thing.
        // Python, Go (`unicode.IsLetter`) and JS (`(?<!\p{L})`) all judge this
        // in Unicode; this was the one port reading it in ASCII.
        let pat = format!(
            r"(^|[^\p{{L}}]){}\s?(\d+(?:[.,]\d+)*)",
            regex::escape(&sym.to_string())
        );
        let re = Regex::new(&pat).unwrap();
        out = re
            .replace_all(&out, |c: &regex::Captures| {
                format!("{}{} {word}", &c[1], priced(&c[2], language_id))
            })
            .to_string();
    }
    // The same amount with the symbol behind it. `2.50 €` and `0.49¢` are prices by
    // exactly the evidence `€2.50` is, and reached the time pass with the dot intact:
    // German answered "zwei Uhr fünfzig Euro". Currency written as a *word* — `5.50
    // zł` — is not covered; telling those from a unit needs a per-language lexicon.
    for sym in CURRENCY_SYMBOLS {
        let Some(word) = unit_word(&sym.to_string(), language_id) else {
            continue;
        };
        if !out.contains(sym) {
            continue;
        }
        let pat = format!(r"(\d+(?:[.,]\d+)*)\s?{}", regex::escape(&sym.to_string()));
        let re = Regex::new(&pat).unwrap();
        out = re
            .replace_all(&out, |c: &regex::Captures| {
                format!("{} {word}", priced(&c[1], language_id))
            })
            .to_string();
    }
    for (sym, (en, pl)) in SYMBOL_WORDS {
        if !out.contains(sym) {
            continue;
        }
        // Not every symbol is a per-language word (arrows, ticks): the old
        // pair table still carries those.
        let fallback = if language == "pl" { pl } else { en };
        let owned;
        let replacement = match unit_word(&sym.to_string(), language) {
            Some(w) => {
                owned = w;
                owned.as_str()
            }
            None => fallback,
        };
        // A word replacement needs spaces around it; a punctuation one must
        // not gain a space BEFORE it or the comma floats.
        let spaced = if replacement.len() == 1 && ",.".contains(replacement) {
            format!("{replacement} ")
        } else {
            format!(" {replacement} ")
        };
        out = out.replace(sym, &spaced);
    }
    out
}

fn drop_footnote_markers(text: &str) -> String {
    if !text.contains('[') {
        return text.to_string();
    }
    FOOTNOTE_RE.replace_all(text, "").to_string()
}

/// `str.isdecimal()`, which is what Python's funnel uses — Unicode category
/// `Nd`, not ASCII `0-9`.
///
/// `char::is_ascii_digit` treats every non-ASCII digit as "not alphanumeric",
/// so `punctuation_for_speech` replaces it with a space and Arabic-Indic and
/// fullwidth numerals are **deleted from the text**: `"١٢٣ items"` comes out as
/// `"items"` where Python reads `"sto dwadzieścia trzy ajtamz"`. `is_numeric`
/// would be the easy reach and is wrong in the other direction — it also
/// admits `No` (½) and `Nl` (Ⅻ), which Python's `isdecimal` refuses.
pub(crate) fn is_decimal_digit(c: char) -> bool {
    use unicode_general_category::{get_general_category, GeneralCategory};
    get_general_category(c) == GeneralCategory::DecimalNumber
}

/// The value of a decimal digit in any script, as `int(token)` reads it in
/// Python.
///
/// `str::parse` understands ASCII digits only, so `"١٢٣"` fails to parse and
/// the Polish number path would decline to spell it — where Python says "sto
/// dwadzieścia trzy". Every `Nd` block is exactly ten consecutive code points,
/// so the value is the distance from the block's zero, found by walking down
/// at most nine.
pub(crate) fn decimal_value(c: char) -> Option<u32> {
    if !is_decimal_digit(c) {
        return None;
    }
    let cp = c as u32;
    let mut zero = cp;
    while zero > 0 {
        match char::from_u32(zero - 1) {
            Some(prev) if is_decimal_digit(prev) => zero -= 1,
            _ => break,
        }
    }
    Some(cp - zero)
}

/// A run of decimal digits in any script, rewritten with ASCII ones.
pub(crate) fn ascii_digits(token: &str) -> Option<String> {
    token
        .chars()
        .map(|c| decimal_value(c).map(|v| char::from_digit(v, 10).unwrap()))
        .collect()
}

fn punctuation_for_speech(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    for (i, &sc) in chars.iter().enumerate() {
        if sc.is_alphabetic() || is_decimal_digit(sc) || sc.is_whitespace() || PROSODIC.contains(sc)
        {
            out.push(sc);
            continue;
        }
        let prev = if i > 0 { Some(chars[i - 1]) } else { None };
        let next = if i + 1 < chars.len() {
            Some(chars[i + 1])
        } else {
            None
        };
        // Between digits, "." and "," are numeric separators and "-" and "/"
        // are ranges and fractions — meaning, not decoration.
        let between_digits =
            prev.is_some_and(is_decimal_digit) && next.is_some_and(is_decimal_digit);
        if between_digits && "-/:.".contains(sc) {
            out.push(sc);
            continue;
        }
        // A hyphen inside a token is part of the token ("well-known", "1e-3").
        // Either end alphanumeric, not both letters: the old test left the
        // exponent in "1e-3" to become a space, so the model was handed "1e 3"
        // after the number pass had already declined to read it.
        // `+` alongside `-`: the number pass declines "1e+3" as a token with a
        // letter in it, and punctuation then took it apart into "1e 3".
        if (sc == '-' || sc == '+')
            && prev.is_some_and(|p| p.is_alphanumeric())
            && next.is_some_and(|n| n.is_alphanumeric())
        {
            out.push(sc);
            continue;
        }
        out.push(' ');
    }
    out
}

/// Prepare `text` to be spoken in `language_id` — the same funnel the shipped
/// Swift engine runs as `SpeechText.prepared`. Same order, same rules.
pub fn speech_text(text: &str, language_id: &str) -> String {
    // The language id is lowercased once, here, and again in the respeller.
    // `GraphemeTextFrontend` lowercases its own tag, so "PL" produced Polish
    // *tokens* while silently skipping the Polish respelling — the same utterance
    // read half one way and half the other, with nothing to indicate it. Python
    // fixed this in `loudkit.frontend.polish.speech_text`, and Swift's
    // `LexicalRespelling.applied` carries the same `.lowercased()` with a
    // comment explaining why.
    let language_id = &language_id.to_lowercase();
    // NFC first, before anything inspects a character — the same opening pass
    // the Python funnel runs, and the one this funnel did not have.
    //
    // Unicode lets the same character arrive two ways: Polish ą as U+0105 or as
    // a + U+0328, Danish å as U+00E5 or a + U+030A. The tokenizer's vocabulary
    // holds one of them, so a decomposed spelling reaches it as a base letter
    // followed by an unknown combining mark — and every rule below, every
    // pattern and lexicon lookup and character class, is matching a string
    // nobody wrote a rule for.
    //
    // Ahead of `strip_invisibles`, which removes format characters:
    // normalisation can compose a sequence into a single character, and running
    // it afterwards would leave that composition unexamined.
    let normalised: String = text.nfc().collect();
    // Beside NFC, and before the symbol pass so the folded percent sign reaches the table that turns it into a word.
    let normalised = crate::numbers::fold_foreign_digits(&normalised, language_id);
    let mut out = strip_invisibles(&normalised);
    out = speak_symbols(&out, language_id);
    out = drop_footnote_markers(&out);
    // Acronyms while the capitals are still capitals: every later pass
    // lowercases or rewrites, and a spelled acronym has to be decided while the
    // only evidence — that the word stands alone in caps — still exists. The
    // pass belongs here rather than in `respell`: a Polish-only table there
    // spells `FBI` *ef-be-i*
    // in a Polish render and leaves the model raw graphemes in the other
    // eleven.
    out = crate::letters::spell_acronyms(&out, language_id);
    // Dates before times and numbers, and this ordering is the whole reason the
    // pass exists: `12.03.2026` is the ordinary written date of five of these
    // languages, and both passes below want a piece of it. The clock pattern
    // matches `12.03` and the digit run matches the lot, so a date recognised
    // any later has already been eaten and read as a time with a stray year.
    out = crate::dates::expand_dates(&out, language_id);
    // Ordinals before numbers, for the same reason: the number pass expands the
    // digits and leaves the suffix stuck to them, so `1st` arrived as *onest*.
    out = crate::dates::expand_ordinals(&out, language_id);
    // Numbers after footnotes and before punctuation — see the Python funnel
    // for the ordering argument; the fixture pins it.
    out = crate::numbers::expand_abbreviations(&out, language_id);
    out = crate::numbers::expand_times(&out, language_id);
    out = crate::numbers::expand_numbers(&out, language_id);
    out = punctuation_for_speech(&out);
    // Polish: respell embedded English the way a Polish reader says it. This
    // is the shipped engine's LexicalRespelling; see respell.rs.
    out = crate::respell::lexical_respelling(&out, language_id);
    // Collapse runs of spaces/tabs — same as the shipped engine.
    out = SPACES_RE.replace_all(&out, " ").to_string();
    // A symbol that became a comma inherits the space that sat in
    // front of it ("0.49 → 0.24" would read "zero point four nine ,").
    out = CLAUSE_RE.replace_all(&out, "$1").to_string();
    // Two clause marks in a row is one clause mark.
    out = MARKS_RE.replace_all(&out, "$1").to_string();
    out.trim().to_string()
}
