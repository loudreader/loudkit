//! The language-agnostic speech funnel: a bit-parity port of
//! `loudkit.frontend.speechtext`, which `docs/design/ARCHITECTURE.md` names as
//! the reference for this concept, and of the Swift, JS and Go funnels beside
//! it.
//!
//! Before tokenising, the shipped engine scrubs the raw text: invisible
//! characters, symbols that carry meaning, footnote markers, and punctuation
//! (prosodic marks stay exactly where they are, the model is a language model
//! trained on punctuated text: everything else becomes a space). Applied by
//! the engine's `encode` path, mirroring `Engine._synthesize_one` in Python,
//! `Engine.encode` in JS and Go.
//!
//! The Polish English-respelling lexicon is ported too, see `respell.rs`,
//! which embeds the generated dictionary and is wired into `speech_text`.
//!
//! Python reference: `loudkit/frontend/speechtext.py`.

use std::sync::LazyLock;

use regex::Regex;
use unicode_normalization::UnicodeNormalization;

use crate::unicode::{is_decimal_digit, is_letter};

const INVISIBLES: &str = "\u{200B}\u{200C}\u{200D}\u{2060}\u{FEFF}\u{00AD}\u{180E}\u{200E}\u{200F}";

// Symbols the model cannot voice, in the order the pass replaces them. Which
// word each takes is a per-language fact and comes from `unit_words` in the
// shared grammar; `SYMBOL_MARKS` carries the rest.
const SPOKEN_SYMBOLS: [char; 28] = [
    '%', '°', '¢', '€', '£', '¥', '₹', '×', '÷', '≈', '≥', '≤', '≠', '±', '→', '←', '⇒', '✓', '✔',
    '✗', '✘', '•', '·', '▪', '◦', '…', '&', '@',
];

// An arrow, a bullet and an ellipsis are the same pause in every language, so
// they are a rule here rather than a row in twelve grammars.
const SYMBOL_MARKS: [(char, &str); 8] = [
    ('→', ","),
    ('←', ","),
    ('⇒', ","),
    ('•', ","),
    ('·', ","),
    ('▪', ","),
    ('◦', ","),
    ('…', "..."),
];

// The ASCII spellings of the comparison operators, longest first, each named
// by the mathematical symbol whose word it shares. Only these six: `-`, `/`,
// `.` and `+` are ranges, paths, decimals and hyphens far more often than
// operators, and a word put on one of them changes prose that reads correctly
// today.
const ASCII_OPERATORS: [(&str, &str); 6] = [
    ("<=", "≤"),
    (">=", "≥"),
    ("!=", "≠"),
    ("==", "="),
    ("<", "<"),
    (">", ">"),
];

/// The operators above, in that order, so the two-character spellings are
/// found before the one-character ones.
///
/// The whitespace Python asserts with `(?<={_SPACE})` and `(?={_SPACE})` is
/// read off the neighbouring characters instead: this engine has no
/// lookaround, and a consumed space is a space the next match cannot stand on.
static OPERATOR_RUN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        &ASCII_OPERATORS
            .iter()
            .map(|(spelling, _)| regex::escape(spelling))
            .collect::<Vec<_>>()
            .join("|"),
    )
    .expect("escaped operator spellings")
});

/// A markup tag, comment or declaration, which is not text anyone reads aloud.
///
/// The name inside the angle brackets otherwise reaches the model as a word,
/// and `<!-- ... -->` additionally leaves its `!` behind as a sentence-final
/// exclamation. A tag is replaced by a space rather than by nothing, because
/// two block tags meeting back to back are two paragraphs and not one glued
/// word.
static MARKUP_TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"</?[A-Za-z!][^<>]*>").unwrap());

// `$` and `£` before a number read as a prefix in writing and a SUFFIX in
// speech: "$5" is "five dollars", not "dollars five". The wording comes from
// `unit_word` (numbers.json); this list only says which symbols are written
// prefix.
const CURRENCY_PREFIXES: [char; 5] = ['$', '£', '€', '¥', '₹'];

/// Also `¢`, which nobody writes in front of a number: it is a suffix in every
/// convention, which is why the prefix pass never saw it.
const CURRENCY_SYMBOLS: [char; 6] = ['$', '£', '€', '¥', '₹', '¢'];

/// Symbol to word, per language, from the shared grammar file.
///
/// Per language and not per script: a table keyed to English and Polish alone
/// makes every other language hear English, so "$5" in a German render says
/// "5 dollars".
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

/// The word `symbol` takes in `language`, or `None` when this language has no
/// wording for it. No fall back to English: a symbol is spoken in the language
/// being read or it is left written, which is what the funnel does everywhere
/// the evidence runs out.
fn unit_word(symbol: &str, language: &str) -> Option<String> {
    UNIT_WORDS
        .get(language)
        .and_then(|w| w.get(symbol).cloned())
}

// Punctuation that carries prosody stays; the rest becomes a space.
const PROSODIC: &str =
    ".,!?;:\u{2014}\u{2013}\u{2026}\"\u{201C}\u{201D}\u{201E}\u{00AB}\u{00BB}()'\u{2019}\u{00BF}\u{00A1}";

static FOOTNOTE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!(r"\[[0-9{WHITE_SPACE},;\-–—]{{1,20}}\]")).unwrap());
static CLAUSE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!(r"[{WHITE_SPACE}]+([.,;:!?])")).unwrap());
/// A run, not a pair: regex substitution does not overlap its matches, so a
/// pair rule turns "..." into ".." on one pass and "." on the next, making the
/// funnel non-idempotent.
static MARKS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!(r"([.,;:])(?:[{WHITE_SPACE}]*[.,;:])+")).unwrap());
/// Unicode White_Space, written out, for every regex in this funnel.
///
/// Python reference: `loudkit.frontend.speechtext.WHITE_SPACE`. `\s` is four
/// different sets across the five implementations and the difference is
/// audible.
const WHITE_SPACE: &str = concat!(
    "\u{9}\u{a}\u{b}\u{c}\u{d}", // the ASCII controls tab through carriage return
    "\u{20}\u{85}\u{a0}",        // space, NEL, no-break space
    "\u{1680}",                  // Ogham space mark
    "\u{2000}\u{2001}\u{2002}\u{2003}\u{2004}\u{2005}\u{2006}\u{2007}\u{2008}\u{2009}\u{200a}", // the en/em quad run
    "\u{2028}\u{2029}",         // line and paragraph separator
    "\u{202f}\u{205f}\u{3000}", // narrow no-break, medium mathematical, ideographic
);

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
/// neunundvierzig Dollar". Only a lone dot with a plain fraction is touched,
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

static PLAIN_DECIMAL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[0-9]+\.[0-9]+$").unwrap());

fn drop_markup_tags(text: &str) -> String {
    if !text.contains('<') {
        return text.to_string();
    }
    MARKUP_TAG.replace_all(text, " ").to_string()
}

/// Whether a Unicode White_Space character stands at `at`, or ends the text.
///
/// The spacing is the evidence that a mark is an operator and not markup or an
/// emoticon: `<p>`, `</div>`, `<3` and `a<b` all keep the mark written, and
/// the funnel leaves written what it cannot read. A string edge is not
/// whitespace, which is what the reference's lookaround says by failing there.
fn space_at(text: &str, at: usize) -> bool {
    text[at..]
        .chars()
        .next()
        .is_some_and(|c| WHITE_SPACE.contains(c))
}

fn space_before(text: &str, at: usize) -> bool {
    text[..at]
        .chars()
        .next_back()
        .is_some_and(|c| WHITE_SPACE.contains(c))
}

/// The ASCII comparison operators, as words in this language.
///
/// See `docs/design/text-funnel.md`.
fn speak_operators(text: &str, language: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cut = 0;
    for m in OPERATOR_RUN.find_iter(text) {
        if !space_before(text, m.start()) || !space_at(text, m.end()) {
            continue;
        }
        let Some((_, symbol)) = ASCII_OPERATORS
            .iter()
            .find(|(spelling, _)| *spelling == m.as_str())
        else {
            continue;
        };
        // An operator no grammar covers stays written, like every other symbol
        // this module has no word for.
        let Some(word) = unit_word(symbol, language).filter(|w| !w.is_empty()) else {
            continue;
        };
        out.push_str(&text[cut..m.start()]);
        out.push_str(&word);
        cut = m.end();
    }
    out.push_str(&text[cut..]);
    out
}

/// The optional magnitude that may follow a currency amount, as a regex.
///
/// Two alternatives and two groups: the abbreviating letter glued to the
/// digits, and the scale noun written beside them. Both cases of each noun are
/// spelled out rather than asked of a case-insensitive flag, because a pattern
/// that needs an inline flag group is a pattern the five implementations
/// cannot share.
///
/// The `(?![A-Za-z])` the reference closes the group with is not here:
/// [`say_amounts`] reads the character after the match instead. A form the
/// guard refuses is a form no shorter alternative could stand in for, because
/// a shorter one is a prefix of it and so is followed by one of its own
/// letters.
fn scale_pattern(language: &str) -> String {
    let nouns = crate::numbers::scale_nouns(language);
    if nouns.is_empty() {
        return String::new();
    }
    let mut written: Vec<String> = Vec::new();
    for noun in &nouns {
        let mut chars = noun.chars();
        let capitalised = match chars.next() {
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            None => noun.clone(),
        };
        written.push(regex::escape(noun));
        if capitalised != *noun {
            written.push(regex::escape(&capitalised));
        }
    }
    format!(
        "(?:({})|[{WHITE_SPACE}]({}))?",
        *crate::numbers::SCALE_SUFFIX_PATTERN,
        written.join("|")
    )
}

/// The magnitude word standing between a price and its currency, or `""`.
///
/// A written scale reaches speech in two shapes and both belong before the
/// currency word: the letter glued to the digits (`$2.5M`) and the noun beside
/// them (`$5 million`). The noun is already this language's own word and is
/// kept as written; the letter is a number, so the grammar's scale noun is
/// asked for the form this count takes.
fn scale_after(amount: &str, suffix: &str, spelled: &str, language: &str) -> String {
    if !spelled.is_empty() {
        return spelled.to_string();
    }
    if suffix.is_empty() {
        return String::new();
    }
    let whole: String = amount
        .split('.')
        .next()
        .unwrap_or_default()
        .split(',')
        .next()
        .unwrap_or_default()
        .chars()
        .filter(char::is_ascii_digit)
        .collect();
    // The scale noun asks two questions of the count, whether it is one and
    // what its last two digits are, so an amount too long for an `i64` keeps
    // both answers as a hundred plus its last two digits.
    let count = whole.parse::<i64>().unwrap_or_else(|_| {
        100 + whole[whole.len().saturating_sub(2)..]
            .parse::<i64>()
            .unwrap_or(0)
    });
    crate::numbers::scale_suffix_word(suffix, count, language).unwrap_or_default()
}

/// One pattern per currency mark and language, compiled once.
///
/// The pattern is a function of the mark and of the language's scale nouns, so
/// a render used to pay for up to eleven compilations of the same eleven
/// patterns per call to [`speech_text`]. Every language is built at once,
/// because the table is read by language and a per-call miss would want a lock
/// around it.
///
/// No letter guard in the pattern: [`letter_before`] decides, the way
/// `loudkit.frontend.speechtext._say_amount` does. A guard written into the
/// pattern has to *consume* the preceding character, because this engine has no
/// lookbehind, and a consumed character is one the scan cannot start the next
/// match at. `$1$2$3` then read as `one dollars twenty-three dollars`: the
/// second `$` fell inside the first match's guard, the third amount matched
/// with `2` as its guard, and `2` and `3` were emitted next to each other and
/// spelled as one number nobody wrote. `docs/design/text-funnel.md:260` is
/// explicit that a confident wrong number is the outcome to avoid.
static CURRENCY_BEFORE: LazyLock<std::collections::HashMap<String, Vec<(char, Regex)>>> =
    LazyLock::new(|| {
        crate::numbers::supported_languages()
            .into_iter()
            .map(|language| {
                let scale = scale_pattern(language);
                let patterns = CURRENCY_PREFIXES
                    .iter()
                    .map(|&sym| {
                        let pat = format!(
                            r"{}[{WHITE_SPACE}]?([0-9]+(?:[.,][0-9]+)*){scale}",
                            regex::escape(&sym.to_string())
                        );
                        (sym, Regex::new(&pat).expect("escaped currency mark"))
                    })
                    .collect();
                (language.to_string(), patterns)
            })
            .collect()
    });

static CURRENCY_AFTER: LazyLock<Vec<(char, Regex)>> = LazyLock::new(|| {
    CURRENCY_SYMBOLS
        .iter()
        .map(|&sym| {
            let pat = format!(
                r"([0-9]+(?:[.,][0-9]+)*)[{WHITE_SPACE}]?{}",
                regex::escape(&sym.to_string())
            );
            (sym, Regex::new(&pat).expect("escaped currency mark"))
        })
        .collect()
});

/// Whether the character before byte offset `at` is a letter, as Unicode
/// category L means it.
///
/// `loudkit.frontend.speechtext._letter_before`. A letter in front means a
/// multi-character currency mark: `R$` is the Brazilian real, `HK$` the Hong
/// Kong dollar, `NT$` the Taiwan dollar, and this table has a wording for none
/// of them. Matching the `$` alone read `R$3,14` as "R3,14 Dollar": the wrong
/// currency, said confidently. The mark itself is still dropped by the
/// punctuation pass, so the amount reads as a plain decimal; losing a symbol is
/// a smaller lie than naming the wrong money.
///
/// Category L and not `[:alpha:]`: this crate's POSIX classes are ASCII even in
/// Unicode mode, so an ASCII test called every non-ASCII letter a non-letter
/// and the guard passed straight through them. `zł€ 000 000` read as "zł000
/// euro nul nul nul", the mark taken for a bare euro sign, where Python's
/// `unicodedata.category`, Go's `unicode.IsLetter` and JS's `\p{L}` all refuse
/// the whole thing.
fn letter_before(text: &str, at: usize) -> bool {
    text[..at].chars().next_back().is_some_and(is_letter)
}

/// Every currency amount in `text` whose mark is not glued to the end of a
/// word, said as a number, its magnitude if it was written, and `word`.
///
/// `loudkit.frontend.speechtext._speak_symbols`' prefix pass, rebuilt rather
/// than expressed as a replacement template, because the letter guard reads a
/// character the match does not cover. The scan is over the original string, so
/// an earlier rewrite cannot move the character a later guard looks at, which
/// is what `re.sub` with a callback gives the reference.
fn say_amounts(text: &str, pattern: &Regex, word: &str, language: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cut = 0;
    for caps in pattern.captures_iter(text) {
        let whole = caps.get(0).expect("group 0 always matches");
        let amount = caps.get(1).expect("group 1 always matches");
        out.push_str(&text[cut..whole.start()]);
        if letter_before(text, whole.start()) {
            // Not a price: a mark glued to the end of a word. Left written,
            // which is what this module does everywhere the evidence runs out.
            out.push_str(whole.as_str());
            cut = whole.end();
            continue;
        }
        // The guard the reference writes as `(?![A-Za-z])`: a letter behind the
        // magnitude means the run is a word rather than a scale, and the
        // optional group matches nothing. `$5kg` is five dollars and a unit
        // nobody abbreviated, so the amount is spoken alone and `kg` is left
        // exactly where it was written.
        let scaled = !text[whole.end()..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic());
        let suffix = if scaled {
            caps.get(2).map_or("", |m| m.as_str())
        } else {
            ""
        };
        let spelled = if scaled {
            caps.get(3).map_or("", |m| m.as_str())
        } else {
            ""
        };
        let said = scale_after(amount.as_str(), suffix, spelled, language);
        out.push_str(&priced(amount.as_str(), language));
        out.push(' ');
        if said.is_empty() {
            out.push_str(word);
            // A scale this language has no noun for. The suffix stays written,
            // which is what it did before the amount was moved.
            out.push_str(suffix);
        } else {
            out.push_str(&said);
            out.push(' ');
            out.push_str(word);
        }
        cut = if scaled { whole.end() } else { amount.end() };
    }
    out.push_str(&text[cut..]);
    out
}

fn speak_symbols(text: &str, language_id: &str) -> String {
    let mut out = text.to_string();
    // A language without a wording table hears English rather than silence.
    let language = if UNIT_WORDS.contains_key(language_id) {
        language_id
    } else {
        "en"
    };
    out = speak_operators(&out, language);
    // Prefix currencies first, while the digits still follow the symbol.
    for (sym, re) in CURRENCY_BEFORE.get(language).into_iter().flatten() {
        let Some(word) = unit_word(&sym.to_string(), language) else {
            continue;
        };
        // The letter guard lives in `say_amounts`, which reads the character
        // in front of the match instead of matching it.
        out = say_amounts(&out, re, &word, language);
    }
    // The same amount with the symbol behind it. `2.50 €` and `0.49¢` are prices by
    // exactly the evidence `€2.50` is, and reached the time pass with the dot intact:
    // German answered "zwei Uhr fünfzig Euro". Currency written as a *word*,
    // `5.50 zł`, is not covered; telling those from a unit needs a per-language
    // lexicon.
    for (sym, re) in CURRENCY_AFTER.iter() {
        let Some(word) = unit_word(&sym.to_string(), language_id) else {
            continue;
        };
        if !out.contains(*sym) {
            continue;
        }
        out = re
            .replace_all(&out, |c: &regex::Captures| {
                format!("{} {word}", priced(&c[1], language_id))
            })
            .to_string();
    }
    for sym in SPOKEN_SYMBOLS {
        if !out.contains(sym) {
            continue;
        }
        let owned;
        let mark = SYMBOL_MARKS
            .iter()
            .find(|(m, _)| *m == sym)
            .map(|(_, w)| *w);
        let replacement = match unit_word(&sym.to_string(), language) {
            Some(w) => {
                owned = w;
                owned.as_str()
            }
            None => match mark {
                Some(w) => w,
                None => continue,
            },
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

/// `No` and `Nl` characters become a space, before anything counts digits.
///
/// A superscript, a vulgar fraction or a circled numeral is a number character
/// this layer has no reading for. It was left in place through the number pass
/// and only removed later, by the punctuation pass, and while it was there it sat
/// inside the word: every "is this a word character" test in the five ports is
/// either `\w`, `\p{N}` or `str.isdigit()`, and all three admit `No`. So `²9` was
/// one token, the number matcher's boundary guard refused it, and what reached the
/// model was a bare `9`.
///
/// A bare digit is the one thing this layer exists to prevent: a grapheme model
/// reads it badly and there is no way to hear that it happened.
///
/// Removed as a space rather than deleted, which is what the punctuation pass
/// does with every other character it has no reading for. `Nd` is excluded on
/// purpose: those are digits in some script, and the number pass wants them.
/// Every number character becomes something a reader can say aloud.
///
/// Python reference: `loudkit.frontend.speechtext.fold_numerals`, which
/// carries the reasoning. A non-ASCII decimal digit becomes the ASCII digit of
/// the same value and joins the run it was in; every other number character
/// becomes the text `numerals.json` names for it, separated from an adjacent
/// alphanumeric so `²9` does not fold into `29`. The table decides *whether* a
/// character is a numeral too: see `folded_numeral`.
fn fold_numerals(text: &str) -> String {
    if !text.chars().any(|c| folded_numeral(c).is_some()) {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    for (i, &c) in chars.iter().enumerate() {
        let Some((spelled, is_digit)) = folded_numeral(c) else {
            out.push(c);
            continue;
        };
        // A slash inside a spelled numeral is a fraction bar, asserted by the
        // character itself: `½` is a half wherever it stands, where a typed
        // `1/2` is a fraction, a date or the `24/7` of ordinary prose. The
        // division sign is the mark the symbol table already has a word for in
        // every language, so the reading comes from the grammar and not from
        // here.
        let spelled = spelled.replace('/', "÷");
        if is_digit {
            out.push_str(&spelled);
            continue;
        }
        if out
            .chars()
            .next_back()
            .is_some_and(|p| is_letter(p) || is_decimal_digit(p))
        {
            out.push(' ');
        }
        out.push_str(&spelled);
        if chars
            .get(i + 1)
            .is_some_and(|&n| is_letter(n) || is_decimal_digit(n))
        {
            out.push(' ');
        }
    }
    out
}

/// What `fold_numerals` puts in a numeral's place, and whether it is a decimal
/// digit, or `None` for a character the table does not name.
///
/// Read from the shared `numerals.json` rather than computed, detection
/// included. Computing either half was wrong: walking down to a decimal block's
/// start walks out of the block where two blocks touch, `MATHEMATICAL
/// DOUBLE-STRUCK DIGIT ZERO` follows `MATHEMATICAL SANS-SERIF DIGIT NINE` with
/// nothing between: NFKC does not reach `ETHIOPIC NUMBER TEN`, the Aegean
/// numbers, the Kaktovik digits or the Meroitic numerals, which then vanished;
/// and asking a general-category crate *whether* to fold put a Unicode version
/// back into an answer the table had taken it out of. `unicode-general-category`
/// moves with its own releases, Python's `unicodedata` with the interpreter, and
/// the five ports report one fingerprint. The table is cut from one pinned UCD
/// and hashed into `TextConfig.grammar`.
///
/// `None` for an ASCII digit, a letter or an ideographic numeral: all left
/// exactly as written.
fn folded_numeral(c: char) -> Option<(String, bool)> {
    if c.is_ascii_digit() {
        return None;
    }
    let table = numerals();
    let code = c as u32;
    if let Some(text) = table.spelled.get(&code) {
        return Some((text.clone(), false));
    }
    let index = table.decimal_zeros.partition_point(|&z| z <= code);
    if index > 0 {
        let offset = code - table.decimal_zeros[index - 1];
        if offset <= 9 {
            return Some((offset.to_string(), true));
        }
    }
    None
}

#[derive(serde::Deserialize)]
struct Numerals {
    decimal_zeros: Vec<u32>,
    spelled: std::collections::HashMap<u32, String>,
}

/// `numerals.json`, parsed once.
fn numerals() -> &'static Numerals {
    static TABLE: LazyLock<Numerals> = LazyLock::new(|| {
        serde_json::from_str(include_str!("numerals.json")).expect("numerals.json unreadable")
    });
    &TABLE
}

fn drop_footnote_markers(text: &str) -> String {
    if !text.contains('[') {
        return text.to_string();
    }
    FOOTNOTE_RE.replace_all(text, "").to_string()
}

fn punctuation_for_speech(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    for (i, &sc) in chars.iter().enumerate() {
        if is_letter(sc) || is_decimal_digit(sc) || sc.is_whitespace() || PROSODIC.contains(sc) {
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
        // are ranges and fractions: meaning, not decoration.
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
            && prev.is_some_and(|p| is_letter(p) || is_decimal_digit(p))
            && next.is_some_and(|n| is_letter(n) || is_decimal_digit(n))
        {
            out.push(sc);
            continue;
        }
        out.push(' ');
    }
    out
}

/// Prepare `text` to be spoken in `language_id`: the same funnel the shipped
/// Swift engine runs as `SpeechText.prepared`. Same order, same rules.
pub fn speech_text(text: &str, language_id: &str) -> String {
    // The language id is lowercased once, here, and again in the respeller.
    // `GraphemeTextFrontend` lowercases its own tag, so without this "PL"
    // produces Polish *tokens* while silently skipping the Polish respelling:
    // the same utterance read half one way and half the other, with nothing to
    // indicate it. `loudkit.frontend.speechtext.speech_text` and Swift's
    // `LexicalRespelling.applied` carry the same lowercasing for the same
    // reason.
    let language_id = &language_id.to_lowercase();
    // NFC first, before anything inspects a character: the same opening pass
    // the Python funnel runs.
    //
    // Unicode lets the same character arrive two ways: Polish ą as U+0105 or as
    // a + U+0328, Danish å as U+00E5 or a + U+030A. The tokenizer's vocabulary
    // holds one of them, so a decomposed spelling reaches it as a base letter
    // followed by an unknown combining mark, and every rule below, every
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
    // Before the symbol pass, which would otherwise read a tag's angle brackets
    // as comparison operators and its attributes as text.
    out = drop_markup_tags(&out);
    // Before the symbol table: see the Python reference. Every pass downstream
    // asks "is this a digit" and the five ports spell it four ways, so folding
    // first means all of them see ASCII.
    out = fold_numerals(&out);
    out = speak_symbols(&out, language_id);
    out = drop_footnote_markers(&out);
    // Before the acronym pass, which spells a Roman numeral letter by letter,
    // and after the numeral fold, which is what turns `Ⅳ` into the `IV` this
    // pass reads.
    out = crate::numbers::expand_roman_numerals(&out, language_id);
    // Acronyms while the capitals are still capitals: every later pass
    // lowercases or rewrites, and a spelled acronym has to be decided while the
    // only evidence, that the word stands alone in caps, still exists. The
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
    // Numbers after footnotes and before punctuation: see the Python funnel
    // for the ordering argument; the fixture pins it.
    out = crate::numbers::expand_abbreviations(&out, language_id);
    out = crate::numbers::expand_times(&out, language_id);
    out = crate::numbers::expand_numbers(&out, language_id);
    out = punctuation_for_speech(&out);
    // Polish: respell embedded English the way a Polish reader says it. This
    // is the shipped engine's LexicalRespelling; see respell.rs.
    out = crate::respell::lexical_respelling(&out, language_id);
    // Collapse runs of spaces/tabs: same as the shipped engine.
    out = SPACES_RE.replace_all(&out, " ").to_string();
    // A symbol that became a comma inherits the space that sat in
    // front of it ("0.49 → 0.24" would read "zero point four nine ,").
    out = CLAUSE_RE.replace_all(&out, "$1").to_string();
    // Two clause marks in a row is one clause mark.
    out = MARKS_RE.replace_all(&out, "$1").to_string();
    out.trim().to_string()
}
