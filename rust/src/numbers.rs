//! Numbers, said out loud — the Rust half of `loudkit.frontend.numbers`.
//!
//! The grammar is data and only the interpreter is code: this module reads the
//! same `numbers.json` every other implementation reads, so a rule lives once.
//! The composition mirrors `loudkit/frontend/numbers.py` function for
//! function; the reasons behind the odd-looking behaviours (joiners carrying
//! their own spacing, per-value agreement scopes, a scale noun with its own
//! gender) live in the Python docstrings and `docs/reference/preprocess.md`,
//! and the hand-written fixture plus the 1300-row CLDR differential pin them.
//!
//! Python reference: `loudkit/frontend/numbers.py`.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

#[derive(Debug, Clone)]
pub struct Scale {
    value: i64,
    forms: Vec<String>,
    /// "~" composes the multiplier; "" uses the bare scale word; anything else
    /// is the literal one-word (German *eine*, Italian *un*).
    one_word: String,
    separate: bool,
    link: String,
    small_joiner: String,
    multiplier_agrees: bool,
    multiplier_gender: String,
}

#[derive(Debug, Clone, Default)]
pub struct Grammar {
    ones: Vec<String>,
    teens: Vec<String>,
    tens: Vec<String>,
    hundred: String,
    hundreds: Vec<String>,
    hundreds_gendered: HashMap<String, Vec<String>>,
    hundred_plural_final: String,
    scales: Vec<Scale>,
    units_before_tens: bool,
    unit_tens_joiner: String,
    time_infix: String,
    abbreviations: Vec<(String, String)>,
    tens_joiner_exceptions: HashMap<i64, String>,
    hundred_joiner: String,
    scale_joiner_on_round_hundreds: bool,
    scale_large_joiner: String,
    one_before_hundred: bool,
    word_join: String,
    minus_word: String,
    decimal_separator: String,
    decimal_word: String,
    exceptions: HashMap<i64, String>,
    genders: HashMap<String, HashMap<i64, String>>,
    gender_scopes: HashMap<i64, String>,
    combining_ones: HashMap<i64, String>,
}

static GRAMMARS: LazyLock<HashMap<String, Grammar>> = LazyLock::new(|| {
    let doc: serde_json::Value =
        serde_json::from_str(include_str!("numbers.json")).expect("numbers.json unreadable");
    let mut out = HashMap::new();
    let Some(langs) = doc["languages"].as_object() else {
        return out;
    };
    let strs = |v: &serde_json::Value| -> Vec<String> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };
    let int_map = |v: &serde_json::Value| -> HashMap<i64, String> {
        v.as_object()
            .map(|m| {
                m.iter()
                    .filter_map(|(k, val)| Some((k.parse().ok()?, val.as_str()?.to_string())))
                    .collect()
            })
            .unwrap_or_default()
    };
    let s = |v: &serde_json::Value| v.as_str().unwrap_or_default().to_string();
    for (lang, e) in langs {
        let mut scales = Vec::new();
        if let Some(list) = e["scales"].as_array() {
            for sc in list {
                scales.push(Scale {
                    value: sc["value"].as_i64().unwrap_or(0),
                    forms: strs(&sc["forms"]),
                    one_word: sc
                        .get("one")
                        .and_then(|v| v.as_str())
                        .unwrap_or("~")
                        .to_string(),
                    separate: sc["separate"].as_bool().unwrap_or(false),
                    link: s(&sc["link"]),
                    small_joiner: s(&sc["small_joiner"]),
                    multiplier_agrees: sc["multiplier_agrees"].as_bool().unwrap_or(false),
                    multiplier_gender: s(&sc["multiplier_gender"]),
                });
            }
        }
        let mut genders = HashMap::new();
        if let Some(m) = e["genders"].as_object() {
            for (name, forms) in m {
                genders.insert(name.clone(), int_map(forms));
            }
        }
        let mut hundreds_gendered = HashMap::new();
        if let Some(m) = e["hundreds_gendered"].as_object() {
            for (name, forms) in m {
                hundreds_gendered.insert(name.clone(), strs(forms));
            }
        }
        out.insert(
            lang.clone(),
            Grammar {
                ones: strs(&e["ones"]),
                teens: strs(&e["teens"]),
                tens: strs(&e["tens"]),
                hundred: s(&e["hundred"]),
                hundreds: strs(&e["hundreds"]),
                hundreds_gendered,
                hundred_plural_final: s(&e["hundred_plural_final"]),
                scales,
                units_before_tens: e["units_before_tens"].as_bool().unwrap_or(false),
                unit_tens_joiner: s(&e["unit_tens_joiner"]),
                time_infix: s(&e["time_infix"]),
                abbreviations: {
                    let mut list: Vec<(String, String)> = e["abbreviations"]
                        .as_object()
                        .map(|m| {
                            m.iter()
                                .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
                                .collect()
                        })
                        .unwrap_or_default();
                    // Longest first, so fr.o.m. cannot be half-eaten.
                    list.sort_by_key(|(w, _)| std::cmp::Reverse(w.len()));
                    list
                },
                tens_joiner_exceptions: int_map(&e["tens_joiner_exceptions"]),
                hundred_joiner: s(&e["hundred_joiner"]),
                scale_joiner_on_round_hundreds: e["scale_joiner_on_round_hundreds"]
                    .as_bool()
                    .unwrap_or(false),
                scale_large_joiner: s(&e["scale_large_joiner"]),
                one_before_hundred: e["one_before_hundred"].as_bool().unwrap_or(false),
                word_join: s(&e["word_join"]),
                minus_word: s(&e["minus_word"]),
                decimal_separator: s(&e["decimal_separator"]),
                decimal_word: s(&e["decimal_word"]),
                exceptions: int_map(&e["exceptions"]),
                genders,
                gender_scopes: int_map(&e["gender_scopes"]),
                combining_ones: int_map(&e["combining_ones"]),
            },
        );
    }
    out
});

impl Grammar {
    /// The form `value` takes in `gender` at `position`, or `None` when it
    /// does not inflect. Position is "standalone" (the whole number), "tail"
    /// (ends a larger number) or "tens_pair" (inside the solid compound).
    fn gendered(&self, value: i64, gender: &str, position: &str) -> Option<&str> {
        if gender.is_empty() {
            return None;
        }
        match self.gender_scopes.get(&value).map(String::as_str) {
            Some("standalone") if position != "standalone" => return None,
            Some("outside_tens") if position == "tens_pair" => return None,
            _ => {}
        }
        self.genders.get(gender)?.get(&value).map(String::as_str)
    }
}

/// First 16 hex characters of the SHA-256 of the grammar file this crate
/// embeds, followed by the respelling lexicon. Hashed as raw bytes, like every
/// other implementation, so the five agree only when they ship the same files.
#[must_use]
pub fn grammar_digest() -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(include_bytes!("numbers.json"));
    // The lexicon alongside the grammar: it is a funnel input exactly as the
    // grammar is and it changes the spoken tokens, so both files hash into the
    // fingerprint. Leaving the lexicon out covers 55 KB of rules but not 6.5 MB
    // of vocabulary, and a build whose lexicon has drifted says different words
    // under the same sixteen hex digits.
    hasher.update(include_bytes!("pl_en_respell.json"));
    // sha2 0.11 returns a generic array rather than something LowerHex, so the
    // bytes are formatted one at a time. Sixteen characters is half a SHA-256,
    // like the fingerprint itself.
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()[..16]
        .to_string()
}

/// The language ids [`cardinal`] can verbalize, sorted — the roster in
/// `numbers.json`, and the allowlist [`crate::frontend`] enforces.
///
/// Mirrors `loudkit.frontend.numbers.supported_languages`. One authority for
/// both questions: a port that keeps a second copy of the roster is a port that
/// disagrees with Python the next time a grammar is added.
#[must_use]
pub fn supported_languages() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = GRAMMARS.keys().map(String::as_str).collect();
    out.sort_unstable();
    out
}

/// `value` as words. An empty gender gives the citation form. Unknown language
/// or a value past the grammar's largest scale is an error — silently reading
/// digits back would be indistinguishable from success.
pub fn cardinal(value: i64, language: &str, gender: &str) -> Result<String, String> {
    let g = GRAMMARS
        .get(language)
        .ok_or_else(|| format!("no number grammar for {language:?}"))?;
    let ceiling = g.scales.first().map_or(1000, |s| s.value * 1000);
    if value.abs() >= ceiling {
        return Err(format!(
            "{value} is past the largest scale {language:?} has a word for"
        ));
    }
    if value < 0 {
        // Always a spaced word, even in solid-writing languages: minus eins.
        return Ok(format!(
            "{} {}",
            g.minus_word,
            cardinal(-value, language, gender)?
        ));
    }
    // Standalone agreement applies to the whole number only: Polish jedna
    // alone, but sto jeden.
    if let Some(w) = g.gendered(value, gender, "standalone") {
        return Ok(w.to_string());
    }
    Ok(compose(value, g, gender, false))
}

fn compose(value: i64, g: &Grammar, gender: &str, as_multiplier: bool) -> String {
    if let Some(w) = g.exceptions.get(&value) {
        return w.clone();
    }
    if value < 100 {
        return below_hundred(value, g, gender, as_multiplier);
    }
    for sc in &g.scales {
        if value >= sc.value {
            return scale_group(value, sc, g, gender);
        }
    }
    hundreds_group(value, g, gender)
}

fn scale_group(value: i64, sc: &Scale, g: &Grammar, gender: &str) -> String {
    let (count, rest) = (value / sc.value, value % sc.value);
    let join = if sc.separate {
        " "
    } else {
        g.word_join.as_str()
    };
    let link_default = if sc.link.is_empty() {
        join
    } else {
        sc.link.as_str()
    };

    let head = if count == 1 && sc.one_word != "~" {
        if sc.one_word.is_empty() {
            scale_word(1, &sc.forms).to_string()
        } else {
            format!("{}{}{}", sc.one_word, join, scale_word(1, &sc.forms))
        }
    } else {
        // Whether the counted noun's gender reaches the multiplier is a fact
        // about the scale noun: Portuguese "duas mil", Polish "dwa tysiące".
        let mg = if !sc.multiplier_gender.is_empty() {
            sc.multiplier_gender.as_str()
        } else if sc.multiplier_agrees {
            gender
        } else {
            ""
        };
        format!(
            "{}{}{}",
            compose(count, g, mg, true),
            join,
            scale_word(count, &sc.forms)
        )
    };
    if rest == 0 {
        return head;
    }

    let round_hundreds = g.scale_joiner_on_round_hundreds && rest >= 100 && rest % 100 == 0;
    let link = if !sc.small_joiner.is_empty() && (rest < 100 || round_hundreds) {
        format!(" {} ", sc.small_joiner)
    } else if rest >= 100 && count >= 100 && !g.scale_large_joiner.is_empty() {
        g.scale_large_joiner.clone()
    } else {
        link_default.to_string()
    };
    format!("{head}{link}{}", compose(rest, g, gender, false))
}

fn scale_word(count: i64, forms: &[String]) -> &str {
    if forms.len() == 1 || count == 1 {
        return &forms[0];
    }
    if forms.len() == 2 {
        // singular / plural: Million / Millionen
        return &forms[1];
    }
    let (last_two, last) = (count % 100, count % 10);
    if (2..=4).contains(&last) && !(12..=14).contains(&last_two) {
        &forms[1]
    } else {
        &forms[2]
    }
}

fn hundreds_group(value: i64, g: &Grammar, gender: &str) -> String {
    let (count, rest) = (value / 100, value % 100);
    let mut parts: Vec<String> = Vec::new();
    let hundreds = if !gender.is_empty() {
        g.hundreds_gendered.get(gender).unwrap_or(&g.hundreds)
    } else {
        &g.hundreds
    };
    if !hundreds.is_empty() {
        parts.push(hundreds[count as usize - 1].clone());
    } else if count == 1 && !g.one_before_hundred {
        parts.push(g.hundred.clone());
    } else {
        parts.push(compose(count, g, "", true));
        // French deux cents / deux cent un: the plural mark appears only when
        // the multiplied hundred ends the number.
        if count > 1 && rest == 0 && !g.hundred_plural_final.is_empty() {
            parts.push(g.hundred_plural_final.clone());
        } else {
            parts.push(g.hundred.clone());
        }
    }
    if rest != 0 {
        if !g.hundred_joiner.is_empty() {
            parts.push(g.hundred_joiner.clone());
        }
        parts.push(below_hundred(rest, g, gender, false));
    }
    parts.retain(|p| !p.is_empty());
    parts.join(&g.word_join)
}

fn unit_word(value: i64, g: &Grammar, gender: &str, as_multiplier: bool) -> String {
    let position = if as_multiplier { "tens_pair" } else { "tail" };
    if let Some(w) = g.gendered(value, gender, position) {
        return w.to_string();
    }
    if as_multiplier {
        if let Some(w) = g.combining_ones.get(&value) {
            return w.clone();
        }
    }
    g.ones[value as usize].clone()
}

fn below_hundred(value: i64, g: &Grammar, gender: &str, as_multiplier: bool) -> String {
    if let Some(w) = g.gendered(value, gender, "tail") {
        return w.to_string();
    }
    if let Some(w) = g.exceptions.get(&value) {
        return w.clone();
    }
    if value < 10 {
        return unit_word(value, g, gender, as_multiplier);
    }
    if value < 20 {
        return g.teens[value as usize - 10].clone();
    }
    let (ten, unit) = (value / 10, value % 10);
    let ten_word = g
        .gendered(ten * 10, gender, "tail")
        .map_or_else(|| g.tens[ten as usize - 2].clone(), String::from);
    if unit == 0 {
        return ten_word;
    }
    // A unit inside a tens pair is always in composition: einundzwanzig holds
    // even when the pair ends the number.
    let unit_w = unit_word(unit, g, gender, true);
    let joiner = g
        .tens_joiner_exceptions
        .get(&value)
        .map_or(g.unit_tens_joiner.as_str(), String::as_str);
    if g.units_before_tens {
        format!("{unit_w}{joiner}{ten_word}")
    } else {
        format!("{ten_word}{joiner}{unit_w}")
    }
}

/// The mark `language` writes between a whole number and its fraction.
///
/// Public because the speech funnel needs it outside the number pass: a
/// currency amount is the one place a dot between digits is known not to be a
/// clock time, and the funnel must say so while the symbol is still in hand.
#[must_use]
pub fn decimal_separator(language: &str) -> &'static str {
    // Borrowed from the `LazyLock`, which lives for the program, rather than
    // leaked per call.
    GRAMMARS
        .get(language)
        .map_or(".", |g| g.decimal_separator.as_str())
}

/// Foreign digit systems and their separators, as this language spells them.
///
/// Beside NFC because it is the same kind of pass: one spelling for every pass
/// that follows, and early enough that the symbol table still sees the folded
/// percent sign.
///
/// Language-dependent for the separators, and that is not a detail. U+066B is a
/// *decimal* separator, so folding it to a dot everywhere turned "٣٫١٤" into
/// "3.14" — which in the eleven languages that write decimals with a comma is
/// the written form of a clock time, read out as *drei Uhr vierzehn*.
#[must_use]
pub fn fold_foreign_digits(text: &str, language: &str) -> String {
    let decimal = GRAMMARS
        .get(language)
        .map_or(".", |g| g.decimal_separator.as_str());
    let grouping = if decimal == "." { "," } else { "." };
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\u{0660}'..='\u{0669}' => out.push(char::from(b'0' + (c as u32 - 0x0660) as u8)),
            '\u{06F0}'..='\u{06F9}' => out.push(char::from(b'0' + (c as u32 - 0x06F0) as u8)),
            '\u{066B}' => out.push_str(decimal),
            '\u{066C}' => out.push_str(grouping),
            '\u{066A}' => out.push('%'),
            other => out.push(other),
        }
    }
    out
}

static PHONE_RUN: LazyLock<Regex> =
    // Python's `_PHONE_RUN`: an E.164 number — a plus, then digits, possibly
    // grouped by spaces — read digit by digit and taken before the digit run,
    // which cannot decline it. "+48 123 456 789" is a valid
    // one-to-three-then-threes grouping, so it read as *forty-eight billion*.
    // The plus is the evidence: E.164 requires one, a grouped thousand has none.
    LazyLock::new(|| Regex::new(r"\+[0-9][0-9 ]*[0-9]").unwrap());

/// Below this a plus-signed run is a delta, not a telephone number: "+5
/// degrees" and "+1 000 000 users" are not numbers to spell out.
/// ISO 8601's 24:00. Admitted as an hour, and only with a zero minute.
const END_OF_DAY_HOUR: i64 = 24;

/// Digits in every thousands group after the first.
const GROUP_DIGITS: usize = 3;

const MIN_E164_DIGITS: usize = 8;

static UNICODE_MINUS: LazyLock<Regex> =
    // U+2212 MINUS SIGN and U+2010 HYPHEN, folded to ASCII where a digit
    // follows. Everything downstream reads the sign as `-`, so a typographic
    // minus was not a sign at all: it reached the punctuation pass, became a
    // space, and "−5" was read as *five*. Not U+2013, which writes a range.
    LazyLock::new(|| Regex::new(r"[\u{2212}\u{2010}]([0-9])").unwrap());

fn expand_phone_numbers(text: &str, language: &str) -> String {
    PHONE_RUN
        .replace_all(text, |caps: &regex::Captures<'_>| {
            let whole = &caps[0];
            let digits: Vec<u32> = whole.chars().filter_map(|c| c.to_digit(10)).collect();
            if digits.len() < MIN_E164_DIGITS {
                return whole.to_string();
            }
            let said: Vec<String> = digits
                .iter()
                .filter_map(|d| cardinal(i64::from(*d), language, "").ok())
                .collect();
            if said.len() != digits.len() {
                return whole.to_string();
            }
            said.join(" ")
        })
        .into_owned()
}

/// Whether the token continues past the match into a letter.
///
/// The mirror of `gluedToAWord`, and it was missing here while Python, JS and
/// Swift had it: `123.de` is one token to them and two to this port, which read
/// "einhundertdreiundzwanzig.de". A grouping space is crossed so `200 000x` is one
/// token; the ordinary space in `2024 200 people` is not, because what follows it
/// is a word.
fn glued_forward(text: &str, end: usize) -> bool {
    let bytes = text.as_bytes();
    let mut i = end;
    while i < text.len() {
        let c = text[i..]
            .chars()
            .next()
            .expect("i < len is a char boundary");
        if c.is_alphabetic() {
            return true;
        }
        if c.is_alphanumeric() || c == '_' || c == '.' || c == ',' || c == '-' || c == '+' {
            i += c.len_utf8();
            continue;
        }
        // A group, not "a digit follows": the space is crossed only when three
        // digits start behind it. The loose question walked out of one number
        // and into the next, so `1000 5.1e+3` refused the `1000` — it found the
        // `e` of an exponent two tokens away and called the whole thing one
        // glued token.
        if c == ' ' && i > 0 && bytes[i - 1].is_ascii_digit() && starts_a_group(text, i + 1) {
            i += 1;
            continue;
        }
        return false;
    }
    false
}

/// Whether three digits start at `i` — the shape `DIGIT_RUN` binds as a group
/// after the first, and so the shape a space in front of them may be grouping.
///
/// The length is checked, not assumed: a three-character slice of a
/// two-character tail is two characters, and a lone digit passing for a group
/// reads the `5` of `R2 5` as one.
fn starts_a_group(text: &str, i: usize) -> bool {
    text.as_bytes()
        .get(i..i + GROUP_DIGITS)
        .is_some_and(|group| group.iter().all(u8::is_ascii_digit))
}

/// ...and no fourth digit behind them: a group the pattern could have *bound*
/// rather than a ragged run that only looks like one.
///
/// Which of the two questions a walk asks differs by direction, and the
/// asymmetry is measured rather than an oversight. Forwards the walk finishes
/// the run the pattern refused to bind and a ragged group is exactly why it
/// refused, so `1 0023R` must stay one token. Backwards the group *is* the
/// match, whose width the pattern already fixed, and the loose question there
/// swallows the `1000` of `e3 1000` — a four-digit number across an ordinary
/// space, unrelated to the exponent behind it.
fn continues_a_group(text: &str, i: usize) -> bool {
    starts_a_group(text, i)
        && !text
            .as_bytes()
            .get(i + GROUP_DIGITS)
            .is_some_and(u8::is_ascii_digit)
}

/// Whether a decimal point with digits behind it follows the match.
///
/// A decimal point with digits behind the match means the fraction group shrank
/// to zero so the right-hand guard could land on the dot instead of a letter:
/// `1.5e3` matched just the `1` and read "one.5e3".
fn truncated_by_a_fraction(text: &str, end: usize) -> bool {
    let mut rest = text[end..].chars();
    matches!(rest.next(), Some('.') | Some(',')) && rest.next().is_some_and(|c| c.is_ascii_digit())
}

/// Whether the digit run at `start` sits inside a token containing a letter —
/// Python's backward walk over word characters and dots, which is the question
/// its one-character lookbehind could not ask.
fn glued_to_a_word(text: &str, start: usize) -> bool {
    let bytes = text.as_bytes();
    let mut i = start;
    while i > 0 {
        let c = text[..i].chars().next_back().expect("i > 0");
        // `-` and `+` are in the walk because an exponent puts one between
        // the letter and the digits: in `1e-3` the scan starts at the `3`,
        // walks back over `-` to `e`, and stops calling it a number.
        if c.is_alphanumeric() || c == '_' || c == '.' || c == ',' || c == '-' || c == '+' {
            i -= c.len_utf8();
        } else if c == ' '
            // A digit, then the space: the byte before it, because only an
            // ASCII digit can be one. The test is not redundant with the group
            // ahead — the walk crosses repeatedly, so `Sold 200 000` went
            // `000` -> `200` -> over the space before `200` into "Sold",
            // refusing a number nothing was glued to.
            && i >= 2
            && bytes[i - 2].is_ascii_digit()
            && continues_a_group(text, i)
        {
            // A thousands space, crossed here as it is in the forward walk, and
            // it was missing: `C0200 000` binds as one match, the lookbehind
            // refuses that match, and the `000` then matches on its own —
            // "C0200 zero zero zero", half a token spoken, where Python, JS and
            // Swift leave the whole thing written.
            i -= 1;
        } else {
            return false;
        }
        if c.is_alphabetic() {
            return true;
        }
    }
    false
}

/// Whether a digit sits at `i`, or at `i + 1` behind a space — the two shapes
/// Python's `(?! ?[0-9])` rejects.
fn digits_follow(text: &str, i: usize) -> bool {
    let bytes = text.as_bytes();
    if i < bytes.len() && bytes[i].is_ascii_digit() {
        return true;
    }
    i + 1 < bytes.len() && bytes[i] == b' ' && bytes[i + 1].is_ascii_digit()
}

static DIGIT_RUN: LazyLock<Regex> =
    // ASCII digits only, explicitly — see the Python module for why.
    //
    // Python's `_DIGIT_RUN` minus its lookbehind, which this crate cannot
    // express; that guard is applied in `expand_numbers` against the character
    // before the match. The sign is read backwards there for the same reason:
    // `regex` does not retry a failed match one position to the right the way
    // Python's engine does, so a captured `-?` swallows the hyphen in `1-5`
    // and leaves the `5` unspoken.
    LazyLock::new(|| {
        Regex::new(r"([0-9]{1,3}(?: [0-9]{3})+|[0-9]+)((?:[.,][0-9]+)*)").unwrap()
    });

/// Every run of digits in `text`, said as words — the seam between the
/// verbalizer and the funnel. Never errors and never leaves digits behind: a
/// number past every scale is read digit by digit (it is almost always an
/// identifier), and only the language's own decimal mark is a decimal mark —
/// the other one is grouping, and is dropped the way a reader drops it.
#[must_use]
pub fn expand_numbers(text: &str, language: &str) -> String {
    let Some(g) = GRAMMARS.get(language) else {
        return text.to_string();
    };
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    // Both before anything looks for a digit run: the sign has to be ASCII by
    // the time the pattern matches one, and a phone number has to be gone
    // before the grouping rule meets a shape it cannot decline.
    let folded = UNICODE_MINUS.replace_all(text, "-$1").into_owned();
    let owned = expand_phone_numbers(&folded, language);
    let text: &str = &owned;
    let mut out = String::with_capacity(text.len());
    // `cursor` is what has been written out and `scan` is where the next match
    // is looked for: a cursor of its own rather than `captures_iter`, because a
    // refused match does not always mean a refused *region* — see the
    // lookbehind below. `scan` never moves backwards, which is what keeps
    // `&text[cursor..start]` from slicing backwards on input like
    // `"1 234 567 12."`, a panic the fuzzer found.
    let mut cursor = 0usize;
    let mut scan = 0usize;
    while let Some(caps) = DIGIT_RUN.captures_at(text, scan) {
        let whole = caps.get(0).expect("group 0 always participates");
        // The whole-number group, whose end is where the pattern asks its
        // boundary question — the match may run past it into a fraction.
        let whole_number = caps.get(1).expect("group 1 always participates");
        let (mut start, mut end) = (whole.start(), whole.end());
        // The lookbehind, in code: a run glued to a word is part of that word,
        // so `iOS18` stays written. A leading minus counts only when it is not
        // itself glued to one.
        let sign = text[..start].ends_with('-')
            && text[..start - 1]
                .chars()
                .next_back()
                .is_none_or(|c| !is_word(c));
        if sign {
            start -= 1;
        } else if text[..start].chars().next_back().is_some_and(is_word) {
            // One character on, not past the whole match: what the lookbehind
            // refuses is this *match*, and a match glued at its left edge can
            // still hold a number that is not. `e3 1000` binds as `3 100`, and
            // taking the iterator's next match after that refusal skipped the
            // rest of the region — the thousand behind it was left written.
            // Python's engine retries at every position, which is why it reads
            // it.
            scan = whole.start() + 1;
            continue;
        }
        // Python's `(?! ?[0-9])`, in code: a space-grouped run is a grouped
        // number only if it reaches a boundary. This engine has no lookahead and
        // does not retry the alternation one branch down the way Python's does,
        // so it took the longest prefix that fit and abandoned the rest:
        // "1 202 555 0199" matched "1 202 555 019" and was read as a ten-digit
        // cardinal with a bare "9" trailing behind it.
        //
        // What Python's engine arrives at instead is the *first group*, matched
        // by the other alternative — so that is what this match is cut back to,
        // before anything else looks at it. Everything else then follows: the
        // guards below ask their questions of a real boundary, the groups behind
        // this one are matched in their own turn (and a tidy tail like the
        // `5 000` of `234 567 5 000` is grouped again, as a reader groups it),
        // and a refusal costs one group rather than the whole run.
        //
        // The boundary question is asked at the end of the whole-number group,
        // which is where the pattern asks it: past the fraction it is a
        // different question, and `1 000.0 3` answered it "ragged" and said a
        // plain thousand one digit at a time.
        if whole_number.as_str().contains(' ') && digits_follow(text, whole_number.end()) {
            end = whole_number.start()
                + whole_number
                    .as_str()
                    .find(' ')
                    .expect("a run with a space in it");
        }
        scan = end;
        // What `(?![\w])` was reaching for, plus the backward walk its
        // one-character lookbehind could not do.
        //
        // A run glued to a word on the left was left alone while a run glued to
        // one on the right was expanded up to the letter and abandoned: "5x3"
        // came out *fivex3* and "1e6" *onee6* — a word welded to a digit, which
        // is not a reading of anything. And the lookbehind sees one character,
        // so an identifier that puts a dot between its letter and its digits
        // slipped past it: in "v1.2.3" the scan starts at the `2` and the
        // version came out "v1.two point three".
        if glued_to_a_word(text, start)
            || glued_forward(text, end)
            || truncated_by_a_fraction(text, end)
        {
            continue;
        }
        // `(?![\w])` itself, on the one character the match ends before.
        if text[end..].chars().next().is_some_and(is_word) {
            continue;
        }
        // Normalised once: thousands spaces gone, the sign carried separately.
        // Read from the match rather than the groups, because a ragged one was
        // cut back above and the groups still hold what the pattern first bound.
        let literal = text[whole_number.start()..end].replace(' ', "");
        if !is_quantity(&literal, g) {
            continue;
        }
        out.push_str(&text[cursor..start]);
        let said = say_number(&literal, g, language);
        if sign && !g.minus_word.is_empty() {
            out.push_str(&g.minus_word);
            out.push(' ');
        }
        out.push_str(&said);
        cursor = end;
    }
    out.push_str(&text[cursor..]);
    out
}

/// Whether a digit run is a number rather than a version, an address or a date.
///
/// `1.2.3`, `192.168.0.1` and `12.03.2026` all match the digit-run pattern and
/// none of them is a quantity. Reading one as a quantity says "nineteen million
/// two hundred sixteen thousand eight hundred one" for an IP address — and in
/// the Python reference is a hard crash.
///
/// A run is a quantity when it has at most one separator, or when its
/// separators genuinely group: every segment after the first exactly three
/// digits, the first one to three. Anything else is left as written.
fn is_quantity(literal: &str, g: &Grammar) -> bool {
    let grouping = if g.decimal_separator == "." { "," } else { "." };
    let (whole, fraction, has_fraction) = match literal.split_once(&g.decimal_separator) {
        Some((w, f)) => (w, f, true),
        None => (literal, "", false),
    };
    // A second mark in what should be the fraction: splitting happens once, so
    // this is where "1.2.3" left "2.3" behind and the reference crashed on it.
    if fraction.contains(grouping) || fraction.contains(&g.decimal_separator) {
        return false;
    }
    let segments: Vec<&str> = whole.split(grouping).collect();
    if segments.len() == 1 {
        return true;
    }
    let grouped =
        (1..=3).contains(&segments[0].len()) && segments[1..].iter().all(|seg| seg.len() == 3);
    if grouped {
        return true;
    }
    // Two segments and no fraction is the "2.5 GB" shape: the mark that is not
    // this language's decimal separator, used as one anyway.
    segments.len() == 2 && !has_fraction
}

fn say_number(literal: &str, g: &Grammar, language: &str) -> String {
    // The non-decimal mark is only grouping when it groups: every following
    // segment exactly three digits. Polish "1.000" is a thousand; Polish "2.5"
    // is a de-facto decimal, and 2.5 read as 25 is a changed meaning.
    let grouping = if g.decimal_separator == "." { "," } else { "." };
    let (mut whole, mut fraction) = match literal.split_once(&g.decimal_separator) {
        Some((w, f)) => (w.to_string(), Some(f.to_string())),
        None => (literal.to_string(), None),
    };
    let segments: Vec<&str> = whole.split(grouping).collect();
    if segments.len() > 1 {
        if segments[1..].iter().all(|s| s.len() == 3) {
            whole = segments.concat();
        } else if fraction.is_none() && segments.len() == 2 {
            fraction = Some(segments[1].to_string());
            whole = segments[0].to_string();
        } else {
            whole = segments.concat();
        }
    }
    let fraction = fraction.map(|f| f.replace(grouping, ""));
    let mut parts = vec![say_integer(&whole, language)];
    if let Some(fraction) = fraction {
        if !fraction.is_empty() {
            parts.push(g.decimal_word.clone());
            // Digit by digit — "point four nine", never "point forty-nine":
            // leading zeros carry meaning there that a cardinal would eat.
            parts.extend(digit_by_digit(&fraction, language));
        }
    }
    parts.join(" ")
}

fn say_integer(digits: &str, language: &str) -> String {
    // Leading zeros mean a code, not a quantity: 0042 is zero zero four two.
    if digits.len() > 1 && digits.starts_with('0') {
        return digit_by_digit(digits, language).join(" ");
    }
    if let Ok(n) = digits.parse::<i64>() {
        if let Ok(said) = cardinal(n, language, "") {
            return said;
        }
    }
    digit_by_digit(digits, language).join(" ")
}

fn digit_by_digit(digits: &str, language: &str) -> Vec<String> {
    digits
        .chars()
        .filter_map(|ch| ch.to_digit(10))
        .filter_map(|d| cardinal(i64::from(d), language, "").ok())
        .collect()
}

static TIME_RUN: LazyLock<Regex> =
    // No `\b`: Python guards this with `(?<![\d.,:]) … (?![.,:]?\d)`, which
    // rejects a digit or separator either side and says nothing about letters.
    // `\b` fires between a letter and a digit too, so `a14:30` matched in
    // Python and not here. Both guards live in the neighbour check below.
    LazyLock::new(|| Regex::new(r"([01]?[0-9]|2[0-4])[:.]([0-5][0-9])").unwrap());

/// Clock times as words — see the Python module for the shape and the
/// deliberate absence of the colloquial clock.
#[must_use]
pub fn expand_times(text: &str, language: &str) -> String {
    let Some(g) = GRAMMARS.get(language) else {
        return text.to_string();
    };
    // Rebuilt by index rather than with `replace_all`, because whether a match
    // is a time depends on what sits *outside* it and this regex engine has no
    // lookaround. `12.03` matches inside `12.03.2026` — the ordinary written
    // date of German, Polish, Danish, Finnish and Norwegian — and must not be
    // read as twelve o'clock three with the year trailing behind it.
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    for caps in TIME_RUN.captures_iter(text) {
        let whole = caps.get(0).expect("group 0 always matches");
        if attached_to_digits(bytes, whole.start(), whole.end()) {
            continue;
        }
        // A dot between an hour and two minutes is a clock time in some of
        // these languages and a decimal point in others, and the grammar file
        // already says which: a language that writes 14.30 for half past two
        // does not use the dot as its decimal mark. German writes "14.30 Uhr"
        // and "2,50 €"; English writes "2:30" and "$2.50". Without this every
        // English decimal with two fraction digits read as the clock — "$0.49"
        // as *zero forty-nine* — and the shared fixture pinned one of them, so
        // all five implementations agreed on it.
        if bytes[caps.get(1).expect("group 1 always matches").end()] == b'.'
            && g.decimal_separator == "."
        {
            continue;
        }
        let hour: i64 = caps[1].parse().unwrap_or(0);
        let minute: i64 = caps[2].parse().unwrap_or(0);
        // 24 is admitted only with a zero minute: ISO 8601 writes end-of-day
        // as 24:00, and without it the two halves were read as unrelated
        // numbers with the colon left standing between them. 24:30 is not a
        // time in any convention and stays as written.
        if hour == END_OF_DAY_HOUR && minute != 0 {
            continue;
        }
        let mut words = Vec::new();
        if let Ok(said) = cardinal(hour, language, "") {
            words.push(said);
        }
        if !g.time_infix.is_empty() {
            words.push(g.time_infix.clone());
        }
        if minute != 0 {
            if let Ok(said) = cardinal(minute, language, "") {
                words.push(said);
            }
        }
        let mut end = whole.end();
        if !g.time_infix.is_empty() {
            end = consume_written_infix(bytes, end, &g.time_infix);
        }
        out.push_str(&text[last..whole.start()]);
        out.push_str(&words.join(" "));
        last = end;
    }
    out.push_str(&text[last..]);
    out
}

/// Extends `end` past a written infix word — German writes `um 14.30 Uhr`,
/// and the spoken reading already puts the infix where it belongs, between
/// hour and minutes (*vierzehn Uhr dreißig*). Leaving the written word
/// standing said it twice. Consumed only when it is a whole word immediately
/// after the time; *Uhrzeit* keeps its head.
///
/// ASCII byte scan throughout: space and tab are single bytes in UTF-8 and a
/// letter or digit touching the infix is detected by range, so this matches
/// the other four implementations exactly.
fn consume_written_infix(bytes: &[u8], end: usize, infix: &str) -> usize {
    let infix = infix.as_bytes();
    let mut i = end;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
        i += 1;
    }
    if i == end || i + infix.len() > bytes.len() || &bytes[i..i + infix.len()] != infix {
        return end;
    }
    let after = i + infix.len();
    if after < bytes.len() && bytes[after].is_ascii_alphanumeric() {
        return end;
    }
    after
}

/// Whether the match at `start..end` has a digit or a separator touching either
/// end — what tells `14:30` from the `12.03` inside a date. A trailing sentence
/// period is fine, because what follows it is not a digit.
fn attached_to_digits(bytes: &[u8], start: usize, end: usize) -> bool {
    if start > 0 && matches!(bytes[start - 1], b'0'..=b'9' | b'.' | b',' | b':') {
        return true;
    }
    match bytes.get(end) {
        Some(b'0'..=b'9') => true,
        Some(b'.' | b',' | b':') => matches!(bytes.get(end + 1), Some(b'0'..=b'9')),
        _ => false,
    }
}

/// The authority-listed abbreviations, written out — longest first, word
/// boundaries only. See the Python module.
#[must_use]
pub fn expand_abbreviations(text: &str, language: &str) -> String {
    let Some(g) = GRAMMARS.get(language) else {
        return text.to_string();
    };
    let mut out = text.to_string();
    for (written, spoken) in &g.abbreviations {
        let pattern = format!(r"(^|[^\w.]){}($|[^\w.])", regex::escape(written));
        let re = Regex::new(&pattern).expect("escaped pattern is valid");
        out = re
            .replace_all(&out, |caps: &regex::Captures<'_>| {
                format!("{}{}{}", &caps[1], spoken, &caps[2])
            })
            .to_string();
    }
    out
}
