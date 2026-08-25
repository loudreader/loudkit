//! Dates and ordinals, said the way each language says them.
//!
//! A port of `loudkit.frontend.dates`: `12.03.2026` is the
//! ordinary written date of five of these twelve languages, and without this
//! funnel it reads as a clock time with a stray year, or as one eight-digit
//! number. `1st`
//! arrives as *onest*, because the number pass expands the digits and leaves the
//! suffix stuck to them.
//!
//! Every rule is data from the shared `numbers.json` — month names, day forms,
//! the infixes Spanish and Portuguese speak between the parts, the German
//! oblique triggers, the ordinal tables. What is code here is the *shape*: which
//! written forms are dates at all, and how each language reads a year.
//!
//! Two refusals are as deliberate as anything it does. A yearless `12.3.` is
//! never matched — its closing period is indistinguishable from a sentence's, so
//! `Die Zahl ist 3.5.` would otherwise come out as *dritte Mai*. And `3/12/2026` is left alone
//! in English, where it is March twelfth to half the world and the third of
//! December to the other half: a listener recovers from hearing digits, not from
//! a confident wrong month.
//! Python reference: `loudkit/frontend/dates.py`.

use crate::numbers::cardinal;
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

/// Above this a four-digit run is an identifier, not a year.
const MAX_YEAR: i64 = 2999;
/// A three-digit year exists; a three-digit *anything* is far more often a
/// quantity, and nothing in the string separates them.
const MIN_YEAR: i64 = 1000;
/// February is 29 on purpose: a plausibility bound, not a calendar. Refusing 29
/// February in a common year would reject a date a human wrote deliberately.
const DAYS_IN_MONTH: [i64; 12] = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

#[derive(Default)]
pub struct Rules {
    day_words: HashMap<i64, String>,
    day_words_oblique: HashMap<i64, String>,
    oblique_triggers: Vec<String>,
    day_one_word: String,
    months: Vec<String>,
    day_month_infix: String,
    month_year_infix: String,
    day_first_prefix: String,
    day_first_infix: String,
    year_rule: String,
    year_units: HashMap<i64, String>,
    year_teens: HashMap<i64, String>,
    year_tens: HashMap<i64, String>,
    year_two_thousand: String,
    dotted_is_ambiguous: bool,
    no_dotted_dates: bool,
    ord_suffixes: Vec<String>,
    ord_units: HashMap<i64, String>,
    ord_teens: HashMap<i64, String>,
    ord_tens: HashMap<i64, String>,
    ord_joiner: String,
}

static RULES: LazyLock<HashMap<String, Rules>> = LazyLock::new(|| {
    let doc: serde_json::Value =
        serde_json::from_str(include_str!("numbers.json")).expect("numbers.json unreadable");
    let Some(langs) = doc["languages"].as_object() else {
        return HashMap::new();
    };
    fn int_keys(v: &serde_json::Value) -> HashMap<i64, String> {
        v.as_object()
            .map(|m| {
                m.iter()
                    .filter_map(|(k, val)| {
                        let s = val.as_str()?;
                        if s.is_empty() {
                            return None;
                        }
                        Some((k.parse::<i64>().ok()?, s.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
    fn strings(v: &serde_json::Value) -> Vec<String> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }
    fn text(v: &serde_json::Value) -> String {
        v.as_str().unwrap_or_default().to_string()
    }

    let mut out = HashMap::new();
    for (lang, entry) in langs {
        let d = &entry["dates"];
        if !d.is_object() {
            continue;
        }
        let o = &entry["ordinals"];
        let joiner = match o["tens_joiner"].as_str() {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => "-".to_string(),
        };
        out.insert(
            lang.clone(),
            Rules {
                day_words: int_keys(&d["day_words"]),
                day_words_oblique: int_keys(&d["day_words_oblique"]),
                oblique_triggers: strings(&d["oblique_triggers"]),
                day_one_word: text(&d["day_one_word"]),
                months: strings(&d["months"]),
                day_month_infix: text(&d["day_month_infix"]),
                month_year_infix: text(&d["month_year_infix"]),
                day_first_prefix: text(&d["day_first_prefix"]),
                day_first_infix: text(&d["day_first_infix"]),
                year_rule: text(&d["year_rule"]),
                year_units: int_keys(&d["year_units"]),
                year_teens: int_keys(&d["year_teens"]),
                year_tens: int_keys(&d["year_tens"]),
                year_two_thousand: text(&d["year_two_thousand"]),
                dotted_is_ambiguous: d["dotted_is_ambiguous"].as_bool().unwrap_or(false),
                no_dotted_dates: d["no_dotted_dates"].as_bool().unwrap_or(false),
                ord_suffixes: strings(&o["suffixes"]),
                ord_units: int_keys(&o["units"]),
                ord_teens: int_keys(&o["teens"]),
                ord_tens: int_keys(&o["tens"]),
                ord_joiner: joiner,
            },
        );
    }
    out
});

fn card(n: i64, lang: &str) -> String {
    cardinal(n, lang, "").unwrap_or_default()
}

/// The month's name in this language, or `None` when it has no table.
pub fn month_name(month: i64, language: &str) -> Option<String> {
    let r = RULES.get(language)?;
    if !(1..=12).contains(&month) || r.months.len() != 12 {
        return None;
    }
    Some(r.months[(month - 1) as usize].clone())
}

/// The day-of-month word, in whatever form this language's dates take.
///
/// `oblique` is German only — the `-en` ending that `am`/`den`/`vom` select.
pub fn ordinal_day(day: i64, language: &str, oblique: bool) -> Option<String> {
    let r = RULES.get(language)?;
    if !(1..=31).contains(&day) {
        return None;
    }
    if oblique {
        if let Some(w) = r.day_words_oblique.get(&day) {
            return Some(w.clone());
        }
    }
    if let Some(w) = r.day_words.get(&day) {
        return Some(w.clone());
    }
    // Cardinal languages: the day is just a number, except where the first of
    // the month is lexicalised.
    if day == 1 && !r.day_one_word.is_empty() {
        return Some(r.day_one_word.clone());
    }
    Some(card(day, language))
}

/// A year, read the way this language reads years.
///
/// English and Norwegian split it; German, Dutch and Swedish group it in
/// hundreds; the rest say one plain cardinal. Spanish is the explicit case — the
/// RAE writes that a year is read as its cardinal and *not* in two-figure blocks
/// as in English, so 2021 is *dos mil veintiuno*.
pub fn say_year(year: i64, language: &str) -> String {
    let Some(r) = RULES.get(language) else {
        return card(year, language);
    };
    match r.year_rule.as_str() {
        "en_split" => year_english(year),
        "de_hundreds" => year_hundreds(year, "de", "hundert", 1100, 1999),
        "nl_hundreds" => year_hundreds(year, "nl", "honderd", 1100, 1999),
        "sv_hundreds" => year_hundreds(year, "sv", "hundra", 1100, 2099),
        "no_split" => year_norwegian(year),
        "da_long" => year_danish(year),
        "pl_ordinal_genitive" => year_polish(year, r),
        _ => card(year, language),
    }
}

fn year_english(year: i64) -> String {
    if year == 1000 || year == 2000 || (2001..=2009).contains(&year) {
        return card(year, "en");
    }
    if (1001..=1999).contains(&year) || year >= 2100 {
        let (century, rest) = (year / 100, year % 100);
        if rest == 0 {
            return format!("{} hundred", card(century, "en"));
        }
        // "nineteen oh five" — never "nineteen five", which nobody says.
        if rest < 10 {
            return format!("{} oh {}", card(century, "en"), card(rest, "en"));
        }
        return format!("{} {}", card(century, "en"), card(rest, "en"));
    }
    if (2010..=2099).contains(&year) {
        return format!("twenty {}", card(year % 100, "en"));
    }
    card(year, "en")
}

/// German, Dutch and Swedish all write `<century><joiner><rest>` solid; only the
/// joiner and the range differ. German stops at 1999 because the GfdS explicitly
/// rejects `zwanzighundert…`; Swedish runs to 2099 because Isof has recommended
/// the `tjugohundra…` series for decades.
fn year_hundreds(year: i64, lang: &str, joiner: &str, lo: i64, hi: i64) -> String {
    if !(lo..=hi).contains(&year) {
        return card(year, lang);
    }
    let (century, rest) = (year / 100, year % 100);
    let head = format!("{}{}", card(century, lang), joiner);
    if rest == 0 {
        head
    } else {
        format!("{head}{}", card(rest, lang))
    }
}

/// Norwegian splits 1100–1999 and drops `hundre`: 1972 is `nittensyttito`.
fn year_norwegian(year: i64) -> String {
    if !(1100..=1999).contains(&year) {
        return card(year, "no");
    }
    let (century, rest) = (year / 100, year % 100);
    if rest == 0 {
        format!("{}hundre", card(century, "no"))
    } else {
        format!("{}{}", card(century, "no"), card(rest, "no"))
    }
}

/// Dansk Sprognævn: the long form works for every year, and the short
/// "telephone-number" form is explicitly poor for a century's first decade.
fn year_danish(year: i64) -> String {
    if !(1100..=1999).contains(&year) {
        return card(year, "da");
    }
    let (century, rest) = (year / 100, year % 100);
    let head = format!("{} hundrede", card(century, "da"));
    if rest == 0 {
        head
    } else {
        format!("{head} og {}", card(rest, "da"))
    }
}

/// Only the tens and units of a Polish year decline. PWN's worked example is
/// *tysiąc dziewięćset dziewięćdziesiątego drugiego*: the thousands and hundreds
/// keep their cardinal form and the ordinal genitive lands on the last two
/// digits. Where those are zero the declension moves left, which is why 2000 has
/// its own word.
fn year_polish(year: i64, r: &Rules) -> String {
    if year == 2000 && !r.year_two_thousand.is_empty() {
        return r.year_two_thousand.clone();
    }
    let (head, rest) = (year / 100, year % 100);
    let lead = if head != 0 {
        card(head * 100, "pl")
    } else {
        String::new()
    };
    if rest == 0 {
        return lead;
    }
    let tail = if let Some(teen) = r.year_teens.get(&rest) {
        teen.clone()
    } else {
        let words = [
            r.year_tens
                .get(&((rest / 10) * 10))
                .cloned()
                .unwrap_or_default(),
            r.year_units.get(&(rest % 10)).cloned().unwrap_or_default(),
        ];
        words
            .iter()
            .filter(|w| !w.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    };
    format!("{lead} {tail}").trim().to_string()
}

fn valid(day: i64, month: i64, year: Option<i64>) -> bool {
    if !(1..=12).contains(&month) {
        return false;
    }
    if !(1..=DAYS_IN_MONTH[(month - 1) as usize]).contains(&day) {
        return false;
    }
    year.is_none_or(|y| (MIN_YEAR..=MAX_YEAR).contains(&y))
}

fn spoken(
    day: i64,
    month: i64,
    year: Option<i64>,
    language: &str,
    oblique: bool,
) -> Option<String> {
    let r = RULES.get(language)?;
    let mut parts = vec![ordinal_day(day, language, oblique)?];
    if !r.day_month_infix.is_empty() {
        parts.push(r.day_month_infix.clone());
    }
    parts.push(month_name(month, language)?);
    if let Some(y) = year {
        if !r.month_year_infix.is_empty() {
            parts.push(r.month_year_infix.clone());
        }
        parts.push(say_year(y, language));
    }
    Some(parts.join(" "))
}

static ISO: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([12][0-9]{3})-([01][0-9])-([0-3][0-9])").unwrap());
/// With the year, which is what makes it a date rather than a guess. The
/// yearless `12.3.` is deliberately not matched — see the module note.
static DOTTED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([0-3]?[0-9])\.([01]?[0-9])\.([12][0-9]{3})").unwrap());
/// Day-first in every language here; English is handled in the callback, where
/// the field order is genuinely ambiguous.
static SLASHED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([0-3]?[0-9])/([01]?[0-9])/([12][0-9]{3})").unwrap());

/// Every written date in `text`, said the way `language` says it.
///
/// Never panics and never invents: a run failing the bounds check, or whose
/// field order cannot be resolved, comes back exactly as it was written.
pub fn expand_dates(text: &str, language: &str) -> String {
    let Some(r) = RULES.get(language) else {
        return text.to_string();
    };
    let out = replace(text, &ISO, &|g: &[String], at, whole: &str| {
        let (y, m, d) = (num(&g[1]), num(&g[2]), num(&g[3]));
        if !bounded_before(whole, at, "0123456789.,:/-")
            || !bounded_after(whole, at + g[0].len(), "0123456789-")
            || !valid(d, m, Some(y))
        {
            return None;
        }
        spoken(d, m, Some(y), language, is_oblique(whole, at, r))
    });
    let out = replace(&out, &DOTTED, &|g: &[String], at, whole: &str| {
        // Swedish marks an ordinal with a colon (`1:a`), never a trailing
        // period, so `12.` there is a list number or a sentence end. English
        // writes dotted dates almost never, and when it does the field order is
        // as unresolvable as in the slashed form.
        if r.no_dotted_dates || r.dotted_is_ambiguous {
            return None;
        }
        let (d, m, y) = (num(&g[1]), num(&g[2]), num(&g[3]));
        if !bounded_before(whole, at, "0123456789.,:/-")
            || !word_boundary_after(whole, at + g[0].len())
            || !valid(d, m, Some(y))
        {
            return None;
        }
        spoken(d, m, Some(y), language, is_oblique(whole, at, r))
    });
    let out = replace(&out, &SLASHED, &|g: &[String], at, whole: &str| {
        let (d, m, y) = (num(&g[1]), num(&g[2]), num(&g[3]));
        if !bounded_before(whole, at, "0123456789.,:/-")
            || !bounded_after(whole, at + g[0].len(), "0123456789/")
        {
            return None;
        }
        // `3/12/2026` is March twelfth to half the English-speaking world and
        // the third of December to the other half, and nothing says which.
        if language == "en" && d <= 12 {
            return None;
        }
        if !valid(d, m, Some(y)) {
            return None;
        }
        spoken(d, m, Some(y), language, is_oblique(whole, at, r))
    });
    textual(&out, language, r)
}

fn num(s: &str) -> i64 {
    s.parse().unwrap_or(-1)
}

/// Rust's `regex` has no lookaround, so the guards Python writes as
/// `(?<![\d.,:/-])` and `(?![\d/])` are checked here against the characters
/// either side of a match. Same rule, expressed where the engine can express it.
fn bounded_before(s: &str, at: usize, class: &str) -> bool {
    s[..at]
        .chars()
        .next_back()
        .is_none_or(|c| !class.contains(c))
}

fn bounded_after(s: &str, end: usize, class: &str) -> bool {
    s[end..].chars().next().is_none_or(|c| !class.contains(c))
}

fn word_boundary_after(s: &str, end: usize) -> bool {
    s[end..]
        .chars()
        .next()
        .is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
}

fn word_boundary_before(s: &str, at: usize) -> bool {
    s[..at]
        .chars()
        .next_back()
        .is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
}

/// German only: `am`/`den`/`vom` before the day select the `-en` ending.
fn is_oblique(whole: &str, at: usize, r: &Rules) -> bool {
    if r.oblique_triggers.is_empty() || at > whole.len() {
        return false;
    }
    let before = whole[..at].trim_end();
    let Some(last) = before.split_whitespace().next_back() else {
        return false;
    };
    let tail = last.to_lowercase();
    let tail = tail.trim_matches(|c| c == ',' || c == ';' || c == ':');
    r.oblique_triggers.iter().any(|w| w.to_lowercase() == tail)
}

/// `12 marca 2026`, `12. März 2026`, `March 12, 2026` — a written month name
/// beside a bare day. The name is the disambiguator, so this runs for every
/// language including English.
fn textual(text: &str, language: &str, r: &Rules) -> String {
    if r.months.len() != 12 {
        return text.to_string();
    }
    let names = r
        .months
        .iter()
        .map(|m| regex::escape(m))
        .collect::<Vec<_>>()
        .join("|");
    // Spanish and Portuguese speak a preposition between every part, so the
    // written form carries it too: "12 de marzo de 2026".
    let infix = if r.day_month_infix.is_empty() {
        String::new()
    } else {
        format!(r"(?:\s+{})?", regex::escape(&r.day_month_infix))
    };
    let yinfix = if r.month_year_infix.is_empty() {
        String::new()
    } else {
        format!(r"(?:\s+{})?", regex::escape(&r.month_year_infix))
    };

    let day_first = Regex::new(&format!(
        r"(?i)([0-3]?[0-9])\.?{infix}\s+({names})(?:{yinfix}\s+([12][0-9]{{3}}))?"
    ))
    .expect("day-first date pattern");
    let out = replace(text, &day_first, &|g: &[String], at, whole: &str| {
        if !word_boundary_before(whole, at) || !word_boundary_after(whole, at + g[0].len()) {
            return None;
        }
        let d = num(&g[1]);
        let m = month_index(&g[2], r)?;
        let y = g.get(3).filter(|s| !s.is_empty()).map(|s| num(s));
        if !valid(d, m, y) {
            return None;
        }
        if !r.day_first_prefix.is_empty() || !r.day_first_infix.is_empty() {
            // English written day-first reads "the twelfth of March": both
            // dialects say it that way, so no locale flag is needed.
            let head = ordinal_day(d, language, false)?;
            let mut rest = vec![month_name(m, language)?];
            if let Some(y) = y {
                rest.push(say_year(y, language));
            }
            let prefix = if r.day_first_prefix.is_empty() {
                String::new()
            } else {
                format!("{} ", r.day_first_prefix)
            };
            let join = if r.day_first_infix.is_empty() {
                " ".to_string()
            } else {
                format!(" {} ", r.day_first_infix)
            };
            return Some(format!("{prefix}{head}{join}{}", rest.join(" ")));
        }
        spoken(d, m, y, language, is_oblique(whole, at, r))
    });

    // Month-first is an English shape. Reading it in a language that never
    // writes it would be inventing a construction nobody used.
    if r.day_first_infix.is_empty() {
        return out;
    }
    let month_first = Regex::new(&format!(
        r"(?i)({names})\s+([0-3]?[0-9])(?:st|nd|rd|th)?,?(?:\s+([12][0-9]{{3}}))?"
    ))
    .expect("month-first date pattern");
    replace(&out, &month_first, &|g: &[String], at, whole: &str| {
        if !word_boundary_before(whole, at) || !word_boundary_after(whole, at + g[0].len()) {
            return None;
        }
        let m = month_index(&g[1], r)?;
        let d = num(&g[2]);
        let y = g.get(3).filter(|s| !s.is_empty()).map(|s| num(s));
        if !valid(d, m, y) {
            return None;
        }
        let mut parts = vec![month_name(m, language)?, ordinal_day(d, language, false)?];
        if let Some(y) = y {
            parts.push(say_year(y, language));
        }
        Some(parts.join(" "))
    })
}

fn month_index(name: &str, r: &Rules) -> Option<i64> {
    let lowered = name.to_lowercase();
    r.months
        .iter()
        .position(|m| m.to_lowercase() == lowered)
        .map(|i| i as i64 + 1)
}

/// `value` as a written-out ordinal, or `None` when this language has no table.
///
/// Composed rather than enumerated past ninety-nine: the hundreds and above stay
/// cardinal and only the last two digits become an ordinal, so *101st* is "one
/// hundred and first".
pub fn ordinal(value: i64, language: &str) -> Option<String> {
    let r = RULES.get(language)?;
    if r.ord_units.is_empty() || value < 0 {
        return None;
    }
    let (head, rest) = (value / 100, value % 100);
    let tail = two_digit_ordinal(rest, r)?;
    if head == 0 {
        return Some(tail);
    }
    let lead = card(head * 100, language);
    Some(if rest != 0 {
        format!("{lead} {tail}")
    } else {
        lead
    })
}

fn two_digit_ordinal(value: i64, r: &Rules) -> Option<String> {
    if let Some(teen) = r.ord_teens.get(&value) {
        return Some(teen.clone());
    }
    let (tens, units) = (value / 10, value % 10);
    if units == 0 {
        return r.ord_tens.get(&(tens * 10)).cloned();
    }
    if tens == 0 {
        return r.ord_units.get(&units).cloned();
    }
    let unit_word = r.ord_units.get(&units)?;
    // The tens word is English's, because English is the only language of the
    // twelve writing an ordinal as digits plus a suffix.
    Some(format!(
        "{}{}{unit_word}",
        card(tens * 10, "en"),
        r.ord_joiner
    ))
}

/// `1st` and `22nd` as words.
///
/// English is the only one of the twelve writing an ordinal as digits plus a
/// letter suffix, so for every other language this is a no-op. It runs before
/// the number pass, which would otherwise expand the digits and leave the
/// suffix stuck to them: *onest*, *fiveth place*, *twenty-twond*.
pub fn expand_ordinals(text: &str, language: &str) -> String {
    let Some(r) = RULES.get(language) else {
        return text.to_string();
    };
    if r.ord_suffixes.is_empty() {
        return text.to_string();
    }
    let Ok(re) = Regex::new(&format!(r"(?i)([0-9]+)({})", r.ord_suffixes.join("|"))) else {
        return text.to_string();
    };
    replace(text, &re, &|g: &[String], at, whole: &str| {
        if !word_boundary_before(whole, at) || !word_boundary_after(whole, at + g[0].len()) {
            return None;
        }
        ordinal(num(&g[1]), language)
    })
}

/// What `replace` calls for each match: capture groups, the match's offset, and
/// the whole string. Named because the signature is three arguments wide and
/// reads worse inline than it does here.
type SubstitutionBody<'a> = dyn Fn(&[String], usize, &str) -> Option<String> + 'a;

/// Rewrite every match, right to left so earlier offsets stay valid.
///
/// The callback gets the capture groups (index 0 is the whole match), the match
/// offset, and the string being scanned — the last two because the German
/// oblique test reads the word *before* the date, and because `regex` cannot
/// express the lookaround Python uses. Returning `None` leaves that match
/// exactly as written, which is this module's answer whenever evidence runs out.
fn replace(text: &str, re: &Regex, body: &SubstitutionBody) -> String {
    let matches: Vec<_> = re.captures_iter(text).collect();
    if matches.is_empty() {
        return text.to_string();
    }
    let mut out = text.to_string();
    for caps in matches.iter().rev() {
        let whole = caps.get(0).expect("group 0 always participates");
        let groups: Vec<String> = caps
            .iter()
            .map(|g| g.map(|m| m.as_str().to_string()).unwrap_or_default())
            .collect();
        if let Some(said) = body(&groups, whole.start(), text) {
            out.replace_range(whole.start()..whole.end(), &said);
        }
    }
    out
}
