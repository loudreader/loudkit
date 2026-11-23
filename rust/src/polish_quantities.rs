//! Polish units take different forms in `1 kg`, `2 kg` and `5 kg`.
//! Resolve the noun while its written unit is still present, then let the
//! shared number grammar speak the digits.

use crate::numbers::scale_word;
use crate::unicode::is_letter;
use regex::{Captures, Match, Regex};
use std::sync::LazyLock;
use unicode_general_category::{get_general_category, GeneralCategory};

static QUANTITY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"([0-9]+(?:[ \x{a0}\x{202f}][0-9]{3})*)(?:,([0-9]+))?(?:[ \t\x{a0}\x{202f}]*(zł|kg|m²))?",
    )
    .unwrap()
});
static CHOICE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"([Ww]ybierz|[Zz]aznacz|[Ww]skaż) +([1-9]) +z +(10|[2-9]) +",
        r"(?:(ofert|opcji|pozycji|propozycji|wiadomości)|(produktów|plików|biletów|elementów))",
    ))
    .unwrap()
});
static MESSAGES: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([Mm]asz) +2( +(?:nowe +)?wiadomości)").unwrap());

fn word_char(c: char) -> bool {
    c == '_'
        || is_letter(c)
        || matches!(
            get_general_category(c),
            GeneralCategory::DecimalNumber
                | GeneralCategory::LetterNumber
                | GeneralCategory::OtherNumber
        )
}

fn bounded(text: &str, m: Match<'_>) -> bool {
    !text[..m.start()].chars().next_back().is_some_and(word_char)
        && !text[m.end()..].chars().next().is_some_and(word_char)
}

fn quantity(text: &str, caps: &Captures<'_>) -> Option<String> {
    let unit = caps.get(3)?.as_str();
    let m = caps.get(0)?;
    let mut before = text[..m.start()].chars().rev();
    if !bounded(text, m)
        || (matches!(before.next(), Some('.' | ','))
            && before.next().is_some_and(|c| c.is_ascii_digit()))
    {
        return None;
    }
    let raw = &caps[1];
    let digits: String = raw.chars().filter(char::is_ascii_digit).collect();
    if digits.len() > 9
        || (digits.starts_with('0') && digits.len() > 1)
        || (raw.len() != digits.len() && raw.bytes().take_while(u8::is_ascii_digit).count() > 3)
    {
        return None;
    }
    let whole = digits.parse().ok()?;
    let fraction = caps.get(2).map_or("", |v| v.as_str());
    match unit {
        "zł" => {
            if fraction.len() > 2 {
                return None;
            }
            let cents = match fraction.len() {
                0 => 0,
                1 => fraction.parse::<i64>().ok()? * 10,
                _ => fraction.parse::<i64>().ok()?,
            };
            let pln = scale_word(whole, &["złoty", "złote", "złotych"]);
            let gr = scale_word(cents, &["grosz", "grosze", "groszy"]);
            Some(if cents == 0 {
                format!("{whole} {pln}")
            } else if whole == 0 {
                format!("{cents} {gr}")
            } else {
                format!("{whole} {pln} i {cents} {gr}")
            })
        }
        unit => {
            if fraction.len() > 6 {
                return None;
            }
            let (forms, fractional) = if unit == "kg" {
                (["kilogram", "kilogramy", "kilogramów"], "kilograma")
            } else {
                (
                    ["metr kwadratowy", "metry kwadratowe", "metrów kwadratowych"],
                    "metra kwadratowego",
                )
            };
            Some(if fraction.bytes().any(|d| d != b'0') {
                format!("{whole},{fraction} {fractional}")
            } else {
                format!("{whole} {}", scale_word(whole, &forms))
            })
        }
    }
}

pub(crate) fn expand(text: &str) -> String {
    const GENITIVE: [&str; 9] = [
        "dwóch",
        "trzech",
        "czterech",
        "pięciu",
        "sześciu",
        "siedmiu",
        "ośmiu",
        "dziewięciu",
        "dziesięciu",
    ];
    let choices = CHOICE.replace_all(text, |caps: &Captures<'_>| {
        if !bounded(text, caps.get(0).unwrap()) {
            return caps[0].to_owned();
        }
        let feminine = caps.get(4);
        let lhs = match (&caps[2], feminine.is_some()) {
            ("1", true) => "jedną",
            ("2", true) => "dwie",
            (digits, _) => digits,
        };
        let rhs = GENITIVE[caps[3].parse::<usize>().unwrap() - 2];
        let noun = feminine.unwrap_or_else(|| caps.get(5).unwrap()).as_str();
        format!("{} {lhs} z {rhs} {noun}", &caps[1])
    });
    let counts = MESSAGES.replace_all(&choices, |caps: &Captures<'_>| {
        if bounded(&choices, caps.get(0).unwrap()) {
            format!("{} dwie{}", &caps[1], &caps[2])
        } else {
            caps[0].to_owned()
        }
    });
    QUANTITY
        .replace_all(&counts, |caps: &Captures<'_>| {
            quantity(&counts, caps).unwrap_or_else(|| caps[0].to_owned())
        })
        .into_owned()
}
