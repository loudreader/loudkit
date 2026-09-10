//! The character classes the funnel is written in.
//!
//! Every pass asks the same four questions, "is this a letter", "is this a
//! digit", "what digit", "the same run in ASCII", and each answer has to be the
//! one Python, Go, JS and Swift give, or a word boundary lands somewhere else
//! in this port than in the other four. They lived in [`crate::speechtext`],
//! the module that drives the passes, so [`crate::numbers`],
//! [`crate::dates`], [`crate::letters`] and [`crate::respell`] each imported
//! their character classes from their own caller. This is a leaf, which is the
//! direction Python's import graph declares.

use unicode_general_category::{get_general_category, GeneralCategory};

/// `\p{L}`, which is what Python's `str.isalpha`, Go's `unicode.IsLetter` and
/// JS's `\p{L}` all mean.
///
/// Not `char::is_alphabetic`, which is the **Alphabetic** property: wider than
/// `\p{L}` by the circled letters (`So`), the Other_Alphabetic marks and the
/// letter numbers. Measured, that made this port answer differently from the
/// other four wherever a boundary asked "is this a letter".
pub(crate) fn is_letter(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

/// `\p{L}` or `\p{Nd}`: the word class the four other ports test.
pub(crate) fn is_letter_or_digit(c: char) -> bool {
    is_letter(c) || is_decimal_digit(c)
}

/// `str.isdecimal()`, which is what Python's funnel uses: Unicode category
/// `Nd`, not ASCII `0-9`.
///
/// `char::is_ascii_digit` treats every non-ASCII digit as "not alphanumeric",
/// so `punctuation_for_speech` replaces it with a space and Arabic-Indic and
/// fullwidth numerals are **deleted from the text**: `"١٢٣ items"` comes out as
/// `"items"` where Python reads `"sto dwadzieścia trzy ajtamz"`. `is_numeric`
/// would be the easy reach and is wrong in the other direction, it also
/// admits `No` (½) and `Nl` (Ⅻ), which Python's `isdecimal` refuses.
pub(crate) fn is_decimal_digit(c: char) -> bool {
    get_general_category(c) == GeneralCategory::DecimalNumber
}

/// Nd block starts whose preceding code point is itself a decimal digit. Only
/// the four mathematical styled runs qualify.
const ADJACENT_BLOCK_STARTS: [u32; 4] = [0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6];

/// The value of a decimal digit in any script, as `int(token)` reads it in
/// Python.
///
/// `str::parse` understands ASCII digits only, so `"١٢٣"` fails to parse and
/// the Polish number path would decline to spell it, where Python says "sto
/// dwadzieścia trzy". Every `Nd` block is exactly ten consecutive code points,
/// so the value is the distance from the block's zero, found by walking down
/// at most nine.
pub(crate) fn decimal_value(c: char) -> Option<u32> {
    if !is_decimal_digit(c) {
        return None;
    }
    let cp = c as u32;
    let mut zero = cp;
    // Four block starts are listed because their predecessor is also a decimal
    // digit: the mathematical double-struck, sans-serif, sans-serif-bold and
    // monospace runs sit back to back, so walking down from a sans-serif '1'
    // crosses into double-struck and keeps going. Sixty-four of the sixty-eight
    // Nd blocks stop the walk on their own. Verified against Python's own
    // unicodedata over every code point in Unicode: zero disagreements.
    while zero > 0 && !ADJACENT_BLOCK_STARTS.contains(&zero) {
        match char::from_u32(zero - 1) {
            Some(prev) if is_decimal_digit(prev) => zero -= 1,
            _ => break,
        }
    }
    Some(cp - zero)
}

/// A run of decimal digits in any script, rewritten with ASCII ones.
pub(crate) fn ascii_digits(token: &str) -> Option<String> {
    // `from_digit` rather than `unwrap`: a value of ten or more declines here
    // instead of aborting. The funnel is the first thing `synthesize` calls,
    // so a panic here kills the render before any model runs.
    token
        .chars()
        .map(|c| decimal_value(c).and_then(|v| char::from_digit(v, 10)))
        .collect()
}
