//! A digit run with two or more separators is a version, an address or a date —
//! never a number.
//!
//! All three used to be read as one: with the comma as the decimal mark the dots
//! were treated as thousands grouping and the segments concatenated, so
//! `192.168.0.1` was spoken as "nineteen million two hundred sixteen thousand
//! eight hundred one". The Python reference additionally crashed on these, which
//! is how the class was found.
//!
//! Regression tests: every literal below is one that shipped wrong.

use loudkit::numbers::{expand_numbers, expand_times, supported_languages};

const NOT_QUANTITIES: [&str; 5] = [
    "1.2.3",
    "1.2.3.4",
    "192.168.0.1",
    "12.03.2026",
    "10.0.0.255",
];

#[test]
fn digits_that_are_not_quantities_are_left_alone() {
    for lang in supported_languages() {
        for literal in NOT_QUANTITIES {
            assert_eq!(
                expand_numbers(literal, lang),
                literal,
                "{lang}: {literal} was read as a number"
            );
        }
    }
}

#[test]
fn real_numbers_still_read() {
    // The guard must not buy correctness by refusing everything.
    for lang in supported_languages() {
        for literal in ["7", "2,5", "2.5"] {
            assert_ne!(
                expand_numbers(literal, lang),
                literal,
                "{lang}: {literal} was left as digits"
            );
        }
    }
}

#[test]
fn grouped_thousands_are_still_a_number() {
    // Two separators that *group* are a number: the rule is "three digits after
    // the first separator", not "at most one separator".
    for lang in supported_languages() {
        if lang == "en" {
            continue; // English groups with commas, not dots
        }
        assert_ne!(expand_numbers("1.234.567", lang), "1.234.567", "{lang}");
    }
    assert_ne!(expand_numbers("1,234,567", "en"), "1,234,567");
}

#[test]
fn a_time_is_not_part_of_a_date() {
    // `12.03` matched inside `12.03.2026`, so the ordinary written date of five
    // of the twelve languages was spoken as a clock time with the year trailing.
    for lang in supported_languages() {
        for literal in ["12.03.2026", "am 05.11.2025 kam"] {
            assert_eq!(expand_times(literal, lang), literal, "{lang}: {literal}");
        }
        for literal in ["14:30", "at 14:30."] {
            assert_ne!(expand_times(literal, lang), literal, "{lang}: {literal}");
        }
        // A dotted time reads only where the dot is not the decimal point: `14.30`
        // is half past two in eleven of these languages and a number in the
        // twelfth. Asserting it for all twelve made every English decimal with two
        // fraction digits a clock time.
        if lang != "en" {
            assert_ne!(expand_times("14.30", lang), "14.30", "{lang}: 14.30");
        } else {
            assert_eq!(expand_times("14.30", lang), "14.30", "{lang}: 14.30");
        }
    }
}
