//! The number verbalizer against both shared corpora: the hand-written fixture
//! (expectations from each language's own reference description) and the CLDR
//! differential (1300 rows Unicode wrote; disputed rows skipped with reasons).

use std::path::PathBuf;

use loudkit::numbers::{
    cardinal, decimal_separator, expand_numbers, expand_times, supported_languages,
};
use serde_json::Value;

fn fixture(name: &str) -> Option<Value> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/data/conformance")
        .join(name);
    let raw = std::fs::read_to_string(p).ok()?;
    serde_json::from_str(&raw).ok()
}

#[test]
fn cardinal_matches_the_hand_fixture() {
    // Not a soft skip. Both fixtures are committed, so an unreadable one is a
    // broken checkout, and printing a line to stderr while returning a pass is
    // how a port stops comparing anything without anyone noticing.
    let fx = fixture("numbers.json").expect("tests/data/conformance/numbers.json is missing");
    let cardinals = fx["cardinals"].as_object().expect("no cardinals");
    assert!(!cardinals.is_empty(), "nothing was compared");
    for (lang, cases) in cardinals {
        for case in cases.as_array().unwrap() {
            let value = case["value"].as_i64().unwrap();
            let want = case["expect"].as_str().unwrap();
            let got = cardinal(value, lang, "").unwrap_or_else(|e| panic!("{lang} {value}: {e}"));
            assert_eq!(got, want, "{lang} {value}");
        }
    }
    for case in fx["gendered"].as_array().unwrap() {
        let lang = case["language"].as_str().unwrap();
        let value = case["value"].as_i64().unwrap();
        let gender = case["gender"].as_str().unwrap();
        let got = cardinal(value, lang, gender).unwrap();
        assert_eq!(
            got,
            case["expect"].as_str().unwrap(),
            "{lang} {value} g={gender}"
        );
    }
}

#[test]
fn cardinal_matches_cldr() {
    let fx =
        fixture("numbers_cldr.json").expect("tests/data/conformance/numbers_cldr.json is missing");
    let mut checked = 0;
    for (lang, cases) in fx["cases"].as_object().unwrap() {
        for case in cases.as_array().unwrap() {
            if case.get("disputed").is_some() {
                continue;
            }
            let value = case["value"].as_i64().unwrap();
            let gender = case["gender"].as_str().unwrap_or("");
            // Past our scale: the refusal is the declared behaviour.
            let Ok(got) = cardinal(value, lang, gender) else {
                continue;
            };
            checked += 1;
            assert_eq!(
                got,
                case["expect"].as_str().unwrap(),
                "{lang} {value} g={gender:?}"
            );
        }
    }
    assert!(
        checked > 1000,
        "only {checked} CLDR rows ran; the corpus went missing"
    );
}

#[test]
fn expand_numbers_in_running_text() {
    for (text, lang, want) in [
        ("I have 21 apples.", "en", "I have twenty-one apples."),
        ("3.5", "en", "three point five"),
        ("1,200", "en", "one thousand two hundred"),
        ("3,5", "pl", "trzy przecinek pięć"),
        (
            "Es kostet 250 Euro.",
            "de",
            "Es kostet zweihundertfünfzig Euro.",
        ),
        ("21 apples", "xx", "21 apples"),
        ("no numbers here", "en", "no numbers here"),
    ] {
        assert_eq!(expand_numbers(text, lang), want, "{text}");
    }
}

/// A ragged run is one group at a time, each group a match of its own.
///
/// The grouped alternative takes what fits and stops, so what this engine binds
/// is a *prefix* of the run; the match is cut back to its first group, which is
/// what a backtracking engine ends up matching, and the rest arrives in its own
/// turn. Reading the prefix instead said `4 5672.5` as far as the dot,
/// "…setenta y dos.5", a fraction welded to a reading, because the pattern
/// attaches the fraction to the group it starts on, not to the one before it.
///
/// The boundary question is asked at the end of the whole-number group, not the
/// end of the match: past a fraction it is a different question, and `1 000.0 3`
/// answered it "ragged" and read a plain thousand one digit at a time.
#[test]
fn a_ragged_run_is_read_one_group_at_a_time() {
    for (text, lang, want) in [
        (
            "1 234 567 12.",
            "fr",
            "un deux cent trente-quatre cinq cent soixante-sept douze.",
        ),
        // The tail that *is* a grouped number is read as one, because it is
        // matched on its own once the ragged head is out of the way.
        (
            "234 567 5 000",
            "en",
            "two hundred and thirty-four five hundred and sixty-seven five thousand",
        ),
        (
            "4 5672.5",
            "es",
            "cuatro cinco mil seiscientos setenta y dos coma cinco",
        ),
        ("1 000.0 3", "nl", "duizend komma nul drie"),
        (
            "200 000.200 000!",
            "en",
            "two hundred thousand point two zero zero zero zero zero!",
        ),
        (
            "1 000 12.5 3",
            "en",
            "one zero zero zero twelve point five three",
        ),
        // Two separators in one group: no single number to read, so that group
        // is left written while the one before it is said.
        ("1 2345.6.7", "nl", "een 2345.6.7"),
    ] {
        assert_eq!(expand_numbers(text, lang), want, "{text}");
    }
}

/// A run touching a word is left written as far as the glue reaches.
///
/// Every group faces the walks, and the forward one crosses a space that has
/// three digits behind it: ragged or not, because a ragged group is exactly
/// why the pattern refused to bind the run. So `1 0023R` is one token from the
/// `1` to the `R` and stays written, where reading a run of segments spoke half
/// of it: "en nul nul to treR". The walk stops where the run stops, which is
/// what leaves the exponent of `1 000 1e6` written while the thousand in front
/// of it is read: `1e6` starts no group.
#[test]
fn a_run_glued_to_a_word_is_refused_as_far_as_the_glue_reaches() {
    for (text, lang, want) in [
        // Glued at the end: `002` is three digits, so the walk crosses the
        // space and the whole token is one.
        ("1 0023R", "da", "1 0023R"),
        ("1 234 567.é", "fr", "1 234 567.é"),
        // Glued only at the tail: the walk stops at the space in front of the
        // exponent, which starts no group.
        ("1 000 1e6", "no", "én null null null 1e6"),
        (
            "1 234 567 2.5E+1",
            "pt",
            "um duzentos e trinta e quatro quinhentos e sessenta e sete 2.5E+1",
        ),
        (
            "200 000 1e-32.5E+1",
            "fr",
            "deux cents zéro zéro zéro 1e-32.5E+1",
        ),
    ] {
        assert_eq!(expand_numbers(text, lang), want, "{text}");
    }
}

/// The backward walk crosses a thousands space, as the forward one does.
///
/// `C0200 000` binds as a single match here, the lookbehind refuses it, and the
/// `000` then matches on its own: "C0200 zero zero zero", half a token spoken.
/// Only a space that groups is crossed: exactly three digits and no fourth,
/// judged of the group the walk steps out of.
#[test]
fn the_backward_walk_crosses_a_grouping_space() {
    for (text, lang, want) in [
        ("C0200 000", "it", "C0200 000"),
        ("x200 000", "en", "x200 000"),
        // Three digits and a fourth: not a group, so the walk stops at the
        // space and the first group is a number of its own.
        ("a1 000 000", "en", "a1 000 000"),
        // One digit behind the space is the first group and says nothing; one
        // digit *ahead* of it is not a group at all, and `R2 5` is two tokens.
        ("R2 5", "en", "R2 five"),
        // The space that ends a word is not a thousands space no matter what
        // follows it: nothing is glued to this number.
        ("Sold 200 000", "en", "Sold two hundred thousand"),
    ] {
        assert_eq!(expand_numbers(text, lang), want, "{text}");
    }
}

/// A match the lookbehind refuses is not a region refused.
///
/// `e3 1000` binds as `3 100`: the grouped alternative reaches across the
/// space, and the `e` refuses it. Taking the iterator's next match then
/// resumed past the whole thing and left the thousand written; Python's engine
/// retries one character on and reads it. The retry must not resurrect half of
/// a token, which is what the walks are for.
#[test]
fn a_refused_lookbehind_rescans_the_tail() {
    for (text, lang, want) in [
        ("e3 1000", "sv", "e3 ettusen"),
        ("iOS18", "en", "iOS18"),
        ("v1.2.3", "en", "v1.2.3"),
        ("1e6 1000", "it", "1e6 mille"),
    ] {
        assert_eq!(expand_numbers(text, lang), want, "{text}");
    }
}

#[test]
fn a_written_infix_is_not_said_twice() {
    // German writes the time with the word the spoken form also carries: the
    // reading puts the infix between hour and minutes, so the written "Uhr"
    // behind the digits is that same token and is consumed, not duplicated.
    for (text, want) in [
        ("um 14:30 Uhr", "um vierzehn Uhr dreißig"),
        // A tab before the word consumes exactly like a space.
        ("um 14:30\tUhr", "um vierzehn Uhr dreißig"),
        ("um 24:00 Uhr an.", "um vierundzwanzig Uhr an."),
        // The dotted form runs through the second pattern.
        ("Termin um 14.30 Uhr.", "Termin um vierzehn Uhr dreißig."),
        // Without the word nothing changes.
        ("um 14:30", "um vierzehn Uhr dreißig"),
        // The noun on its own is not part of any time.
        (
            "Es ist 14:30 Uhr und die Uhr tickt.",
            "Es ist vierzehn Uhr dreißig und die Uhr tickt.",
        ),
        // Infix inside a longer word keeps its head.
        (
            "Die Uhrzeit ist 14:30.",
            "Die Uhrzeit ist vierzehn Uhr dreißig.",
        ),
    ] {
        assert_eq!(expand_times(text, "de"), want, "{text}");
    }
    // Eleven of the twelve grammars carry an empty infix: nothing to consume.
    assert_eq!(
        expand_times("at 14:30 sharp", "en"),
        "at fourteen thirty sharp"
    );
    // The word needs no space in front of it.
    for (text, want) in [
        ("um 14:30Uhr", "um vierzehn Uhr dreißig"),
        ("Termin um 14.30Uhr.", "Termin um vierzehn Uhr dreißig."),
    ] {
        assert_eq!(expand_times(text, "de"), want, "{text}");
    }
}

/// Both spellings in both cases, plus the dotted forms, which are letters too.
const MERIDIEMS: [&str; 8] = ["am", "pm", "AM", "PM", "Am", "pM", "a.m.", "p.m."];

/// A spoken time is not written against a letter.
///
/// `3:45pm` read *three forty-fivepm*, one word to a listener, where `3:45 pm`
/// read correctly: a space in the source was deciding whether the meridiem was
/// a word at all, and nothing in any of the five suites asked.
#[test]
fn a_spoken_time_is_not_written_against_a_letter() {
    for meridiem in MERIDIEMS {
        let text = format!("Call at 3:45{meridiem}.");
        let want = format!("Call at three forty-five {meridiem}.");
        assert_eq!(expand_times(&text, "en"), want, "{text}");
    }
}

/// Every hour and minute of the clock, in every language: the glued form reads
/// exactly as the spaced one. The separator is the one the language treats as a
/// time, and the hour is written both bare and zero-padded, two matches.
#[test]
fn the_space_in_the_source_decides_nothing() {
    for lang in supported_languages() {
        let separator = if decimal_separator(lang) == "." {
            ":"
        } else {
            "."
        };
        for meridiem in ["pm", "a.m."] {
            for hour in 0..=24 {
                for written_hour in [format!("{hour}"), format!("{hour:02}")] {
                    for minute in 0..60 {
                        let written = format!("{written_hour}{separator}{minute:02}");
                        let glued = expand_times(&format!("at {written}{meridiem} sharp"), lang);
                        let spaced = expand_times(&format!("at {written} {meridiem} sharp"), lang);
                        if expand_times(&written, lang) == written {
                            // Not a clock time here, `24:01` being the whole
                            // set: both forms keep every character.
                            assert_eq!(glued, format!("at {written}{meridiem} sharp"), "{lang}");
                            continue;
                        }
                        assert_eq!(glued, spaced, "{lang} {written}");
                    }
                }
            }
        }
    }
}

/// A letter is the only thing the rule reads: what a following digit or
/// separator refused, it still refuses, and a letter in front of the time is a
/// different question the shared fixture answers.
#[test]
fn only_a_letter_separates_a_spoken_time() {
    for lang in supported_languages() {
        // A seconds field is two digits and the last of them: `10:30:45:60` is
        // a separator run, not a clock, and the refusal that reads the
        // character after the time is what says so.
        for literal in ["12.03.2026", "1.2.3", "24:30", "10:30:45:60"] {
            assert_eq!(expand_times(literal, lang), literal, "{lang}: {literal}");
        }
        for text in ["at 14:30", "at 14:30.", "at 14:30, yes", "at 14:30!"] {
            let said = expand_times(text, lang);
            assert!(!said.contains("  "), "{lang}: {said}");
            assert_eq!(said.trim_end(), said, "{lang}: {said}");
        }
    }
    assert_eq!(
        expand_times("Meet at a14:30.", "en"),
        "Meet at afourteen thirty."
    );
}

/// A clock time carries its seconds, and a zero seconds field says nothing the
/// hour and the minute have not already said.
///
/// The two forms are compared with each other rather than with twelve spellings
/// of the reading, because agreeing is the whole rule. The dotted form is left
/// out on purpose: `10.30.45` is a version string as readily as a timestamp.
#[test]
fn zero_seconds_read_as_no_seconds_at_all() {
    for lang in supported_languages() {
        for (with, without) in [
            ("10:30:00", "10:30"),
            ("3:45:00pm", "3:45pm"),
            ("24:00:00", "24:00"),
        ] {
            assert_eq!(
                expand_times(with, lang),
                expand_times(without, lang),
                "{lang}: {with}"
            );
        }
        // A dotted time takes no seconds anywhere, whatever the language does
        // with the dot between an hour and its minutes.
        assert_eq!(expand_times("10.30.45", lang), "10.30.45", "{lang}");
        assert_ne!(expand_times("10:30:45", lang), "10:30:45", "{lang}");
    }
    // A zero minute is dropped from `10:30` and kept in `10:00:45`, where
    // dropping it would move the seconds into the minutes' place.
    assert_eq!(
        expand_times("10:00:45 and 10:30:00", "en"),
        "ten zero forty-five and ten thirty"
    );
}

/// The one value with no magnitude is an error, not a crash.
///
/// `i64::MIN.abs()` overflows: in debug it panicked, and in release it wrapped
/// back to `i64::MIN`, still negative and still under the ceiling, so the
/// negative arm recursed on the same value until the stack ran out. `cardinal`
/// is public, so the caller is any consumer of the crate: rust-17.
#[test]
fn the_smallest_integer_is_an_error_rather_than_a_panic() {
    for language in ["en", "pl", "de"] {
        let err = cardinal(i64::MIN, language, "")
            .expect_err("i64::MIN is past every grammar's largest scale");
        assert!(err.contains("largest scale"), "{err}");
    }
    // The value next to it still spells, so the guard cost nothing reachable.
    assert!(cardinal(i64::MIN + 1, "en", "").is_err());
    assert_eq!(
        cardinal(-999, "en", "").unwrap(),
        "minus nine hundred and ninety-nine"
    );
}
