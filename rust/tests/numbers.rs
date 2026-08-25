//! The number verbalizer against both shared corpora: the hand-written fixture
//! (expectations from each language's own reference description) and the CLDR
//! differential (1300 rows Unicode wrote; disputed rows skipped with reasons).

use std::path::PathBuf;

use loudkit::numbers::{cardinal, expand_numbers, expand_times};
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
    let Some(fx) = fixture("numbers.json") else {
        eprintln!("fixture not found; skipping");
        return;
    };
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
    let Some(fx) = fixture("numbers_cldr.json") else {
        eprintln!("fixture not found; skipping");
        return;
    };
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
/// turn. Reading the prefix instead said `4 5672.5` as far as the dot —
/// "…setenta y dos.5", a fraction welded to a reading — because the pattern
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
/// three digits behind it — ragged or not, because a ragged group is exactly
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
/// Only a space that groups is crossed — exactly three digits and no fourth,
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
/// `e3 1000` binds as `3 100` — the grouped alternative reaches across the
/// space — and the `e` refuses it. Taking the iterator's next match then
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
}
