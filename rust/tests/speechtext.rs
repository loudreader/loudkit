//! The speech funnel, against the fixture every port is checked with.
//!
//! Hand-written cases in five languages are five tests of five different
//! things. tests/data/conformance/speechtext.json is one test of one thing, and
//! a disagreement names itself. That file's own note says so: "Every port must
//! reproduce these exactly; a difference is a divergence, not a dialect": and
//! until now only Swift read the `cases` section. The three bindings read the
//! `chunking` section and hand-wrote their funnel expectations, which is how
//! three separate divergences (an uppercase language tag, non-ASCII digits, a
//! typographic apostrophe sliced by bytes) stayed green in all three at once.

use loudkit::speechtext::speech_text;

#[derive(serde::Deserialize)]
struct Case {
    text: String,
    language: Option<String>,
    expected: String,
}

#[derive(serde::Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

/// The conformance directory holding speechtext.json.
///
/// LOUDKIT_FIXTURE_DIR, not LOUDKIT_FIXTURE: the two names are not
/// interchangeable. _DIR is the directory, LOUDKIT_FIXTURE is the vectors.json
/// file inside it, which `weightfree.rs` reads directly. Appending
/// "speechtext.json" to the file path resolves to vectors.json/speechtext.json
/// and fails with "not a directory".
fn fixture_dir() -> String {
    std::env::var("LOUDKIT_FIXTURE_DIR").unwrap_or_else(|_| "../tests/data/conformance".to_string())
}

#[test]
fn funnel_matches_the_shared_fixture() {
    let dir = fixture_dir();
    let raw = std::fs::read_to_string(format!("{dir}/speechtext.json"))
        .expect("cannot read the shared fixture");
    let fixture: Fixture = serde_json::from_str(&raw).expect("cannot parse the shared fixture");
    // A renamed key would leave this loop comparing nothing and reporting a
    // pass, which is the failure this whole file exists to prevent.
    assert!(
        !fixture.cases.is_empty(),
        "the fixture has no cases; nothing was compared"
    );

    let mut bad = Vec::new();
    for c in &fixture.cases {
        let lang = c.language.as_deref().unwrap_or("");
        let got = speech_text(&c.text, lang);
        if got != c.expected {
            bad.push(format!(
                "  text={:?} lang={:?}\n    want {:?}\n    got  {:?}",
                c.text, lang, c.expected, got
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "the funnel disagrees with the shared fixture in {}/{} cases:\n{}",
        bad.len(),
        fixture.cases.len(),
        bad.join("\n")
    );
    eprintln!("{} cases compared", fixture.cases.len());
}

/// Ordinary text that aborts the process without the cursor guard.
///
/// A ragged-run branch that sets the cursor past the end of its own match, by
/// reading a whole run of segments the regex sees as several, leaves the next
/// of those still arriving from the iterator, and `&text[cursor..start]`
/// panics: `"1 234 567 12."` dies with `begin > end (12 > 10) when slicing`.
/// Found by `tools/fuzz_parity.py`, not by a fixture.
///
/// A ragged match is cut back to its first group and the scan is a cursor of
/// its own (a refused lookbehind resumes one character on rather than past the
/// match), so "the cursor is never past the match" is an invariant of two
/// moving indices instead of one. These strings are the fuzzer's, and every one
/// of them makes both of them move.
#[test]
fn ragged_runs_beside_other_numbers_do_not_panic() {
    for (text, language) in [
        ("1 234 567 12.", "fr"),
        ("١٢٣ - 200 000 1 CIA 0 - koszt.!", "it"),
        ("200 000.200 000!", "en"),
        ("121 euros 234 567 5 000", "en"),
        ("e3 1000", "sv"),
        ("koszt 1e61 000 1000.$ 1 000.CIA?", "nl"),
        ("and.zł.−000 2024 −", "es"),
        ("1e6 %, 24:00 é kg 1e+3 1000 -?", "sv"),
        ("14.30 24:00.1 234 567 1 000 1e6?", "no"),
    ] {
        let said = loudkit::speechtext::speech_text(text, language);
        assert!(!said.is_empty(), "{text:?} produced nothing");
    }
}

/// The `chunking` section: the same 18 splits Python's `split_text` produces,
/// asserted here so "shared fixture" names all five implementations. This
/// file's header explains what happens when a section goes unread.
#[test]
fn chunking_matches_the_shared_fixture() {
    let dir = fixture_dir();
    let raw = std::fs::read_to_string(format!("{dir}/speechtext.json"))
        .expect("cannot read the shared fixture");

    #[derive(serde::Deserialize)]
    struct ChunkCase {
        text: String,
        max_tokens: usize,
        prefix_tokens: usize,
        split_on: Vec<String>,
        abbreviations: Vec<String>,
        mid_sentence_period: String,
        chunks: Vec<String>,
    }

    #[derive(serde::Deserialize)]
    struct ChunkFixture {
        chunking: Vec<ChunkCase>,
    }

    let fixture: ChunkFixture =
        serde_json::from_str(&raw).expect("cannot parse the shared fixture");
    assert!(
        !fixture.chunking.is_empty(),
        "the fixture has no chunking cases; nothing was compared"
    );

    use loudkit::chunking::{split_text, ChunkConfig, WORD_CAP_RESPLIT};

    let mut bad = Vec::new();
    for case in &fixture.chunking {
        let cfg = ChunkConfig {
            // Not a splitter input: `cap_resplit` decides what the engine does
            // with a window that overran, so the fixture does not vary it.
            cap_resplit: WORD_CAP_RESPLIT.to_string(),
            enabled: true,
            max_tokens: case.max_tokens,
            prefix_tokens: case.prefix_tokens,
            split_on: case.split_on.clone(),
            abbreviations: case.abbreviations.clone(),
            mid_sentence_period: case.mid_sentence_period.clone(),
        };
        let got = split_text(&case.text, &cfg);
        if got != case.chunks {
            bad.push(format!("{}: {:?} != {:?}", case.text, got, case.chunks));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} chunking cases disagree with the fixture:\n  {}",
        bad.len(),
        fixture.chunking.len(),
        bad.join("\n  ")
    );
}

/// A number past the grammar's largest scale leaves the text as written.
///
/// The date passes reach `cardinal` through a helper that turned its error
/// into the empty string, and every caller concatenates the result, so an
/// unspellable value shipped as a fragment beginning with a space in the
/// middle of a sentence. Declining is what `replace` already does whenever
/// evidence runs out: rust-18.
#[test]
fn an_unspellable_ordinal_is_left_as_written() {
    // The last two digits spell and the head does not, which is the shape that
    // produced " twenty-first" where the number had been.
    let text = "the 1000000000000000021st time";
    let got = loudkit::dates::expand_ordinals(text, "en");
    assert_eq!(got, text, "a value with no words for it stays as digits");
    assert!(!got.contains("  "), "no fragment with a hole in it: {got}");
    // The ordinals the funnel actually reads are untouched.
    assert_eq!(
        loudkit::dates::expand_ordinals("the 21st time", "en"),
        "the twenty-first time"
    );
}

/// The same decline for a year, where the public function returns a `String`
/// and so cannot say `None`: it answers with the digits: rust-18.
#[test]
fn an_unspellable_year_comes_back_as_its_digits() {
    assert_eq!(
        loudkit::dates::say_year(i64::MAX, "en"),
        i64::MAX.to_string()
    );
    // A language with no grammar at all took the same path to an empty string.
    assert_eq!(loudkit::dates::say_year(1984, "zz"), "1984");
    // The years the funnel actually reads are untouched.
    assert_eq!(loudkit::dates::say_year(1984, "en"), "nineteen eighty-four");
}

/// Two prices written side by side stay two prices.
///
/// The prefix pass has no lookbehind, so its "not glued to a word" guard used
/// to *match* the preceding character rather than look at it. That character
/// then belonged to the match, the scan resumed past it, and every amount
/// after the first was taken apart differently from the reference: the unit
/// word duplicated, the number moved behind it, and with three amounts the
/// second and third joined into one number nobody wrote. The values below are
/// the reference's, which leaves the second amount's digits glued to the first
/// amount's unit word so the number pass declines to read them: written digits
/// are the intended outcome, because a confident wrong number cannot be
/// undone.
#[test]
fn adjacent_currency_amounts_stay_separate() {
    assert_eq!(speech_text("$1$2$3", "en"), "one dollars2 dollars3 dollars");
    assert_eq!(speech_text("$5$10", "en"), "five dollars10 dollars");
    assert_eq!(speech_text("£3£4", "en"), "three pounds4 pounds");
    assert_eq!(speech_text("£3£4", "pl"), "trzy funtów4 funtów");
    assert_eq!(speech_text("€1€2", "de"), "eins Euro2 Euro");
    assert_eq!(speech_text("$5$10", "pt"), "cinco dólares10 dólares");
    // The guard the consumed character was there for still holds: a letter in
    // front means a multi-character mark this table has no wording for, so the
    // amount reads as a plain decimal and the mark is dropped.
    assert_eq!(speech_text("R$3,14", "pt"), "R três vírgula um quatro");
    // And an ordinary price is unchanged.
    assert_eq!(
        speech_text("It costs $5 today.", "en"),
        "It costs five dollars today."
    );
}

/// An abbreviation is written out every time it appears, not every other time.
///
/// The same shape as the currency guard, three functions away: the reference's
/// word boundaries are zero-width, this port's were characters the match
/// consumed, and the single space between two occurrences could only serve one
/// of them.
#[test]
fn every_occurrence_of_an_abbreviation_is_written_out() {
    assert_eq!(speech_text("tzn. tzn.", "pl"), "to znaczy to znaczy");
    assert_eq!(
        speech_text("np. np. np.", "pl"),
        "na przykład na przykład na przykład"
    );
    assert_eq!(speech_text("etc. etc.", "en"), "etcetera etcetera");
    assert_eq!(speech_text("z.B. z.B.", "de"), "zum Beispiel zum Beispiel");
    // The boundary itself is unchanged: two run together are not two
    // abbreviations, because neither stands at a word boundary.
    assert_eq!(speech_text("tzn.tzn.", "pl"), "tzn.tzn.");
    assert_eq!(speech_text("etc.etc.", "en"), "etc.etc.");
}
