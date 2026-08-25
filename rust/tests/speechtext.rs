//! The speech funnel, against the fixture every port is checked with.
//!
//! Hand-written cases in five languages are five tests of five different
//! things. tests/data/conformance/speechtext.json is one test of one thing, and
//! a disagreement names itself. That file's own note says so: "Every port must
//! reproduce these exactly; a difference is a divergence, not a dialect" — and
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

/// Ordinary text that used to abort the process.
///
/// The ragged-run branch used to set the cursor past the end of its own match,
/// because it read a whole run of segments the regex saw as several. The next of
/// those still arrived from the iterator, and `&text[cursor..start]` panics:
/// `"1 234 567 12."` died with `begin > end (12 > 10) when slicing`.
///
/// The identical guard went into the Go port first and did not reach here,
/// which is the second time a fix has landed in four implementations of five.
/// Both were found by `tools/fuzz_parity.py`, neither by a fixture.
///
/// A ragged match is cut back to its first group now and the scan is a cursor of
/// its own — a refused lookbehind resumes one character on rather than past the
/// match — so "the cursor is never past the match" is an invariant of two moving
/// indices instead of one. These strings are the fuzzer's, and every one of them
/// makes both of them move.
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

    use loudkit::chunking::{split_text, ChunkConfig};

    let mut bad = Vec::new();
    for case in &fixture.chunking {
        let cfg = ChunkConfig {
            enabled: true,
            max_tokens: case.max_tokens,
            prefix_tokens: case.prefix_tokens,
            split_on: case.split_on.clone(),
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
