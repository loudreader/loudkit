//! The postprocess layer, against the shared conformance fixture.
//!
//! Every case in `tests/data/conformance/postprocess.json` is a regression from
//! the shipped reader or a named device trace, and every port runs the same
//! file. A rule that drifts in one language fails in one language.

use std::collections::HashSet;
use std::path::PathBuf;

use loudkit::postprocess::{
    ceiling_for, desperation_cut, ended_tail_trim, inspect, is_dropout, is_trailing_filler,
    pacing_outliers, repetition_cut, terminal_echo_cut, Config, Mode, Request,
};
use serde_json::Value;

fn fixture_path() -> Option<PathBuf> {
    let p = std::env::var("LOUDKIT_POSTPROCESS_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../tests/data/conformance/postprocess.json"));
    p.exists().then_some(p)
}

fn fixture() -> Value {
    let p = fixture_path().expect("fixture not found; set LOUDKIT_POSTPROCESS_FIXTURE");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// The fixture's token-shape builder, spelled out in its header.
fn build(shape: &Value) -> Vec<usize> {
    let mut out = Vec::new();
    for seg in shape.as_array().unwrap() {
        let kind = seg[0].as_str().unwrap();
        let count = seg[1].as_u64().unwrap() as usize;
        match kind {
            "speech" => out.extend((0..count).map(|i| 20 + i % 60)),
            "quiet" => out.extend((0..count).map(|i| i % 8)),
            "cycle" => {
                // `count` is the period here; seg[2] the repeat count.
                let cycle: Vec<usize> = (0..count).map(|i| 20 + i % 60).collect();
                for _ in 0..seg[2].as_u64().unwrap() {
                    out.extend(&cycle);
                }
            }
            "cycle_mixed" => {
                // Second half silence: the word-then-pause stutter.
                let half = count / 2;
                let mut cycle: Vec<usize> = (0..count - half).map(|i| 20 + i).collect();
                cycle.extend((0..half).map(|i| i % 8));
                for _ in 0..seg[2].as_u64().unwrap() {
                    out.extend(&cycle);
                }
            }
            other => panic!("unknown segment kind {other:?}"),
        }
    }
    out
}

fn silence(fx: &Value) -> HashSet<usize> {
    fx["silence_token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

/// Build the detector config out of the fixture, so the numbers the test runs
/// on are the numbers the fixture declares rather than this port's own defaults
/// — which is the whole point of a shared file.
fn config_from(fx: &Value, mode: Option<&str>) -> Config {
    let c = &fx["config"];
    let num = |k: &str| {
        c[k].as_f64()
            .unwrap_or_else(|| panic!("fixture config missing {k:?}"))
    };
    let n = |k: &str| {
        c[k].as_u64()
            .unwrap_or_else(|| panic!("fixture config missing {k:?}")) as usize
    };
    Config {
        mode: Mode::parse(mode.unwrap_or_else(|| c["mode"].as_str().unwrap())).unwrap(),
        ceiling_speech_per_text_token: num("ceiling_speech_per_text_token"),
        ceiling_slack_tokens: n("ceiling_slack_tokens"),
        trailing_filler_threshold: num("trailing_filler_threshold"),
        trailing_silence_run_tokens: n("trailing_silence_run_tokens"),
        // The band keys predate the fixture; absent means the shipping value,
        // exactly as the manifest readers treat absence.
        desperation_band_ratio: c["desperation_band_ratio"]
            .as_f64()
            .unwrap_or_else(|| Config::default().desperation_band_ratio),
        desperation_band_floor: c["desperation_band_floor"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or_else(|| Config::default().desperation_band_floor),
        filler_min_eos_probability: num("filler_min_eos_probability"),
        filler_max_speech_after_run: n("filler_max_speech_after_run"),
        desperation_speech_per_text_token: num("desperation_speech_per_text_token"),
        desperation_min_text_tokens: n("desperation_min_text_tokens"),
        ended_tail_silence_run: n("ended_tail_silence_run"),
        ended_tail_blip_max: n("ended_tail_blip_max"),
        ended_tail_word_max: n("ended_tail_word_max"),
        ended_tail_keep: n("ended_tail_keep"),
        echo_strong_eos_probability: num("echo_strong_eos_probability"),
        echo_strong_max_tail: n("echo_strong_max_tail"),
        echo_strong_min_position_pct: n("echo_strong_min_position_pct"),
        echo_weak_eos_probability: num("echo_weak_eos_probability"),
        echo_weak_max_tail: n("echo_weak_max_tail"),
        echo_weak_min_position_pct: n("echo_weak_min_position_pct"),
        repetition_max_period: n("repetition_max_period"),
        repetition_min_cycles: n("repetition_min_cycles"),
        repetition_min_span: n("repetition_min_span"),
        dropout_min_tokens: n("dropout_min_tokens"),
        retry_max_attempts: n("retry_max_attempts"),
        pacing_tolerance: num("pacing_tolerance"),
    }
}

/// The fixture's nullable `expect`, as this port's `Option`.
fn want(v: &Value) -> Option<usize> {
    v.as_u64().map(|n| n as usize)
}

/// The shipping constants are the fixture's, or the cases below prove nothing
/// about what actually runs.
#[test]
fn shipping_defaults_match_the_fixture() {
    let fx = fixture();
    assert_eq!(
        Config::default(),
        config_from(&fx, None),
        "Config::default() has drifted from the conformance fixture"
    );
}

#[test]
fn ceiling_matches_the_fixture() {
    let fx = fixture();
    let cfg = config_from(&fx, None);
    for case in fx["ceiling"].as_array().unwrap() {
        let got = ceiling_for(
            case["text_tokens"].as_u64().unwrap() as usize,
            &cfg,
            case["window"].as_u64().unwrap() as usize,
        );
        assert_eq!(
            got,
            case["expect"].as_u64().unwrap() as usize,
            "{}: {}",
            case["name"],
            case["why"]
        );
    }
}

#[test]
fn trailing_filler_matches_the_fixture() {
    let fx = fixture();
    let cfg = config_from(&fx, None);
    let sil = silence(&fx);
    for case in fx["trailing_filler"].as_array().unwrap() {
        let tokens = build(&case["shape"]);
        let got = is_trailing_filler(&tokens, case["from"].as_u64().unwrap() as usize, &sil, &cfg);
        assert_eq!(
            got,
            case["expect"].as_bool().unwrap(),
            "{}: {}",
            case["name"],
            case["why"]
        );
    }
}

#[test]
fn desperation_matches_the_fixture() {
    let fx = fixture();
    let cfg = config_from(&fx, None);
    let sil = silence(&fx);
    for case in fx["desperation"].as_array().unwrap() {
        let got = desperation_cut(
            &build(&case["shape"]),
            case["text_tokens"].as_u64().unwrap() as usize,
            case["min_tokens"].as_u64().unwrap() as usize,
            case["eos_peak_at"].as_i64().unwrap(),
            &sil,
            &cfg,
            case["peak_allowed"].as_bool().unwrap(),
        );
        assert_eq!(
            got,
            want(&case["expect"]),
            "{}: {}",
            case["name"],
            case["why"]
        );
    }
}

#[test]
fn ended_tail_matches_the_fixture() {
    let fx = fixture();
    let cfg = config_from(&fx, None);
    let sil = silence(&fx);
    for case in fx["ended_tail"].as_array().unwrap() {
        let got = ended_tail_trim(
            &build(&case["shape"]),
            &sil,
            &cfg,
            case["is_terminal"].as_bool().unwrap(),
        );
        assert_eq!(
            got,
            want(&case["expect"]),
            "{}: {}",
            case["name"],
            case["why"]
        );
    }
}

#[test]
fn terminal_echo_matches_the_fixture() {
    let fx = fixture();
    let cfg = config_from(&fx, None);
    for case in fx["terminal_echo"].as_array().unwrap() {
        let got = terminal_echo_cut(
            case["token_count"].as_u64().unwrap() as usize,
            case["eos_peak_at"].as_i64().unwrap(),
            case["eos_peak_prob"].as_f64().unwrap(),
            case["min_tokens"].as_u64().unwrap() as usize,
            case["is_terminal"].as_bool().unwrap(),
            case["hit_ceiling"].as_bool().unwrap(),
            &cfg,
        );
        assert_eq!(
            got,
            want(&case["expect"]),
            "{}: {}",
            case["name"],
            case["why"]
        );
    }
}

/// The precedence, which is the part a caller cannot get right by itself.
#[test]
fn resolve_matches_the_fixture() {
    let fx = fixture();
    let sil = silence(&fx);
    for case in fx["resolve"].as_array().unwrap() {
        let cfg = config_from(&fx, case["mode"].as_str());
        let got = inspect(
            &build(&case["shape"]),
            &Request {
                text_token_count: case["text_tokens"].as_u64().unwrap() as usize,
                min_tokens: case["min_tokens"].as_u64().unwrap() as usize,
                eos_peak_at: case["eos_peak_at"].as_i64().unwrap(),
                eos_peak_prob: case["eos_peak_prob"].as_f64().unwrap(),
                ended: case["ended"].as_bool().unwrap(),
                is_terminal: case["is_terminal"].as_bool().unwrap(),
                hit_ceiling: case["hit_ceiling"].as_bool().unwrap(),
            },
            &sil,
            &cfg,
        );
        let expect = &case["expect"];
        let why = format!("{}: {}", case["name"], case["why"]);
        assert_eq!(got.keep, expect["keep"].as_u64().unwrap() as usize, "{why}");
        assert_eq!(
            got.reason.as_str(),
            expect["reason"].as_str().unwrap(),
            "{why}"
        );
        assert_eq!(got.suspect, expect["suspect"].as_bool().unwrap(), "{why}");
    }
}

/// The ceiling was settled on English traces; nine languages ship.
///
/// Speech tokens per *text* token is a property of the orthography, so a
/// constant tuned on one language is an assumption everywhere else — and the
/// expensive direction of that assumption is a guard that truncates correct
/// speech in a language nobody measured. Measured with one voice held constant
/// across nine language tags, because the voice-to-voice spread on a single
/// sentence is larger than the language-to-language spread.
#[test]
fn language_guard_matches_the_fixture() {
    let fx = fixture();
    let cfg = config_from(&fx, None);
    let cases = fx["language_guard"]["cases"]
        .as_array()
        .expect("the fixture has no language_guard cases; nothing was compared");
    assert!(!cases.is_empty());

    let mut stopped: Vec<&str> = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let ceiling = ceiling_for(
            case["text_tokens"].as_u64().unwrap() as usize,
            &cfg,
            case["window"].as_u64().unwrap() as usize,
        );
        assert_eq!(
            ceiling,
            case["expect"].as_u64().unwrap() as usize,
            "{name}: {}",
            case["why"]
        );
        let hit = case["measured_speech_tokens"].as_u64().unwrap() as usize >= ceiling;
        assert_eq!(
            hit,
            case["expect_stopped_by_ceiling"].as_bool().unwrap(),
            "{name} changed side of the ceiling: {}",
            case["why"]
        );
        if hit {
            stopped.push(name);
        }
    }
    // One row belongs here and it is not a false positive: a Spanish three-word
    // phrase whose decoder never emitted a stop token. The guard caught a
    // runaway; it did not cut a legitimate read.
    assert_eq!(
        stopped,
        ["es_short"],
        "a new entry is a language being truncated by an English-tuned constant"
    );
}

/// The loop the tail rules cannot see, because it happens mid-row.
///
/// Every other rule reads the end of the chunk. A stuck decoder repeats inside
/// it, and the literature puts that failure first or second in every ranking of
/// what goes wrong with autoregressive speech models.
#[test]
fn repetition_matches_the_fixture() {
    let fx = fixture();
    let cfg = config_from(&fx, None);
    let sil = silence(&fx);
    let cases = fx["repetition"].as_array().expect("no repetition cases");
    assert!(!cases.is_empty(), "the fixture has no repetition cases");

    let mut negatives = 0;
    for case in cases {
        let want = case["expect"].as_u64().map(|v| v as usize);
        if want.is_none() {
            negatives += 1;
        }
        let got = repetition_cut(&build(&case["shape"]), &sil, &cfg);
        assert_eq!(got, want, "{}: {}", case["name"], case["why"]);
    }
    // A mid-sequence cut is the most destructive thing this layer can do, so
    // the cases that must NOT fire carry more weight than the ones that must.
    assert!(negatives >= 6, "only {negatives} negative cases; too few");
}

/// Early truncation — the failure a listener cannot hear.
///
/// Every other rule says the end of the row is wrong. This one says the row is
/// incomplete, which is why it reports rather than cuts.
#[test]
fn dropout_matches_the_fixture() {
    let fx = fixture();
    let cfg = config_from(&fx, None);
    let cases = fx["dropout"]["cases"].as_array().expect("no dropout cases");
    assert!(!cases.is_empty(), "the fixture has no dropout cases");
    for case in cases {
        let got = is_dropout(
            case["tokens"].as_u64().unwrap() as usize,
            case["text_tokens"].as_u64().unwrap() as usize,
            &cfg,
        );
        assert_eq!(
            got,
            case["expect"].as_bool().unwrap(),
            "{}: {}",
            case["name"],
            case["why"]
        );
    }
}

/// Long-form drift, report-only, in the same integer-derived domain.
#[test]
fn pacing_matches_the_fixture() {
    let fx = fixture();
    let cfg = config_from(&fx, None);
    let cases = fx["pacing"]["cases"].as_array().expect("no pacing cases");
    assert!(!cases.is_empty());
    for case in cases {
        let ratios: Vec<f64> = case["ratios"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let want: Vec<usize> = case["expect"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(
            pacing_outliers(&ratios, &cfg),
            want,
            "{}: {}",
            case["name"],
            case["why"]
        );
    }
}
