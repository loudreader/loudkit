//! The postprocess layer, against the shared conformance fixture.
//!
//! Every case in `tests/data/conformance/postprocess.json` is a regression from
//! the shipped reader or a named device trace, and every port runs the same
//! file. A rule that drifts in one language fails in one language.

use std::collections::HashSet;
use std::path::PathBuf;

use loudkit::postprocess::{
    ceiling_for, desperation_cut, ended_tail_trim, inspect, is_dropout, is_stalled,
    is_trailing_filler, pacing_outliers, repetition_cut, terminal_echo_cut, Config, Mode, Reason,
    RepetitionResume, RepetitionSilence, Request,
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
            // True digital silence in the stall section's two-class scheme.
            "sil" => out.extend((0..count).map(|i| i % 4)),
            // The contextually-quiet family: extends a dead-air run without
            // counting toward its gate.
            "breath" => out.extend((0..count).map(|i| 4 + i % 4)),
            // True silence only the render census knows (the 6405 class):
            // outside the fixture's sampler list, inside its silence census.
            "dead" => out.extend(std::iter::repeat_n(12, count)),
            // Breath only the quiet census knows.
            "sigh" => out.extend(std::iter::repeat_n(13, count)),
            "cycle_dead" => {
                // The stutter with its pause on a census-only id.
                let half = count / 2;
                let mut cycle: Vec<usize> = (0..count - half).map(|i| 20 + i).collect();
                cycle.extend(std::iter::repeat_n(12, half));
                for _ in 0..seg[2].as_u64().unwrap() {
                    out.extend(&cycle);
                }
            }
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
/// on are the numbers the fixture declares rather than this port's own
/// defaults, which is the whole point of a shared file.
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
        desperation_min_keep_per_text_token: c["desperation_min_keep_per_text_token"]
            .as_f64()
            .unwrap_or_else(|| Config::default().desperation_min_keep_per_text_token),
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
        // String fields, absent from the fixture's config block like the
        // band keys, so the shipping values apply.
        repetition_resume: c["repetition_resume"]
            .as_str()
            .map(|v| RepetitionResume::parse(v).unwrap())
            .unwrap_or_else(|| Config::default().repetition_resume),
        repetition_silence: c["repetition_silence"]
            .as_str()
            .map(|v| RepetitionSilence::parse(v).unwrap())
            .unwrap_or_else(|| Config::default().repetition_silence),
        // Like the band keys: absent from the fixture's config block, so the
        // shipping value applies: Python builds its config the same way.
        stall_run_tokens: c["stall_run_tokens"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or_else(|| Config::default().stall_run_tokens),
        silence_render_ids: Vec::new(),
        quiet_render_ids: Vec::new(),
        dropout_min_tokens: n("dropout_min_tokens"),
        retry_max_attempts: n("retry_max_attempts"),
        pacing_tolerance: num("pacing_tolerance"),
    }
}

/// `config_from` plus the stall section's render-id censuses. The fallback arm
/// (`render_ids` false) runs without them: a checkpoint packed before the
/// census, where only the run trigger fires.
fn stall_config(fx: &Value, render_ids: bool) -> Config {
    let mut cfg = config_from(fx, None);
    if render_ids {
        let ids = |key: &str| -> Vec<usize> {
            fx["stall"][key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        };
        cfg.silence_render_ids = ids("silence_render_ids");
        cfg.quiet_render_ids = ids("quiet_render_ids");
    }
    cfg
}

/// `config_from` plus the repetition_silence section's render-id censuses,
/// the ids only the censuses know, which the loop exemption must union in
/// under the acoustic family.
fn rep_silence_config(fx: &Value) -> Config {
    let mut cfg = config_from(fx, None);
    let ids = |key: &str| -> Vec<usize> {
        fx["repetition_silence"][key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    };
    cfg.silence_render_ids = ids("silence_render_ids");
    cfg.quiet_render_ids = ids("quiet_render_ids");
    cfg
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
/// constant tuned on one language is an assumption everywhere else, and the
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

/// The decoder trapped in silence: the failure no tail rule can see.
///
/// Every mute chunk and every mid-row hole in the interior-stall study shipped
/// as clean, because all six other rules anchor on the tail. The stall rule
/// condemns instead of cutting: the failure is a hole, and the fix is the
/// retry ladder. Detection is two-class (a true-silence gate, a quiet-family
/// continuation), which the fixture pins because single-set counting was
/// measured broken.
#[test]
fn stall_matches_the_fixture() {
    let fx = fixture();
    let sil = silence(&fx);
    let cases = fx["stall"]["cases"].as_array().expect("no stall cases");
    assert!(
        !cases.is_empty(),
        "the fixture has no stall cases; nothing was compared"
    );
    for case in cases {
        let cfg = stall_config(&fx, case["render_ids"].as_bool().unwrap());
        let got = is_stalled(
            &build(&case["shape"]),
            case["hit_ceiling"].as_bool().unwrap(),
            &sil,
            &cfg,
        );
        assert_eq!(
            got,
            case["stalled"].as_bool().unwrap(),
            "{}: {}",
            case["name"],
            case["why"]
        );
    }
}

/// The wiring is part of the contract: after repetition, before every tail
/// rescue, condemned like dropout.
#[test]
fn stall_resolver_matches_the_fixture() {
    let fx = fixture();
    let sil = silence(&fx);
    for case in fx["stall"]["cases"].as_array().unwrap() {
        let cfg = stall_config(&fx, case["render_ids"].as_bool().unwrap());
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

/// A cap-hit desperation cut that keeps less than any full read.
///
/// The one row that survived the stall fix: soren/da0028 chunk 4 burned 132
/// tokens to the ceiling and the seam cut kept 36: 1.44 s in which 33 of the
/// 36 kept tokens render near-silent through ids outside both manifest
/// censuses, invisible to every set-membership rule. The keep's *length* is
/// the only evidence there is: a keep under
/// `desperation_min_keep_per_text_token` per text token cannot hold a full
/// read, so the verdict is condemned into the retry ladder. The cut stands as
/// the keep, an exhausted ladder ships the trim, flagged, rather than the
/// untrimmed babble. No censuses configured here on purpose: the trigger is a
/// length test and must fire identically on a checkpoint packed before them.
#[test]
fn starved_rescue_matches_the_fixture() {
    let fx = fixture();
    let sil = silence(&fx);
    let cfg = config_from(&fx, None);
    let cases = fx["starved_rescue"]["cases"]
        .as_array()
        .expect("no starved_rescue cases");
    assert!(
        !cases.is_empty(),
        "the fixture has no starved_rescue cases; nothing was compared"
    );
    for case in cases {
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

/// keep == floor ships: `<`, not `<=`, so the pinned law has no ambiguity at
/// the boundary for a port to resolve differently. Text 20 puts the floor at
/// exactly 34.0.
#[test]
fn starved_rescue_floor_is_exclusive() {
    let fx = fixture();
    let cfg = config_from(&fx, None);
    let row = build(&serde_json::json!([
        ["speech", 34],
        ["sil", 12],
        ["speech", 90]
    ]));
    let got = inspect(
        &row,
        &Request {
            text_token_count: 20,
            min_tokens: 24,
            eos_peak_at: -1,
            eos_peak_prob: 0.0,
            ended: false,
            is_terminal: true,
            hit_ceiling: true,
        },
        &silence(&fx),
        &cfg,
    );
    assert_eq!(got.reason, Reason::Desperation);
    assert_eq!(got.keep, 34);
    assert!(!got.suspect, "a keep exactly at the floor is not under it");
}

#[test]
fn starved_rescue_zero_disables_the_trigger() {
    let fx = fixture();
    let mut cfg = config_from(&fx, None);
    cfg.desperation_min_keep_per_text_token = 0.0;
    let case = &fx["starved_rescue"]["cases"][0];
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
        &silence(&fx),
        &cfg,
    );
    assert_eq!(got.reason, Reason::Desperation);
    assert!(!got.suspect, "zero must disable the trigger");
}

#[test]
fn stall_never_cuts() {
    let fx = fixture();
    let cfg = stall_config(&fx, true);
    let row = build(&serde_json::json!([
        ["speech", 30],
        ["sil", 30],
        ["speech", 30]
    ]));
    let got = inspect(
        &row,
        &Request {
            text_token_count: 40,
            min_tokens: 48,
            eos_peak_at: -1,
            eos_peak_prob: 0.0,
            ended: true,
            is_terminal: true,
            hit_ceiling: false,
        },
        &silence(&fx),
        &cfg,
    );
    assert_eq!(got.reason, Reason::Stall);
    assert_eq!(
        got.keep,
        row.len(),
        "a stalled row must be handed back whole; the hole is mid-row and no \
         cut can remove it"
    );
    assert!(
        got.suspect,
        "the caller has to be told, since nothing was changed"
    );
}

/// A row that both loops and stalls answers to the loop: an exactly repeated
/// cycle pins where the failure began. Here the decoder resumed after the
/// region, so the loop condemns rather than cuts, but it still outranks the
/// stall's condemnation, and the verdict names the anchor that was found.
#[test]
fn repetition_outranks_stall() {
    let fx = fixture();
    let cfg = stall_config(&fx, true);
    let row = build(&serde_json::json!([
        ["cycle", 4, 8],
        ["sil", 30],
        ["speech", 20]
    ]));
    let got = inspect(
        &row,
        &Request {
            text_token_count: 40,
            min_tokens: 48,
            eos_peak_at: -1,
            eos_peak_prob: 0.0,
            ended: true,
            is_terminal: true,
            hit_ceiling: false,
        },
        &silence(&fx),
        &cfg,
    );
    assert_eq!(
        got.reason,
        Reason::Repetition,
        "the exact anchor outranks the condemnation"
    );
}

/// The loop exemption keys on acoustic silence, not the sampler list.
///
/// The specimen is told once, on `Config::repetition_silence`. Keyed to the
/// sampler list alone, the exemption cannot see a pause parked on a
/// census-only id, the period-1 run fires as a loop, and the cut deletes the
/// pause plus the speech behind it. The law unions the sampler list with both
/// render censuses for this one rule; the tail rules keep the list they were
/// calibrated against.
#[test]
fn repetition_silence_matches_the_fixture() {
    let fx = fixture();
    let sil = silence(&fx);
    let cfg = rep_silence_config(&fx);
    let cases = fx["repetition_silence"]["cases"]
        .as_array()
        .expect("no repetition_silence cases");
    assert!(
        !cases.is_empty(),
        "the fixture has no repetition_silence cases; nothing was compared"
    );
    for case in cases {
        let want = case["loop"].as_u64().map(|v| v as usize);
        let got = repetition_cut(&build(&case["shape"]), &sil, &cfg);
        assert_eq!(got, want, "{}: {}", case["name"], case["why"]);
    }
}

/// The cascade is part of the contract: a declined loop falls through to
/// stall, which condemns the specimen's pause into the retry ladder instead
/// of shipping the cut.
#[test]
fn repetition_silence_resolver_matches_the_fixture() {
    let fx = fixture();
    let sil = silence(&fx);
    let cfg = rep_silence_config(&fx);
    for case in fx["repetition_silence"]["cases"].as_array().unwrap() {
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

/// The pre-amendment behaviour stays nameable: a checkpoint measured under
/// it can declare what it measured, and this is what it did: cut at the
/// pause and delete everything behind it.
#[test]
fn sampling_names_the_old_law() {
    let fx = fixture();
    let mut cfg = rep_silence_config(&fx);
    cfg.repetition_silence = RepetitionSilence::Sampling;
    let case = &fx["repetition_silence"]["cases"][0];
    let got = repetition_cut(&build(&case["shape"]), &silence(&fx), &cfg);
    assert_eq!(
        got,
        Some(31),
        "the old law cut one token past the pause's start"
    );
}

/// A checkpoint packed before the censuses changes nothing: nothing on such a
/// build knows id 12 is silent, so the run still reads as a loop there, under
/// either field value.
#[test]
fn without_censuses_the_union_is_the_sampler_list() {
    let fx = fixture();
    let cfg = config_from(&fx, None);
    let case = &fx["repetition_silence"]["cases"][0];
    let got = repetition_cut(&build(&case["shape"]), &silence(&fx), &cfg);
    assert_eq!(got, Some(31));
}

fn resume_request(case: &Value) -> Request {
    Request {
        text_token_count: case["text_tokens"].as_u64().unwrap() as usize,
        min_tokens: case["min_tokens"].as_u64().unwrap() as usize,
        eos_peak_at: case["eos_peak_at"].as_i64().unwrap(),
        eos_peak_prob: case["eos_peak_prob"].as_f64().unwrap(),
        ended: case["ended"].as_bool().unwrap(),
        is_terminal: case["is_terminal"].as_bool().unwrap(),
        hit_ceiling: case["hit_ceiling"].as_bool().unwrap(),
    }
}

/// A loop the decoder resumed from is condemned, never cut.
///
/// The census fix (`repetition_silence`) needs a manifest that names the
/// silent ids, and the published pack has none: on it the en0023 pause fired
/// again as a period-1 loop and the cut kept 52 of 206 tokens, deleting two
/// sentences of correctly-read speech, verdict repetition, no retry. The
/// guard here needs no silence knowledge at all: a genuine lock-up runs its
/// cycle to the end of the row (a ceiling truncates at most one incomplete
/// copy, `period - 1` tokens), so a qualifying loop followed by a full period
/// or more of other content is a decoder that resumed, and a decoder that
/// resumed was never locked. Such a row is handed back whole, suspect, into
/// the retry ladder. The section configures no censuses on purpose: it is
/// the arm `repetition_silence` cannot reach.
#[test]
fn repetition_resume_matches_the_fixture() {
    // The bare rule still reports the loop; the law lives in the resolver.
    let fx = fixture();
    let sil = silence(&fx);
    let cfg = config_from(&fx, None);
    let cases = fx["repetition_resume"]["cases"]
        .as_array()
        .expect("no repetition_resume cases");
    assert!(
        !cases.is_empty(),
        "the fixture has no repetition_resume cases; nothing was compared"
    );
    for case in cases {
        let want = case["loop"].as_u64().map(|v| v as usize);
        let got = repetition_cut(&build(&case["shape"]), &sil, &cfg);
        assert_eq!(got, want, "{}: {}", case["name"], case["why"]);
    }
}

#[test]
fn repetition_resume_resolver_matches_the_fixture() {
    let fx = fixture();
    let sil = silence(&fx);
    let cfg = config_from(&fx, None);
    for case in fx["repetition_resume"]["cases"].as_array().unwrap() {
        let got = inspect(&build(&case["shape"]), &resume_request(case), &sil, &cfg);
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

/// The pre-amendment behaviour stays nameable: a checkpoint measured under
/// it can declare what it measured, and this is what it did: cut at the
/// pause and delete everything behind it.
#[test]
fn cut_names_the_old_law() {
    let fx = fixture();
    let mut cfg = config_from(&fx, None);
    cfg.repetition_resume = RepetitionResume::Cut;
    let case = &fx["repetition_resume"]["cases"][0];
    let got = inspect(
        &build(&case["shape"]),
        &resume_request(case),
        &silence(&fx),
        &cfg,
    );
    assert_eq!(got.reason, Reason::Repetition);
    assert_eq!(
        got.keep,
        case["loop"].as_u64().unwrap() as usize,
        "the old law shipped the specimen's cut"
    );
    assert!(!got.suspect);
}

/// With the censuses configured the specimen's pause is exempt from the loop
/// rule entirely and stall condemns it: the `repetition_silence` contract,
/// byte for byte, guard or no guard.
#[test]
fn the_census_arm_is_untouched() {
    let fx = fixture();
    let cfg = rep_silence_config(&fx);
    let case = &fx["repetition_silence"]["cases"][0];
    let got = inspect(
        &build(&case["shape"]),
        &resume_request(case),
        &silence(&fx),
        &cfg,
    );
    assert_eq!(got.reason, Reason::Stall);
    assert_eq!(got.keep, case["expect"]["keep"].as_u64().unwrap() as usize);
}

/// Early truncation: the failure a listener cannot hear.
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

/// A NaN ratio orders last instead of aborting the process.
///
/// `pacing_outliers` is public and has no caller inside the crate, so its
/// ratios are whatever a consumer passes, and the median sort reached them
/// through `partial_cmp().expect(..)`: rust-27.
#[test]
fn a_nan_ratio_does_not_abort_the_pacing_sort() {
    let cfg = Config::default();
    let finite = [1.0, 1.0, 1.0, 9.0];
    let want = pacing_outliers(&finite, &cfg);

    let with_nan = [1.0, 1.0, 1.0, 9.0, f64::NAN];
    let got = pacing_outliers(&with_nan, &cfg);
    assert_eq!(
        got.iter()
            .filter(|i| **i < finite.len())
            .copied()
            .collect::<Vec<_>>(),
        want,
        "total_cmp orders NaN last, so the finite verdicts must not move"
    );
}

/// The shipping configuration passes its own validator.
///
/// The first thing a validator must not do is refuse the release, so this runs
/// beside every refusal below: rust-04.
#[test]
fn the_shipping_configuration_validates() {
    Config::default()
        .validate()
        .expect("the shipped detector constants must pass their own validator");
}

/// Every shape `PostprocessConfig._validate_ranges` refuses, refused here.
///
/// Two of them were reachable crashes rather than bad taste:
/// `repetition_min_cycles: 0` divides by zero inside `repetition_cut`, and a
/// `retry_max_attempts` past the ladder's headroom derives a seed from a
/// stream a chunk already owns. The rest are the constants whose ordering the
/// detectors assume: rust-04.
#[test]
fn the_validator_refuses_what_python_refuses() {
    /// The field the refusal must name, and the edit that provokes it.
    type Case = (&'static str, fn(&mut Config));

    let cases: [Case; 12] = [
        ("pacing_tolerance", |c| c.pacing_tolerance = f64::NAN),
        ("desperation_band_ratio", |c| {
            c.desperation_band_ratio = f64::INFINITY;
        }),
        ("retry_max_attempts", |c| c.retry_max_attempts = 8),
        ("repetition_min_cycles", |c| c.repetition_min_cycles = 0),
        ("repetition_max_period", |c| c.repetition_max_period = 0),
        ("stall_run_tokens", |c| c.stall_run_tokens = 0),
        ("repetition_min_span", |c| c.repetition_min_span = 2),
        ("ceiling_speech_per_text_token", |c| {
            c.ceiling_speech_per_text_token = 0.0;
        }),
        ("desperation_speech_per_text_token", |c| {
            c.desperation_speech_per_text_token = c.ceiling_speech_per_text_token;
        }),
        ("desperation_min_keep_per_text_token", |c| {
            c.desperation_min_keep_per_text_token = c.desperation_band_ratio + 0.1;
        }),
        ("trailing_filler_threshold", |c| {
            c.trailing_filler_threshold = 0.0;
        }),
        ("filler_min_eos_probability", |c| {
            c.filler_min_eos_probability = 1.0;
        }),
    ];
    for (field, break_it) in cases {
        let mut cfg = Config::default();
        break_it(&mut cfg);
        let err = cfg
            .validate()
            .expect_err("a configuration Python refuses must be refused here");
        assert!(err.contains(field), "the error must name the field: {err}");
    }
}

/// A position percentage above 100 is refused.
///
/// Only one end of Python's `0 <= pct <= 100` survives the port: the other is
/// the `usize` itself.
#[test]
fn a_position_percentage_over_a_hundred_is_refused() {
    let mut strong = Config {
        echo_strong_min_position_pct: 101,
        ..Config::default()
    };
    let err = strong.validate().expect_err("a percentage over 100");
    assert!(err.contains("echo_strong_min_position_pct"), "{err}");

    strong.echo_strong_min_position_pct = 68;
    strong.echo_weak_min_position_pct = 101;
    let err = strong.validate().expect_err("a percentage over 100");
    assert!(err.contains("echo_weak_min_position_pct"), "{err}");
}

/// The division `repetition_cut` performs on `repetition_min_cycles`.
///
/// The validator is what stands between a manifest that sets it to zero and an
/// integer division by zero several seconds into a synthesis: rust-04.
#[test]
fn zero_min_cycles_would_divide_by_zero_in_the_period_search() {
    let mut cfg = Config {
        repetition_min_cycles: 0,
        ..Config::default()
    };
    assert!(cfg.validate().is_err());

    // With the smallest value the validator does admit, the same search runs.
    cfg.repetition_min_cycles = 2;
    cfg.validate().expect("two cycles is the definition of one");
    let tokens: Vec<usize> = vec![5; 8];
    let _ = repetition_cut(&tokens, &HashSet::new(), &cfg);
}
