//! Weight-free conformance vectors: RNG, sampler, frontend, seed derivation.
//! These run against `tests/data/conformance/vectors.json` — the same fixture
//! pytest, swift test, and the JS/Go bindings verify. A drift here is a broken
//! port, not "close enough".

use std::path::PathBuf;

use loudkit::frontend::Frontend;
use loudkit::rng;
use loudkit::sampler::{self, Sampler};

fn fixture_path() -> Option<PathBuf> {
    let p = std::env::var("LOUDKIT_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../tests/data/conformance/vectors.json"));
    p.exists().then_some(p)
}

fn tokenizer_path() -> Option<PathBuf> {
    let p = std::env::var("LOUDKIT_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../tests/data/conformance/tokenizer.json"));
    p.exists().then_some(p)
}

fn vectors() -> serde_json::Value {
    let p = fixture_path().expect("fixture not found; set LOUDKIT_FIXTURE");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn to_f64(v: &serde_json::Value) -> f64 {
    if v.is_null() {
        return 0.0;
    }
    v.as_f64().unwrap_or_else(|| v.as_u64().unwrap() as f64)
}

/// The cases for one fixture section, refusing an empty list.
///
/// Every loop below iterates a slice pulled out of the fixture by key. A
/// regeneration that renamed one — `philox` to `rng`, say — would leave the
/// loop comparing nothing and the test reporting a pass, which is the entire
/// cross-language determinism claim quietly switched off. The correct pattern
/// already existed three times in this repo; it was missing from the rest.
fn cases_of<'a>(section: &'a serde_json::Value, key: &str) -> &'a Vec<serde_json::Value> {
    let list = section
        .get(key)
        .unwrap_or_else(|| panic!("the fixture has no {key:?} section; nothing was compared"))
        .as_array()
        .unwrap_or_else(|| panic!("fixture section {key:?} is not a list"));
    assert!(
        !list.is_empty(),
        "fixture section {key:?} is empty; nothing was compared"
    );
    list
}

#[test]
fn philox_kat() {
    let fixture = vectors();
    let philox = cases_of(&fixture["philox"], "kat");
    for c in philox {
        let counter: Vec<u32> = c["counter"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| to_f64(x) as u32)
            .collect();
        let key: Vec<u32> = c["key"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| to_f64(x) as u32)
            .collect();
        let want: Vec<u32> = c["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| to_f64(x) as u32)
            .collect();
        let got = rng::philox4x32(
            counter[0], counter[1], counter[2], counter[3], key[0], key[1],
        );
        assert_eq!(got.to_vec(), want, "counter {:?}", counter);
    }
}

#[test]
fn uniform_bits() {
    let fixture = vectors();
    let philox = cases_of(&fixture["philox"], "uniform_bits");
    for p in philox {
        let seed =
            u64::from_str_radix(p["seed"].as_str().unwrap().trim_start_matches("0x"), 16).unwrap();
        let stream = to_f64(&p["stream"]) as u32;
        let step0 = to_f64(&p["step0"]) as usize;
        let n_steps = to_f64(&p["n_steps"]) as usize;
        let width = to_f64(&p["width"]) as usize;
        let u = rng::uniforms(seed, stream, step0, n_steps, width);
        let got: Vec<u32> = u
            .iter()
            .map(|x| (x * 4294967296.0 - 0.5).round() as u32)
            .collect();
        let want: Vec<u32> = p["bits"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|x| to_f64(x) as u32))
            .collect();
        assert_eq!(got, want, "seed {}", p["seed"]);
    }
}

#[test]
fn gumbel() {
    let fixture = vectors();
    let philox = cases_of(&fixture["philox"], "gumbel");
    for p in philox {
        let seed = to_f64(&p["seed"]) as u64;
        let stream = to_f64(&p["stream"]) as u32;
        let step = to_f64(&p["step"]) as usize;
        let width = to_f64(&p["width"]) as usize;
        let g = rng::gumbel_noise(seed, stream, step, 1, width);
        let vals = p["values"].as_array().unwrap();
        for (i, w) in vals.iter().enumerate() {
            let rel = ((g[i] - to_f64(w)) / to_f64(w)).abs();
            assert!(rel < 1e-12, "seed {} idx {} rel {}", seed, i, rel);
        }
    }
}

#[test]
fn sampler_choices() {
    let fixture = vectors();
    let cases = cases_of(&fixture["sampler"], "cases");
    for c in cases {
        let cfg = &c["config"];
        let config = sampler::Config {
            temperature: to_f64(&cfg["temperature"]),
            repetition_penalty: to_f64(&cfg["repetition_penalty"]),
            min_p: to_f64(&cfg["min_p"]),
            max_new_tokens: to_f64(&cfg["max_new_tokens"]) as usize,
            silence_token_ids: cfg["silence_token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| to_f64(x) as usize)
                .collect(),
            min_tokens_floor: 0,
            min_tokens_text_ratio: 0.0,
        };
        let mut s = Sampler::new(config, to_f64(&c["seed"]) as u64);
        let rows: Vec<Vec<f32>>;
        if let Some(r) = c["logits_recipe"].as_object() {
            rows = (0..to_f64(&r["steps"]) as usize)
                .map(|step| {
                    let u = rng::uniforms(
                        to_f64(&r["seed"]) as u64,
                        to_f64(&r["stream"]) as u32,
                        step,
                        1,
                        to_f64(&r["vocab"]) as usize,
                    );
                    u.iter()
                        .map(|x| (x * to_f64(&r["scale"]) + to_f64(&r["offset"])) as f32)
                        .collect()
                })
                .collect();
        } else {
            let base: Vec<f32> = c["logits"][0]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| to_f64(x) as f32)
                .collect();
            let repeat = to_f64(&c["repeat_logits"]) as usize;
            rows = vec![
                base;
                if repeat > 0 {
                    repeat
                } else {
                    c["logits"].as_array().unwrap().len()
                }
            ];
        }
        let mut seen = vec![false; rows[0].len()];
        let mut got = Vec::new();
        for (step, row) in rows.iter().enumerate() {
            let t = s.call(row, step, &seen);
            got.push(t);
            seen[t] = true;
        }
        let want: Vec<usize> = c["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| to_f64(x) as usize)
            .collect();
        assert_eq!(got, want, "case {}", c["name"]);
    }
}

#[test]
fn frontend_ids() {
    let tp = tokenizer_path().expect("tokenizer not found; set LOUDKIT_TOKENIZER");
    let fe = Frontend::load(tp.to_str().unwrap()).unwrap();
    let fixture = vectors();
    let cases = cases_of(&fixture["frontend"], "cases");
    for c in cases {
        let ids = fe
            .encode(c["text"].as_str().unwrap(), c["language"].as_str().unwrap())
            .unwrap();
        let want: Vec<usize> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| to_f64(x) as usize)
            .collect();
        assert_eq!(ids, want, "text {}", c["text"]);
    }
}

/// The ceiling `Engine::load` checks the checkpoint's text embedding table
/// against.
///
/// `encode` can return any id in the vocabulary, and every one of them indexes
/// that table; a tokenizer paired with a checkpoint from another release used to
/// read past its end mid-synthesis. The shipped weights carry 2454 rows
/// (`TorchTokenGenerator.TEXT_VOCAB`), so 2453 is the last id that fits — the
/// margin is one row, which is why a regenerated fixture must show up here, as
/// a line to read, rather than in a panic on someone's laptop.
#[test]
fn the_vocabulary_ceiling_is_known() {
    let tp = tokenizer_path().expect("tokenizer not found; set LOUDKIT_TOKENIZER");
    let fe = Frontend::load(tp.to_str().unwrap()).unwrap();
    assert_eq!(fe.max_token_id(), 2453);
}

#[test]
fn seed_derivation() {
    const PHI: u64 = 0x9e3779b97f4a7c15;
    const PSI: u64 = 0xbf58476d1ce4e5b9;
    let deriv = &vectors()["seeds"]["derivation"];
    for p in deriv.as_array().unwrap() {
        let seed = to_f64(&p["seed"]) as u64;
        let stream = to_f64(&p["stream"]) as u64;
        let got = seed
            .wrapping_mul(PHI)
            .wrapping_add(stream.wrapping_mul(PSI));
        let want = u64::from_str_radix(p["derived"].as_str().unwrap().trim_start_matches("0x"), 16)
            .unwrap();
        assert_eq!(got, want, "seed {} stream {}", seed, stream);
    }
}

/// The four chunking recipes Python refuses, and this port used to accept.
///
/// `ChunkConfig` was a plain struct read straight from the manifest. The
/// zero-budget case is why it matters: `split_text` cuts nothing and loops
/// forever, which on a server is a wedged request holding the engine. Python
/// fixed that in `d8742aa`, on the Python side only.
#[test]
fn chunk_config_refuses_what_python_refuses() {
    use loudkit::chunking::ChunkConfig;

    assert!(
        ChunkConfig::default().validate().is_ok(),
        "the shipping recipe must validate"
    );

    let seps: Vec<String> = vec![". ".to_string()];
    let cases: Vec<(&str, ChunkConfig, &str)> = vec![
        (
            "zero max",
            ChunkConfig {
                enabled: true,
                max_tokens: 0,
                prefix_tokens: 0,
                split_on: seps.clone(),
            },
            "must be positive",
        ),
        (
            "no budget",
            ChunkConfig {
                enabled: true,
                max_tokens: 1,
                prefix_tokens: 0,
                split_on: seps.clone(),
            },
            "no character budget",
        ),
        (
            "prefix >= max",
            ChunkConfig {
                enabled: true,
                max_tokens: 20,
                prefix_tokens: 20,
                split_on: seps.clone(),
            },
            "prefix_tokens must be in",
        ),
        (
            "no separators",
            ChunkConfig {
                enabled: true,
                max_tokens: 20,
                prefix_tokens: 6,
                split_on: vec![],
            },
            "nowhere to break",
        ),
    ];
    for (name, cfg, want) in cases {
        let err = cfg.validate().expect_err(name);
        assert!(
            err.contains(want),
            "{name}: got {err:?}, want it to mention {want:?}"
        );
    }
}

/// An explicit Euler grid overrides the cosine schedule.
///
/// `time_grid` took only the step count, so a checkpoint shipping an explicit
/// grid rendered on a different integration schedule here — silently, and
/// under a fingerprint that recorded the grid being ignored. An explicit grid
/// exists precisely because "cosine" is a formula two codebases can write two
/// ways (`config.py:296`).
#[test]
fn time_grid_honours_an_explicit_grid() {
    let cosine = loudkit::windowing::time_grid(2, None);
    assert_eq!(cosine.len(), 3);
    assert!((cosine[0]).abs() < 1e-12);
    assert!((cosine[2] - 1.0).abs() < 1e-12);

    let explicit = loudkit::windowing::time_grid(2, Some(&[0.0, 0.25, 1.0]));
    assert_eq!(
        explicit,
        vec![0.0, 0.25, 1.0],
        "the explicit grid was ignored"
    );
}

/// A static window shorter than the tokens it must hold is an error, not a panic.
///
/// `frame_windows` guarded on `max_speech_tokens` and then wrote into a buffer
/// `static_length` long, so a manifest declaring 300/255 index-panicked and
/// killed the process where Python raises a ValueError from numpy.
#[test]
fn a_static_window_shorter_than_the_token_budget_is_refused() {
    use loudkit::windowing::{frame_windows, WindowConfig};

    let cfg = WindowConfig {
        max_speech_tokens: 300,
        static_length: Some(255),
        pad_token_id: Some(4254),
        static_prompt_tokens: Some(238),
    };
    let err = frame_windows(&cfg, Some(4254), &[], &[1, 2, 3], &[1, 2], &[0.0; 80])
        .expect_err("a window that cannot hold its own budget must be refused");
    assert!(err.contains("static_length"), "got {err:?}");
}

/// The algorithm fingerprint, against the one the fixture ships.
///
/// Every other cross-language check here compares a behaviour somebody thought
/// to compare. This compares the *whole* configuration in one string, so a
/// field nobody wrote a test for still cannot drift — which is not
/// hypothetical: `euler_grid` was ignored by this port, `silence_token_ids`
/// accepted a string, and `chunking.prefix_tokens` was guessed rather than
/// read. Each was found by hand. This finds the next one for free.
#[test]
fn fingerprint_matches_the_shared_fixture() {
    use loudkit::chunking::ChunkConfig;
    use loudkit::engine::EngineConfig;
    use loudkit::fingerprint::{canonical_form, fingerprint};
    use loudkit::windowing::WindowConfig;

    let fixture = vectors();
    let algorithm = &fixture["algorithm"];

    // The production algorithm, spelled out rather than loaded, so this runs
    // with no checkpoint: the fingerprint is a property of the values, and the
    // values are what the fixture pins.
    let cfg = EngineConfig {
        text: loudkit::engine::TextConfig::default(),
        recipe_version: "loudkit-1".to_string(),
        guidance: "single_path".to_string(),
        guidance_rate: 0.0,
        sample_rate: 24_000,
        token_rate_hz: 25.0,
        euler_steps: 2,
        euler_grid: None,
        start_speech: 6561,
        stop_speech: 6562,
        speech_vocab_size: 8194,
        window: WindowConfig {
            max_speech_tokens: 255,
            static_length: Some(255),
            pad_token_id: Some(4254),
            static_prompt_tokens: Some(238),
        },
        sampling: loudkit::sampler::Config {
            temperature: 0.8,
            repetition_penalty: 1.2,
            min_p: 0.05,
            max_new_tokens: 255,
            silence_token_ids: vec![
                1731, 1821, 1822, 1824, 1975, 2058, 2068, 3190, 3377, 3918, 3927, 3928, 3930, 4008,
                4009, 4011, 4012, 4137, 4146, 4161, 4171, 4173, 4174, 4218, 4245, 4251, 4252, 4254,
                4255, 4260, 4282,
            ],
            min_tokens_floor: 10,
            min_tokens_text_ratio: 1.2,
        },
        chunking: ChunkConfig::default(),
        postprocess: loudkit::postprocess::Config::default(),
    };

    // The blob first: a mismatch there names the field that drifted, while a
    // mismatch in the hash alone says only that something did.
    assert_eq!(
        canonical_form(&cfg),
        algorithm["canonical_form"].as_str().unwrap(),
        "canonical form differs — the field that differs is visible in the diff"
    );
    assert_eq!(
        fingerprint(&cfg),
        algorithm["fingerprint"].as_str().unwrap()
    );
}

/// The stop-token observation the postprocess layer reads.
///
/// Pinned across languages because it is hand-written in five of them and it is
/// *audible*: two of the detector rules compare it against a threshold, so a
/// port that computes it differently cuts a chunk somewhere else. The quantity
/// has two subtleties either of which a reimplementation gets wrong silently —
/// the numerator is the stop token's weight taken BEFORE the min_p cutoff, and
/// the peak is recorded only PAST the floor.
#[test]
fn eos_peak_matches_the_shared_fixture() {
    let fixture = vectors();
    let section = &fixture["eos_peak"];
    let cases = section["cases"]
        .as_array()
        .expect("the fixture has no eos_peak section; nothing was compared");
    assert!(!cases.is_empty(), "eos_peak section is empty");
    let rtol = section["prob_rtol"].as_f64().unwrap();

    for case in cases {
        let cfg = sampler::Config {
            temperature: case["config"]["temperature"].as_f64().unwrap(),
            repetition_penalty: case["config"]["repetition_penalty"].as_f64().unwrap(),
            min_p: case["config"]["min_p"].as_f64().unwrap(),
            max_new_tokens: 255,
            silence_token_ids: case["config"]["silence_token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect(),
            min_tokens_floor: 0,
            min_tokens_text_ratio: 0.0,
        };
        let mut s = Sampler::new(cfg, case["seed"].as_u64().unwrap());
        s.observe_eos(
            case["stop_token"].as_u64().unwrap() as usize,
            case["eos_floor"].as_u64().unwrap() as usize,
        );

        let r = &case["logits_recipe"];
        let vocab = r["vocab"].as_u64().unwrap() as usize;
        let scale = r["scale"].as_f64().unwrap();
        let offset = r["offset"].as_f64().unwrap();
        let mut seen = vec![false; vocab];
        for step in 0..r["steps"].as_u64().unwrap() as usize {
            let u = rng::uniforms(
                r["seed"].as_u64().unwrap(),
                r["stream"].as_u64().unwrap() as u32,
                step,
                1,
                vocab,
            );
            let row: Vec<f32> = u.iter().map(|x| (x * scale + offset) as f32).collect();
            let tok = s.call(&row, step, &seen);
            seen[tok] = true;
        }
        let (at, prob) = s.eos_peak();
        let want_prob = case["expected_prob"].as_f64().unwrap();
        assert_eq!(
            at,
            case["expected_at"].as_i64().unwrap(),
            "{}",
            case["name"]
        );
        assert!(
            (prob - want_prob).abs() <= rtol * want_prob.abs(),
            "{}: peak prob {prob}, want {want_prob}",
            case["name"]
        );
    }
}
