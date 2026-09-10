//! The shipped 0.1.1 manifest, read to the shipped fingerprint, through the
//! reader rather than around it.
//!
//! Every other fingerprint check in this port builds the configuration by hand
//! and compares the canonical form, which pins the *writer*: it holds even if
//! the reader ignores every key and answers its defaults. This one starts where
//! a user starts, at the manifest a released checkpoint carries, so a reader
//! change that drops a key or reads it differently moves the number here
//! instead of reaching a listener. `go/config/freeze_test.go` is the same test
//! in Go, over the same manifest.

use std::collections::HashMap;

use loudkit::checkpoint::Checkpoint;
use loudkit::engine::EngineConfig;
use loudkit::fingerprint::{canonical_form, fingerprint};

/// `7cd75498ad4e7531` is the algorithm this build implements, recorded in
/// CHANGELOG.md and in the conformance vectors. A published voice records the
/// fingerprint it was rendered under, which is a fact about that artefact and
/// does not move with this one.
const SHIPPED_FINGERPRINT: &str = "7cd75498ad4e7531";

/// The manifest 0.1.1 ships, as `tools/amend_manifest.py` writes it over the
/// values the packed checkpoint already carried.
///
/// Written out rather than assembled from the shipping constants, because those
/// are what the reader is being checked against: a manifest built from them
/// would agree with a reader that ignored every key and answered its defaults,
/// which is the exact failure this pins.
fn shipped_011() -> serde_json::Value {
    serde_json::json!({
        "format": "loudkit-checkpoint",
        "format_version": 1,
        "edge_fade_seconds": 0.02,
        "guidance": "single_path",
        "guidance_rate": 0.0,
        "recipe_version": "loudkit-1",
        "postprocess": {
            "mode": "trim",
            "ceiling_speech_per_text_token": 4.0,
            "ceiling_slack_tokens": 40,
            "trailing_filler_threshold": 0.7,
            "trailing_silence_run_tokens": 12,
            "filler_min_eos_probability": 0.05,
            "filler_max_speech_after_run": 10,
            "desperation_speech_per_text_token": 4.5,
            "desperation_min_text_tokens": 10,
            "ended_tail_silence_run": 6,
            "ended_tail_blip_max": 2,
            "ended_tail_word_max": 10,
            "ended_tail_keep": 5,
            "echo_strong_eos_probability": 0.1,
            "echo_strong_max_tail": 30,
            "echo_strong_min_position_pct": 68,
            "echo_weak_eos_probability": 0.003,
            "echo_weak_max_tail": 16,
            "echo_weak_min_position_pct": 85
        },
        "window": {
            "max_speech_tokens": 255,
            "static_length": 255,
            "pad_token_id": 4254,
            "static_prompt_tokens": 238
        },
        "eos_floor": {"min_tokens_floor": 10, "min_tokens_text_ratio": 1.2},
        "chunking": {
            "enabled": true,
            "max_tokens": 255,
            "prefix_tokens": 6,
            "split_on": [". ", "! ", "? ", "; ", ", "],
            "abbreviations": [
                "A", "B", "Cpn", "D", "Dr", "F", "H", "Hr", "I", "J", "K", "M",
                "Mr", "Mrs", "R", "S", "St", "T", "V", "Vors", "dr", "mrs",
                "prof", "\u{15b}w"
            ],
            "mid_sentence_period": "hold"
        },
        "sample_rate": 24000,
        "token_rate_hz": 25.0,
        "speech_vocab_size": 8194,
        "n_cfm_timesteps": 2,
        "speech_tokens": {"start": 6561, "stop": 6562},
        "sampling_defaults": {
            "temperature": 0.8,
            "repetition_penalty": 1.2,
            "min_p": 0.05,
            "max_new_tokens": 255
        },
        "silence_token_ids": [
            1731, 1821, 1822, 1824, 1975, 2058, 2068, 3190, 3377, 3918, 3927,
            3928, 3930, 4008, 4009, 4011, 4012, 4137, 4146, 4161, 4171, 4173,
            4174, 4218, 4245, 4251, 4252, 4254, 4255, 4260, 4282
        ],
        "silence_render_ids": [4137, 4215, 4218, 4299, 6162, 6324, 6405, 6486],
        "quiet_render_ids": [
            1458, 1461, 1488, 1701, 1704, 1707, 1716, 1731, 1785, 1788, 1869,
            1947, 1950, 1951, 1959, 1978, 2028, 2031, 2040, 2058, 2076, 2112,
            2139, 3645, 3648, 3651, 3704, 3888, 3894, 4188, 5838, 6081, 6183,
            6537
        ]
    })
}

/// A safetensors file carrying nothing but a manifest.
///
/// Small enough to write in a test and complete enough for `Checkpoint::open`
/// to run every check it makes, which is the point: the reader path this pins
/// starts at the bytes on disk, not at a `HashMap` assembled beside it.
fn write_manifest_only(name: &str, manifest: &serde_json::Value) -> String {
    let dir = std::env::temp_dir().join(format!("loudkit-freeze-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let header =
        serde_json::json!({"__metadata__": {"manifest": manifest.to_string()}}).to_string();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path.to_string_lossy().into_owned()
}

fn read_shipped(manifest: &serde_json::Value, name: &str) -> EngineConfig {
    let path = write_manifest_only(name, manifest);
    let ckpt = Checkpoint::open(&path).expect("the shipped manifest must load");
    EngineConfig::from_checkpoint(&ckpt).expect("the shipped manifest must read")
}

/// The canonical form Python writes for this algorithm, from the shared
/// fixture, or nothing when the fixture is not beside this crate.
///
/// The fixture is not shipped in the published crate, and the hash below is
/// asserted whether it is there or not: the fingerprint is a digest of the
/// canonical form, so the constant already pins the bytes. The comparison
/// against the fixture exists to name the field that moved when it does.
fn fixture_canonical_form() -> Option<String> {
    let path = std::env::var("LOUDKIT_FIXTURE")
        .unwrap_or_else(|_| "../tests/data/conformance/vectors.json".to_string());
    let raw = std::fs::read_to_string(path).ok()?;
    let fixture: serde_json::Value = serde_json::from_str(&raw).ok()?;
    Some(fixture["algorithm"]["canonical_form"].as_str()?.to_string())
}

#[test]
fn the_shipped_manifest_still_reads_to_the_shipped_fingerprint() {
    let cfg = read_shipped(&shipped_011(), "shipped.safetensors");

    // The blob first: a mismatch there names the field that drifted, while a
    // mismatch in the hash alone says only that something did. The fixture is
    // written by Python, so this is the byte-for-byte comparison against the
    // reference, not against another copy of this port's own writer.
    if let Some(want) = fixture_canonical_form() {
        assert_eq!(
            canonical_form(&cfg),
            want,
            "the shipped manifest no longer reads to the canonical form Python writes"
        );
    }
    assert_eq!(
        fingerprint(&cfg),
        SHIPPED_FINGERPRINT,
        "the shipped 0.1.1 manifest now reads as a different algorithm\ncanonical form: {}",
        canonical_form(&cfg)
    );

    // Named separately because it is the block whose sub-keys are easiest to
    // drop silently: the window recipe is the whole measured deviation between
    // two backends' renders, and a dropped static_length frames it ragged.
    assert_eq!(cfg.window.max_speech_tokens, 255);
    assert_eq!(cfg.window.static_length, Some(255));
    assert_eq!(cfg.window.static_prompt_tokens, Some(238));
    assert_eq!(cfg.window.pad_token_id, Some(4254));
}

/// The manifest is read, not recognised.
///
/// One key moved, one fingerprint moved: without this, a reader that answered
/// its defaults for everything would pass the test above whenever the defaults
/// happened to be the shipped values.
///
/// The moved value is 300 and not 240, because a static window shorter than
/// `max_speech_tokens` is a window recipe the reference refuses: the point
/// here is a key the reader must *read*, so it has to be one both readers
/// accept.
#[test]
fn a_moved_key_moves_the_fingerprint() {
    let mut manifest = shipped_011();
    manifest["window"]["static_length"] = serde_json::json!(300);
    let cfg = read_shipped(&manifest, "moved.safetensors");
    assert_eq!(cfg.window.static_length, Some(300));
    assert_ne!(
        fingerprint(&cfg),
        SHIPPED_FINGERPRINT,
        "a window recipe the manifest did not declare must not fingerprint as the shipped one"
    );
}

/// A sub-key this port cannot read is refused at the door, not defaulted.
///
/// `"255"` is the case the reference coerces and this port refuses; the rest
/// have no reading anywhere. Read as the default, each of them frames the
/// window ragged under a manifest that asked for a padded one, and the
/// canonical form then records the ragged window as if the manifest had
/// declared it.
#[test]
fn an_unreadable_window_length_is_refused() {
    for bad in [
        serde_json::json!("255"),
        serde_json::json!("abc"),
        serde_json::json!(true),
        serde_json::json!([255]),
        serde_json::json!({"n": 255}),
        serde_json::json!(-1),
    ] {
        let mut manifest = shipped_011();
        manifest["window"]["static_length"] = bad.clone();
        let path = write_manifest_only("bad-window.safetensors", &manifest);
        let err = Checkpoint::open(&path)
            .err()
            .unwrap_or_else(|| panic!("static_length {bad} must be refused"));
        assert!(
            err.contains("static_length"),
            "the refusal must name the key: {err}"
        );
    }
}

/// The manifest's own spellings of "unset" still load.
///
/// `window: null` is the ragged window said out loud, an explicit null length
/// is the same reading one level down, and `max_speech_tokens` has no null
/// reading at all: `_window_from` reads it with a bare `int()`.
#[test]
fn the_null_asymmetry_is_the_references() {
    let mut manifest = shipped_011();
    manifest["window"]["static_length"] = serde_json::Value::Null;
    let cfg = read_shipped(&manifest, "null-length.safetensors");
    assert_eq!(cfg.window.static_length, None);

    let mut manifest = shipped_011();
    manifest["window"] = serde_json::Value::Null;
    let cfg = read_shipped(&manifest, "null-window.safetensors");
    assert_eq!(cfg.window.static_length, None);
    assert_eq!(cfg.window.max_speech_tokens, 255);

    let mut manifest = shipped_011();
    manifest["window"]["max_speech_tokens"] = serde_json::Value::Null;
    let path = write_manifest_only("null-max.safetensors", &manifest);
    let err = Checkpoint::open(&path)
        .err()
        .expect("max_speech_tokens has no null reading");
    assert!(
        err.contains("max_speech_tokens"),
        "the refusal must name the key: {err}"
    );
}

/// Every block whose sub-keys the reader takes numbers from, and the two lists
/// of ids beside them.
///
/// One case each, because the reading is one rule: a value with no reading is
/// refused by the key's name rather than replaced by the constant next to it.
#[test]
fn an_unreadable_value_is_refused_wherever_it_sits() {
    // Each key on its own, so the refusal names it rather than whichever the
    // accumulator reached first.
    for (block, key, bad) in [
        ("sampling_defaults", "temperature", serde_json::json!("0.8")),
        ("eos_floor", "min_tokens_floor", serde_json::json!("10")),
        ("speech_tokens", "start", serde_json::json!("6561")),
        ("chunking", "max_tokens", serde_json::json!("255")),
        ("chunking", "enabled", serde_json::json!(0)),
        ("chunking", "mid_sentence_period", serde_json::json!(5)),
        ("postprocess", "stall_run_tokens", serde_json::json!("30")),
        ("postprocess", "mode", serde_json::json!(5)),
    ] {
        let mut manifest = shipped_011();
        manifest[block][key] = bad.clone();
        let path = write_manifest_only("bad-block.safetensors", &manifest);
        let err = Checkpoint::open(&path)
            .err()
            .unwrap_or_else(|| panic!("{block}.{key} = {bad} must be refused"));
        assert!(err.contains(key), "the refusal must name the key: {err}");
    }

    for (key, bad) in [
        ("sample_rate", serde_json::json!("24000")),
        ("token_rate_hz", serde_json::json!(true)),
        ("n_cfm_timesteps", serde_json::json!("2")),
        ("guidance_rate", serde_json::json!("0.0")),
        ("edge_fade_seconds", serde_json::json!("0.02")),
        ("guidance", serde_json::json!(5)),
        ("silence_token_ids", serde_json::json!(["1731"])),
        ("silence_render_ids", serde_json::json!(["4137"])),
        ("quiet_render_ids", serde_json::json!(["1458"])),
        ("euler_grid", serde_json::json!(["0.5"])),
    ] {
        let mut manifest = shipped_011();
        manifest[key] = bad.clone();
        let path = write_manifest_only("bad-top.safetensors", &manifest);
        let err = Checkpoint::open(&path)
            .err()
            .unwrap_or_else(|| panic!("{key} = {bad} must be refused"));
        assert!(err.contains(key), "the refusal must name the key: {err}");
    }
}

/// A separator set the manifest declares empty is refused, not replaced.
///
/// It used to fall back to the shipping five, which is the splitting recipe of
/// a manifest other than the one being read: the text breaks in five places the
/// manifest named none of. `ChunkConfig::validate` already carried this
/// refusal; the fallback simply ran first.
#[test]
fn an_empty_split_on_is_refused_not_replaced() {
    let mut manifest = shipped_011();
    manifest["chunking"]["split_on"] = serde_json::json!([]);
    let path = write_manifest_only("empty-split.safetensors", &manifest);
    let ckpt = Checkpoint::open(&path).expect("an empty list is a shape the door accepts");
    assert!(
        ckpt.chunking().split_on.is_empty(),
        "the shipping separators must not stand in for the ones the manifest declared"
    );
    let err = EngineConfig::from_checkpoint(&ckpt)
        .err()
        .expect("a splitter with nowhere to break must be refused");
    assert!(err.contains("split_on"), "{err}");
}

/// Absence still means what the reference says it means.
///
/// The accumulator refuses a value it cannot read, and a key that is not there
/// is not such a value: a manifest carrying only its format still reads to the
/// defaults `algorithm_from` gives it.
#[test]
fn an_empty_manifest_still_takes_the_reference_defaults() {
    let manifest = serde_json::json!({
        "format": "loudkit-checkpoint",
        "format_version": 1,
    });
    let cfg = read_shipped(&manifest, "bare.safetensors");
    assert_eq!(cfg.sample_rate, 24_000);
    assert_eq!(cfg.euler_steps, 2);
    assert_eq!(cfg.speech_vocab_size, 8194);
    assert_eq!(cfg.start_speech, 6561);
    assert_eq!(cfg.stop_speech, 6562);
    assert_eq!(cfg.window.static_length, None);
    assert_eq!(cfg.sampling.min_tokens_floor, 0);
    assert!(cfg.sampling.min_tokens_text_ratio.abs() < f64::EPSILON);
    assert!((cfg.edge_fade_seconds - 0.005).abs() < f64::EPSILON);
    assert!(cfg.sampling.silence_token_ids.is_empty());

    // And a zero a manifest declares out loud is kept, not read as absence.
    let mut manifest = shipped_011();
    manifest["sampling_defaults"]["min_p"] = serde_json::json!(0);
    manifest["eos_floor"]["min_tokens_floor"] = serde_json::json!(0);
    let cfg = read_shipped(&manifest, "explicit-zero.safetensors");
    assert!(cfg.sampling.min_p.abs() < f64::EPSILON);
    assert_eq!(cfg.sampling.min_tokens_floor, 0);
}

/// Every value the reference refuses at the manifest door, refused here with
/// the reference's own sentence.
///
/// This port carried none of `AlgorithmConfig.__post_init__`, none of
/// `SamplingConfig.__post_init__` and none of `WindowConfig.__post_init__`, so
/// each of these loaded, reported a fingerprint, and computed something. A
/// temperature of zero divides the logit row by it. `n_cfm_timesteps: 0` runs
/// the flow loop zero times and renders the prior noise as audio.
/// `token_rate_hz: 0` divides every duration by zero. A start token past the
/// embedding table indexes off the end of it. A static window shorter than the
/// token budget failed at frame time instead, mid-stream, after earlier chunks
/// had played.
#[test]
fn the_manifest_door_refuses_what_the_reference_refuses() {
    // (path into the manifest, value, the reference's message)
    let cases: Vec<(&[&str], serde_json::Value, &str)> = vec![
        (
            &["sampling_defaults", "temperature"],
            serde_json::json!(0),
            "temperature out of range: 0.0",
        ),
        (
            &["sampling_defaults", "temperature"],
            serde_json::json!(5),
            "temperature out of range: 5.0",
        ),
        (
            &["sampling_defaults", "repetition_penalty"],
            serde_json::json!(0.5),
            "repetition_penalty below 1.0 rewards repetition: 0.5",
        ),
        (
            &["sampling_defaults", "min_p"],
            serde_json::json!(1.5),
            "min_p out of range: 1.5",
        ),
        (
            &["sampling_defaults", "max_new_tokens"],
            serde_json::json!(0),
            "max_new_tokens must be positive: 0",
        ),
        (
            &["eos_floor", "min_tokens_text_ratio"],
            serde_json::json!(-1),
            "min_tokens_text_ratio must be >= 0: -1.0",
        ),
        (
            &["n_cfm_timesteps"],
            serde_json::json!(0),
            "euler_steps must be >= 1: 0",
        ),
        (
            &["token_rate_hz"],
            serde_json::json!(0),
            "token_rate_hz must be > 0: 0.0",
        ),
        (
            &["speech_vocab_size"],
            serde_json::json!(0),
            "speech_vocab_size must be >= 1: 0",
        ),
        (
            &["sample_rate"],
            serde_json::json!(0),
            "sample_rate must be > 0: 0",
        ),
        (
            &["speech_tokens", "start"],
            serde_json::json!(99999),
            "start_speech_token must be in [0, 8194): 99999",
        ),
        (
            &["speech_tokens", "start"],
            serde_json::json!(6562),
            "start_speech_token and stop_speech_token must differ: both are 6562",
        ),
        (
            &["edge_fade_seconds"],
            serde_json::json!(0),
            "edge_fade_seconds must be in [0.001, 0.05]: 0.0",
        ),
        (
            &["edge_fade_seconds"],
            serde_json::json!(10),
            "edge_fade_seconds must be in [0.001, 0.05]: 10.0",
        ),
        (
            &["euler_grid"],
            serde_json::json!([0.0, 1.0]),
            "euler_grid has 2 points, expected 3",
        ),
        (
            &["euler_grid"],
            serde_json::json!([1.0, 0.5, 0.0]),
            "euler_grid must be strictly increasing",
        ),
        (
            &["euler_grid"],
            serde_json::json!([0.1, 0.5, 1.0]),
            "euler_grid must run from 0.0 to 1.0",
        ),
        (
            &["guidance_rate"],
            serde_json::json!(0.5),
            "guidance_rate must be 0.0 in single_path mode",
        ),
        (
            &["window", "static_length"],
            serde_json::json!(254),
            "static_length 254 cannot be shorter than max_speech_tokens 255",
        ),
        (
            &["window", "static_prompt_tokens"],
            serde_json::json!(0),
            "static_prompt_tokens must be positive: 0",
        ),
        (
            &["chunking", "max_tokens"],
            serde_json::json!(300),
            "chunking.max_tokens 300 exceeds the render window (255): every chunk would be \
             sized past what the renderer accepts, and the refusal would land mid-stream, \
             after audio had already been delivered",
        ),
        (
            &["sampling_defaults", "max_new_tokens"],
            serde_json::json!(300),
            "sampling.max_new_tokens 300 exceeds the render window (255): generation is \
             allowed to produce more speech than the renderer will accept, so a long \
             utterance fails after it has been generated rather than before",
        ),
        // The detectors' own validator was already here; what moved is how it
        // spells a float back, which is the reference's `repr` and not Rust's
        // shortest form. "0" for a value written 0.0 is a different sentence
        // in a place where the two ports are supposed to say the same thing.
        (
            &["postprocess", "ceiling_speech_per_text_token"],
            serde_json::json!(0),
            "ceiling_speech_per_text_token must be positive: 0.0",
        ),
        (
            &["postprocess", "trailing_filler_threshold"],
            serde_json::json!(2),
            "trailing_filler_threshold must be in (0, 1]: 2.0",
        ),
        (
            &["postprocess", "filler_min_eos_probability"],
            serde_json::json!(2),
            "filler_min_eos_probability out of range: 2.0",
        ),
    ];

    for (n, (path, value, want)) in cases.iter().enumerate() {
        let mut manifest = shipped_011();
        let mut node = &mut manifest;
        for key in &path[..path.len() - 1] {
            node = &mut node[*key];
        }
        node[path[path.len() - 1]] = value.clone();
        let file = write_manifest_only(&format!("refused-{n}.safetensors"), &manifest);
        // Either door may be the one that answers: the detector block is read
        // and checked inside `Checkpoint::open`, the rest at
        // `from_checkpoint`. The sentence is what is being pinned, and
        // `Checkpoint::open` prefixes its own with the file it was reading.
        let err = match Checkpoint::open(&file) {
            Err(e) => e.replace(&format!("{file}: "), ""),
            Ok(ckpt) => match EngineConfig::from_checkpoint(&ckpt) {
                Ok(_) => panic!("{path:?}={value} must be refused, it loaded"),
                Err(e) => e,
            },
        };
        assert_eq!(&err, want, "{path:?}={value}");
    }
}

/// The values on the other side of each boundary still load.
///
/// Without this, a reader could buy agreement with the reference by refusing
/// everything, and every assertion above would still pass.
#[test]
fn the_manifest_door_keeps_taking_what_the_reference_takes() {
    let cases: Vec<(&[&str], serde_json::Value)> = vec![
        (
            &["sampling_defaults", "temperature"],
            serde_json::json!(4.0),
        ),
        (
            &["sampling_defaults", "repetition_penalty"],
            serde_json::json!(1.0),
        ),
        (&["sampling_defaults", "min_p"], serde_json::json!(0.999)),
        (
            &["sampling_defaults", "max_new_tokens"],
            serde_json::json!(255),
        ),
        (
            &["eos_floor", "min_tokens_text_ratio"],
            serde_json::json!(0),
        ),
        (&["n_cfm_timesteps"], serde_json::json!(1)),
        (&["token_rate_hz"], serde_json::json!(0.001)),
        (&["speech_vocab_size"], serde_json::json!(8194)),
        (&["speech_tokens", "start"], serde_json::json!(8193)),
        (&["edge_fade_seconds"], serde_json::json!(0.001)),
        (&["edge_fade_seconds"], serde_json::json!(0.05)),
        (&["euler_grid"], serde_json::json!([0.0, 0.5, 1.0])),
        (&["window", "static_length"], serde_json::json!(300)),
        (&["window", "static_length"], serde_json::json!(null)),
        (&["window", "static_prompt_tokens"], serde_json::json!(1)),
        (&["chunking", "max_tokens"], serde_json::json!(255)),
    ];

    for (n, (path, value)) in cases.iter().enumerate() {
        let mut manifest = shipped_011();
        let mut node = &mut manifest;
        for key in &path[..path.len() - 1] {
            node = &mut node[*key];
        }
        node[path[path.len() - 1]] = value.clone();
        let file = write_manifest_only(&format!("allowed-{n}.safetensors"), &manifest);
        let ckpt = Checkpoint::open(&file).expect("the container is well formed");
        if let Err(e) = EngineConfig::from_checkpoint(&ckpt) {
            panic!("{path:?}={value} must load: {e}");
        }
    }
}

/// The reader is reachable from the manifest a `Checkpoint` carries, and the
/// two agree.
///
/// `Checkpoint::manifest` is public, so a caller can read a key itself; this
/// pins that the reader above it sees the same map.
#[test]
fn the_checkpoint_carries_the_manifest_it_read() {
    let path = write_manifest_only("carried.safetensors", &shipped_011());
    let ckpt = Checkpoint::open(&path).unwrap();
    let manifest: &HashMap<String, serde_json::Value> = &ckpt.manifest;
    assert_eq!(manifest["window"]["static_length"], serde_json::json!(255));
    assert_eq!(ckpt.window().static_length, Some(255));
}
