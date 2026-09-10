//! End-to-end conformance: the ONNX engine vs the shared fixture. Free-run
//! tokens must be exact and fixed-token renders must land inside the fixture's
//! correlation bands, and a long-form passage must produce the fixture's exact
//! token stream in every one of its chunks. Needs the checkpoint, the exported
//! graphs, the reference voice and the onnxruntime shared library
//! (ORT_DYLIB_PATH); skips when any are absent.

use std::path::PathBuf;

use loudkit::engine::{Engine, Options};
use loudkit::sampler::{self, Sampler};
use loudkit::voice;

fn fixture_dir() -> Option<PathBuf> {
    let p = std::env::var("LOUDKIT_FIXTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../tests/data/conformance"));
    p.join("vectors.json").exists().then_some(p)
}

fn need(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}

/// Report a missing prerequisite, and decide whether that is a skip or a fail.
///
/// Cargo has no runtime skip, and `eprintln!` is swallowed by libtest without
/// `--nocapture`, so a run with no assets printed `test engine_conformance ...
/// ok` / `1 passed`, indistinguishable in any CI summary from a real
/// conformance pass. Go prints `--- SKIP:` and JS prints `# SKIP`; this was the
/// one that lied.
///
/// Two changes make it honest. The test is `#[ignore]`, which libtest reports
/// distinctly (`1 ignored`) and which the asset-backed CI job overrides with
/// `--ignored`; and `LOUDKIT_REQUIRE_ASSETS=1` turns the skip into a panic,
/// matching the Python suite's `requires()`.
fn skip(reason: &str) {
    if std::env::var("LOUDKIT_REQUIRE_ASSETS").is_ok_and(|v| !v.is_empty() && v != "0") {
        panic!("LOUDKIT_REQUIRE_ASSETS is set but {reason}");
    }
    eprintln!("SKIPPED (not a pass): {reason}");
}

fn to_f64(v: &serde_json::Value) -> f64 {
    if v.is_null() {
        return 0.0;
    }
    v.as_f64().unwrap_or_else(|| v.as_u64().unwrap() as f64)
}

/// Pearson correlation, on the explicit condition that the two are the same
/// length.
///
/// Correlating `min(a.len(), b.len())` samples makes a truncated render score
/// perfectly against the prefix it did produce. The length *is* the finding in
/// that case, so it is asserted before the correlation rather than quietly
/// discarded by it.
fn corr(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: correlating a prefix would hide a truncated render"
    );
    let n = a.len();
    let ma: f64 = a[..n].iter().map(|x| f64::from(*x)).sum::<f64>() / n as f64;
    let mb: f64 = b[..n].iter().map(|x| f64::from(*x)).sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let x = f64::from(a[i]) - ma;
        let y = f64::from(b[i]) - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da * db).sqrt()
}

/// The RMS ratio of a render to its reference, in dB.
///
/// Correlation subtracts the mean and divides by the deviation, so it reports
/// 1.0 for a render at half volume, at twenty times volume, or with a DC
/// offset. Level is exactly what that normalisation discards, so it is the one
/// amplitude fact worth its own gate.
fn level_db(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let sa: f64 = a.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    let sb: f64 = b.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
    assert!(sa > 0.0, "rendered silence");
    20.0 * (sa / sb).sqrt().log10()
}

/// The loudest sample, against the `[-1, 1]` a waveform is declared to occupy.
/// Everything downstream clips to that range, so a render outside it is audibly
/// wrong and needs no tolerance to say so.
fn peak_of(a: &[f32]) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(f64::from(*v).abs()))
}

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let buf = std::fs::read(path).unwrap();
    buf.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn derive(seed: u64, stream: u64) -> u64 {
    const PHI: u64 = 0x9e3779b97f4a7c15;
    const PSI: u64 = 0xbf58476d1ce4e5b9;
    seed.wrapping_mul(PHI)
        .wrapping_add(stream.wrapping_mul(PSI))
}

// `#[ignore]` by default: this needs the 1.27 GB checkpoint, the exported
// graphs, a voice and an onnxruntime shared library. Run it with
// `cargo test -- --ignored`, which the parity job does with
// LOUDKIT_REQUIRE_ASSETS=1 so a missing asset is a failure rather than a
// silent pass. Without the attribute, an asset-less run reported `ok`.
#[test]
#[ignore = "needs LOUDKIT_CKPT, LOUDKIT_ONNX_DIR, LOUDKIT_VOICE and ORT_DYLIB_PATH"]
fn engine_conformance() {
    run_conformance(false);
}

#[test]
#[ignore = "needs fusion checkpoint, ONNX graphs, voice and ORT_DYLIB_PATH"]
fn fusion_engine_conformance() {
    run_conformance(true);
}

fn run_conformance(fused: bool) {
    let ckpt = need(if fused {
        "LOUDKIT_FUSION_CKPT"
    } else {
        "LOUDKIT_CKPT"
    });
    let onnx = need(if fused {
        "LOUDKIT_FUSION_ONNX_DIR"
    } else {
        "LOUDKIT_ONNX_DIR"
    });
    let vp = need("LOUDKIT_VOICE");
    let lib = need("ORT_DYLIB_PATH");
    let fixture = match fixture_dir() {
        Some(f) => f,
        None => {
            skip("LOUDKIT_FIXTURE_DIR has no vectors.json");
            return;
        }
    };
    let missing: Vec<&str> = [
        ("LOUDKIT_CKPT", &ckpt),
        ("LOUDKIT_ONNX_DIR", &onnx),
        ("LOUDKIT_VOICE", &vp),
        ("ORT_DYLIB_PATH", &lib),
    ]
    .iter()
    .filter(|(_, v)| v.is_empty())
    .map(|(k, _)| *k)
    .collect();
    if !missing.is_empty() {
        skip(&format!("these are unset: {}", missing.join(", ")));
        return;
    }

    let vectors: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture.join(if fused {
            "vectors_fusion_mtp2.json"
        } else {
            "vectors.json"
        }))
        .unwrap(),
    )
    .unwrap();
    let cases = vectors["end_to_end"].as_array().unwrap();

    let mut eng = Engine::load_paths(
        &ckpt,
        &onnx,
        fixture.join("tokenizer.json").to_str().unwrap(),
    )
    .unwrap();
    let v = voice::load(&vp).unwrap();

    for c in cases {
        let name = c["name"].as_str().unwrap();
        let seed = to_f64(&c["seed"]) as u64;
        let sampling = {
            let cfg = eng.config();
            sampler::Config {
                temperature: cfg.sampling.temperature,
                repetition_penalty: cfg.sampling.repetition_penalty,
                min_p: cfg.sampling.min_p,
                max_new_tokens: cfg.sampling.max_new_tokens,
                silence_token_ids: cfg.sampling.silence_token_ids.clone(),
                min_tokens_floor: cfg.sampling.min_tokens_floor,
                min_tokens_text_ratio: cfg.sampling.min_tokens_text_ratio,
            }
        };
        let start_speech = eng.config().start_speech;

        // free-run tokens: exact
        let ids = eng
            .encode(c["text"].as_str().unwrap(), c["language"].as_str().unwrap())
            .unwrap();
        let mut s = Sampler::new(sampling, seed);
        let raw = eng.generate(&ids, &v, &mut s, None, None, &[]).unwrap();
        let stripped: Vec<usize> = raw.iter().copied().filter(|t| *t < start_speech).collect();
        let want: Vec<usize> = c["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| to_f64(x) as usize)
            .collect();
        assert_eq!(stripped, want, "{name} tokens");
        for prefix in [&[][..], &want[..want.len().min(6)]] {
            let mut sampler = Sampler::new(eng.config().sampling.clone(), seed);
            let started = std::time::Instant::now();
            let first = eng
                .generate(&ids, &v, &mut sampler, None, None, prefix)
                .unwrap();
            let elapsed = started.elapsed();
            let mut sampler = Sampler::new(eng.config().sampling.clone(), seed);
            let repeated = eng
                .generate(&ids, &v, &mut sampler, None, None, prefix)
                .unwrap();
            assert_eq!(
                first,
                repeated,
                "{name}: repeat with prefix {}",
                prefix.len()
            );
            eprintln!(
                "{name}: warm prefix={} tokens={} elapsed_ms={:.2}",
                prefix.len(),
                first.len(),
                elapsed.as_secs_f64() * 1000.0
            );
        }

        if fused {
            for cap in [1, 2, 3] {
                let mut sampler = Sampler::new(eng.config().sampling.clone(), seed);
                let capped = eng
                    .generate(&ids, &v, &mut sampler, Some(cap), None, &[])
                    .unwrap();
                assert_eq!(capped, want[..cap], "{name}: cap {cap}");
            }
            let mut sampler = Sampler::new(eng.config().sampling.clone(), seed);
            let even = eng
                .generate(&ids, &v, &mut sampler, Some(8), None, &want[..6])
                .unwrap();
            let mut sampler = Sampler::new(eng.config().sampling.clone(), seed);
            let odd = eng
                .generate(&ids, &v, &mut sampler, Some(8), None, &want[..7])
                .unwrap();
            assert_eq!(even, odd, "{name}: incomplete prefix pair");
        }

        // The two public paths agree with the generator: chunk 0 draws the
        // caller's seed, so a text that fits one window renders the same
        // tokens through synthesize_window and through synthesize.
        let options = Options {
            seed,
            language: Some(c["language"].as_str().unwrap().to_string()),
            ..Default::default()
        };
        let text = c["text"].as_str().unwrap();
        let window = eng.synthesize_window(text, &v, &options).unwrap();
        let whole = eng.synthesize(text, &v, &options).unwrap();
        assert_eq!(window.tokens, want, "{name}: synthesize_window");
        assert_eq!(whole.tokens, want, "{name}: synthesize");

        // fixed-token render: within the band
        let mel = eng.decode_mel(&want, &v, derive(seed, 1)).unwrap();
        let audio = eng.vocode(&mel, derive(seed, 2)).unwrap();
        let mel_ref = read_f32(&fixture.join(c["mel"]["file"].as_str().unwrap()));
        let wav_ref = read_f32(&fixture.join(c["wav"]["file"].as_str().unwrap()));
        let gates = &c["gates"];
        let mel_corr = corr(&mel, &mel_ref);
        let wave_corr = corr(&audio, &wav_ref);
        assert!(
            mel_corr >= to_f64(&gates["mel_corr"]),
            "{name} mel corr {mel_corr} below gate"
        );
        assert!(
            wave_corr >= to_f64(&gates["wave_corr"]),
            "{name} wave corr {wave_corr} below gate"
        );
        let level = level_db(&audio, &wav_ref);
        let band = to_f64(&gates["wave_rms_db"]);
        assert!(
            level.abs() <= band,
            "{name} level {level:+.4} dB against the reference, outside the {band} dB \
             band; correlation cannot see this"
        );
        let peak = peak_of(&audio);
        assert!(
            peak <= 1.0,
            "{name} peak {peak:.4} is outside the declared [-1, 1] waveform"
        );
        eprintln!(
            "{name}: tokens PASS, render mel {mel_corr:.6} wave {wave_corr:.4} \
             level {level:+.4} dB"
        );
    }

    long_form(&mut eng, &v, &vectors);
}

/// The long-form chain: a passage that does not fit one window, chunk by chunk.
///
/// Everything above is a single window with an empty prefix, and with an empty
/// prefix `prefix.len() + step + 1` and `step + 1` are the same number and a
/// repetition mask seeded from the prefix is the empty one. This port wrote
/// both short forms and this fixture passed anyway. A carried prefix is what
/// tells them apart.
///
/// Asserted per chunk rather than on the concatenation: a divergence inside
/// chunk *k* shifts every token after it, so a whole-passage comparison reports
/// one enormous mismatch instead of naming the chunk and the step.
///
/// Runs inside `engine_conformance` rather than as its own `#[test]` because
/// loading six ONNX sessions again to read three more token streams costs more
/// than the isolation is worth, and libtest would run the two in parallel
/// against the same graphs.
fn long_form(eng: &mut Engine, v: &voice::Profile, vectors: &serde_json::Value) {
    let section = &vectors["long_form"];
    if section.is_null() {
        skip("the fixture has no long_form section");
        return;
    }
    let prefix_tokens = section["prefix_tokens"].as_u64().unwrap() as usize;
    assert_eq!(
        eng.config().chunking.prefix_tokens,
        prefix_tokens,
        "this port carries a different number of tokens across a join"
    );

    for case in section["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let language = case["language"].as_str().unwrap();
        // Funnel first, then split: the order the engine uses, and the order
        // the character budget assumes.
        let prepared = loudkit::speechtext::speech_text(case["text"].as_str().unwrap(), language);
        assert_eq!(
            prepared,
            case["prepared"].as_str().unwrap(),
            "{name}: the speech funnel drifted"
        );
        let chunks = case["chunks"].as_array().unwrap();
        assert!(
            chunks.len() > 1,
            "{name} is a single window and proves nothing"
        );
        let want_texts: Vec<&str> = chunks.iter().map(|c| c["text"].as_str().unwrap()).collect();
        let got_texts = loudkit::chunking::split_text(&prepared, &eng.config().chunking);
        assert_eq!(
            got_texts, want_texts,
            "{name}: the split moved, so every chunk below is asking about different text"
        );

        for chunk in chunks {
            let index = chunk["index"].as_u64().unwrap() as usize;
            let want: Vec<usize> = chunk["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| to_f64(x) as usize)
                .collect();
            let prefix: Vec<usize> = chunk["prefix"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| to_f64(x) as usize)
                .collect();
            // The chain the streaming path walks: chunk k is conditioned on the
            // tail of chunk k-1. Spelled out in the fixture so a mismatch names
            // the carry rather than the tokens that followed from it.
            if index > 0 {
                let previous: Vec<usize> = chunks[index - 1]["tokens"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| to_f64(x) as usize)
                    .collect();
                assert_eq!(
                    prefix,
                    loudkit::engine::carry_pair_aligned(
                        &previous,
                        prefix_tokens,
                        eng.config().decode_mode == "fusion_mtp2"
                    ),
                    "{name} chunk {index}: carry"
                );
            }
            // Hex, because a derived 64-bit seed does not survive a JSON double.
            let seed =
                u64::from_str_radix(chunk["seed"].as_str().unwrap().trim_start_matches("0x"), 16)
                    .unwrap();
            let ids = eng
                .encode(chunk["text"].as_str().unwrap(), language)
                .unwrap();
            let mut s = Sampler::new(eng.config().sampling.clone(), seed);
            let raw = eng.generate(&ids, v, &mut s, None, None, &prefix).unwrap();
            let start_speech = eng.config().start_speech;
            let got: Vec<usize> = raw.iter().copied().filter(|t| *t < start_speech).collect();
            assert_eq!(got, want, "{name} chunk {index}");
        }
        // The engine's own long-form path, not the generator driven with the
        // fixture's seeds: this holds the chunk seed law (chunk 0 the caller's
        // seed, chunk k derive(seed, 16 + k)) and the carry to the fixture.
        let whole = eng
            .synthesize(
                case["text"].as_str().unwrap(),
                v,
                &Options {
                    seed: to_f64(&case["seed"]) as u64,
                    language: Some(language.to_string()),
                    ..Default::default()
                },
            )
            .unwrap();
        let public_tokens = if eng.config().decode_mode == "fusion_mtp2" {
            case.get("public_tokens")
                .expect("fusion fixture requires measured public_tokens")
        } else {
            &case["tokens"]
        };
        let flat: Vec<usize> = public_tokens
            .as_array()
            .unwrap()
            .iter()
            .map(|x| to_f64(x) as usize)
            .collect();
        assert_eq!(
            whole.tokens, flat,
            "{name}: synthesize against the fixture's flat list"
        );
        eprintln!("{name}: long-form tokens PASS ({} chunks)", chunks.len());
    }
}
