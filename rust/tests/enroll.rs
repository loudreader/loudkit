//! Enrollment conformance: the enrollment pipeline vs the shared fixture.
//! The same reference clip must yield the fixture's prompt tokens exactly and
//! its embeddings to cosine > 0.9999. Needs the exported enrollment graphs and
//! the onnxruntime shared library (ORT_DYLIB_PATH); skips when absent, with a
//! named reason.

use std::path::PathBuf;

use loudkit::enroll::Enroller;

fn fixture_dir() -> Option<PathBuf> {
    let p = std::env::var("LOUDKIT_ENROLL_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../tests/data/enrollment"));
    p.join("ref_audio.f32").exists().then_some(p)
}

fn need(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}

fn skip(reason: &str) {
    if std::env::var("LOUDKIT_REQUIRE_ASSETS").is_ok_and(|v| !v.is_empty() && v != "0") {
        panic!("LOUDKIT_REQUIRE_ASSETS is set but {reason}");
    }
    eprintln!("SKIPPED (not a pass): {reason}");
}

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let buf = std::fs::read(path).unwrap();
    buf.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn read_i64(path: &PathBuf) -> Vec<i64> {
    let buf = std::fs::read(path).unwrap();
    buf.as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        dot += f64::from(a[i]) * f64::from(b[i]);
        na += f64::from(a[i]) * f64::from(a[i]);
        nb += f64::from(b[i]) * f64::from(b[i]);
    }
    dot / (na * nb).sqrt()
}

fn enroll() -> (loudkit::enroll::Enrolled, PathBuf) {
    let onnx = need("LOUDKIT_ONNX_DIR");
    let lib = need("ORT_DYLIB_PATH");
    let fixture = match fixture_dir() {
        Some(f) => f,
        None => {
            skip("LOUDKIT_ENROLL_FIXTURE has no ref_audio.f32");
            panic!("unreachable");
        }
    };
    let missing: Vec<&str> = [("LOUDKIT_ONNX_DIR", &onnx), ("ORT_DYLIB_PATH", &lib)]
        .iter()
        .filter(|(_, v)| v.is_empty())
        .map(|(k, _)| *k)
        .collect();
    if !missing.is_empty() {
        skip(&format!("these are unset: {}", missing.join(", ")));
        panic!("unreachable");
    }

    let mut enr = Enroller::load(&PathBuf::from(&onnx)).unwrap();
    let audio = read_f32(&fixture.join("ref_audio.f32"));
    let res = enr.enroll(&audio, 24_000).unwrap();
    (res, fixture)
}

#[test]
#[ignore = "needs LOUDKIT_ONNX_DIR and ORT_DYLIB_PATH"]
fn prompt_tokens_exact() {
    let (res, fixture) = enroll();
    let want = read_i64(&fixture.join("prompt_tokens.i64"));
    assert_eq!(res.prompt_tokens, want, "prompt tokens must match exactly");
}

#[test]
#[ignore = "needs LOUDKIT_ONNX_DIR and ORT_DYLIB_PATH"]
fn cond_tokens_exact() {
    let (res, fixture) = enroll();
    let want = read_i64(&fixture.join("cond_prompt_tokens.i64"));
    assert_eq!(
        res.cond_prompt_tokens, want,
        "cond tokens must match exactly"
    );
}

#[test]
#[ignore = "needs LOUDKIT_ONNX_DIR and ORT_DYLIB_PATH"]
fn embeddings_match() {
    let (res, fixture) = enroll();
    let flow = read_f32(&fixture.join("flow_embedding.f32"));
    let speaker = read_f32(&fixture.join("speaker_embedding.f32"));
    let cf = cos(&res.flow_embedding, &flow);
    let cs = cos(&res.speaker_embedding, &speaker);
    assert!(cf > 0.9999, "flow embedding cosine {cf} <= 0.9999");
    assert!(cs > 0.9999, "speaker embedding cosine {cs} <= 0.9999");
}

#[test]
#[ignore = "needs both synthesis graph sets, enrollment graphs and ORT_DYLIB_PATH"]
fn clone_once_runs_on_both_models() {
    let (enrolled, _) = enroll();
    let profile = enrolled.profile("shared", "en");
    let fixture = PathBuf::from("../tests/data/conformance");
    for (checkpoint, graphs) in [
        ("LOUDKIT_CKPT", "LOUDKIT_ONNX_DIR"),
        ("LOUDKIT_FUSION_CKPT", "LOUDKIT_FUSION_ONNX_DIR"),
    ] {
        let mut engine = loudkit::engine::Engine::load_paths(
            &need(checkpoint),
            &need(graphs),
            fixture.join("tokenizer.json").to_str().unwrap(),
        )
        .unwrap();
        let options = loudkit::engine::Options {
            seed: 0,
            ..Default::default()
        };
        let first = engine
            .synthesize_window("Hello, how are you?", &profile, &options)
            .unwrap();
        let repeated = engine
            .synthesize_window("Hello, how are you?", &profile, &options)
            .unwrap();
        assert!(!first.tokens.is_empty());
        assert_eq!(first.tokens, repeated.tokens, "{checkpoint} repeat tokens");
        assert_eq!(first.audio, repeated.audio, "{checkpoint} repeat audio");
        assert!(first.audio.iter().all(|v| v.is_finite()));
    }
}
