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
    buf.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i64(path: &PathBuf) -> Vec<i64> {
    let buf = std::fs::read(path).unwrap();
    buf.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
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
