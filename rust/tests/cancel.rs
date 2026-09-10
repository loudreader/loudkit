//! The cancellation contract every port shares: cancelled at a decode step
//! inside the second chunk, `synthesize` returns `CANCELLED` and no
//! `Synthesis`, and `stream` delivers the first chunk exactly as an
//! uncancelled run does and nothing after, with `Ok(())`. Needs the same
//! assets as `engine_conformance`; `#[ignore]` without them.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use loudkit::engine::{Engine, Options};
use loudkit::error::{is_cancelled, CANCELLED};
use loudkit::voice;

fn need(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}

fn skip(reason: &str) {
    if std::env::var("LOUDKIT_REQUIRE_ASSETS").is_ok_and(|v| !v.is_empty() && v != "0") {
        panic!("LOUDKIT_REQUIRE_ASSETS is set but {reason}");
    }
    eprintln!("SKIPPED (not a pass): {reason}");
}

/// `Options` whose `should_cancel` counts its polls and fires past `at`
/// (`usize::MAX` never fires), plus the counter.
fn cancelling(seed: u64, at: usize) -> (Options, Arc<AtomicUsize>) {
    let polls = Arc::new(AtomicUsize::new(0));
    let seen = polls.clone();
    let options = Options {
        seed,
        language: Some("en".to_string()),
        should_cancel: Some(Arc::new(move || {
            seen.fetch_add(1, Ordering::SeqCst) + 1 > at
        })),
        ..Default::default()
    };
    (options, polls)
}

#[test]
#[ignore = "needs LOUDKIT_CKPT, LOUDKIT_ONNX_DIR, LOUDKIT_VOICE and ORT_DYLIB_PATH"]
fn cancel_at_a_decode_step() {
    check_cancel(false);
}

#[test]
#[ignore = "needs fusion assets and ORT_DYLIB_PATH"]
fn fusion_cancel_at_a_decode_step() {
    check_cancel(true);
}

fn check_cancel(fused: bool) {
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
    let fixture = std::env::var("LOUDKIT_FIXTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../tests/data/conformance"));
    if [&ckpt, &onnx, &vp, &lib].iter().any(|v| v.is_empty()) {
        skip("LOUDKIT_CKPT, LOUDKIT_ONNX_DIR, LOUDKIT_VOICE or ORT_DYLIB_PATH is unset");
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
    let case = &vectors["long_form"]["cases"][0];
    let text = case["text"].as_str().unwrap();
    let seed = case["seed"].as_u64().unwrap();

    let mut eng = Engine::load_paths(
        &ckpt,
        &onnx,
        fixture.join("tokenizer.json").to_str().unwrap(),
    )
    .unwrap();
    let v = voice::load(&vp).unwrap();

    // Uncancelled first, counting the polls: the step to cancel at has to
    // land inside the second chunk's decode loop.
    let (options, polls) = cancelling(seed, usize::MAX);
    let mut polls_at_chunk: Vec<usize> = Vec::new();
    let mut first: Vec<usize> = Vec::new();
    eng.stream(text, &v, &options, None, &mut |chunk| {
        polls_at_chunk.push(polls.load(Ordering::SeqCst));
        if first.is_empty() {
            first = chunk.tokens.to_vec();
        }
        true
    })
    .unwrap();
    assert!(polls_at_chunk.len() >= 2, "the passage must split");
    let step = polls_at_chunk[0] + 5;
    assert!(
        step < polls_at_chunk[1],
        "step {step} is not inside chunk 1"
    );

    // stream: the first chunk as it was, nothing after, Ok.
    let (options, _) = cancelling(seed, step);
    let mut got: Vec<Vec<usize>> = Vec::new();
    eng.stream(text, &v, &options, None, &mut |chunk| {
        got.push(chunk.tokens.to_vec());
        true
    })
    .expect("a cancel ends the stream without an error");
    assert_eq!(got.len(), 1, "only the chunk before the cancel arrives");
    assert_eq!(
        got[0], first,
        "the chunk before the cancel is what an uncancelled run delivered"
    );

    // synthesize: no Synthesis, CANCELLED.
    let (options, _) = cancelling(seed, step);
    match eng.synthesize(text, &v, &options) {
        Err(e) => assert!(
            is_cancelled(&e),
            "synthesize failed with {e:?}, want {CANCELLED:?}"
        ),
        Ok(out) => panic!(
            "synthesize handed back {} tokens after a cancel",
            out.tokens.len()
        ),
    }

    // synthesize_window, three steps in: the same signal.
    let (options, _) = cancelling(seed, 3);
    let err = eng
        .synthesize_window("Hello from loudkit.", &v, &options)
        .err()
        .expect("synthesize_window returned a Synthesis after a cancel");
    assert!(is_cancelled(&err), "{err:?}");
}
