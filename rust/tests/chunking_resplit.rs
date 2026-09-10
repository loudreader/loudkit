//! The repair for a chunk the window could not hold.
//!
//! The Python reference's `TestSplitInHalf`, case for case.

use loudkit::chunking::{split_in_half, ChunkConfig, OFF_CAP_RESPLIT, WORD_CAP_RESPLIT};

#[test]
fn it_halves_at_a_word_boundary() {
    let (a, b) = split_in_half("one two three four five six").unwrap();
    assert_eq!(format!("{a} {b}"), "one two three four five six");
}

#[test]
fn it_does_not_seek_punctuation() {
    // A weaker separator sits nearer the middle than the comma does, and the
    // comma has no pull of its own: of 27 re-splits taken at the separator
    // nearest the middle, every seam over a second fell on a comma.
    let (a, b) = split_in_half("aa bb, cc dddddddddddd ee").unwrap();
    assert_eq!(a, "aa bb, cc");
    assert_eq!(b, "dddddddddddd ee");
}

/// `split_text`'s last resort, when a window holds no punctuation. The
/// boundary table was introduced for `split_in_half` and this fallback, ten
/// lines away, was left matching U+0020 alone -- so text whose every space is
/// non-breaking was cut mid-word.
#[test]
fn the_word_boundary_fallback_breaks_on_a_non_breaking_space() {
    let cfg = ChunkConfig::default();
    let words: Vec<String> = (0..30).map(|i| format!("ord{i:02}")).collect();
    let chunks = loudkit::chunking::split_text(&words.join("\u{00a0}"), &cfg);
    assert!(chunks.len() > 1, "the case needs to cross a window");
    for c in &chunks {
        let last = c.rsplit('\u{00a0}').next().unwrap();
        assert!(words.iter().any(|w| w == last), "cut mid-word: {last:?}");
    }
}

#[test]
fn it_counts_scalars_not_grapheme_clusters() {
    // CRLF is one grapheme cluster and two Unicode scalars; Swift's
    // Array(text) yields clusters and halved this one word later until it was
    // switched to unicodeScalars. The funnel passes CRLF through verbatim.
    let (a, b) = split_in_half("xx\r\nxx xx xxxxx").unwrap();
    assert_eq!(a, "xx\r\nxx");
    assert_eq!(b, "xx xxxxx");
}

#[test]
fn it_cuts_on_boundaries_the_funnel_keeps() {
    // NBSP and tab survive the funnel and are word boundaries. Matching only
    // U+0020 made a capped chunk whose separators were all non-breaking come
    // back unsplittable, so it shipped its truncation.
    for sep in ["\u{00a0}", "\t", "\u{202f}"] {
        let (a, b) = split_in_half(&format!("alpha{sep}beta{sep}gamma")).unwrap();
        assert_eq!(a, format!("alpha{sep}beta"));
        assert_eq!(b, "gamma");
    }
}

#[test]
fn a_single_unbroken_run_is_refused() {
    // Splitting it would have to cut a word, which is worse than the
    // truncation it would be repairing.
    assert!(split_in_half("omringden.").is_none());
    assert!(split_in_half("").is_none());
    // Trimming can empty a half the scan thought was interior. Unreachable
    // through the engine, which trims first, but this is public.
    assert!(split_in_half("x \t").is_none());
}

#[test]
fn neither_half_is_empty() {
    for text in ["a bb", "aaaaaaaa b", "a bbbbbbbb"] {
        let (a, b) = split_in_half(text).unwrap_or_else(|| panic!("no split for {text:?}"));
        assert!(!a.is_empty() && !b.is_empty(), "empty half for {text:?}");
    }
}

#[test]
fn the_default_is_the_law_and_unknown_spellings_are_refused() {
    let mut cfg = ChunkConfig::default();
    assert_eq!(cfg.cap_resplit, WORD_CAP_RESPLIT);
    cfg.cap_resplit = OFF_CAP_RESPLIT.to_string();
    assert!(cfg.validate().is_ok());
    cfg.cap_resplit = "halve".to_string();
    let err = cfg.validate().unwrap_err();
    assert!(err.contains("cap_resplit"), "{err}");
}

fn derive_seed(seed: u64, stream: u64) -> u64 {
    seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(stream.wrapping_mul(0xBF58_476D_1CE4_E5B9))
}

/// Holds this port to the `cap_resplit` law without weights.
///
/// The token streams need a model; the *plan* does not. Which window carries
/// which index, where the split falls and which seed each window draws are all
/// computable from the fixture alone, and they are exactly the three things
/// that went wrong while this law was written: a second half seeded from the
/// next chunk's stream, a Swift split counting grapheme clusters, and a queue
/// that did not advance. A port that gets any of them wrong produces different
/// audio while every other test still passes.
///
/// Whether a chunk overran needs the model, so that one bit is read from the
/// fixture; everything the port decides given it is recomputed and compared.
#[test]
fn the_resplit_plan_matches_the_shared_fixture() {
    let path = std::env::var("LOUDKIT_FIXTURE")
        .unwrap_or_else(|_| "../tests/data/conformance/vectors.json".to_string());
    // Not a silent `return`: a suite that reports a pass because it could not
    // find its input is the failure this whole file exists to prevent.
    // Verified with LOUDKIT_FIXTURE=/nonexistent: every other fixture-backed
    // test in this crate failed loudly and this one reported a tick.
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture unreadable at {path}: {e}; nothing was compared"));
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let section = v
        .get("resplit")
        .expect("the fixture has no resplit section; nothing was compared");
    let resplit_stream = section["resplit_stream"].as_u64().unwrap();
    let chunk_base = section["chunk_stream_base"].as_u64().unwrap();
    let prefix_tokens = section["prefix_tokens"].as_u64().unwrap() as usize;

    for case in section["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let seed = case["seed"].as_u64().unwrap();
        // The case moves the window, so the chunk budget moves with it: the
        // config refuses a budget larger than the window, and the engine gates
        // the re-split on the window rather than on any cap.
        let window = case["window"].as_u64().unwrap() as usize;
        let cfg = ChunkConfig {
            max_tokens: window,
            ..Default::default()
        };
        assert_eq!(
            cfg.cap_resplit, WORD_CAP_RESPLIT,
            "the fixture pins the law"
        );
        let prepared = case["prepared"].as_str().unwrap();
        let windows = case["windows"].as_array().unwrap();
        let texts = loudkit::chunking::split_text(prepared, &cfg);

        let mut wi = 0usize;
        let mut carry: Vec<i64> = Vec::new();
        for (index, chunk_text) in texts.iter().enumerate() {
            // Chunk 0 draws the caller's seed itself; the base applies from
            // chunk 1 up.
            let chunk_seed = if index == 0 {
                seed
            } else {
                derive_seed(seed, chunk_base + index as u64)
            };
            let was_split = windows[wi]["split"].as_bool().unwrap();
            // `at` rather than the loop counter, so the closure borrows nothing
            // the loop then assigns.
            let check =
                |at: usize, w: &serde_json::Value, text: &str, s: u64, pre: &[i64], split: bool| {
                    assert_eq!(
                        w["index"].as_u64().unwrap(),
                        index as u64,
                        "{name} window {at}: a moved index moves every later chunk's seed"
                    );
                    assert_eq!(
                        w["text"].as_str().unwrap(),
                        text,
                        "{name} window {at}: text"
                    );
                    assert_eq!(
                        w["split"].as_bool().unwrap(),
                        split,
                        "{name} window {at}: split"
                    );
                    assert_eq!(
                        w["seed"].as_str().unwrap(),
                        format!("0x{s:x}"),
                        "{name} window {at}: the second half draws from its own stream"
                    );
                    let want: Vec<i64> = w["prefix"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_i64().unwrap())
                        .collect();
                    assert_eq!(want, pre, "{name} window {at}: carry");
                };
            if was_split {
                let (first, second) = split_in_half(chunk_text).unwrap_or_else(|| {
                    panic!("{name} chunk {index}: fixture split it, this port cannot")
                });
                check(wi, &windows[wi], &first, chunk_seed, &carry, true);
                let a: Vec<i64> = windows[wi]["tokens"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect();
                let pre: Vec<i64> = a[a.len().saturating_sub(prefix_tokens)..].to_vec();
                wi += 1;
                check(
                    wi,
                    &windows[wi],
                    &second,
                    derive_seed(chunk_seed, resplit_stream),
                    &pre,
                    true,
                );
                let b: Vec<i64> = windows[wi]["tokens"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect();
                carry = b[b.len().saturating_sub(prefix_tokens)..].to_vec();
                wi += 1;
            } else {
                check(wi, &windows[wi], chunk_text, chunk_seed, &carry, false);
                let t: Vec<i64> = windows[wi]["tokens"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect();
                carry = t[t.len().saturating_sub(prefix_tokens)..].to_vec();
                wi += 1;
            }
        }
        assert_eq!(wi, windows.len(), "{name}: window count");
    }
}
