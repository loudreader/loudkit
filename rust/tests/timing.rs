//! Timestamps: exact at the chunk, estimated at the word.
//!
//! The whole value of this feature is that a reading app can trust the first
//! tier and be told, loudly, not to trust the second in the same way. So the
//! tests are split the same way: the chunk assertions are equalities, the word
//! assertions are invariants (monotonic, inside the chunk, every word present)
//! and nothing here claims a word lands where a listener would say it does.
//!
//! Needs no assets. `loudkit::timing` is arithmetic over sample counts, which is
//! the whole of it — the engine parts that fill these in cannot run without the
//! checkpoint, the exported graphs and the onnxruntime shared library, and the
//! same is true in the other four ports.

use loudkit::timing::{estimate_words, timeline, ChunkSpan};

const SAMPLE_RATE: usize = 24_000;

fn span(text: &str, samples: usize, tokens: usize) -> ChunkSpan {
    ChunkSpan {
        text: text.to_string(),
        samples,
        tokens,
    }
}

/// A highlight that switches on `time >= start` flickers on a gap and
/// double-lights on an overlap, and both are invisible to a comparison with a
/// tolerance. Offsets accumulate as integer samples for exactly this reason.
#[test]
fn chunks_are_adjacent_to_the_last_bit() {
    let got = timeline(
        &[span("a b", 7_001, 3), span("c d e", 13_337, 5)],
        SAMPLE_RATE,
    );
    assert_eq!(got[0].start, 0.0);
    assert_eq!(got[1].start, got[0].end);
    assert_eq!(got[1].end, (7_001.0 + 13_337.0) / SAMPLE_RATE as f64);
}

#[test]
fn the_spans_cover_the_whole_render_with_nothing_left_over() {
    let got = timeline(
        &[
            span("one", 100, 1),
            span("two", 200, 2),
            span("three", 300, 3),
        ],
        SAMPLE_RATE,
    );
    let total: f64 = got.iter().map(loudkit::timing::ChunkTiming::duration).sum();
    assert!((total - 600.0 / SAMPLE_RATE as f64).abs() < 1e-12);
    assert_eq!(
        got.iter().map(|c| c.tokens).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}

#[test]
fn an_empty_render_is_an_empty_timeline() {
    assert!(timeline(&[], SAMPLE_RATE).is_empty());
}

/// The text a chunk carries is the text that was tokenised, handed straight
/// through: the engine funnels before it splits, and the timeline reports what
/// was spoken rather than what was typed.
#[test]
fn the_span_text_reaches_the_timing() {
    let got = timeline(&[span("I have three apples.", 4_800, 7)], SAMPLE_RATE);
    assert_eq!(got[0].text, "I have three apples.");
    assert_eq!(got[0].words.len(), 4);
}

/// Every chunk of a stream starts at zero, so a caller who is stitching them
/// has to do the adding. That is the one arithmetic they should not have to
/// write themselves.
#[test]
fn shifting_moves_the_chunk_and_its_words_together() {
    let base = timeline(&[span("alpha beta", 24_000, 4)], SAMPLE_RATE).remove(0);
    let moved = base.shifted(2.5);
    assert_eq!(moved.start, base.start + 2.5);
    assert_eq!(moved.end, base.end + 2.5);
    for (was, now) in base.words.iter().zip(&moved.words) {
        assert_eq!(now.text, was.text);
        assert_eq!(now.start, was.start + 2.5);
        assert_eq!(now.end, was.end + 2.5);
    }
}

#[test]
fn words_tile_the_chunk_without_gaps() {
    let words = estimate_words("alpha beta gamma", 1.0, 4.0);
    assert_eq!(
        words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
        vec!["alpha", "beta", "gamma"]
    );
    assert_eq!(words[0].start, 1.0);
    assert_eq!(words[2].end, 4.0);
    for pair in words.windows(2) {
        assert_eq!(pair[0].end, pair[1].start);
    }
}

#[test]
fn times_are_monotonic_and_inside_the_chunk() {
    let words = estimate_words("a bb ccc dddd e", 2.5, 3.25);
    let mut previous = 2.5;
    for w in &words {
        assert!(2.5 <= w.start && w.start <= w.end && w.end <= 3.25);
        assert!(w.start >= previous);
        previous = w.start;
    }
}

/// The whole content of the estimate: characters stand in for seconds. Nothing
/// else here knows how long a word takes.
#[test]
fn a_longer_word_is_given_longer() {
    let words = estimate_words("hi internationalisation", 0.0, 1.0);
    assert!(words[1].end - words[1].start > words[0].end - words[0].start);
}

/// A caller highlighting `"end."` wants the full stop lit with the word, and a
/// caller matching back against their own text needs the substring to be a
/// substring.
#[test]
fn punctuation_stays_with_its_word() {
    let words = estimate_words("Hello, world!", 0.0, 1.0);
    assert_eq!(
        words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
        vec!["Hello,", "world!"]
    );
}

#[test]
fn no_text_is_no_words_rather_than_a_division_by_zero() {
    assert!(estimate_words("   ", 0.0, 1.0).is_empty());
    assert!(estimate_words("", 0.0, 1.0).is_empty());
}

/// The five ports count code points. A byte count would give Polish and
/// Japanese text different word weights in Rust than in Python, for text that
/// reads identically — `żółć` is four characters and eight bytes.
#[test]
fn length_is_counted_in_characters_not_bytes() {
    let words = estimate_words("aaaa żółć", 0.0, 1.0);
    let ascii = words[0].end - words[0].start;
    let accented = words[1].end - words[1].start;
    assert!((ascii - accented).abs() < 1e-12, "{ascii} vs {accented}");
}
