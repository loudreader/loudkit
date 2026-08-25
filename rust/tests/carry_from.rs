//! Cross-call prosody context: the chunk prefix, exposed across calls.
//!
//! `previous_tokens` on `synthesize`, `stream` and `synthesize_long` does one
//! thing — it seeds the same carry variable the streaming loop already passes
//! between two chunks. A request boundary and a chunk boundary are the same
//! join, so there is deliberately no second conditioning mechanism, and the
//! whole of the new behaviour is the slice asserted here.
//!
//! Tested against the free function rather than through the engine because this
//! port has no weight-free engine seam: `Engine` holds six concrete
//! `ort::session::Session` values and `Engine::load` is its only constructor, so
//! nothing can drive the pipeline without a checkpoint, the exported graphs and
//! the onnxruntime shared library. Python asserts the same rules against a fake
//! generator that records what it was handed; `carry_from` is the honest unit
//! here, and it needs no assets.
//!
//! What this therefore does NOT cover is the wiring: if the helper's result
//! stopped being handed to the generator's prefix, every assertion here would
//! still pass. That half is pinned in Python, by
//! tests/test_engine.py::TestCrossRequestContext, against a fake generator
//! that records the context it was given — building an equivalent seam in four
//! more languages would cost four engine refactors to re-assert one fact.

use loudkit::engine::carry_from;

/// The shipping recipe's values, so the numbers below read like a real call.
const PREFIX_TOKENS: usize = 6;
const START_SPEECH: usize = 6561;

#[test]
fn the_first_chunk_is_conditioned_on_the_tail() {
    let got = carry_from(Some(&[10, 11, 12, 13, 14]), 3, START_SPEECH).unwrap();
    assert_eq!(got, vec![12, 13, 14]);
}

/// Chaining should be `previous_tokens = Some(&earlier_tokens)` with no
/// arithmetic at the call site: a caller who had to know the prefix length
/// would be keeping their own copy of an algorithm value.
#[test]
fn a_long_history_is_sliced_rather_than_refused() {
    let history: Vec<usize> = (0..200).collect();
    let got = carry_from(Some(&history), 2, START_SPEECH).unwrap();
    assert_eq!(got, vec![198, 199]);
}

/// A history shorter than the prefix is taken whole. Nothing is padded: there is
/// no token that means "silence before the utterance began".
#[test]
fn a_short_history_is_taken_whole() {
    let got = carry_from(Some(&[4, 5]), PREFIX_TOKENS, START_SPEECH).unwrap();
    assert_eq!(got, vec![4, 5]);
    assert!(carry_from(Some(&[]), PREFIX_TOKENS, START_SPEECH)
        .unwrap()
        .is_empty());
}

/// The default is byte-for-byte the behaviour from before the parameter
/// existed: an empty carry is what the streaming loop has always started with.
#[test]
fn absent_is_todays_behaviour() {
    assert!(carry_from(None, PREFIX_TOKENS, START_SPEECH)
        .unwrap()
        .is_empty());
}

/// Python's `tokens[-0:]` is the whole list. At the setting that means "chunks
/// are independent" that would condition on the entire previous utterance — the
/// exact opposite — and every port has to refuse the same way even where the
/// slicing bug cannot be spelled.
#[test]
fn zero_prefix_tokens_means_no_context_not_all_of_it() {
    assert!(carry_from(Some(&[1, 2, 3]), 0, START_SPEECH)
        .unwrap()
        .is_empty());
}

/// An id the renderer cannot look up would index off the embedding table. Named
/// at the boundary rather than three stages in.
#[test]
fn a_token_outside_the_codebook_is_refused() {
    let err = carry_from(Some(&[START_SPEECH]), PREFIX_TOKENS, START_SPEECH)
        .expect_err("a control token is not an acoustic token");
    assert!(err.contains("not an acoustic speech token"), "{err}");
    assert!(err.contains(&START_SPEECH.to_string()), "{err}");
}

/// The whole input is checked, not only the slice that will be used: an id out
/// of range means the sequence was built wrong, and reporting that only when it
/// happens to land in the last six tokens would make the failure depend on how
/// long the caller's text happened to be.
#[test]
fn the_head_is_validated_even_though_only_the_tail_is_used() {
    let err = carry_from(Some(&[9_999, 1, 2, 3]), 2, START_SPEECH)
        .expect_err("an out-of-range id anywhere in the history is a refusal");
    assert!(err.contains("9999"), "{err}");
}
