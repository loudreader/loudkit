//! The decode loop's two prefix-dependent numbers: the learned speech
//! positional row a generated token reads, and the repetition mask it starts
//! from.
//!
//! The prefill lays the speech stream out as `cond ‖ text ‖ BOS@0 ‖
//! prefix[i]@(i+1)`, so a prefix of length P owns speech positions 1..P and the
//! first generated token sits at P+1. This port asked for `step + 1`, a row the
//! prefill had just written for a carried token, and never reached P+1 or
//! above — right for a chunk with no prefix and wrong for every other one,
//! which is why single-window synthesis matched Python and long-form did not.
//! Python (`backends/onnx_backend.py:353`) and Swift
//! (`TokenGenerator.swift:586`) have always indexed `len(prefix) + step + 1`.
//!
//! The mask was the same omission: the prefix ids were never marked as spoken,
//! so the repetition penalty treated tokens said moments ago as new.
//!
//! Asserted against `DecodeState` rather than through `Engine::generate`,
//! because this port has no weight-free engine seam — `Engine` holds six
//! concrete `ort::session::Session` values and `Engine::load` is its only
//! constructor, so nothing can drive the decode loop without the checkpoint, the
//! exported graphs and the onnxruntime shared library. JS drives its `generate`
//! against stand-in sessions and pins the wiring there. What is pinned here is
//! the arithmetic and the mask, in the value the loop delegates both to.

use loudkit::engine::DecodeState;
use loudkit::sampler::{self, Sampler};

/// Small enough to write logits out by hand.
const VOCAB: usize = 64;

/// A carried tail. It repeats an id, so the mask has something to have been
/// seeded with, and holds no silence id, so the penalty is not exempt from it.
const PREFIX: [usize; 4] = [3, 5, 3, 7];

fn sampler_config(silence: Vec<usize>) -> sampler::Config {
    sampler::Config {
        // Low enough that the Gumbel draw cannot outrank a whole logit of
        // margin: what is under test is the penalty, not the sampler's noise.
        temperature: 0.1,
        repetition_penalty: 2.0,
        min_p: 0.0,
        max_new_tokens: 8,
        silence_token_ids: silence,
        min_tokens_floor: 0,
        min_tokens_text_ratio: 0.0,
    }
}

/// Token 3 leads token 11 by one logit, and halving it (the penalty, on a
/// positive logit) puts token 11 in front.
fn logits() -> Vec<f32> {
    let mut row = vec![0.0f32; VOCAB];
    row[3] = 4.0;
    row[11] = 3.0;
    row
}

#[test]
fn the_first_generated_token_reads_the_row_after_the_prefix() {
    let state = DecodeState::new(&PREFIX, VOCAB);
    assert_eq!(state.position(0), PREFIX.len() + 1);
    assert_ne!(state.position(0), 1, "that is `step + 1`, the prefix's row");
}

#[test]
fn generation_continues_above_the_rows_the_prefill_wrote() {
    let prefill: Vec<usize> = (0..PREFIX.len())
        .map(DecodeState::prefix_position)
        .collect();
    assert_eq!(prefill, vec![1, 2, 3, 4], "BOS holds row 0");

    let state = DecodeState::new(&PREFIX, VOCAB);
    assert_eq!(state.position(0), prefill[PREFIX.len() - 1] + 1);
    for step in 0..16 {
        assert!(
            !prefill.contains(&state.position(step)),
            "step {step} re-reads a row the prefill wrote"
        );
    }
}

#[test]
fn no_prefix_is_the_single_window_behaviour_it_always_was() {
    let state = DecodeState::new(&[], VOCAB);
    for step in 0..16 {
        assert_eq!(state.position(step), step + 1);
    }
}

#[test]
fn the_repetition_mask_starts_from_the_prefix() {
    let state = DecodeState::new(&PREFIX, VOCAB);
    let seen = state.seen();
    assert_eq!(seen.len(), VOCAB);
    for (id, marked) in seen.iter().enumerate() {
        assert_eq!(*marked, PREFIX.contains(&id), "id {id}");
    }
}

/// The seeding is worth the line only if it changes a draw, so this asserts the
/// draw: the same logits and the same seed pick a different token depending on
/// whether the carried tail was marked as spoken.
#[test]
fn a_carried_token_is_penalised_on_the_first_step() {
    let row = logits();
    let mut penalised = Sampler::new(sampler_config(Vec::new()), 0);
    let state = DecodeState::new(&PREFIX, VOCAB);
    assert_eq!(penalised.call(&row, 0, state.seen()), 11);

    let mut unpenalised = Sampler::new(sampler_config(Vec::new()), 0);
    assert_eq!(unpenalised.call(&row, 0, &[false; VOCAB]), 3);
}

/// Why nothing caught this: the penalty exempts the manifest silence ids, and a
/// tail of silence is what a chunk boundary usually carries. A seeded mask and
/// an empty one draw the same token there.
#[test]
fn a_silent_tail_seeds_the_mask_and_changes_nothing() {
    let row = logits();
    let silence = vec![3, 5, 7];
    let mut sampler = Sampler::new(sampler_config(silence.clone()), 0);
    let state = DecodeState::new(&PREFIX, VOCAB);
    assert_eq!(sampler.call(&row, 0, state.seen()), 3);

    let mut empty = Sampler::new(sampler_config(silence), 0);
    assert_eq!(empty.call(&row, 0, &[false; VOCAB]), 3);
}

#[test]
fn a_generated_token_joins_the_mask() {
    let mut state = DecodeState::new(&PREFIX, VOCAB);
    assert!(!state.seen()[11]);
    state.mark(11);
    assert!(state.seen()[11]);
}
