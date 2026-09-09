//! The algorithm fingerprint: one string that says whether two engines agree.
//!
//! Every other cross-language check in this project compares a behaviour
//! somebody thought to compare: the funnel because there are 30 fixture cases
//! for it, the splitter because there are 48. The fingerprint compares the
//! *whole* algorithm configuration in one comparison, so a field nobody wrote a
//! test for still cannot drift silently.
//!
//! The failure mode is concrete: an `euler_grid` parsed by one port and
//! ignored by another; a `silence_token_ids` that accepts a string and
//! iterates its characters; a `chunking.prefix_tokens` read from the manifest
//! by some ports and guessed by others, each invisible to behaviour
//! comparison alone. A fingerprint check finds all of them at once, and finds the next one
//! for free.
//!
//! The canonical form is specified rather than incidental: see
//! `AlgorithmConfig.canonical_form` in `loudkit/config.py`. Three rules make it
//! portable across languages:
//!
//! * **floats are their shortest round-tripping decimal, as a JSON string.**
//!   Python emits `repr(float)`; `"0.8"`, not `0.8`. Quoted, so no JSON parser
//!   anywhere gets to re-render the number with its own idea of precision.
//! * **keys are sorted**, at every level.
//! * **only schema-known fields are hashed**, with an explicit schema version,
//!   so adding a field with a default does not re-fingerprint an algorithm that
//!   did not change.

use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use crate::engine::EngineConfig;

/// Bumped only when the *set* of hashed fields changes, never when a value does.
pub const FINGERPRINT_SCHEMA: u32 = 1;

/// Python's `repr(float)`: the shortest decimal that round-trips, with a `.0`
/// on anything integral.
///
/// Rust's `{}` for `f64` already prints the shortest round-tripping form, but
/// renders `25.0` as `25` where Python renders it `25.0`. That one character is
/// the difference between a matching fingerprint and a mysterious one.
///
/// Two ranges are not Python's yet: `{}` never uses exponent form, so a
/// magnitude below 1e-4 prints as `0.00005` where `repr` gives `5e-05`, and one
/// at or above 1e16 prints in full where `repr` gives `1e+16`. Both are out of
/// reach of every value any shipped manifest carries; the canonical form moves
/// for them, so closing it is one commit across all five ports rather than a
/// change here.
pub(crate) fn repr_float(value: f64) -> String {
    if value.is_finite() && value == value.trunc() && value.abs() < 1e16 {
        format!("{value:.1}")
    } else {
        format!("{value}")
    }
}

/// A JSON string literal, escaped the way `json.dumps` escapes.
///
/// One case is not Python's yet, for the same reason as the two ranges on
/// [`repr_float`]: backspace and form feed go out as `\u0008` and `\u000c`
/// where `json.dumps` writes the short `\b` and `\f`. No hashed string carries
/// either character, so the canonical form is unaffected today; closing it
/// moves that form and is one commit across all five ports.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            // Python's `json.dumps` defaults to `ensure_ascii=True`, so every
            // non-ASCII character in the canonical form is a \uXXXX escape and
            // an astral one is a surrogate pair. No hashed string had a
            // non-ASCII character in it until `chunking.abbreviations` carried
            // "\u015bw", and the raw UTF-8 spelling hashed differently from
            // the reference in every port at once.
            c if (c as u32) < 0x20 || (c as u32) >= 0x7f => {
                let v = c as u32;
                if v > 0xffff {
                    let v = v - 0x10000;
                    let _ = write!(
                        out,
                        "\\u{:04x}\\u{:04x}",
                        0xd800 + (v >> 10),
                        0xdc00 + (v & 0x3ff)
                    );
                } else {
                    let _ = write!(out, "\\u{v:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_num_str(value: f64) -> String {
    json_str(&repr_float(value))
}

fn json_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |v| v.to_string())
}

fn json_list<T: ToString>(items: &[T]) -> String {
    let joined: Vec<String> = items.iter().map(ToString::to_string).collect();
    format!("[{}]", joined.join(","))
}

/// The exact string that gets hashed.
///
/// Built by hand rather than through a serialiser: the byte-for-byte output is
/// the contract, and a serialiser is free to change how it renders a float or
/// orders a map between releases.
#[must_use]
pub fn canonical_form(cfg: &EngineConfig) -> String {
    let split_on: Vec<String> = cfg.chunking.split_on.iter().map(|s| json_str(s)).collect();
    // Keys sorted, as everywhere in this form: "abbreviations" before
    // "cap_resplit" before "enabled", and "mid_sentence_period" between
    // "max_tokens" and "prefix_tokens". Five canonical forms are hand-written and a new field's
    // position in that order is part of the contract.
    let abbreviations: Vec<String> = cfg
        .chunking
        .abbreviations
        .iter()
        .map(|s| json_str(s))
        .collect();
    let chunking = format!(
        "{{\"abbreviations\":[{}],\"cap_resplit\":{},\"enabled\":{},\"max_tokens\":{},\
         \"mid_sentence_period\":{},\"prefix_tokens\":{},\"split_on\":[{}]}}",
        abbreviations.join(","),
        json_str(&cfg.chunking.cap_resplit),
        cfg.chunking.enabled,
        cfg.chunking.max_tokens,
        json_str(&cfg.chunking.mid_sentence_period),
        cfg.chunking.prefix_tokens,
        split_on.join(",")
    );

    let sampling = format!(
        "{{\"max_new_tokens\":{},\"min_p\":{},\"min_tokens_floor\":{},\
         \"min_tokens_text_ratio\":{},\"repetition_penalty\":{},\
         \"silence_token_ids\":{},\"temperature\":{}}}",
        cfg.sampling.max_new_tokens,
        json_num_str(cfg.sampling.min_p),
        cfg.sampling.min_tokens_floor,
        json_num_str(cfg.sampling.min_tokens_text_ratio),
        json_num_str(cfg.sampling.repetition_penalty),
        json_list(&cfg.sampling.silence_token_ids),
        json_num_str(cfg.sampling.temperature),
    );

    let window = format!(
        "{{\"max_speech_tokens\":{},\"pad_token_id\":{},\"static_length\":{},\
         \"static_prompt_tokens\":{}}}",
        cfg.window.max_speech_tokens,
        json_opt_usize(cfg.window.pad_token_id),
        json_opt_usize(cfg.window.static_length),
        json_opt_usize(cfg.window.static_prompt_tokens),
    );

    // Keys sorted, as everywhere in this form. The detectors remove tokens, so
    // a port using a different threshold produces different audio: exactly the
    // silent drift a whole-config hash exists to catch.
    let pp = &cfg.postprocess;
    let postprocess = format!(
        "{{\"ceiling_slack_tokens\":{},\"ceiling_speech_per_text_token\":{},\
         \"desperation_band_floor\":{},\"desperation_band_ratio\":{},\
         \"desperation_min_keep_per_text_token\":{},\
         \"desperation_min_text_tokens\":{},\"desperation_speech_per_text_token\":{},\
         \"dropout_min_tokens\":{},\
         \"echo_strong_eos_probability\":{},\"echo_strong_max_tail\":{},\
         \"echo_strong_min_position_pct\":{},\"echo_weak_eos_probability\":{},\
         \"echo_weak_max_tail\":{},\"echo_weak_min_position_pct\":{},\
         \"ended_tail_blip_max\":{},\"ended_tail_keep\":{},\
         \"ended_tail_silence_run\":{},\"ended_tail_word_max\":{},\
         \"filler_max_speech_after_run\":{},\"filler_min_eos_probability\":{},\
         \"mode\":{},\"pacing_tolerance\":{},\"quiet_render_ids\":{},\
         \"repetition_max_period\":{},\"repetition_min_cycles\":{},\
         \"repetition_min_span\":{},\"repetition_resume\":{},\
         \"repetition_silence\":{},\
         \"retry_max_attempts\":{},\
         \"silence_render_ids\":{},\"stall_run_tokens\":{},\
         \"trailing_filler_threshold\":{},\
         \"trailing_silence_run_tokens\":{}}}",
        pp.ceiling_slack_tokens,
        json_num_str(pp.ceiling_speech_per_text_token),
        pp.desperation_band_floor,
        json_num_str(pp.desperation_band_ratio),
        json_num_str(pp.desperation_min_keep_per_text_token),
        pp.desperation_min_text_tokens,
        json_num_str(pp.desperation_speech_per_text_token),
        pp.dropout_min_tokens,
        json_num_str(pp.echo_strong_eos_probability),
        pp.echo_strong_max_tail,
        pp.echo_strong_min_position_pct,
        json_num_str(pp.echo_weak_eos_probability),
        pp.echo_weak_max_tail,
        pp.echo_weak_min_position_pct,
        pp.ended_tail_blip_max,
        pp.ended_tail_keep,
        pp.ended_tail_silence_run,
        pp.ended_tail_word_max,
        pp.filler_max_speech_after_run,
        json_num_str(pp.filler_min_eos_probability),
        json_str(pp.mode.as_str()),
        json_num_str(pp.pacing_tolerance),
        json_list(&pp.quiet_render_ids),
        pp.repetition_max_period,
        pp.repetition_min_cycles,
        pp.repetition_min_span,
        json_str(pp.repetition_resume.as_str()),
        json_str(pp.repetition_silence.as_str()),
        pp.retry_max_attempts,
        json_list(&pp.silence_render_ids),
        pp.stall_run_tokens,
        json_num_str(pp.trailing_filler_threshold),
        pp.trailing_silence_run_tokens,
    );

    let euler_grid = cfg.euler_grid.as_ref().map_or_else(
        || "null".to_string(),
        |grid| {
            let values: Vec<String> = grid.iter().map(|v| json_num_str(*v)).collect();
            format!("[{}]", values.join(","))
        },
    );

    // The funnel's identity travels in the fingerprint: its code version, and the
    // digest of the grammar file this port reads. Each implementation hashes its own
    // copy, so a port whose data has drifted computes a different fingerprint and the
    // engine refuses to start, which is how drift is caught, rather than by someone
    // eventually hearing it.
    let text = format!(
        "{{\"grammar\":{},\"recipe\":{}}}",
        json_str(&cfg.text.grammar),
        json_str(&cfg.text.recipe)
    );

    let decode = if cfg.decode_mode == "single" {
        String::new()
    } else {
        format!("\"decode_mode\":{},", json_str(&cfg.decode_mode))
    };
    let fade = if cfg.edge_fade_seconds == 0.005 {
        String::new()
    } else {
        format!(
            "\"edge_fade_seconds\":{},",
            json_num_str(cfg.edge_fade_seconds)
        )
    };
    let body = format!(
        "{{\"chunking\":{chunking},{decode}{fade}\"euler_grid\":{euler_grid},\"euler_steps\":{},\
         \"guidance\":{},\"guidance_rate\":{},\"postprocess\":{postprocess},\
         \"recipe_version\":{},\"sample_rate\":{},\
         \"sampling\":{sampling},\"speech_vocab_size\":{},\"start_speech_token\":{},\
         \"stop_speech_token\":{},\"text\":{text},\"token_rate_hz\":{},\"window\":{window}}}",
        cfg.euler_steps,
        json_str(&cfg.guidance),
        json_num_str(cfg.guidance_rate),
        json_str(&cfg.recipe_version),
        cfg.sample_rate,
        cfg.speech_vocab_size,
        cfg.start_speech,
        cfg.stop_speech,
        json_num_str(cfg.token_rate_hz),
    );

    format!("{{\"algorithm\":{body},\"schema\":{FINGERPRINT_SCHEMA}}}")
}

/// First 16 hex characters of SHA-256 over [`canonical_form`].
///
/// Two engines whose fingerprints differ are computing different things,
/// whatever their outputs happen to sound like, which is the whole point: the
/// guidance defect this project was built around produced plausible audio on
/// both sides of the mismatch.
#[must_use]
pub fn fingerprint(cfg: &EngineConfig) -> String {
    let digest = Sha256::digest(canonical_form(cfg).as_bytes());
    crate::hex(&digest)[..16].to_string()
}
