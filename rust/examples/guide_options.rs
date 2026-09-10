//! The extended synthesis example in docs/guides/09-rust.md, compiled with examples.
use loudkit::engine::{Engine, Options, Synthesis};
use loudkit::voice::Profile;

#[allow(dead_code)]
fn synthesize(
    engine: &mut Engine,
    text: &str,
    voice: Profile,
    earlier: &Synthesis,
    path: &str,
) -> Result<(), String> {
    // BEGIN GUIDE
    let out = engine.synthesize(
        text,
        &voice,
        &Options {
            seed: 7,                                       // 0 by default
            language: Some("pl".into()),                   // None: the voice's own
            speed: 1.25, // [0.5, 2.0], pitch preserved; 1.0 is an exact bypass
            previous_tokens: Some(earlier.tokens.clone()), // continue an earlier result's pitch contour
            ..Default::default()
        },
    )?;
    let _audio = &out.audio; // Vec<f32> at out.sample_rate
    let _tokens = &out.tokens; // the speech tokens
    let _chunks = &out.chunks; // where each chunk lands, and an estimate of each word
    let _truncated = out.hit_token_cap; // generation stopped at the token cap: probably cut off
    out.save_wav(path)?;
    // END GUIDE
    Ok(())
}

fn main() {}
