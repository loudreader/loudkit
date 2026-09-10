//! Fetch loudr-1, speak one line, write hello.wav: `cargo run --example hello`.
use loudkit::engine::{Engine, Options};

fn main() -> Result<(), String> {
    let mut engine = Engine::load("loudreader/loudr-1")?;
    let joe = engine.voice("joe")?;
    let out = engine.synthesize(
        "Hello from loudkit.",
        &joe,
        &Options {
            seed: 7,
            ..Default::default()
        },
    )?;
    out.save_wav("hello.wav")?;
    println!("hello.wav: {:.2}s", out.duration());
    Ok(())
}
