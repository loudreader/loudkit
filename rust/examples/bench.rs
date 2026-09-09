//! Four timed CPU runs of one text, as one JSON record on stdout.

use loudkit::engine::{Engine, Options};
use serde_json::json;
use std::{env, time::Instant};

fn main() -> Result<(), String> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 {
        // A usage line and exit 2, not `assert_eq!`: wrong arguments are a
        // usage error, and a panic answers them with a backtrace hint. The
        // same exit code `src/main.rs` uses for the same case.
        eprintln!("usage: bench BUNDLE VOICE TEXT");
        std::process::exit(2);
    }
    let mut start = Instant::now();
    let mut engine = Engine::load_with(
        &args[1],
        &loudkit::execution::ExecutionConfig {
            onnx_provider: loudkit::execution::OnnxProvider::Cpu,
        },
    )?;
    let load = start.elapsed().as_secs_f64();
    // The checkpoint's rate, not 24000: at any other rate the hardcoded
    // divisor reports the wrong duration for every run.
    let sample_rate = f64::from(u32::try_from(engine.config().sample_rate).unwrap_or(u32::MAX));
    let voice = loudkit::voice::load(&args[2])?;
    let mut runs = Vec::new();
    for run in 0..4 {
        start = Instant::now();
        let (mut first, mut samples, mut tokens, mut chunks) = (0., 0, 0, 0);
        engine.stream(
            &args[3],
            &voice,
            &Options {
                seed: 7,
                ..Default::default()
            },
            None,
            &mut |c| {
                if chunks == 0 {
                    first = start.elapsed().as_secs_f64();
                }
                chunks += 1;
                samples += c.audio.len();
                tokens += c.tokens.len();
                true
            },
        )?;
        runs.push(
            json!({"run":run,"seconds":start.elapsed().as_secs_f64(),"ttfa_s":first,
                        "audio_s":samples as f64/sample_rate,"tokens":tokens,"chunks":chunks}),
        );
    }
    println!(
        "{}",
        json!({"runtime":"rust","bundle":args[1],"load_s":load,"text":args[3],"seed":7,"runs":runs})
    );
    Ok(())
}
