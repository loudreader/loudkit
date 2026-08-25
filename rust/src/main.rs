use std::env;

use loudkit::engine::Engine;
use loudkit::execution::{ExecutionConfig, OnnxProvider};
use loudkit::voice;

/// What this binary takes, and — because someone comparing the two ports will
/// reach for both — how it differs from the Go CLI.
///
/// The two argv surfaces are deliberately not the same: each grew around what
/// that port needed in order to be driven by hand. Saying so here is the
/// alternative to converging them, and it keeps a reader from reading the
/// difference as a port that fell behind. Parity between the ports is the
/// library API and the conformance fixture, never these two flag lists.
const USAGE: &str = "usage: loudkit <checkpoint> <onnx-dir> <voice> [--text TEXT] [--seed N]
               [--language LANG] [--speed X] [--provider P] [--tokens] [--json OUT]

--provider is auto (the default), cpu, cuda, coreml or directml. A provider
this build or this libonnxruntime does not carry is refused, not demoted.

A dev tool for driving this port by hand. The Go CLI (go/cmd/loudkit) is
deliberately a different surface: it carries -timestamps, which this one has no
equivalent for, and neither --language nor --json, which this one has.";

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    // Only in first position: further along it is a value, and `--text -h`
    // asks for a reading of "-h", not for this text.
    if matches!(args.get(1).map(String::as_str), Some("-h" | "--help")) {
        println!("{USAGE}");
        return Ok(());
    }
    if args.len() < 4 {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    let ckpt = &args[1];
    let onnx_dir = &args[2];
    let voice_path = &args[3];
    let mut text = "Hello from Rust.".to_string();
    let mut seed = 0u64;
    // The kit ships voices for nine languages and the funnel is language-aware
    // (Polish respells embedded English, which changes the token count). This
    // was pinned to "en", so the CLI could not speak eight of them and silently
    // ran the wrong funnel for any text that was not English.
    //
    // `None`, not "en": an omitted --language now means the voice's own
    // language, resolved by the engine. A Polish voice needs no flag.
    let mut language: Option<String> = None;
    // Playback speed, in [0.5, 2.0]. The default is the bypass: with no --speed
    // the samples are the vocoder's own, so every existing invocation of this
    // CLI — including the JSON records the cross-port comparison hashes —
    // produces the same bytes it did before the flag existed.
    let mut speed = 1.0f64;
    let mut tokens_only = false;
    // Writes what was produced rather than a summary of it: the speech tokens
    // and a digest of the samples. "Do two ports render the same thing?" is a
    // question about those two values, and a printed duration cannot answer it.
    let mut json_out: Option<String> = None;
    // The knob the benchmark figures needed: every ONNX path here ran on the
    // CPU provider with no way to say otherwise, so a Rust number and a torch
    // number were being compared as if they described the same hardware.
    let mut execution = ExecutionConfig::default();
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--text" => {
                text = args.get(i + 1).ok_or("--text requires a value")?.clone();
                i += 1;
            }
            "--seed" => {
                seed = args
                    .get(i + 1)
                    .ok_or("--seed requires a value")?
                    .parse()
                    .map_err(|_| "bad seed")?;
                i += 1;
            }
            "--language" => {
                language = Some(
                    args.get(i + 1)
                        .ok_or("--language requires a value")?
                        .clone(),
                );
                i += 1;
            }
            "--speed" => {
                speed = args
                    .get(i + 1)
                    .ok_or("--speed requires a value")?
                    .parse()
                    .map_err(|_| "bad speed")?;
                i += 1;
            }
            "--json" => {
                json_out = Some(args.get(i + 1).ok_or("--json requires a value")?.clone());
                i += 1;
            }
            "--provider" => {
                execution.onnx_provider =
                    OnnxProvider::parse(args.get(i + 1).ok_or("--provider requires a value")?)?;
                i += 1;
            }
            "--tokens" => tokens_only = true,
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }

    // The onnxruntime shared library is loaded from ORT_DYLIB_PATH at first
    // ort::init(); ort's load-dynamic feature reads it on the first session.
    // The release ships `tokenizer.json` beside the checkpoint; appending
    // ".tokenizer.json" to the whole filename names a file no release
    // contains.
    let tokenizer = env::var("LOUDKIT_TOKENIZER").unwrap_or_else(|_| {
        std::path::Path::new(ckpt)
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("tokenizer.json")
            .to_string_lossy()
            .into_owned()
    });
    let mut eng = Engine::load_with(ckpt, onnx_dir, &tokenizer, &execution)?;
    // On stderr, on every run, beside the truncation warning: stdout is the
    // record a cross-port comparison reads and must not change shape, but a
    // benchmark figure with no provider beside it is not a figure anyone can
    // reuse.
    eprintln!("{}", eng.describe());
    let v = voice::load(voice_path)?;
    // The language this run actually speaks in, resolved by the library rather
    // than restated here: `--language`, then the voice, then English. Reported
    // in the JSON record and the summary line, because "which language" is the
    // question a cross-port comparison is asking.
    let language = loudkit::engine::resolve_language(language.as_deref(), &v).to_string();
    if tokens_only {
        // `encode` takes no voice and so cannot run the chain itself.
        let ids = eng.encode(&text, &language)?;
        let mut s = loudkit::sampler::Sampler::new(eng.config().sampling.clone(), seed);
        let raw = eng.generate(&ids, &v, &mut s, None, None, &[])?;
        println!("{raw:?}");
        return Ok(());
    }

    // `None` for previous_tokens: one CLI invocation is one utterance, and there
    // is no earlier call in this process for it to continue.
    let (audio, tokens, _, sr, chunks, hit_cap) =
        eng.synthesize_long(&text, &v, seed, Some(&language), speed, None, None)?;
    if hit_cap {
        // The flag exists so truncation cannot pass silently: the audio is
        // real but incomplete. Same warning the Python CLI prints.
        eprintln!(
            "warning: generation stopped at the token cap rather than at a stop \
             token, so the reading is probably truncated"
        );
    }
    if let Some(path) = json_out {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for sample in &audio {
            hasher.update(sample.to_le_bytes());
        }
        let digest: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        // Built by serde_json rather than format!: `--text` reaches this record
        // only through its token count, but the language does not — it comes
        // from a voice header or a flag, and either can hold a quote or a
        // backslash that string interpolation would happily emit as invalid
        // JSON.
        let record = serde_json::json!({
            "language": language,
            "seed": seed,
            "n_tokens": tokens.len(),
            "samples": audio.len(),
            "wav_sha256": digest[..16],
            "tokens": tokens,
        })
        .to_string();
        std::fs::write(&path, record).map_err(|e| e.to_string())?;
        println!(
            "rust-onnx    {language} tokens={} sha={}",
            tokens.len(),
            &digest[..16]
        );
        return Ok(());
    }

    let peak = audio.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    println!(
        "tokens={} audio={} samples @ {sr} Hz = {:.2}s peak={:.3} chunks={}",
        tokens.len(),
        audio.len(),
        audio.len() as f64 / sr as f64,
        peak,
        chunks.len()
    );
    Ok(())
}
