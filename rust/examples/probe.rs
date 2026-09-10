//! Stage-by-stage probe of the Rust decode loop, for cross-port comparison.
//!
//! The same stages in the same order as `tools/decode_probe/probe.py`, so a
//! comparison stops at the first stage that disagrees instead of reporting
//! noise from everything downstream of a divergence.
//!
//! ```text
//! cargo run --release --example probe -- BUNDLE VOICE TEXT SEED LANGUAGE OUTDIR
//! ```
//!
//! Writes `OUTDIR/rust.json` and `OUTDIR/rust.<stage>.bin` (raw
//! little-endian float32).

use std::env;
use std::fs;
use std::path::Path;

use loudkit::engine::Engine;
use loudkit::execution::{ExecutionConfig, OnnxProvider};
use loudkit::sampler::Sampler;
use loudkit::voice;
use sha2::{Digest, Sha256};

fn f32_bytes(a: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(a.len() * 4);
    for x in a {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn sha_of(a: &[f32]) -> String {
    let mut h = Sha256::new();
    h.update(f32_bytes(a));
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn head(a: &[f32], n: usize) -> Vec<f32> {
    a.iter().take(n).copied().collect()
}

fn json_f32(a: &[f32]) -> String {
    let parts: Vec<String> = a.iter().map(|x| format!("{x:?}")).collect();
    format!("[{}]", parts.join(","))
}

fn json_usize(a: &[usize]) -> String {
    let parts: Vec<String> = a.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(","))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 {
        eprintln!("usage: probe BUNDLE VOICE TEXT SEED LANGUAGE OUTDIR");
        std::process::exit(2);
    }
    let (bundle, voice_path, text) = (&args[1], &args[2], &args[3]);
    let seed: u64 = args[4].parse().map_err(|_| "bad seed")?;
    let (language, outdir) = (&args[5], &args[6]);

    // Which binary and which library actually answered.
    eprintln!(
        "rust probe: {}",
        env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_default()
    );
    eprintln!(
        "ORT_DYLIB_PATH: {}",
        env::var("ORT_DYLIB_PATH").unwrap_or_default()
    );

    fs::create_dir_all(outdir).map_err(|e| e.to_string())?;
    let dir = Path::new(bundle);
    let ckpt = dir.join("loudr-1.safetensors");
    let onnx = dir.join("onnx");
    let tokenizer = dir.join("tokenizer.json");
    let execution = ExecutionConfig {
        onnx_provider: OnnxProvider::parse("cpu")?,
    };
    let mut eng = Engine::load_paths_with(
        ckpt.to_str().unwrap(),
        onnx.to_str().unwrap(),
        tokenizer.to_str().unwrap(),
        &execution,
    )?;
    eprintln!("{}", eng.describe());
    let v = voice::load(voice_path)?;
    // Copied out, not held: `config()` hands back a reference, so a `cfg`
    // that is still borrowing the engine blocks every `&mut self` call below.
    let sampling = eng.config().sampling.clone();
    let limit = eng.config().start_speech;
    let decode_mode = eng.config().decode_mode.clone();
    let sample_rate = eng.config().sample_rate;
    let describe = eng.describe();

    // 1. text tokens.
    let text_tokens = eng.encode(text, language)?;

    // 3. the token sequence. (2, the prefill row, is private to the engine.)
    let mut s = Sampler::new(sampling, seed);
    let raw = eng.generate(&text_tokens, &v, &mut s, None, None, &[])?;
    let stripped: Vec<usize> = raw.iter().copied().filter(|t| *t < limit).collect();

    // 4. the mel frames.
    let mel = eng.decode_mel(&stripped, &v, seed)?;
    fs::write(Path::new(outdir).join("rust.mel.bin"), f32_bytes(&mel))
        .map_err(|e| e.to_string())?;

    // 5. the rendered samples.
    let audio = eng.vocode(&mel, seed)?;
    fs::write(Path::new(outdir).join("rust.audio.bin"), f32_bytes(&audio))
        .map_err(|e| e.to_string())?;

    // 6. the long-form path.
    let out = eng.synthesize(
        text,
        &v,
        &loudkit::engine::Options {
            seed,
            language: Some(language.clone()),
            speed: 1.0,
            previous_tokens: None,
            should_cancel: None,
        },
    )?;
    fs::write(
        Path::new(outdir).join("rust.longform.bin"),
        f32_bytes(&out.audio),
    )
    .map_err(|e| e.to_string())?;

    let record = format!(
        concat!(
            "{{\n \"port\": \"rust\",\n \"loaded_from\": {:?},\n \"bundle\": {:?},\n",
            " \"voice\": {:?},\n \"text\": {:?},\n \"seed\": {},\n \"language\": {:?},\n",
            " \"decode\": {:?},\n \"fingerprint\": {:?},\n \"sample_rate\": {},\n",
            " \"text_tokens\": {},\n \"speech_tokens_raw\": {},\n \"speech_tokens\": {},\n",
            " \"mel\": {{\"len\": {}, \"sha\": {:?}, \"head\": {}}},\n",
            " \"audio\": {{\"len\": {}, \"sha\": {:?}, \"head\": {}}},\n",
            " \"longform\": {{\"tokens\": {}, \"audio_len\": {}, \"audio_sha\": {:?},",
            " \"audio_head\": {}, \"n_chunks\": {}, \"hit_token_cap\": {}}}\n}}\n"
        ),
        env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
        bundle,
        voice_path,
        text,
        seed,
        language,
        decode_mode,
        describe,
        sample_rate,
        json_usize(&text_tokens),
        json_usize(&raw),
        json_usize(&stripped),
        mel.len(),
        sha_of(&mel),
        json_f32(&head(&mel, 8)),
        audio.len(),
        sha_of(&audio),
        json_f32(&head(&audio, 8)),
        json_usize(&out.tokens),
        out.audio.len(),
        sha_of(&out.audio),
        json_f32(&head(&out.audio, 8)),
        out.chunks.len(),
        out.hit_token_cap,
    );
    fs::write(Path::new(outdir).join("rust.json"), record).map_err(|e| e.to_string())?;
    eprintln!("wrote {}/rust.json", outdir);
    Ok(())
}
