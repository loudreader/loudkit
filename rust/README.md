# loudkit for Rust

Text to speech in Rust, on ONNX Runtime through the `ort` crate. No Python,
no torch.

## Hello

One shared library that cannot be vendored: `brew install onnxruntime` on
macOS, `apt install libonnxruntime-dev` on Linux, the
[onnxruntime-win-x64 archive](https://github.com/microsoft/onnxruntime/releases)
on Windows. It is found automatically; set `LOUDKIT_ONNXRUNTIME_LIB` if yours is
somewhere unusual.

For a new application:

```bash
cargo new hello
cd hello
cargo add loudkit@0.1.1
```

`src/main.rs`:

```rust
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
```

```bash
cargo run
```

The first run downloads the model files into
`~/Library/Caches/loudkit/loudreader--loudr-1` on macOS and
`~/.cache/loudkit/loudreader--loudr-1` elsewhere (`$LOUDKIT_CACHE` moves it),
the directory the Go, JS and Swift ports share, and checks every file
against the release's own `SHA256SUMS`; later runs read what is there.
`engine.voices()` names the 28 voices.

The snippets on this page need loudkit 0.1.1. From a checkout, `cargo run
--example hello` runs `examples/hello.rs`, which is this file.

Both `loudr-1` and `loudr-1-turbo` use this API in 0.1.1. Change the model
name to switch; keep the same voice profile. A local release directory works
as well as a published model name.


## The rest of the front door

```rust
loudkit::download("loudreader/loudr-1", "loudr-1")?;          // a directory of your own
hub::download_with(repo, dir, &hub::Options { cloning: true, ..Default::default() })?;
Engine::load("loudr-1")?;                                     // a directory or a repo id
engine.voices()?;                                             // the names in the release
engine.voice("joe")?;                                         // one of them
engine.synthesize(text, &voice, &Options { seed, language, speed, previous_tokens, ..Default::default() })?;
engine.stream(text, &voice, &options, None, &mut |chunk| { play(chunk.audio); true })?;
let mine = engine.enroll_wav("me.wav", "mine", "en")?;         // clone; fetches the enrollment graphs once
mine.save("mine.safetensors")?;                               // a portable profile
voice::load("mine.safetensors")?;
out.save_wav(path)?;                                          // 16-bit PCM
```

`synthesize` takes text of any length: it splits at sentence boundaries and
joins the audio. `Options::default()` is seed 0, the voice's own language and
normal speed. `stream` hands out chunks as they are made; return `false` to
stop, or set `Options.should_cancel` (read by `synthesize` too, which then
returns `Err(error::CANCELLED)`) to stop within one decode step. `enroll_wav`
on an engine loaded by repo id fetches the enrollment graphs once; a directory
of your own needs `cloning: true`.
`Engine::load_paths(checkpoint, onnx_dir, tokenizer)` opens a layout of your
own.

Streaming, timestamps, speed and barge-in: `docs/guides/09-rust.md`.

## Execution provider

`Engine::load_with` and `Enroller::load_with` take an `ExecutionConfig` whose
`onnx_provider` is one of `auto` (the default), `cpu`, `cuda`, `coreml` or
`directml`, the same five values every port accepts.

```rust
use loudkit::execution::{ExecutionConfig, OnnxProvider};

let execution = ExecutionConfig { onnx_provider: OnnxProvider::Cuda };
let mut engine = Engine::load_with("loudreader/loudr-1", &execution)?;
println!("{}", engine.describe());
```

Two things must be true for a provider to run, and the refusal says which one
is missing:

| provider | cargo feature | shared library |
| --- | --- | --- |
| `cpu` | none | any `libonnxruntime` |
| `cuda` | `--features cuda` | the onnxruntime-gpu (CUDA) build |
| `coreml` | `--features coreml` | an Apple-platform onnxruntime built with CoreML |
| `directml` | `--features directml` | the `Microsoft.ML.OnnxRuntime.DirectML` build, on Windows |

No provider feature is on by default, so `auto` resolves to `cpu` in a default
build. A named provider that is not available is an error, never a quiet
demotion to CPU. A GPU provider can change the token stream and waveform;
conformance runs pin CPU. See
[`docs/benchmarks.md`](../docs/benchmarks.md#onnx-execution-providers).

## Build and test

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test                            # weight-free; `cargo test -- --ignored` runs the engine conformance with LOUDKIT_* assets
cargo clippy --all-targets --no-default-features -- -D warnings
cargo test --no-default-features      # the build a consumer gets without `download`
```

`download` is on by default: it pulls `ureq` with rustls, the whole of this
crate's TLS surface. `default-features = false` drops it and keeps the
engine, which then takes a release directory and no repo id.
