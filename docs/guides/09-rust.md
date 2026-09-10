# 9. Rust

The same engine as a Rust crate over ONNX Runtime, through `ort`. No Python,
no torch.

## Hello

One shared library that cannot be vendored: `brew install onnxruntime` on
macOS, `apt install libonnxruntime-dev` on Linux, the
[onnxruntime-win-x64 archive](https://github.com/microsoft/onnxruntime/releases)
on Windows. `Engine::load` finds it; set `LOUDKIT_ONNXRUNTIME_LIB` if yours is
somewhere unusual.

For a new application:

```bash
cargo new hello
cd hello
cargo add loudkit@0.1.1
```

Save this as `src/main.rs`, then run `cargo run`:

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
    Ok(())
}
```

The first run downloads the model files into
`~/Library/Caches/loudkit/loudreader--loudr-1` on macOS and
`~/.cache/loudkit/loudreader--loudr-1` elsewhere (`$LOUDKIT_CACHE` moves it),
the directory the Go, JS and Swift ports share, and checks every file
against the release's own `SHA256SUMS`; later runs read what is there.
`engine.voices()` names the 28 voices. The snippets need loudkit 0.1.1;
from a checkout, `cargo run --example hello`.

Both `loudr-1` and `loudr-1-turbo` use this API in 0.1.1. Change the model
name to switch; keep the same voice profile. A local release directory works
as well as a published model name.


## A directory of your own

```rust
use loudkit::hub;

let options = hub::Options { revision: "v0.1.1".into(), ..Default::default() };
let dir = hub::download_with("loudreader/loudr-1", "loudr-1", &options)?;
let mut engine = Engine::load(&dir)?;
```

`download` writes a receipt, `.loudkit-release.json`; a later call whose
revision still resolves to the same commit fetches and hashes nothing, a moved
revision keeps every file that still hashes to the new `SHA256SUMS` and fetches
the rest, and an interrupted fetch resumes. Pin `revision` for anything reproducible. Under `loudreader/`,
`release.json` must say the bundle passed the builder's gate, and it is
checked before any weight moves.

## Synthesize

`synthesize` takes text of any length: it splits at sentence boundaries,
gives each chunk its own seed, carries the pitch contour across the joins and
returns one `Synthesis`. `Options::default()` is every default.

```rust
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
```

`synthesize_window` renders exactly one model window and returns an error on
longer text; it is for the conformance harness.

## Streaming and barge-in

```rust
let mut stop = || interrupted();
engine.stream(text, &voice, &Options::default(), Some(&mut stop), &mut |chunk| {
    play(chunk.audio); // chunk.timing starts at zero; shifted() adds your offset
    true               // false stops at the next chunk
})?;
```

`stream` hands out chunks as they are made, so playback starts before the
passage is finished. The cancel closure is polled on every decode step, and
the chunk being generated is discarded. `Options.should_cancel` is the same
flag for `synthesize`, which returns `Err(error::CANCELLED)` instead of a
passage cut short.

## Timestamps and speed

`out.chunks` is exact at the chunk level and an estimate at the word level;
read [timestamps.md](../reference/timestamps.md) before building on the word
times. `speed` is refused outside `[0.5, 2.0]`; see
[speed.md](../reference/speed.md).

## Cloning a voice

```rust
let mine = engine.enroll_wav("me.wav", "mine", "en")?;
mine.save("mine.safetensors")?;
```

The first `enroll_wav` on an engine loaded by repo id fetches the three
enrollment graphs into the same cache directory,
`~/Library/Caches/loudkit/loudreader--loudr-1` on macOS and
`~/.cache/loudkit/loudreader--loudr-1` elsewhere; later calls read them from
there. An engine loaded from a directory of your own needs them fetched with
`hub::download_with(repo, dir, &hub::Options { cloning: true, ..Default::default() })`.
Five to ten seconds of clean speech is the input this was tuned for.
`enroll_wav` reads 16-bit PCM and 32-bit float WAVs at any rate and channel
count; `enroll` takes samples. `voice::load(path)` reads the profile back.

## Execution provider

`Engine::load_with` and `Enroller::load_with` take an `ExecutionConfig` whose
`onnx_provider` is `auto` (the default), `cpu`, `cuda`, `coreml` or
`directml`.

```rust
use loudkit::execution::{ExecutionConfig, OnnxProvider};

let execution = ExecutionConfig { onnx_provider: OnnxProvider::Cpu };
let mut engine = Engine::load_with("loudreader/loudr-1", &execution)?;
println!("{}", engine.describe());
```

Two things must be true for a provider to run, and the refusal says which one
is missing: the cargo feature (`--features cuda`, `coreml`, `directml`; none
is on by default, so `auto` resolves to `cpu` in a default build) and a
shared library that carries it. A named provider that is not available is an
error, never a quiet demotion to CPU.

`coreml` runs the renderer on CoreML and keeps the generator on CPU, so the
speech tokens are identical to a `cpu` run and the waveform is not
bit-identical. The first run compiles the graphs, about two minutes, cached
under `~/Library/Caches/loudkit/coreml` (`$LOUDKIT_COREML_CACHE` moves it).

CUDA measured 3.60x on an RTX 3090 (measured on 0.1.0), against 0.70x for the
CPU provider on the same host. On an Apple M3 Pro the CPU provider runs the
shared passage at 1.21x with loudr-1 and 1.74x with loudr-1-turbo, measured on
0.1.1. Build with `--features cuda` and use a CUDA-enabled ONNX Runtime.

## Your own layout

`Engine::load_paths(checkpoint, onnx_dir, tokenizer)` opens three paths you
assembled yourself, and `hub::Bundle::open(dir)` reads a release's paths
without loading it.

## Verify against the shared fixture

```bash
cd rust && cargo test                                              # weight-free vectors
LOUDKIT_CKPT=… LOUDKIT_ONNX_DIR=… LOUDKIT_VOICE=… ORT_DYLIB_PATH=… cargo test -- --ignored
```

The engine conformance is `#[ignore]` so a run without the assets reports it
as ignored rather than passed. It holds `synthesize` and `synthesize_window`
to the fixture's tokens, chunk by chunk.
