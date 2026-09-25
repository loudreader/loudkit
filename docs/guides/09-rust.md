# 9. Rust

The Rust port of loudkit runs on ONNX Runtime through the `ort` crate. It does
not need Python or PyTorch.

## Hello

The crate loads the ONNX Runtime shared library at run time. You need version
1.27 or newer. The library is not part of the crate. Install it separately:

- macOS: `brew install onnxruntime`. Check that its version is 1.27 or newer.
- Linux: unpack the `onnxruntime-linux-x64` archive from the
  [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases).
- Windows: unpack the `onnxruntime-win-x64` archive from the same page.

`Engine::load` looks for the library in this order:

1. the file that `ORT_DYLIB_PATH` names
2. the file that `LOUDKIT_ONNXRUNTIME_LIB` names
3. `/opt/homebrew/lib` and `/usr/local/lib` on macOS; `/usr/local/lib`,
   `/usr/lib`, `/usr/lib/x86_64-linux-gnu` and `/usr/lib/aarch64-linux-gnu` on
   Linux; `C:\Program Files\onnxruntime\lib` and the working directory on
   Windows

Set `ORT_DYLIB_PATH` to the library file in two cases:

- A provider feature (`cuda`, `coreml`, `directml`) is on. The crate then loads
  the library before this search, so `LOUDKIT_ONNXRUNTIME_LIB` and the
  directories above are not used.
- The program loads an `Enroller` with `Enroller::load_with` before it loads
  any `Engine`. `Enroller::load_with` does not search for the library.

Distribution packages such as `libonnxruntime-dev` can be older than 1.27.

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

The first run downloads the model files into the user cache:

- macOS: `~/Library/Caches/loudkit/loudreader--loudr-1`
- Linux: `$XDG_CACHE_HOME/loudkit/loudreader--loudr-1`, or
  `~/.cache/loudkit/loudreader--loudr-1` when `XDG_CACHE_HOME` is not set
- Windows: `%LOCALAPPDATA%\loudkit\loudreader--loudr-1`

Set `LOUDKIT_CACHE` to use `$LOUDKIT_CACHE/loudreader--loudr-1` instead. The
Go, JS and Swift ports use the same directory. Each downloaded file is checked
against the release's `SHA256SUMS`. Later runs reuse the cache and fetch only
the files that changed when the repo's `main` branch moves. `engine.voices()`
lists the 28 voices.

The examples on this page need loudkit 0.1.1. To run the example from a
repository checkout, run `cargo run --example hello` in `rust/`.

`loudr-1` and `loudr-1-turbo` use the same API. To switch models, change the
model name. The same voice profiles work with both models. `Engine::load` also
accepts a local release directory.


## Download to a local directory

```rust
use loudkit::hub;

let options = hub::Options { revision: "v0.1.1".into(), ..Default::default() };
let dir = hub::download_with("loudreader/loudr-1", "loudr-1", &options)?;
let mut engine = Engine::load(&dir)?;
```

`download` writes a receipt, `.loudkit-release.json`, into the directory. On
a later call:

- If the revision still resolves to the same commit, `download` fetches and
  hashes no weight file.
- If the revision moved, it keeps every file that matches the new
  `SHA256SUMS` and fetches the rest.
- An interrupted fetch resumes.

Pin `revision` to a tag or commit for a reproducible build. For repos under
`loudreader/`, `release.json` must show that the release passed its build
checks. `download` reads it before it fetches any weight file.

## Synthesize

`synthesize` splits long text into chunks at sentence boundaries. It gives
each chunk its own seed, conditions each chunk on the speech tokens at the end
of the chunk before it, and returns one `Synthesis` that holds all the audio.
`Options::default()` selects every default.

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
longer text. The conformance tests use it.

## Streaming and barge-in

```rust
let mut stop = || interrupted();
engine.stream(text, &voice, &Options::default(), Some(&mut stop), &mut |chunk| {
    play(chunk.audio); // chunk.timing starts at zero; shifted() adds your offset
    true               // false stops at the next chunk
})?;
```

`stream` passes each chunk to the callback when it is ready, so playback can
start before the passage is finished. The cancel closure is checked on every
decode step. When it returns true, the chunk in progress is discarded.

`synthesize` reads the same flag from `Options.should_cancel`. When it
returns true, `synthesize` returns `Err(error::CANCELLED)` and no audio.

## Timestamps and speed

`out.chunks` is exact at the chunk level and an estimate at the word level.
Read [timestamps.md](../reference/timestamps.md) before you use the word
times. `speed` outside `[0.5, 2.0]` is refused; see
[speed.md](../reference/speed.md).

## Cloning a voice

```rust
let mine = engine.enroll_wav("me.wav", "mine", "en")?;
mine.save("mine.safetensors")?;
```

On an engine loaded by repo id, the first `enroll_wav` call fetches the three
enrollment graphs into the model's cache directory. Later calls use the cached
graphs. For an engine loaded from a local directory, fetch the graphs first
with
`hub::download_with(repo, dir, &hub::Options { cloning: true, ..Default::default() })`.

Use five to ten seconds of clean speech. `enroll_wav` reads 16-bit PCM and
32-bit float WAV files at any sample rate and channel count. `enroll` takes
samples. `voice::load(path)` loads a saved profile.

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

A provider other than `cpu` runs only if two conditions are true:

- Its cargo feature is on. Enable it when you add the crate, for example
  `cargo add loudkit@0.1.1 --features cuda`. The features are `cuda`, `coreml`
  and `directml`.
- The ONNX Runtime library at `ORT_DYLIB_PATH` contains the provider.

No provider feature is on by default, so `auto` selects `cpu` in a default
build. If you name a provider that is not available, `load_with` returns an
error that names the missing condition.

`coreml` runs the renderer on CoreML and keeps the token generator on the CPU.
The speech tokens are identical to a `cpu` run, but the waveform is not
bit-identical. The first run compiles the graphs, which takes about two
minutes. The compiled graphs are cached in `~/Library/Caches/loudkit/coreml`.
Set `LOUDKIT_COREML_CACHE` to use another directory.

On an RTX 3090, measured on 0.1.0, the CUDA provider runs at 3.60x real time
and the CPU provider at 0.70x on the same host. On an Apple M3 Pro, measured on
0.1.1, the CPU provider runs the shared passage at 1.21x with loudr-1 and 1.74x
with loudr-1-turbo. For CUDA, add the crate with `--features cuda` as shown
above and use a CUDA build of ONNX Runtime.

## Explicit asset paths

`Engine::load_paths(checkpoint, onnx_dir, tokenizer)` loads a checkpoint, a
directory of ONNX graphs and a tokenizer file from the paths you give.
`hub::Bundle::open(dir)` finds these paths in a release directory without
loading the engine.

## Verify against the shared fixture

```bash
cd rust && cargo test                                              # weight-free vectors
LOUDKIT_CKPT=… LOUDKIT_ONNX_DIR=… LOUDKIT_VOICE=… ORT_DYLIB_PATH=… cargo test -- --ignored
```

The engine conformance test is marked `#[ignore]`, so a run without the assets
reports it as ignored. Run it with `cargo test -- --ignored` after you set the
asset variables. It compares the tokens from `synthesize` and
`synthesize_window` with the fixture, chunk by chunk.
