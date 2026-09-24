# loudkit for Rust

Text to speech in Rust, on ONNX Runtime through the `ort` crate. It does not
need Python or PyTorch.

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

The first run downloads the model files into the user cache:

- macOS: `~/Library/Caches/loudkit/loudreader--loudr-1`
- Linux: `$XDG_CACHE_HOME/loudkit/loudreader--loudr-1`, or
  `~/.cache/loudkit/loudreader--loudr-1` when `XDG_CACHE_HOME` is not set
- Windows: `%LOCALAPPDATA%\loudkit\loudreader--loudr-1`

Set `LOUDKIT_CACHE` to use `$LOUDKIT_CACHE/loudreader--loudr-1` instead. The
Go, JS and Swift ports use the same directory. Each downloaded file is checked
against the release's `SHA256SUMS`. `engine.voices()` lists the 28 voices.

The examples on this page need loudkit 0.1.1. To run the example from a
repository checkout, run `cargo run --example hello` in `rust/`.
`examples/hello.rs` is the program above.

`loudr-1` and `loudr-1-turbo` use the same API. To switch models, change the
model name. The same voice profiles work with both models. `Engine::load` also
accepts a local release directory.


## API overview

```rust
loudkit::download("loudreader/loudr-1", "loudr-1")?;          // download to a local directory
hub::download_with(repo, dir, &hub::Options { cloning: true, ..Default::default() })?;
Engine::load("loudr-1")?;                                     // a local directory or a repo id
engine.voices()?;                                             // the voice names in the release
engine.voice("joe")?;                                         // load one voice by name
engine.synthesize(text, &voice, &Options { seed, language, speed, previous_tokens, ..Default::default() })?;
engine.stream(text, &voice, &options, None, &mut |chunk| { play(chunk.audio); true })?;
let mine = engine.enroll_wav("me.wav", "mine", "en")?;         // clone; fetches the enrollment graphs once
mine.save("mine.safetensors")?;                               // a portable profile
voice::load("mine.safetensors")?;
out.save_wav(path)?;                                          // 16-bit PCM
```

`synthesize` splits long text into chunks at sentence boundaries and joins
the audio in memory. `Options::default()` selects seed 0, the voice's own
language and speed 1.0. `stream` passes each chunk to the callback when it is
ready. Return `false` to stop, or set `Options.should_cancel`, which is checked
on every decode step. `synthesize` reads it too, and then returns
`Err(error::CANCELLED)`.

On an engine loaded by repo id, the first `enroll_wav` call fetches the
enrollment graphs. For a local directory, fetch them with `cloning: true`.
`Engine::load_paths(checkpoint, onnx_dir, tokenizer)` loads assets from the
paths you give.

Streaming, timestamps, speed and barge-in: `docs/guides/09-rust.md`.

## Execution provider

`Engine::load_with` and `Enroller::load_with` take an `ExecutionConfig` whose
`onnx_provider` is `auto` (the default), `cpu`, `cuda`, `coreml` or
`directml`. The Python, Go and JS implementations accept the same five values.

```rust
use loudkit::execution::{ExecutionConfig, OnnxProvider};

let execution = ExecutionConfig { onnx_provider: OnnxProvider::Cuda };
let mut engine = Engine::load_with("loudreader/loudr-1", &execution)?;
println!("{}", engine.describe());
```

A provider other than `cpu` runs only if its cargo feature is on and the ONNX
Runtime library at `ORT_DYLIB_PATH` contains it. Enable a feature when you add
the crate, for example `cargo add loudkit@0.1.1 --features cuda`.

| provider | cargo feature | shared library |
| --- | --- | --- |
| `cpu` | none | any `libonnxruntime` |
| `cuda` | `cuda` | the onnxruntime-gpu (CUDA) build |
| `coreml` | `coreml` | an Apple-platform onnxruntime built with CoreML |
| `directml` | `directml` | the `Microsoft.ML.OnnxRuntime.DirectML` build, on Windows |

No provider feature is on by default, so `auto` selects `cpu` in a default
build. If you name a provider that is not available, `load_with` returns an
error that names the missing condition. A GPU provider can change the token
stream and waveform. Conformance runs use CPU. See
[`docs/benchmarks.md`](../docs/benchmarks.md#onnx-execution-providers).

## Build and test

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test                            # weight-free; `cargo test -- --ignored` runs the engine conformance with LOUDKIT_* assets
cargo clippy --all-targets --no-default-features -- -D warnings
cargo test --no-default-features      # the build a consumer gets without `download`
```

The `download` feature is on by default. It adds `ureq` with rustls, the only
TLS code in the crate. With `default-features = false`, the crate has no
`download`. It then loads only from a local release directory and refuses a
repo id.
