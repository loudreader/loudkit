# loudkit Rust binding

The loudkit engine over the `ort` crate (load-dynamic). The rng, sampler,
tokenizer, windowing, engine and the Polish respelling lexicon are ported to
Rust, with **no torch** at runtime.

## Status: supported

Passes the shared conformance fixture: weight-free vectors exact, free-run
tokens and render band against the checkpoint. The full walkthrough is
`docs/guides/09-rust.md` in the repo root.

## Requirements

- Rust edition 2021 (stable toolchain)
- a `libonnxruntime` shared library (load-dynamic; point `ORT_DYLIB_PATH` at it)
- the exported ONNX graphs, the packed checkpoint, a voice profile, and
  `tokenizer.json`

## Synthesise

```toml
[dependencies]
loudkit = "0.1"
```

```bash
pip install "loudkit[hub]"
loudkit download loudreader/loudr-1 --for onnx --local-dir loudr-1
```

Everything lands inside `loudr-1/`, which is what the paths below are relative
to. `--with-cloning` adds the three enrollment graphs.

```rust
use loudkit::engine::Engine;
use loudkit::voice;

fn main() -> Result<(), String> {
    let mut eng = Engine::load(
        "loudr-1/loudr-1.safetensors", // packed checkpoint
        "loudr-1/onnx",                // exported graphs
        "loudr-1/tokenizer.json",      // text tokenizer
    )?;
    let v = voice::load("loudr-1/voices/joe.safetensors")?;

    // Use synthesize_long. synthesize renders one window and errors on
    // anything longer instead of clipping it. The `None`s are language (the voice's own),
    // previous_tokens and should_cancel.
    let (audio, tokens, _mel, sr, _chunks, _capped) =
        eng.synthesize_long("Hello from loudkit.", &v, 7, None, 1.0, None, None)?;
    println!("{} tokens, {:.2}s", tokens.len(), audio.len() as f64 / sr as f64);
    // `audio` is f32 at `sr`. Hand it to your own WAV writer or audio
    // device. This crate writes no files.
    Ok(())
}
```

`ORT_DYLIB_PATH` must point at a `libonnxruntime` before this runs. Streaming,
timestamps, speed and barge-in: `docs/guides/09-rust.md`.

## Execution provider

The default build runs on the CPU provider. `Engine::load_with` and
`Enroller::load_with` take an `ExecutionConfig` whose `onnx_provider` is one of
`auto` (the default), `cpu`, `cuda`, `coreml` or `directml`, the same five
values the Python, Go and TypeScript bindings accept.

```rust
use loudkit::execution::{ExecutionConfig, OnnxProvider};

let execution = ExecutionConfig { onnx_provider: OnnxProvider::Cuda };
let mut eng = Engine::load_with(ckpt, onnx_dir, tokenizer, &execution)?;
println!("{}", eng.describe());  // ... | exec[onnx provider=cuda]
```

`auto` takes CUDA where the build offers it and CPU otherwise, and the describe line
names the one it took. A named provider that is not available is an
error, never a quiet demotion to the CPU: a benchmark row that still says `cuda`
over a CPU number is worse than a refusal.

Two things have to be true for a provider to run, and the refusal says which one
is missing.

| provider | cargo feature | shared library |
| --- | --- | --- |
| `cpu` | none | any `libonnxruntime` |
| `cuda` | `--features cuda` | the onnxruntime-gpu (CUDA) build |
| `coreml` | `--features coreml` | an Apple-platform onnxruntime built with CoreML |
| `directml` | `--features directml` | the `Microsoft.ML.OnnxRuntime.DirectML` build, on Windows |

No feature is on by default, and the default build needs nothing but a plain
`libonnxruntime`. Because the crate uses `ort`'s `load-dynamic`, a feature only
compiles the registration code in. The library at `ORT_DYLIB_PATH` still has to
carry the provider.

A GPU provider can change the token stream and waveform. CoreML is available by
name when both the feature and a compatible runtime are present, but `auto`
does not select it because its first compile is expensive. The conformance
fixture pins CPU. See
[`docs/benchmarks.md`](../docs/benchmarks.md#onnx-execution-providers).

The default build compiles no provider feature in, so `auto` resolves to `cpu`
here. In every port, `auto` prefers CUDA when available and otherwise uses CPU.

CUDA has been measured on an RTX 3090. DirectML is unit-tested for resolution
and refusal text but has not been measured on Windows.

The CLI carries the same knob as `--provider`.

## Build and test

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --test weightfree          # weight-free conformance vectors
cargo test                            # + engine conformance (needs LOUDKIT_* assets)

cargo clippy --all-targets --features cuda,coreml,directml -- -D warnings
```

## Runtime libraries and assets

The ONNX Runtime shared library and the checkpoint/graphs are **not** bundled.
The crate loads them from the paths you provide.
