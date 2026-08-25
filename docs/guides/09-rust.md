# 9. Rust

The same engine as a Rust crate over the `ort` ONNX Runtime binding, running
the fp32 ONNX graphs exported by `tools/export_onnx.py`. No torch at runtime,
only the graphs, the checkpoint's embedding tables and the onnxruntime shared
library.

This guide assumes you have:

- the synthesis checkpoint (`loudr-1.safetensors`);
- the exported graphs beside it (`onnx/`); the release ships them, and the
  fetch below gets them with everything else;
- a voice profile (see guide 3);
- the text tokenizer (`tokenizer.json`, ships beside the checkpoint);
- a `libonnxruntime` shared library. The `ort` crate loads it at runtime. Point
  at it with `ORT_DYLIB_PATH`. Any recent release works, including the venv's
  `onnxruntime/capi/libonnxruntime.*.dylib`.

**One fetch gets the whole set.** The graphs alone cannot run: this port also
reads the checkpoint's embedding tables, the tokenizer and a voice.
`loudkit download --for onnx` fetches all of it together, and nothing the
torch path alone would need.

```bash
pip install "loudkit[hub]"      # the fetch tool only; no torch
loudkit download loudreader/loudr-1 --for onnx --local-dir loudr-1
```

To clone a voice from this port, add `--with-cloning`. It fetches the three
enrollment graphs this port enrols through, and not the torch enrollment
weights, which only Python reads:

```bash
loudkit download loudreader/loudr-1 --for onnx --with-cloning --local-dir loudr-1
```

Exporting instead needs a Python checkout with torch installed, once, on any
machine. Nothing in this crate
links torch, and the machine that runs it never needs torch. If that machine
cannot have torch even once, export on another one and copy the `onnx/`
directory across. The graphs are plain files with no host affinity.

## Build

```bash
cd rust
cargo build --release
```

## Speak

```rust
use loudkit::engine::Engine;
use loudkit::voice;

fn main() -> Result<(), String> {
    let mut eng = Engine::load(
        "loudr-1/loudr-1.safetensors",
        "loudr-1/onnx",
        "loudr-1/tokenizer.json",
    )?;
    let v = voice::load("loudr-1/voices/joe.safetensors")?;
    // synthesize_long, not synthesize: synthesize renders one window and
    // returns an error for anything longer rather than clipping it.
    //
    // The arguments after the seed are, in order:
    //   language        `None` means the voice's own: the argument, then
    //                   `voice.language`, then "en". Pass `Some("pl")` only to
    //                   read Polish text in a voice not enrolled in Polish.
    //   speed           playback speed in [0.5, 2.0], pitch preserved. `1.0`
    //                   is an exact bypass: the vocoder's own samples.
    //   previous_tokens the token vector of an earlier call, so this one
    //                   continues its pitch contour instead of restarting
    //                   like a fresh sentence. `None` for a standalone call.
    //   should_cancel   `None` here. Pass `Some(&mut closure)` for barge-in
    //                   and the decode loop stops within one forward pass.
    //
    // It returns the audio, the speech tokens, the mel, the sample rate, the
    // chunk timeline, and the token-cap flag. The flag is true when generation
    // stopped at the cap rather than at a stop token, which means the reading
    // is probably truncated.
    let (audio, tokens, _, sr, chunks, capped) =
        eng.synthesize_long("Hello from loudkit.", &v, 7, None, 1.0, None, None)?;
    if capped {
        eprintln!("warning: the reading is probably truncated");
    }
    println!("{} tokens, {:.2}s audio", tokens.len(), audio.len() as f64 / sr as f64);
    for c in &chunks {
        println!("  [{:.2}s .. {:.2}s] {}", c.start, c.end, c.text);
    }
    Ok(())
}
```

Same seed, same tokens as the Python engine. The conformance fixture is
verified by `pytest`, `swift test`, and the JS, Go and Rust bindings, down to
exact Philox bits, sampler choices, frontend ids and free-running tokens.

### Passages longer than one window

`eng.synthesize` renders exactly one window. It returns an error for anything
longer rather than clipping the end of a paragraph. `eng.synthesize_long` is
the long-form path. It runs the speech funnel over the whole text, splits on
the manifest's `chunking` recipe, gives each chunk its own derived seed, and
carries a token prefix across the joins so the pitch contour does not restart
at every boundary. Same algorithm as Python's `Engine.synthesize_long`, same
fingerprint.

### Timestamps, speed, and continuing an earlier call

The fifth return value is the chunk timeline. Chunk boundaries are **exact**:
they are sample offsets the engine already knew, so chunk *k*'s `end` is the
same `f64` as chunk *k+1*'s `start`, and the last `end` is the whole duration.
The per-word times inside each entry are an **estimate**, allocated in
proportion to each word's length in characters. There is no alignment model
here. Read `src/timing.rs` before building anything that leans on them.

`speed` is the video-player control: 1.5x is the same voice, sooner, with the
pitch unmoved (WSOLA, `src/timestretch.rs`). It is applied last, after the
artifact detectors have inspected the render, and the timeline is measured on
the stretched waveform. There is no `1/speed` correction to apply anywhere.
Outside `[0.5, 2.0]` the call is refused rather than clamped.

`previous_tokens` makes one call continue another. Pass the token vector an
earlier call returned and this one is conditioned on its tail, exactly as an
interior chunk is conditioned on its predecessor. Only the last
`chunking.prefix_tokens` are used, so passing the whole vector is the intended
usage. Both `speed` and `previous_tokens` are execution inputs like the seed:
neither is in the fingerprint, and the defaults (`1.0`, `None`) are
byte-for-byte the behaviour from before they existed.

## Choose the execution provider

`Engine::load_with` and `Enroller::load_with` take an `ExecutionConfig`. Its
`onnx_provider` accepts the same five values as the other ports.

```rust
use loudkit::execution::{ExecutionConfig, OnnxProvider};

let execution = ExecutionConfig { onnx_provider: OnnxProvider::Cpu };
let mut eng = Engine::load_with(ckpt, onnx_dir, tokenizer, &execution)?;
println!("{}", eng.describe());  // ... | exec[onnx provider=cpu]
```

The CLI carries the same knob as `--provider`.

`auto` is the default. It takes CUDA where the build offers it, and CPU
otherwise; it reaches neither CoreML nor DirectML. A named provider that is not
available is an error, never a quiet demotion to CPU.

Two things must be true for a provider to run here, and the refusal says which
one is missing. The cargo feature must be on, because `ort` puts each provider's
registration code behind its own feature. The `libonnxruntime` at
`ORT_DYLIB_PATH` must also carry the provider, because `load-dynamic` makes the
provider set a property of the library you supply at runtime.

The default build compiles no provider feature in, so `auto` resolves to `cpu`
here whatever the machine offers. Build with `--features coreml` or
`--features cuda` to make one reachable.

**`coreml` runs the renderer, not the whole engine.** It puts the three renderer
graphs on CoreML with `ModelFormat=MLProgram` and keeps the generator on CPU,
because the generator has no winning CoreML configuration. The speech tokens are
therefore identical to a `cpu` run, index for index; the waveform is not
bit-identical, which is what
[`../reference/IDENTITY-CONTRACT.md`](../reference/IDENTITY-CONTRACT.md) says
about running the renderer elsewhere.

The first run on a machine compiles those graphs, which takes about two minutes.
The result is cached in `~/Library/Caches/loudkit/coreml`, about 1.6 GB, and
later runs open in about 25 s against 3 s for `cpu`. `$LOUDKIT_COREML_CACHE`
moves the directory. That startup cost is why `auto` does not pick `coreml`.
Measured RTF is in [`../benchmarks.md`](../benchmarks.md#onnx-execution-providers).

CUDA measured 3.60x on an RTX 3090, against 0.70x for the CPU provider on the
same host. Build with `--features cuda` and use a CUDA-enabled ONNX Runtime.

## Verify against the shared fixture

The weight-free vectors (RNG, sampler, frontend, seed derivation) run anywhere:

```bash
just rust-test          # or: cd rust && cargo test --test weightfree
```

The full engine (tokens + render band) needs the assets and the shared library
and skips without them:

```bash
ORT_DYLIB_PATH=…/libonnxruntime.dylib \
LOUDKIT_CKPT=…/loudr-1/loudr-1.safetensors \
LOUDKIT_ONNX_DIR=…/loudr-1/onnx \
LOUDKIT_VOICE=…/tests/data/reference/testvoice.voice.safetensors \
cargo test
```

The ported pieces live in `rust/src/`: `rng.rs` (Philox-4x32-10,
native u32), `sampler.rs` (LR-SAMPLER-v1), `tokenizer.rs` (grapheme BPE),
`windowing.rs`, `noise.rs`, `checkpoint.rs`, `voice.rs`, `safetensors.rs`,
`timing.rs`, `timestretch.rs`, `engine.rs`. Each mirrors a Python, JS or Go
module 1:1 so a fix on any side is a one-line diff on the others.

## Scope

- **Enrollment.** Ported, and held to the shared enrollment fixture (prompt and
  conditioning tokens exact, embeddings cosine > 0.9999).

Not ported:

- **fp16 / int8.** The graphs are fp32, the registered and measured
  configuration. int8 stays blocked; nothing int8 is produced.
- **The server.** `engine.stream` delivers chunks in process, and
  `synthesize_long` is built on it; only Python streams over a transport. Wrap
  it in your own server if you need one over the wire.
