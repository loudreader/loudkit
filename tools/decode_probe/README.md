# Compare the five decode loops

The library has five implementations of the same decode: `python/loudkit/models/generator.py`
(and `backends/onnx_backend.py`), `go/engine/engine.go`, `rust/src/engine.rs`,
`js/src/engine.ts` and `swift/LoudKit/TokenGenerator.swift`. Each ONNX binding
needs a different ONNX Runtime build, so they rarely run side by side. This
harness runs them on one input and compares the stages in the order a
divergence has to be localised in.

Each probe writes one JSON record and the raw float32 arrays beside it:

1. `text_tokens`: what the funnel handed the generator
2. `prefill` / `cond`: the embedding rows (Python and JS; unexported elsewhere)
3. `speech_tokens_raw` and `speech_tokens`: the decode, before and after the
   stop marker is stripped
4. `mel`: the flow decoder's frames
5. `audio`: the vocoder's samples
6. `longform`: the whole `synthesize` path: chunking, the prefix carry, the
   retry ladder

`compare.py` stops at the first stage that disagrees. Resolve that difference
first: the later stages may differ only because of it.

## Running

Build the probes first, from the repository root:

```sh
mkdir -p out
(cd go && go build -o ../out/probe-go ../tools/decode_probe/probe.go)
cargo build --release --manifest-path rust/Cargo.toml --example probe
npm --prefix js run build
swift build -c release
swiftc -O -I .build/arm64-apple-macosx/release/Modules \
  tools/decode_probe/probe.swift \
  .build/arm64-apple-macosx/release/LoudKit.build/*.o \
  .build/arm64-apple-macosx/release/LoudKitText.build/*.o \
  -o out/probe-swift
```

`all.sh` runs `out/probe-swift` directly, so link it before the first run.
`tools/decode_probe/swift.sh` runs the same link step and then the probe.

Then run:

```sh
tools/decode_probe/all.sh TEXT SEED LANGUAGE VOICE OUTDIR
```

A probe that fails prints `<name> FAILED`, and `compare.py` compares the
records that exist. Check that all five records are present before you call a
result a five-way comparison.

`env.sh` holds the asset paths and the two ONNX Runtime paths, with macOS
defaults; override any of them from the caller. The Go binding needs ORT
API 28, which is ONNX Runtime 1.28 or newer. The Rust `ort` build refuses
anything older than 1.27. So the default Rust library, the 1.27 build that
`onnxruntime-node` ships, does not work for Go. JS ships its own runtime.
Swift runs CoreML and reads `coreml/` out of the same bundle.

Every probe prints the module or binary it actually loaded before it measures.

## Limits

The four ONNX ports here run three different ONNX Runtime builds. A
sample-level difference between them can come from the runtime, the execution
settings or the port code. Tokens are the better comparison across runtimes,
but they are not guaranteed to match either. The identity contract
(`docs/reference/IDENTITY-CONTRACT.md`) does not promise waveform identity
across backends.

## The other two probes

- `behaviour.py`: determinism, the two-thread split in `python/loudkit/stream.py`,
  and the cancellation poll, none of which a single decode exercises.
- `cancel_py.py` and `cancel.go`: the same two callback shapes in two ports,
  for measuring what each does with a partial row.
