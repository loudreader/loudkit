# Compare the public streaming paths

Each runner accepts `BUNDLE VOICE TEXT` (Python adds `DEVICE`). It loads once,
streams the same passage four times with seed 7, and writes one JSON object.
Run 0 includes cold caches; report medians of runs 1–3. `seconds` covers the
whole passage, `ttfa_s` stops at its first delivered chunk. `audio_s / seconds`
is end-to-end RTF, not generator throughput. Model load is separate.
Python also accepts `coreml-hybrid` to measure the retained fp16 PyTorch
generator with CoreML rendering.

Use complete local releases. Keep the text, voice, sample rate (24 kHz) and
seed fixed; differing models may produce different token counts and durations.
Run one process at a time, after builds, exports and tests finish. These are
local comparisons, not GPU or cross-machine claims. Report the actual runtime
versions and execution settings with results.

From the repository root, build first:

```bash
(cd go && go build -o ../out/bench-go ../tools/bench_ports/bench.go)
cargo build --release --manifest-path rust/Cargo.toml --example bench
npm --prefix js run build
swift build -c release
swiftc -O -I .build/arm64-apple-macosx/release/Modules \
  tools/bench_ports/bench.swift \
  .build/arm64-apple-macosx/release/LoudKit.build/*.o \
  .build/arm64-apple-macosx/release/LoudKitText.build/*.o -o out/bench-swift
```

Create `out/` before building. The Swift link command is for Apple silicon;
use the build directory for your architecture on another Mac.

Set `LOUDKIT_ONNXRUNTIME_LIB` (Go) and `ORT_DYLIB_PATH` (Rust) to a compatible
ONNX Runtime shared library. Both runners explicitly select its CPU provider;
JS uses the version supplied by `onnxruntime-node`. Python records its resolved
execution settings. Swift uses its native generator and default CoreML placement.

Then run, substituting your paths and the same text each time:

```bash
out/bench-go BUNDLE VOICE TEXT
rust/target/release/examples/bench BUNDLE VOICE TEXT
node tools/bench_ports/bench.mjs BUNDLE VOICE TEXT
out/bench-swift BUNDLE VOICE TEXT
PYTHONPATH=python .venv/bin/python tools/bench_ports/stream_bench.py BUNDLE VOICE TEXT onnx
PYTHONPATH=python .venv/bin/python tools/bench_ports/stream_bench.py BUNDLE VOICE TEXT coreml
```

On macOS, prefix each command with `/usr/bin/time -l`: its maximum resident
set size includes model loading and all four runs. Keep stderr with the JSON;
RSS units are bytes on macOS. Check repeated token, sample and chunk counts
before comparing medians. For Python cancellation timing and the standard
multi-passage suite, use the existing `tools/bench.py` instead.
