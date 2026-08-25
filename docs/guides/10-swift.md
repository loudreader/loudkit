# 10. Swift

*the same engine from Swift, over CoreML, on the Neural Engine*

Swift is the only port that does not run on ONNX. It loads CoreML packages,
which means an export step the other ports do not have and a device placement
the other ports cannot use.

## What you need

- macOS 14 or iOS 17, and Xcode's CoreML runtime;
- the checkpoint (`loudr-1.safetensors`);
- **the CoreML packages beside it**: `flow_encoder.mlpackage`,
  `flow_estimator.mlpackage`, `vocoder.mlpackage`. These are not the ONNX
  graphs the other ports use, and they are not interchangeable with them;
- a voice profile (guide 3);
- the text tokenizer (`tokenizer.json`, ships beside the checkpoint).

There is no `libonnxruntime` to find and no environment variable to set. CoreML
is already on the system, and the release ships its packages. One fetch gets
the whole set: checkpoint, tokenizer, voices and the `coreml/` directory:

```bash
pip install "loudkit[hub]"      # the fetch tool only; no torch
loudkit download loudreader/loudr-1 --for coreml --local-dir loudr-1
```

They can also be re-exported from the checkpoint with
`tools/export_coreml.py`; see [the Apple page](../platforms/apple.md).

## Add it to your project

`Package.swift` is at the repository root, because SwiftPM reads a manifest
from the root or not at all. The sources live under `swift/`, beside
`python/`, `go/`, `rust/` and `js/`.

```swift
.package(url: "https://github.com/loudreader/loudkit", from: "0.1.0")
```

Pin the released version so a build does not move under you.

There are two products:

```swift
.product(name: "LoudKit", package: "loudkit"),      // the engine
.product(name: "LoudKitText", package: "loudkit"),  // the text funnel alone
```

`LoudKitText` has no CoreML dependency and no weights. It is the funnel alone:
numbers, dates, acronyms, twelve languages, the Polish respelling lexicon. At
400 KB it links into anything, including a command-line tool that never
synthesises.

## Speak

```swift
import Foundation
import LoudKit

let engine = try Engine.load(
    checkpoint: URL(fileURLWithPath: "loudr-1/loudr-1.safetensors"))
let voice = try VoiceProfile.load(
    url: URL(fileURLWithPath: "loudr-1/voices/joe.safetensors"))

let result = try engine.synthesize("Hello from Swift.", voice: voice, seed: 7)
try result.save(to: URL(fileURLWithPath: "hello.wav"))
```

`Engine.load` takes `coremlAssets:` if the packages are not beside the
checkpoint, and `execution:` for device placement (see below). `seed:` is the
same seed the other four take and gives the same audio.

## Long text, and streaming

```swift
// One result for the whole passage.
let whole = try engine.synthesizeLong(passage, voice: voice, seed: 7)

// Or a chunk at a time, so playback starts before generation finishes.
try engine.stream(passage, voice: voice, seed: 7) { chunk in
    player.enqueue(chunk.audio)
    return true          // false stops the generation
}
```

Both take `shouldCancel:`. Return `true` and the decode loop stops within a
step. Your playback layer must also discard audio it has already queued. All
five implementations stream in-process; only Python streams over a transport.

## Where the stages run

```swift
var execution = ExecutionConfig()
execution.estimatorComputeUnits = .all     // the ANE citizen
execution.encoderComputeUnits = .cpuOnly   // fp32 graph, stays on CPU
let engine = try Engine.load(
    checkpoint: URL(fileURLWithPath: "loudr-1/loudr-1.safetensors"),
    execution: execution)
```

**This port is slower than the Python engine.** End to end on an M3 Pro, the
whole pipeline runs the third benchmark passage at 2.28x real time warm, median
of three, and 2.11x cold. The Python engine on `--device mps`, same machine and
same passage, runs the same call at 3.25x warm and 3.50x cold. Its generator is
a native fp32 implementation whose attention runs through BLAS. The renderer,
which is the CoreML half, runs at the same speed here as anywhere. Use this
package for an Apple target that cannot host Python; use the Python engine when
speed decides. The figures are in [apple.md](../platforms/apple.md).

The token generator runs natively on the CPU in fp32, and that placement is
measured rather than assumed. The autoregressive stage is faster on CPU than on
GPU or ANE at batch one, and fp32 is the precision the conformance fixture
declares. The estimator is the stage worth putting on the Neural Engine.

`ExecutionConfig` exposes the knob and holds no opinion about the tuned
per-stage placement. `docs/platforms/apple.md` carries the measurements instead
of a recipe.

## What Swift does that the others do not

**Bit parity on the funnel, verified in the fixture.** `LoudKitText` is the
implementation the Python, Go, Rust and JS funnels were ported *from*, so its
tests are the reference the other four are measured against: 122 shared cases,
`swift test`.

**Enrollment on device.** `Enrollment` turns ten seconds of audio into a
profile without leaving the phone. It needs the enrollment packages, which are
a separate export again.

## Where it is weaker

- **Coarser errors.** One enum with five cases against Python's seven classes.
  `docs/reference/errors.md` has the matrix, including the seventeen conditions
  all five detect and what each of them tells you.
- **No transport.** No HTTP server, no gRPC, no MCP. Those are Python's. A
  Swift app that wants them talks to `loudkit serve` over the network, which
  now includes an OpenAI-compatible route (guide 4).
- **Two lexicon channels.** `ChatterboxAssets` first, then `Bundle.module`. An
  app shipping its own `pl_en_respell.json` overrides the packaged one. The
  grammar digest hashes whichever file it actually read. The two used to
  disagree, and the fingerprint then described a file that was not in use.

## Next

This is the last guide. Guides 1 through 4 are written in Python but describe
behaviour that applies to all five implementations.
