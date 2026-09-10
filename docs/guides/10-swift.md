# 10. Swift

The same engine as a Swift package over CoreML, for macOS 14 and iOS 17. No
Python, no torch, no ONNX Runtime: CoreML is already on the system.

## Hello

`Package.swift` is at the repository root.

```swift
.package(url: "https://github.com/loudreader/loudkit", from: "0.1.1")
```

For an existing app, also add the `LoudKit` product to its target. For a new
command-line application, create `Sources/Hello/main.swift` and this complete
`Package.swift`, then run `swift run Hello`:

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Hello",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/loudreader/loudkit", from: "0.1.1")
    ],
    targets: [
        .executableTarget(name: "Hello", dependencies: [
            .product(name: "LoudKit", package: "loudkit")
        ])
    ]
)
```

Two products: `LoudKit` is the engine, `LoudKitText` is the text funnel alone
(numbers, dates, acronyms, twelve languages) with nothing to download.

```swift
import Foundation
import LoudKit

let engine = try await Engine.load("loudreader/loudr-1")
let result = try engine.synthesize("Hello from loudkit.", voice: engine.voice(named: "joe"), seed: 7)
try result.saveWav("hello.wav")
```

The first run downloads the model files into
`~/Library/Caches/loudkit/loudreader--loudr-1` (`$LOUDKIT_CACHE` moves it),
the directory the Go, Rust and JS ports share, and checks every file against
the release's own `SHA256SUMS`; later runs read what is there.
`engine.voiceNames` names the 28 voices. The snippets need loudkit
0.1.1; from a checkout, `swift run Hello`.

Both `loudr-1` and `loudr-1-turbo` use this API in 0.1.1. Change the model
name to switch; keep the same voice profile. A local release directory works
as well as a published model name.


## A directory of your own

```swift
let dir = try await LoudKit.download(
    repo: "loudreader/loudr-1", to: URL(fileURLWithPath: "loudr-1"), revision: "v0.1.1")
let engine = try Engine.load(bundle: dir)
```

`download` writes a receipt, `.loudkit-release.json`; a later call whose
revision still resolves to the same commit fetches and hashes nothing, a moved
revision keeps every file that still hashes to the new `SHA256SUMS` and fetches
the rest, and an interrupted fetch resumes. Pass a `progress:` closure
to draw a bar; without one, a line per file goes to stderr. Pin `revision:`
for anything reproducible. Under `loudreader/`, `release.json` must say the
bundle passed the builder's gate, and it is checked before any weight moves.

## Synthesize

`synthesize` takes text of any length: it splits at sentence boundaries,
gives each chunk its own seed, carries the pitch contour across the joins and
returns one `Result`. Every argument after the voice has a default.

```swift
let result = try engine.synthesize(text, voice: voice,
    seed: 7,                            // 0 by default
    language: "pl",                     // nil: the voice's own
    speed: 1.25,                        // [0.5, 2.0], pitch preserved; 1.0 is an exact bypass
    previousTokens: earlier.tokens)     // continue an earlier result's pitch contour
result.audio         // [Float] at result.sampleRate
result.tokens        // the speech tokens
result.chunks        // where each chunk lands, and an estimate of each word
result.hitTokenCap   // generation stopped at the token cap: probably cut off
try result.saveWav("hello.wav")     // 16-bit PCM
```

`synthesizeWindow` renders exactly one model window and throws on longer
text; it is for the conformance harness. `saveFloat32Wav(to:)` writes the
same audio as float32, for the harness too.

## Streaming and barge-in

```swift
try engine.stream(text, voice: voice, seed: 7, shouldCancel: { stopped }) { chunk in
    player.enqueue(chunk.audio)   // chunk.timing starts at zero; add your own offset
    return true                   // false stops at the next chunk
}
```

`stream` hands out chunks as they are made, so playback starts before the
passage is finished. `shouldCancel` is polled on every decode step, and the
chunk being generated is discarded. Your playback layer must also discard
audio it has already queued.

## Timestamps and speed

`result.chunks` is exact at the chunk level and an estimate at the word
level; read [timestamps.md](../reference/timestamps.md) before building on
the word times. `speed` is refused outside `[0.5, 2.0]`; see
[speed.md](../reference/speed.md).

## Cloning a voice

```swift
let mine = try await engine.enroll(contentsOf: URL(fileURLWithPath: "me.m4a"), name: "mine", language: "en")
try mine.save("mine.safetensors")
```

The first `enroll` on an engine loaded by repo id fetches the three
enrollment packages into the same cache directory,
`~/Library/Caches/loudkit/loudreader--loudr-1`; later calls read them from
there. An engine loaded from a directory of your own needs them fetched with
`LoudKit.download(repo:to:cloning: true)`. Five to ten seconds of clean
speech is the input this was tuned for. The reader takes anything
AVFoundation opens: WAV, CAF, AIFF, m4a. `Enrollment.Enroller.enroll(_:sampleRate:)`
takes samples, for audio that never was a file. `VoiceProfile.load(url:)`
reads the profile back.

## Where the stages run

Each CoreML stage can be pinned to a compute unit. The defaults are the
measured optimum on Apple silicon; [platforms/apple.md](../platforms/apple.md)
has the knobs for hardware where they are not.

**This port is slower than the Python engine.** End to end on an M3 Pro, the
whole pipeline runs the third benchmark passage at 2.49x real time with
loudr-1 and 3.44x with loudr-1-turbo, medians of three warm streams. The
Python engine on `--device mps`, same machine and same passage, runs the same
call at 3.29x and 5.77x. Every figure in this paragraph was measured on 0.1.1.
Its generator is
a native fp32 implementation whose attention runs through BLAS. The renderer,
which is the CoreML half, runs at the same speed here as anywhere. Use this
package for an Apple target that cannot host Python; use the Python engine when
speed decides. The figures are in [apple.md](../platforms/apple.md).

The token generator runs natively on the CPU in fp32, which is the measured
placement for an autoregressive stage at batch one; the renderer's middle
stage is the one worth putting on the Neural Engine. `ExecutionConfig`
exposes the knob and holds no opinion; [apple.md](../platforms/apple.md)
carries the measurements and the stage names.

## Your own layout

`Engine.load(checkpoint:coremlAssets:)` opens a checkpoint beside a directory
of `.mlpackage` or precompiled `.mlmodelc` stages, and `ModelBundle(directory:)`
reads a release's paths without loading it.

## Verify against the shared fixture

```bash
swift test                                                   # weight-free vectors
LOUDKIT_ASSET_ROOT=/path/to/assets swift test                # + the engine, against the checkpoint
```

The second needs the checkpoint and the CoreML packages, and holds
`synthesize` and `synthesizeWindow` to the fixture's tokens, chunk by chunk.
