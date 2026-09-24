# 10. Swift

The Swift port of loudkit runs on CoreML, on macOS 14 or iOS 17 and later.
CoreML is part of the operating system, so the package does not need Python,
PyTorch or ONNX Runtime.

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

The package has two products:

- `LoudKit`: speech synthesis.
- `LoudKitText`: text normalization only (numbers, dates, acronyms) in twelve
  languages. It needs no model download.

```swift
import Foundation
import LoudKit

let engine = try await Engine.load("loudreader/loudr-1")
let result = try engine.synthesize("Hello from loudkit.", voice: engine.voice(named: "joe"), seed: 7)
try result.saveWav("hello.wav")
```

The first run downloads the model files into
`~/Library/Caches/loudkit/loudreader--loudr-1` on macOS, or into the
application's Caches directory on iOS. Set `LOUDKIT_CACHE` to use
`$LOUDKIT_CACHE/loudreader--loudr-1` instead. The Go, Rust and JS ports use
the same directory. Each downloaded file is checked against the release's
`SHA256SUMS`. Later runs reuse the cache and fetch only the files that changed
when the repo's `main` branch moves. `engine.voiceNames` lists the 28 voices.

The examples on this page need loudkit 0.1.1. To run the example from a
repository checkout, run `swift run Hello`.

`loudr-1` and `loudr-1-turbo` use the same API. To switch models, change the
model name. The same voice profiles work with both models. `Engine.load` also
accepts a local release directory.


## Download to a local directory

```swift
let dir = try await LoudKit.download(
    repo: "loudreader/loudr-1", to: URL(fileURLWithPath: "loudr-1"), revision: "v0.1.1")
let engine = try Engine.load(bundle: dir)
```

`download` writes a receipt, `.loudkit-release.json`, into the directory. On
a later call:

- If the revision still resolves to the same commit, `download` fetches and
  hashes no weight file.
- If the revision moved, it keeps every file that matches the new
  `SHA256SUMS` and fetches the rest.
- An interrupted fetch resumes.

Pass a `progress:` closure to show download progress. Without one, `download`
writes one line per file to stderr. Pin `revision:` to a tag or commit for a
reproducible build. For repos under `loudreader/`, `release.json` must show
that the release passed its build checks. `download` reads it before it
fetches any weight file.

## Synthesize

`synthesize` splits long text into chunks at sentence boundaries. It gives
each chunk its own seed, conditions each chunk on the speech tokens at the end
of the chunk before it, and returns one `Result` that holds all the audio.
Every argument after the voice has a default.

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
text. `saveFloat32Wav(to:)` writes the same audio as float32. The conformance
tests use both.

## Streaming and barge-in

```swift
try engine.stream(text, voice: voice, seed: 7, shouldCancel: { stopped }) { chunk in
    player.enqueue(chunk.audio)   // chunk.timing starts at zero; add your own offset
    return true                   // false stops at the next chunk
}
```

`stream` passes each chunk to the closure when it is ready, so playback can
start before the passage is finished. `shouldCancel` is checked on every
decode step. When it returns true, the chunk in progress is discarded. Your
playback layer must also discard the audio it has already queued.

## Timestamps and speed

`result.chunks` is exact at the chunk level and an estimate at the word
level. Read [timestamps.md](../reference/timestamps.md) before you use the
word times. `speed` outside `[0.5, 2.0]` is refused; see
[speed.md](../reference/speed.md).

## Cloning a voice

```swift
let mine = try await engine.enroll(contentsOf: URL(fileURLWithPath: "me.m4a"), name: "mine", language: "en")
try mine.save("mine.safetensors")
```

On an engine loaded by repo id, the first `enroll` call fetches the three
enrollment packages into the model's cache directory. Later calls use the
cached packages. For an engine loaded from a local directory, fetch the
packages first: call `LoudKit.download(repo:to:revision:cloning:progress:)`
with `cloning: true`.

Use five to ten seconds of clean speech. `enroll(contentsOf:)` reads any file
that AVFoundation opens, such as WAV, CAF, AIFF or m4a. For samples in
memory, get an enroller with `ModelBundle(directory:).enroller()` and call
`enroll(_:sampleRate:)`. It returns an `EnrolledVoice`; make a profile from
it with `VoiceProfile(_:name:language:)`. `VoiceProfile.load(url:)` loads a
saved profile.

## Where the stages run

The token generator runs as native code on the CPU in fp32. The renderer has
three CoreML stages. `ExecutionConfig` sets the compute units of each stage.
By default the middle stage runs on the CPU and the Neural Engine, and the
first and last stages run on the CPU. The defaults come from measurements on
an M3 Pro. On other hardware, other settings can be faster.
[platforms/apple.md](../platforms/apple.md) lists the `ExecutionConfig`
properties, their defaults and the measurements.

This port is slower than the Python engine. On an M3 Pro, the
whole pipeline runs the third benchmark passage at 2.49x real time with
loudr-1 and 3.44x with loudr-1-turbo, measured on 0.1.1 as medians of three
warm streams. On the same machine and passage, the Python engine with
`--device mps` runs at 3.29x and 5.77x.

This port's token generator is a native fp32 implementation whose attention
runs through BLAS. Its renderer runs the same CoreML graphs as the Python
`coreml` backend. Use this package for an Apple target that cannot run
Python. For the highest speed on a Mac, use the Python engine. The full
figures are in [apple.md](../platforms/apple.md).

## Explicit asset paths

`Engine.load(checkpoint:coremlAssets:)` loads a checkpoint and a directory of
`.mlpackage` or precompiled `.mlmodelc` stages. `ModelBundle(directory:)`
finds the paths in a release directory without loading the engine.

## Verify against the shared fixture

```bash
swift test                                                   # weight-free vectors
LOUDKIT_ASSET_ROOT=/path/to/assets swift test                # + the engine, against the checkpoint
```

The second command needs the checkpoint and the CoreML packages. It compares
the tokens from `synthesize` and `synthesizeWindow` with the fixture, chunk by
chunk.
