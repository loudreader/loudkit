# loudkit for Swift

Text to speech on CoreML, on macOS 14 or iOS 17. No torch, no ONNX Runtime,
no Python.

## Hello

`Package.swift` is at the repository root.

```swift
.package(url: "https://github.com/loudreader/loudkit", from: "0.1.1")
```

Two products: `LoudKit` is the engine, `LoudKitText` is the text funnel alone
(numbers, dates, acronyms, twelve languages) with nothing to download.

`main.swift`:

```swift
import Foundation
import LoudKit

let engine = try await Engine.load("loudreader/loudr-1")
let result = try engine.synthesize("Hello from loudkit.", voice: engine.voice(named: "joe"), seed: 7)
try result.saveWav("hello.wav")
print("hello.wav: \(String(format: "%.2f", result.duration))s")
```

The first run downloads the model files into
`~/Library/Caches/loudkit/loudreader--loudr-1` (`$LOUDKIT_CACHE` moves it),
the directory the Go, Rust and JS ports share, and checks every file against
the release's own `SHA256SUMS`; later runs read what is there.
`engine.voiceNames` names the 28 voices.

The snippets on this page need loudkit 0.1.1. From a checkout, `swift run
Hello` runs `swift/Examples/Hello/main.swift`, which is this file.

Both `loudr-1` and `loudr-1-turbo` use this API in 0.1.1. Change the model
name to switch; keep the same voice profile. A local release directory works
as well as a published model name.


## The rest of the front door

```swift
import Foundation
import LoudKit

try await LoudKit.download(repo: "loudreader/loudr-1", to: URL(fileURLWithPath: "loudr-1"))
try await LoudKit.download(repo:to:revision:cloning:progress:)   // pin a revision, add cloning
try await Engine.load("loudr-1")                       // a directory or a repo id
engine.voiceNames                                      // the names in the release
try engine.voice(named: "joe")                         // one of them
try engine.synthesize(text, voice: voice, seed: 7, language: nil, speed: 1.0, previousTokens: nil)
try engine.stream(text, voice: voice, seed: 7) { chunk in play(chunk.audio); return true }
let mine = try await engine.enroll(contentsOf: URL(fileURLWithPath: "me.m4a"), name: "mine", language: "en")
try mine.save("mine.safetensors")                      // a portable profile
try VoiceProfile.load(url: URL(fileURLWithPath: "mine.safetensors"))
try result.saveWav("hello.wav")                        // 16-bit PCM
```

`synthesize` takes text of any length: it splits at sentence boundaries and
joins the audio. Every argument after the voice has a default: seed 0, the
voice's own language, normal speed. `stream` hands out chunks as they are
made; return `false` to stop, or pass `shouldCancel:` to stop within one
decode step. `enroll` on an engine loaded by repo id fetches the enrollment
packages once; a directory of your own needs `cloning: true`. The reader
takes anything AVFoundation opens. `Engine.load(checkpoint:coremlAssets:)`
opens a layout of your own.

## Where the stages run

`ExecutionConfig` places each CoreML stage. The token generator runs natively
on the CPU in fp32, which is the measured-right placement for an
autoregressive stage at batch one; the renderer's middle stage is the one
worth putting on the Neural Engine. `docs/platforms/apple.md` carries the
measurements.

## Build and test

```bash
swift build
swift test            # weight-free conformance vectors; the engine tests skip without the checkpoint
```
