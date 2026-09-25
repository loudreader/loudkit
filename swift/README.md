# loudkit for Swift

Text to speech on CoreML, on macOS 14 or iOS 17 and later. It does not need
Python, PyTorch or ONNX Runtime.

## Hello

`Package.swift` is at the repository root.

```swift
.package(url: "https://github.com/loudreader/loudkit", from: "0.1.1")
```

The package has two products:

- `LoudKit`: speech synthesis.
- `LoudKitText`: text normalization only (numbers, dates, acronyms) in twelve
  languages. It needs no model download.

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
`~/Library/Caches/loudkit/loudreader--loudr-1` on macOS, or into the
application's Caches directory on iOS. Set `LOUDKIT_CACHE` to use
`$LOUDKIT_CACHE/loudreader--loudr-1` instead. The Go, Rust and JS ports use
the same directory. Each downloaded file is checked against the release's
`SHA256SUMS`. `engine.voiceNames` lists the 28 voices.

The examples on this page need loudkit 0.1.1. To run the example from a
repository checkout, run `swift run Hello`. `swift/Examples/Hello/main.swift`
is the program above.

`loudr-1` and `loudr-1-turbo` use the same API. To switch models, change the
model name. The same voice profiles work with both models. `Engine.load` also
accepts a local release directory.


## API overview

```swift
import Foundation
import LoudKit

try await LoudKit.download(repo: "loudreader/loudr-1", to: URL(fileURLWithPath: "loudr-1"))
try await LoudKit.download(repo: "loudreader/loudr-1", to: URL(fileURLWithPath: "loudr-1"),
                           revision: "v0.1.1", cloning: true)   // pin a revision, add cloning
try await Engine.load("loudr-1")                       // a local directory or a repo id
engine.voiceNames                                      // the voice names in the release
try engine.voice(named: "joe")                         // load one voice by name
try engine.synthesize(text, voice: voice, seed: 7, language: nil, speed: 1.0, previousTokens: nil)
try engine.stream(text, voice: voice, seed: 7) { chunk in play(chunk.audio); return true }
let mine = try await engine.enroll(contentsOf: URL(fileURLWithPath: "me.m4a"), name: "mine", language: "en")
try mine.save("mine.safetensors")                      // a portable profile
try VoiceProfile.load(url: URL(fileURLWithPath: "mine.safetensors"))
try result.saveWav("hello.wav")                        // 16-bit PCM
```

`synthesize` splits long text into chunks at sentence boundaries and joins
the audio in memory. Every argument after the voice has a default: seed 0, the
voice's own language, speed 1.0. `stream` passes each chunk to the closure
when it is ready. Return `false` to stop, or pass `shouldCancel:`, which is
checked on every decode step.

On an engine loaded by repo id, the first `enroll` call fetches the enrollment
packages. For a local directory, fetch them with `cloning: true`. `enroll`
reads any audio file that AVFoundation opens.
`Engine.load(checkpoint:coremlAssets:)` loads a checkpoint and CoreML
packages from the paths you give.

## Where the stages run

The token generator runs as native code on the CPU in fp32. `ExecutionConfig`
sets the compute units of the three CoreML stages of the renderer. By default
the middle stage runs on the CPU and the Neural Engine, and the other two run
on the CPU. `docs/platforms/apple.md` lists the properties, their defaults
and the measurements.

## Build and test

```bash
swift build
swift test            # weight-free conformance vectors; the engine tests skip without the checkpoint
```
