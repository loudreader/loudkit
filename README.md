<p align="center">
  <img src="https://raw.githubusercontent.com/loudreader/loudkit/main/assets/logo-wordmark.png" alt="LoudKit" width="640">
</p>

# loudkit

**Natural-sounding text-to-speech that runs on your own hardware.**

[![CI](https://github.com/loudreader/loudkit/actions/workflows/ci.yml/badge.svg)](https://github.com/loudreader/loudkit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/loudreader/loudkit/blob/main/LICENSE)
[![Model](https://img.shields.io/badge/Model-loudr--1-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/loudreader/loudr-1)
[![Model](https://img.shields.io/badge/Model-loudr--1--turbo-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/loudreader/loudr-1-turbo)
[![Spaces](https://img.shields.io/badge/Spaces-Try%20it-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/jer3mi/loudkit)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/loudreader/loudkit/blob/main/notebooks/loudkit_quickstart.ipynb)

Twenty-eight voices in ten languages, voice cloning from about ten seconds of
audio, and native SDKs for Python, Swift, Go, Rust and TypeScript. Download the
model once and run offline, with no account, telemetry or usage bill.

loudkit is the speech engine inside [LoudReader](https://loudreader.io), a
reading app that speaks articles, PDFs and books on device. The engine is here
under Apache-2.0 for anyone who wants to build with it directly.

[**Hear the voices**](https://loudreader.github.io/loudkit/demo/) |
[**Try it in the browser**](https://huggingface.co/spaces/jer3mi/loudkit) |
[**Open in Colab**](https://colab.research.google.com/github/loudreader/loudkit/blob/main/notebooks/loudkit_quickstart.ipynb) |
[Model](https://huggingface.co/loudreader/loudr-1) |
[Documentation](https://loudreader.github.io/loudkit/)

## Hear a voice in one command

```bash
pip install "loudkit[torch,audio,hub]"
loudkit speak --voice joe "Hello from loudkit." --play
```

The first run downloads the 747 MB model and the 28 voices, then it runs
offline. `--play` uses the system player; add `-o hello.wav` to keep the file.
Every other voice works the same way: `--voice kathleen`, `--voice dave`.

## The same from Python

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
voice = engine.voice("joe")

engine.synthesize("Hello from loudkit.", voice, seed=7).save("hello.wav")
```

`engine.voices()` lists the 28 names. `synthesize` takes text of any
length, and `engine.stream(...)` delivers the same audio sentence by sentence
so playback can start early. All 28 voices, including Henry, Oliver, Oscar and
Sophie, are included in both model downloads and load by name.

[Getting started](https://github.com/loudreader/loudkit/blob/main/docs/guides/01-getting-started.md)
is the page to read next.

## The same in Swift, Go, Rust and TypeScript

Every port loads the model by name, fetches and verifies it on the first call,
and runs offline after that. The snippets need version 0.1.1 of each package.

**Swift** (CoreML, macOS 14 or iOS 17). Add
`.package(url: "https://github.com/loudreader/loudkit", from: "0.1.1")` to
`Package.swift`. [Guide](https://github.com/loudreader/loudkit/blob/main/docs/guides/10-swift.md).

```swift
import LoudKit

let engine = try await Engine.load("loudreader/loudr-1")
let voice = try engine.voice(named: "joe")
try engine.synthesize("Hello from loudkit.", voice: voice, seed: 7).saveWav("hello.wav")
```

**Go** (ONNX Runtime). `go get github.com/loudreader/loudkit/go@v0.1.1`.
[Guide](https://github.com/loudreader/loudkit/blob/main/docs/guides/08-go.md).

```go
eng, err := loudkit.Load("loudreader/loudr-1")
if err != nil { log.Fatal(err) }
defer eng.Close()
v, err := eng.Voice("joe")
if err != nil { log.Fatal(err) }
res, err := eng.Synthesize("Hello from loudkit.", v, loudkit.Options{Seed: 7})
if err != nil { log.Fatal(err) }
if err := res.SaveWav("hello.wav"); err != nil { log.Fatal(err) }
```

**Rust** (ONNX Runtime). `cargo add loudkit@0.1.1`.
[Guide](https://github.com/loudreader/loudkit/blob/main/docs/guides/09-rust.md).

```rust
use loudkit::{Engine, Options};

let mut engine = Engine::load("loudreader/loudr-1")?;
let voice = engine.voice("joe")?;
let options = Options { seed: 7, ..Default::default() };
engine.synthesize("Hello from loudkit.", &voice, &options)?.save_wav("hello.wav")?;
```

**TypeScript** (ONNX Runtime, Node 20). `npm install loudkit@0.1.1`.
[Guide](https://github.com/loudreader/loudkit/blob/main/docs/guides/07-js-ts.md).

```typescript
import { Engine } from "loudkit";

const engine = await Engine.load("loudreader/loudr-1");
const voice = engine.voice("joe");
(await engine.synthesize("Hello from loudkit.", voice, { seed: 7 })).saveWav("hello.wav");
await engine.close();
```

Go and Rust need `libonnxruntime` on the machine (`brew install onnxruntime`,
or a build from the ONNX Runtime releases); the guides say where each port
looks for it. The same text, voice and seed give the same speech tokens in all
five languages.

## Two models

`loudreader/loudr-1` is the default and runs on every backend and in every
language. `loudreader/loudr-1-turbo` is faster, at a small cost in
naturalness, and in 0.1.1 runs in all five SDKs. The string passed
to `load` is the whole choice:
[Choosing a model](https://github.com/loudreader/loudkit/blob/main/docs/guides/11-choosing-a-model.md).

## Voices

[The voice gallery](https://loudreader.github.io/loudkit/demo/) plays all
28 voices next to the recording each was enrolled from. English, Spanish,
French, German, Italian, Polish, Portuguese, Dutch, Swedish and Danish, ten English voices and two for each other language. [VOICES.md](https://github.com/loudreader/loudkit/blob/main/VOICES.md)
records the source, licence and consent basis of every one; they come from
recordings donated for speech technology or from CC0 and CC-BY corpora.

We have evaluated English by ear and do not speak the other nine languages
well enough to judge them. If you do, please listen and tell us what sounds
wrong.

## Clone a voice

From a recording you own or have permission to use, five to ten seconds of one
speaker:

```bash
pip install "loudkit[torch,audio,enroll,hub]"
loudkit clone my-recording.wav --checkpoint loudreader/loudr-1 --name my-voice --language en
loudkit speak --voice voices/my-voice.safetensors "Now in a cloned voice." -o cloned.wav
```

The result is a portable profile of about 150 KB, not another copy of the
model. In Python the same path is `lk.enroll(...)`; every port has `enroll`.
See [Cloning a voice](https://github.com/loudreader/loudkit/blob/main/docs/guides/03-cloning-a-voice.md)
and [Responsible use](https://github.com/loudreader/loudkit/blob/main/RESPONSIBLE_USE.md).

## Measured speed

| path | hardware | loudr-1 | loudr-1-turbo |
|---|---|---:|---:|
| split PyTorch engine\* | Apple M3 Pro | 3.29x | 5.77x |
| Swift, native generator plus CoreML renderer | Apple M3 Pro | 2.49x | 3.44x |
| ONNX Runtime, CPU provider | Apple M3 Pro | 1.14x | 1.59x |
| PyTorch with CUDA graphs | RTX 3090 | 8.55x | 13.05x |
| PyTorch with CUDA graphs | Jetson Orin Nano | 1.85x | 2.50x |

\* "Split" describes device placement, not a different model or checkpoint.
The token generator runs on the CPU while the mel and vocoder renderer runs on
the Apple GPU through MPS. Adjacent windows can overlap across the two devices.

Higher is faster, and 1.0x means real time. Every row was measured on 0.1.1
(2026-09-06, both models): the Apple rows on one laptop in ordinary use, the
NVIDIA rows on the named parts, with the same passage, voice and seed. The
A100, L4 and T4 rows are on the
[benchmark page](https://github.com/loudreader/loudkit/blob/main/docs/benchmarks.md).

For batched workloads, the token generator reaches 16.7x aggregate throughput
at batch 1 and 57.3x at batch 64 on the RTX 3090 with loudr-1, and 42.0x to
155.0x with loudr-1-turbo. The highest measured result is 223.6x, turbo on an
A100 at batch 64 (85.3x with loudr-1), measured on 0.1.1. They are
generator-only throughput numbers, not single-request latency or end-to-end
RTF. Full commands, hardware and caveats are in
[Benchmarks](https://github.com/loudreader/loudkit/blob/main/docs/benchmarks.md).

## Integrations

For a process that keeps one engine warm rather than paying the load cost per
call. The contracts are in
[Server and agents](https://github.com/loudreader/loudkit/blob/main/docs/guides/04-server-and-agents.md).

- `loudkit serve`: an HTTP server with loudkit's own routes and an
  OpenAI-compatible `/v1/audio/speech`, streaming over Server-Sent Events.
- `loudkit serve --grpc`: the same engine behind a typed schema with
  backpressure.
- `loudkit serve --mcp`: an MCP server on stdio for agent hosts (preview).
- A Speech Dispatcher module for Linux screen readers, in `integrations/`.
- [Docker images and compose](https://github.com/loudreader/loudkit/blob/main/docs/platforms/docker.md)
  for the server.

WAVs saved from Python and the server's replies carry a machine-readable note
about how the audio was made, naming the model, voice, seed and backend and
carrying a checksum that ties the note to the sound; the Go, Rust, JS and Swift
ports write plain PCM. loudkit writes that note and `loudkit verify` reads it.
It is not C2PA Content Credentials: nothing signs it and no C2PA tool reads it.

## Scope

loudkit is an inference toolbox, not a hosted speech platform. It does not
provide accounts, billing, multi-tenancy, model training or an emotion control.
The local server expects you to provide any public-facing authentication, rate
limits and TLS. It will not help with undisclosed impersonation, bypassing voice
authentication or stripping Content Credentials from generated audio.
[SUPPORTED.md](https://github.com/loudreader/loudkit/blob/main/SUPPORTED.md)
states the boundary.

## Documentation

The [documentation index](https://github.com/loudreader/loudkit/blob/main/docs/README.md)
lists the twelve pages a user needs: getting started, one page per language,
choosing a model, cloning, long text and streaming, servers, troubleshooting
and the two model cards. The
[model card](https://github.com/loudreader/loudkit/blob/main/docs/MODEL_CARD.md)
covers lineage, limits and what each download contains.

## Licence

The code and the loudr-1 release are
[Apache-2.0](https://github.com/loudreader/loudkit/blob/main/LICENSE). The
tokenizer and speaker encoder keep their upstream MIT licence from Chatterbox.
[`NOTICE`](https://github.com/loudreader/loudkit/blob/main/NOTICE) lists every
upstream component and licence.
