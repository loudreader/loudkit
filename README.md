<p align="center">
  <img src="https://raw.githubusercontent.com/loudreader/loudkit/main/assets/logo-wordmark.png" alt="loudkit" width="640">
</p>

# loudkit

**Text-to-speech that runs on your own hardware.**

[![CI](https://github.com/loudreader/loudkit/actions/workflows/ci.yml/badge.svg)](https://github.com/loudreader/loudkit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/loudreader/loudkit/blob/main/LICENSE)
[![Model](https://img.shields.io/badge/Model-loudr--1-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/loudreader/loudr-1)
[![Model](https://img.shields.io/badge/Model-loudr--1--turbo-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/loudreader/loudr-1-turbo)
[![Spaces](https://img.shields.io/badge/Spaces-Try%20it-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/jer3mi/loudkit)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/loudreader/loudkit/blob/main/notebooks/loudkit_quickstart.ipynb)
[![loudkit MCP server score on Glama](https://glama.ai/mcp/servers/loudreader/loudkit/badges/score.svg)](https://glama.ai/mcp/servers/loudreader/loudkit)

loudkit has 28 voices in ten languages, clones a voice from about ten seconds
of audio, and has native SDKs for Python, Swift, Go, Rust and TypeScript. After
the first model download it runs offline. It needs no account, sends no
telemetry and has no usage fees.

[LoudReader](https://loudreader.io), a reading app, uses loudkit as its speech
engine. loudkit is licensed under Apache-2.0.

[**Hear the voices**](https://loudreader.github.io/loudkit/demo/) |
[**Try the online demo**](https://huggingface.co/spaces/jer3mi/loudkit) |
[**Open in Colab**](https://colab.research.google.com/github/loudreader/loudkit/blob/main/notebooks/loudkit_quickstart.ipynb) |
[Model](https://huggingface.co/loudreader/loudr-1) |
[Documentation](https://loudreader.github.io/loudkit/)

## Quickstart

```bash
pip install "loudkit[torch,audio,hub]"
loudkit speak --voice joe "Hello from loudkit." --play
```

The first run downloads the 747 MB model and the 28 voices. After that, it runs
offline. `speak` writes `out.wav` by default, and `--play` plays it with the
system player. Use `-o hello.wav` to choose the file name. Other English voices
work the same way, for example `--voice kathleen` or `--voice henry`. A voice
reads text in its own language unless you pass `--language`.

## Python

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
voice = engine.voice("joe")

engine.synthesize("Hello from loudkit.", voice, seed=7).save("hello.wav")
```

`engine.voices()` lists the 28 names. All 28 voices are in both model downloads
and load by name. `synthesize` splits long text into chunks and returns one
waveform. `engine.stream(...)` yields the same audio chunk by chunk, so
playback can start early.

Read [Getting started](https://github.com/loudreader/loudkit/blob/main/docs/guides/01-getting-started.md)
next.

## Swift, Go, Rust and TypeScript

Each port loads the model by name. The first call downloads and verifies it,
and later calls run offline. The snippets need version 0.1.1 of each package.

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
looks for it. All five implementations pass the same conformance fixture. The
[identity contract](https://github.com/loudreader/loudkit/blob/main/docs/reference/IDENTITY-CONTRACT.md)
states where their speech tokens match and where they can differ.

## Two models

`loudreader/loudr-1` is the default. `loudreader/loudr-1-turbo` is faster and
sounds slightly less natural. Both run on every backend and in all five SDKs.
Select the model with the string you pass to `load`. See
[Choosing a model](https://github.com/loudreader/loudkit/blob/main/docs/guides/11-choosing-a-model.md).

## Voices

[The voice gallery](https://loudreader.github.io/loudkit/demo/) plays all
28 voices next to the recording each was enrolled from. The ten languages are
English, Spanish, French, German, Italian, Polish, Portuguese, Dutch, Swedish
and Danish. English has ten voices, and each other language has two.
[VOICES.md](https://github.com/loudreader/loudkit/blob/main/VOICES.md) records
the source, licence and consent basis of each voice. The voices come from
recordings donated for speech technology, or from CC0 and CC-BY corpora.

Only English has been evaluated by ear. The other nine languages have automated
checks but no native-speaker review. If you speak one of them, please listen
and report what sounds wrong.

## Clone a voice

Use five to ten seconds of one speaker, from a recording you own or have
permission to use:

```bash
pip install "loudkit[torch,audio,enroll,hub]"
loudkit clone my-recording.wav --checkpoint loudreader/loudr-1 --name my-voice --language en
loudkit speak --voice voices/my-voice.safetensors "Now in a cloned voice." -o cloned.wav
```

The result is a portable voice profile of about 150 KB. By default,
`loudkit clone` cuts the recording at a pause and pads its end with silence.
`lk.enroll(...)` in Python and `enroll` in each port keep the recording as
given. To get the `loudkit clone` result from Python, pass
`end_in_silence=True` to `lk.enroll`. See
[Cloning a voice](https://github.com/loudreader/loudkit/blob/main/docs/guides/03-cloning-a-voice.md)
and [Responsible use](https://github.com/loudreader/loudkit/blob/main/RESPONSIBLE_USE.md).

## Measured speed

| path | hardware | loudr-1 | loudr-1-turbo |
|---|---|---:|---:|
| split PyTorch engine\* | Apple M3 Pro | 3.29x | 5.77x |
| Swift, native generator plus CoreML renderer | Apple M3 Pro | 2.49x | 3.44x |
| ONNX Runtime, CPU provider | Apple M3 Pro | 1.14x | 1.59x |
| PyTorch with CUDA graphs | RTX 3090 | 8.55x | 13.05x |
| PyTorch with CUDA graphs | Jetson Orin Nano Super, 25 W | 1.85x | 2.50x |

\* "Split" is a device placement of the same model. The token generator runs on
the CPU, and the renderer (mel decoder and vocoder) runs on the Apple GPU
through MPS. Adjacent windows can overlap across the two devices.

Higher is faster, and 1.0x is real time. Every row was measured on 0.1.1 on
2026-09-06, for both models, with the same passage, voice and seed. The Apple
rows come from one M3 Pro laptop in ordinary use, with no isolation from
background load. The A100, L4 and T4 rows are on the
[benchmark page](https://github.com/loudreader/loudkit/blob/main/docs/benchmarks.md).

For batched workloads, the token generator reaches 16.7x aggregate throughput
at batch 1 and 57.3x at batch 64 on the RTX 3090 with loudr-1, and 42.0x to
155.0x with loudr-1-turbo. The highest measured result is 223.6x, turbo on an
A100 at batch 64 (85.3x with loudr-1), measured on 0.1.1. These figures come
from `research/bench_batch.py`, which replays a recorded token sequence through
the token generator. Sampling, prefill and rendering are outside the timed
loop, so the figures are neither single-request latency nor end-to-end RTF.
Full commands, hardware and caveats are in
[Benchmarks](https://github.com/loudreader/loudkit/blob/main/docs/benchmarks.md).

## Integrations

These keep one loaded engine across many requests. The contracts are in
[Server and agents](https://github.com/loudreader/loudkit/blob/main/docs/guides/04-server-and-agents.md).

- `loudkit serve`: an HTTP server with loudkit's own routes, Server-Sent Events
  streaming on `/v1/synthesize/stream`, and an OpenAI-compatible
  `/v1/audio/speech` that returns the whole utterance in one response.
  Agents that use OpenAI's speech API, such as Hermes Agent and OpenClaw,
  [connect with configuration only](https://github.com/loudreader/loudkit/blob/main/docs/guides/04-server-and-agents.md#connect-your-agent).
- `loudkit serve --grpc`: the same engine behind a typed schema with
  backpressure.
- `loudkit serve --mcp`: an MCP server on stdio for agent hosts (preview).
- A Speech Dispatcher module for Linux screen readers, in `integrations/`.
- [Docker images and compose](https://github.com/loudreader/loudkit/blob/main/docs/platforms/docker.md)
  for the server.

By default, a WAV saved from Python and a WAV reply from the server carry an
unsigned, machine-readable note on how the audio was made. The note names the
model, voice, seed and backend, and holds a checksum of the audio.
`loudkit verify` reads it. The Go, Rust, JS and Swift ports write plain PCM
without the note. The note is not C2PA: nothing signs it, and C2PA tools do not
read it.

## Scope

loudkit is an inference toolkit. It does not provide accounts, billing,
multi-tenancy, model training or emotion control. For a public deployment, put
your own authentication, rate limits and TLS in front of the server. The
maintainers close issues and pull requests that ask for help with undisclosed
impersonation, bypassing voice authentication, or stripping the note from
generated audio.
[SUPPORTED.md](https://github.com/loudreader/loudkit/blob/main/SUPPORTED.md)
lists what is supported.

## Documentation

The [documentation index](https://github.com/loudreader/loudkit/blob/main/docs/README.md)
lists the user guides: getting started (Python), one guide per port, choosing
a model, cloning, long text and streaming, servers, troubleshooting and the two
model cards. It also links the reference, platform and design pages. The
[model card](https://github.com/loudreader/loudkit/blob/main/docs/MODEL_CARD.md)
covers lineage, limits and what each download contains.

## Licence

The code and the loudr-1 and loudr-1-turbo releases are
[Apache-2.0](https://github.com/loudreader/loudkit/blob/main/LICENSE). The
tokenizer and speaker encoder keep their upstream MIT licence from Chatterbox.
[`NOTICE`](https://github.com/loudreader/loudkit/blob/main/NOTICE) lists every
upstream component and licence.
