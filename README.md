<p align="center">
  <img src="https://raw.githubusercontent.com/loudreader/loudkit/main/assets/logo-wordmark.png" alt="LoudKit" width="640">
</p>

# loudkit

**Natural-sounding text-to-speech for products, scripts and experiments.**

loudkit runs on your own hardware. It includes twenty voices across ten
languages, voice cloning from about ten seconds of audio, and native SDKs for
Python, Swift, Go, Rust and TypeScript. Download the model once and run offline
with no account, telemetry or usage bill.

[![CI](https://github.com/loudreader/loudkit/actions/workflows/ci.yml/badge.svg)](https://github.com/loudreader/loudkit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/loudreader/loudkit/blob/main/LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/loudreader/loudkit/blob/main/notebooks/loudkit_quickstart.ipynb)

[**Hear all 20 voices**](https://loudreader.github.io/loudkit/demo/) |
[**Open in Colab**](https://colab.research.google.com/github/loudreader/loudkit/blob/main/notebooks/loudkit_quickstart.ipynb) |
[Model](https://huggingface.co/loudreader/loudr-1) |
[Documentation](https://loudreader.github.io/loudkit/)

- **20 voices in 10 languages:** English, Spanish, French, German, Italian,
  Polish, Portuguese, Dutch, Swedish and Danish.
- **Voice cloning:** a reusable profile of about 150 KB from roughly ten
  seconds of audio.
- **Runs locally:** PyTorch, ONNX Runtime and CoreML.
- **Five native SDKs:** Python, Swift, Go, Rust and TypeScript.

## Make a WAV

```bash
pip install "loudkit[torch,audio,hub]"
```

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
voice = lk.voice("joe", repo="loudreader/loudr-1")

engine.synthesize("Hello from loudkit.", voice, seed=7).save("hello.wav")
```

From the shell:

```bash
loudkit speak --checkpoint loudreader/loudr-1 --voice joe \
  "Hello from loudkit." -o hello.wav
```

The first run downloads the 747 MB synthesis checkpoint and the voices. Later
runs use the local cache. See
[Getting started](https://github.com/loudreader/loudkit/blob/main/docs/guides/01-getting-started.md)
for revision pinning, local directories and a complete first-run walkthrough.

## Voices

[Open the voice gallery](https://loudreader.github.io/loudkit/demo/) to compare
all twenty shipped voices. Each generated sample is shown next to the enrollment
reference used to build its profile.

The [voice roster](https://github.com/loudreader/loudkit/blob/main/VOICES.md)
records the source, licence and consent basis for every profile. The voices come
from recordings donated for speech technology or from CC0 and CC-BY speech
corpora. No scraped celebrity voices ship with the project.

We have evaluated English by ear. We do not speak the other nine languages well
enough to judge their naturalness reliably. If you do, please listen and tell us
what sounds good or wrong. Reports from native speakers are especially welcome.

## Clone a voice

Use a recording that you own or have permission to use:

```bash
pip install "loudkit[torch,audio,enroll,hub]"

loudkit clone my-recording.wav --checkpoint loudreader/loudr-1 \
  --name my-voice --language en
loudkit speak --checkpoint loudreader/loudr-1 \
  --voice voices/my-voice.safetensors "Now in a cloned voice." -o cloned.wav
```

The result is a portable profile, not another copy of the model. Python users
can call the same path through `lk.enroll`. See
[Cloning a voice](https://github.com/loudreader/loudkit/blob/main/docs/guides/03-cloning-a-voice.md)
for recording advice and
[Responsible use](https://github.com/loudreader/loudkit/blob/main/RESPONSIBLE_USE.md)
for the consent rules.

## Download only the runtime you need

The Hugging Face repository contains every supported format. The downloader
selects one runtime and leaves the rest behind.

| use case | command |
|---|---|
| Python with PyTorch | `loudkit download loudreader/loudr-1 --for torch` |
| Python, JS, Go or Rust with ONNX Runtime | `loudkit download loudreader/loudr-1 --for onnx --local-dir loudr-1` |
| Swift or Python with CoreML | `loudkit download loudreader/loudr-1 --for coreml --local-dir loudr-1` |

Add `--with-cloning` if that installation also needs enrollment. The
[model card](https://huggingface.co/loudreader/loudr-1) lists exact download
sizes and files.

### Measured speed

| path | hardware | real-time factor |
|---|---|---:|
| PyTorch with CUDA graphs | RTX 3090 | 7.47x |
| PyTorch with CUDA graphs | Jetson Orin Nano | 1.83x |
| split PyTorch engine\* | Apple M3 Pro | 3.43x |
| ONNX Runtime, CPU provider | Apple M3 Pro | 1.21x |
| PyTorch CPU reference | Apple M3 Pro | 0.33x |

\* "Split" describes device placement, not a different model or checkpoint.
The token generator runs on the CPU while the mel and vocoder renderer runs on
the Apple GPU through MPS. Adjacent windows can overlap across the two devices.

Higher is faster, and 1.0x means real time. CPU performance depends heavily on
the runtime: ONNX Runtime is faster than real time on the measured M3 Pro, while
the PyTorch CPU reference path is not.

For batched workloads, the token generator reaches 20.1x aggregate throughput
at batch 1 and 153.1x at batch 64 on the RTX 3090. The highest measured result
is 170.8x on an A100 at batch 64. These are generator-only throughput numbers,
not single-request latency or end-to-end RTF. Full commands, hardware and
caveats are in
[Benchmarks](https://github.com/loudreader/loudkit/blob/main/docs/benchmarks.md).

## SDKs and local APIs

Python is the reference implementation. Swift, Go, Rust and TypeScript are
native ports checked against the same conformance fixtures.

| SDK | runtime | install |
|---|---|---|
| Python | PyTorch, ONNX Runtime or CoreML | `pip install "loudkit[torch]"` plus the extra for your backend |
| Swift | CoreML | Swift Package Manager, from `0.1.0` |
| Go | ONNX Runtime | `go get github.com/loudreader/loudkit/go` |
| Rust | ONNX Runtime through `ort` | `cargo add loudkit` |
| TypeScript | `onnxruntime-node` | `npm install loudkit` |

For processes that should keep one engine warm, loudkit also includes:

- a local HTTP server with an OpenAI-compatible speech endpoint;
- a small MCP preview server;
- typed gRPC streaming with backpressure;
- a Speech Dispatcher module for Linux screen readers.

[Servers and agents](https://github.com/loudreader/loudkit/blob/main/docs/guides/04-server-and-agents.md)
documents the exact contract of each integration.

## Output and reproducibility

Synthesis results support chunk and estimated word timestamps, pitch-preserving
speed from 0.5x to 2.0x, and C2PA Content Credentials. Saved WAVs and server
responses carry an unsigned claim-only C2PA manifest by default. It records the
model fingerprint, checkpoint and voice digests, seed, backend and audio digest.

For a fixed build, device and backend, the same text, voice and seed produce the
same waveform. Different devices or backends can differ at the waveform level
because of floating-point arithmetic. The precise guarantees and measurements
live in the
[Identity contract](https://github.com/loudreader/loudkit/blob/main/docs/reference/IDENTITY-CONTRACT.md)
and
[Measured parity](https://github.com/loudreader/loudkit/blob/main/docs/parity-measured.md).

## Scope

loudkit is an inference toolbox, not a hosted speech platform. It does not
provide accounts, billing, multi-tenancy, model training or an emotion control.
The local server expects you to provide any public-facing authentication, rate
limits and TLS.

The project ships twenty permitted voice profiles and the code to enroll your
own. It will not help with undisclosed impersonation, bypassing voice
authentication or removing provenance from generated audio.

## Documentation

- [Getting started](https://github.com/loudreader/loudkit/blob/main/docs/guides/01-getting-started.md):
  first synthesis, caches and local files.
- [Guides](https://github.com/loudreader/loudkit/tree/main/docs/guides): long
  form, cloning, servers and every SDK.
- [Model card](https://github.com/loudreader/loudkit/blob/main/docs/MODEL_CARD.md):
  model lineage, limitations and release layout.
- [Supported in v0.1](https://github.com/loudreader/loudkit/blob/main/SUPPORTED.md):
  the public compatibility boundary.
- [Troubleshooting](https://github.com/loudreader/loudkit/blob/main/docs/reference/troubleshooting.md):
  common failures and concrete fixes.
- [Architecture](https://github.com/loudreader/loudkit/blob/main/docs/reference/ARCHITECTURE.md):
  components and package layout for contributors.

## Licence

The code and loudr-1 release are
[Apache-2.0](https://github.com/loudreader/loudkit/blob/main/LICENSE). The
tokenizer and speaker encoder retain their upstream MIT licence from Chatterbox.
[`NOTICE`](https://github.com/loudreader/loudkit/blob/main/NOTICE) lists every
upstream component and licence. The voice encoder's chain is in its
[provenance record](https://github.com/loudreader/loudkit/blob/main/docs/PROVENANCE-voice-encoder.md).
