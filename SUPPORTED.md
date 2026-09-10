# What loudkit 0.1 supports

The scope, stated once, so the README cannot promise more than the project
keeps. The current release is 0.1.1. Anything not listed under *Supported* is
either *Experimental* (works, shipped, feedback wanted, may change) or *Out of
scope for 0.1* (not attempted). Removing something from *Supported* is a
breaking change. Promoting something into it is not.

## Supported

- **Local synthesis, offline.** No account, no network dependency, no
  per-character billing. The engine loads a packed checkpoint and runs on
  your hardware.
- **Two models.** `loudreader/loudr-1` and `loudreader/loudr-1-turbo` run in
  all five SDKs. Python supports PyTorch, ONNX Runtime and CoreML; Swift
  uses CoreML rendering, and Go, Rust and TypeScript use ONNX Runtime.
  Both models accept the same voice profiles.
- **Voice profiles as files.** A voice is a ~150 KB `.safetensors` you can
  copy, mail and keep; cloning one takes about ten seconds of audio, from the
  shell with `loudkit clone` or from any SDK with `enroll`.
- **Deterministic rendering.** Same text, voice, seed and build give a
  bit-identical waveform on a given device and backend, with the contract and
  its limits written down in
  [docs/reference/IDENTITY-CONTRACT.md](docs/reference/IDENTITY-CONTRACT.md).
- **Python as the reference implementation**, with the full API: `synthesize`
  at any length, streaming with token-level cancellation, `previous_tokens`
  continuation, speed control, WAVs that carry Content Credentials.
- **Backends:** torch (CPU, CUDA, MPS) as the reference execution, ONNX
  Runtime as the portable no-torch deployment, CoreML for Apple. Execution
  may differ in speed per backend, never in what is computed.
- **Compatibility ports:** Swift, Go, Rust and TypeScript implement the same
  algorithm against shared conformance fixtures. They track the reference:
  features land in Python first and reach the ports with the fixture that
  proves them.
- **The CLI**, eight commands: `speak`, `text`, `clone`, `voices`, `download`,
  `serve`, `verify`, `doctor`. `loudkit --help` lists these and no others.
  `serve` takes one of `--grpc` and `--mcp` to pick a transport other than
  HTTP; `--mcp` is preview and not part of this contract.
- **Transports over one synthesis path:** the local HTTP server (`loudkit
  serve`: `/v1`, SSE streaming, OpenAI-compatible route), gRPC (`loudkit
  serve --grpc`, typed schema and streaming backpressure, contract in
  [proto/loudkit.proto](proto/loudkit.proto)), and Speech Dispatcher. All
  answer byte-for-byte what the library answers, with one frozen error-code
  catalog ([docs/reference/errors.md](docs/reference/errors.md)).
- **Speech Dispatcher integration**: loudkit as a system voice for Linux
  screen readers, with the protocol codes and rate control handled properly.
- **Text handling in 12 languages** (numbers, dates, currency, units,
  abbreviations), verbalised from first principles and fuzz-tested.
- **Voices for ten languages** (en, pl, de, fr, nl, es, it, pt, sv, da), each
  with its donor or source, licence and consent basis recorded in
  [the voice roster](VOICES.md).
- **A note on how the audio was made**: a WAV saved from Python and every
  server reply carries an unsigned, machine-readable note naming the
  algorithm, checkpoint, voice profile, backend and seed; the Go, Rust, JS and
  Swift ports write plain PCM. It is loudkit's own and is not C2PA Content
  Credentials.

## Supported with stated limits

- **Language quality.** We evaluated English by ear. We do not speak the other
  nine shipped languages well enough to judge their naturalness reliably.
  Automated checks cover them, but those checks are not a native-speaker
  verdict. Feedback from native speakers is very welcome.
- **Waveforms are not identical across devices or backends.** Equivalence is
  measured and banded. Identity holds per build/device/backend.
- **CUDA graphs are an opt-in throughput mode** (`ExecutionConfig(cuda_graphs=True)`)
  in the identity contract's "equivalent" class: deterministic, not
  token-identical to eager.
- **`ChunkConfig.first_chunk_max_tokens` is Python-only for now.** Setting it
  changes where the first chunk ends and so changes the audio. The ports
  follow the usual fixture-first route.
- **Text preparation reads abbreviations, units and currency as written**,
  apart from the expansions the grammar tables list (for Polish: np., itd.,
  itp., tzn., tzw.). Numbers, ordinals, years, times and percentages are
  expanded; `2000 zł` is read as "dwa tysiące zł", `12 m²` as "dwanaście m
  dwa", and a numeric date such as `03/04` digit by digit in English.
  `loudkit text` prints exactly what will be spoken. Unit and currency reading
  with number agreement is planned for 0.1.2.
- **`loudkit clone` reads a local WAV or FLAC file.** No URLs, no recording,
  no denoising, no batch, no preview. By default it cuts the prompt at a
  suitable pause and pads its ending with silence; `--no-end-in-silence`
  keeps the recording as given. `--name` and `--language`
  are stated, not guessed. The API path, `enroll`, exists in every SDK's
  language (Python, Swift, Go, Rust, TypeScript). There is no MCP clone tool.

## Experimental

- **MCP over stdio** (`loudkit serve --mcp`). It answers the same bytes as
  every other transport and stays in the package, but its tool surface may
  change shape.
- The systemd unit and other deployment scaffolding in `integrations/`
  beyond the Speech Dispatcher module.
- The OpenAI-compatible route's parameter mapping outside the documented
  fields.

## Out of scope for 0.1

- Equal listening quality in every language.
- A production public server. `loudkit serve` is a local runtime with a
  hardened loopback default, not a multi-tenant service: no TLS, no quotas,
  no per-voice authorization.
- Expressiveness or emotion control.
- Robotics integrations (ROS, Wyoming), Android, WebAssembly, additional
  backends beyond the three shipped.
- Model training and fine-tuning.

## Model and platform evidence for 0.1.1

Both models implement the same public API. Implementation support is distinct
from model-backed validation on a particular machine:

| Runtime / platform | Base | Turbo |
|---|---|---|
| Python torch, ONNX CPU and CoreML / Apple silicon | measured | measured |
| Swift CoreML / macOS Apple silicon | measured | measured |
| Go, Rust, TypeScript ONNX CPU / macOS Apple silicon | measured | measured |
| Go, Rust ONNX CPU / Linux x86_64 | both fixture cases and long-form measured | both fixture cases and long-form measured |
| Python torch / CUDA | RTX 3090: two fixture cases, eager and CUDA graphs | RTX 3090: two fixture cases, eager and CUDA graphs |
| Python / ONNX CUDA | RTX 3090: two fixture cases | RTX 3090: two fixture cases |
| Go, Rust ONNX CPU / Jetson Orin | both fixture cases and long-form measured | both fixture cases and long-form measured |
| Python torch / Jetson Orin CUDA | two fixture cases, eager and CUDA graphs | two fixture cases, eager and CUDA graphs |
| Python ONNX CPU and CUDA / Jetson Orin | two fixture cases measured | two fixture cases measured |
| Go, Rust / ONNX CUDA | supported; validate target runtime | supported; validate target runtime |
| Windows / ONNX CPU or DirectML | supported; model-backed acceptance required | supported; model-backed acceptance required |
| Swift / iOS | supported; device acceptance required | supported; device acceptance required |

Tested conversion environment: Python 3.12, torch 2.13.0, coremltools 9.0;
ONNX Runtime 1.29.0 (Python/Go/Rust), 1.27.0 (TypeScript). Package minimums are
compatibility declarations, not a claim that every allowed combination was
measured. Go requires 1.25; Rust uses the pinned ort release candidate and needs
a compatible ONNX Runtime library through `ORT_DYLIB_PATH`. Node uses its bundled
CPU library; optional CUDA builds need the driver's supported CUDA runtime.

The RTX 3090 smoke used torch 2.8.0+cu128 with driver 575.51.03. Both models'
two fixture cases produced the CPU reference's exact tokens in eager and captured
execution. This does not extend the identity guarantee to arbitrary CUDA graph
inputs; that mode remains opt-in.

On the same Linux x86_64 machine, ONNX Runtime GPU 1.23.2 passed both models
with CPU and CUDA providers: exact reference tokens in both fixture cases.
CUDA waveform correlation was at least 0.99088 for base and 0.99943 for turbo.
Go and Rust also passed both models including the three-chunk long-form fixture.

Jetson Orin (Jetson Linux R36.5.0, NVIDIA torch 2.5.0a0+872d972e41.nv24.08)
produced exact reference tokens for both models in eager and captured CUDA
execution, with finite audio. JetPack CUDA 12.6 ONNX Runtime GPU 1.24.0
also produced exact reference tokens with CPU and CUDA providers for both
cases of both models. Minimum
CUDA waveform correlation was 0.99094 for base and 0.99959 for turbo.
These are measured cases, not an exhaustive cross-platform identity claim.

The HF Space and the ignored iOS demo scaffolding are not 0.1.1 deliverables.
The supported deployment examples are the tracked SDK quickstarts and Docker.

## Download sizes for 0.1.1

Approximate decimal sizes for the release files; backend weights are included.
Cloning adds enrollment assets only when requested. Both models ship separate
synthesis and enrollment checkpoints.

| Model | Torch | Torch + cloning | ONNX | ONNX + cloning | CoreML | CoreML + cloning |
|---|---:|---:|---:|---:|---:|---:|
| loudr-1 | 0.75 GB | 1.28 GB | 2.60 GB | 3.13 GB | 2.46 GB | 2.99 GB |
| loudr-1-turbo | 0.72 GB | 1.25 GB | 2.44 GB | 2.97 GB | 2.44 GB | 2.97 GB |
