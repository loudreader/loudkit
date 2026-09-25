# What loudkit 0.1 supports

loudkit 0.1.1 supports the features below. Each item is *Supported*,
*Supported with stated limits*, *Experimental* or *Out of scope for 0.1*.
Experimental features ship and work, but they may change, and feedback on them
is welcome. Out-of-scope items are not attempted in 0.1. Removing an item from
*Supported* is a breaking change. Adding an item is not.

## Supported

- Synthesis runs locally on your hardware. After the model download it needs no
  network. It needs no account and has no usage fees.
- Two models, `loudreader/loudr-1` and `loudreader/loudr-1-turbo`, run in all
  five SDKs. Python runs on PyTorch, ONNX Runtime or CoreML. Swift uses CoreML
  rendering. Go, Rust and TypeScript use ONNX Runtime. Both models accept the
  same voice profiles.
- A voice profile is a portable `.safetensors` file of about 150 KB.
  `loudkit clone`, or the `enroll` call in any SDK, makes one from about ten
  seconds of audio.
- The same text, voice and seed give a bit-identical waveform on the same
  build, device, backend and execution settings.
  [docs/reference/IDENTITY-CONTRACT.md](docs/reference/IDENTITY-CONTRACT.md)
  states the contract and its limits.
- Python is the reference implementation and has the full API: `synthesize`
  for text of any length, streaming with token-level cancellation,
  `previous_tokens` continuation, speed control, and saved WAVs that record the
  model, voice and seed.
- The Python backends are torch (CPU, CUDA, MPS) as the reference, ONNX Runtime
  for deployment without torch, and CoreML for Apple. All three run the same
  algorithm. Their speed differs, and so can their numerical output (see
  *Supported with stated limits*).
- Swift, Go, Rust and TypeScript are compatibility ports. They implement the
  same algorithm and pass the shared conformance fixtures. A new feature ships
  in Python first and reaches the ports with its fixture cases.
- The CLI has eight commands: `speak`, `text`, `clone`, `voices`, `download`,
  `serve`, `verify` and `doctor`. `loudkit --help` lists them, and
  [the command reference](docs/reference/cli.md) describes each one. `serve`
  runs HTTP by default; `--grpc` or `--mcp` selects another transport. `--mcp`
  is a preview and is not part of this contract.
- Three transports share the library's synthesis path: the local HTTP server
  (`loudkit serve`: `/v1` routes, SSE streaming and an OpenAI-compatible
  route), gRPC (`loudkit serve --grpc`: a typed schema with streaming
  backpressure, contract in [proto/loudkit.proto](proto/loudkit.proto)) and
  Speech Dispatcher. They share one frozen error-code catalog
  ([docs/reference/errors.md](docs/reference/errors.md)).
- A Speech Dispatcher module makes loudkit a system voice for Linux screen
  readers. It maps the Speech Dispatcher protocol codes and rate control.
- Text preparation in 12 languages expands numbers, ordinals, years, times,
  percentages and the currency signs €, $ and £. It uses loudkit's own grammar
  tables and is fuzz-tested.
- Voices for ten languages (en, pl, de, fr, nl, es, it, pt, sv, da).
  [The voice roster](VOICES.md) lists the source and licence of each voice and
  links the record of its donor and consent basis.
- By default, a WAV saved from Python and a WAV reply from the server carry an
  unsigned, machine-readable note on how the audio was made. The note names the
  algorithm, checkpoint, voice profile, backend and seed. One-shot server
  replies also carry it in a header. The Go, Rust, JS and Swift ports write
  plain PCM without it. The note is loudkit's own format. It is not C2PA.

## Supported with stated limits

- Only English has been evaluated by ear. The other nine voice languages have
  automated checks but no native-speaker review of their naturalness. Feedback
  from native speakers is welcome.
- Waveforms differ across devices and backends. The difference is measured
  and held within bands. Bit identity holds only for one build, device,
  backend and execution configuration. Speech tokens can differ too: Polish
  text on ONNX Runtime diverges from torch (see the identity contract).
- CUDA graphs (`ExecutionConfig(cuda_graphs=True)`) are an opt-in throughput
  mode in the identity contract's "equivalent" class. The output is
  deterministic but not token-identical to eager execution.
- `ChunkConfig.first_chunk_max_tokens` exists only in Python. Setting it
  changes where the first chunk ends, and so changes the audio. The ports
  refuse a checkpoint that sets it.
- Text preparation reads units, currency codes and most abbreviations as
  written. It expands the currency signs €, $ and £, and the abbreviations the
  grammar tables list (for Polish: np., itd., itp., tzn., tzw.). So `2000 zł`
  is read as "dwa tysiące zł", `12 m²` as "dwanaście m dwa", and a numeric
  date such as `03/04` digit by digit in English. `loudkit text` prints what
  will be spoken. Unit and currency reading with number agreement is planned
  for 0.1.2.
- `loudkit clone` reads a local WAV or FLAC file. It does not accept URLs, and
  it does not record, denoise, batch or preview. By default it cuts the prompt
  at a suitable pause and pads its end with silence. `--no-end-in-silence`
  keeps the recording as given. `--name` and `--language` are required.
- Every SDK (Python, Swift, Go, Rust, TypeScript) has an `enroll` call, and it
  keeps the recording as given. In Python, `end_in_silence=True` applies the
  same cut and padding as `loudkit clone`; the ports have no such option.
  There is no MCP clone tool.

## Experimental

- MCP over stdio (`loudkit serve --mcp`). It uses the same synthesis path as
  the other transports. Its tools may change.
- The systemd unit and other deployment scaffolding in `integrations/`
  beyond the Speech Dispatcher module.
- The OpenAI-compatible route's parameter mapping outside the documented
  fields.

## Out of scope for 0.1

- Equal listening quality in every language.
- A production public server. `loudkit serve` is a local runtime that binds to
  loopback by default. It has no TLS, quotas or per-voice authorization. See
  [SECURITY.md](SECURITY.md) for public binds.
- Expressiveness or emotion control.
- Robotics integrations (ROS, Wyoming), Android, WebAssembly, additional
  backends beyond the three shipped.
- Model training and fine-tuning.

## Model and platform evidence for 0.1.1

Both models use the same public API. The table separates implementation
support from model-backed validation on a particular machine:

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
ONNX Runtime 1.29.0 (Python/Go/Rust), 1.27.0 (TypeScript). Package minimums
state compatibility; not every allowed combination was measured. Go requires
1.25. Rust uses the pinned ort release candidate and needs a compatible ONNX
Runtime library through `ORT_DYLIB_PATH`. Node uses its bundled CPU library;
optional CUDA builds need the driver's supported CUDA runtime.

The RTX 3090 check used torch 2.8.0+cu128 with driver 575.51.03. Both models'
two fixture cases produced the CPU reference's exact tokens in eager and captured
execution. This does not extend the identity guarantee to arbitrary CUDA graph
inputs; that mode stays opt-in.

On the same Linux x86_64 machine, ONNX Runtime GPU 1.23.2 passed both models
with CPU and CUDA providers: exact reference tokens in both fixture cases.
CUDA waveform correlation was at least 0.99088 for base and 0.99943 for turbo.
Go and Rust also passed both models including the three-chunk long-form fixture.

Jetson Orin (Jetson Linux R36.5.0, NVIDIA torch 2.5.0a0+872d972e41.nv24.08)
produced exact reference tokens for both models in eager and captured CUDA
execution, with finite audio. JetPack CUDA 12.6 ONNX Runtime GPU 1.24.0
also produced exact reference tokens with CPU and CUDA providers for both
cases of both models. Minimum CUDA waveform correlation was 0.99094 for base
and 0.99959 for turbo.
These results cover the measured cases only.

The Hugging Face Space is a demo and not a supported deployment. The supported
deployment examples are the SDK quickstarts and Docker.

## Download sizes for 0.1.1

Approximate decimal sizes for the release files; backend weights are included.
Cloning adds enrollment assets only when requested. Both models ship separate
synthesis and enrollment checkpoints.

| Model | Torch | Torch + cloning | ONNX | ONNX + cloning | CoreML | CoreML + cloning |
|---|---:|---:|---:|---:|---:|---:|
| loudr-1 | 0.75 GB | 1.28 GB | 2.60 GB | 3.13 GB | 2.46 GB | 2.99 GB |
| loudr-1-turbo | 0.72 GB | 1.25 GB | 2.44 GB | 2.97 GB | 2.44 GB | 2.97 GB |
