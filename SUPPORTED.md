# What loudkit 0.1 supports

The scope, stated once, so the README cannot promise more than the project
keeps. Anything not listed under *Supported* is either *Experimental* (works,
shipped, feedback wanted, may change) or *Out of scope for 0.1* (not
attempted). Removing something from *Supported* is a breaking change. Promoting
something into it is not.

## Supported

- **Local synthesis, offline.** No account, no network dependency, no
  per-character billing. The engine loads a packed checkpoint and runs on
  your hardware.
- **Voice profiles as files.** A voice is a ~150 KB `.safetensors` you can
  copy, mail and keep; cloning one takes about ten seconds of audio, from the
  shell with `loudkit clone` or from any SDK with `enroll`.
- **Deterministic rendering.** Same text, voice, seed and build give a
  bit-identical waveform on a given device and backend, with the contract and
  its limits written down in
  [docs/reference/IDENTITY-CONTRACT.md](docs/reference/IDENTITY-CONTRACT.md).
- **Python as the reference implementation**, with the full API: `synthesize`,
  `synthesize_long`, streaming with token-level cancellation, `previous_tokens`
  continuation, speed control, provenance-carrying WAVs.
- **Backends:** torch (CPU, CUDA, MPS) as the reference execution, ONNX
  Runtime as the portable no-torch deployment, CoreML for Apple. Execution
  may differ in speed per backend, never in what is computed.
- **Compatibility ports:** Swift, Go, Rust and TypeScript implement the same
  algorithm against shared conformance fixtures. They track the reference:
  features land in Python first and reach the ports with the fixture that
  proves them.
- **The CLI**, eight commands: `speak`, `clone`, `voices`, `download`,
  `serve`, `verify`, `doctor`, `grpc`. `loudkit --help` lists these and no
  others. `describe`, `bench` and `profile` stay in the package as repo tools
  and stay runnable, and `mcp` is preview; none of the four is part of this
  contract.
- **Transports over one synthesis path:** the local HTTP server (`/v1`,
  SSE streaming, OpenAI-compatible route), gRPC (`loudkit grpc`, typed schema
  and streaming backpressure, contract in
  [proto/loudkit.proto](proto/loudkit.proto)), and Speech Dispatcher. All
  answer byte-for-byte what the library answers, with one frozen error-code
  catalog ([docs/reference/errors.md](docs/reference/errors.md)).
- **Speech Dispatcher integration**: loudkit as a system voice for Linux
  screen readers, with the protocol codes and rate control handled properly.
- **Text handling in 12 languages** (numbers, dates, currency, units,
  abbreviations), verbalised from first principles and fuzz-tested.
- **Voices for ten languages** (en, pl, de, fr, nl, es, it, pt, sv, da), each
  with its donor or source, licence and consent basis recorded in
  [docs/voices/roster/provenance.json](docs/voices/roster/provenance.json).
- **Provenance**: every saved WAV and server reply carries a C2PA claim-only
  manifest naming the algorithm, checkpoint, voice profile, backend and seed.

## Supported with stated limits

- **Language quality.** We evaluated English by ear. We do not speak the other
  nine shipped languages well enough to judge their naturalness reliably.
  Automated checks cover them, but those checks are not a native-speaker
  verdict. Feedback from native speakers is very welcome.
- **Waveforms are not identical across devices or backends.** Equivalence is
  measured and banded. Identity holds per build/device/backend.
- **`--cuda-graphs` is an opt-in throughput mode** in the identity contract's
  "equivalent" class: deterministic, not token-identical to eager.
- **`ChunkConfig.first_chunk_max_tokens` is Python-only for now.** It is an
  algorithm-level knob that re-fingerprints when set. The ports follow the
  usual fixture-first route.
- **`loudkit clone` reads a local WAV or FLAC file.** No URLs, no recording,
  no denoising, no trimming, no batch, no preview. `--name` and `--language`
  are stated, not guessed. The API path, `loudkit.enroll(...)`, exists in every
  SDK's language (Python, Swift, Go, Rust, TypeScript) and takes samples
  directly. There is no MCP clone tool.

## Experimental

- **MCP over stdio** (`loudkit mcp`). It answers the same bytes as every other
  transport and stays in the package, but it is not in the top-level help and
  its tool surface may change shape.
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
