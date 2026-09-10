# loudkit documentation

Twelve pages cover using loudkit. Everything else is in [reference/](reference/)
for someone who has already shipped, [platforms/](platforms/) for one machine,
and [design/](design/) for anyone changing the engine.

[Open the voice gallery](https://loudreader.github.io/loudkit/demo/) to search, listen,
compare English voices from both models. [Voices](../VOICES.md)
explains which profiles are included and records their sources and licences.

## Using loudkit

1. [Getting started](guides/01-getting-started.md): Python, first WAV, voices,
   the seed, devices.
2. [Swift](guides/10-swift.md): the same engine over CoreML.
3. [Go](guides/08-go.md): over ONNX Runtime.
4. [Rust](guides/09-rust.md): over ONNX Runtime.
5. [JavaScript and TypeScript](guides/07-js-ts.md): over `onnxruntime-node`.
6. [Choosing a model](guides/11-choosing-a-model.md): loudr-1 or loudr-1-turbo,
   and where each one runs.
7. [Cloning a voice](guides/03-cloning-a-voice.md): a profile of your own from
   ten seconds of audio.
8. [Long text and streaming](guides/02-streaming-and-long-form.md): first
   audio before the passage finishes.
9. [Server and agents](guides/04-server-and-agents.md): HTTP, gRPC, MCP and
   Speech Dispatcher over one warm engine, and how to connect an agent such as
   Hermes Agent or OpenClaw.
10. [Troubleshooting](reference/troubleshooting.md): symptoms, causes, fixes.
11. [Model card](MODEL_CARD.md): loudr-1.
12. [Turbo model card](MODEL_CARD-turbo.md): loudr-1-turbo.

Beside them: [Voices](../VOICES.md), [What 0.1 supports](../SUPPORTED.md) and
[Responsible use](../RESPONSIBLE_USE.md).

## Reference

- [Compatibility](reference/COMPATIBILITY.md): what may change between
  releases, and how to pin one.
- [Errors](reference/errors.md): what each implementation raises.
- [Timestamps](reference/timestamps.md) and [speed](reference/speed.md):
  what a result carries and how playback speed works.
- [Content Credentials](reference/provenance.md): what a saved WAV records.
- [Identity contract](reference/IDENTITY-CONTRACT.md): what "same input, same
  audio" means across backends, and what it does not.
- [Voice encoder licence chain](PROVENANCE-voice-encoder.md).

## Platforms

[Apple](platforms/apple.md) (CoreML and the Swift package),
[Docker](platforms/docker.md), [Jetson](platforms/jetson.md).

## Performance

[Benchmarks](benchmarks.md): the measured figures, the machines and the
commands. [Measured parity](parity-measured.md): the cross-runtime report.

## Design

[design/](design/) holds the notes for anyone changing the engine: the
architecture map, text normalization, postprocess, the ONNX graphs, typing,
embedding, the benchmark tools, silence classes, two-token decode and the
evaluation method. None of it is needed to use loudkit.
