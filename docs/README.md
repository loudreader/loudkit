# loudkit documentation

The guides below cover using loudkit. [reference/](reference/) holds the
contracts and details, [platforms/](platforms/) covers specific machines, and
[design/](design/) holds engine notes for contributors.

The [voice gallery](https://loudreader.github.io/loudkit/demo/) plays every
voice and compares the English voices across both models. [Voices](../VOICES.md)
lists the included profiles with their sources and licences.

## Using loudkit

1. [Getting started](guides/01-getting-started.md): Python, first WAV, voices,
   the seed, devices.
2. [Swift](guides/10-swift.md): the Swift port, with CoreML rendering.
3. [Go](guides/08-go.md): the Go port, on ONNX Runtime.
4. [Rust](guides/09-rust.md): the Rust port, on ONNX Runtime.
5. [JavaScript and TypeScript](guides/07-js-ts.md): the TypeScript port, on
   `onnxruntime-node`.
6. [Choosing a model](guides/11-choosing-a-model.md): loudr-1 or loudr-1-turbo,
   and where each one runs.
7. [Cloning a voice](guides/03-cloning-a-voice.md): a profile of your own from
   ten seconds of audio.
8. [Long text and streaming](guides/02-streaming-and-long-form.md): first
   audio before the passage finishes.
9. [Server and agents](guides/04-server-and-agents.md): HTTP, gRPC, MCP and
   Speech Dispatcher over one loaded engine, and how to connect an agent such as
   Hermes Agent or OpenClaw.
10. [Troubleshooting](reference/troubleshooting.md): symptoms, causes, fixes.
11. [Model card](MODEL_CARD.md): loudr-1.
12. [Turbo model card](MODEL_CARD-turbo.md): loudr-1-turbo.

Also see [Voices](../VOICES.md), [What 0.1 supports](../SUPPORTED.md) and
[Responsible use](../RESPONSIBLE_USE.md).

## Reference

- [Command line](reference/cli.md): the eight `loudkit` commands and their
  main options.
- [Compatibility](reference/COMPATIBILITY.md): what may change between
  releases, and how to pin one.
- [Errors](reference/errors.md): what each implementation raises.
- [Timestamps](reference/timestamps.md) and [speed](reference/speed.md):
  what a result carries and how playback speed works.
- [What a saved WAV records](reference/provenance.md): the unsigned note on how
  it was made.
- [Identity contract](reference/IDENTITY-CONTRACT.md): what "same input, same
  audio" means across backends, and its limits.
- [Voice encoder licence chain](PROVENANCE-voice-encoder.md).

## Platforms

[Apple](platforms/apple.md) (CoreML and the Swift package),
[Docker](platforms/docker.md), [Jetson](platforms/jetson.md).

## Performance

[Benchmarks](benchmarks.md): the measured figures, the machines and the
commands. [Measured parity](parity-measured.md): the cross-runtime report.

## Design

[design/](design/) holds notes for contributors who change the engine: the
architecture map, text normalization, postprocess, the ONNX graphs, typing,
embedding, the benchmark tools, silence classes, two-token decode and the
evaluation method. You do not need them to use loudkit.
