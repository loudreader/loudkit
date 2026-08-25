# loudkit documentation

Start with [the guides](guides/) if you are using the library, and
[reference](reference/) if you are looking up a specific behaviour.

## Guides

Guides 1-6 form the Python path. Guides 7-10 are standalone quickstarts for the
other four implementations.

[Getting started](guides/01-getting-started.md) ·
[Streaming and long-form](guides/02-streaming-and-long-form.md) ·
[Cloning a voice](guides/03-cloning-a-voice.md) ·
[Server, streaming API and MCP](guides/04-server-and-agents.md) ·
[Benchmarking](guides/05-benchmarking.md) ·
[Embedding](guides/06-embedding.md) ·
[JavaScript / TypeScript](guides/07-js-ts.md) ·
[Go](guides/08-go.md) ·
[Rust](guides/09-rust.md) ·
[Swift](guides/10-swift.md)

## Reference

What the library guarantees, and what each implementation does.

- [Troubleshooting](reference/troubleshooting.md): symptoms, causes and fixes.
- [Compatibility](reference/COMPATIBILITY.md): versioning and breaking changes.
- [Errors](reference/errors.md): errors exposed by each implementation.
- [Timestamps](reference/timestamps.md), [speed](reference/speed.md) and
  [provenance](reference/provenance.md): output behaviour.
- [Architecture](reference/ARCHITECTURE.md),
  [ONNX graphs](reference/onnx-graphs.md) and
  [identity](reference/IDENTITY-CONTRACT.md): implementation details for
  developers changing or porting the engine.
- [Text normalization](reference/preprocess.md),
  [postprocess](reference/postprocess.md) and [typing](reference/typing.md):
  deeper reference material.

## Platforms

[Apple: CoreML artefacts and the Swift package](platforms/apple.md) ·
[Docker: variants, compose and the port-mapping boundary](platforms/docker.md) ·
[Jetson: the JetPack environment the Orin rows were measured with](platforms/jetson.md)

## The model

[Model card](MODEL_CARD.md) ·
[Voice encoder provenance](PROVENANCE-voice-encoder.md) ·
[Voices](../VOICES.md)

## Performance

[Benchmarks](benchmarks.md) · [Measured parity](parity-measured.md)
