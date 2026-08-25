# Guides

Start with the quickstart, then open only the guide you need. The four port
guides include their own download and build steps. Python snippets assume
`pip install "loudkit[torch,audio,hub]"` and weights reachable by repo id or by
a checkpoint file at `loudr-1.safetensors`. See
[the quickstart](../../README.md#make-a-wav) for how to get both.

| # | Guide | What you will have when it is done |
|---|---|---|
| 1 | [Getting started](01-getting-started.md) | Your first WAV, from a two-line Python API |
| 2 | [Streaming and long-form](02-streaming-and-long-form.md) | First audio before the passage finishes; long texts split across windows |
| 3 | [Cloning a voice](03-cloning-a-voice.md) | A voice profile that is yours, from ten seconds of audio |
| 4 | [Server, streaming API and MCP](04-server-and-agents.md) | Any local app or agent can speak |
| 5 | [Benchmarking and profiling](05-benchmarking.md) | A reproducible number for your hardware, stage by stage |
| 6 | [Embedding loudkit](06-embedding.md) | loudkit inside your own program, on the right device |
| 7 | [JavaScript / TypeScript](07-js-ts.md) | the same engine from Node, over onnxruntime-node |
| 8 | [Go](08-go.md) | the same engine from Go, over yalue/onnxruntime_go |
| 9 | [Rust](09-rust.md) | the same engine from Rust, over the `ort` crate |
| 10 | [Swift](10-swift.md) | the same engine from Swift, over CoreML, with the estimator on the Neural Engine |

The [documentation index](../README.md) links the API and implementation
reference. Guides 1 through 6 use Python; guides 7 through 10 cover the other
four implementations.
