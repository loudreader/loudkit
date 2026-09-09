# loudkit for JavaScript and TypeScript

Text to speech in Node, on `onnxruntime-node`. No Python, no torch.

## Hello

Node 20 or newer.

```bash
npm install loudkit
```

The `onnxruntime-node` dependency has an install script for optional native
binaries, including CUDA. If your package manager blocks dependency scripts,
review and allow that dependency's script when you need those binaries. The
bundled CPU runtime does not require the optional CUDA download; selecting CUDA
also requires a driver compatible with the installed runtime.

`hello.mjs`:

```javascript
import { Engine } from "loudkit";

const engine = await Engine.load("loudreader/loudr-1");
const result = await engine.synthesize("Hello from loudkit.", engine.voice("joe"), { seed: 7 });
result.saveWav("hello.wav");
console.log(`hello.wav, ${(result.audio.length / result.sampleRate).toFixed(2)}s`);
await engine.close();
```

```bash
node hello.mjs
```

The first run downloads the model files into
`~/Library/Caches/loudkit/loudreader--loudr-1` on macOS and
`~/.cache/loudkit/loudreader--loudr-1` on Linux (`$LOUDKIT_CACHE` moves it),
the directory the Go, Rust and Swift ports share, and checks every file
against the release's own `SHA256SUMS`; later runs read what is there.
`engine.voices()` names the 28 voices. `close()` hands back the
runtime's memory; it matters when you build a second engine.

The snippets on this page need loudkit 0.1.1. From a checkout: `npm run build`,
then `node examples/hello.mjs`, which is this file.

Both `loudr-1` and `loudr-1-turbo` use this API in 0.1.1. Change the model
name to switch; keep the same voice profile. A local release directory works
as well as a published model name.


## The rest of the front door

```javascript
await download("loudreader/loudr-1", "loudr-1");          // a directory of your own
await download(repo, dir, { revision: "v0.1.1", cloning: true });
await Engine.load("loudr-1");                            // a directory or a repo id
engine.voices();                                         // the names in the release
engine.voice("joe");                                     // one of them
await engine.synthesize(text, voice, { seed, language, speed, previousTokens });
for await (const chunk of engine.stream(text, voice, options)) play(chunk.audio);
const mine = await engine.enroll("me.wav", { name: "mine", language: "en" });
saveVoice(mine, "mine.safetensors");                     // a portable profile
loadVoice("mine.safetensors");
result.saveWav(path);  result.toWav();                   // 16-bit PCM
```

`synthesize` takes text of any length: it splits at sentence boundaries and
joins the audio. Every option has a default: seed 0, the voice's own language,
normal speed. `stream` yields chunks as they are made; `options.shouldCancel`
stops within one decode step. `enroll` on an engine loaded by repo id fetches
the enrollment graphs once; a directory of your own needs `{ cloning: true }`.
`Engine.loadPaths(checkpoint, onnxDir, tokenizer)` opens a layout of your own.

Streaming, timestamps, speed and barge-in: `docs/guides/07-js-ts.md`.

## Execution provider

`onnxProvider` picks the onnxruntime execution provider; the five values are
the same in every port: `auto` (the default), `cpu`, `cuda`, `coreml`,
`directml`.

```javascript
const engine = await Engine.load("loudr-1", { onnxProvider: "auto" });
console.log(engine.onnxProvider);  // the one that ran, never "auto"
console.log(engine.describe());
```

`auto` takes CUDA where the build offers it and CPU otherwise. A named provider
the build does not carry is an error, never a quiet fall back to CPU. Which
providers exist is fixed when `onnxruntime-node` is installed:

| platform | providers |
| --- | --- |
| darwin/x64, darwin/arm64 | `cpu`, `coreml` |
| linux/x64 | `cpu`, `cuda` |
| win32/x64, win32/arm64 | `cpu`, `directml` |

`availableProviders()` reports what the installed build offers. This package
refuses `coreml`, because `onnxruntime-node` cannot keep its compile cache;
use the Swift package for CoreML. A GPU provider can change the token stream
and waveform; conformance runs pin CPU. See
[`docs/benchmarks.md`](../docs/benchmarks.md#onnx-execution-providers).

## Build and test

```bash
npm ci
npm test                 # weight-free conformance vectors
npm run test:all         # + engine conformance (needs checkpoint + graphs)
```

`onnxruntime-node` ships the native runtime as a package dependency. The
checkpoint, graphs and tokenizer are not bundled; `Engine.load` fetches them.
