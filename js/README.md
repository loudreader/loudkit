# loudkit for JavaScript and TypeScript

Text to speech in Node, on `onnxruntime-node`. It does not need Python or
PyTorch.

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

The first run downloads the model files into the user cache:

- macOS: `~/Library/Caches/loudkit/loudreader--loudr-1`
- Linux: `$XDG_CACHE_HOME/loudkit/loudreader--loudr-1`, or
  `~/.cache/loudkit/loudreader--loudr-1` when `XDG_CACHE_HOME` is not set
- Windows: `%LOCALAPPDATA%\loudkit\loudreader--loudr-1`

Set `LOUDKIT_CACHE` to use `$LOUDKIT_CACHE/loudreader--loudr-1` instead. The
Go, Rust and Swift ports use the same directory. Each downloaded file is
checked against the release's `SHA256SUMS`. `engine.voices()` lists the 28
voices. `close()` releases the native runtime's memory, which matters when a
process loads a second engine.

The examples on this page need loudkit 0.1.1. To run the example from a
repository checkout, run `npm run build`, then `node examples/hello.mjs`, in
`js/`. `examples/hello.mjs` is the program above.

`loudr-1` and `loudr-1-turbo` use the same API. To switch models, change the
model name. The same voice profiles work with both models. `Engine.load` also
accepts a local release directory.


## API overview

```javascript
await download("loudreader/loudr-1", "loudr-1");          // download to a local directory
await download(repo, dir, { revision: "v0.1.1", cloning: true });
await Engine.load("loudr-1");                            // a local directory or a repo id
engine.voices();                                         // the voice names in the release
engine.voice("joe");                                     // load one voice by name
await engine.synthesize(text, voice, { seed, language, speed, previousTokens });
for await (const chunk of engine.stream(text, voice, options)) play(chunk.audio);
const mine = await engine.enroll("me.wav", { name: "mine", language: "en" });
saveVoice(mine, "mine.safetensors");                     // a portable profile
loadVoice("mine.safetensors");
result.saveWav(path);  result.toWav();                   // 16-bit PCM
```

`synthesize` splits long text into chunks at sentence boundaries and joins
the audio in memory. Every option has a default: seed 0, the voice's own
language, speed 1.0. `stream` yields each chunk when it is ready.
`options.shouldCancel` is checked on every decode step.

On an engine loaded by repo id, the first `enroll` call fetches the enrollment
graphs. For a local directory, fetch them with `{ cloning: true }`.
`Engine.loadPaths(checkpoint, onnxDir, tokenizer)` loads assets from the paths
you give.

Streaming, timestamps, speed and barge-in: `docs/guides/07-js-ts.md`.

## Execution provider

`onnxProvider` selects the ONNX Runtime execution provider: `auto` (the
default), `cpu`, `cuda`, `coreml` or `directml`. The Python, Go and Rust
implementations accept the same five values. This package refuses `coreml`.

```javascript
const engine = await Engine.load("loudr-1", { onnxProvider: "auto" });
console.log(engine.onnxProvider);  // the one that ran, never "auto"
console.log(engine.describe());
```

`auto` selects CUDA where the build has it, and CPU otherwise. If you name a
provider that the build does not have, `Engine.load` throws. The installed
`onnxruntime-node` binary decides which providers this package can use:

| platform | providers |
| --- | --- |
| darwin/x64, darwin/arm64 | `cpu` |
| linux/x64 | `cpu`, `cuda` |
| win32/x64, win32/arm64 | `cpu`, `directml` |

`availableProviders()` lists the providers this package can use with the
installed build. The darwin binaries also contain CoreML, but this package
refuses `coreml`, because `onnxruntime-node` cannot keep its compile cache. For
CoreML, use the Swift package. A GPU provider can change the token stream and
waveform. Conformance runs use CPU. See
[`docs/benchmarks.md`](../docs/benchmarks.md#onnx-execution-providers).

## Build and test

```bash
npm ci
npm test                 # weight-free conformance vectors
npm run test:all         # + engine conformance (needs checkpoint + graphs)
```

`onnxruntime-node` ships the native runtime as a package dependency. The
checkpoint, graphs and tokenizer are not bundled; `Engine.load` fetches them.
