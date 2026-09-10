# 7. JavaScript and TypeScript

The same engine as a Node package over `onnxruntime-node`. No Python, no
torch.

## Hello

Node 20 or newer.

```bash
npm install loudkit
```

Save this as `hello.mjs`, then run `node hello.mjs`:

```javascript
import { Engine } from "loudkit";

const engine = await Engine.load("loudreader/loudr-1");
const result = await engine.synthesize("Hello from loudkit.", engine.voice("joe"), { seed: 7 });
result.saveWav("hello.wav");
console.log(`hello.wav, ${(result.audio.length / result.sampleRate).toFixed(2)}s`);
await engine.close();
```

The first run downloads the model files into
`~/Library/Caches/loudkit/loudreader--loudr-1` on macOS and
`~/.cache/loudkit/loudreader--loudr-1` on Linux (`$LOUDKIT_CACHE` moves it),
the directory the Go, Rust and Swift ports share, and checks every file
against the release's own `SHA256SUMS`; later runs read what is there.
`engine.voices()` names the 28 voices. The snippets need loudkit
0.1.1; from a checkout, `npm run build` then `node examples/hello.mjs`.

Both `loudr-1` and `loudr-1-turbo` use this API in 0.1.1. Change the model
name to switch; keep the same voice profile. A local release directory works
as well as a published model name.


## A directory of your own

```javascript
import { download, Engine } from "loudkit";

const dir = await download("loudreader/loudr-1", "loudr-1", { revision: "v0.1.1" });
const engine = await Engine.load(dir);
```

`download` writes a receipt, `.loudkit-release.json`; a later call whose
revision still resolves to the same commit fetches and hashes nothing, a moved
revision keeps every file that still hashes to the new `SHA256SUMS` and fetches
the rest, and an interrupted fetch resumes. Pin `revision` for anything reproducible. Under
`loudreader/`, `release.json` must say the bundle passed the builder's gate,
and it is checked before any weight moves.

## Synthesize

`synthesize` takes text of any length: it splits at sentence boundaries,
gives each chunk its own seed, carries the pitch contour across the joins and
returns one result. Every option has a default.

```javascript
const result = await engine.synthesize(text, voice, {
  seed: 7,                // 0 when omitted
  language: "pl",         // the voice's own when omitted
  speed: 1.25,            // [0.5, 2.0], pitch preserved; 1.0 is an exact bypass
  previousTokens: earlier.tokens,   // continue an earlier result's pitch contour
});
result.audio;        // Float32Array at result.sampleRate
result.tokens;       // the speech tokens
result.chunks;       // where each chunk lands, and an estimate of each word
result.hitTokenCap;       // generation stopped at the token cap: probably truncated
result.saveWav(path); result.toWav();
```

`synthesizeWindow` renders exactly one model window and throws on longer
text; it is for the conformance harness.

## Streaming and barge-in

```javascript
for await (const chunk of engine.stream(text, voice, { seed: 7, shouldCancel: () => stop })) {
  play(chunk.audio);   // chunk.timing starts at zero; add your own offset
}
```

`stream` yields chunks as they are made, so playback starts before the
passage is finished. `shouldCancel` is polled on every decode step, and the
chunk being generated is discarded.

## Timestamps and speed

`result.chunks` is exact at the chunk level and an estimate at the word
level; read [timestamps.md](../reference/timestamps.md) before building on
the word times. `speed` is refused outside `[0.5, 2.0]`; see
[speed.md](../reference/speed.md).

## Cloning a voice

```javascript
import { saveVoice } from "loudkit";

const mine = await engine.enroll("me.wav", { name: "mine", language: "en" });
saveVoice(mine, "mine.safetensors");
```

The first `enroll` on an engine loaded by repo id fetches the three
enrollment graphs into the same cache directory,
`~/Library/Caches/loudkit/loudreader--loudr-1` on macOS and
`~/.cache/loudkit/loudreader--loudr-1` on Linux; later calls read them from
there. An engine loaded from a directory of your own needs them fetched with
`download(repo, dir, { cloning: true })`. Five to ten seconds of clean speech
is the input this was tuned for. `enroll` takes a WAV path, WAV bytes, or
samples with their rate. `loadVoice(path)` reads the profile back.

## Execution provider

`onnxProvider` picks the onnxruntime execution provider: `auto` (the
default), `cpu`, `cuda`, `coreml`, `directml`.

```javascript
const engine = await Engine.load("loudreader/loudr-1", { onnxProvider: "cpu" });
engine.onnxProvider;   // the provider that ran, never "auto"
engine.describe();
```

`auto` takes CUDA where the installed `onnxruntime-node` offers it and CPU
otherwise. A named provider the build does not carry throws and names what is
available; nothing falls back to CPU in silence. `availableProviders()`
reports what the installed build offers.

`coreml` throws in this port: `onnxruntime-node` cannot keep the compiled
graphs between processes, and compiling them costs about two minutes every
run. Use the Swift package for CoreML. `cuda` needs a driver from the 580
series or newer; `onnxruntime-node@1.26.0` is the last version built against
CUDA 12.

## Your own layout

`Engine.loadPaths(checkpoint, onnxDir, tokenizer)` opens three paths you
assembled yourself.

## Verify against the shared fixture

```bash
npm test                 # weight-free vectors
LOUDKIT_CKPT=… LOUDKIT_ONNX_DIR=… LOUDKIT_VOICE=… LOUDKIT_TOKENIZER=… npm run test:all
```

The second needs the checkpoint, the graphs and the reference voice, and
holds `synthesize` and `stream` to the fixture's tokens, chunk by chunk.
