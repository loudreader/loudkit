# 7. JavaScript and TypeScript

The JavaScript and TypeScript port of loudkit runs on `onnxruntime-node`. It
does not need Python or PyTorch.

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

The first run downloads the model files into the user cache:

- macOS: `~/Library/Caches/loudkit/loudreader--loudr-1`
- Linux: `$XDG_CACHE_HOME/loudkit/loudreader--loudr-1`, or
  `~/.cache/loudkit/loudreader--loudr-1` when `XDG_CACHE_HOME` is not set
- Windows: `%LOCALAPPDATA%\loudkit\loudreader--loudr-1`

Set `LOUDKIT_CACHE` to use `$LOUDKIT_CACHE/loudreader--loudr-1` instead. The
Go, Rust and Swift ports use the same directory. Each downloaded file is
checked against the release's `SHA256SUMS`. Later runs reuse the cache and
fetch only the files that changed when the repo's `main` branch moves.
`engine.voices()` lists the 28 voices.

The examples on this page need loudkit 0.1.1. To run the example from a
repository checkout, run `npm run build`, then `node examples/hello.mjs`, in
`js/`.

`loudr-1` and `loudr-1-turbo` use the same API. To switch models, change the
model name. The same voice profiles work with both models. `Engine.load` also
accepts a local release directory.


## Download to a local directory

```javascript
import { download, Engine } from "loudkit";

const dir = await download("loudreader/loudr-1", "loudr-1", { revision: "v0.1.1" });
const engine = await Engine.load(dir);
```

`download` writes a receipt, `.loudkit-release.json`, into the directory. On
a later call:

- If the revision still resolves to the same commit, `download` fetches and
  hashes no weight file.
- If the revision moved, it keeps every file that matches the new
  `SHA256SUMS` and fetches the rest.
- An interrupted fetch resumes.

Pin `revision` to a tag or commit for a reproducible build. For repos under
`loudreader/`, `release.json` must show that the release passed its build
checks. `download` reads it before it fetches any weight file.

## Synthesize

`synthesize` splits long text into chunks at sentence boundaries. It gives
each chunk its own seed, conditions each chunk on the speech tokens at the end
of the chunk before it, and returns one result that holds all the audio. Every
option has a default.

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
text. The conformance tests use it.

## Streaming and barge-in

```javascript
for await (const chunk of engine.stream(text, voice, { seed: 7, shouldCancel: () => stop })) {
  play(chunk.audio);   // chunk.timing starts at zero; add your own offset
}
```

`stream` yields each chunk when it is ready, so playback can start before the
passage is finished. `shouldCancel` is checked on every decode step. When it
returns true, the chunk in progress is discarded.

## Timestamps and speed

`result.chunks` is exact at the chunk level and an estimate at the word
level. Read [timestamps.md](../reference/timestamps.md) before you use the
word times. `speed` outside `[0.5, 2.0]` is refused; see
[speed.md](../reference/speed.md).

## Cloning a voice

```javascript
import { saveVoice } from "loudkit";

const mine = await engine.enroll("me.wav", { name: "mine", language: "en" });
saveVoice(mine, "mine.safetensors");
```

On an engine loaded by repo id, the first `enroll` call fetches the three
enrollment graphs into the model's cache directory. Later calls use the cached
graphs. For an engine loaded from a local directory, fetch the graphs first
with `download(repo, dir, { cloning: true })`.

Use five to ten seconds of clean speech. `enroll` takes a WAV path, WAV
bytes, or samples with their sample rate. `loadVoice(path)` loads a saved
profile.

## Execution provider

`onnxProvider` selects the ONNX Runtime execution provider: `auto` (the
default), `cpu`, `cuda` or `directml`. This port refuses `coreml`; see below.

```javascript
const engine = await Engine.load("loudreader/loudr-1", { onnxProvider: "cpu" });
engine.onnxProvider;   // the provider that ran, never "auto"
engine.describe();
```

`auto` selects CUDA where the installed `onnxruntime-node` has it, and CPU
otherwise. It never selects DirectML. If CUDA fails to open under `auto`, the
load emits a `LoudkitProviderWarning` and uses CPU. If you name a provider that the build does not have, `Engine.load`
throws and lists the providers it has. `availableProviders()` lists the
providers this port can use with the installed build.

`coreml` throws in this port. `onnxruntime-node` cannot keep the compiled
CoreML graphs between processes, so every process would compile them again,
which takes about two minutes. For CoreML, use the Swift package.

`cuda` needs an NVIDIA driver from the 580 series or newer.
`onnxruntime-node@1.26.0` is the last version built against CUDA 12.

## Explicit asset paths

`Engine.loadPaths(checkpoint, onnxDir, tokenizer)` loads a checkpoint, a
directory of ONNX graphs and a tokenizer file from the paths you give.

## Verify against the shared fixture

```bash
npm test                 # weight-free vectors
LOUDKIT_CKPT=… LOUDKIT_ONNX_DIR=… LOUDKIT_VOICE=… LOUDKIT_TOKENIZER=… npm run test:all
```

The second command needs the checkpoint, the graphs and the reference voice.
It compares the tokens from `synthesize` and `stream` with the fixture, chunk
by chunk.
