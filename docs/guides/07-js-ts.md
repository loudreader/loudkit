# 7. JavaScript / TypeScript

The same engine as a Node package over `onnxruntime-node`. All three stages run
as fp32 ONNX graphs exported by `tools/export_onnx.py`. This bindings directory
needs **no torch** at runtime, only the graphs and the checkpoint's embedding
tables.

This guide assumes you have:

- the synthesis checkpoint (`loudr-1.safetensors`);
- the exported graphs beside it (`onnx/`); the release ships them, and the
  fetch below gets them with everything else;
- a voice profile (see guide 3);
- the text tokenizer, which ships beside the checkpoint as `tokenizer.json`.

**One fetch gets the whole set.** The graphs alone cannot run: this port also
reads the checkpoint's embedding tables, the tokenizer and a voice.
`loudkit download --for onnx` fetches all of it together, and nothing the
torch path alone would need.

```bash
pip install "loudkit[hub]"      # the fetch tool only; no torch
loudkit download loudreader/loudr-1 --for onnx --local-dir loudr-1
```

To clone a voice from this port, add `--with-cloning`. It fetches the three
enrollment graphs this port enrols through, and not the torch enrollment
weights, which only Python reads:

```bash
loudkit download loudreader/loudr-1 --for onnx --with-cloning --local-dir loudr-1
```

Exporting instead needs a Python checkout with torch installed, once, on any
machine. Nothing in this package
imports torch, and the machine that runs it never needs torch. If that machine
cannot have torch even once, export on another one and copy the `onnx/`
directory across. The graphs are plain files with no host affinity.

## Install

```bash
cd js
npm install
npm run build
```

## Synthesise

```typescript
import { Engine, loadVoice } from "./dist/index.js";

const engine = await Engine.load(
  "loudr-1/loudr-1.safetensors",  // checkpoint
  "loudr-1/onnx",                 // exported ONNX graphs
  "loudr-1/tokenizer.json"        // text tokenizer
);
const voice = loadVoice("loudr-1/voices/joe.safetensors");

// No language argument: the voice's own is used. The chain is the argument,
// then `voice.language`, then "en". Pass one only to read text in a language
// the voice was not enrolled in.
const result = await engine.synthesize("Hello from loudkit.", voice, 7);
// result.tokens: the 25 Hz speech tokens
// result.mel:    the 80-bin mel
// result.audio:  Float32Array at 24 kHz
console.log(`${result.audio.length / result.sampleRate}s, ${result.tokens.length} tokens`);
```

Same seed, same tokens as the Python engine. The conformance fixture
(`tests/data/conformance/vectors.json`) is verified by both `pytest` and the
JS suite, down to exact Philox bits, sampler token choices, frontend ids and
free-running tokens.

### Passages longer than one window

`engine.synthesize` renders exactly one window. It returns an error for
anything longer rather than clipping the end of a paragraph.
`engine.synthesizeLong` is the long-form path. It runs the speech funnel over
the whole text, splits on the manifest's `chunking` recipe, gives each chunk
its own derived seed, and carries a token prefix across the joins so the pitch
contour does not restart at every boundary. Same algorithm as Python's
`Engine.synthesize_long`, same fingerprint.

### Timestamps, speed, and continuing a previous call

`synthesize` and `synthesizeLong` return `chunks: ChunkTiming[]`. Chunk
positions in the waveform are exact, taken from sample offsets. Word positions
are an **estimate** by proportional allocation, not a forced alignment. Read
[`docs/reference/timestamps.md`](../reference/timestamps.md) before you rely on
the word times. Each streamed chunk carries its own `timing` starting at zero.
`ChunkTiming` is a plain object in this port, so add the running offset to
`start`, `end` and each word yourself as you stitch.

The two execution inputs ride in a trailing options object. The alternative
reads `synthesize(t, v, 7, undefined, undefined, 1.5, prev)`.

```ts
const first = await engine.synthesizeLong("Part one.", voice, 7, undefined, undefined, {
  speed: 1.25,                    // pitch preserved; 1.0 is an exact bypass
});
const second = await engine.synthesizeLong("Part two.", voice, 8, undefined, undefined, {
  previousTokens: first.tokens,   // the join stops being audible
});
```

`speed` is refused rather than clamped outside `[0.5, 2.0]`; see
[`docs/reference/speed.md`](../reference/speed.md). `previousTokens` accepts any
length and only the tail is used, so passing the whole previous `tokens` is the
intended call.

## Choose the execution provider

`Engine.load` and `Enroller.load` take an `onnxProvider` option. It accepts the
same five values in every loudkit port: `auto`, `cpu`, `cuda`, `coreml`,
`directml`.

```typescript
const engine = await Engine.load(ckpt, onnxDir, tokenizerPath, {
  onnxProvider: "cpu",
});
console.log(engine.onnxProvider);   // the provider that ran, never "auto"
console.log(engine.describe());     // ... | exec[onnx provider=cpu prec[all=fp32]]
```

`auto` is the default. It takes CUDA where the build offers it, and CPU
otherwise. Any other value is a requirement: if the build does not carry that
provider, `load` throws and names what is available. It never falls back to CPU
in silence.

Which providers exist is fixed when `onnxruntime-node` is installed, not when
loudkit runs. Call `availableProviders()` to read what the installed build
offers. Do not infer it from `process.platform`, which describes the download
and not a locally built binding.

## CoreML is not available here

`coreml` throws in this port. It is the one place where the five spellings are
not five choices.

CoreML is worth using only when the compiled models are cached. Compiling the
renderer graphs takes about two minutes, and the Python, Rust and Go ports pay
that once per machine by naming a cache directory. Node cannot name one: the
native addon reads `coreMlFlags` and nothing else, and session-config entries
are ignored, so ONNX Runtime's own EPContext caching is out of reach as well.
Every process would pay the compile again — measured on an M3 Pro, 116.7 s to
open the graphs against 2.1 s on `cpu`.

For CoreML on Apple hardware use the Swift package, or the Python, Rust or Go
ports.

## CUDA

`cuda` runs the same graphs on an NVIDIA GPU and produces the same speech
tokens as `cpu`, byte for byte.

It needs a driver from the 580 series or newer. The CUDA provider is not in the
npm tarball: `onnxruntime-node`'s install step fetches it separately, and from
1.27 that build links CUDA 13, which older drivers cannot load. On a driver
from the CUDA 12 era the provider fails to load with a missing `libcudart.so.13`
rather than falling back, and `npm install onnxruntime-node@1.26.0` is the last
version built against CUDA 12.

## Verify your build against the shared fixture

The weight-free vectors (RNG, sampler, frontend, seed derivation) run anywhere:

```bash
npm test
```

The full engine (tokens + render band) needs the assets and skips without them:

```bash
LOUDKIT_CKPT=…/loudr-1/loudr-1.safetensors \
LOUDKIT_ONNX_DIR=…/loudr-1/onnx \
LOUDKIT_VOICE=…/tests/data/reference/testvoice.voice.safetensors \
LOUDKIT_TOKENIZER=…/loudr-1/tokenizer.json \
npm run test:all
```

Or run the end-to-end conformance script directly (prints PASS/FAIL per case):

```bash
node dist/test/run_conformance.js \
  --ckpt …/loudr-1/loudr-1.safetensors \
  --onnx …/loudr-1/onnx \
  --voice …/tests/data/reference/testvoice.voice.safetensors \
  --fixture ../../tests/data/conformance
```

The ported pieces live in `src/`: `rng.ts` (Philox-4x32-10), `sampler.ts`
(LR-SAMPLER-v1), `frontend.ts` (grapheme tokenizer), `windowing.ts` (the
static window recipe), `noise.ts`, `engine.ts` (the three ONNX sessions and
the generation loop). Each mirrors a Python module 1:1 so a fix on either side
is a one-line diff on the other.

## Scope

- **Enrollment.** Ported, and held to the shared enrollment fixture (prompt and
  conditioning tokens exact, embeddings cosine > 0.9999).

Not ported:

- **fp16 / int8.** The graphs are fp32, the registered and measured
  configuration. int8 stays blocked; nothing int8 is produced.
- **The server.** The Node package is a library. `engine.stream` is an async
  generator that yields chunks in process; only Python streams over a
  transport. Wrap it in your own server if you need one over the wire.
