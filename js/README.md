# loudkit JS/TS binding

The loudkit engine over `onnxruntime-node`. The sampler, RNG (Philox),
tokenizer, windowing and the generator loop are ported to TypeScript, with
**no torch** at runtime.

## Status: supported

Passes the shared conformance fixture: weight-free vectors exact, free-run
tokens and render band against the checkpoint. The full walkthrough is
`docs/guides/07-js-ts.md` in the repo root.

## Requirements

- Node ≥ 20 (uses `node:test`; `package.json` requires it, and CI runs 20 and 22)
- the exported ONNX graphs (`onnx/` beside the checkpoint; run
  `python tools/export_onnx.py` once from the repo root)
- the packed checkpoint, a voice profile, and `tokenizer.json`

## Synthesise

```bash
npm install loudkit
```

```bash
pip install "loudkit[hub]"
loudkit download loudreader/loudr-1 --for onnx --local-dir loudr-1
```

Everything lands inside `loudr-1/`, which is what the paths below are relative
to. `--with-cloning` adds the three enrollment graphs.

```javascript
import { Engine, loadVoice } from "loudkit";

const engine = await Engine.load(
  "loudr-1/loudr-1.safetensors",  // packed checkpoint
  "loudr-1/onnx",                 // exported graphs
  "loudr-1/tokenizer.json"        // text tokenizer
);
const voice = loadVoice("loudr-1/voices/joe.safetensors");

// Use synthesizeLong. synthesize renders one window and errors on anything
// longer instead of clipping it. Omit the language to read in the voice's own.
const r = await engine.synthesizeLong("Hello from loudkit.", voice, 7);
console.log(`${r.tokens.length} tokens, ${(r.audio.length / r.sampleRate).toFixed(2)}s`);
// r.audio is a Float32Array at r.sampleRate. Hand it to your own WAV writer
// or audio device. This package writes no files.
```

Streaming, timestamps, speed and barge-in: `docs/guides/07-js-ts.md`.

## Execution provider

`onnxProvider` picks the onnxruntime execution provider. The five accepted
values are the same in every loudkit port: `auto`, `cpu`, `cuda`, `coreml`,
`directml`.

```javascript
const engine = await Engine.load(ckpt, onnxDir, tokenizerPath, {
  onnxProvider: "auto",         // the default
});
console.log(engine.onnxProvider);  // the one that ran, never "auto"
console.log(engine.describe());    // algo[...] loudkit-1 | exec[onnx provider=coreml prec[all=fp32]]
```

`auto` takes CUDA where the build offers it and CPU otherwise. It reaches
neither CoreML, which this port refuses, nor DirectML, which nobody has
measured. Any other value is a requirement: if the
build does not carry that provider, `load` throws and names what is available.
It never falls back to CPU without saying so, because a benchmark row that
reads `cuda` and ran on CPU is worse than a failure.

Which providers you get is fixed when `onnxruntime-node` is installed, not
when loudkit runs. The package ships one prebuilt binary per platform and arch:

| platform | providers |
| --- | --- |
| darwin/x64, darwin/arm64 | `cpu`, `coreml` |
| linux/x64 | `cpu`, `cuda` |
| win32/x64, win32/arm64 | `cpu`, `directml` |

There is no separate npm package to install for CUDA or DirectML. For CUDA the
provider's own shared libraries are fetched by the package's postinstall step,
which `--onnxruntime-node-install=skip` turns off. For anything your platform's
prebuilt binary does not carry, build `onnxruntime-node` from source.

`availableProviders()` reports what the installed build actually offers. Ask it
rather than inferring from `process.platform`, which describes the download and
not a locally built binding.

`Enroller.load` takes the same option.

A GPU provider can change the token stream and waveform. This port refuses
CoreML because `onnxruntime-node` cannot persist its compile cache; use Swift or
one of the other ports for CoreML. Conformance runs pin CPU, and
`npm run test:fixture` accepts `--provider` for explicit comparisons. See
[`docs/benchmarks.md`](../docs/benchmarks.md#onnx-execution-providers).

CUDA was measured on an RTX 3090 with `onnxruntime-node` 1.26.0. The declared
1.27 series needs a newer NVIDIA driver than that machine had, so the result is
not a measurement of the default install. DirectML is unit-tested for
resolution and refusal text but has not been measured on Windows.

## Build and test

```bash
npm ci
npm test                 # weight-free conformance vectors
npm run test:all         # + engine conformance (needs checkpoint + graphs)
```

## Runtime libraries and assets

`onnxruntime-node` ships the native runtime as a package dependency. The
checkpoint, graphs and tokenizer are **not** bundled. Point the engine at the
paths on disk.
