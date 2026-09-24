# Apple: CoreML packages and the Swift package

**Requirements:** macOS 14 or later. The Swift package also runs on iOS 17 or
later and declares `swift-tools-version: 5.9`. The CoreML packages are
exported with `coremltools` 9. All figures on this page come from one Apple
Silicon laptop, an M3 Pro.

loudkit on Apple has two parts:

- the CoreML packages, exported from the checkpoint and loaded by the Python
  `coreml` backend and the Swift package;
- the Swift package, `LoudKit`, with its `Package.swift` at the repository
  root.

The Swift package and the Python engine read one shared conformance fixture.
The tests compare their tokens, mel spectrograms and waveforms.
[Conformance tests](#conformance-tests) lists the checks and the measured
results.

## The CoreML packages

The exported packages are not in git. The release ships them in a `coreml/`
directory beside the checkpoint. This command fetches everything the Swift
package and the Python `coreml` backend need:

```bash
loudkit download loudreader/loudr-1 --for coreml --with-cloning --local-dir loudr-1
```

For synthesis only, leave out `--with-cloning`. With `--for coreml`,
`--with-cloning` adds the three CoreML enrollment packages, which Swift
enrollment needs. It does not add the PyTorch enrollment weights
(`loudr-1-enrollment.safetensors`, `ve.safetensors`), which the Swift package
does not use.

The Python `coreml` backend and the Swift package look for `coreml/` beside
the checkpoint by default:

```
loudr-1/
  loudr-1.safetensors           # synthesis checkpoint, 747 MB
  manifest.json                 # human-readable mirror
  tokenizer.json
  voices/                       # all 28 profiles, always fetched
  coreml/
    export.json                 # export record: source checkpoint, tool versions
    flow_encoder.mlpackage      # fp32, CPU
    flow_estimator.mlpackage    # fp16, CPU+ANE
    vocoder.mlpackage           # fp32, CPU
    t3_cond.mlpackage           # token generator, Python coreml backend only
    t3_prefill.mlpackage        # token generator, Python coreml backend only
    t3_step.mlpackage           # token generator, Python coreml backend only
    s3_tokenizer.mlpackage      # enrollment, with --with-cloning
    camp.mlpackage              # enrollment, with --with-cloning
    voice_encoder.mlpackage     # enrollment, with --with-cloning
```

The flow encoder and the vocoder stay fp32. In fp16, the encoder's mel
correlation drops to 0.619, and the vocoder adds a tone at the Nyquist
frequency. [execution-config.md](../design/execution-config.md) has the
measurements.

### Rebuild the packages

Rebuild the packages from the matching checkpoint:

```bash
python tools/export_coreml.py \
    --checkpoint /path/to/loudr-1.safetensors
```

The script checks every stage against the PyTorch module loaded from the same
checkpoint before it moves a package into place. A stage that fails the check
leaves the existing package unchanged. Results of the loudr-1 renderer export:

| stage | against PyTorch (same weights) | note |
|---|---|---|
| flow_encoder | corr 1.0000000, max\|Δ\| 1.8e-06 | fp32 conversion |
| flow_estimator | corr 0.9999590, max\|Δ\| 1.2e-01 | fp16, inside the fp16 tolerance |
| vocoder | corr 1.0000000, max\|Δ\| 3.8e-05 | STFT rewritten as convolutions, 8.8e-07 from the original STFT before conversion |

These loudr-1 exports used torch 2.6.0. The turbo estimator was exported with
torch 2.13.0 and coremltools 9.0 and passed the waveform conformance test.
That result does not cover every exporter with that combination. `export.json`
records the installed tool versions of each new export in its `toolchain`
field. Entries without that field have no recorded tool versions.

The graphs are static: a query of 255 tokens and a prompt of 238 tokens give
986 mel frames, and the vocoder (HiFT) runs at 510 frames. All weights come
from the one checkpoint file. All render randomness (flow prior, harmonic
phases, excitation noise) is a graph input, drawn from loudkit's Philox
streams in both languages.

The checkpoint manifest holds the static window (query length, prompt length,
pad token) and the EOS floor. The Swift package reads them from the manifest
and has no built-in window values. It refuses a checkpoint whose manifest does
not declare the 255/238 static window. A manifest without an `eos_floor` block
gets the shared default, no floor, as in the other implementations.

## The Swift package

`Package.swift` is at the repository root. The sources are in `swift/LoudKit`
and the tests are in `tests/LoudKitTests`. `LoudKit` chunks long text and
streams (`Chunking.swift`, `Engine.stream`).

The Python and Swift APIs have the same shape:

```python
import loudkit as lk

engine = lk.load("loudr-1/loudr-1.safetensors")
voice = lk.VoiceProfile.load("loudr-1/voices/joe.safetensors")
engine.synthesize("Hello there.", voice, seed=7).save("out.wav")
```

```swift
import LoudKit
let engine = try Engine.load(checkpoint: checkpointURL)   // coreml/ found beside it
let voice  = try VoiceProfile.load(url: voiceURL)
let result = try engine.synthesize("Hello there.", voice: voice, seed: 7)
try result.saveWav(to: outURL)                             // 16-bit PCM WAV
```

Both split the configuration the same way:

- `AlgorithmConfig` is built from the checkpoint manifest. Its fingerprint is
  compatible with Python's; see [Conformance tests](#conformance-tests).
- `ExecutionConfig` holds the per-device settings: the compute units of each
  CoreML stage.

| `ExecutionConfig` property | default | stage |
|---|---|---|
| `encoderComputeUnits` | `.cpuOnly` | flow encoder |
| `estimatorComputeUnits` | `.cpuAndNeuralEngine` | flow estimator |
| `vocoderComputeUnits` | `.cpuOnly` | vocoder |

Each property takes `.cpuOnly`, `.cpuAndNeuralEngine` or `.all`. Pass the
config to `Engine.load(checkpoint:coremlAssets:execution:)`,
`Engine.load(bundle:execution:)` or `Engine.load(_:revision:execution:progress:)`.
`engine.withExecution(_:)` rebuilds only the CoreML stages with a new config.
`tokenGeneratorPrecision` is informational: the generator always computes in
fp32. The defaults come from measurements on the M3 Pro. On other hardware,
other settings can be faster.

Where each stage runs by default:

| stage | where | precision |
|---|---|---|
| token generator | native Swift (Accelerate BLAS), CPU | fp32 compute over the packed fp16 weights |
| flow encoder | CoreML, CPU | fp32 |
| flow estimator | CoreML, CPU + Neural Engine | fp16 for loudr-1; fp32 for turbo |
| vocoder | CoreML, CPU | fp32 |

The token generator is native Swift code. It does not use the
`t3_*.mlpackage` graphs. On Apple silicon it measured faster on the CPU than
on the GPU or the Neural Engine at batch one. It computes in fp32 from the
fp16 weights in the checkpoint, the same precision the Python conformance
engine runs. The fixture declares fp32,
because token identity across implementations holds only at matched precision
(see the [identity contract](../reference/IDENTITY-CONTRACT.md)). No
conformance harness covers a stateful CoreML export of the generator; see
[Limits](#limits).

### The first synthesis

The first synthesis after launch is slow. CoreML specializes a loaded
`.mlmodelc` for the device on first use, as the PyTorch MPS backend compiles a
Metal pipeline on first use. Both caches belong to the OS and outlive the
process. Measured on the M3 Pro with the shipped vocoder graph:

- first fresh process: 0.43 s to load, 0.39 s for the first prediction, 0.19 s
  for each later prediction;
- next fresh process: 0.12 s to load, 0.25 s for the first prediction.

Run one synthesis in a background task at launch and discard the audio. Every
stage draws its random numbers from the call's own seed (see
[the pipeline notes](../design/engine-pipeline.md)), so this warm-up does not
change the output of later calls. The server entry points can warm up the
same way; see [transports](../design/transports.md).

## Enrollment in the Swift package

The Swift package enrolls a voice with the three exported CoreML graphs
(`s3_tokenizer.mlpackage`, `camp.mlpackage`, `voice_encoder.mlpackage`). The
same enrollment fixture checks it and the Python, Go, Rust and JS
implementations. `camp.mlpackage` is exported at a fixed input of 998 frames,
the length that the 10 s enrollment limit always produces, because
coremltools converts `avg_pool1d(ceil_mode=True)` incorrectly under a dynamic
dimension.

## Speed of the Swift package

On an M3 Pro (11-core CPU, 14-core GPU, 36 GB, macOS 26.1, Swift 6.2.1,
release build), the Swift package runs the third benchmark passage at 2.49x
real time with loudr-1 and 3.44x with loudr-1-turbo, measured on 0.1.1
(2026-09-06) as medians of three warm streams. First audio arrives after
1.93 s and 1.38 s. Engine load takes 11 s for loudr-1 and 4 s for turbo. On the
same machine, passage, voice and seed, the Python engine with `--device mps`
runs at 3.29x and 5.77x. The raw runs are in
[2026-09-06-m3pro-both-models.json](../measurements/2026-09-06-m3pro-both-models.json).

In these runs the token generator ran natively on the CPU in fp32, the flow
estimator on `cpuAndNeuralEngine`, and the encoder and vocoder on `cpuOnly`.
The renderer runs the same CoreML graphs as the Python `coreml` backend.

## Conformance tests

One fixture, `tests/data/conformance/`, generated by
`tools/make_conformance.py`, is read by both `pytest`
(`tests/test_conformance.py`) and `swift test`. It has these layers:

| layer | what the tests check |
|---|---|
| Philox | The three Random123 known-answer vectors. The raw uniform bits for fixed `(seed, stream, step, index)`, exact as integers. Gumbel probes at a tolerance of 1e-12, for last-ulp differences between math libraries. |
| LR-SAMPLER-v1 | Token choices for literal logits, a silence-exemption case, and a full-vocabulary case (8194 entries) whose logits come from Philox bits, so both languages build the same float32 input. The Swift sampler makes the same choice as the Python sampler in every case. |
| Text frontend | Token ids for edge-case sentences (punctuation, Polish diacritics, doubled whitespace). The fixture holds the tokenizer JSON, so this layer needs no weights. |
| Algorithm identity | The fingerprint and the exact canonical form it hashes. Swift builds `canonicalForm()` from the same rules (floats as shortest round-trip `repr` strings, sorted keys, schema envelope) and does not store Python's output. Both languages compute the same fingerprint, `7cd75498ad4e7531`. |
| Seed derivation | The per-stage splitting constants, as hex strings, because a JSON double cannot hold every u64 exactly. |
| End to end | Two sentences: text, voice and seed to speech tokens (exact), and to mel and waveform (within a tolerance band), rendered by the Python `coreml` backend with the generator declared fp32. |

Measured results:

| comparison | s0 (79 tok) | s2 (157 tok) |
|---|---|---|
| Swift tokens vs Python tokens, same seed | **79/79 exact** | **157/157 exact** |
| Swift mel vs Python coreml mel | corr 1.000000000 | corr 1.000000000 |
| Swift waveform vs Python coreml waveform | **bit-identical** (max\|Δ\| 0) | **bit-identical** (max\|Δ\| 0) |
| Python coreml (re-export) vs torch reference, fixed tokens¹ | mel 0.9999923 | mel 0.9999914 |

¹ The fourth row was measured on the parity sentences s0, s1 and s2 against
`tests/data/reference` (s1: 0.9999914). The waveform correlations are 0.9973,
0.9886 and 0.9717: the vocoder's predicted-phase channel lowers the waveform
correlation, while the spectrum does not change. Mel correlation is the
quality check recorded in [Parity, measured](../parity-measured.md).

The tests require a mel correlation of at least 0.999. The bit-identical
results in rows 2 and 3 were measured on one machine. Swift reads the same
Philox bytes and runs the same deterministic graphs, but bit identity is not
promised on other machines or Neural Engine generations.

Reproduce:

```bash
.venv/bin/python -m pytest -q             # the Python suite
swift test                                # same fixture, from Swift
# regenerate the fixture only after an intended change to the algorithm:
.venv/bin/python tools/make_conformance.py --checkpoint …
```

The weight-free vectors need no model files. The algorithm-identity and
end-to-end tests need the checkpoint and the exported packages. Set
`LOUDKIT_CHECKPOINT` to the checkpoint file, or `LOUDKIT_ASSET_ROOT` to a
directory that holds it. The default is the repository's `assets/` directory.
Without the files, these tests skip and name the reason.
`LOUDKIT_REQUIRE_ASSETS=1` turns those skips into failures, the same rule as
the Python suite.

## Limits

- Neither language runs the token generator (T3) on the Neural Engine. No
  validated cross-implementation harness exists for a stateful multi-function
  CoreML export of T3. The Swift package runs its native generator on the CPU.
  In Python, the PyTorch decode loop crashes (segfault) when coremltools is
  loaded in the same process, so `tools/make_conformance.py` generates the
  fixture tokens in a separate process.
- fp16 generator tokens are not part of the conformance fixture, which
  declares fp32. On the two fixture sentences, the shipped fp16 generator
  produced the same tokens, but that is not promised. fp16 changes about 1
  token in 1000, and each changed token changes every token after it.
- The package builds for iOS 17 and later, but it has not been measured on an
  iPhone.

## The Python `coreml` backend in a long-running process

`loudkit.load(device="coreml")` can run in a server, a notebook or any process
that continues after a synthesis. The backend guards against a coremltools 9.0
bug that can crash such a process about one second after a synthesis returns
correct audio
([apple/coremltools#2827](https://github.com/apple/coremltools/issues/2827)).

coremltools wraps each prediction input without copying it. CoreML releases
that input about one second after `predict` returns, on a dispatch thread that
does not hold the GIL. If Python holds no other reference by then, the release
corrupts the Python allocator, and the process crashes in whatever it does
next.

The backend keeps its own reference to each input buffer for the life of the
model, and every prediction copies into that buffer. The final release then
happens on a thread that holds the GIL. A buffer grows when its input grows,
for example the token generator's KV cache.
[execution-config.md](../design/execution-config.md) describes the mechanism.

Measured on an M3 Pro, macOS 26.1, coremltools 9.0: the waveform is
bit-identical to the path without the guard, and the wall time is 2.09 s
against 2.06 s for a 4.96 s sentence (median of four), inside run-to-run noise.

`tests/test_coreml_lifetime.py` renders in a child process and checks the
child's exit status after the release delay.

The guard covers only predictions made through this backend. Other code that
calls `MLModel.predict` directly has the same risk.
`tools/export_enroll_coreml.py`, `tools/make_conformance.py` and the release
tool (`tools/build_release.py`) run their CoreML predictions in subprocesses.
It is not known whether the T3 decode-loop crash in [Limits](#limits) has the
same cause.

## CoreML packages and the ONNX CoreML provider

loudkit reaches CoreML in two ways, with different files:

- The exported packages described above (`coreml/*.mlpackage`, built by
  `tools/export_coreml.py`). The Swift package and the Python `coreml` backend
  use them.
- The ONNX CoreML execution provider: `onnx_provider="coreml"` in the Python
  engine and the Rust and Go ports, or `--device onnx --provider coreml` on the
  command line. CoreML then runs the same exported ONNX graphs that the CPU
  provider runs.

The ONNX CoreML provider moves only the three renderer graphs to CoreML, with
`ModelFormat=MLProgram`. The token generator stays on the CPU. `t3_step` runs
once per speech token and takes 9.8 ms on the CPU against 17.6 ms for the
fastest CoreML variant, and `t3_prefill` and `t3_step` do not compile under
MLProgram. Because the generator runs on the CPU, the speech tokens are
identical to a CPU run. The waveform is not bit-identical.

The first run on a machine compiles the renderer graphs, which takes about two
minutes. The result is cached in `~/Library/Caches/loudkit/coreml`, about
1.6 GB, and later runs open in about 25 s against 2.5 s to 2.9 s for the CPU
provider. These figures were measured before 0.1.0; see
[benchmarks](../benchmarks.md).
Set `LOUDKIT_COREML_CACHE` to use another directory. `auto` never selects
`coreml`. Request it by name.

The JS port refuses `coreml`. `onnxruntime-node` cannot set a cache directory,
so every process would compile the graphs again. See
[the JS guide](../guides/07-js-ts.md).

## Python: native or PyTorch token generator

By default, the Python `coreml` backend runs the native fp32 token generator
from the `t3_*.mlpackage` packages and does not need PyTorch. It has fewer
dependencies, but for loudr-1 it is not always the fastest choice. A loudr-1
release that ships no generator packages uses the PyTorch generator on the CPU.

To use the fp16 PyTorch generator with the CoreML renderer, set
`generator_device="cpu"`. The precision map then sets its dtype:

```python
import loudkit as lk
from loudkit.config import ExecutionConfig

engine = lk.load("loudreader/loudr-1", device="coreml",
                 execution=ExecutionConfig(generator_device="cpu", precision={
                     "token_generator": "fp16",
                     "mel_decoder.estimator": "fp16",
                 }))
```

This path needs the `torch` extra. For turbo, the CoreML estimator is fp32:
set `mel_decoder.estimator` to `fp32` in the example. The native generator
requires fp32. With the native generator, `"token_generator": "fp16"` raises
an error that asks for `generator_device="cpu"`.

Both generators implement the same sampling law, but different precisions can
give different logits, and so different tokens on arbitrary text. The native
fp32 token fixture does not cover fp16. Compare repeatability within one
execution setup. Swift always uses its native Accelerate generator; this
Python setting does not change it.
