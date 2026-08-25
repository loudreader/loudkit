# The Apple path: CoreML artefacts and the Swift package

**Requirements:** Apple Silicon, macOS 14+ (iOS 17+ for the package). The
package declares `swift-tools-version: 5.9`.
CoreML artefacts are exported with `coremltools` 9 under torch ≤ 2.7. Figures on
this page were measured on an Apple Silicon laptop. Results depend on hardware.

Two deliverables, one gate:

- the CoreML renderer graphs, re-exported from the packed checkpoint;
- a Swift package (`LoudKit`, at the repo root) that renders the same audio as
  the Python engine. A conformance fixture that both languages read proves it.

## The artefacts and how to rebuild them

The exported packages are **not** in git (they are weights). The release ships
them in a `coreml/` directory beside the checkpoint, and one fetch gets the
whole working set:

```bash
loudkit download loudreader/loudr-1 --for coreml --with-cloning --local-dir loudr-1
```

Drop `--with-cloning` for synthesis alone. For CoreML it adds the three
enrollment packages, which Swift enrollment needs. It does not add the torch
enrollment weights: those are the Python enroller's, and Swift never opens
them.

Both the Python coreml backend and the Swift package look beside the
checkpoint by default:

```
checkpoints/loudr-1/
  loudr-1.safetensors           # synthesis checkpoint, 747 MB
  manifest.json                 # human-readable mirror
  tokenizer.json
  voices/                       # all twenty profiles, always fetched
  coreml/
    flow_encoder.mlpackage      # fp32, CPU; fp16 here is measured fatal
    flow_estimator.mlpackage    # fp16, CPU+ANE
    vocoder.mlpackage           # fp32, CPU; fp16 puts a tone at Nyquist
    s3_tokenizer.mlpackage      # enrollment, with --with-cloning
    camp.mlpackage              # enrollment, with --with-cloning
    voice_encoder.mlpackage     # enrollment, with --with-cloning
```

Rebuild from the packed checkpoint. `coremltools` needs torch ≤ 2.7. These were
built against 2.6.0, and 2.13 traces fail inside the converter:

```bash
<torch2.6-venv>/bin/python tools/export_coreml.py \
    --checkpoint /path/to/loudr-1.safetensors
```

The script gates every stage against the torch modules loaded from the same
checkpoint before a package is moved into place. This build's gates:

| stage | vs torch (same weights) | note |
|---|---|---|
| flow_encoder | corr 1.0000000, max\|Δ\| 1.8e-06 | fp32 conversion, transparent |
| flow_estimator | corr 0.9999590, max\|Δ\| 1.2e-01 | the fp16 pipeline band |
| vocoder | corr 1.0000000, max\|Δ\| 3.8e-05 | conv-STFT rewrite proven at 8.8e-07 first |

The graph geometry is the shipped app's (query 255 / prompt 238 → T986 mel,
HiFT at 510 frames), and the weights trace to one file. All render randomness
(flow prior, harmonic phases, excitation noise) is a graph *input*, drawn from
loudkit's Philox streams on both sides.

The static window recipe and the EOS floor are manifest-borne, and the tensor
payload is re-hashed unchanged before and after the manifest amendment.
The Swift implementation carries **no fallback constants** for them: it refuses
an un-amended checkpoint rather than re-guess the framing that was once the
entire measured ANE-vs-torch deviation.

## The Swift package

`Package.swift` sits at the repo root, sources in `swift/LoudKit`, tests in
`tests/LoudKitTests`. The lowercase test path is deliberate: on the
case-insensitive dev filesystem `Tests` and the Python `tests` directory are one
directory, and the manifest must name the real one for case-sensitive checkouts.

The API mirrors the Python engine closely enough to show side by side:

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
try result.save(to: outURL)                                // audio, tokens, mel, timings
```

The config split is mirrored too: `AlgorithmConfig` (built from the checkpoint
manifest, fingerprint-compatible with Python, see below) and a per-device
`ExecutionConfig` (compute units per CoreML stage, generator precision).

Execution layout, declared not implied:

| stage | where | precision |
|---|---|---|
| token generator | native Swift (Accelerate BLAS), CPU | fp32 compute over the packed fp16 weights |
| flow encoder | CoreML, CPU | fp32 |
| flow estimator | CoreML, CPU + Neural Engine | fp16 pipeline |
| vocoder | CoreML, CPU | fp32 |

The generator is native rather than a CoreML graph for two reasons. The app's
stateful T3 export has no validated cross-implementation harness (see "not
covered" below). CPU is also the measured-right placement for the autoregressive
stage on Apple silicon. fp32 is the declared conformance precision: token
identity across implementations holds *at matched precision* (identity
contract), and fp32-from-fp16-storage is what the Python conformance engine
runs.

## Conformance: the gate, and how it is checked

One fixture, `tests/data/conformance/`, generated by
`tools/make_conformance.py` and read by **both** `pytest`
(`tests/test_conformance.py`) and `swift test`. Layers:

- **Philox**: the three Random123 KAT vectors, raw uniform *bits* for fixed
  `(seed, stream, step, index)` (exact, integer), and gumbel probes at 1e-12
  (allowance for a foreign libm's last ulp).
- **LR-SAMPLER-v1**: token choices for literal logits, a silence-exemption
  case, and a full-vocab (8194) case whose logits are derived from Philox bits
  so both languages regenerate the identical float32 input. The Swift sampler
  is the third independent implementation of the law, and it matches choice for
  choice.
- **Text frontend**: token ids for trap sentences (punctuation, Polish
  diacritics, doubled whitespace), with the tokenizer JSON in the fixture so
  this layer needs no weights.
- **Algorithm identity**: the production fingerprint *and* the exact canonical
  form it hashes. Swift implements `canonicalForm()` against the same rules
  (floats as shortest-round-trip `repr` strings, sorted keys, schema
  envelope) rather than storing Python's output. Both languages compute the
  fingerprint independently and agree: `79f71f5821477353`.
- **Seed derivation**: the per-stage splitting constants, as hex (a u64 does
  not survive a JSON double).
- **End to end**: two sentences, text + voice + seed → speech tokens (exact),
  mel and waveform (banded), rendered by the Python coreml backend with the
  generator declared fp32.

Measured on the reference Apple Silicon build. Results vary by hardware and OS
version:

| comparison | s0 (79 tok) | s2 (157 tok) |
|---|---|---|
| Swift tokens vs Python tokens, same seed | **79/79 exact** | **157/157 exact** |
| Swift mel vs Python coreml mel | corr 1.000000000 | corr 1.000000000 |
| Swift waveform vs Python coreml waveform | **bit-identical** (max\|Δ\| 0) | **bit-identical** (max\|Δ\| 0) |
| Python coreml (re-export) vs torch reference, fixed tokens¹ | mel 0.9999923 | mel 0.9999914 |

¹ third row measured on the parity sentences s0/s1/s2 vs
`tests/data/reference` (s1: 0.9999914; waveform 0.9973 / 0.9886 / 0.9717, since
the vocoder's predicted-phase channel decorrelates the waveform while the
spectrum stays put; mel is the quality gate recorded in
[Parity, measured](../parity-measured.md)). The
measured band (≥ 0.999) is the **gate** the tests assert. The bit-identity in
rows 2 and 3 is what this machine *observed* (Swift consumes identical Philox
bytes and drives the identical deterministic graphs) and is not promised across
machines or ANE generations.

Reproduce:

```bash
.venv/bin/python -m pytest -q             # the Python suite
swift test                                # same fixture, from Swift
# regenerate the fixture only when the engine legitimately changes:
.venv/bin/python tools/make_conformance.py --checkpoint …
```

The weight-free vectors run on any machine. The algorithm-identity and
end-to-end tests need the checkpoint (`LOUDKIT_CHECKPOINT`, or the dev default)
and the exported packages. Without them they skip with a named reason.
`LOUDKIT_REQUIRE_ASSETS=1` turns those skips into failures, same rule as the
Python suite.

## Not covered

- **T3 on the Neural Engine is still not covered by either language.** The
  app's stateful multi-function T3 export remains without a validated
  cross-implementation harness. On the Python side torch's decode loop
  segfaults after coremltools loads in the same process (hit again while
  generating this fixture, worked around by a two-process split). The Swift
  package runs the generator natively on CPU instead. A row synthesised from an
  unvalidated export is worse than a missing row.
- **fp16 generator tokens are not the conformance claim.** The fixture
  declares fp32. On the two fixture sentences the shipping fp16 map produced
  identical tokens: observed, not promised. fp16 flips ~1 token in a thousand,
  and one flip re-routes everything after it.
- **iPhone numbers are not from this pass.** Everything above was measured on
  an Apple Silicon Mac. The package compiles for iOS 17+, but per this
  project's own rule (measure on device) the A16 row in
  [benchmarks](../benchmarks.md) keeps its provenance from the app's engine,
  not from this package, until it is measured here.
- **Enrollment is Swift-supported over CoreML.** The chunking, streaming and
  long-form composition landed in `LoudKit` (`Chunking.swift`, `Engine.stream`).
  Enrollment runs over the three exported CoreML graphs
  (`s3_tokenizer.mlpackage`, `camp.mlpackage`, `voice_encoder.mlpackage`), held
  to the same enrollment fixture as the Python, Go, Rust and JS ports. One
  Apple-specific caveat: `camp.mlpackage` is exported at the fixed 998-frame
  geometry the 10 s-capped enrollment always produces, because coremltools
  lowers `avg_pool1d(ceil_mode=True)` wrong under a dynamic dimension.
- **The port is measured, and it is slower than the Python engine.** On an
  M3 Pro (11-core CPU, 14-core GPU, 36 GB, macOS 26.1, Swift 6.2, release
  build) the whole pipeline runs the third benchmark passage at 2.11x real
  time cold and 2.28x warm, median of three. Engine load takes 11.8s. The
  Python engine on `--device mps`, same machine, same passage, same voice and
  seed, runs the same call at 3.50x cold and 3.25x warm. Both figures come
  from `synthesize_long`. The generator runs natively on the CPU in fp32,
  because that is where the port puts it; the estimator ran on
  `cpuAndNeuralEngine`, the encoder and vocoder on `cpuOnly`. The renderer,
  which is the CoreML part, is the same graphs at the same speed on both
  sides.

## The in-process crash, and why the backend is safe to embed

Until this was fixed, `loudkit.load(device="coreml")` killed its host about one
second after a synthesis returned. The audio was correct; the process died
afterwards, so a server, a notebook or any script that did a second thing was
taken down by a call that had already succeeded.

The mechanism is upstream, in coremltools, and it is still unfixed in 9.0
(apple/coremltools#2827, open, no Apple response). coremltools wraps each
prediction input without copying and keeps the Python array as an Objective-C
ivar on the feature value. CoreML does not release that feature value when
`predict` returns: the MLE5 execution stream lingers and resets itself about a
second later on `com.apple.coreml.MLE5ExecutionStream.resetQueue`. The release
therefore runs on a dispatch thread that holds no GIL. If the interpreter's own
reference is gone by then, the release reaches `_PyObject_Free` and corrupts the
allocator, and the fault lands in whatever the main thread is doing.

The backend now keeps a reference of its own. Each model owns one buffer per
input for its lifetime and every predict copies into it, so CoreML's off-thread
release only ever drops a count from two to one and the real free happens on a
thread that holds the GIL. The exported graphs are static, so this is one buffer
per input, allocated once. Measured on an M3 Pro, macOS 26.1, coremltools 9.0:
waveform bit-identical to the unpinned path, wall time 2.09 s against 2.06 s on
a 4.96 s sentence, median of four, which is inside run-to-run noise.

`tests/test_coreml_lifetime.py` is the regression test. It renders in a child
process and asserts the child's exit status after it has outlived the linger
window, because a test that only checks the synthesis would pass while the
process was already doomed.

Two things this does **not** claim. It does not make coremltools safe for other
callers: anything in this repo that calls `MLModel.predict` outside this backend
carries the same hazard, which is why `tools/export_enroll_coreml.py`,
`tools/make_conformance.py` and `tools/build_release.py` still isolate their
CoreML work in subprocesses. And it is not known to be the same fault as the T3
decode-loop segfault recorded above; that one was never traced to a stack.

## Two doors to CoreML, and which one you are using

CoreML is reachable two ways, and they are different artefacts with different
properties.

**The exported packages**, described above: `coreml/*.mlpackage`, built by
`tools/export_coreml.py`. This is what the Swift package uses, and what the
Python `coreml` backend uses. It is the Apple-native path.

**The ONNX CoreML execution provider**, reached with `onnx_provider="coreml"`
from the Python, Rust and Go ports, and from the command line with
`--device onnx --provider coreml`. Here CoreML runs the same exported ONNX
graphs the CPU provider runs. It is a *placement*, not a whole-engine switch:
the three renderer graphs go to CoreML with `ModelFormat=MLProgram`, and the
generator stays on the CPU. The generator has no winning CoreML configuration —
`t3_step` runs once per speech token at 9.8 ms on CPU against 17.6 ms for the
best CoreML variant, and `t3_prefill` and `t3_step` fail to compile under
MLProgram at all.

Because nothing that decides a token runs on CoreML, the speech tokens are
identical to a CPU run. The waveform is not bit-identical.

The first run on a machine compiles the renderer graphs, which takes about two
minutes. That is cached in `~/Library/Caches/loudkit/coreml`, about 1.6 GB, and
later runs open in about 25 s against 3 s for the CPU provider.
`$LOUDKIT_COREML_CACHE` moves the directory. `auto` never selects `coreml`,
because a default may not spend two minutes and 1.6 GB without being asked.

**JS has no CoreML.** `onnxruntime-node` cannot name a cache directory, so every
process would pay the compile again. The port refuses the provider rather than
offering that. See [the JS guide](../guides/07-js-ts.md).
