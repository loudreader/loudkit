# Identity contract

What this project promises about its output, and what it does not. Current
cross-runtime measurements are in [Parity, measured](../parity-measured.md).

## The recipe in force: `loudkit-1`

Production fingerprint `79f71f5821477353`. Python, Swift, Go, Rust and
TypeScript compute it independently and agree.

`recipe_version` names the parts of the algorithm that are *code* rather than
settings: the sampling law, the window framing, the joins, the artifact
detectors. Two builds that agree on every configured value can still compute
different things, so the code's own version travels inside the fingerprint. It
is bumped whenever what comes out changes, and a bump re-bases the goldens.

What `loudkit-1` includes beyond the bare law:

- **Join continuity.** `ChunkConfig.prefix_tokens` is 6, not 0. Each chunk
  carries a tail of the previous chunk's tokens, so the pitch contour crosses a
  join instead of restarting ~74 Hz higher (measured on the reference voice;
  ~7 Hz with the prefix).
- **The postprocess layer.** The detectors in `docs/reference/postprocess.md`
  remove hallucinated tails, which makes the audio for an affected chunk shorter
  than the raw generation. Clean rows are untouched. The two end-to-end
  conformance cases produce byte-identical waveforms with the layer on and off.

## What we promise

**I-2, per-build determinism.** Same seed, same build, same backend, same input
produces a **bit-identical waveform**, every time, forever. This is the property
users observe and tests need, and we hold it absolutely.

**I-4, one sampling law.** Every backend implements the same sampling
mathematics from the same counter-based RNG stream, so the same seed means the
same *decisions* given the same logits.

## What we do not promise

- Bit-identical output **across backends** (CUDA / CPU / ANE). Ruled out by
  reduction order, FMA contraction, differing transcendental implementations, and
  the ANE's fixed fp16 pipeline. Our own static-cache measurement is this in
  miniature: 1.34e-05 of drift from nothing but a padded reduction.
- Identical **speech tokens across ONNX execution providers**. `onnx_provider`
  changes which kernels run, and it sits outside the fingerprint. The providers
  loudkit offers are measured to agree today (see below), but that is a
  measurement to repeat, not a promise. The CPU provider is the reference.
- Bit-identical output **across releases**. When the engine changes, the goldens
  are re-baselined once, under a bumped contract version, by script only.
- That a given sentence renders identically on your machine and ours.

## Exact cross-backend token identity, and its precondition

The headline "same speech tokens on every backend" has a measured precondition
and a measured exception.

* **Holds** for the default path at matched precision. The conformance fixture
  pins exact free-run tokens across torch (CPU/CUDA/MPS), ONNX, CoreML and all
  three bindings. This is what I-4 means in practice: same logits from the same
  precision, same counter-based RNG decisions. **The ONNX half of that fixture
  runs on the CPU execution provider.** It says nothing about the others; see
  the next bullet.
* **Holds across the ONNX execution providers loudkit offers, as measured.**
  `onnx_provider` is an execution knob and stays outside the fingerprint, so
  this is a measurement rather than a guarantee. What it says today:

  * **CUDA.** The same speech tokens as the CPU provider, byte for byte, in
    Python, Rust, Go and JS. Measured on an RTX 3090.
  * **CoreML.** The same speech tokens as the CPU provider, index for index.
    This is a consequence of placement rather than of numerics: `coreml` puts
    only the three renderer graphs on CoreML and keeps the generator on CPU, so
    nothing that decides a token ever runs there. Measured on an Apple M3 Pro,
    macOS 26.1, onnxruntime 1.28.0, on the third passage of the shipped
    benchmark set (voice `joe`, seed 1234): 392 tokens with digest
    `bf01efb39a3dbcda` in Python, Rust and Go alike, on both providers.
  * **The waveform is a separate question, and it does differ.** The same
    passage rendered `68f0de69...` on the CPU provider and `87988cc9...` on
    CoreML. Running the renderer somewhere else changes the last bits of the
    audio, which is exactly what the clause above declines to promise.

  Two things follow.

  1. **The token stream is reproducible from the recipe and the seed**, across
     the providers named here, without also pinning the provider. Pin `cpu`
     anyway when checking against the conformance fixture, which is a `cpu`
     measurement.
  2. **A CoreML configuration is part of the measurement.** These numbers are
     for `ModelFormat=MLProgram`. The CoreML default (NeuralNetwork) is a
     different computation, not merely a slower one: a vocoder under it sums
     217.70 where the CPU provider sums 211.15 and MLProgram sums 211.149.
     loudkit never selects the default; a caller reaching past it is outside
     what was measured.

* **English is the measured envelope; Polish is not guaranteed to be
  token-identical across every backend.** ONNX-Runtime's CPU graph fusion
  differs numerically from torch eager even at matched fp32: measured per-step
  logit drift ~1e-2 (torch fp32 vs ORT fp32, teacher-forced). English speech has
  a sparse top-set, so that drift almost never crosses a sampling decision
  boundary. The fixture's three sentences match token-for-token (79/79, 190/190,
  178/178) and free-run agree on short and long English (82/82, 204/204). Polish,
  after the respelling funnel, has a denser top-set, and the same ~1e-2 drift
  *does* cross a boundary. Measured on the Polish sentence in
  `tools/compare_backends.py`: ONNX diverges from torch fp32 at token 4, and
  both torch and MPS agree with each other 156/156. Polish is therefore the
  contract's `equivalent` class on ONNX: same distribution, not the same stream.
  MPS fp32 matches torch fp32 bit-for-bit (156/156) on the same sentence.
* **Does not hold** when an execution flag changes the reduction order without
  changing the sampling law. `--cuda-graphs` runs the decode over a static KV
  cache whose padded attention switches the cuBLAS kernel at large widths.
  Measured logit drift is ~2e-4 per layer at a 750-token prefill, and it flips a
  sampled token on long windows (a 255-token utterance diverges around token
  26-130). That is the identity contract's `equivalent` class: deterministic,
  same distribution, not the same stream. The default (dynamic cache) path stays
  bit-identical.
* **Warning**: the engine emits a `RuntimeWarning` when `--cuda-graphs` (or
  `compile_model`) is enabled, since that path is outside the proven
  token-identical envelope. If a future change proves token identity on long
  windows, the warning can be dropped. Until then it gates the flag honestly.

## Classification every change must declare

| class | meaning | examples | test that catches a violation |
|---|---|---|---|
| **bit-exact** | same arithmetic, same order | removing a weight-norm reparameterisation, deleting a host sync | golden waveform hash, unchanged |
| **equivalent** | deterministic, different reduction order | CUDA graphs / static KV cache, kernel fusion, Triton GEMVs, flash attention | golden tokens unchanged; logit drift within band; waveform correlation band |
| **changes-maths** | different numerics | fp16, TF32, int8 | full gate: KL, top-1, length/EOS watchdog, audio band; goldens re-baselined under a new contract version |

A change that cannot state its class does not merit review.

## The sampler: LR-SAMPLER-v1

I-4 needs one sampling algorithm that three implementations can agree on, which
rules out anything defined by a vendor library. The reference implementation in
`python/loudkit/sampler.py` is the specification, ported verbatim to
`go/sampler`, `rust/src/sampler.rs`, `swift/LoudKit/Sampler.swift` and
`js/src/sampler.ts`:

- **Philox-4x32-10**, counter-based, addressed by `(seed, stream, step, index)`.
  Integer-only, so its algorithm defines it rather than whoever compiled it.
  Verified bit-exact against the three Random123 published KAT vectors, which
  makes a Swift or Rust port checkable against a standard rather than against us.
- **Stateless.** A token's random number depends on its counter alone, not on how
  many numbers were drawn before it, so CPU and GPU may compute them in any order
  and still agree.
- **min_p in logit space**: keep `i` where `z_i/T >= max(z/T) + ln(min_p)`.
  Selection is identical to `p_i >= min_p * p_max` because softmax is monotone
  and its normaliser cancels. There is no exponential, no sum and no
  renormalisation, so no reduction whose order a backend could vary.
- **Gumbel-argmax** instead of a CDF scan, ties broken by lowest index.

Measured: **CPU and CUDA agree on 64 of 64 tokens.** The vendor-library
sampler this replaces agrees on **17 of 64**. Total variation against the
upstream sampling law is 0.00286 against a sampling-noise floor of 0.00227, so
it is the same law drawn from a different stream.

Cost: **+0.078 ms/token, about 4% of a T3 step**, because in eager torch ops this
is several elementwise kernels plus an argmax over 8194. It fuses away when the
sampler moves into the decode kernel. The cost is tracked but not gated: a speed
gate here would only tempt us to make the sampler non-portable again, which is
the thing it exists to fix.

Adopting it re-bases the goldens once: same law, different stream, so different
tokens. That is what a contract version is for.

## Measurement rules that make the above meaningful

1. **Pin both TF32 flags and record them.** `cudnn.allow_tf32` defaults True and
   `cuda.matmul.allow_tf32` defaults False, so "PyTorch fp32" is neither fp32 nor
   bit-exact with it. Measured 5% faster and non-identical on s3gen.
2. **Seed the global torch RNG before every render.** s3gen draws its CFM prior
   and HiFiGAN excitation from it. Unseeded, two renders of identical tokens
   correlate at 0.11-0.31.
3. **Never judge quality on a render from random speech tokens.** That is babble.
4. Every golden carries a manifest: torch version, GPU, both TF32 flags, dtype,
   contract version. A number without its manifest is folklore.

## How to describe this in prose

> Deterministic by build: the same text, voice, and seed produce a bit-identical
> waveform every time on a given build and device. Output is **not** guaranteed
> to match across backends (CUDA / CPU / Apple Neural Engine) or across releases.
> Different hardware reduces floating-point sums in different orders, and engine
> changes are re-baselined under a versioned identity contract. What is held
> constant across all of them is the sampling law and the voice.

Never write "identical waveform" unqualified.
