# Identity contract

What this project promises about its output, and what it does not. Current
cross-runtime measurements are in [Parity, measured](../parity-measured.md).

## The recipe in force: `loudkit-1`

Production fingerprint `7cd75498ad4e7531`. Python, Swift, Go, Rust and
TypeScript compute it independently and agree. `loudr-1-turbo` runs the same
recipe in the `fusion_mtp2` decode mode with one Euler step; its fingerprint is
`e5303ba243087222`, and the same five runtimes agree on it.

A written fingerprint answers one of two questions, and which one decides
whether it moves. **The algorithm this build implements** is the pair above,
the conformance vectors, the ports' freeze tests, the changelog entry and
[COMPATIBILITY.md](COMPATIBILITY.md): they move together, in the commit that
moves the algorithm, and `tests/test_docs.py` holds them to the fixture.
**What an artefact was made under** is a record of a past event and does not
move: `docs/voices/roster/provenance.json` and
`docs/voices/preview/catalog.json` name the algorithm each published voice was
rendered with, `docs/measurements/` names the algorithm each run measured, and
the CI workflow names the algorithm each published bundle was built and
verified with. Re-pinning one of those to today's value states something that
did not happen. A voice is re-recorded, or its record stands.

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
- **Not every period ends a sentence.** `ChunkConfig.mid_sentence_period` is
  `"hold"`. A period whose next word starts in lower case, or whose token is in
  `ChunkConfig.abbreviations`, is not a chunk boundary, so `"But Mr. Smith went
  home"` no longer produces the chunk `"But Mr."`. `"break"` names the law as it
  stood before, for a pack measured under it.
- **A chunk the window could not hold is split, not shipped.**
  `ChunkConfig.cap_resplit` is `"word"`. The character budget is an estimate,
  and a chunk that overruns it stops mid-word with the remainder never spoken,
  because chunk texts are fixed before anything renders. Such a chunk is
  replaced by its two halves, cut at the word boundary nearest the middle. Both
  halves keep the original chunk's index, so no later chunk's seed moves; the
  audio of later chunks in that passage does move, because the carry now comes
  from the second half. `"off"` names the law as it stood before, for a pack
  measured under it.
- **The postprocess layer.** The detectors in `docs/design/postprocess.md`
  remove hallucinated tails, which makes the audio for an affected chunk shorter
  than the raw generation. Clean rows are untouched. The two end-to-end
  conformance cases produce byte-identical waveforms with the layer on and off.
- **The repetition penalty covers silence.** Since the interior-stall fix the
  penalty applies to silence ids like any other token; silence keeps only the
  `min_p` exemption. A `stall` verdict condemns a row the decoder spent
  trapped in silence into the retry ladder, and a starved `desperation` cut
  (a cap-hit trim keeping under `desperation_min_keep_per_text_token` speech
  tokens per text token) is condemned the same way, with the trim as the
  fallback. The repetition rule's all-silence-cycle exemption reads acoustic
  silence (`repetition_silence: acoustic`, the union of the sampler list and
  both render censuses), so a pause parked on a census-only silent id is
  never cut as a loop; it stalls and retries instead. And a loop the decoder
  resumed from is condemned into the retry ladder rather than cut
  (`repetition_resume: condemn`), which holds on a checkpoint with no
  censuses at all: an applied repetition cut only ever removes a tail. These
  are amendments to the law under this recipe name: each moved the
  fingerprint above and re-based the goldens, and the sampler's and
  postprocess docstrings record the measurements.

- **The decode loop is a property of the weights.** `decode.mode` enters the
  fingerprint, and a checkpoint carrying `fusion_mtp2` decodes two speech
  tokens per transformer forward ([why](../design/two-token-decode.md)). An
  absent block stays absent rather than defaulting to an explicit `"single"`,
  so no loudr-1 fingerprint moved for a checkpoint that decodes exactly as it
  always did. The mode is not a runtime option: a one-token loop over fusion
  weights reads half the context the model expects and speaks fluently about
  something else, so the mode is declared in the manifest, the file is
  `format_version 2`, and every engine that has not implemented the loop
  refuses it, by version and by mode, because the version is only as honest
  as the packer that wrote it.

## What we promise

**I-2, per-build determinism.** Same seed, same build, same backend, same execution configuration, same input
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
  `research/compare_backends.py`: ONNX diverges from torch fp32 at token 4, and
  both torch and MPS agree with each other 156/156. Polish is therefore the
  contract's `equivalent` class on ONNX: same distribution, not the same stream.
  MPS fp32 matches torch fp32 bit-for-bit (156/156) on the same sentence.
* **Does not hold** when an execution flag changes the reduction order without
  changing the sampling law. `cuda_graphs` runs the decode over a static KV
  cache whose padded attention switches the cuBLAS kernel at large widths.
  Measured logit drift is ~2e-4 per layer at a 750-token prefill, and it flips a
  sampled token on long windows (a 255-token utterance diverges around token
  26-130). That is the identity contract's `equivalent` class: deterministic,
  same distribution, not the same stream. The default (dynamic cache) path stays
  bit-identical.
* **Does not hold for the waveform** when `ExecutionConfig.vocoder_ragged` is
  set, which it is **by default**, on a CUDA renderer. The vocoder pads a
  chunk's mel by the receptive field instead of out to the static window, so
  cuDNN picks its algorithms for a different width and the last bits of the
  audio move: measured 1e-6 against the full-window output, which is 120 dB
  down on a waveform in [-1, 1] and is not a sound. Speech tokens are
  untouched, this is a renderer stage, so I-4 is unaffected and I-2 holds
  absolutely: the flag is deterministic, and the same build renders the same
  bytes every time.

  It is the `equivalent` class and it is declared here rather than left to be
  discovered, because it is the first flag in that class to ship **on**. Turn
  it off (`vocoder_ragged=False` in the execution config; `--no-vocoder-ragged` in `tools/bench.py`)
  when you need byte agreement with a 0.1.0 build rather than the same reading;
  `tests/test_models.py` pins both the agreement and that turning it off
  restores the padded bytes exactly. On CPU, MPS and the graph backends the
  flag does nothing at all, and `describe()` says which of those you have.
* **Warning**: the engine emits a `RuntimeWarning` when `cuda_graphs` (or
  `compile_model`) is enabled, since that path is outside the proven
  token-identical envelope. If a future change proves token identity on long
  windows, the warning can be dropped. Until then it gates the flag honestly.

## Classification every change must declare

| class | meaning | examples | test that catches a violation |
|---|---|---|---|
| **bit-exact** | same arithmetic, same order | removing a weight-norm reparameterisation, deleting a host sync | golden waveform hash, unchanged |
| **equivalent** | deterministic, different reduction order | CUDA graphs / static KV cache, the ragged vocoder, kernel fusion, Triton GEMVs, flash attention | golden tokens unchanged; logit drift within band; waveform correlation band |
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

## Execution and waveform boundaries in 0.1.1

The output edge fade is a half-cosine ramp on both edges of every rendered
window, `edge_fade_seconds` in the algorithm. It changes non-silent edges
without changing speech tokens. The default is 0.02 s (20 ms), serialized as
`"edge_fade_seconds":"0.02"`. The historical 0.005 s
stays out of the canonical form, the way `decode_mode` does; any other length
enters it and moves the fingerprint, in every implementation. See
[postprocess](../design/postprocess.md). Old manifests without the key retain
5 ms; new release manifests explicitly set 0.02.

The ramp itself is bit-pinned, not recomputed abroad. Python takes the cosine in
float32, and both lengths a release can apply, 5 ms and 20 ms at 24 kHz, are
carried in Go, Rust, TypeScript and Swift as those float32 bits, from
`tests/data/conformance/edge_fade.json`. Any other length is computed in double
and narrowed, which agrees with the reference to within two units in the last
place, under 1.2e-07 at full scale, about -138 dBFS. That is the `equivalent`
class above, not bit identity.

Precision, CPU thread count and backend placement belong to execution identity.
Changing reduction order or fp32 to fp16 can change logits and sampled tokens;
it is not covered by same-configuration bit identity. Record `engine.describe()`
and runtime versions when comparing output. Do not pin a global thread count
implicitly to make one measurement repeat. Native CoreML generation is fp32;
`generator_device="cpu"` explicitly selects torch, whose precision is separate.
CUDA graph capture remains opt-in and equivalent, not token-identical to eager.
