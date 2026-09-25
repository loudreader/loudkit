# Identity contract

The output guarantees of loudkit, and their limits. The cross-runtime results
that the test suite measures are in [Parity, measured](../parity-measured.md).

## The recipe in force: `loudkit-1`

Production fingerprint `7cd75498ad4e7531`. Python, Swift, Go, Rust and
TypeScript compute it independently and agree. `loudr-1-turbo` runs the same
recipe in the `fusion_mtp2` decode mode with one Euler step. Its fingerprint is
`e5303ba243087222`, and the same five implementations agree on it.

`recipe_version` names the parts of the algorithm that live in code: the
sampling rule, the window framing, the joins and the artifact detectors. Their
settings are hashed separately. `recipe_version` is hashed into the
fingerprint too. `loudkit-1` is the only recipe. An audible change under this
recipe adds or changes a setting, and that setting moves the fingerprint. The
recipe name stays `loudkit-1`.

The fingerprint values on this page, in the conformance fixture, in the ports'
freeze tests and in [COMPATIBILITY.md](COMPATIBILITY.md) change together when
the algorithm changes. `tests/test_docs.py` checks them against the fixture.

A record of a past render keeps the fingerprint it was made with, and is not
updated when the fingerprint moves:

- `docs/voices/roster/provenance.json` and `docs/voices/preview/catalog.json`
  name the algorithm each published voice sample was rendered with.
- `docs/measurements/` names the algorithm each run measured.
- The CI workflow names the algorithm each published bundle was built and
  verified with.

A voice sample with an older fingerprint is re-rendered, or its record stays as
it is.

## What `loudkit-1` includes

- Join continuity. `ChunkConfig.prefix_tokens` is 6: each chunk gets the last
  six speech tokens of the previous chunk as context, so the pitch contour
  continues across a join. Measured on the reference voice before 0.1.0: the
  pitch restarts about 74 Hz higher at a join without the prefix, and about
  7 Hz higher with it.
- Periods that do not end a sentence. `ChunkConfig.mid_sentence_period` is
  `"hold"`. A period followed by a word in lower case, or a period after a word
  in `ChunkConfig.abbreviations`, is not a chunk boundary. For example, `"But
  Mr. Smith went home"` stays one chunk. With `"break"`, every separator match
  is a boundary.
- Re-splitting a chunk that fills its window. `ChunkConfig.cap_resplit` is
  `"word"`. The character budget of a chunk is an estimate, and chunk texts are
  fixed before rendering starts. A chunk that reaches the token cap would stop
  mid-word and lose the rest of its text. loudkit replaces such a chunk with its
  two halves, cut at the word boundary nearest the middle. Both halves keep the
  original chunk index, so the seeds of later chunks do not change. The audio of
  later chunks in the passage can change, because their context now comes from
  the second half. `"off"` disables the re-split.
- The postprocess layer. The detectors in
  [postprocess](../design/postprocess.md) remove hallucinated tails, so the
  audio for an affected chunk is shorter than the raw generation. Chunks that
  no detector flags are not changed. On the two end-to-end conformance cases,
  the waveform is byte-identical with the layer on and off.
- The repetition penalty covers silence. The penalty applies to the silence
  token ids like any other token. Silence keeps only the `min_p` exemption.
- Retries for stalls and starved cuts. A `stall` verdict (a row that stays in
  silence) sends the row to the retry ladder. A `desperation` cut that keeps
  fewer than `desperation_min_keep_per_text_token` speech tokens per text token
  does the same, and the trim is the fallback.
- Silence in the repetition rule. The rule exempts a cycle made only of
  silence. With `repetition_silence: acoustic`, silence means the sampler's
  silence ids plus the manifest's `silence_render_ids` and `quiet_render_ids`
  (token ids measured to render as silence or as breath and decay). A pause on
  one of those ids is not cut as a loop. It stalls and retries instead.
- Resumed loops. With `repetition_resume: condemn`, a loop that the decoder
  resumed from goes to the retry ladder instead of being cut. This holds on a
  checkpoint without the two render id lists too. An applied repetition cut
  only removes a tail.
- The decode loop is a property of the weights. `decode.mode` is part of the
  fingerprint. A checkpoint with `fusion_mtp2` decodes two speech tokens per
  transformer forward ([why](../design/two-token-decode.md)). A manifest
  without a decode block decodes one token per forward, and the fingerprint
  does not include an explicit `"single"`. The mode is not a runtime option: a
  one-token loop over fusion weights produces wrong speech. The manifest
  declares the mode, and the file is `format_version 2`. An engine without the
  two-token loop refuses the file, both by version and by mode.

The sampler and postprocess docstrings record the measurements behind these
settings.

## What is guaranteed

**Per-build determinism.** The same seed, build, device, backend, execution
configuration and input produce a bit-identical waveform. The test suite
checks this on every backend it runs.

**One sampling algorithm.** Every implementation uses the same sampling
algorithm, drawn from the same counter-based random number stream. The same
seed makes the same choices from the same logits.

## What is not guaranteed

- Bit-identical output across backends or devices (CUDA, CPU, the Apple Neural
  Engine). Reduction order, fused multiply-add, different implementations of
  transcendental functions and the fixed fp16 pipeline of the Neural Engine all
  change the result. Measured before 0.1.0: the static KV cache alone adds
  1.34e-05 of drift through a padded reduction.
- Identical speech tokens across ONNX execution providers. `onnx_provider`
  changes which kernels run, and it is not part of the fingerprint. The
  providers were measured to agree (see below), but this is a measurement, not
  a guarantee. The CPU provider is the reference.
- Bit-identical output across releases. When the engine output changes, the
  reference fixtures are regenerated by script.
- Identical output for a given sentence on two different machines.

## Cross-backend token identity

Tokens match across backends only under the conditions below. Exact
free-running tokens depend on the CPU architecture and the BLAS library.

- On the CI machine (macOS on arm64), tests pin exact free-running tokens on
  the conformance cases for torch on CPU and MPS, ONNX Runtime on the CPU
  provider, and the four ports. CUDA is not in CI.
- ONNX execution providers, measured with loudr-1:
  - CUDA gave the same speech tokens as the CPU provider, byte for byte, in
    Python, Rust, Go and JS, on an RTX 3090 with 0.1.0.
  - CoreML gave the same speech tokens as the CPU provider, because the
    `coreml` provider runs only the three renderer graphs on CoreML. The token
    generator stays on CPU. Measured on 2026-08-23, before the 0.1.0
    release, on an Apple M3 Pro, macOS 26.1, onnxruntime 1.28.0, on the third
    passage of the benchmark set (voice `joe`, seed 1234): 392 tokens with
    digest `bf01efb39a3dbcda` in Python, Rust and Go, on both providers.
  - The waveform differs between those two providers. The same passage
    rendered `68f0de69...` on the CPU provider and `87988cc9...` on CoreML.
- Use the `cpu` provider when you compare output against the conformance
  fixture, which is a `cpu` measurement.
- The CoreML provider measurements use `ModelFormat=MLProgram`, and loudkit
  always sets it. The CoreML default format (NeuralNetwork) computes different
  values: on the same input, the vocoder output sums to 217.70 under
  NeuralNetwork, 211.15 on the CPU provider and 211.149 under MLProgram.
- ONNX Runtime's CPU graph fusion differs from torch eager even at fp32: the
  logits drift by about 1e-2 per step (torch fp32 against ONNX Runtime fp32,
  teacher-forced). In the measured English cases this drift did not change a
  sampled token. The three reference sentences (80, 191 and 175 tokens) match
  token for token on the ONNX CPU provider. Measured before 0.1.0, short and
  long English free-running passages also matched (82/82 and 204/204 tokens).
- Polish is not guaranteed to be token-identical across backends. Measured
  before 0.1.0 on the Polish sentence in `research/compare_backends.py`: ONNX
  diverged from torch fp32 at token 4, while torch on CPU and on MPS matched
  (156/156). `tests/test_onnx.py` checks that the free-running output is not
  empty and that the renderer agrees on fixed tokens; it allows the sampled
  token streams to differ. Polish on ONNX is in
  the `equivalent` class: the same sampling algorithm, but not the same token
  stream.
- `cuda_graphs` does not give the same tokens as eager. It runs the decode over
  a static KV cache, and the padded attention switches the cuBLAS kernel at
  large widths. Measured before 0.1.0: the logits drift by about 2e-4 per layer
  at a 750-token prefill, and a 255-token utterance diverges between token 26
  and token 130. This is the `equivalent` class: deterministic, but not the
  same token stream. The default path (dynamic cache) is not affected.
- `ExecutionConfig.vocoder_ragged` changes the waveform on a CUDA renderer. It
  is on by default there. The vocoder pads the mel of a chunk by its receptive
  field instead of to the full static window, so cuDNN picks its algorithms for
  a different width. The measured difference from the full-window output is
  about 1e-6, which is 120 dB below full scale. Speech tokens do not change,
  because this is a renderer stage. The flag is deterministic, so per-build
  determinism still holds. It is in the `equivalent` class.

  To get the padded-vocoder bytes of this build, set `vocoder_ragged=False` in
  the execution config (`--no-vocoder-ragged` in `tools/bench.py`).
  `tests/test_models.py` checks that the two paths agree and that turning the
  flag off gives the padded bytes exactly. On CPU, MPS and the graph backends
  the flag has no effect. `describe()` shows which one you have.
- The engine emits a `RuntimeWarning` when `cuda_graphs` or `compile_model` is
  on, because that path is not token-identical to eager.

## Change classes

A change to execution or numerics states its class and the test it passed.

| class | meaning | examples | test that catches a violation |
|---|---|---|---|
| **bit-exact** | same arithmetic, same order | removing a weight-norm reparameterisation, deleting a host sync | golden waveform hash, unchanged |
| **equivalent** | deterministic, different reduction order | CUDA graphs / static KV cache, the ragged vocoder, kernel fusion, Triton GEMVs, flash attention | deterministic re-render; logit drift within band; waveform correlation band |
| **changes-maths** | different numerics | fp16, TF32, int8 | full gate: KL, top-1, length/EOS watchdog, audio band; goldens regenerated |

## The sampler: LR-SAMPLER-v1

The one-sampling-algorithm guarantee needs an algorithm that all five
implementations can compute identically. The reference implementation in
`python/loudkit/sampler.py` is the specification. It is ported to
`go/sampler`, `rust/src/sampler.rs`, `swift/LoudKit/Sampler.swift` and
`js/src/sampler.ts`:

- The random numbers come from Philox-4x32-10, a counter-based generator
  addressed by `(seed, stream, step, index)`. It uses integer arithmetic only.
  It is verified bit-exact against the three published Random123 known-answer
  vectors, and every port is checked against the same vectors.
- The generator is stateless. The random number for a token depends only on
  its counter, not on how many numbers were drawn before it. CPU and GPU can
  compute them in any order and still agree.
- `min_p` is applied in logit space: keep `i` where
  `z_i/T >= max(z/T) + ln(min_p)`. This selects the same tokens as
  `p_i >= min_p * p_max`, because softmax is monotone and its normaliser
  cancels. The selection uses no exponential, no sum and no renormalisation,
  so no reduction order can vary.
- The choice is a Gumbel-argmax instead of a CDF scan, with ties broken by
  the lowest index.

Measured before 0.1.0: CPU and CUDA agree on 64 of 64 tokens. A
vendor-library sampler agrees on 17 of 64. The total variation against the
upstream sampling law is 0.00286, against a sampling-noise floor of 0.00227.

Cost, measured before 0.1.0: +0.078 ms per token, about 4% of a
token-generator step. In eager torch the sampler is several elementwise
kernels plus an argmax over 8194 ids. The cost is recorded, and no gate applies
to it.

## Measurement rules

1. Set both TF32 flags and record them. In PyTorch, `cudnn.allow_tf32` defaults
   to True and `cuda.matmul.allow_tf32` defaults to False, so the default
   "fp32" uses TF32 convolutions and is not bit-exact with true fp32. Measured
   before 0.1.0: about 5% faster and not bit-identical on the renderer. loudkit
   sets both flags from `ExecutionConfig.allow_tf32` on every path.
2. Use the same seed to compare renders. The renderer draws its prior noise and
   its excitation noise from Philox streams under that seed, not from the
   global torch RNG. Measured before 0.1.0, with unseeded noise, two renders of
   the same tokens correlate at only 0.11 to 0.31.
3. Judge quality only on renders of generated speech tokens. Random speech
   tokens render as babble.
4. Record with every reference result: the torch version, the GPU, both TF32
   flags, the dtype, the fingerprint and the identity contract version.

## Execution and waveform boundaries in 0.1.1

The output edge fade is a half-cosine ramp on both edges of every rendered
window, `edge_fade_seconds` in the algorithm. It changes non-silent edges
without changing speech tokens. The release manifests set 0.02 s (20 ms),
serialized as `"edge_fade_seconds":"0.02"`. A manifest without the key uses
5 ms. The 0.005 s value is left out of the canonical form, like an absent
`decode_mode`. Any other length is part of the canonical form and moves the
fingerprint, in every implementation. See
[postprocess](../design/postprocess.md).

The ports read the ramp's float32 bits from
`tests/data/conformance/edge_fade.json` for the two lengths a release can
apply: 5 ms and 20 ms at 24 kHz. Python computes the cosine in float32. The
ports compute any other length in double precision and round it to float32.
That result agrees with the reference to within two units in the last place,
under 1.2e-07 at full scale (about -138 dBFS). This is the `equivalent` class,
not bit identity.

Precision, CPU thread count and backend placement are part of the execution
configuration. A change in reduction order, or from fp32 to fp16, can change
logits and sampled tokens, so bit identity applies only within one execution
configuration. Record `engine.describe()` and the runtime versions when you
compare output. Native CoreML generation is fp32. `generator_device="cpu"`
selects the torch generator, which has its own precision setting. CUDA graph
capture is opt-in and in the `equivalent` class. It is not token-identical to
eager.
