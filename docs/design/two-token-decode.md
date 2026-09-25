# Two speech tokens per forward

The token generator is autoregressive. One second of audio takes 25 sequential
transformer forwards, and each forward needs the token that the previous one
produced, so they cannot run in parallel. CUDA graphs, static KV caches and
precision settings reduce the cost of one forward. `fusion_mtp2` reduces the
number of forwards.

`fusion_mtp2` decodes a pair of tokens per forward. It needs weights trained
for it: a second head (`head2`) and a fusion MLP (`fuse`). A checkpoint that
carries them declares `decode.mode: "fusion_mtp2"` and `format_version 2`.

## The loop

From one hidden state `h` at the slot for pair *n*:

    a  ~  speech_head(h)                    the pair's first token
    b  ~  head2([h ; speech_emb(a)])        its second, conditioned on the first
    slot = 0.5*(ea + eb) + fuse([ea ; eb])  what the pair leaves in the cache
                                            (ea, eb = speech_emb of a and b)

`slot` then gets the pair's learned position and is the input of the next
forward. Three invariants follow from this. A port can break each of them
without an error:

- Positions advance once per pair. One slot is one position. A loop that
  advances positions per token puts the model at twice the trained distance
  by the end of a sentence.
- Either head can stop. The sequence ends on whichever head emits the stop
  token. When the first token of a pair is the stop token, the second token is
  not part of the output, and an odd token count is a normal end. The captured
  device graph still draws the second token and discards it; see `select` in
  [sampler and noise](sampler-and-noise.md#select).
- The sampler step counts tokens. The RNG is addressed by step index, so
  the step is the token's index, not the forward's. With any other index, a
  sequence decoded in pairs draws different numbers than the same sequence
  decoded one token at a time.

The slot contains the mean of the two embeddings because the fusion MLP was
initialised as a zero delta from it: the slot starts as the average, and
training learns how far to move it. A single-token loop carries only `eb`.
That input blurs phonemes and raises no error.

## Why a new format version

A version-1 reader can load a turbo checkpoint. It finds `t3.*` weights it
knows, runs the single-token loop, and produces fluent speech that does not
match the text, without an error. The project treats that failure as worse
than a crash. These checks prevent it:

- The mode is declared in the manifest as `decode.mode` and enters the
  fingerprint. An absent `decode` block and an explicit `"single"` both leave
  the field out of the canonical form, so every single-token fingerprint
  stays the same.
- The fusion submodules are built only in fusion mode. Fusion weights
  without the mode are unexpected keys, and the mode without the weights is
  missing keys. Either disagreement is a load error.
- `format_version 2` is checked against the mode.
  `checkpoint.DECODE_FORMAT_VERSION` requires version 2 for `fusion_mtp2`.
  Python, Go, Rust, Swift and TypeScript each parse `decode.mode`, refuse
  `fusion_mtp2` under `format_version 1`, and refuse a mode they do not
  implement, whatever version the file declares.
- `AlgorithmConfig.describe()` names the mode, so a log line or a pasted bug
  report shows which loop ran.

Python, Go, Rust, Swift and TypeScript implement the pair loop. The ONNX and
CoreML generator exports add `t3_pair_step` and `t3_head2` for fusion, and the
manifest selects them. Swift runs its native generator. Both models use the
same enrollment assets and voice profiles.

`recipe_version` stays `loudkit-1`, a strict equality gate. The format version
and the fingerprinted `decode` block already identify the file, so a second
version gate adds no information.

## Where the time goes

On CUDA with CUDA graphs and the built-in sampler,
`_generate_static_fused_ondevice` runs the whole pair in one captured graph:
the forward, both heads, both draws and the slot for the next replay. Per
pair, two token ids cross to the host, which is the one synchronisation. The
opening pair is drawn on the host from the prefill's hidden state. Without a
device sampler (no CUDA, or an injected sampler without `on_device`),
`_generate_static_fused` captures only the forward. The logits then come back
to the host, and `head2` runs outside the graph, because its input is the
pair's first token, which the host samples from those logits.

Measured on 0.1.1 (2026-09-06), voice `joe`, seed 7, the third benchmark
passage (48 words), from [Benchmarks](../benchmarks.md):

| hardware | path | loudr-1 RTF | loudr-1-turbo RTF |
|---|---|---:|---:|
| RTX 3090 | eager | 2.36x | 4.99x |
| RTX 3090 | CUDA graphs | 8.55x | 13.05x |
| Jetson Orin Nano Super, 25 W | eager | 0.66x | 1.33x |
| Jetson Orin Nano Super, 25 W | CUDA graphs | 1.85x | 2.50x |

On the RTX 3090, turbo is 2.1x faster eager and 1.5x faster under CUDA graphs.
Pair decoding halves only the forwards. With the forward captured, the
per-token sampler work and the per-pair synchronisation are a larger share of
what remains. The two models also differ in the renderer: the turbo manifest
sets one flow step (`n_cfm_timesteps: 1`) and loudr-1 sets two. The
end-to-end ratio therefore includes both changes.

A development measurement, recorded on 2026-08-30 before the release, compared
the 0.1.0 single-token checkpoint with the pair decode on one checkpoint
lineage: CUDA graphs, `python tools/bench.py --seed 7`, RTF on the short /
medium / long benchmark samples. It matches neither shipped model in
[Benchmarks](../benchmarks.md). The same table is in the 0.1.1 entry of
[CHANGELOG.md](../../CHANGELOG.md).

| | 0.1.0, single token | development build: pair decode, one-step renderer |
|---|---|---|
| RTX 3090, RTF (short / medium / long) | 1.22 / 6.03 / 7.73 | 1.43 / 7.47 / 8.45 |
| Jetson Orin Nano, RTF | 0.53 / 1.60 / 1.84 | 0.63 / 1.82 / 2.16 |

The end-to-end ratios in this table are 1.09x to 1.24x. The notes of the same
date also record a generator-stage gain of 1.18x to 1.40x, and a further 1.2x
to 1.6x on the mel stage from the one-step renderer; the stage timings behind
those two figures are not in the repository. Neither change affects the
vocoder. On the Jetson it took 3.4 s of the 9.2 s that a sixteen-second
passage cost, the largest single item.

## Tests that hold the loop

`tests/test_models.py::TestFusionDecode` compares three implementations of
this decode on a two-layer model with random weights:

- the eager loop;
- the static loop that samples on the host;
- the static loop that samples on the device inside the graph.

The third loop runs only when `_device_sampler` finds CUDA. The test replaces
that one gate, because `DeviceSamplerV1` also runs on CPU torch. Its four
cases are floor 16 with cap 24, floor 0 (the stop token can win at step one),
an odd cap of 9 (a pair stops half-way), and cap 40 (several pairs). The loop
is also checked against `teacher_forced_logits`, which reads the whole
sequence in one causal forward and shares no code with the three loops.

`TestTheFusionLoopAgainstRealWeights` checks the shipping entry point against
real weights. It pins one token stream for a fixed text, voice and seed,
through the checkpoint file and through an assembled release directory, so it
checks the resolver's answer and the decode together. A mis-paired slot
changes the token stream first.

Release CI requires both checkpoints (`LOUDKIT_REQUIRE_ASSETS`).

Related: [the identity contract](../reference/IDENTITY-CONTRACT.md) for what
"equivalent" permits a static KV cache to change, and
[silence classes](silence-classes.md) for how the pauses are counted.

## Release integration

Each synthesis checkpoint ships without its enrollment weights, which are a
separate `loudr-1-enrollment.safetensors` file. Export each model's graphs from
its own synthesis file. The split changes file digests, not tensors. Both
models share the voice profiles and the enrollment graphs. Both model
repositories are public before the code is tagged. CI requires both
checkpoints and tests both decode modes. The full procedure is in
[RELEASING.md](../../RELEASING.md).
