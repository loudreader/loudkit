# Two speech tokens per forward

The token generator is autoregressive, so a second of audio costs 25 sequential
transformer forwards and there is nothing to batch: each one needs the token
the last one produced. Every other lever in this engine — CUDA graphs, static
KV caches, precision — attacks the cost of a forward. This one attacks the
count.

`fusion_mtp2` decodes a **pair** of tokens per forward. It is the reason a
checkpoint carries `format_version 2`, and it is a property of the weights, not
a switch.

## The loop

From one hidden state `h` at the slot for pair *n*:

    a  ~  speech_head(h)                    the pair's first token
    b  ~  head2([h ; speech_emb(a)])        its second, conditioned on the first
    slot = 0.5*(ea + eb) + fuse([ea ; eb])  what the pair leaves in the cache
                                            (ea, eb = speech_emb of a and b)

`slot` then gets the pair's learned position and becomes the next forward's
input. Three consequences fall out of that last line, and all three are places
a port can be subtly wrong:

- **The position clock ticks per pair.** One slot is one position. A loop that
  advanced positions per token would put the model at twice the distance it was
  trained to expect by the end of a sentence.
- **Either head can stop.** The sequence ends on whichever emits the stop token,
  so a pair's first token ending it means its second is never sampled — and an
  odd number of tokens is a normal way for a generation to end.
- **The sampler still counts tokens.** Its RNG is addressed by step index, so
  step must be the token's index, not the forward's, or a sequence decoded in
  pairs draws different numbers than the same sequence decoded singly.

The mean of the two embeddings is in the slot because the fusion MLP was
initialised as a zero-delta from it: the slot starts as the average and the MLP
learns how far to move it. Carrying only `eb` — which is what a single-token
loop would naturally produce — is the specific failure this design exists to
avoid, and it does not crash. It blurs phonemes.

## Why a new format version

A version-1 reader handed a turbo checkpoint finds `t3.*` weights it knows, runs
the loop it knows, and emits fluent nonsense. No error, no silence, no clipping
— a plausible voice reading something that is not the text. That is the failure
mode this project treats as worse than a crash, so:

- the mode is declared in the manifest as `decode.mode` and **enters the
  fingerprint**, while an absent block stays absent rather than defaulting to
  an explicit `"single"` — otherwise every loudr-1 fingerprint would move for
  checkpoints that decode exactly as they always did;
- the fusion submodules are built **only** in fusion mode, so weights without
  the mode are unexpected keys and the mode without weights is missing ones:
  a disagreement is a load error in either direction, never half a model;
- the file is `format_version 2`, and that is **checked against the mode**
  rather than trusted. The version is the portable half of this contract: Rust,
  Go, TypeScript and Swift gate on the number and none of them parses the
  `decode` block, so a manifest saying `format_version 1` beside
  `decode.mode: "fusion_mtp2"` would clear every gate and be misread by four of
  the five engines. `checkpoint.DECODE_FORMAT_VERSION` refuses that file, and
  each port refuses a `decode.mode` it does not implement whatever version the
  file claims to be. Refusing is the correct answer from an engine that has not
  implemented the loop, and now neither field can be believed alone;
- Python, Go, Rust, Swift and TypeScript implement the pair loop. ONNX and
  CoreML generator exports use `t3_pair_step` and `t3_head2` for fusion,
  selected by the manifest. Swift keeps its native generator. Both models
  use the same enrollment assets and voice profiles;
- `AlgorithmConfig.describe()` names the mode, so the line a log carries and a
  bug report pastes distinguishes two engines that would otherwise differ only
  by a hash.

The recipe version does **not** move. `loudkit-1` stays a strict equality gate;
the format version and the fingerprinted `decode` block already distinguish the
file, and a second loosened gate would buy nothing.

## What it is worth

Measured with CUDA graphs, one checkpoint lineage, `python tools/bench.py --seed 7`:

| | 0.1.0 | 0.1.1 |
|---|---|---|
| RTX 3090, RTF (short / medium / long) | 1.22 / 6.03 / 7.73 | 1.43 / 7.47 / 8.45 |
| Jetson Orin Nano, RTF | 0.53 / 1.60 / 1.84 | 0.63 / 1.82 / 2.16 |

The generator gains 1.18–1.40x, not 2x. Halving the forwards halves only the
forwards: with the forward already captured in a graph, what remains per token
is the sampler in NumPy, and per pair a device-to-host copy of the logits and
the second head's matmul, which sits outside the graph because its input is the
token just sampled. Those are now a measurable share of decode time, and they
are where the next factor is, if there is one.

The 0.1.1 checkpoint also folds in a renderer distilled from six Euler steps to
one, worth a further 1.2–1.6x on the mel stage. Neither change touches the
vocoder, which is untouched by both and is now the largest single item on the
Jetson: 3.4s of the 9.2s a sixteen-second passage costs.

## What holds the loop, and what does not

Three implementations of this decode exist and all three are compared against
each other in `tests/test_models.py::TestFusionDecode`, on a two-layer model
with random weights: the eager loop, the static loop that samples on the host,
and the static loop that samples on the device inside the graph. The third is
reached only when `_device_sampler` finds CUDA, so the test stubs that one gate
— `DeviceSamplerV1` runs on CPU torch — and the four cases it runs are the
boundaries rather than the middle: a floor that unmasks mid-pair, a floor of
zero so the stop token can win at step one, an odd cap that must stop a pair
half-way, and a run long enough to cross several pairs. The loop is also
checked against `teacher_forced_logits`, which reads the whole sequence in one
causal forward and shares no code with any of them.

Against real weights, `TestTheFusionLoopAgainstRealWeights` pins one token
stream from the shipping entry point: a fixed text, voice and seed, once
through the checkpoint file and once through an assembled release directory,
so the resolver's answer is pinned beside the decode's. That is the layer a
mis-paired slot changes first.

Both checkpoints are required by `LOUDKIT_REQUIRE_ASSETS` in release CI.

Related: [the identity contract](../reference/IDENTITY-CONTRACT.md) for what
"equivalent" permits a static KV cache to change, and
[silence classes](silence-classes.md) for how the pauses are counted.

## Release integration

Both synthesis checkpoints are split from their enrollment weights. Export each
model's graphs against its own final synthesis file; file digests intentionally
change during splitting even though every tensor is unchanged. Shared voice
profiles and enrollment graph math are unchanged. Both model repositories must
be public before tagging code; CI requires both checkpoints and tests both modes.
