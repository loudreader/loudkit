# Sampler and noise

Maintainer notes for `loudkit/models/noise.py`, `loudkit/rng.py` and
`loudkit/sampler.py`: the reasoning and the measurements behind each symbol.
Each heading names the module and the symbol whose docstring points here.


## `loudkit/models/noise.py`


### `module`

The flow prior and the vocoder excitation are pipeline inputs that happen to
be random. Device RNG state makes renders irreproducible. Unseeded, two renders
of identical tokens correlate at 0.109. Seeded, `torch.randn` streams still
differ across devices, so a CPU render and a GPU render of the same seed
disagree for no algorithmic reason.

The noise therefore comes from the Philox counter in `loudkit.rng`. Each
uniform is a pure function of `(seed, stream, row, column)`, and its bits are
the same on every backend. The ONNX and CoreML backends consume the NumPy field
(`gaussian_field`). The torch backend uses it too on CPU and MPS, and computes
the same field on the card on CUDA (`gaussian_field_torch`, below). A
cross-backend waveform comparison then measures only the arithmetic.

The Gaussian transform draws a fresh uniform pair per sample and uses only
the cosine output of Box–Muller: `sqrt(-2 ln u1) * cos(2 pi u2)`, with `u1`
and `u2` from sub-streams `stream` and `stream + 1`. Callers therefore space
their stream ids by two. An earlier implementation emitted the sine and cosine
outputs of one pair as consecutive samples. Its raw stream measured +5.3 dB at
12 kHz (Nyquist at 24 kHz), audible as a whine after the vocoder's excitation
path. The sine and cosine outputs of a correct Box–Muller pair are independent
standard normals, so their shared radius does not explain that measurement;
its cause is not recorded. The cosine-only form costs two uniforms per
Gaussian, and it is the form the ports and the fixtures reproduce.


### `gaussian_field_torch`

The vocoder excitation is `9 x samples` standard normals, 2.2 million for a
full window. Drawn in NumPy on the host, it took 105 ms of a 156 ms vocoder
stage for a typical chunk on an RTX 3090, before the array was copied to the
card. On CUDA, `gaussian_field_torch` computes the field on the card that
uses it. On CPU the NumPy path stays, as the reference. MPS has no float64
datapath for the transform, so it also takes the NumPy path.

It computes the same numbers as NumPy:

- Philox is integer arithmetic and is reproduced exactly, including the
  known-answer vectors.
- The conversion to uniforms, `(bits + 0.5) / 2**32`, divides by a power of
  two and is exact.
- IEEE-754 rounds `sqrt` correctly.

`log` and `cos` are not bound that tightly: their last bits depend on the math
library. Measured, the field came back bit-identical to NumPy across 2.2
million values on CPU and on CUDA, at every shape tried. In practice this is
not an `equivalent`-class difference, but only measurement supports that.

`tests/test_core.py` (`test_the_field_matches_numpy`) checks the agreement on
CPU, where CI runs. The CUDA check was done by hand. A new CUDA toolchain can
break it without the suite noticing.

NumPy is the reference. A port in another language matches the NumPy field.


## `loudkit/rng.py`


### `module`

`torch.multinomial` depends on the platform: given an identical probability
vector and an identical generator stream, it returns different samples on x86
and on arm64. A sampler built on a library RNG inherits that.

loudkit defines the stream by an algorithm: Philox-4x32-10, a
counter-based generator from Salmon et al., *Parallel Random Numbers: As Easy
as 1, 2, 3*. Three of its properties matter here:

- *Integer-only.* Every operation is a 32-bit multiply, xor or add. No
  floating-point accumulation can round differently, so a correct
  implementation in any language produces identical bits.
- *Counter-based.* The n-th number is a pure function of `(seed, stream, step,
  index)`, not of how many numbers came before it. Backends may generate in
  any order, one number at a time or a whole block ahead, and still agree.
  Generated per token, the noise costs 2.2 ms, more than a whole forward of
  the token generator. Generated a block at a time, it costs under 0.01 ms per
  token, and the block boundary changes no value.
- *Verifiable.* The implementation is checked against the published
  known-answer vectors of the reference library (`KAT_VECTORS`, `selftest`),
  so a port validates against a standard and not against this implementation.


### `uniforms`

Returns values in the open interval (0, 1). The Gumbel transform takes two
logarithms, and a zero would produce an infinity in the argmax. Adding a half
before scaling by 2^-32 keeps every value away from both ends in one
expression, with no branch and no clamp.

| parameter | meaning |
|---|---|
| `seed` | the user-visible seed, or a per-stage seed derived from it |
| `stream` | sub-stream id; two ids under one seed never draw the same numbers |
| `step0` | first decode step in this block |
| `n_steps` | how many steps to generate |
| `width` | numbers per step: the vocabulary size, for sampling |

Stages are separated by seed, not by stream id. The first sampling attempt of a
window uses the window seed itself. A retry uses
`_derive(seed, _STREAM_RETRY + attempt)`. The flow prior uses
`_derive(seed, _STREAM_FLOW)` and the vocoder `_derive(seed, _STREAM_VOCODER)`,
both from the window seed (`window.py`). Each stage starts at stream 0. Inside
one stage the stream id separates draws: the
vocoder's phase offsets use stream 0 and its excitation noise uses streams 1
and 2.


## `loudkit/sampler.py`


### `module`

The sampler chooses the same token from the same logits, sampling
configuration, history and seed, on any hardware and in any language. That is
a stronger requirement than "the same distribution". LR-SAMPLER-v1 differs from
the textbook sampler in three places:

- The RNG is counter-based. `torch.multinomial` returns different samples
  on x86 and on arm64 for identical inputs (see `loudkit/rng.py` above). With a
  library RNG, every cross-host comparison diverges at token zero.
- `min_p` is evaluated in logit space. The usual form normalises to
  probabilities, drops anything below `min_p * p_max`, renormalises and scans
  a CDF. That form has two reductions, a sum and a scan, whose order a backend
  may vary. Softmax is monotone and its normaliser cancels on both sides of the
  comparison, so `p_i >= min_p * p_max` is exactly
  `z_i/T >= max(z/T) + ln(min_p)`. The cutoff then needs only a maximum and a
  comparison.
- Selection is Gumbel-argmax. Adding `-log(-log(u))` to the scaled logits
  and taking the argmax is a categorical draw. An argmax does not depend on
  evaluation order, except for ties, which go to the lowest index.

The draw method determines which tokens come out, even under the same law. A
change to it therefore needs a new sampler version and new goldens.

Two rules apply to the silence ids (`silence_token_ids`):

- The repetition penalty applies to every seen token, silence ids included.
  `seen` is a boolean mask, so a seen token is penalised once. The penalty does
  not grow with the length of a run.
- Silence ids are exempt from the `min_p` cutoff: they stay selectable when
  `min_p` would drop them.

Evidence, from the interior-stall study (August 2026, details in
[silence classes](silence-classes.md)):

- With silence exempt from both the penalty and the cutoff, a silence run was
  absorbing. With silence as `p_max`, `min_p` usually removed every
  non-silence candidate (on 87% of stall steps in an instrumented trace), the
  exemption re-admitted silence, and no penalty acted on it.
  Measured: zero escapes in 1,031 instrumented trap steps; 33.0% of long-form
  paragraphs carried a hole over 1 s; 74 chunks in 1705 passages rendered no
  speech at all.
- With the penalty on silence, holes fall to 4.3% and mute chunks to 1, at 120
  passages per arm across ten languages. Pause rates stay inside the natural
  band on 8 of 9 healthy voices, and WER is flat or better.
- Removing the `min_p` exemption instead raised the median long-form gap from
  2.46 s to 4.64 s.

Long silence runs are still possible under the penalty. The `stall` detector
in [postprocess](postprocess.md) catches them.


### `LRSamplerV1`

The numbers it draws are stateless: a token's randomness is a pure function of
`(seed, step)`. It caches a block of precomputed Gumbel noise, because ten
Philox rounds per token cost more than a forward of the model (see
`loudkit/rng.py` above).

```python
>>> import numpy as np
>>> from loudkit.config import SamplingConfig
>>> from loudkit.sampler import LRSamplerV1
>>> sampler = LRSamplerV1(SamplingConfig(silence_token_ids=(0, 1)), seed=7)
>>> logits = np.zeros(16, dtype=np.float32); logits[3] = 10.0
>>> seen = np.zeros(16, dtype=bool)
>>> sampler(logits, step=0, seen=seen)
3
```


### `__init__`

| parameter | meaning |
|---|---|
| `config` | the sampling law; read once and never mutated |
| `seed` | the user-visible seed. The same seed with the same logits, history and configuration gives the same tokens on any backend. |
| `block` | how many steps of noise to precompute at a time. It does not change the result: step 300 gets the same number whether it is drawn alone or inside a block that starts at 256. |
| `stop_token` | enables `eos_peak`. `None` disables the observation and its cost: one exponential and one sum over the vocabulary per step. |
| `eos_floor` | the EOS floor this generation runs under. The peak is recorded only after it: below the floor the generator masks the stop token, so its probability there describes the mask, not the model. |

The EOS observation happens in the sampler, not in `TokenGenerator.generate`.
Every backend calls the injected sampler on every step, because the sampler
owns the RNG stream and a backend that skipped it would produce different
tokens. The observation therefore reaches torch, ONNX and CoreML without a
change to the `TokenGenerator` protocol that external implementations use.


### `_observe_eos`

The quantity is the stop token's softmax weight divided by the sum of the
weights that survive `min_p`. Two details matter:

- The numerator is taken before the cutoff. A step where the stop token was
  itself filtered out still reports how near it came. The value measures how
  close the step came to ending. It is not the probability of sampling the
  stop token after filtering. Postprocess needs the first quantity, because
  its rescue rules act on rows where the stop token was never sampled.
- The floor test is `step > eos_floor`, not `>=`. At the floor step the
  generator unmasks the stop token at that step, so recording starts at the
  next step.


### `select`

The step counter advances by one per call. A caller that draws twice from one
forward gets two consecutive RNG steps. A pair decoded together therefore uses
the same RNG rows as the same two tokens decoded one at a time. The logits
differ between the two modes; the addressing does not.

`valid` is a 0-dim device bool that says whether the caller keeps this draw. A
fused pair draws twice per replay, whether or not both tokens belong to the
sequence. The host discards the second token when the first was the stop token
or when the cap closes mid-pair. For an invalid draw, `select` still draws and
still advances the step, because a captured graph replays a fixed sequence of
ops. It does not mark `seen` and does not move the EOS peak. Otherwise a token
that is not in the output could set `eos_peak_at` past the end of the sequence
and break the `eos_peak_at < len(tokens)` guards in postprocess. Measured on a
scripted pair without this gate: the peak was `(1, 2.5e-08)` after the kept
token and `(2, 1.0)` after the discarded one. `None` means valid, which is
every caller that draws one token at a time.

The `seen` update is a scatter, not a branch. Scattering a value that is
already on the device keeps the update inside the captured graph. An invalid
draw writes back the value already in `seen`, so the op sequence stays fixed
and the state stays unchanged. A data-dependent branch cannot be captured, and
skipping the scatter would change the op sequence between replays.


### `DeviceSamplerV1`

`DeviceSamplerV1` selects tokens on the device, so the decode loop does not
copy a vocabulary of logits to the host for every token. A whole pair (the
forward, both heads, both draws and the fused slot) is one captured graph, and
only two token ids cross back to the host.

It does not reimplement the randomness. Gumbel noise depends only on
`(seed, stream, step)`, so the device uploads blocks that the host sampler
generated with `loudkit.rng.gumbel_noise` (`LRSamplerV1.noise_block`) and
indexes a row. The RNG stays the implementation that the Philox test vectors
and the ports check, and no logarithm runs on the device.

It recomputes the arithmetic around the noise: the EOS floor mask, the
repetition penalty, the temperature, the `min_p` cutoff in logit space, the
silence exemption and the selection. Each step is elementwise or an
order-independent reduction (`max`, and the lowest index among the maxima),
computed in float64 as on the host. `TestDeviceSampler` in
`tests/test_core.py` checks that the device form selects the same tokens as
the host form for identical inputs.

The EOS observation is the one value that is not bit-identical. It divides the
stop token's weight by a sum over the kept tokens. The host takes that sum in a
fixed left-to-right order, because a parallel reduction may reassociate. The
device takes it as a parallel reduction. Only `loudkit.postprocess` reads the
value, against thresholds, and only the CUDA-graph path computes it on the
device. The identity contract classes that path as `equivalent`, the same
clause that covers the static KV cache moving a logit. A last-bit difference
can still flip a postprocess verdict: see "The EOS peak on the device" in
[postprocess](postprocess.md). `TestDeviceSampler` and
`TestFusionDecode.test_the_on_device_loop_is_the_same_loop` in
`tests/test_models.py` measure the agreement on CPU: the peak step exactly,
the probability to a relative tolerance of 1e-9.


### `eos_peak`

Returns `(step, probability)` for the largest recorded EOS observation. It
returns `(-1, 0.0)` when no positive observation was recorded after the floor,
or when the sampler was built without a `stop_token`.

When the model never samples the stop token, the peak marks the step where it
came closest to stopping. Postprocess uses it as a heuristic anchor for a cut.
Three postprocess rules read it: `silence_tail` and `terminal_echo` compare the
probability against thresholds, and the `desperation` fallback compares the
step against a position band. The value is therefore audible, although it
never feeds back into sampling, and the conformance fixture pins it.


### `__call__`

| parameter | meaning |
|---|---|
| `logits` | `(vocab,)` scores straight from the model head |
| `step` | decode step index; addresses the RNG row |
| `seen` | `(vocab,)` mask of already-emitted tokens |

Returns the chosen token id.


### `prepare`

`DeviceSamplerV1.prepare(stop_token=...)` sets the id that the EOS floor
masks. It is separate from the host sampler's own `stop_token`, which is
optional and only enables the EOS observation. `Engine` turns the observation
off when postprocess is `off`. The floor always applies: the host loops apply
it whether or not anything is observed. Without an id of its own, a build with
postprocess off would let the stop token win from the second token onward and
cut the utterance short.


### `advance_to`

The host calls it between replays. It refills the noise block into the
same tensor, because a captured graph holds tensor addresses: a new allocation
would leave the graph reading a buffer that nothing updates.

| parameter | meaning |
|---|---|
| `step` | the step of the next draw |
| `draws` | how many consecutive draws follow before the host calls again: two for a fused pair, where the graph advances the step for the second token. Both draws must fall inside the resident block, or it raises. |


## Typing

`loudkit/models/noise.py` imports torch only inside its device functions, so
the ONNX and CoreML backends import it without torch installed. The torch
names are still imported under `TYPE_CHECKING`, so mypy checks the device half
like the rest of the package. See [typing](typing.md); `loudkit.sampler` uses
the same pattern.
