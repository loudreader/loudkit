# Sampler and noise

Maintainer notes moved out of the runtime's docstrings: the reasoning and
the measurements behind each symbol. Each heading names the module and
the symbol the note belongs to. Not user documentation.


## `loudkit/models/noise.py`


### `module`

The flow prior and the vocoder excitation are *inputs* to the pipeline that
happen to be random. Treating them as device RNG state was measured to be the
single largest source of irreproducibility in this engine's history: unseeded,
two renders of identical tokens correlate at 0.109 (EXP-010), and even seeded,
``torch.randn`` streams differ across devices, so a CPU render and a GPU
render of the same seed would disagree for no algorithmic reason.

So the noise comes from :mod:`loudkit.rng`'s Philox counter instead: a value is
a pure function of ``(seed, stream, row, column)``, identical on every backend
by construction. A CoreML or ONNX backend consumes the *same bytes* as the
torch one, which is what lets a cross-backend waveform comparison measure
arithmetic rather than RNG plumbing.

The Gaussian transform draws a **fresh uniform pair per sample** (cos-only
Box–Muller). The classic cache-the-spare-``sin`` optimisation makes every
second sample share its radius, a period-2 structure that lands exactly on
Nyquist, measured at +5.3 dB at 12 kHz in the raw stream and audible as a
whine after the vocoder's excitation path (see the shipped engine's RNG note).
Two uniforms per Gaussian is the price of not shipping that artefact again.


### `gaussian_field_torch`

The excitation the vocoder needs is ``9 x samples`` of standard normals -
2.2 million of them for a full window, and drawing that in NumPy was two
thirds of the vocoder stage (105 ms of 156 ms for a typical chunk on a
3090), on the host, before a byte of it reached the card it was destined
for. Nothing about the arithmetic wanted to be there.

**The stream is the same stream.** Philox is integer arithmetic and is
reproduced here exactly, known-answer vectors and all; the conversion to
uniforms is ``(bits + 0.5) / 2**32``, a divide by a power of two, which is
exact; and ``sqrt`` is correctly rounded by IEEE-754. Two operations are
not bound that tightly: ``log`` and ``cos``, whose last bits are a libm's
to choose. Measured, the two libms agree, the field came back bit-identical
to NumPy's across 2.2 million values on CPU and on CUDA, at every shape
tried, so this is not an *equivalent*-class trade in practice. It is not
guaranteed to stay that way by anything stronger than measurement, which is
why the agreement is a test rather than a remark.

``tests/test_core.py`` pins it on CPU, which is where CI runs; the CUDA
check was done by hand and is the one a new toolchain could break without
the suite noticing.

NumPy stays the reference: a port to a third language should match it.


## `loudkit/rng.py`


### `module`

``torch.multinomial`` does not: given an identical probability vector and an
identical generator stream it returns different samples on x86 and on arm64.
Any sampler built on a library RNG inherits that.

So the stream is defined by an algorithm instead of by a vendor:
**Philox-4x32-10**, counter-based, from Salmon et al., *Parallel Random Numbers:
As Easy as 1, 2, 3*. Three properties make it the right choice here.

*Integer-only.* Every operation is a 32-bit multiply, xor or add. There is no
floating-point accumulation whose rounding could vary, so a correct
implementation in Python, Swift or Rust produces identical bits by construction
rather than by luck.

*Counter-based.* The n-th random number is a pure function of ``(seed, stream,
step, index)``, not of how many numbers came before it. Two backends may
therefore generate in any order, one number at a time, or a whole block ahead
- and still agree. That freedom is what makes the sampler affordable: generated
per token it costs 2.2 ms, more than the entire 16-layer model; generated a
block at a time it costs under 0.01 ms, and because the numbers are addressed
rather than streamed, the block boundary is invisible.

*Verifiable.* The implementation is checked against the published known-answer
vectors from the reference library, so a port can be validated against a
standard rather than against this implementation.


## `loudkit/sampler.py`


### `module`

The shipped engine and this library must choose the same token from the same
logits, on any hardware, in any language. That is a stronger requirement than
"same distribution", and it rules out the obvious implementation.

Three things had to change from the textbook version, each for a measured
reason.

**The RNG is counter-based, not a library generator.** ``torch.multinomial``
returns different samples for an identical probability vector and an identical
generator on x86 versus arm64. A library-RNG sampler makes every cross-host comparison
diverged at token zero.

**min_p is evaluated in logit space.** The usual form, normalise to
probabilities, drop anything below ``min_p * p_max``, renormalise, scan a CDF -
contains two reductions (a sum and a scan) whose order a backend is free to
vary. Because softmax is monotone and its normaliser cancels on both sides of
the comparison, ``p_i >= min_p * p_max`` is exactly ``z_i/T >= max(z/T) +
ln(min_p)``. Same selection, no exponential, no sum, no scan.

**Selection is Gumbel-argmax, not a CDF walk.** Adding ``-log(-log(u))`` to the
scaled logits and taking the argmax is a categorical draw, and an argmax is
order-independent apart from ties, which are broken by lowest index.

Adopting this changes which tokens come out, same law, different stream, so it
re-bases goldens once, under a bumped identity-contract version.

One later amendment to the law itself: **the repetition penalty applies to
silence ids.** They were exempt from both the penalty and the ``min_p`` cutoff;
with both exemptions in place a silence run is absorbing, ``min_p`` removes
every non-silence candidate, the exemption re-admits silence, and nothing ever
degrades the state (zero escapes in 1,031 instrumented trap steps; 33.0% of
long-form paragraphs carried a >1 s hole). The penalty exemption is gone and the
``min_p`` exemption stays: removing that one instead was measured catastrophic
(median long-form gap 2.46 s -> 4.64 s). The change re-based the goldens and
moved the algorithm fingerprint.


### `DeviceSamplerV1`

The point is to stop the decode loop pulling a vocabulary of logits back to
the host for every token. With selection on the device, a whole pair -
forward, both heads, both draws, the fused slot, is one captured graph and
the only thing crossing back is two token ids.

**The randomness is not reimplemented.** Gumbel noise depends on
``(seed, stream, step)`` and nothing else, so a block of it is generated by
the same :func:`~loudkit.rng.gumbel_noise` the host sampler uses and
uploaded; the device only indexes a row. That keeps the RNG, the part with
a published Philox test vector and five ports to agree with, bit-for-bit
the shipped one, and leaves no logarithm for a device libm to round its own
way.

What *is* recomputed here is the arithmetic around it: the repetition
penalty, the temperature, the ``min_p`` cutoff in logit space, the silence
exemption, the EOS floor and the argmax. Every one of those is elementwise
or an order-independent reduction (``max``, ``argmax``), computed in float64
exactly as on the host, so the selection is the same selection.

The one quantity that is *not* bit-identical is the EOS observation. It
divides the stop token's weight by a sum over the survivors, and the host
takes that sum in a fixed left-to-right order precisely because a parallel
reduction may reassociate. On the device it is a parallel reduction. This
is used only by :mod:`loudkit.postprocess`, and only against thresholds, and
it is confined to the CUDA-graph path which the identity contract already
classes as *equivalent* rather than exact, the same clause that covers the
static KV cache moving a logit. It is not a licence to be careless: see
``TestDeviceSampler`` in ``tests/test_core.py`` for the measured agreement,
and ``TestFusionDecode.test_the_on_device_loop_is_the_same_loop`` in
``tests/test_models.py`` for the loop that consumes it.


### `__init__`

seed: the user-visible seed. Same seed, same tokens, on any backend. block: how many steps of noise to precompute at a time. Invisible to the result, statelessness means step 300 gets the same number whether it was drawn alone or inside a block starting at 256. stop_token: enables :attr:`eos_peak`. ``None`` disables the observation entirely, and with it its cost, one exponential and one sum over the vocabulary per step.

Observation is done **here**, in the sampler, rather than by
        changing :meth:`~loudkit.contracts.TokenGenerator.generate`.
        Every backend already calls the injected sampler on every step -
        it owns the RNG stream, so a backend that skipped it would
        produce different tokens, which means this reaches torch, ONNX
        and CoreML without touching a protocol that other people have
        written implementations against.
    eos_floor: the EOS floor this generation runs under. The peak is
        only recorded past it, matching the shipped engine: below the
        floor the generator masks the stop token, so its probability
        there describes the mask rather than the model.


### `_observe_eos`

The quantity is the shipped engine's, reproduced exactly: the stop
token's softmax weight over the sum of the weights that *survived*
``min_p``. Two details are deliberate and neither is an oversight.

The numerator is taken **before** the cutoff is applied, so a step where
the stop token was itself filtered out still reports how near it came.
The number answers "how close was this to being the end", not "what was
the chance of stopping", and the first question is the one
:mod:`loudkit.postprocess` needs, because the rows it exists to rescue
are precisely the ones where stopping never won.

The floor is ``>`` and not ``>=``: at exactly the floor step the
generator has only just unmasked the stop token, and the shipped engine
records from the step after.


### `select`

The step counter advances by one here, so a caller that draws twice from
one forward gets two consecutive steps of the RNG stream, which is what
makes a pair decoded together indistinguishable from the same two tokens
decoded one at a time.

``valid`` is a 0-dim device bool saying whether this draw is one the
caller will actually keep, and it exists because a fused pair draws
**twice per replay whether or not both tokens belong to the sequence**.
The host discards the second when the first was the stop token, or when
the cap closes mid-pair, and until this argument existed, that
discarded token still marked ``seen`` and still moved the EOS peak. The
peak is the audible one: it is compared against thresholds in
:mod:`loudkit.postprocess`, so a token that is not in the output could
set ``eos_peak_at`` past the end of the sequence and break the
``eos_peak_at < len(tokens)`` guards there. Measured on a scripted pair:
peak ``(1, 2.5e-08)`` after the kept token, ``(2, 1.0)`` after the
discarded one.

The draw itself still happens, a captured graph has one shape, and
the step still advances, because both are what makes the replay a fixed
sequence of ops. What is suppressed is the two pieces of *state* an
invalid draw would otherwise leave behind. ``None`` means always valid,
which is every caller that draws one token at a time.


## Notes moved from the runtime docstrings


## `loudkit/rng.py`


### `uniforms`

Open at both ends deliberately: the Gumbel transform takes two logarithms,
and a zero would produce an infinity that poisons an argmax. Adding a half
before scaling by 2^-32 keeps every value clear of both ends in a single
expression, with no branch and no clamp to get wrong.

Args:
    seed: the user-visible seed.
    stream: independent sub-stream, so that (say) sampling and the flow
        prior never draw the same numbers even at the same step.
    step0: first decode step in this block.
    n_steps: how many steps to generate.
    width: numbers per step, the vocabulary size, for sampling.


## `loudkit/sampler.py`


### `LRSamplerV1`

Stateless with respect to *which* numbers it draws, a token's randomness is
a pure function of ``(seed, step)``, but it caches a block of precomputed
Gumbel noise, because generating ten Philox rounds per token costs more than
running the entire model.

Example:
    >>> from loudkit.config import SamplingConfig
    >>> sampler = LRSamplerV1(SamplingConfig(silence_token_ids=(0, 1)), seed=7)
    >>> logits = np.zeros(16, dtype=np.float32); logits[3] = 10.0
    >>> seen = np.zeros(16, dtype=bool)
    >>> sampler(logits, step=0, seen=seen)
    3


### `eos_peak`

``(-1, 0.0)`` when the stop token was never plausible, or when this
sampler was built without a ``stop_token``.

**If the model never stops, that peak is where the sentence really
ended**, which is what makes the number worth carrying. It is read by
:mod:`loudkit.postprocess`, and because two of the rules there compare it
against a threshold, it is an *audible* value despite never feeding back
into sampling. The conformance fixture pins it for that reason.


### `__call__`

Args:
    logits: ``(vocab,)`` scores straight from the model head.
    step: decode step index, which addresses the RNG.
    seen: ``(vocab,)`` mask of already-emitted tokens.

Returns:
    The chosen token id.


### `prepare`

``stop_token`` is the id the EOS floor masks, and it is separate from
the sampler's own ``stop_token`` on purpose: that one is optional and
only enables the EOS *observation*, which `Engine` switches off with
the postprocess rules. The floor is not optional, the host loops apply
it whether or not anything is being observed, so the mask needs an id
of its own or a build with postprocess off would let the stop token win
from the second token onward and cut the utterance short.


### `advance_to`

Called from the host between replays. The block is refilled *into* the
same tensor rather than replacing it: a captured graph holds addresses,
not objects, so a fresh allocation here would be a graph reading a
buffer nobody writes any more.

Args:
    step: the step the next draw is for.
    draws: how many consecutive draws will follow before the host gets
        to call this again, two for a fused pair, which advances the
        step counter itself for the second token.


## Notes moved from attribute docstrings and comments


### `loudkit/models/noise.py`


#### `module` imports the deferred names for typing

The names behind the deferred import are still imported for typing, so the
deferred half of this file is checked like the rest of the package rather than
typed as `object` and proved nothing about. See docs/design/typing.md and the
same pattern in `loudkit.sampler`.


### `loudkit/sampler.py`


#### `__call__` penalises seen silence

The repetition penalty applies to silence like everything else. Exempt, a
silence run had no exit: zero escapes in 1,031 instrumented trap steps, 33.0%
of long-form paragraphs carrying a >1 s hole, 74 of 1705 passages rendering
chunks of no speech at all. Penalised, holes fall to 4.3% and mute chunks to 1
at 120 passages per arm across ten languages, with pause rates inside the
natural band on 8 of 9 healthy voices and WER flat or better.


#### `__call__` keeps the min_p re-admission

Silence is re-admitted below the min_p floor; this is the one exemption it
keeps, because removing it was measured catastrophic (median long-form gap
2.46 s to 4.64 s). With the penalty above applying to silence, a pause that
overstays decays instead of never ending.


#### `select` scatters instead of branching

Scattering a value that already lives on the device keeps the whole update
inside the captured graph. An invalid draw writes back whatever was already
there, a no-op with a fixed shape: a branch would not be capturable, and
skipping the scatter would change the op sequence between replays.
