"""LR-SAMPLER-v1 — one sampling law, specified tightly enough to reimplement.

The shipped engine and this library must choose the same token from the same
logits, on any hardware, in any language. That is a stronger requirement than
"same distribution", and it rules out the obvious implementation.

Three things had to change from the textbook version, each for a measured
reason.

**The RNG is counter-based, not a library generator.** ``torch.multinomial``
returns different samples for an identical probability vector and an identical
generator on x86 versus arm64. A library-RNG sampler makes every cross-host comparison
diverged at token zero.

**min_p is evaluated in logit space.** The usual form — normalise to
probabilities, drop anything below ``min_p * p_max``, renormalise, scan a CDF —
contains two reductions (a sum and a scan) whose order a backend is free to
vary. Because softmax is monotone and its normaliser cancels on both sides of
the comparison, ``p_i >= min_p * p_max`` is exactly ``z_i/T >= max(z/T) +
ln(min_p)``. Same selection, no exponential, no sum, no scan.

**Selection is Gumbel-argmax, not a CDF walk.** Adding ``-log(-log(u))`` to the
scaled logits and taking the argmax is a categorical draw, and an argmax is
order-independent apart from ties, which are broken by lowest index.

Adopting this changes which tokens come out — same law, different stream — so it
re-bases goldens once, under a bumped identity-contract version.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import SamplingConfig
from .rng import gumbel_noise

__all__ = ["LRSamplerV1", "SAMPLER_VERSION"]

SAMPLER_VERSION = "LR-SAMPLER-v1"

_STREAM_SAMPLING = 0
"""Sub-stream id for token sampling. The flow prior and the vocoder excitation
use their own, so no two consumers can ever draw the same numbers."""


class LRSamplerV1:
    """The sampling law of :data:`SAMPLER_VERSION`.

    Stateless with respect to *which* numbers it draws — a token's randomness is
    a pure function of ``(seed, step)`` — but it caches a block of precomputed
    Gumbel noise, because generating ten Philox rounds per token costs more than
    running the entire model.

    Example:
        >>> from loudkit.config import SamplingConfig
        >>> sampler = LRSamplerV1(SamplingConfig(silence_token_ids=(0, 1)), seed=7)
        >>> logits = np.zeros(16, dtype=np.float32); logits[3] = 10.0
        >>> seen = np.zeros(16, dtype=bool)
        >>> sampler(logits, step=0, seen=seen)
        3
    """

    __slots__ = (
        "_cfg",
        "_seed",
        "_block",
        "_noise",
        "_base",
        "_silence",
        "_stop_token",
        "_eos_floor",
        "_peak_at",
        "_peak_prob",
    )

    def __init__(
        self,
        config: SamplingConfig,
        *,
        seed: int,
        block: int = 256,
        stop_token: int | None = None,
        eos_floor: int = 0,
    ) -> None:
        """
        Args:
            config: the law. Read once; this object never mutates it.
            seed: the user-visible seed. Same seed, same tokens, on any backend.
            block: how many steps of noise to precompute at a time. Invisible to
                the result — statelessness means step 300 gets the same number
                whether it was drawn alone or inside a block starting at 256.
            stop_token: enables :attr:`eos_peak`. ``None`` disables the
                observation entirely, and with it its cost — one exponential and
                one sum over the vocabulary per step.

                Observation is done **here**, in the sampler, rather than by
                changing :meth:`~loudkit.contracts.TokenGenerator.generate`.
                Every backend already calls the injected sampler on every step —
                it owns the RNG stream, so a backend that skipped it would
                produce different tokens — which means this reaches torch, ONNX
                and CoreML without touching a protocol that other people have
                written implementations against.
            eos_floor: the EOS floor this generation runs under. The peak is
                only recorded past it, matching the shipped engine: below the
                floor the generator masks the stop token, so its probability
                there describes the mask rather than the model.
        """
        self._cfg = config
        self._seed = seed
        self._block = block
        self._noise: NDArray[np.float64] | None = None
        self._base = 0
        self._silence = np.asarray(config.silence_token_ids, dtype=np.int64)
        self._stop_token = stop_token
        self._eos_floor = eos_floor
        self._peak_at = -1
        self._peak_prob = 0.0

    @property
    def eos_peak(self) -> tuple[int, float]:
        """Where the model came closest to stopping, as ``(step, probability)``.

        ``(-1, 0.0)`` when the stop token was never plausible, or when this
        sampler was built without a ``stop_token``.

        **If the model never stops, that peak is where the sentence really
        ended** — which is what makes the number worth carrying. It is read by
        :mod:`loudkit.postprocess`, and because two of the rules there compare it
        against a threshold, it is an *audible* value despite never feeding back
        into sampling. The conformance fixture pins it for that reason.
        """
        return self._peak_at, self._peak_prob

    @property
    def config(self) -> SamplingConfig:
        return self._cfg

    @property
    def seed(self) -> int:
        return self._seed

    def _noise_for(self, step: int, width: int) -> NDArray[np.float64]:
        cache = self._noise
        if (
            cache is None
            or step < self._base
            or step >= self._base + self._block
            or cache.shape[1] != width
        ):
            self._base = (step // self._block) * self._block
            self._noise = gumbel_noise(
                self._seed, _STREAM_SAMPLING, self._base, self._block, width
            )
        assert self._noise is not None
        row: NDArray[np.float64] = self._noise[step - self._base]
        return row

    def __call__(
        self,
        logits: NDArray[np.float32],
        *,
        step: int,
        seen: NDArray[np.bool_],
    ) -> int:
        """Choose the next token from raw, unnormalised logits.

        Args:
            logits: ``(vocab,)`` scores straight from the model head.
            step: decode step index, which addresses the RNG.
            seen: ``(vocab,)`` mask of already-emitted tokens.

        Returns:
            The chosen token id.
        """
        cfg = self._cfg
        z = np.asarray(logits, dtype=np.float64)

        if cfg.repetition_penalty != 1.0:
            mask = seen.copy()
            if self._silence.size:
                # A reader pauses repeatedly. Penalising silence removes pauses,
                # which is audible and was measured: pause ratio 0.112 -> 0.085.
                mask[self._silence] = False
            penalised = np.where(z > 0, z / cfg.repetition_penalty, z * cfg.repetition_penalty)
            z = np.where(mask, penalised, z)

        s = z / cfg.temperature

        # min_p in logit space: identical selection to p_i >= min_p * p_max,
        # with no softmax and therefore no order-dependent normalisation.
        keep = s >= (s.max() + np.log(cfg.min_p)) if cfg.min_p > 0.0 else np.ones_like(s, bool)
        if self._silence.size:
            # Silence stays available even when min_p would drop it: a pause
            # token is what makes a reader pause, and a filter that removes the
            # only way to pause is a filter that removes prosody. This mirrors
            # the repetition-penalty exemption above — silence is not content,
            # it is punctuation of content, so neither filter applies to it.
            keep[self._silence] = True

        if self._stop_token is not None:
            self._observe_eos(s, keep, step)

        g = s + self._noise_for(step, s.shape[0])
        g = np.where(keep, g, -np.inf)
        return int(np.argmax(g))  # argmax already breaks ties toward low indices

    def _observe_eos(self, s: NDArray[np.float64], keep: NDArray[np.bool_], step: int) -> None:
        """Record how close this step came to stopping. Never changes the draw.

        The quantity is the shipped engine's, reproduced exactly: the stop
        token's softmax weight over the sum of the weights that *survived*
        ``min_p``. Two details are deliberate and neither is an oversight.

        The numerator is taken **before** the cutoff is applied, so a step where
        the stop token was itself filtered out still reports how near it came.
        The number answers "how close was this to being the end", not "what was
        the chance of stopping" — and the first question is the one
        :mod:`loudkit.postprocess` needs, because the rows it exists to rescue
        are precisely the ones where stopping never won.

        The floor is ``>`` and not ``>=``: at exactly the floor step the
        generator has only just unmasked the stop token, and the shipped engine
        records from the step after.
        """
        if step <= self._eos_floor:
            return
        assert self._stop_token is not None
        # A separate max() from the one in the min_p test above, rather than a
        # shared temporary: the sampling law is the thing five ports must agree
        # on bit for bit, and it does not get restructured to save an O(vocab)
        # pass that the exponential below dwarfs anyway.
        weights = np.exp(s - s.max())
        # Fixed-order accumulation, not `np.sum()`: numpy pairs additions by
        # SIMD width, so the result could differ in the last bits between
        # hosts, and this probability sits against hard postprocess thresholds.
        # A left-to-right loop over ascending indices is the order every port
        # reproduces trivially.
        kept = weights[keep]
        total = 0.0
        for i in range(kept.shape[0]):
            total += float(kept[i])
        if total <= 0.0:
            return
        prob = float(weights[self._stop_token]) / total
        if prob > self._peak_prob:
            self._peak_prob = prob
            self._peak_at = step

    def __repr__(self) -> str:
        c = self._cfg
        return (
            f"{SAMPLER_VERSION}(seed={self._seed}, temp={c.temperature}, "
            f"rep={c.repetition_penalty}, min_p={c.min_p}, "
            f"silence={len(c.silence_token_ids)})"
        )
