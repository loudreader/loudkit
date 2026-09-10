"""LR-SAMPLER-v1, one sampling law, specified tightly enough to reimplement.

See ``docs/design/sampler-and-noise.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

# torch stays a runtime import inside the device sampler: this module is the
# portable law, and the ONNX and CoreML backends load it without torch
# installed at all.
if TYPE_CHECKING:
    import torch
    from torch import Tensor

from .config import SamplingConfig
from .rng import gumbel_noise

__all__ = ["LRSamplerV1", "DeviceSamplerV1", "SAMPLER_VERSION"]

SAMPLER_VERSION = "LR-SAMPLER-v1"

_STREAM_SAMPLING = 0
"""Sub-stream id for token sampling, under this stage's own seed.

The flow prior and the vocoder phase carry this same id, 0. What keeps the
three from drawing the same numbers is the per-stage seed ``window._derive``
mixes, not the id: a reader who takes the id for the separation renumbers one
of them and moves every sample the model draws.
"""


class LRSamplerV1:
    """The sampling law of :data:`SAMPLER_VERSION`.

    See ``docs/design/sampler-and-noise.md``.
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
        """Args: config: the law. Read once; this object never mutates it.

        See ``docs/design/sampler-and-noise.md``.
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

        See ``docs/design/sampler-and-noise.md``.
        """
        return self._peak_at, self._peak_prob

    @property
    def config(self) -> SamplingConfig:
        return self._cfg

    @property
    def seed(self) -> int:
        return self._seed

    def noise_block(self, base: int, width: int) -> NDArray[np.float64]:
        """The Gumbel block starting at ``base``, generated once and kept.

        Public because the device sampler needs the *same* block rather than
        its own: generating one is a Philox pass over ``block x width`` values,
        which for a production vocabulary is comparable to the whole decode it
        is meant to serve, and a second copy of a number that is a pure
        function of ``(seed, stream, step)`` buys nothing.
        """
        cache = self._noise
        if cache is None or self._base != base or cache.shape[1] != width:
            self._base = base
            self._noise = gumbel_noise(self._seed, _STREAM_SAMPLING, base, self._block, width)
        assert self._noise is not None
        return self._noise

    def _noise_for(self, step: int, width: int) -> NDArray[np.float64]:
        base = (step // self._block) * self._block
        row: NDArray[np.float64] = self.noise_block(base, width)[step - base]
        return row

    def __call__(
        self,
        logits: NDArray[np.float32],
        *,
        step: int,
        seen: NDArray[np.bool_],
    ) -> int:
        """Choose the next token from raw, unnormalised logits.

        See ``docs/design/sampler-and-noise.md``.
        """
        cfg = self._cfg
        z = np.asarray(logits, dtype=np.float64)

        if cfg.repetition_penalty != 1.0:
            # The penalty applies to every seen token, silence included.
            penalised = np.where(z > 0, z / cfg.repetition_penalty, z * cfg.repetition_penalty)
            z = np.where(seen, penalised, z)

        s = z / cfg.temperature

        # min_p in logit space: identical selection to p_i >= min_p * p_max,
        # with no softmax and therefore no order-dependent normalisation.
        keep = s >= (s.max() + np.log(cfg.min_p)) if cfg.min_p > 0.0 else np.ones_like(s, bool)
        if self._silence.size:
            # Silence stays available even when min_p would drop it: a pause token is
            # what makes a reader pause, and a filter that removes the only way to pause.
            keep[self._silence] = True

        if self._stop_token is not None:
            self._observe_eos(s, keep, step)

        g = s + self._noise_for(step, s.shape[0])
        g = np.where(keep, g, -np.inf)
        return int(np.argmax(g))  # argmax already breaks ties toward low indices

    def _observe_eos(self, s: NDArray[np.float64], keep: NDArray[np.bool_], step: int) -> None:
        """Record how close this step came to stopping. Never changes the draw.

        See ``docs/design/sampler-and-noise.md``.
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

    def absorb_peak(self, step: int, prob: float) -> None:
        """Fold in an EOS observation made somewhere else.

        The graph-captured decode evaluates the same quantity on the device, so
        that a step costs no copy back; at the end of a generation it hands the
        running peak here, and :attr:`eos_peak` answers for the whole run
        whichever path produced it.
        """
        if prob > self._peak_prob:
            self._peak_prob = prob
            self._peak_at = step

    def on_device(self, device: str, *, block_steps: int | None = None) -> DeviceSamplerV1:
        """This same law, as buffers and tensor ops a CUDA graph can capture.

        Returns a companion, not a replacement: it is built *from* this
        sampler's configuration, seed and stream, so the two draw the same
        numbers for the same step. See :class:`DeviceSamplerV1` for what is
        identical and what is not.
        """
        import torch

        # `is None`, not `or`: 0 is a value a caller can ask for, and the
        # companion refuses a block that does not match this sampler's. Under
        # `or` that refusal was skipped and the host's block used instead.
        return DeviceSamplerV1(
            self,
            torch.device(device),
            block_steps=self._block if block_steps is None else block_steps,
        )

    def __repr__(self) -> str:
        c = self._cfg
        return (
            f"{SAMPLER_VERSION}(seed={self._seed}, temp={c.temperature}, "
            f"rep={c.repetition_penalty}, min_p={c.min_p}, "
            f"silence={len(c.silence_token_ids)})"
        )


class DeviceSamplerV1:
    """LR-SAMPLER-v1 as capturable tensor ops, for the graph decode path.

    See ``docs/design/sampler-and-noise.md``.
    """

    def __init__(
        self,
        host: LRSamplerV1,
        device: torch.device,
        *,
        block_steps: int,
    ) -> None:
        import torch

        cfg = host.config
        self._host = host
        self._cfg = cfg
        self._device = device
        if block_steps != host._block:
            # The device rows are the host's array uploaded verbatim, so a
            # different block length here would be a shape mismatch at best and
            # a silently shifted RNG stream at worst.
            raise ValueError(
                f"device block of {block_steps} steps must match the host "
                f"sampler's {host._block}"
            )
        self._block_steps = block_steps
        self._width = 0
        self._base = -1

        self._temperature = float(cfg.temperature)
        self._rep = float(cfg.repetition_penalty)
        self._min_p_log = float(np.log(cfg.min_p)) if cfg.min_p > 0.0 else None
        self._stop_token = host._stop_token

        # Scalars the graph reads instead of Python values, so a replay can be
        # told which step it is without being recaptured.
        self._step = torch.zeros((), dtype=torch.int64, device=device)
        self._floor = torch.zeros((), dtype=torch.int64, device=device)
        self._base_t = torch.zeros((), dtype=torch.int64, device=device)
        self._peak_prob = torch.zeros((), dtype=torch.float64, device=device)
        self._true = torch.ones(1, dtype=torch.bool, device=device)
        self._peak_at = torch.full((), -1, dtype=torch.int64, device=device)
        self._floor_stop = self._stop_token
        self._noise: Tensor | None = None
        self._seen: Tensor | None = None
        self._silence_mask: Tensor | None = None
        self._arange: Tensor | None = None

    # -- host-side control ---------------------------------------------------

    def prepare(
        self,
        width: int,
        seen: NDArray[np.bool_],
        *,
        floor: int,
        step: int,
        stop_token: int | None = None,
    ) -> None:
        """Size the buffers to a vocabulary and adopt the host's state.

        See ``docs/design/sampler-and-noise.md``.
        """
        import torch

        if width != self._width:
            self._width = width
            self._seen = torch.zeros(width, dtype=torch.bool, device=self._device)
            self._arange = torch.arange(width, dtype=torch.int64, device=self._device)
            self._noise = torch.zeros(
                self._block_steps, width, dtype=torch.float64, device=self._device
            )
            mask = np.zeros(width, dtype=bool)
            if self._cfg.silence_token_ids:
                mask[np.asarray(self._cfg.silence_token_ids, dtype=np.int64)] = True
            self._silence_mask = torch.from_numpy(mask).to(self._device)
            self._base = -1
        assert self._seen is not None
        self._seen.copy_(torch.from_numpy(np.ascontiguousarray(seen)))
        self._floor.fill_(floor)
        self._floor_stop = self._stop_token if stop_token is None else stop_token
        self.advance_to(step)
        self._peak_prob.zero_()
        self._peak_at.fill_(-1)

    def advance_to(self, step: int, *, draws: int = 1) -> None:
        """Point the buffers at ``step``, refilling the noise block if needed.

        See ``docs/design/sampler-and-noise.md``.
        """
        import torch

        self._step.fill_(step)
        base = (step // self._block_steps) * self._block_steps
        # A pair draws twice from one replay: the host sets the first token's
        # step and the graph increments for the second, so both must live in the
        # block that is resident. With an even block and pairs starting on even
        # steps they always do, but a silent out-of-range row read inside a
        # captured graph is not a failure anyone would trace back to here.
        last = step + draws - 1
        if last // self._block_steps * self._block_steps != base:
            raise ValueError(
                f"a noise block of {self._block_steps} steps cannot hold draws "
                f"{step}..{last}; the block must be a multiple of the draw group"
            )
        if base != self._base:
            assert self._noise is not None
            # From the host sampler's cache, not a second Philox pass: the
            # opening pair of a fused generation is drawn on the host, so that
            # block already exists by the time this runs.
            block = self._host.noise_block(base, self._width)
            self._noise.copy_(torch.from_numpy(block))
            self._base = base
            self._base_t.fill_(base)

    def rebind(self, host: LRSamplerV1) -> None:
        """Point at a different host sampler, keeping every buffer address.

        A captured graph holds addresses, and the seed changes per chunk, so a
        reused graph needs the noise *contents* swapped rather than the tensor
        replaced. Dropping the resident block forces the next
        :meth:`advance_to` to upload the new stream's rows into the same
        memory.
        """
        if host.config is not self._cfg and host.config != self._cfg:
            raise ValueError("a rebound sampler must carry the same sampling law")
        if host._stop_token != self._stop_token:
            raise ValueError("a rebound sampler must carry the same stop token")
        self._host = host
        self._base = -1

    def flush_peak(self) -> None:
        """Hand the device's EOS observation back to the host sampler."""
        prob = float(self._peak_prob.item())
        at = int(self._peak_at.item())
        if at >= 0:
            self._host.absorb_peak(at, prob)

    @property
    def law(self) -> tuple[object, ...]:
        """What a captured graph baked in, as a cache key.

        The temperature, the penalty and the cutoff are Python constants at
        capture time, not buffers, so a graph built under one law cannot serve
        another. Naming them here means a second law gets a second graph
        instead of silently getting the first one's arithmetic.
        """
        c = self._cfg
        return (
            c.temperature,
            c.repetition_penalty,
            c.min_p,
            tuple(c.silence_token_ids),
            self._stop_token,
            self._block_steps,
        )

    @property
    def host(self) -> LRSamplerV1:
        return self._host

    @property
    def seen(self) -> Tensor | None:
        return self._seen

    # -- the law, as capturable ops -----------------------------------------

    def select(self, logits: Tensor, *, valid: Tensor | None = None) -> Tensor:
        """One token from ``(1, vocab)`` logits. Capturable; updates ``seen``.

        See ``docs/design/sampler-and-noise.md``.
        """
        import torch

        assert self._seen is not None
        assert self._noise is not None
        assert self._arange is not None
        assert self._silence_mask is not None
        z = logits.reshape(-1).to(torch.float64)

        if self._floor_stop is not None:
            # The EOS floor, which the eager loop applies before calling the
            # sampler: below it the stop token cannot win.
            below = self._step < self._floor
            z = torch.where(
                below & (self._arange == self._floor_stop),
                torch.full_like(z, float("-inf")),
                z,
            )

        if self._rep != 1.0:
            penalised = torch.where(z > 0, z / self._rep, z * self._rep)
            z = torch.where(self._seen, penalised, z)

        s = z / self._temperature
        if self._min_p_log is not None:
            keep = s >= (s.max() + self._min_p_log)
        else:
            keep = torch.ones_like(s, dtype=torch.bool)
        keep = keep | self._silence_mask

        if self._stop_token is not None:
            self._observe_eos(s, keep, valid)

        row = self._noise.index_select(0, (self._step - self._base_t).reshape(1))[0]
        g = torch.where(keep, s + row, torch.full_like(s, float("-inf")))

        # Not `argmax`: its tie-breaking is unspecified, and the law says lowest
        # index. Taking the smallest index among the maxima says so outright.
        top = g.max()
        token = torch.where(
            g == top, self._arange, self._arange.new_full((), self._width)
        ).min()

        # `seen[token] = True` looks equivalent and is not: the Python `True` becomes a
        # host tensor and its copy to the device is "an operation not permitted when
        # stream is capturing". Scattering a device value keeps the update in the graph.
        at = token.reshape(1)
        mark = self._true if valid is None else torch.where(valid, self._true, self._seen[at])
        self._seen.scatter_(0, at, mark)
        self._step += 1
        return token

    def _observe_eos(self, s: Tensor, keep: Tensor, valid: Tensor | None = None) -> None:
        """The running EOS peak, updated in place. Never changes the draw.

        ``valid`` gates the update rather than the computation: see
        :meth:`select` for why a fused pair can draw a token the sequence does
        not contain, and why that token must not move the peak.
        """
        import torch

        assert self._stop_token is not None
        weights = torch.exp(s - s.max())
        total = torch.where(keep, weights, torch.zeros_like(weights)).sum()
        prob = torch.where(
            (total > 0) & (self._step > self._floor),
            weights[self._stop_token] / total.clamp_min(torch.finfo(torch.float64).tiny),
            torch.zeros_like(total),
        )
        better = prob > self._peak_prob
        if valid is not None:
            better = better & valid
        self._peak_at.copy_(torch.where(better, self._step, self._peak_at))
        self._peak_prob.copy_(torch.where(better, prob, self._peak_prob))
