"""Render randomness as data, not as generator state.

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
second sample share its radius — a period-2 structure that lands exactly on
Nyquist, measured at +5.3 dB at 12 kHz in the raw stream and audible as a
whine after the vocoder's excitation path (see the shipped engine's RNG note).
Two uniforms per Gaussian is the price of not shipping that artefact again.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from ..rng import uniforms

__all__ = ["gaussian_field", "symmetric_uniforms"]


def gaussian_field(seed: int, stream: int, rows: int, cols: int) -> NDArray[np.float32]:
    """``(rows, cols)`` standard normals, addressed by ``(seed, stream)``.

    Uses two Philox sub-streams (``stream`` and ``stream + 1``) for the two
    Box–Muller uniforms, so callers must space their stream ids by two. The
    uniforms are open-interval by construction (see :func:`loudkit.rng.uniforms`),
    hence the logarithm is finite without a clamp.
    """
    u1 = uniforms(seed, stream, 0, rows, cols)
    u2 = uniforms(seed, stream + 1, 0, rows, cols)
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return cast(NDArray[np.float32], z.astype(np.float32))


def symmetric_uniforms(
    seed: int, stream: int, n: int, half_width: float
) -> NDArray[np.float32]:
    """``(n,)`` uniforms in ``(-half_width, half_width)``, counter-addressed."""
    u: NDArray[np.float64] = uniforms(seed, stream, 0, 1, n)[0]
    return ((u * 2.0 - 1.0) * half_width).astype(np.float32)
