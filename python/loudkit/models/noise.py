"""Render randomness as data, not as generator state.

See ``docs/design/sampler-and-noise.md``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

# torch stays a runtime import inside the device functions below: this module is the
# portable definition of the noise, and the ONNX and CoreML backends import it without
# torch installed. The names are imported for typing only; see docs/design/typing.md.
if TYPE_CHECKING:
    import torch
    from torch import Tensor

from ..rng import uniforms

__all__ = ["gaussian_field", "gaussian_field_torch", "symmetric_uniforms"]


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


def gaussian_field_torch(
    seed: int, stream: int, rows: int, cols: int, device: str | torch.device
) -> Tensor:
    """:func:`gaussian_field`, computed where it is used.

    See ``docs/design/sampler-and-noise.md``.
    """
    import torch

    dev = torch.device(device)
    u1 = _uniforms_torch(seed, stream, rows, cols, dev)
    u2 = _uniforms_torch(seed, stream + 1, rows, cols, dev)
    z = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    return z.to(torch.float32)


def _uniforms_torch(
    seed: int, stream: int, rows: int, cols: int, device: torch.device
) -> Tensor:
    """``(rows, cols)`` uniforms in (0, 1), on ``device``, addressed as NumPy's are."""
    import torch

    quads = (cols + 3) // 4
    idx = torch.arange(quads, dtype=torch.int64, device=device).expand(rows, quads)
    stp = torch.arange(rows, dtype=torch.int64, device=device).unsqueeze(1).expand(rows, quads)
    zero = torch.zeros((), dtype=torch.int64, device=device).expand(rows, quads)
    r = _philox_torch(idx, stp, zero + (stream & 0xFFFFFFFF), zero, seed, seed >> 32)
    bits = torch.stack(r, dim=-1).reshape(rows, quads * 4)[:, :cols]
    return (bits.to(torch.float64) + 0.5) / 4294967296.0


def _mulhilo_torch(a: Tensor, b: int) -> tuple[Tensor, Tensor]:
    """32x32 -> (high, low) without a 64-bit unsigned type.

    torch has no ``uint64``, and the product of two 32-bit values overflows a
    *signed* 64-bit lane's positive range, where ``>> 32`` would then shift in
    sign bits. Splitting both operands into 16-bit limbs keeps every partial
    product under 2**33 and reconstructs the same two words the NumPy path gets
    from a single multiply.
    """
    a_lo, a_hi = a & 0xFFFF, a >> 16
    b_lo, b_hi = b & 0xFFFF, b >> 16

    t0 = a_lo * b_lo
    w0 = t0 & 0xFFFF
    t1 = a_hi * b_lo + (t0 >> 16)
    w1 = t1 & 0xFFFF
    w2 = t1 >> 16
    t2 = a_lo * b_hi + w1
    high = a_hi * b_hi + w2 + (t2 >> 16)
    low = ((t2 & 0xFFFF) << 16) | w0
    return high & 0xFFFFFFFF, low & 0xFFFFFFFF


def _philox_torch(
    c0: Tensor, c1: Tensor, c2: Tensor, c3: Tensor, k0: int, k1: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Ten rounds of Philox-4x32, mirroring :func:`loudkit.rng.philox_4x32_10`."""
    key0 = k0 & 0xFFFFFFFF
    key1 = k1 & 0xFFFFFFFF
    for _ in range(10):
        hi0, lo0 = _mulhilo_torch(c0, 0xD2511F53)
        hi1, lo1 = _mulhilo_torch(c2, 0xCD9E8D57)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ key0) & 0xFFFFFFFF,
            lo1,
            (hi0 ^ c3 ^ key1) & 0xFFFFFFFF,
            lo0,
        )
        key0 = (key0 + 0x9E3779B9) & 0xFFFFFFFF
        key1 = (key1 + 0xBB67AE85) & 0xFFFFFFFF
    return c0, c1, c2, c3
