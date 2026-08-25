"""A random number generator that gives the same answer everywhere.

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
therefore generate in any order — one number at a time, or a whole block ahead
— and still agree. That freedom is what makes the sampler affordable: generated
per token it costs 2.2 ms, more than the entire 16-layer model; generated a
block at a time it costs under 0.01 ms, and because the numbers are addressed
rather than streamed, the block boundary is invisible.

*Verifiable.* The implementation is checked against the published known-answer
vectors from the reference library, so a port can be validated against a
standard rather than against this implementation.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = ["philox_4x32_10", "uniforms", "gumbel_noise", "KAT_VECTORS"]

_M0 = np.uint64(0xD2511F53)
_M1 = np.uint64(0xCD9E8D57)
_W0 = np.uint32(0x9E3779B9)  # golden ratio
_W1 = np.uint32(0xBB67AE85)  # sqrt(3) - 1
_MASK = np.uint64(0xFFFFFFFF)
_ROUNDS = 10

KAT_VECTORS: tuple[
    tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int, int, int]], ...
] = (
    ((0x00000000,) * 4, (0x00000000,) * 2, (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
)
"""Known-answer vectors from the Random123 reference test suite.

The all-zero and all-ones cases pin the round function; the third (digits of pi
as the counter) pins the key schedule, which is where reimplementations usually
go wrong. :func:`loudkit.rng.selftest` checks all three.
"""


def _mulhilo(
    a: NDArray[np.uint64], b: np.uint64
) -> tuple[NDArray[np.uint64], NDArray[np.uint64]]:
    """32x32 -> (high, low), computed in 64 bits so the product is exact."""
    prod = a * b
    return (prod >> np.uint64(32)) & _MASK, prod & _MASK


def philox_4x32_10(
    c0: NDArray[np.uint64],
    c1: NDArray[np.uint64],
    c2: NDArray[np.uint64],
    c3: NDArray[np.uint64],
    k0: int,
    k1: int,
) -> tuple[NDArray[np.uint64], ...]:
    """Ten rounds of Philox-4x32 over element-wise counters.

    Counters and keys are 32-bit values carried in ``uint64`` lanes so the
    intermediate products do not overflow. Returns four uint32 streams.
    """
    key0 = np.uint64(k0 & 0xFFFFFFFF)
    key1 = np.uint64(k1 & 0xFFFFFFFF)
    for _ in range(_ROUNDS):
        hi0, lo0 = _mulhilo(c0, _M0)
        hi1, lo1 = _mulhilo(c2, _M1)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ key0) & _MASK,
            lo1,
            (hi0 ^ c3 ^ key1) & _MASK,
            lo0,
        )
        key0 = (key0 + np.uint64(_W0)) & _MASK
        key1 = (key1 + np.uint64(_W1)) & _MASK
    return c0, c1, c2, c3


def uniforms(
    seed: int, stream: int, step0: int, n_steps: int, width: int
) -> NDArray[np.float64]:
    """``(n_steps, width)`` uniforms in the open interval (0, 1).

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
        width: numbers per step — the vocabulary size, for sampling.
    """
    quads = (width + 3) // 4
    idx = np.arange(quads, dtype=np.uint64)[None, :].repeat(n_steps, axis=0)
    stp = (np.arange(n_steps, dtype=np.uint64) + np.uint64(step0))[:, None].repeat(
        quads, axis=1
    )
    zero = np.zeros_like(idx)
    r0, r1, r2, r3 = philox_4x32_10(
        idx, stp & _MASK, zero + np.uint64(stream & 0xFFFFFFFF), zero, seed, seed >> 32
    )
    bits = np.stack([r0, r1, r2, r3], axis=-1).reshape(n_steps, quads * 4)[:, :width]
    return (bits.astype(np.float64) + 0.5) / 4294967296.0


def gumbel_noise(
    seed: int, stream: int, step0: int, n_steps: int, width: int
) -> NDArray[np.float64]:
    """``-log(-log(u))``, the additive form of a categorical draw.

    Precomputed for a whole block because the two logarithms depend only on the
    counter. Sampling then costs one add, one mask and one argmax per token, and
    the argmax is order-independent up to ties — which are broken by lowest
    index, so two backends cannot disagree even there.
    """
    u = uniforms(seed, stream, step0, n_steps, width)
    return cast(NDArray[np.float64], -np.log(-np.log(u)))


def selftest() -> None:
    """Check the implementation against the published vectors.

    Raises:
        AssertionError: if any vector fails, which means this is not Philox and
            no port can be validated against it.
    """
    for ctr, key, want in KAT_VECTORS:
        c0, c1, c2, c3 = (np.array([c], dtype=np.uint64) for c in ctr)
        got = philox_4x32_10(c0, c1, c2, c3, key[0], key[1])
        got_t = tuple(int(g[0]) for g in got)
        if got_t != want:
            raise AssertionError(
                f"Philox KAT failed for ctr={ctr[0]:#010x} key={key[0]:#010x}: "
                f"got {tuple(hex(g) for g in got_t)}, want {tuple(hex(w) for w in want)}"
            )
