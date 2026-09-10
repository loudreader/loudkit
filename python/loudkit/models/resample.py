"""The enrollment resampler, one law, ported everywhere.

See ``docs/design/models-notes.md``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["sinc_hann_kernel", "resample"]


def reduced_rates(orig_freq: int, new_freq: int) -> tuple[int, int]:
    """``(orig, new)`` divided by their gcd: the phase count the kernel is built for.

    Both the kernel and the FIR that walks it are indexed by these, so they are
    reduced once here rather than once in each.
    """
    gcd = math.gcd(orig_freq, new_freq)
    return orig_freq // gcd, new_freq // gcd


def sinc_hann_kernel(
    orig_freq: int,
    new_freq: int,
    *,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
) -> tuple[NDArray[np.float32], int]:
    """The float32 Hann-windowed-sinc kernel and its half-width, after GCD reduction.

    Mirrors torchaudio's ``_get_sinc_resample_kernel`` for
    ``resampling_method="sinc_interp_hann"``, with the gcd reduction its
    ``transforms.Resample`` applies. Returns ``(kernel, width)`` where
    ``kernel`` is ``[new, 1, 2*width + orig]`` float32.
    """
    orig, new = reduced_rates(orig_freq, new_freq)

    base_freq = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base_freq)

    idx = np.arange(-width, width + orig, dtype=np.float64)[None, None] / orig
    t = np.arange(0, -new, -1, dtype=np.float64)[:, None, None] / new + idx
    t = np.clip(t * base_freq, -lowpass_filter_width, lowpass_filter_width)

    window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t = t * math.pi
    scale = base_freq / orig

    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = np.sin(t) / t
    sinc[t == 0.0] = 1.0

    kernel = (sinc * window * scale).astype(np.float32)
    return kernel, width


def resample(
    waveform: NDArray[np.float32], orig_freq: int, new_freq: int
) -> NDArray[np.float32]:
    """Downsample/upsample ``waveform`` with the sinc-hann kernel, in float32.

    ``waveform`` is 1-D float32. The FIR walks each output sample's taps left
    to right in float32; the output length matches torchaudio's (``ceil(new /
    orig * length)`` after gcd reduction).
    """
    if orig_freq == new_freq:
        return np.asarray(waveform, dtype=np.float32)

    orig, new = reduced_rates(orig_freq, new_freq)
    kernel, width = sinc_hann_kernel(orig_freq, new_freq)
    taps = kernel.shape[2]

    x = np.asarray(waveform, dtype=np.float32)
    padded = np.pad(x, (width, width + orig), mode="constant")

    n_out = (padded.shape[0] - taps) // orig + 1
    # One accumulator per phase, advanced tap by tap across *every* output sample at
    # once.
    acc = np.zeros((new, n_out), dtype=np.float32)
    bases = np.arange(n_out) * orig
    for phase in range(new):
        row = acc[phase]
        for c in range(taps):
            row += np.float32(kernel[phase, 0, c]) * padded[bases + c]
    # Phase-major to sample-major: output `i * new + phase`.
    out = acc.T.reshape(-1)

    target = math.ceil(new * x.shape[0] / orig)
    return np.asarray(out[:target], dtype=np.float32)
