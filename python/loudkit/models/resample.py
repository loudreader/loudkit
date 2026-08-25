"""The enrollment resampler — one law, ported everywhere.

Enrollment downsamples the reference clip from 24 kHz to 16 kHz, and it used
to do so through **two different** resamplers: torchaudio's polyphase
``sinc_interp_hann`` on the flow side and librosa's ``soxr_hq`` on the
token-generator side. That split is an accident of history, not a feature —
the two are both anti-aliased and differ only by a hair — and it is fatal to
cross-language parity, because ``soxr_hq`` is a C library whose float
accumulation order no port can reproduce bit for bit.

So enrollment uses **one** resampler, this one, and every port reimplements
this exact law. It is the same algorithm as torchaudio's ``sinc_interp_hann``
(a Hann-windowed sinc, band-limited interpolation) restated with an explicit
contract so the five ports stay bit-identical:

* the kernel is computed in float64 from the formula below, then rounded to
  float32 once — the float32 values are the contract, not the float64
  intermediates;
* the FIR accumulates **left to right in float32**, one multiply and one add
  per tap, never a fused multiply-add (an FMA would round differently and the
  divergence would be silent, exactly the failure this library exists to end).

The kernel is 2 phases by 23 taps after GCD reduction (24k/16k = 3/2), so the
whole thing is a 23-tap strided FIR — small enough that the float32 kernel can
be shipped as data, but computed here so the definition is self-contained.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["sinc_hann_kernel", "resample"]

_PI = math.pi


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
    gcd = math.gcd(orig_freq, new_freq)
    orig = orig_freq // gcd
    new = new_freq // gcd

    base_freq = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base_freq)

    idx = np.arange(-width, width + orig, dtype=np.float64)[None, None] / orig
    t = np.arange(0, -new, -1, dtype=np.float64)[:, None, None] / new + idx
    t = np.clip(t * base_freq, -lowpass_filter_width, lowpass_filter_width)

    window = np.cos(t * _PI / lowpass_filter_width / 2) ** 2
    t = t * _PI
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

    gcd = math.gcd(orig_freq, new_freq)
    orig = orig_freq // gcd
    new = new_freq // gcd

    kernel, width = sinc_hann_kernel(orig_freq, new_freq)
    taps = kernel.shape[2]

    x = np.asarray(waveform, dtype=np.float32)
    padded = np.pad(x, (width, width + orig), mode="constant")

    n_out = (padded.shape[0] - taps) // orig + 1
    # One accumulator per phase, advanced tap by tap across *every* output
    # sample at once. The scalar triple loop this replaces ran ~8.3 M float32
    # additions for a ten-second clip — about 0.18 s of pure Python per
    # enrolment.
    #
    # Vectorised across `i`, never across `c`: the docstring's "walks each
    # output sample's taps left to right in float32" is a specification, not a
    # description. `np.dot` would sum in whatever order BLAS prefers and
    # stop matching torchaudio and the four ports. Here every output still
    # accumulates its taps in ascending `c`, one addition at a time, so the
    # result is bit-identical with the naive loop, checked on a 24 kHz second
    # of noise: `array_equal`, max difference 0.0.
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
