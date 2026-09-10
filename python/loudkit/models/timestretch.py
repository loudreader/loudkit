"""Playing faster without talking higher, WSOLA, from first principles.

See ``docs/design/models-notes.md``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["MAX_SPEED", "MIN_SPEED", "stretched_length", "time_stretch", "validate_speed"]

Waveform = NDArray[np.float32]

MIN_SPEED = 0.5
MAX_SPEED = 2.0
"""The range worth offering, not the range that runs.

Outside it the alignment search stops finding matches often enough, the
required shift exceeds the ±10 ms it may look over, and the output is
recognisably processed rather than merely faster. Refused rather than clamped: a
caller who asked for 3x and silently got 2x has a bug that only a stopwatch
finds.
"""

_FRAME_MS = 25.0
"""Analysis/synthesis frame. Long enough to hold two periods of the lowest voiced
pitch this is used on (~80 Hz), short enough that a frame is inside one phone."""

_SEARCH_MS = 10.0
"""How far the read position may move to find a better join, a bit under one
pitch period at the low end of the voiced range, which is what the search is
looking for."""

_HANN_COLA_HOP = 2
"""Frames overlap by half. A periodic Hann window at hop = frame/2 sums to
exactly one, so the overlap-add needs no normalisation of its own, the
denominator below only ever corrects the ends and the places the alignment
search moved a frame off the grid."""


def validate_speed(speed: float) -> float:
    """``speed`` if it is usable, or a ``ValueError`` that says the range.

    Kept beside the stretch rather than in the engine because
    :func:`time_stretch` has to refuse the same values whoever calls it, and
    the engine's two synthesis doors go through it. The transports refuse
    earlier, at their own door, so a bad speed comes back as that transport's
    own bad-request rather than as a ``ValueError`` raised inside the engine;
    they read ``MIN_SPEED`` and ``MAX_SPEED`` from here so the range is
    written once.
    """
    if not math.isfinite(speed):
        raise ValueError(f"speed must be a finite number, not {speed!r}")
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(
            f"speed {speed} is outside [{MIN_SPEED}, {MAX_SPEED}]. Beyond that "
            "range the time-stretch is audibly processed rather than merely "
            "faster or slower, so it is refused rather than clamped."
        )
    return speed


def stretched_length(n_samples: int, speed: float) -> int:
    """How long ``n_samples`` becomes at ``speed``.

    Written as ``floor(n / speed + 0.5)`` rather than ``round()`` on purpose:
    Python rounds halves to even, Go, Rust, Swift and JavaScript do not, and a
    one-sample disagreement between ports on an exact half is the kind of thing
    that is found six months later in a conformance run.
    """
    return int(math.floor(n_samples / speed + 0.5))


def time_stretch(audio: Waveform, *, sample_rate: int, speed: float) -> Waveform:
    """``audio`` played at ``speed``, same pitch.

    At speed 1.0 the argument itself comes back, where every other path returns
    a new array. That asymmetry is the contract, not an oversight: the default
    render must not depend on this DSP path being lossless, so it must not
    enter it at all, and identity is what proves it did not. Pinned by
    ``tests/test_timestretch.py``. Callers who intend to edit the result in
    place copy it.

    See ``docs/design/models-notes.md``.
    """
    validate_speed(speed)
    if speed == 1.0:
        return audio

    n = int(audio.shape[0])
    out_len = stretched_length(n, speed)
    frame = int(math.floor(sample_rate * _FRAME_MS / 1000.0 + 0.5))
    hop = frame // _HANN_COLA_HOP
    if n <= frame or out_len <= 0 or hop <= 0:
        # Nothing to overlap-add: a fragment shorter than one frame has no second frame
        # to align against.
        out = np.zeros(max(out_len, 0), dtype=np.float32)
        out[: min(out_len, n)] = audio[: min(out_len, n)]
        return out

    search = int(math.floor(sample_rate * _SEARCH_MS / 1000.0 + 0.5))
    # Periodic Hann, i.e. 2*pi*i/frame and not /(frame-1). The periodic form is
    # the one that sums to exactly one at 50 % overlap; the symmetric form is off
    # by a hair at every frame boundary, which reads as a low-level buzz at the
    # frame rate, 40 Hz here, right in the range a listener notices.
    window = 0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(frame, dtype=np.float64) / frame)

    x = audio.astype(np.float64, copy=False)
    # Room for the last frame to be written whole; trimmed at the end.
    acc = np.zeros(out_len + frame, dtype=np.float64)
    weight = np.zeros(out_len + frame, dtype=np.float64)

    last_frame_at = 0
    write_at = 0
    k = 0
    while write_at < out_len:
        ideal = int(math.floor(k * hop * speed + 0.5))
        if k == 0:
            read_at = 0
        else:
            # What the previous frame would naturally have been followed by. The
            # search asks which nearby segment continues *this*, not which one
            # the arithmetic pointed at.
            target = x[last_frame_at + hop : last_frame_at + hop + frame]
            read_at = _best_match(x, target, ideal, search, frame)
        read_at = min(max(read_at, 0), n - frame)

        segment = x[read_at : read_at + frame]
        acc[write_at : write_at + frame] += window * segment
        weight[write_at : write_at + frame] += window

        last_frame_at = min(read_at, n - frame - hop) if n >= frame + hop else read_at
        write_at += hop
        k += 1

    # The Hann pair sums to one in the interior, so this division is the
    # identity almost everywhere; it earns its place at the two ends, where only
    # one frame contributes and the raw sum would fade in and out.
    out = np.zeros(out_len, dtype=np.float64)
    head = weight[:out_len]
    np.divide(acc[:out_len], head, out=out, where=head > 1e-12)
    return out.astype(np.float32)


def _best_match(
    x: NDArray[np.float64],
    target: NDArray[np.float64],
    ideal: int,
    search: int,
    frame: int,
) -> int:
    """The offset within ±``search`` of ``ideal`` whose frame best continues ``target``.

    See ``docs/design/models-notes.md``.
    """
    n = int(x.shape[0])
    lo = max(0, ideal - search)
    hi = min(n - frame, ideal + search)
    if hi < lo or target.shape[0] < frame:
        return min(max(ideal, 0), n - frame)

    best_at = lo
    best_score = -math.inf
    for at in range(lo, hi + 1):
        candidate = x[at : at + frame]
        energy = float(np.dot(candidate, candidate))
        # A silent candidate scores zero rather than dividing by nothing.
        score = 0.0 if energy <= 0.0 else float(np.dot(candidate, target)) / math.sqrt(energy)
        if score > best_score:
            best_score = score
            best_at = at
    return best_at
