"""Judge and cut reference recordings without loading a model runtime.

Refusing a clip the enrollment contract cannot honour, and cutting the prompt
so it ends in silence. Both are signal processing over the samples a user
handed in, needing no weights and no torch, so they sit below every enroller.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Waveform = NDArray[np.float32]  # as models/timestretch.py spells it; a leaf

_MIN_ENROLL_SECONDS = 1.0
"""Below this there is not enough signal to estimate a speaker: the utterance
encoder's first partial alone covers 1.6 s and is zero-padded under it, so a
sub-second clip enrolls mostly padding."""

_MAX_ENROLL_SECONDS = 30.0
"""Above this the input contract stops being honest. The prompt uses the first
10 s and the speaker embedding reads the whole clip, so a five-minute recording
produces a voice mostly shaped by audio the docs say is ignored. Refused rather
than truncated: the user picked that recording for a reason, and silently using
a different slice of it is worse than asking them to choose."""

_SILENCE_PEAK = 1e-4
"""A clip whose loudest sample is under this is silence at any playback level;
there is no voice in it to enroll."""

_GOOD_INPUT = (
    "A good input is 5 to 10 seconds of one person speaking, clean, "
    "without music or a second voice."
)


def validate_reference_audio(wav: Waveform, sample_rate: int) -> None:
    """Refuse a recording the enrollment contract cannot honour.

    See ``docs/design/models-notes.md``.
    """
    # The two shape checks word their failures as "positive" and "mono":
    # callers and tests match on those words, so they are part of the contract.
    if sample_rate <= 0:
        raise ValueError(f"sample rate must be positive, got {sample_rate}")
    if wav.ndim != 1:
        raise ValueError(f"audio must be mono 1-D, got shape {wav.shape}")
    # Finiteness before anything arithmetic: one NaN poisons every statistic
    # below and every tensor downstream.
    if not bool(np.isfinite(wav).all()):
        raise ValueError(
            "the recording contains NaN or Inf samples, so no voice can be "
            "derived from it. Re-export the file. " + _GOOD_INPUT
        )
    seconds = wav.size / sample_rate
    if seconds < _MIN_ENROLL_SECONDS:
        raise ValueError(
            f"the recording is {seconds:.2f} s: too short to enroll a speaker "
            f"from (minimum {_MIN_ENROLL_SECONDS:g} s). " + _GOOD_INPUT
        )
    if seconds > _MAX_ENROLL_SECONDS:
        raise ValueError(
            f"the recording is {seconds:.1f} s. Only the first 10 s become the "
            "voice prompt, and the whole clip shapes the speaker embedding, so "
            "a long recording enrolls something the prompt does not carry. Trim "
            f"it to the best 5 to 10 seconds (at most {_MAX_ENROLL_SECONDS:g} s). "
            + _GOOD_INPUT
        )
    peak = float(np.abs(wav).max())
    if peak < _SILENCE_PEAK:
        raise ValueError(
            f"the recording is silent (peak {peak:.1e}); there is no voice in "
            "it to enroll. " + _GOOD_INPUT
        )


PROMPT_SECONDS = 10.0
"""How much of a reference clip becomes the renderer's prompt; see :func:`loudkit.enroll`."""

PAUSE_FRAME_SECONDS = 0.005
"""Seconds each loudness reading covers: short enough to see the gap between
two words, long enough for the RMS to mean something."""

PAUSE_FLOOR_DBFS = -45.0
"""A frame quieter than this is silence, when looking for a pause to end on."""

PAUSE_MIN_SECONDS = 0.06
"""The shortest run of quiet frames that counts as a pause."""

CLIP_MIN_SECONDS = 3.0
"""Ignore pauses that would leave less than this much reference audio."""

END_SILENCE_SECONDS = 0.4
"""Silence appended after the cut, so the prompt ends in it whatever the clip did."""


def with_prompt_ending_in_silence(samples: Waveform, sample_rate: int) -> Waveform:
    """Append silence after the last usable pause in the ten-second prompt.

    Prefer the last pause of at least 60 ms after the first three seconds.
    A pause near ten seconds leaves only part of the 0.4-second pad inside
    the prompt; keep it rather than cutting an earlier word to fit the pad.
    Without a pause, truncate at 9.6 seconds if needed to leave room for silence.
    ``enroll(..., end_in_silence=True)`` asks for this cut; the default leaves
    the recording as it is.

    Measurement and limits: docs/design/embedding.md.
    """
    tail = np.zeros(int(END_SILENCE_SECONDS * sample_rate), dtype=samples.dtype)
    limit = int(PROMPT_SECONDS * sample_rate)
    cut = min(len(samples), limit - len(tail))
    frame = max(1, int(PAUSE_FRAME_SECONDS * sample_rate))
    n = min(len(samples), limit) // frame
    if n:
        rms = np.sqrt((samples[: n * frame].reshape(n, frame).astype(np.float64) ** 2).mean(1))
        min_frames = max(1, int(np.ceil(PAUSE_MIN_SECONDS * sample_rate / frame)))
        run = 0
        for k, quiet in enumerate(rms < 10 ** (PAUSE_FLOOR_DBFS / 20)):
            run = run + 1 if quiet else 0
            if run == min_frames:
                at = (k + 1 - run) * frame
                if at >= CLIP_MIN_SECONDS * sample_rate:
                    cut = at
    return np.concatenate((samples[:cut], tail))
