"""The renderer's geometry and randomness addressing, torch-free.

See ``docs/design/models-notes.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..config import AlgorithmConfig
from ..contracts import MEL_BINS, TOKEN_MEL_RATIO
from ..errors import WindowOverflowError
from ..voice import VoiceProfile

__all__ = [
    "START_TEXT_TOKEN",
    "STOP_TEXT_TOKEN",
    "FLOW_NOISE_STREAM",
    "VOCODER_PHASE_STREAM",
    "VOCODER_NOISE_STREAM",
    "VOCODER_HARMONICS",
    "UPSAMPLE_PER_FRAME",
    "time_grid",
    "pad_token_id",
    "frame_windows",
    "FramedWindow",
    "eos_floor",
]

START_TEXT_TOKEN = 255
"""Text framing token that opens the transcript segment. A property of the T3
text tokenizer family (shared with the torch generator module), and it lives
here rather than beside the torch generator so the ONNX backend can frame the
same way without importing torch."""

STOP_TEXT_TOKEN = 0
"""Text framing token that closes the transcript segment. See
:data:`START_TEXT_TOKEN`."""

FLOW_NOISE_STREAM = 0
"""Philox sub-stream (under the stage seed) for the CFM prior. Streams 0 and 1
are consumed by the Box–Muller pair; keep any future draw at >= 2."""

VOCODER_PHASE_STREAM = 0
"""Philox sub-stream for the 8 harmonic phase offsets (row 0 is pinned to 0 -
the voiced fundamental must start at a zero crossing)."""

VOCODER_NOISE_STREAM = 1
"""Philox sub-streams 1 and 2 (Box–Muller pair) for the excitation noise."""

VOCODER_HARMONICS = 9
"""Excitation rows the source module carries: harmonic_num 8 plus the fundamental.

Here rather than beside the torch vocoder for the same reason as
:data:`START_TEXT_TOKEN`: the graph backends draw the same noise and cannot
import torch to learn its shape."""

UPSAMPLE_PER_FRAME = 480
"""Audio samples one mel frame becomes: 24 kHz over the 50 Hz mel rate.

Every renderer sizes its excitation and cuts its output with this, so it is one
number rather than one per backend."""


def time_grid(config: AlgorithmConfig) -> list[float]:
    """The Euler time grid: the explicit one if the config carries it, else
    the cosine schedule ``t_i = 1 − cos(i/K · π/2)``, one implementation,
    shared by every renderer so "cosine" cannot be written two ways."""
    if config.euler_grid is not None:
        return list(config.euler_grid)
    k = config.euler_steps
    return [1.0 - math.cos(i / k * math.pi / 2.0) for i in range(k + 1)]


def pad_token_id(config: AlgorithmConfig) -> int:
    """The token that fills unused static-window slots."""
    if config.window.pad_token_id is not None:
        return config.window.pad_token_id
    if config.sampling.silence_token_ids:
        return config.sampling.silence_token_ids[0]
    raise ValueError(
        "static window needs a pad token: set WindowConfig.pad_token_id or "
        "provide silence_token_ids: padding with token 0 bleeds +3 dB of "
        "high-band energy into the tail through the encoder's attention"
    )


def eos_floor(n_text_tokens: int, config: AlgorithmConfig) -> int:
    """Minimum speech tokens before the stop token becomes sampleable."""
    s = config.sampling
    return max(s.min_tokens_floor, int(n_text_tokens * s.min_tokens_text_ratio))


class FramedWindow(NamedTuple):
    """One window as the renderers take it, from :func:`frame_windows`.

    A tuple, so the positional unpack every backend already writes keeps
    working; named, so a reader of ``framed[2]`` three modules away can tell
    the prompt length from the token count.
    """

    row: NDArray[np.int64]
    """Prompt and query token ids, concatenated."""
    cond: NDArray[np.float32]
    """The prompt mel the flow decoder conditions on."""
    prompt_frames: int
    """Mel frames of prompt, the offset the rendered audio starts at."""
    n: int
    """Speech tokens in the query, before any static padding."""


def frame_windows(
    config: AlgorithmConfig, tokens: Sequence[int], voice: VoiceProfile
) -> FramedWindow:
    """Apply the window recipe; shared by every renderer backend.

    See ``docs/design/models-notes.md``.
    """
    w = config.window
    # Refused rather than trimmed, which is what Rust, Go, JS and Swift already do in
    # this same function and what `Engine` does one layer up.
    toks_all = [int(t) for t in tokens]
    if len(toks_all) > w.max_speech_tokens:
        raise WindowOverflowError(
            f"{len(toks_all)} speech tokens exceeds the "
            f"{w.max_speech_tokens}-token window; split the text into chunks",
            n_tokens=len(toks_all),
            window=w.max_speech_tokens,
        )
    toks = np.asarray(toks_all, dtype=np.int64)
    n = len(toks)
    prompt_tokens = np.asarray(voice.prompt_tokens, dtype=np.int64)
    prompt_mel = np.asarray(voice.prompt_mel, dtype=np.float32)

    prompt: NDArray[np.int64]
    query: NDArray[np.int64]
    if w.static_length is not None:
        pad = pad_token_id(config)
        p_len = (
            w.static_prompt_tokens if w.static_prompt_tokens is not None else len(prompt_tokens)
        )
        prompt = np.full(p_len, pad, dtype=np.int64)
        keep = min(len(prompt_tokens), p_len)
        prompt[:keep] = prompt_tokens[:keep]
        query = np.full(w.static_length, pad, dtype=np.int64)
        query[:n] = toks
    else:
        prompt = prompt_tokens
        query = toks

    row = np.concatenate([prompt, query])[None]
    t_mel = TOKEN_MEL_RATIO * row.shape[1]
    prompt_frames = TOKEN_MEL_RATIO * len(prompt)
    cond = np.zeros((1, MEL_BINS, t_mel), dtype=np.float32)
    keep_f = min(prompt_mel.shape[1], prompt_frames)
    cond[0, :, :keep_f] = prompt_mel[:, :keep_f]
    return FramedWindow(row, cond, prompt_frames, n)
