"""The renderer's geometry and randomness addressing — torch-free.

Everything a renderer backend needs from the flow and vocoder modules that is
*not* a torch module: the window framing recipe, the Euler time grid, and the
Philox sub-stream ids that address the render randomness. A runtime-only
backend (ONNX, CoreML) imports this file and never touches a torch module,
which is the checkpoint module's promise ("a future runtime-only backend can
load the same file without dragging torch in") kept for the parts that are
pure geometry and bookkeeping.

The window recipe is the entire measured ANE-vs-torch mel deviation (corr
0.975–0.993) when implementations disagree, so it lives here as data, shared
by every renderer — not re-derived per backend.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ..config import AlgorithmConfig
from ..errors import WindowOverflowError
from ..voice import VoiceProfile

__all__ = [
    "START_TEXT_TOKEN",
    "STOP_TEXT_TOKEN",
    "FLOW_NOISE_STREAM",
    "VOCODER_PHASE_STREAM",
    "VOCODER_NOISE_STREAM",
    "time_grid",
    "pad_token_id",
    "frame_windows",
    "eos_floor",
]

START_TEXT_TOKEN = 255
"""Text framing token that opens the transcript segment. A property of the T3
text tokenizer family (shared with the torch generator module), moved here so
the ONNX backend can frame the same way without importing torch."""

STOP_TEXT_TOKEN = 0
"""Text framing token that closes the transcript segment. See
:data:`START_TEXT_TOKEN`."""

FLOW_NOISE_STREAM = 0
"""Philox sub-stream (under the stage seed) for the CFM prior. Streams 0 and 1
are consumed by the Box–Muller pair; keep any future draw at >= 2."""

VOCODER_PHASE_STREAM = 0
"""Philox sub-stream for the 8 harmonic phase offsets (row 0 is pinned to 0 —
the voiced fundamental must start at a zero crossing)."""

VOCODER_NOISE_STREAM = 1
"""Philox sub-streams 1 and 2 (Box–Muller pair) for the excitation noise."""

_TOKEN_MEL_RATIO = 2  # 25 Hz tokens -> 50 Hz mel frames
_MEL_BINS = 80


def time_grid(config: AlgorithmConfig) -> list[float]:
    """The Euler time grid: the explicit one if the config carries it, else
    the cosine schedule ``t_i = 1 − cos(i/K · π/2)`` — one implementation,
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
        "provide silence_token_ids — padding with token 0 bleeds +3 dB of "
        "high-band energy into the tail through the encoder's attention"
    )


def eos_floor(n_text_tokens: int, config: AlgorithmConfig) -> int:
    """Minimum speech tokens before the stop token becomes sampleable."""
    s = config.sampling
    return max(s.min_tokens_floor, int(n_text_tokens * s.min_tokens_text_ratio))


def frame_windows(
    config: AlgorithmConfig, tokens: Sequence[int], voice: VoiceProfile
) -> tuple[NDArray[np.int64], NDArray[np.float32], int, int]:
    """Apply the window recipe; shared by every renderer backend.

    Returns ``(token_row (1, P+Q), cond (1, 80, 2·(P+Q)), prompt_frames, n)``
    where ``n`` is the count of real speech tokens and ``prompt_frames`` the
    mel region to cut after integration. In static mode the prompt is framed
    to exactly ``static_prompt_tokens`` (truncate long, silence-pad short) and
    the query to ``static_length`` — the production recipe, which is the
    entire measured ANE-vs-torch mel deviation when implementations disagree.
    """
    w = config.window
    # Refused rather than trimmed, which is what Rust, Go, JS and Swift already
    # do in this same function and what `Engine` does one layer up. Python was
    # the last place a `.decode` called directly — bypassing the engine — took
    # 300 tokens and returned 255 tokens of audio with nothing to say the rest
    # had gone. In a reading tool that is the end of a passage simply not
    # existing while the audio sounds perfectly fine; the only listener who
    # notices is one who knows the text.
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
        p_len = w.static_prompt_tokens or len(prompt_tokens)
        prompt = np.full(p_len, pad, dtype=np.int64)
        keep = min(len(prompt_tokens), p_len)
        prompt[:keep] = prompt_tokens[:keep]
        query = np.full(w.static_length, pad, dtype=np.int64)
        query[:n] = toks
    else:
        prompt = prompt_tokens
        query = toks

    row = np.concatenate([prompt, query])[None]
    t_mel = _TOKEN_MEL_RATIO * row.shape[1]
    prompt_frames = _TOKEN_MEL_RATIO * len(prompt)
    cond = np.zeros((1, _MEL_BINS, t_mel), dtype=np.float32)
    keep_f = min(prompt_mel.shape[1], prompt_frames)
    cond[0, :, :keep_f] = prompt_mel[:, :keep_f]
    return row, cond, prompt_frames, n
