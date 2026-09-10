"""The seams. Five components, five protocols, one direction of data.

See ``docs/design/engine-pipeline.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .config import AlgorithmConfig
from .voice import VoiceProfile

__all__ = [
    "TextFrontend",
    "VoiceEnroller",
    "TokenGenerator",
    "MelDecoder",
    "Vocoder",
    "Sampler",
    "SpeechTokens",
    "Mel",
    "Waveform",
    "MEL_BINS",
    "TOKEN_RATE_HZ",
    "TOKEN_MEL_RATIO",
]

SpeechTokens = Sequence[int]
"""Discrete speech tokens at 25 Hz. The interface between the two stages, and
the reason the whole pipeline is comparable: two backends either chose the same
tokens or they did not, and that is a yes-or-no question."""

Mel = NDArray[np.float32]
"""Log-mel spectrogram, ``(80, frames)``."""

Waveform = NDArray[np.float32]
"""Mono audio in [-1, 1] at ``AlgorithmConfig.sample_rate``."""


# -- the geometry of the seams above, stated once ----------------------------
# These three numbers describe the shapes that cross the boundaries this module
# draws.

MEL_BINS = 80
"""Mel bins per frame. The height of :data:`Mel` and the vocoder's input width."""

TOKEN_RATE_HZ = 25
"""Speech tokens per second of audio: one token is 40 ms."""

TOKEN_MEL_RATIO = 2
"""Mel frames per speech token, 25 Hz tokens become 50 Hz mel frames."""


@runtime_checkable
class TextFrontend(Protocol):
    """Text to text-tokens. Deterministic, no model state."""

    def encode(self, text: str, language: str = "en") -> NDArray[np.int64]:
        """Normalise and tokenise. Same text and language give the same ids."""
        ...


@runtime_checkable
class VoiceEnroller(Protocol):
    """Reference audio to a :class:`VoiceProfile`.

    Enrollment is deliberately separate from synthesis: it is slow, it needs
    models synthesis does not (a speaker encoder, a speech tokenizer, together
    about 40% of the checkpoint), and its result is a few hundred kilobytes of
    tensors that can be cached, shipped and versioned on their own.
    """

    def enroll(self, audio: Waveform, sample_rate: int, *, name: str = "") -> VoiceProfile: ...


@runtime_checkable
class Sampler(Protocol):
    """Logits to one token. The whole sampling law, and nothing else.

    See ``docs/design/engine-pipeline.md``.
    """

    def __call__(
        self,
        logits: NDArray[np.float32],
        *,
        step: int,
        seen: NDArray[np.bool_],
    ) -> int:
        """Choose the next token.

        See ``docs/design/engine-pipeline.md``.
        """
        ...

    @property
    def eos_peak(self) -> tuple[int, float]:
        """Where the model came closest to stopping, as ``(step, probability)``.

        See ``docs/design/engine-pipeline.md``.
        """
        ...


@runtime_checkable
class TokenGenerator(Protocol):
    """Text tokens and a voice to speech tokens. The autoregressive stage.

    Owns the loop but not the law: the sampler is injected, so a backend can
    change how the forward pass is executed without touching what is sampled.
    """

    config: AlgorithmConfig

    def generate(
        self,
        text_tokens: NDArray[np.int64],
        voice: VoiceProfile,
        *,
        sampler: Sampler,
        max_new_tokens: int | None = None,
        prefix: SpeechTokens = (),
        should_cancel: Callable[[], bool] | None = None,
    ) -> SpeechTokens:
        """Run to the stop token or the cap, whichever comes first.

        See ``docs/design/engine-pipeline.md``.
        """
        ...

    def teacher_forced_logits(
        self,
        text_tokens: NDArray[np.int64],
        voice: VoiceProfile,
        forced: SpeechTokens,
    ) -> NDArray[np.float32]:
        """Logits at each step when the given tokens are fed back regardless.

        See ``docs/design/engine-pipeline.md``.
        """
        ...


@runtime_checkable
class MelDecoder(Protocol):
    """Speech tokens and a voice to a mel. Non-autoregressive, whole sequence.

    The opposite shape from the token generator, one large parallel pass rather
    than hundreds of tiny serial ones, which is why the two stages disagree
    about which hardware they want: on Apple silicon this stage is faster on the
    GPU while the token generator is faster on the CPU.
    """

    config: AlgorithmConfig

    def decode(self, tokens: SpeechTokens, voice: VoiceProfile, *, seed: int) -> Mel:
        """Integrate the flow to a mel.

        ``seed`` is mandatory, not optional with a default. The prior is drawn
        from it, and an unseeded implementation of this stage produced waveforms
        correlating at 0.109 across two runs of *identical* tokens, larger than
        every effect we have ever tried to measure here.
        """
        ...


@runtime_checkable
class Vocoder(Protocol):
    """Mel to waveform.

    Ships in fp32 and should stay there. Half precision here puts an audible
    tone at Nyquist: the source module accumulates phase with a running sum that
    reaches ~1400 cycles, where fp16 resolution is coarser than the per-sample
    increment, and the excitation degenerates.
    """

    config: AlgorithmConfig

    def synthesize(self, mel: Mel, voice: VoiceProfile, *, seed: int) -> Waveform:
        """Render audio. ``seed`` drives the excitation noise; see
        :meth:`MelDecoder.decode` for why it is not optional."""
        ...
