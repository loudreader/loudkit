"""The seams. Five components, five protocols, one direction of data.

::

    text ──▶ TextFrontend ──▶ text tokens
                                   │
    voice ─▶ VoiceEnroller ──▶ VoiceProfile
                                   │
                                   ▼
                            TokenGenerator ──▶ speech tokens   (25 Hz, discrete)
                                   │
                                   ▼
                              MelDecoder ──▶ mel               (80 bins)
                                   │
                                   ▼
                                Vocoder ──▶ waveform           (24 kHz)

Every protocol takes an :class:`~loudkit.config.AlgorithmConfig` and is forbidden
from carrying algorithm state of its own. A backend supplies implementations;
it does not supply behaviour.

Why protocols rather than base classes: an implementation may be a torch module,
an ONNX session, or a CoreML package, and none of those want a shared ancestor.
What they share is a shape of call, which is what a protocol says and a base
class only implies.

**The contract that matters most** is that these boundaries are *value*
boundaries. Tokens are integers, a mel is an array, a waveform is an array. No
component hands another a live model object, a device handle, or a cache. That
is what makes it possible to run the token generator on the CPU and the mel
decoder on the GPU — which, on Apple silicon, is measurably the right split —
and to compare two backends stage by stage when they disagree.
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
]

SpeechTokens = Sequence[int]
"""Discrete speech tokens at 25 Hz. The interface between the two stages, and
the reason the whole pipeline is comparable: two backends either chose the same
tokens or they did not, and that is a yes-or-no question."""

Mel = NDArray[np.float32]
"""Log-mel spectrogram, ``(80, frames)``."""

Waveform = NDArray[np.float32]
"""Mono audio in [-1, 1] at ``AlgorithmConfig.sample_rate``."""


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
    models synthesis does not (a speaker encoder, a speech tokenizer — together
    about 40% of the checkpoint), and its result is a few hundred kilobytes of
    tensors that can be cached, shipped and versioned on their own.
    """

    def enroll(self, audio: Waveform, sample_rate: int, *, name: str = "") -> VoiceProfile: ...


@runtime_checkable
class Sampler(Protocol):
    """Logits to one token. The whole sampling law, and nothing else.

    Kept a component rather than a function because it owns the RNG stream, and
    the RNG stream is the single thing most likely to make two correct backends
    disagree. ``torch.multinomial`` gives different samples for the same
    probability vector and the same generator on x86 and arm64.
    """

    def __call__(
        self,
        logits: NDArray[np.float32],
        *,
        step: int,
        seen: NDArray[np.bool_],
    ) -> int:
        """Choose the next token.

        Args:
            logits: raw scores over the speech vocabulary, unnormalised.
            step: index of this decode step. The RNG is addressed by it, so the
                result does not depend on how many tokens were drawn before —
                which is what lets two backends agree while computing in
                different orders.
            seen: which tokens have already been emitted, for the repetition
                penalty. Silence tokens are exempt; see
                :class:`~loudkit.config.SamplingConfig`.
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

        Args:
            prefix: speech tokens from the preceding chunk, fed in as context
                and **not** included in the return value.

                This parameter is why long-form reading does not stutter.
                Generated independently, each chunk restarts its pitch contour
                like a fresh sentence, and the restart is audible at every join;
                conditioning on the tail of the previous chunk removes it.

                It is in the protocol from the first release on purpose. Adding
                a parameter to a ``Protocol`` after other people have written
                implementations against it breaks all of them, and this is the
                one extension we already know is coming.
        """
        ...

    def teacher_forced_logits(
        self,
        text_tokens: NDArray[np.int64],
        voice: VoiceProfile,
        forced: SpeechTokens,
    ) -> NDArray[np.float32]:
        """Logits at each step when the given tokens are fed back regardless.

        Required by every implementation because it is the only comparison
        between backends that is not confounded by chaos: once two free-running
        generations differ by one token they are reading different histories,
        and every later logit is incomparable. Teacher forcing holds the context
        identical and asks only what the arithmetic did.

        Returns ``(len(forced) + 1, vocab)``.
        """
        ...


@runtime_checkable
class MelDecoder(Protocol):
    """Speech tokens and a voice to a mel. Non-autoregressive, whole sequence.

    The opposite shape from the token generator — one large parallel pass rather
    than hundreds of tiny serial ones — which is why the two stages disagree
    about which hardware they want. Measured on an M3 Pro: this stage is 2.6x
    faster on the GPU, while the token generator is 1.7x slower there.
    """

    config: AlgorithmConfig

    def decode(self, tokens: SpeechTokens, voice: VoiceProfile, *, seed: int) -> Mel:
        """Integrate the flow to a mel.

        ``seed`` is mandatory, not optional with a default. The prior is drawn
        from it, and an unseeded implementation of this stage produced waveforms
        correlating at 0.109 across two runs of *identical* tokens — larger than
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
