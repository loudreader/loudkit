"""The engine: four components and one algorithm, composed at load time.

Sequencing and seeding live in :mod:`loudkit.window`; the long-form pipeline
in :mod:`loudkit.stream`. What is left here is the public surface and the
one check the library enforces: every component computes the same algorithm.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from .config import AlgorithmConfig, ExecutionConfig
from .contracts import MelDecoder, SpeechTokens, TextFrontend, TokenGenerator, Vocoder
from .errors import CancelledError, InvalidTokensError, NothingToSpeakError
from .frontend.chunking import split_text
from .models.timestretch import validate_speed
from .result import Provenance, Result, StageTimings
from .timing import ChunkSpan, timeline
from .voice import VoiceProfile
from .window import (
    _STREAM_FLOW,
    _STREAM_VOCODER,
    GeneratedWindow,
    _derive,
    carry_from,
    generate_window,
    provenance_for,
    refuse_truncated_window,
    render_window,
    require_a_voice_profile,
    resolve_language,
    strip_specials,
    validate_speech_tokens,
    windows_for_chunk,
)

__all__ = ["Engine", "Provenance", "Result", "StageTimings", "validate_speech_tokens"]


@dataclass(frozen=True)
class Engine:
    """Text to speech.

    >>> engine = lk.load("loudreader/loudr-1")
    >>> engine.synthesize("Hello there.", engine.voice("joe"), seed=7).save("out.wav")

    Frozen so no component can be swapped after the algorithm check.
    """

    frontend: TextFrontend
    token_generator: TokenGenerator
    mel_decoder: MelDecoder
    vocoder: Vocoder
    algorithm: AlgorithmConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backend: str = ""
    """``torch``, ``onnx`` or ``coreml``. Provenance, not dispatch."""
    checkpoint_sha256: str = ""
    checkpoint_path: str = ""
    """Its directory is the release :meth:`voice` and :meth:`voices` read."""

    _wedged: str | None = field(default=None, init=False, repr=False, compare=False)
    """Why this engine stopped being usable. Set once, per instance, never cleared.

    Out of ``__init__`` but not out of the instance: ``repr`` and ``eq`` skip it,
    so two engines that differ only in whether one has wedged still compare equal
    and print the same line. A wedge is a runtime state, not part of identity.
    """

    def __post_init__(self) -> None:
        self._assert_one_algorithm()
        self._assert_the_rate_it_renders_at()

    def _assert_the_rate_it_renders_at(self) -> None:
        """The declared token rate must be the one this renderer produces.

        ``token_rate_hz`` is a manifest field. The geometry underneath it is
        not: a speech token becomes :data:`~loudkit.contracts.TOKEN_MEL_RATIO`
        mel frames and a mel frame becomes
        :data:`~loudkit.models.windowing.UPSAMPLE_PER_FRAME` samples, in every
        backend. A checkpoint free to declare one rate while the renderer
        produces another is a checkpoint whose duration claims do not describe
        its audio: at 12.5 Hz the 255-token window is announced as 20.4 s of
        speech and renders 10.2 s. The two numbers describe one thing, so they
        are required to agree here, where the renderer is.
        """
        from .contracts import TOKEN_MEL_RATIO
        from .models.windowing import UPSAMPLE_PER_FRAME

        per_token = TOKEN_MEL_RATIO * UPSAMPLE_PER_FRAME
        renders_at = self.algorithm.sample_rate / per_token
        if abs(renders_at - self.algorithm.token_rate_hz) > 1e-9:
            raise ValueError(
                f"token_rate_hz is {self.algorithm.token_rate_hz} but this renderer "
                f"produces {renders_at} Hz: one speech token is {TOKEN_MEL_RATIO} mel "
                f"frames of {UPSAMPLE_PER_FRAME} samples at "
                f"{self.algorithm.sample_rate} Hz. Every duration the engine states "
                "comes from the declared rate and every sample comes from the "
                "geometry, so the two disagreeing means a claim that does not "
                "describe the audio."
            )

    def _assert_one_algorithm(self) -> None:
        """Every component must carry the engine's exact algorithm, or it is a different
        engine."""
        want = self.algorithm.fingerprint()
        for label in ("token_generator", "mel_decoder", "vocoder"):
            cfg = getattr(getattr(self, label), "config", None)
            if cfg is None:
                raise ValueError(
                    f"{label} exposes no `config`, so its algorithm cannot be "
                    "checked against the engine's. Every TokenGenerator, "
                    "MelDecoder and Vocoder must carry the AlgorithmConfig it "
                    "was built with."
                )
            if cfg.fingerprint() != want:
                raise ValueError(
                    f"{label} was built with a different algorithm config "
                    f"({cfg.fingerprint()} != {want}).\n"
                    f"  engine:    {self.algorithm.describe()}\n"
                    f"  {label}: {cfg.describe()}\n"
                    "Algorithm values are shared, not per-component. If this is "
                    "deliberate, it is a different engine."
                )

    @property
    def wedged(self) -> str | None:
        """Why this engine can no longer render, or ``None`` while it can.

        A liveness probe reads this. An engine that has wedged answers every
        synthesis with a fault and cannot recover, so a probe that cannot see
        the state keeps reporting a server nothing can be routed to.
        """
        return self._wedged

    def _refuse_if_wedged(self) -> None:
        """A plain RuntimeError: the boundaries report it as the server's fault."""
        if self._wedged is not None:
            raise RuntimeError(self._wedged)

    def _wedge(self, reason: str) -> None:
        object.__setattr__(
            self,
            "_wedged",
            f"this engine is unusable: {reason}. Nothing can reclaim its stages "
            "from here - load a new engine.",
        )

    def describe(self) -> str:
        """One line naming both layers."""
        return f"{self.algorithm.describe()} | {self.execution.describe()}"

    # -- the release this engine came from ----------------------------------

    def voice(self, name: str) -> VoiceProfile:
        """A voice from this engine's own release, by name."""
        from .hub import resolve_voice

        return VoiceProfile.load(resolve_voice(name, repo=self._release()))

    def voices(self) -> tuple[str, ...]:
        """The names :meth:`voice` accepts, sorted."""
        from .hub import list_voices

        return list_voices(repo=self._release())

    def _release(self) -> str:
        if not self.checkpoint_path:
            # `FileNotFoundError` although no file is missing: the release this
            # would have named is what is absent, and callers already catch it
            # that way, pinned by tests/test_public_api.py. Narrowing it to a
            # `ValueError` is a 0.1.2 change.
            raise FileNotFoundError(
                "this engine was assembled from components, so it does not know "
                "which release it came from. Use loudkit.voice(name, repo=...)."
            )
        from pathlib import Path

        return str(Path(self.checkpoint_path).parent)

    # -- synthesis ----------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice: VoiceProfile,
        *,
        seed: int = 0,
        language: str | None = None,
        speed: float = 1.0,
        previous_tokens: SpeechTokens | None = None,
        single_window: bool = False,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Result:
        """Speak ``text`` in ``voice``, whatever its length.

        The whole passage, or :class:`~loudkit.errors.CancelledError` when
        ``should_cancel`` returned true: a passage cut short is never handed
        back as a result. See ``docs/design/engine-pipeline.md``.
        """
        self._refuse_if_wedged()
        require_a_voice_profile(voice)
        if not text.strip():
            raise NothingToSpeakError("nothing to speak")
        validate_speed(speed)
        if not single_window:
            from .stream import joined

            return joined(
                self,
                text,
                voice,
                seed=seed,
                language=language,
                speed=speed,
                previous_tokens=previous_tokens,
                should_cancel=should_cancel,
            )
        return self._synthesize_one(
            text,
            voice,
            seed=seed,
            language=resolve_language(language, voice),
            speed=speed,
            prefix=self._carry_from(previous_tokens),
            should_cancel=should_cancel,
            refuse_overflow=True,
        )

    def stream(
        self,
        text: str,
        voice: VoiceProfile,
        *,
        seed: int = 0,
        language: str | None = None,
        speed: float = 1.0,
        previous_tokens: SpeechTokens | None = None,
        should_cancel: Callable[[], bool] | None = None,
        latency_mode: bool = True,
    ) -> Iterator[Result]:
        """Yield one :class:`Result` per chunk as each becomes ready.

        When ``should_cancel`` returns true the stream ends: the chunks already
        yielded are the partial, the one in flight is discarded, and nothing is
        raised, since the caller flipped the flag. See
        ``docs/design/engine-pipeline.md``.
        """
        try:
            yield from self._stream(
                text,
                voice,
                seed=seed,
                language=language,
                speed=speed,
                previous_tokens=previous_tokens,
                should_cancel=should_cancel,
                latency_mode=latency_mode,
            )
        except CancelledError:
            return

    def _stream(
        self,
        text: str,
        voice: VoiceProfile,
        *,
        seed: int,
        language: str | None,
        speed: float,
        previous_tokens: SpeechTokens | None,
        should_cancel: Callable[[], bool] | None,
        latency_mode: bool,
    ) -> Iterator[Result]:
        """:meth:`stream`, raising :class:`CancelledError` where it stops."""
        self._refuse_if_wedged()
        language = resolve_language(language, voice)
        if not text.strip():
            raise NothingToSpeakError("nothing to speak")
        validate_speed(speed)
        # The funnel runs on the whole text before splitting, so the splitter
        # budgets the text that will be spoken.
        from .frontend.speechtext import speech_text

        prepared = speech_text(text, language)
        if not prepared.strip():
            raise NothingToSpeakError("nothing to speak")
        chunks = split_text(prepared, self.algorithm.chunking)
        carry = self._carry_from(previous_tokens)

        if len(chunks) == 1:
            windows = self._windows_for_chunk(
                chunks[0],
                voice,
                index=0,
                seed=seed,
                language=language,
                prefix=carry,
                is_terminal=True,
                should_cancel=should_cancel,
            )
            for window in windows:
                yield self._render_window(
                    window, voice, speed=speed, should_cancel=should_cancel
                )
            return

        from .stream import pipelined

        yield from pipelined(
            self,
            chunks,
            voice,
            seed=seed,
            language=language,
            speed=speed,
            carry=carry,
            prefix_len=self.algorithm.chunking.prefix_tokens,
            should_cancel=should_cancel,
            latency_mode=latency_mode,
        )

    def synthesize_tokens(
        self, tokens: SpeechTokens, voice: VoiceProfile, *, seed: int = 0
    ) -> Result:
        """Render tokens that already exist: the diagnostic that takes the sampler out of
        a comparison.

        The diagnostic path renders the vocoder's own bytes: no edge fade, no
        time stretch. Both belong to the window renderer, and a comparison that
        wants to hear the model wants them out of the way. So this is the one
        rendered audio the identity contract's "every rendered window" does not
        cover. Swift's ``synthesizeTokens`` makes the same choice, so the two
        agree; routing it through the window renderer is a 0.1.2 change,
        because it would move these bytes.
        """
        import time

        self._refuse_if_wedged()
        if len(tokens) == 0:
            raise InvalidTokensError(
                "tokens is empty: there is nothing to render.",
                token=0,
                limit=self.algorithm.start_speech_token,
            )
        validate_speech_tokens(tokens, limit=self.algorithm.start_speech_token, field="tokens")
        speech = strip_specials(self.algorithm, tokens)
        t0 = time.perf_counter()
        mel = self.mel_decoder.decode(speech, voice, seed=_derive(seed, _STREAM_FLOW))
        t1 = time.perf_counter()
        audio = self.vocoder.synthesize(mel, voice, seed=_derive(seed, _STREAM_VOCODER))
        t2 = time.perf_counter()
        return Result(
            audio=audio,
            tokens=speech,
            mel=mel,
            seed=seed,
            sample_rate=self.algorithm.sample_rate,
            timings=StageTimings(0.0, t1 - t0, t2 - t1),
            chunks=timeline(
                [ChunkSpan(text="", samples=len(audio), tokens=len(speech))],
                sample_rate=self.algorithm.sample_rate,
            ),
            provenance=provenance_for(self, voice, ""),
        )

    def warm(self, voice: VoiceProfile) -> None:
        """Pay the first-use costs (kernel autotuning, graph capture) now.

        Every seed in the pipeline is derived from the caller's own, so this
        render is invisible to every later one. See
        ``docs/design/engine-pipeline.md``. The transports reach it through
        :func:`loudkit.synthesis.warm_engine`, which is where the environment
        override and the log line live.
        """
        self._refuse_if_wedged()
        self._synthesize_one("Ready.", voice, seed=0, language=resolve_language(None, voice))

    # -- one window; the seams the pipeline and the tests reach ---------------

    def _synthesize_one(
        self,
        text: str,
        voice: VoiceProfile,
        *,
        seed: int,
        language: str,
        max_new_tokens: int | None = None,
        prefix: SpeechTokens = (),
        prepared: bool = False,
        is_terminal: bool = True,
        speed: float = 1.0,
        should_cancel: Callable[[], bool] | None = None,
        refuse_overflow: bool = False,
    ) -> Result:
        """Both phases of one window. The overflow refusal sits between them,
        before the expensive half."""
        window = self._generate_window(
            text,
            voice,
            seed=seed,
            language=language,
            max_new_tokens=max_new_tokens,
            prefix=prefix,
            prepared=prepared,
            is_terminal=is_terminal,
            should_cancel=should_cancel,
        )
        if refuse_overflow:
            refuse_truncated_window(self.algorithm, window.n_generated, window.hit_window_cap)
        return self._render_window(window, voice, speed=speed, should_cancel=should_cancel)

    def _generate_window(
        self,
        text: str,
        voice: VoiceProfile,
        *,
        seed: int,
        language: str,
        max_new_tokens: int | None = None,
        prefix: SpeechTokens = (),
        prepared: bool = False,
        is_terminal: bool = True,
        should_cancel: Callable[[], bool] | None = None,
    ) -> GeneratedWindow:
        return generate_window(
            self,
            text,
            voice,
            seed=seed,
            language=language,
            max_new_tokens=max_new_tokens,
            prefix=prefix,
            prepared=prepared,
            is_terminal=is_terminal,
            should_cancel=should_cancel,
        )

    def _render_window(
        self,
        window: GeneratedWindow,
        voice: VoiceProfile,
        *,
        speed: float,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Result:
        return render_window(self, window, voice, speed=speed, should_cancel=should_cancel)

    def _windows_for_chunk(
        self,
        chunk: str,
        voice: VoiceProfile,
        *,
        index: int,
        seed: int,
        language: str,
        prefix: SpeechTokens,
        is_terminal: bool,
        should_cancel: Callable[[], bool] | None,
    ) -> list[GeneratedWindow]:
        return windows_for_chunk(
            self,
            chunk,
            voice,
            index=index,
            seed=seed,
            language=language,
            prefix=prefix,
            is_terminal=is_terminal,
            should_cancel=should_cancel,
        )

    def _carry_from(self, previous_tokens: Sequence[int] | None) -> list[int]:
        return carry_from(self.algorithm, previous_tokens)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        device: str | None = None,
        execution: ExecutionConfig | None = None,
        algorithm: AlgorithmConfig | None = None,
    ) -> Engine:
        """Build an engine from a packed checkpoint. The manifest is the authority on the
        algorithm.

        ``execution`` names the fields to change; the rest come from the
        checkpoint's defaults for ``device``.
        """
        from .backends import build_engine

        return build_engine(path, device=device, execution=execution, algorithm=algorithm)
