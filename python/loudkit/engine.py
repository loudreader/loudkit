"""The engine: five components, one line of data, no hidden state.

:class:`Engine` composes a text frontend, a token generator, a mel decoder and a
vocoder into text-to-speech, and does almost nothing else. That is intentional —
the interesting decisions live in :class:`~loudkit.config.AlgorithmConfig`, and
the interesting speed lives in the backends. What is left here is sequencing,
seeding, and refusing to run a mismatched configuration.

Two behaviours are worth knowing about before reading the code.

**Stages may live on different devices.** ``Engine`` never assumes its
components share hardware, because on Apple silicon they should not: the token
generator is 1.7x *slower* on the GPU than on the CPU (an autoregressive step at
batch one is a few hundred tiny dispatches), while the mel decoder is 2.6x
faster there (one large parallel pass). Splitting them measured 1.35x -> 1.54x.

**Seeds are derived, not shared.** Each stage draws from its own sub-stream of
one counter-based generator, so the sampler, the flow prior and the vocoder
excitation can never collide, and a change in how many numbers one stage
consumes cannot shift another stage's stream.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np

from .config import AlgorithmConfig, ExecutionConfig, ExecutionOverrides
from .contracts import (
    Mel,
    MelDecoder,
    SpeechTokens,
    TextFrontend,
    TokenGenerator,
    Vocoder,
    Waveform,
)
from .errors import InvalidTokensError, NothingToSpeakError, WindowOverflowError
from .frontend.chunking import split_text
from .models.timestretch import time_stretch, validate_speed
from .models.windowing import eos_floor
from .postprocess import Inspection, ceiling_for, inspect
from .sampler import LRSamplerV1
from .timing import ChunkSpan, ChunkTiming, timeline
from .voice import VoiceProfile

__all__ = ["Engine", "Result", "StageTimings", "validate_speech_tokens"]

_LOG = logging.getLogger("loudkit.engine")

_FALLBACK_LANGUAGE = "en"
"""What a synthesis reads as when neither the caller nor the voice says.

Reached less often than it looks. Every loader defaults a *missing* header key
to ``"en"``, and Python has always written the key — so the empty string only
arrives from a profile built in memory without one, or a header hand-edited to
``""``.

Worth stating plainly, because it bounds what the chain can do for anyone's
existing files: a profile written before this field was read back loads as
``"en"``, not as blank, so it inherits **nothing**. A non-English voice from an
older writer still needs an explicit ``language`` or a re-save.
"""

_STREAM_FLOW = 1
_STREAM_VOCODER = 2
_PIPELINE_DEPTH = 2
"""How many rendered-or-rendering windows :meth:`Engine.stream` holds in flight.

The producer generates ahead of the consumer by at most this many windows.
Two is enough to keep the renderer busy while the next window generates; more
buys nothing but memory and wasted work when a consumer walks away.
"""

_PRODUCER_JOIN_TIMEOUT = 60.0
"""How long a closing :meth:`Engine.stream` waits for its producer thread.

Generous because the wait is for the render already in flight, not for the
passage: a window that has not finished rendering in a minute is stuck, not
slow. Exceeding it is terminal for the engine rather than swallowed: the thread
still holds the stages, so the engine refuses every later call — see the join
below.
"""

_STREAM_RETRY = 8  # retry attempts draw derive(seed, 8 + attempt); clear of the
# stage streams above and below the chunk streams at 16
_STREAM_CHUNK = 16  # chunk seeds start here, clear of the stage streams


@dataclass(frozen=True, slots=True)
class StageTimings:
    """Wall time per stage, in seconds."""

    tokens: float
    mel: float
    audio: float

    @property
    def total(self) -> float:
        return self.tokens + self.mel + self.audio

    def rtf(self, audio_seconds: float) -> float:
        """Real-time factor: seconds of audio produced per second of work."""
        return audio_seconds / self.total if self.total > 0 else float("inf")

    def describe(self, audio_seconds: float) -> str:
        return (
            f"tokens {self.tokens:.3f}s  mel {self.mel:.3f}s  audio {self.audio:.3f}s  "
            f"-> {audio_seconds:.2f}s @ RTF {self.rtf(audio_seconds):.2f}x"
        )


@dataclass(frozen=True, slots=True)
class Result:
    """A synthesis, with everything needed to explain or reproduce it.

    The intermediates are kept rather than discarded because they are how two
    backends get compared when they disagree: tokens localise a difference to
    the first stage, the mel to the second, the waveform to the third. Without
    them a mismatch is one number and no diagnosis.
    """

    audio: Waveform
    tokens: SpeechTokens
    mel: Mel
    seed: int
    sample_rate: int
    timings: StageTimings
    algorithm_fingerprint: str
    hit_token_cap: bool = False
    """True if generation stopped at the cap rather than at a stop token —
    usually a sign of a broken EOS path, and always worth surfacing."""

    inspections: tuple[Inspection, ...] = ()
    """What the postprocess detectors concluded, one entry per chunk.

    A tuple rather than a single verdict because a passage is many chunks and
    they fail independently: one hallucinated tail in the middle of six clean
    ones is the case worth seeing, and an aggregate would hide it.
    """

    speed: float = 1.0
    """The time-stretch this render was asked for. ``1.0`` means none was
    applied — the waveform came straight out of the vocoder.

    Recorded rather than inferred, because it cannot be inferred: a stretched
    reading and a naturally faster one are the same numbers afterwards, and
    ``duration`` alone cannot tell a caller which it is holding.
    """

    chunks: tuple[ChunkTiming, ...] = ()
    """Where each chunk lands in ``audio``, and where its words probably do.

    One entry per chunk, in order and adjacent: chunk *k*'s ``end`` is the same
    float as chunk *k+1*'s ``start``, and the last ``end`` is
    :attr:`duration`. A single-window synthesis gets one entry covering the
    whole result.

    Chunk boundaries are exact — they are sample offsets, which the engine
    already knows because it concatenated the chunks. The per-word times inside
    each entry are an **estimate**; see :mod:`loudkit.timing` before building
    anything that depends on them.

    Measured on the returned waveform, so they already account for
    :attr:`speed`.
    """

    recipe_version: str = ""
    """The algorithm recipe this render was computed under.

    Carried so ``save`` can put it in the provenance manifest alongside the
    fingerprint: the fingerprint plus the seed reproduce the waveform, and the
    recipe names which code did it.
    """

    voice_name: str = ""
    """The name of the profile that voiced this render, for the manifest."""

    language: str = ""
    """The resolved language used by the text frontend, for the manifest.

    Empty only for :meth:`Engine.synthesize_tokens`, where no text frontend ran
    and therefore no language was selected.
    """

    voice_sha256: str = ""
    """SHA-256 of the profile file that voiced this render, or ``""`` when the
    profile never touched disk. A name is a label anyone can reuse; the digest
    names the bytes."""

    checkpoint_sha256: str = ""
    """SHA-256 of the checkpoint file this engine was built from — the value a
    release's ``SHA256SUMS`` lists. The algorithm fingerprint says *what* was
    computed; this says *with which weights*."""

    backend: str = ""
    """Which renderer produced this: ``torch``, ``onnx`` or ``coreml``."""

    execution: str = ""
    """The execution layer, described: device placement and per-module
    precision. Execution never changes what is computed, but fp16 does perturb
    it within measured bands, so a manifest that names the waveform should name
    the datapath too."""

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def suspect(self) -> bool:
        """Any chunk was impossibly long for its text and no rule could say
        where to cut. Nothing was removed; you are being told."""
        return any(i.suspect for i in self.inspections)

    def save(
        self,
        path: str,
        *,
        voice: str = "",
        language: str = "",
        include_provenance: bool = True,
    ) -> None:
        """Write a WAV, with C2PA claim-only provenance by default.

        The manifest (a JUMBF ``c2pa`` box trailing the audio) carries the
        algorithm fingerprint, the recipe, the seed, and the SHA-256 of the
        audio it binds to — the machine-readable marking Article 50 asks for,
        at the metadata layer the field ships (see :mod:`loudkit.provenance`).
        ``voice`` and ``language`` default to what the result already knows;
        pass them to override the labels. Requires ``soundfile``.
        """
        if not include_provenance:
            import soundfile as sf

            sf.write(path, self.audio, self.sample_rate)
            return

        from .provenance import write_wav

        try:
            from importlib.metadata import version as _pkg_version

            version = _pkg_version("loudkit")
        except Exception:  # pragma: no cover - metadata present in any install
            version = "0.1.0"
        write_wav(
            path,
            self.audio,
            self.sample_rate,
            algorithm_fingerprint=self.algorithm_fingerprint,
            recipe_version=self.recipe_version,
            seed=self.seed,
            voice=voice or self.voice_name,
            language=language or self.language,
            text=" ".join(c.text for c in self.chunks),
            speed=self.speed,
            version=version,
            voice_sha256=self.voice_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            backend=self.backend,
            execution=self.execution,
        )

    def __repr__(self) -> str:
        cap = ", HIT CAP" if self.hit_token_cap else ""
        trimmed = sorted({i.reason for i in self.inspections if i.cut})
        cut = f", cut={'+'.join(trimmed)}" if trimmed else ""
        flag = ", SUSPECT" if self.suspect else ""
        # Only when it is not the default: a stretched reading and a naturally
        # fast one print the same duration, and this is the only thing that
        # tells them apart in a log.
        rate = f", speed={self.speed:g}x" if self.speed != 1.0 else ""
        return (
            f"Result({self.duration:.2f}s, {len(self.tokens)} tokens, seed={self.seed}, "
            f"RTF {self.timings.rtf(self.duration):.2f}x{cap}{rate}{cut}{flag})"
        )


@dataclass(frozen=True, slots=True)
class _GeneratedWindow:
    """A window after the token phase, before the render phase.

    Everything :meth:`Engine._render_window` needs and nothing more. Kept
    deliberately thin: whatever crosses here is what the pipeline in
    :meth:`Engine.stream` holds in flight.
    """

    text: str
    language: str
    speech: list[int]
    seed: int
    verdict: Inspection
    hit_token_cap: bool
    generate_seconds: float


@dataclass(frozen=True)
class Engine:
    """Text to speech, composed from four components and one config.

    Example:
        >>> engine = Engine.from_checkpoint("loudr-1.safetensors", device="cpu")
        >>> voice = VoiceProfile.load("voices/james.safetensors")
        >>> result = engine.synthesize("Hello there.", voice, seed=7)
        >>> result.save("out.wav")

    Frozen on purpose. The fingerprint check runs once at construction, so a
    mutable engine would let a caller swap in a component with a different
    algorithm afterwards and defeat the only enforcement this library has.

    Args:
        algorithm: what is computed. Shared by every component; a component that
            disagrees is rejected at construction.
        execution: how fast. Differs per backend and is not checked for
            agreement, because differing is its purpose.
    """

    frontend: TextFrontend
    token_generator: TokenGenerator
    mel_decoder: MelDecoder
    vocoder: Vocoder
    algorithm: AlgorithmConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    backend: str = ""
    """Which backend built this engine: ``torch``, ``onnx`` or ``coreml``.

    Provenance, not dispatch — nothing branches on it. Carried into every
    :class:`Result` and from there into the C2PA manifest.
    """

    checkpoint_sha256: str = ""
    """SHA-256 of the checkpoint file the components were loaded from.

    The value a release's ``SHA256SUMS`` lists. The algorithm fingerprint
    already pins *what* is computed; two different checkpoints can share a
    fingerprint, so a manifest that stopped there could not say which weights
    spoke. Empty when the engine was assembled from components by hand.
    """

    _wedged: ClassVar[str | None] = None
    """Why this engine stopped being usable, or ``None`` while it is.

    Not a dataclass field: it is a fact about one engine's fate, not about what
    it computes, and two engines built from the same components are still equal
    when one of them is wedged. Written once per instance through
    ``object.__setattr__`` — the frozen guard exists to stop a component being
    swapped after the fingerprint check, and this changes no component.

    Set by :meth:`_stream_pipelined` when a producer thread outlives the join,
    and by a transport that cannot reclaim a stream it abandoned. Never
    cleared: the thread is still inside the stages, and nothing here can prove
    it has left.
    """

    def __post_init__(self) -> None:
        self._assert_one_algorithm()

    def _refuse_if_wedged(self) -> None:
        """Refuse the call when a previous render never let go.

        Every public synthesis entry calls this first. A stuck producer holds
        the token generator and the renderer, so the next call does not fail —
        it contends, silently, with a thread nobody is waiting on, and the
        result is a slow request or a wrong one depending on how much state the
        two share. Naming the cause once is worth more than either.

        A plain ``RuntimeError`` and not a
        :class:`~loudkit.errors.LoudkitError`: the boundaries classify every
        ``LoudkitError`` as the caller's fault, and this is the server's. It
        reaches a client as ``server_fault``, which is exactly what it is.
        """
        if self._wedged is not None:
            raise RuntimeError(self._wedged)

    def _wedge(self, reason: str) -> None:
        """Mark this engine unusable, naming what still holds it.

        ``object.__setattr__`` because the dataclass is frozen. The freeze
        guards the *components* — swapping one after the fingerprint check is
        what it exists to stop — and this swaps none of them.

        ``reason`` is a phrase rather than a thread name because the producer's
        join is not the only place that can fail to prove the stages are free:
        a transport reclaiming a stream it abandoned reaches the same verdict by
        a different route, and the caller who reads the message needs the route.
        """
        object.__setattr__(
            self,
            "_wedged",
            f"this engine is unusable: {reason}. Nothing can reclaim its stages "
            "from here — load a new engine.",
        )

    def _assert_one_algorithm(self) -> None:
        """Refuse to run components that disagree about what to compute.

        Algorithm values decide what is computed, so every component must carry
        the engine's exact config. Two configurations can both produce
        plausible audio — dual-path guidance applied to an estimator distilled
        for single-path use, for instance — so output comparison alone cannot
        detect a mismatch.
        """
        want = self.algorithm.fingerprint()
        # The vocoder is checked too. It looks like a pure renderer, but it reads
        # `config.window` for padding geometry, which is algorithm-bearing: a
        # vocoder framing the tail differently produces a different reading.
        for label, component in (
            ("token_generator", self.token_generator),
            ("mel_decoder", self.mel_decoder),
            ("vocoder", self.vocoder),
        ):
            cfg = getattr(component, "config", None)
            if cfg is None:
                # Fail closed. The protocols declare `config` on all three of
                # these, so a component without one is either not implementing
                # the protocol or has been swapped for something that only
                # looks like it — and skipping the check is exactly the shape
                # of hole this method exists to close: an unchecked component
                # can compute anything at all while every reported fingerprint
                # agrees. (`TextFrontend` genuinely carries no config and is
                # not in this list; it is bound to the checkpoint by digest
                # instead — see `Checkpoint.verified_sibling`.)
                raise ValueError(
                    f"{label} exposes no `config`, so its algorithm cannot be "
                    "checked against the engine's. Every TokenGenerator, "
                    "MelDecoder and Vocoder must carry the AlgorithmConfig it "
                    "was built with."
                )
            got = cfg.fingerprint()
            if got != want:
                raise ValueError(
                    f"{label} was built with a different algorithm config "
                    f"({got} != {want}).\n"
                    f"  engine:    {self.algorithm.describe()}\n"
                    f"  {label}: {cfg.describe()}\n"
                    "Algorithm values are shared, not per-component. If this is "
                    "deliberate, it is a different engine."
                )

    def describe(self) -> str:
        """One line naming both layers. Log it on every run."""
        return f"{self.algorithm.describe()} | {self.execution.describe()}"

    # -- synthesis ----------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice: VoiceProfile,
        *,
        seed: int = 0,
        language: str | None = None,
        max_new_tokens: int | None = None,
        speed: float = 1.0,
        previous_tokens: SpeechTokens | None = None,
    ) -> Result:
        """Speak ``text`` in ``voice``.

        Args:
            text: what to say. Normalisation is the frontend's business.
            voice: who says it.
            seed: same seed and same build give a bit-identical waveform. Not
                guaranteed across backends or releases — see the identity
                contract.

                Tokens are more portable than audio, because the sampler is
                hardware-agnostic by construction — but *only at matched
                precision*. fp16 in the generator moves a logit enough to flip
                roughly one token in a thousand, and one flip re-routes every
                token after it, so two backends running different precisions
                will produce different readings. Same precision, same tokens;
                different precision, same law and a different sample.
            language: language id for the text frontend. ``None`` — the default
                — takes ``voice.language``, and falls back to ``"en"`` only for
                a profile that carries none. Pass one explicitly to read text in
                a language the voice was not enrolled in; that is what
                cross-lingual synthesis is, and the argument always wins.
            max_new_tokens: override the configured cap. Rarely useful; the cap
                exists to bound a runaway EOS, not to shape output.
            speed: playback speed, in ``[0.5, 2.0]``. Greater than one is
                faster; pitch is preserved. ``1.0`` — the default — is an exact
                bypass: the waveform is the vocoder's own bytes, untouched.

                Applied last, after the postprocess detectors have inspected the
                render, because those detectors measure pacing against the text
                (duration per token) and a stretch applied first would move every
                measurement they make. It is a change to the *delivery*, not to
                the reading.

                Not part of the algorithm config and not in the fingerprint: it
                is an execution input like ``seed`` and ``text``, and two engines
                that disagree about it are still computing the same thing.
            previous_tokens: speech tokens this utterance continues from —
                :attr:`Result.tokens` of the call before it. The first (and only)
                window is then conditioned on their tail exactly as an interior
                chunk is conditioned on its predecessor, which is what stops a
                second request from restarting the pitch contour like a fresh
                sentence.

                Only the last ``chunking.prefix_tokens`` are used, so passing a
                whole previous result is the intended usage and costs nothing —
                the slice happens here rather than at the call site.

                Also an execution input: ``None`` is byte-for-byte today's
                behaviour, and the fingerprint does not move.

        Note:
            The seed is applied to the single chunk *directly*. Streaming and
            long-form synthesis derive a per-chunk seed from it, so even for a
            text that fits one window ``synthesize(t, seed=7)`` and
            ``synthesize_long(t, seed=7)`` produce different audio (the
            single-chunk seed is the raw one, the streamed chunk seeds are
            ``derive(7, chunk)``). Same seed, same *path* — same bytes.

        Returns:
            A :class:`Result` carrying the audio and every intermediate.

        Example:
            Chaining two calls so the second continues the first::

                first = engine.synthesize("Part one.", voice, seed=7)
                second = engine.synthesize(
                    "Part two.", voice, seed=8, previous_tokens=first.tokens
                )
        """
        # All refused here, before the six seconds of generation they would
        # otherwise be discovered after.
        #
        # All three entry points refuse empty text identically: generating
        # against an empty prompt yields near-silence with no error, the one
        # failure a caller is least likely to notice.
        self._refuse_if_wedged()
        if not text.strip():
            raise NothingToSpeakError("nothing to speak")
        validate_speed(speed)
        result = self._synthesize_one(
            text,
            voice,
            seed=seed,
            language=_resolve_language(language, voice),
            max_new_tokens=max_new_tokens,
            speed=speed,
            prefix=self._carry_from(previous_tokens),
        )
        if result is None:  # pragma: no cover - unreachable without should_cancel
            # `assert` here vanished under `python -O`, and this returns
            # `Result` rather than `Result | None`: the invariant is part of the
            # signature, so breaking it has to raise rather than hand a caller a
            # `None` the type says cannot arrive.
            raise RuntimeError(
                "synthesis returned nothing without a cancellation callback — "
                "this is a bug in loudkit"
            )
        return result

    def warm(self, voice: VoiceProfile) -> None:
        """Pay the first-use costs now, so the first request does not.

        The first synthesis on a device is the slowest one it will ever run:
        cuDNN autotunes its kernels, cuFFT builds plans, the CUDA-graph decode
        step is captured, allocator pools fill. Measured on a 3090 with
        graphs, first audio is 1.09 s cold against 0.60 s warm; on an Orin,
        3.5 s against 2.4 s. None of that belongs on a user's first request —
        a long-running process calls this once at startup and the cost moves
        to load time, where a listener is not waiting on it.

        Renders a short throwaway sentence end-to-end in ``voice`` — every
        stage must run, or its first-use cost survives — and discards the
        result. Purely execution-layer: caches and pools change, output bytes
        never do. Idempotent and cheap when already warm.
        """
        self._refuse_if_wedged()
        self._synthesize_one(
            "Ready.",
            voice,
            seed=0,
            language=_resolve_language(None, voice),
        )

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
    ) -> _GeneratedWindow | None:
        """One window's token phase: funnel, generate, judge, retry, trim.

        The token phase; :meth:`_render_window` is the render phase. Split so
        :meth:`stream` can render window *k* while
        generating window *k+1* — the two phases share nothing but the tokens
        that cross between them, which is also why the split cannot change a
        byte of output.

        Returns ``None`` when ``should_cancel`` fired during generation. The
        partial tokens are discarded rather than rendered: the listener has
        already interrupted, so the mel decoder and the vocoder would be
        producing audio no one will hear — and on a slow device that render is
        the larger half of the barge-in latency the caller is trying to avoid.
        Only a caller that passes ``should_cancel`` can receive ``None``.
        """
        # An override is a number of tokens: zero is not (it would fall
        # through to the configured cap, yielding a full utterance for a call
        # that asked for none), and a negative would reach the sampler
        # unexamined.
        if max_new_tokens is not None and max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1: {max_new_tokens}")
        cap = (
            max_new_tokens
            if max_new_tokens is not None
            else self.algorithm.sampling.max_new_tokens
        )
        # The speech funnel the shipped Swift engine runs before tokenising
        # (SpeechText.prepared): scrub invisibles/symbols/footnotes/punctuation,
        # and for Polish respell embedded English. Applied here, on the one
        # path that renders, so single-shot and streaming cannot drift apart.
        # ``prepared`` skips the funnel when the caller has already run it on
        # the whole text before splitting (the streaming path, which must budget
        # the post-funnel length).
        if not prepared:
            from .frontend.polish import speech_text

            text = speech_text(text, language)

        # Refused *after* the funnel, not only before it.
        #
        # The emptiness check at the public entry runs on what the caller
        # passed, and the funnel is entitled to remove everything: `[12]` is a
        # footnote marker, `💩` and `©®™` are symbols with no word, and each
        # comes out as "". The tokeniser then encoded nothing, the sampler was
        # asked for speech about it, and the caller got a `Result` with audio in
        # it — a render of a sentence that does not exist.
        #
        # Here rather than at the three entry points, because this is the one
        # place all three pass through: `synthesize`, `stream` and
        # `synthesize_long` cannot disagree about it from here.
        if not text.strip():
            raise NothingToSpeakError(
                "nothing to speak: the text funnel removed every character. "
                "Footnote markers, emoji and symbols with no word in the "
                "render language are dropped, and this input was only those."
            )

        text_tokens = self.frontend.encode(text, language)

        pp = self.algorithm.postprocess
        floor = eos_floor(len(text_tokens), self.algorithm)
        if pp.mode != "off":
            # The length ceiling is applied *during* generation, not after it:
            # the tokens past it cost real time on a device and are certain to
            # be discarded. It only ever stops a row that was going to run away
            # — a model that stops on its own never reaches it.
            cap = min(
                cap,
                ceiling_for(
                    len(text_tokens),
                    config=pp,
                    window=self.algorithm.window.max_speech_tokens,
                ),
            )

        # Selective re-roll: a window whose verdict is unfixable — dropout
        # (content missing) or suspect (certainly wrong, nowhere to cut) — is
        # regenerated from a derived seed, up to `retry_max_attempts` times.
        # Only condemned windows pay; a clean render costs exactly one pass.
        # The ladder is a pure function of the caller's seed, so the same seed
        # still gives the same audio, retries included.
        attempt = 0
        tokens_elapsed = 0.0
        while True:
            attempt_seed = seed if attempt == 0 else _derive(seed, _STREAM_RETRY + attempt)
            sampler = LRSamplerV1(
                self.algorithm.sampling,
                seed=attempt_seed,
                stop_token=self.algorithm.stop_speech_token if pp.mode != "off" else None,
                eos_floor=floor,
            )

            t0 = time.perf_counter()
            tokens = self.token_generator.generate(
                text_tokens,
                voice,
                sampler=sampler,
                max_new_tokens=cap,
                prefix=prefix,
                should_cancel=should_cancel,
            )
            t1 = time.perf_counter()
            # Condemned attempts count too: a retry that burned 8 s of decode
            # is wall time the caller waited, and hiding it made RTF flatter
            # than the run was.
            tokens_elapsed += t1 - t0

            if should_cancel is not None and should_cancel():
                return None

            # `gen` is what the shipped engine calls a row: every token the
            # model committed to, with the stop marker itself excluded. Indices
            # into it are decode-step indices, which is what makes
            # `eos_peak_at` comparable against it — so the detectors run here,
            # before `_strip_specials` is free to renumber anything.
            stop = self.algorithm.stop_speech_token
            gen = list(tokens)
            ended = bool(gen) and gen[-1] == stop
            if ended:
                gen.pop()

            peak_at, peak_prob = sampler.eos_peak
            hit_cap = not ended and len(gen) >= cap
            verdict = inspect(
                gen,
                text_token_count=len(text_tokens),
                min_tokens=floor,
                eos_peak_at=peak_at,
                eos_peak_prob=peak_prob,
                ended=ended,
                is_terminal=is_terminal,
                hit_ceiling=hit_cap,
                silence=self.algorithm.sampling.silence_token_ids,
                config=pp,
            )
            condemned = verdict.reason == "dropout" or verdict.suspect
            if not condemned or pp.mode == "off" or attempt >= pp.retry_max_attempts:
                break
            attempt += 1

        if pp.mode == "trim" and verdict.keep < len(gen):
            gen = gen[: verdict.keep]

        speech = self._strip_specials(gen)
        if len(speech) == 0:
            # The EOS floor is opt-in (`eos_floor` returns 0 by default), so a
            # sampler is allowed to accept the stop token at step zero. Every
            # retry then produces the same nothing, and what came back was a
            # valid `Result` holding silence — a caller asked for speech and got
            # a blank answer with no error to distinguish it from a very quiet
            # one. Raised where it is known rather than discovered downstream.
            raise ValueError(
                "generation produced no speech tokens: the stop token was "
                "accepted immediately. Set sampling.min_tokens_floor above 0 to "
                "refuse that during sampling."
            )
        return _GeneratedWindow(
            text=text,
            language=language,
            speech=speech,
            seed=seed,
            verdict=verdict,
            hit_token_cap=hit_cap,
            generate_seconds=tokens_elapsed,
        )

    def _render_window(
        self,
        window: _GeneratedWindow,
        voice: VoiceProfile,
        *,
        speed: float,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Result | None:
        """One window's render phase: mel, vocoder, stretch, `Result`.

        Deterministic given the window: the render seeds derive from the
        window's own seed, so rendering on the caller's thread and rendering
        on :meth:`stream`'s worker produce the same bytes.

        ``should_cancel`` is polled between stages, the same discipline the
        token phase applies per decode step. A kernel already executing cannot
        be interrupted; what must stop is *starting the next stage*. A deadline
        that expires inside the mel decoder must not buy the caller a vocoder
        pass on an RPC that has already ended. Returns ``None`` when it fires,
        and only a caller that passes ``should_cancel`` can receive ``None``.
        """
        speech, seed = window.speech, window.seed
        if should_cancel is not None and should_cancel():
            return None
        t1 = time.perf_counter()
        mel = self.mel_decoder.decode(speech, voice, seed=_derive(seed, _STREAM_FLOW))
        t2 = time.perf_counter()

        if should_cancel is not None and should_cancel():
            return None
        audio = self.vocoder.synthesize(mel, voice, seed=_derive(seed, _STREAM_VOCODER))
        t3 = time.perf_counter()

        # Last, and after `inspect` in the token phase rather than before it:
        # the detectors judge pacing by duration per token, and stretching
        # first would move every number they compare against. `speed=1.0`
        # returns the vocoder's array itself, so the default costs nothing and
        # changes no byte.
        audio = time_stretch(audio, sample_rate=self.algorithm.sample_rate, speed=speed)

        return Result(
            audio=audio,
            tokens=speech,
            mel=mel,
            seed=seed,
            sample_rate=self.algorithm.sample_rate,
            timings=StageTimings(window.generate_seconds, t2 - t1, t3 - t2),
            algorithm_fingerprint=self.algorithm.fingerprint(),
            recipe_version=self.algorithm.recipe_version,
            voice_name=voice.name,
            language=window.language,
            voice_sha256=voice.source_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            backend=self.backend,
            execution=self.execution.describe(),
            hit_token_cap=window.hit_token_cap,
            inspections=(window.verdict,),
            speed=speed,
            # One window is one chunk, and it starts at zero: a streamed chunk
            # is its own Result and cannot know what preceded it. `stream`'s
            # caller — or `synthesize_long` — adds the offsets.
            chunks=timeline(
                [ChunkSpan(text=window.text, samples=len(audio), tokens=len(speech))],
                sample_rate=self.algorithm.sample_rate,
            ),
        )

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
    ) -> Result | None:
        """One window, both phases. The only path that renders, so single-shot
        and streaming cannot drift apart."""
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
        if window is None:
            return None
        return self._render_window(window, voice, speed=speed, should_cancel=should_cancel)

    def synthesize_tokens(
        self, tokens: SpeechTokens, voice: VoiceProfile, *, seed: int = 0
    ) -> Result:
        """Render a token sequence that already exists.

        The single most useful diagnostic in the library. Two backends that
        disagree on free-running output may disagree because they *sampled*
        differently, which is chaos rather than damage; feeding both the same
        tokens removes the first stage from the comparison and asks only whether
        the renderer agrees. That is how the Neural Engine's deviation was
        traced to a padding recipe rather than to the hardware.
        """
        # The same check the streaming route runs on `previous_tokens`, which
        # this path never ran on its own. `_strip_specials` drops ids at or
        # above `start_speech_token` and says nothing about the bottom, so `-1`
        # went straight through to an embedding lookup — negative indices work
        # in torch and read from the end of the table, and are an out-of-bounds
        # read on the ONNX path. A public render entry has to bound what it is
        # handed, not assume a caller derived it from a previous render.
        self._refuse_if_wedged()
        if len(tokens) == 0:
            # An empty sequence rendered audio: the decoder was asked for
            # speech about nothing and produced a Result the caller could save.
            # Silence is a thing a caller can ask for; a render of nothing is
            # not, and returning one hides the mistake that produced it.
            raise InvalidTokensError(
                "tokens is empty: there is nothing to render.",
                token=0,
                limit=self.algorithm.start_speech_token,
            )
        validate_speech_tokens(tokens, limit=self.algorithm.start_speech_token, field="tokens")
        speech = self._strip_specials(tokens)
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
            algorithm_fingerprint=self.algorithm.fingerprint(),
            recipe_version=self.algorithm.recipe_version,
            voice_name=voice.name,
            voice_sha256=voice.source_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            backend=self.backend,
            execution=self.execution.describe(),
            # No text reached this path, so there are no words to estimate — but
            # the span still covers the whole render, so a caller stitching
            # results does not have to special-case it.
            chunks=timeline(
                [ChunkSpan(text="", samples=len(audio), tokens=len(speech))],
                sample_rate=self.algorithm.sample_rate,
            ),
        )

    def synthesize_long(
        self,
        text: str,
        voice: VoiceProfile,
        *,
        seed: int = 0,
        language: str | None = None,
        speed: float = 1.0,
        previous_tokens: SpeechTokens | None = None,
    ) -> Result:
        """Speak a passage, splitting it across windows and joining the audio.

        Equivalent to concatenating :meth:`stream`. Use this when you want one
        waveform; use :meth:`stream` when you want to start playing before the
        passage is finished.

        **Bounded by memory, not by the splitter.** The whole stream is drained
        before anything is joined, so peak memory is the returned audio and mel
        plus every chunk they were built from — roughly twice the passage, at
        the moment it is largest. A minute of speech is a few hundred
        megabytes; a book is not. For content long enough to matter, consume
        :meth:`stream` and write each chunk as it arrives, which holds one
        chunk at a time whatever the length::

            import soundfile as sf

            with sf.SoundFile(
                "book.wav", "w", samplerate=engine.algorithm.sample_rate, channels=1
            ) as out:
                for chunk in engine.stream(text, voice, seed=7):
                    out.write(chunk.audio)

        That writes a plain WAV: the provenance manifest belongs to
        :meth:`Result.save`, which needs the whole render to describe it.

        ``language`` resolves as it does everywhere: the argument, then
        ``voice.language``, then ``"en"``. Left ``None`` here so :meth:`stream`
        resolves it once, on the one path that renders.

        ``speed`` is applied per chunk, exactly as :meth:`stream` applies it, so
        the two paths still produce the same waveform. ``previous_tokens``
        conditions the *first* chunk; every chunk after it is conditioned on the
        one before, as always.
        """
        self._refuse_if_wedged()
        parts = list(
            self.stream(
                text,
                voice,
                seed=seed,
                language=language,
                speed=speed,
                previous_tokens=previous_tokens,
                # Throughput path: the whole passage is drained before anyone
                # hears a byte, so first-audio protection would only give up
                # overlap. Measured on a 3090 with CUDA graphs: RTF 8.94
                # unprotected vs 7.46 protected, byte-identical output.
                latency_mode=False,
            )
        )
        if not parts:
            raise NothingToSpeakError("nothing to speak")
        if len(parts) == 1:
            # `replace`, not `parts[0]`: a chunk's Result carries the *chunk*
            # seed, `_derive(seed, 16 + index)`. Returning it verbatim made
            # `Result.seed` report `derive(seed, 16)` for a one-chunk text and
            # `seed` for a two-chunk one — the same call answering differently
            # depending on how the text splits, from the field
            # whose docstring says it holds "everything needed to explain or
            # reproduce it" and which `__repr__` prints.
            return replace(parts[0], seed=seed)

        # Preallocated rather than `np.concatenate`, which holds the chunk list
        # and the joined array at once: a long passage peaked at twice the audio
        # it was producing, at exactly the moment it was largest. The mel below
        # is joined the same way for the same reason.
        # Held honestly: every chunk's Result stays alive until return, so peak
        # memory is the joined array plus all chunks. Passages beyond that
        # budget should consume `stream()` directly.
        total = sum(len(p.audio) for p in parts)
        audio = np.empty(total, dtype=parts[0].audio.dtype)
        at = 0
        for part in parts:
            audio[at : at + len(part.audio)] = part.audio
            at += len(part.audio)
        tokens = [t for p in parts for t in p.tokens]
        frames = sum(p.mel.shape[1] for p in parts)
        mel = np.empty((parts[0].mel.shape[0], frames), dtype=parts[0].mel.dtype)
        at = 0
        for part in parts:
            mel[:, at : at + part.mel.shape[1]] = part.mel
            at += part.mel.shape[1]
        timings = StageTimings(
            sum(p.timings.tokens for p in parts),
            sum(p.timings.mel for p in parts),
            sum(p.timings.audio for p in parts),
        )
        return Result(
            audio=audio,
            tokens=tokens,
            mel=mel,
            seed=seed,
            sample_rate=self.algorithm.sample_rate,
            timings=timings,
            algorithm_fingerprint=self.algorithm.fingerprint(),
            recipe_version=self.algorithm.recipe_version,
            voice_name=voice.name,
            language=parts[0].language,
            voice_sha256=voice.source_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            backend=self.backend,
            execution=self.execution.describe(),
            hit_token_cap=any(p.hit_token_cap for p in parts),
            inspections=tuple(i for p in parts for i in p.inspections),
            speed=speed,
            # Rebuilt from the parts rather than shifting each part's own
            # timing by a running float: `timeline` accumulates sample offsets
            # as integers, so the joins are exact and every chunk's `end` is
            # the next one's `start` down to the last bit.
            chunks=timeline(
                [
                    ChunkSpan(
                        text=p.chunks[0].text if p.chunks else "",
                        samples=len(p.audio),
                        tokens=len(p.tokens),
                    )
                    for p in parts
                ],
                sample_rate=self.algorithm.sample_rate,
            ),
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
        """Yield one :class:`Result` per chunk, as each becomes ready.

        For a reading app this is the difference between waiting for a paragraph
        and hearing the first sentence immediately. Time to first audio is set by
        the first chunk, not by the passage.

        Each chunk is conditioned on the tail of the previous one when
        ``ChunkConfig.prefix_tokens`` is non-zero. Without that, every chunk
        restarts its pitch contour like a fresh sentence and the restart is
        audible at every join.

        Each chunk also gets its own derived seed, so the same passage streams
        identically whether or not the caller stops early — a chunk's audio does
        not depend on how many chunks came before it.

        ``should_cancel`` is polled on **every decode step** of the token
        generator: when it returns true, generation stops within one forward
        pass (the token that was about to be sampled is discarded). This is how
        a voice agent does barge-in — a human interrupts, the speaker goes
        quiet within a few tokens, not after a ~10 s chunk. The partial chunk is
        then **discarded without being rendered**: those tokens are speech the
        listener has already stopped wanting, and the mel decode plus vocode is
        pure waste at that point.

        **Cancelling does not un-deliver audio.** Chunks already yielded from
        this generator are the caller's, and they will play unless the caller
        drops them. Stopping the engine is half of a barge-in; flushing the
        playback buffer is the other half, and it is usually the larger one.
        See ``docs/design/barge-in.md``.

        ``language`` resolves as it does everywhere: the argument, then
        ``voice.language``, then ``"en"``. Resolved once here rather than per
        chunk, so every chunk of a passage is read in the same language.

        ``speed`` stretches each chunk independently, which is the same
        independence the seeds and the prefix already have: a chunk's audio must
        not depend on how many came before it, or a listener who stops early
        would have heard something different from one who did not.

        ``latency_mode`` (default ``True``) protects time-to-first-audio when
        both stages share one device: generation of window 1 waits until
        window 0's render is out the door, giving up one render's worth of
        overlap (measured on a 3090 with CUDA graphs: first audio 0.86 s
        protected vs 1.14 s not, RTF 7.46 vs 8.94). A caller that drains the
        whole stream before playing anything — :meth:`synthesize_long` — turns
        it off and keeps the full overlap. On split placements (generator on
        CPU, renderer on GPU) the stages do not contend and the flag changes
        nothing.

        ``previous_tokens`` seeds the carry, so the first chunk of *this* call
        is conditioned on the tail of a *previous* one. It is the same
        conditioning the joins inside a passage already use — the carry variable
        below simply starts non-empty — which is why a request boundary stops
        being audible without a second mechanism existing to maintain.
        """
        # A generator body, so this runs on the first `next()` rather than at
        # the call — the same point at which the empty-text refusal below
        # raises. Both are the caller's first look at the stream.
        self._refuse_if_wedged()
        language = _resolve_language(language, voice)
        # Empty input refuses here exactly as in `synthesize` and
        # `synthesize_long`: an empty stream with no error is indistinguishable
        # from a clean one.
        if not text.strip():
            raise NothingToSpeakError("nothing to speak")
        validate_speed(speed)
        # Run the speech funnel on the whole text BEFORE splitting. The funnel
        # (SpeechText.prepared: invisibles, symbols, and for Polish the
        # respelling of embedded English) can change the text length — Polish
        # respelling expands "download" to "daunlod". If it ran per chunk after
        # the budget was computed, a chunk could silently overflow its window.
        # Running it first means the splitter budgets the text it will actually
        # speak.
        from .frontend.polish import speech_text

        prepared = speech_text(text, language)
        # The funnel may remove everything (emoji, bare symbols, invisible
        # marks): refuse here, identically to ``synthesize`` and
        # ``synthesize_long``, so a transport cannot answer a request whose
        # words were all stripped with a clean, empty success.
        if not prepared.strip():
            raise NothingToSpeakError("nothing to speak")
        chunks = split_text(prepared, self.algorithm.chunking)
        prefix_len = self.algorithm.chunking.prefix_tokens
        carry: list[int] = self._carry_from(previous_tokens)

        if len(chunks) == 1:
            # No pipeline for one window: there is nothing to overlap, and the
            # serial path keeps the single-sentence call free of threads.
            if should_cancel is not None and should_cancel():
                return
            result = self._synthesize_one(
                chunks[0],
                voice,
                seed=_derive(seed, _STREAM_CHUNK),
                language=language,
                prefix=carry,
                prepared=True,
                is_terminal=True,
                speed=speed,
                should_cancel=should_cancel,
            )
            if result is not None:
                yield result
            return

        yield from self._stream_pipelined(
            chunks,
            voice,
            seed=seed,
            language=language,
            speed=speed,
            carry=carry,
            prefix_len=prefix_len,
            should_cancel=should_cancel,
            latency_mode=latency_mode,
        )

    def _stream_pipelined(  # noqa: PLR0915 — one producer, one consumer, one teardown
        self,
        chunks: Sequence[str],
        voice: VoiceProfile,
        *,
        seed: int,
        language: str,
        speed: float,
        carry: list[int],
        prefix_len: int,
        should_cancel: Callable[[], bool] | None,
        latency_mode: bool,
    ) -> Generator[Result, None, None]:
        """Generate window *k+1* while window *k* renders.

        The chunk chain is sequential only through the token phase — window
        *k+1*'s prefix is the tail of window *k*'s **tokens**, which exist
        before its render begins. Rendering is a pure function of the window,
        so it moves to a worker while generation continues. Same windows, same
        seeds, same math, one render at a time: the audio is byte-identical to
        the serial path, which a test asserts.

        A producer thread walks the token chain and hands each finished window
        to a single-thread render pool; this generator drains the futures in
        order. ``_PIPELINE_DEPTH`` bounds how far generation runs ahead of the
        consumer, so an abandoned or slow consumer stops the engine instead of
        letting it speak the whole book into memory.

        Cancellation keeps its token-level latency: the producer polls the
        caller's ``should_cancel`` (and this generator's own close) inside the
        decode loop, exactly as the serial path did.
        """
        import queue as queue_mod
        from concurrent.futures import Future, ThreadPoolExecutor, wait

        stop = threading.Event()
        # First audio is the product number, and on a single device the
        # pipeline can hurt it: generating window 1 contends with window 0's
        # render for the same GPU (measured on a 3090 with CUDA graphs: +0.35 s
        # to first audio). When the stages share a device, hold generation
        # until window 0's render is out the door — it costs one render's
        # worth of overlap at the start of the passage and protects the
        # latency the stream exists to deliver. Split placements (gen on CPU,
        # render on GPU) do not contend and keep the full overlap.
        protect_first_render = latency_mode and (
            self.execution.resolved_generator_device()
            == self.execution.resolved_renderer_device()
        )

        def cancelled() -> bool:
            return stop.is_set() or (should_cancel is not None and should_cancel())

        out: queue_mod.Queue[tuple[str, Future[Result | None] | BaseException | None]] = (
            queue_mod.Queue(maxsize=_PIPELINE_DEPTH)
        )
        """("item", future) | ("done", None) | ("error", exc), in production order."""

        submitted: list[Future[Result | None]] = []
        submitted_lock = threading.Lock()
        """Every render handed to the pool that has not finished, in order.

        Kept so that teardown can *cancel* the ones nobody is going to read.
        ``ThreadPoolExecutor.__exit__`` drains its own queue before it returns,
        so a render queued behind the one in flight still runs to completion
        after the consumer has walked away, and the producer's join waits for
        all of it. That turns the join timeout into a bound on ``depth ×
        render`` rather than on one render, and a renderer merely slower than
        that arithmetic is then indistinguishable here from a thread that will
        never come back — a healthy engine reported wedged.

        The lock is not for the list alone: it pairs the ``stop`` check with the
        submit, so a window generated in the instant before teardown cannot slip
        into the pool after the cancellation sweep has already passed over it.
        """

        def produce() -> None:
            try:
                with ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="loudkit-render"
                ) as renderer:
                    prefix = carry
                    for index, chunk in enumerate(chunks):
                        if cancelled():
                            break
                        window = self._generate_window(
                            chunk,
                            voice,
                            seed=_derive(seed, _STREAM_CHUNK + index),
                            language=language,
                            prefix=prefix,
                            prepared=True,
                            # Only the last chunk ends the passage — see the
                            # serial path.
                            is_terminal=index == len(chunks) - 1,
                            should_cancel=cancelled,
                        )
                        if window is None:
                            # Cancelled mid-generation: nothing to render,
                            # nothing to carry.
                            break
                        if prefix_len:
                            prefix = list(window.speech[-prefix_len:])
                        with submitted_lock:
                            if stop.is_set():
                                # Teardown has already swept `submitted`; a
                                # render submitted now would be the one nobody
                                # can cancel.
                                break
                            # Backpressure stays *after* the submit, not before
                            # it: holding a queue slot before handing the window
                            # to the renderer would idle the render thread for
                            # exactly the overlap the pipeline exists to create.
                            # What was missing is the cancellation below, not a
                            # tighter bound here.
                            # `cancelled` rides along so a cancel that lands
                            # while the mel decoder runs stops the window
                            # before the vocoder, not after it.
                            future = renderer.submit(
                                self._render_window,
                                window,
                                voice,
                                speed=speed,
                                should_cancel=cancelled,
                            )
                            submitted[:] = [f for f in submitted if not f.done()]
                            submitted.append(future)
                        # Blocks when the consumer is _PIPELINE_DEPTH behind —
                        # the backpressure that keeps generation from running
                        # arbitrarily ahead.
                        out.put(("item", future))
                        if index == 0 and protect_first_render:
                            # See protect_first_render above. Wait, do not
                            # read: an exception belongs to the consumer, who
                            # gets it from the same future.
                            wait([future])
                out.put(("done", None))
            except BaseException as exc:  # noqa: BLE001 — ferried to the consumer
                out.put(("error", exc))

        producer = threading.Thread(target=produce, name="loudkit-generate", daemon=True)
        producer.start()
        try:
            while True:
                kind, payload = out.get()
                if kind == "done":
                    return
                if kind == "error":
                    assert isinstance(payload, BaseException)  # narrowed by kind
                    raise payload
                # The serial contract: once the caller cancels, nothing more
                # is yielded — including a window the pipeline already
                # rendered ahead. The wasted render is bounded by
                # ``_PIPELINE_DEPTH``; the semantics are not.
                if should_cancel is not None and should_cancel():
                    return
                assert isinstance(payload, Future)  # narrowed by kind
                rendered = payload.result()
                if rendered is None:
                    # The render phase saw the cancel between its stages. The
                    # producer is winding down for the same reason; nothing
                    # after a half-rendered window is worth yielding.
                    return
                yield rendered
        finally:
            # Consumer closed early (or an error above): tell the producer,
            # then unblock any put() it is waiting on so it can exit, and let
            # the render in flight finish before the engine is used again.
            stop.set()
            # Take back every render that has not started. The one already
            # running cannot be preempted — it is inside the vocoder, holding
            # exactly the state this teardown exists to reclaim — so `cancel()`
            # returns False for it and the join below waits for that one and
            # only that one. Without this the join waits for the whole pipeline
            # depth, and the wedge verdict below stops being about a stuck
            # thread and starts being about a slow renderer.
            with submitted_lock:
                doomed = list(submitted)
                submitted.clear()
            for future in doomed:
                future.cancel()
            try:
                while True:
                    out.get_nowait()
            except queue_mod.Empty:
                pass
            producer.join(timeout=_PRODUCER_JOIN_TIMEOUT)
            if producer.is_alive():
                # The thread outliving the join still holds this engine's
                # stages, so the engine stops being usable here — reporting it
                # and carrying on left the next call to contend with a render
                # nobody is waiting on, which reads as a slow request rather
                # than as a stuck thread.
                #
                # One way only. The thread is still inside the stages and
                # nothing on this side can prove it has left, so there is no
                # honest test for "recovered": a long-running process replaces
                # the engine, which is the same cost it pays for any other
                # unrecoverable component.
                #
                # Reported by name — the name `threading.enumerate()` shows —
                # so the log line and the stack dump name the same thread.
                self._wedge(
                    f"{producer.name} was still running "
                    f"{_PRODUCER_JOIN_TIMEOUT:.0f}s after a stream closed, and it "
                    "still holds the token generator and the renderer"
                )
                _LOG.error(
                    "%s is still running %.0fs after the stream closed; this "
                    "engine refuses every call from here",
                    producer.name,
                    _PRODUCER_JOIN_TIMEOUT,
                )

    def _carry_from(self, previous_tokens: SpeechTokens | None) -> list[int]:
        """The conditioning context a call inherits from the one before it.

        The same slice the streaming loop takes between two chunks — last
        ``chunking.prefix_tokens`` — applied to tokens that came from a
        different call. There is deliberately no second mechanism: a request
        boundary and a chunk boundary are the same join, and the reason chunk
        joins do not stutter is the reason request joins should not either.

        Any length is accepted because only the tail is used, so
        ``previous_tokens=result.tokens`` is the intended call and a caller
        should never have to know the prefix length to make it.

        Raises:
            ValueError: for an id outside the acoustic codebook. The whole input
                is checked rather than only the slice that will be used: an id
                out of range means the sequence was built wrong, and reporting
                that only when it happens to land in the last six tokens would
                make the failure depend on the length of the caller's text.
        """
        if previous_tokens is None:
            return []
        wanted = self.algorithm.chunking.prefix_tokens
        tokens = [int(t) for t in previous_tokens]
        validate_speech_tokens(tokens, limit=self.algorithm.start_speech_token)
        # Not `tokens[-wanted:]`: a zero there is the whole list rather than
        # nothing, which would condition on the entire previous utterance at
        # exactly the setting that means "chunks are independent".
        return tokens[-wanted:] if wanted > 0 else []

    def _strip_specials(self, tokens: SpeechTokens) -> list[int]:
        """Drop start/stop markers and anything above the acoustic codebook.

        The generator emits its own control tokens; the renderer only
        understands codebook entries.
        """
        limit = self.algorithm.start_speech_token
        out = [int(t) for t in tokens if int(t) < limit]
        window = self.algorithm.window.max_speech_tokens
        if len(out) > window:
            # Loud, not sliced: silent truncation loses text while the audio
            # still sounds fine, and only a listener who knows the passage
            # notices.
            dropped = len(out) - window
            raise WindowOverflowError(
                f"{len(out)} speech tokens exceed the {window}-token window by "
                f"{dropped} (~{dropped / self.algorithm.token_rate_hz:.1f}s of "
                "speech would be lost).\n"
                "Split the text first: Engine.synthesize_long() does it for you.",
                n_tokens=len(out),
                window=window,
            )
        return out

    # -- construction -------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        device: str = "cpu",
        execution: ExecutionConfig | ExecutionOverrides | None = None,
        algorithm: AlgorithmConfig | None = None,
    ) -> Engine:
        """Build an engine from the synthesis checkpoint.

        The checkpoint's manifest is the authority on algorithm values that are
        properties of the weights — silence ids, step count, vocabulary — so
        that a new backend cannot re-guess them.

        Args:
            path: packed ``.safetensors``.
            device: ``cpu``, ``cuda`` or ``mps``. A backend is selected for it.
            execution: an :class:`~loudkit.config.ExecutionOverrides` to change
                named fields and inherit the manifest's defaults for the rest,
                or a full :class:`~loudkit.config.ExecutionConfig` to specify
                everything. ``None`` takes the manifest's shipping map.
            algorithm: override the manifest's algorithm. Deliberate deviations
                only; the fingerprint will differ from the shipping one and
                conformance will say so.
        """
        from .backends import build_engine

        return build_engine(path, device=device, execution=execution, algorithm=algorithm)


def validate_speech_tokens(
    tokens: SpeechTokens | None, *, limit: int, field: str = "previous_tokens"
) -> None:
    """Refuse a caller-supplied token sequence that the renderer cannot look up.

    Module level and public so a transport can run the check *before* it commits
    anything to it. The HTTP streaming route is the reason: it takes the
    single-flight engine slot before the response starts, so a request whose
    token list was never usable would otherwise queue behind every other
    synthesis in order to fail — and fail after the 200, where a status code can
    no longer be sent. The check is pure arithmetic on the request's own body;
    nothing about its answer changes while the server runs.

    Raises:
        InvalidTokensError: naming the first offending id and the bound.
    """
    if tokens is None:
        return
    for token in tokens:
        # Integer-*valued*, not merely integer-convertible. `int(1.9)` is 1, so
        # a float array of "almost" ids passed here and then meant something
        # different in every backend: torch refuses a float index outright, ONNX
        # truncates silently, and the four ports read the field as `i64` and
        # never see the fraction at all. One profile, several readings.
        value = int(token)
        if value != token:
            raise InvalidTokensError(
                f"{field} contains {token!r}, which is not a whole number. "
                "A speech token id indexes a table; a fraction is not an index, "
                "and truncating it silently renders something else.",
                token=token,
                limit=limit,
            )
        if not 0 <= value < limit:
            raise InvalidTokensError(
                f"{field} contains {value}, which is not an acoustic "
                f"speech token (expected 0 <= id < {limit}). Pass "
                "`Result.tokens` from an earlier call; the generator's own "
                "control tokens are already stripped from it.",
                token=value,
                limit=limit,
            )


def _resolve_language(language: str | None, voice: VoiceProfile) -> str:
    """The language a synthesis is actually read in: argument, voice, English.

    The chain is ``language`` if the caller gave one, else ``voice.language`` if
    the profile carries one, else ``"en"``.

    Without the voice link, ``engine.synthesize("Cześć", polish_voice)`` runs
    Polish text through the English frontend — English number words, English
    abbreviation expansion, no Polish respelling — and nothing in the audio
    reports the mismatch. A profile records the language of the audio it was
    enrolled from, so the voice is the better default than a constant, and a
    Polish voice reading Polish text needs no argument at all.

    Passing ``language`` explicitly is how **cross-lingual** synthesis is
    requested: an English voice reading Polish text is
    ``synthesize(text, english_voice, language="pl")``, and the explicit
    argument always wins over the profile.
    """
    if language is not None:
        return language
    return voice.language or _FALLBACK_LANGUAGE


def _derive(seed: int, stream: int) -> int:
    """Per-stage seed from one user seed.

    Splitting rather than sharing means a change in how many numbers the sampler
    consumes cannot shift the flow prior, so an optimisation in one stage cannot
    silently alter another stage's output.
    """
    return (seed * 0x9E3779B97F4A7C15 + stream * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
