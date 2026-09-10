"""One window: the token phase, the render phase, and the seed law between them.

Free functions over the engine's stages, so :mod:`loudkit.stream` can run the
two phases on different threads. Seeds are derived, not shared: each stage
draws its own sub-stream of the caller's seed, so a change in how many numbers
one stage consumes cannot shift another's. ``docs/design/engine-pipeline.md``
has the stream numbers and why chunk 0 draws the raw seed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import AlgorithmConfig, ExecutionConfig
from .contracts import MelDecoder, SpeechTokens, TextFrontend, TokenGenerator, Vocoder, Waveform
from .errors import (
    CancelledError,
    InvalidTokensError,
    NothingToSpeakError,
    WindowOverflowError,
)
from .frontend.chunking import split_in_half
from .models.timestretch import time_stretch
from .models.windowing import eos_floor
from .postprocess import Inspection, ceiling_for, inspect
from .result import Provenance, Result, StageTimings
from .sampler import LRSamplerV1
from .timing import ChunkSpan, timeline
from .voice import VoiceProfile

_FALLBACK_LANGUAGE = "en"

_STREAM_FLOW = 1
_STREAM_VOCODER = 2
_STREAM_RETRY = 8
"""Retry attempt n draws derive(seed, 8 + n); the ladder stays below the chunk streams."""
_STREAM_CHUNK = 16
"""Chunk k > 0 draws derive(seed, 16 + k). Chunk 0 draws the caller's seed itself, so a
text that fits one window renders what the ports' single-window call renders."""
_STREAM_RESPLIT = 4096
"""The second half of a re-split chunk, derived from the chunk's own seed."""


def _derive(seed: int, stream: int) -> int:
    """Per-stage seed from one user seed."""
    return (seed * 0x9E3779B97F4A7C15 + stream * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF


class Stages(Protocol):
    """What the window functions need of an engine."""

    @property
    def frontend(self) -> TextFrontend: ...
    @property
    def token_generator(self) -> TokenGenerator: ...
    @property
    def mel_decoder(self) -> MelDecoder: ...
    @property
    def vocoder(self) -> Vocoder: ...
    @property
    def algorithm(self) -> AlgorithmConfig: ...
    @property
    def execution(self) -> ExecutionConfig: ...
    @property
    def backend(self) -> str: ...
    @property
    def checkpoint_sha256(self) -> str: ...


@dataclass(frozen=True, slots=True)
class GeneratedWindow:
    """A window after the token phase, before the render phase."""

    text: str
    language: str
    speech: list[int]
    seed: int
    verdict: Inspection
    hit_token_cap: bool
    hit_window_cap: bool
    """Whether the window filled, measured before postprocess trimmed anything."""
    n_generated: int
    generate_seconds: float


def resolve_language(language: str | None, voice: VoiceProfile) -> str:
    """The argument, else the voice's language, else English."""
    if language is not None:
        return language
    return voice.language or _FALLBACK_LANGUAGE


def provenance_for(stages: Stages, voice: VoiceProfile, language: str) -> Provenance:
    return Provenance(
        algorithm_fingerprint=stages.algorithm.fingerprint(),
        recipe_version=stages.algorithm.recipe_version,
        voice=voice.name,
        language=language,
        voice_sha256=voice.source_sha256,
        checkpoint_sha256=stages.checkpoint_sha256,
        backend=stages.backend,
        execution=stages.execution.describe(),
    )


def generate_window(
    stages: Stages,
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
    """The token phase: funnel, generate, judge, retry, trim.

    Raises :class:`CancelledError` when ``should_cancel`` fired; the partial
    tokens are discarded, not rendered. ``prepared`` skips the funnel for a
    caller that ran it on the whole text before splitting.
    """
    algorithm = stages.algorithm
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1: {max_new_tokens}")
    cap = max_new_tokens if max_new_tokens is not None else algorithm.sampling.max_new_tokens
    if not prepared:
        from .frontend.speechtext import speech_text

        text = speech_text(text, language)
    # The funnel may remove everything (a footnote marker, an emoji): refused
    # here, where synthesize and stream both pass.
    if not text.strip():
        raise NothingToSpeakError(
            "nothing to speak: the text funnel removed every character. "
            "Footnote markers, emoji and symbols with no word in the "
            "render language are dropped, and this input was only those."
        )

    text_tokens = stages.frontend.encode(text, language)
    pp = algorithm.postprocess
    floor = eos_floor(len(text_tokens), algorithm)
    if pp.mode != "off":
        # The length ceiling stops a runaway during generation, not after it.
        cap = min(
            cap,
            ceiling_for(len(text_tokens), config=pp, window=algorithm.window.max_speech_tokens),
        )

    # A condemned window (dropout, or suspect) is regenerated from a derived
    # seed up to retry_max_attempts times; the attempt that ships is the one
    # with the fewest true-silence tokens, earliest on a tie.
    dead_air = frozenset(pp.silence_render_ids or algorithm.sampling.silence_token_ids)
    attempt = 0
    tokens_elapsed = 0.0
    best: tuple[int, list[int], Inspection, bool, bool] | None = None
    while True:
        attempt_seed = seed if attempt == 0 else _derive(seed, _STREAM_RETRY + attempt)
        sampler = LRSamplerV1(
            algorithm.sampling,
            seed=attempt_seed,
            stop_token=algorithm.stop_speech_token if pp.mode != "off" else None,
            eos_floor=floor,
        )
        t0 = time.perf_counter()
        tokens = stages.token_generator.generate(
            text_tokens,
            voice,
            sampler=sampler,
            max_new_tokens=cap,
            prefix=prefix,
            should_cancel=should_cancel,
        )
        tokens_elapsed += time.perf_counter() - t0
        _refuse_if_cancelled(should_cancel)

        # `gen` is the row: every committed token, the stop marker excluded.
        # The detectors index into it, so they run before anything renumbers.
        stop = algorithm.stop_speech_token
        gen = list(tokens)
        ended = bool(gen) and gen[-1] == stop
        if ended:
            gen.pop()
        peak_at, peak_prob = sampler.eos_peak
        # Both measured here on the raw row: a trim may leave one token behind.
        hit_cap, hit_window = (
            not ended and len(gen) >= n for n in (cap, algorithm.window.max_speech_tokens)
        )
        verdict = inspect(
            gen,
            text_token_count=len(text_tokens),
            min_tokens=floor,
            eos_peak_at=peak_at,
            eos_peak_prob=peak_prob,
            ended=ended,
            is_terminal=is_terminal,
            hit_ceiling=hit_cap,
            silence=algorithm.sampling.silence_token_ids,
            config=pp,
        )
        condemned = verdict.reason == "dropout" or verdict.suspect
        if not condemned or pp.mode == "off":
            break
        silence_count = sum(1 for t in gen if t in dead_air)
        if best is None or silence_count < best[0]:
            best = (silence_count, gen, verdict, hit_cap, hit_window)
        if attempt >= pp.retry_max_attempts:
            _, gen, verdict, hit_cap, hit_window = best
            break
        attempt += 1

    n_generated = len(gen)
    if pp.mode == "trim" and verdict.keep < len(gen):
        gen = gen[: verdict.keep]
    speech = strip_specials(algorithm, gen)
    if len(speech) == 0:
        raise ValueError(
            "generation produced no speech tokens: the stop token was "
            "accepted immediately. Set sampling.min_tokens_floor above 0 to "
            "refuse that during sampling."
        )
    return GeneratedWindow(
        text=text,
        language=language,
        speech=speech,
        seed=seed,
        verdict=verdict,
        hit_token_cap=hit_cap,
        hit_window_cap=hit_window,
        n_generated=n_generated,
        generate_seconds=tokens_elapsed,
    )


def _fade_edges(audio: Waveform, *, sample_rate: int, seconds: float) -> Waveform:
    """Fade waveform edges; rationale and measurements: docs/design/postprocess.md.

    The reference ramp. The cosine is taken in float32, which no other libm
    reproduces, so the ports carry the two shipped lengths as bits read off this
    function: ``tests/data/conformance/edge_fade.json``.
    """
    n = int(seconds * sample_rate)
    if n <= 0 or len(audio) < 2 * n:
        return audio
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32))).astype(
        audio.dtype
    )
    out = np.array(audio, copy=True)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def render_window(
    stages: Stages,
    window: GeneratedWindow,
    voice: VoiceProfile,
    *,
    speed: float,
    should_cancel: Callable[[], bool] | None = None,
) -> Result:
    """The render phase: mel, vocoder, stretch. Deterministic given the window.

    ``should_cancel`` is polled between stages; a kernel already running is
    not interrupted, the next stage is not started.
    """
    algorithm = stages.algorithm
    speech, seed = window.speech, window.seed
    _refuse_if_cancelled(should_cancel)
    t1 = time.perf_counter()
    mel = stages.mel_decoder.decode(speech, voice, seed=_derive(seed, _STREAM_FLOW))
    t2 = time.perf_counter()
    _refuse_if_cancelled(should_cancel)
    audio = stages.vocoder.synthesize(mel, voice, seed=_derive(seed, _STREAM_VOCODER))
    t3 = time.perf_counter()
    # After the detectors, which judge pacing on the unstretched render.
    audio = time_stretch(audio, sample_rate=algorithm.sample_rate, speed=speed)
    audio = _fade_edges(audio, sample_rate=algorithm.sample_rate, seconds=algorithm.edge_fade)
    _refuse_if_cancelled(should_cancel)
    return Result(
        audio=audio,
        tokens=speech,
        mel=mel,
        seed=seed,
        sample_rate=algorithm.sample_rate,
        timings=StageTimings(window.generate_seconds, t2 - t1, t3 - t2),
        hit_token_cap=window.hit_token_cap,
        inspections=(window.verdict,),
        speed=speed,
        chunks=timeline(
            [ChunkSpan(text=window.text, samples=len(audio), tokens=len(speech))],
            sample_rate=algorithm.sample_rate,
        ),
        provenance=provenance_for(stages, voice, window.language),
    )


def windows_for_chunk(
    stages: Stages,
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
    """One chunk's windows: one, or two when the chunk filled its window.

    The chunker budgets characters and a slow voice can fill the window before
    the text runs out. Both halves derive from ``index``, so a re-split cannot
    move any later chunk's seed. A half that still overruns is not split again.
    """
    algorithm = stages.algorithm
    chunk_seed = seed if index == 0 else _derive(seed, _STREAM_CHUNK + index)
    window = generate_window(
        stages,
        chunk,
        voice,
        seed=chunk_seed,
        language=language,
        prefix=prefix,
        prepared=True,
        is_terminal=is_terminal,
        should_cancel=should_cancel,
    )
    # The window has to be what stopped it, not the length ceiling: halving a
    # runaway short text gives two runaways.
    if not window.hit_window_cap or algorithm.chunking.cap_resplit == "off":
        return [window]
    halves = split_in_half(chunk)
    if halves is None:
        return [window]
    first = generate_window(
        stages,
        halves[0],
        voice,
        seed=chunk_seed,
        language=language,
        prefix=prefix,
        prepared=True,
        is_terminal=False,
        should_cancel=should_cancel,
    )
    prefix_len = algorithm.chunking.prefix_tokens
    second = generate_window(
        stages,
        halves[1],
        voice,
        seed=_derive(chunk_seed, _STREAM_RESPLIT),
        language=language,
        prefix=carry_pair_aligned(algorithm, first.speech, prefix_len)
        if prefix_len
        else prefix,
        prepared=True,
        is_terminal=is_terminal,
        should_cancel=should_cancel,
    )
    return [first, second]


def _refuse_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise CancelledError("cancelled: should_cancel returned true")


def refuse_truncated_window(
    algorithm: AlgorithmConfig, n_tokens: int, hit_window_cap: bool
) -> None:
    """The one-window contract: text that did not fit is refused, not shipped short."""
    if not hit_window_cap:
        return
    window = algorithm.window.max_speech_tokens
    seconds = window / algorithm.token_rate_hz
    raise WindowOverflowError(
        f"the text did not fit one {window}-token window (~{seconds:.1f}s of "
        f"speech) and its tail was not spoken. Drop single_window: "
        f"synthesize() splits at sentence boundaries and joins the audio, "
        f"and stream() is the same path chunk by chunk.",
        n_tokens=n_tokens,
        window=window,
    )


def carry_pair_aligned(
    algorithm: AlgorithmConfig, tokens: Sequence[int], wanted: int
) -> list[int]:
    """The last ``wanted`` tokens, extended to start where a pair does under ``fusion_mtp2``.

    The generator re-pairs a prefix from its own start, so a tail beginning on
    the second half of a pair would fuse the right tokens with the wrong
    partners. One extra token restores the pairing.
    """
    if not wanted:
        return []
    end = len(tokens)
    start = end - wanted
    if algorithm.decode == "fusion_mtp2":
        end -= end % 2
        start = end - wanted
        if start % 2:
            start -= 1
    return [int(t) for t in tokens[max(start, 0) : end]]


def carry_from(algorithm: AlgorithmConfig, previous_tokens: SpeechTokens | None) -> list[int]:
    """The conditioning tail a call inherits from the one before it.

    The same slice a chunk join takes. The whole input is validated, not only
    the slice used, so a bad id is refused whatever the text length.
    """
    if previous_tokens is None:
        return []
    validate_speech_tokens(previous_tokens, limit=algorithm.start_speech_token)
    tokens = [int(t) for t in previous_tokens]
    return carry_pair_aligned(algorithm, tokens, algorithm.chunking.prefix_tokens)


def strip_specials(algorithm: AlgorithmConfig, tokens: SpeechTokens) -> list[int]:
    """Drop start/stop markers and anything above the acoustic codebook. Loud on overflow."""
    limit = algorithm.start_speech_token
    out = [int(t) for t in tokens if int(t) < limit]
    window = algorithm.window.max_speech_tokens
    if len(out) > window:
        dropped = len(out) - window
        raise WindowOverflowError(
            f"{len(out)} speech tokens exceed the {window}-token window by "
            f"{dropped} (~{dropped / algorithm.token_rate_hz:.1f}s of "
            "speech would be lost).\n"
            "Engine.synthesize() splits at sentence boundaries; "
            "single_window=True is what refuses instead.",
            n_tokens=len(out),
            window=window,
        )
    return out


def validate_speech_tokens(
    tokens: SpeechTokens | None, *, limit: int, field: str = "previous_tokens"
) -> None:
    """Refuse a token sequence the renderer cannot look up. Public, so a
    transport can check a request before it takes the engine slot.

    Raises:
        InvalidTokensError: naming the first offending id and the bound.
    """
    if tokens is None:
        return
    for token in tokens:
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


def require_a_voice_profile(voice: object) -> None:
    """Refuse anything that is not a :class:`~loudkit.voice.VoiceProfile`, naming the remedy."""
    if isinstance(voice, VoiceProfile):
        return
    got = type(voice).__name__
    remedy = (
        "lk.voice('joe', repo='loudreader/loudr-1') fetches one by name, and "
        "VoiceProfile.load(path) reads one from disk."
    )
    if isinstance(voice, str):
        raise TypeError(f"voice must be a VoiceProfile, not the name {voice!r}. {remedy}")
    raise TypeError(f"voice must be a VoiceProfile, not {got}. {remedy}")
