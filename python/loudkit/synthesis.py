"""The synthesis surface every transport shares.

See ``docs/design/transports.md``.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .engine import Engine
from .errors import VoiceNotFoundError
from .result import Result
from .voice import VoiceProfile
from .window import carry_pair_aligned

_LOG = logging.getLogger("loudkit.synthesis")
"""Where this module's notices go. A library writing straight to stderr gives
a host application no way to route or silence them; a logger at WARNING still
reaches stderr through ``logging.lastResort`` on an unconfigured process, so
the warm-up lines guide 04 documents are not lost."""

_MAX_TEXT_LEN = 10_000
"""Upper bound on a synthesis request's text, in characters.

There is no auth on the server and the MCP transport is meant for agents, so
a request with an unbounded text field is a memory/latency DoS: chunking and
generation scale with input length. Capped here, at the single place audio is
made, so all three transports inherit it.
"""


_MAX_WAIT_S = 120.0
"""How long a request may wait for the single-flight engine before 503.

See ``docs/design/transports.md``.
"""


AudioFormat = Literal["wav", "pcm16", "flac", "ogg", "mp3", "opus"]
"""What a synthesis can be encoded as.

Everything here is written by the ``soundfile`` the server already depends on;
none of it adds a package. ``mp3`` and ``opus`` use the MPEG and Opus codecs of
the libsndfile that soundfile loads, which its wheels bundle; a libsndfile built
without them is refused by name before the engine runs. ``ogg`` and ``opus``
carry a random Ogg stream serial, so their bytes differ from run to run while
the samples do not; every other format repeats byte for byte. See
``docs/design/transports.md``.
"""


_ENCODINGS: dict[str, tuple[str, str, str]] = {
    # format -> (soundfile format, soundfile subtype, media type)
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "ogg": ("OGG", "VORBIS", "audio/ogg"),
    # What OpenAI clients ask for by default: MPEG-2 Layer III at the engine's
    # 24 kHz.
    "mp3": ("MP3", "MPEG_LAYER_III", "audio/mpeg"),
    # Ogg Opus, what chat voice notes carry. The codecs parameter is what tells
    # it apart from the Vorbis above, which shares the container and nothing
    # else.
    "opus": ("OGG", "OPUS", "audio/ogg; codecs=opus"),
    # Header-less frames, little-endian, for a caller feeding a device or a socket
    # directly.
    "pcm16": ("RAW", "PCM_16", "application/octet-stream"),
}


_MAX_PREVIOUS_TOKENS = 4096
"""Longest ``previous_tokens`` a request may carry.

The engine uses only the last ``chunking.prefix_tokens`` of it, six by default,
so anything beyond a previous utterance's worth is already pointless, and the
body bound above would not stop a caller from sending a megabyte of integers
that this server then parses into a list of Python ints. Bounded explicitly, and
generously: a whole window is 255 tokens, so 4096 is sixteen of them.
"""


_VOICE_CACHE_BYTES = 64 * 1024 * 1024
"""How much of the voice directory stays parsed in memory.

See ``docs/design/transports.md``.
"""


_MAX_VOICE_NAME = 128
"""Longest voice name a request may name.

Every shipped name is under twenty characters. The bound exists because the
filesystem has one of its own and enforces it with `OSError(63, 'File name too
long')`, which is not a refusal any transport was catching.
"""


@dataclass(frozen=True, slots=True)
class VoiceLibrary:
    """Voices available to the server, resolved by name.

    A directory rather than an open path parameter on purpose: a request naming
    a filesystem path would let anyone who can reach the port read any
    ``.safetensors`` on the machine.
    """

    root: Path

    _cache: dict[str, tuple[tuple[int, int], VoiceProfile]] = field(
        default_factory=dict, repr=False, compare=False
    )
    """name -> ((mtime_ns, size), profile). See :data:`_VOICE_CACHE_BYTES`.

    See ``docs/design/transports.md``.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    """Guards :attr:`_cache`, which several request threads share.

    See ``docs/design/transports.md``.
    """

    def names(self) -> list[str]:
        from .release import release_confinement

        boundary = release_confinement(self.root)
        return sorted(
            p.stem
            for p in self.root.glob("*.safetensors")
            if p.resolve().is_relative_to(boundary)
        )

    def load(self, name: str) -> VoiceProfile:
        # Reject separators outright rather than resolving and comparing: the
        # request supplies a name, and a name has no path in it. NUL is here
        # for the same reason and not because it escapes anything: no path can
        # hold one, so `resolve()` below raises `ValueError: lstat: embedded
        # null character in path`, which the transports answered with verbatim
        # -- an OS-layer sentence naming neither the field it came from nor
        # what a caller should send instead.
        if not name or "/" in name or "\\" in name or "\x00" in name or name.startswith("."):
            raise ValueError(f"not a voice name: {name!r}")
        # Bounded, because the filesystem bounds it and answers with an `OSError` that
        # is neither `VoiceNotFoundError` nor `ValueError`, so it would escape every
        # transport's error ladder. No real voice name is anywhere near this long.
        if len(name) > _MAX_VOICE_NAME:
            raise ValueError(
                f"not a voice name: {len(name)} characters, limit {_MAX_VOICE_NAME}"
            )
        # A name cannot escape the directory, but a symlink sitting inside it
        # can, and `glob` follows links: `voices/x.safetensors -> /elsewhere/y`
        # would otherwise hand any `.safetensors` on the host to an
        # unauthenticated caller. Resolve first, then confine, the same check
        # `hub.resolve_voice` makes, arriving here by a different door. The
        # boundary is the release, which for a Hub-cached one includes the
        # blob store its snapshot links into.
        from .release import release_confinement

        root = self.root.resolve()
        path = (root / f"{name}.safetensors").resolve()
        if not path.is_relative_to(release_confinement(self.root)) or not path.is_file():
            # Deliberately omits `self.root`: it's an absolute filesystem path
            # on the host, and this error is returned verbatim as an HTTP 404
            # detail to an unauthenticated client. The voice-name list is
            # already exposed on purpose via /v1/voices; the path isn't.
            available = tuple(self.names())
            raise VoiceNotFoundError(
                f"no voice {name!r}; have: {', '.join(available) or 'none'}",
                ref=name,
                available=available,
            )

        # Keyed on (mtime_ns, size) rather than on the name alone: a voice re-enrolled
        # under the same name has to be picked up on the next request, and on
        # nanoseconds because an enrolment loop overwrites within one second.
        info = path.stat()
        stamp = (info.st_mtime_ns, info.st_size)
        with self._lock:
            cached = self._cache.pop(name, None)
            if cached is not None and cached[0] == stamp:
                self._cache[name] = cached  # reinserted: last position is most recent
                return cached[1]
        profile = VoiceProfile.load(path)
        with self._lock:
            # A racing caller may have stored its own parse of the same file in
            # the meantime; last write wins, and both are the same voice.
            self._cache[name] = (stamp, profile)
            self._evict()
        return profile

    def _evict(self) -> None:
        """Drop least-recently-used profiles until the cache is inside its budget.

        See ``docs/design/transports.md``.
        """
        total = sum(profile.n_bytes for _, profile in self._cache.values())
        while total > _VOICE_CACHE_BYTES and len(self._cache) > 1:
            _, dropped = self._cache.pop(next(iter(self._cache)))
            total -= dropped.n_bytes


NO_WARM_ENV = "LOUDKIT_NO_WARM"
"""Any non-empty value skips every warm-up, ``0`` and ``false`` included.

A switch, not a boolean: it is either in the environment or it is not, and
there is nothing for a value to say. The first request pays instead.
"""


def warm_engine(engine: Engine, voices: VoiceLibrary) -> None:
    """Pay the first render's extra cost here, on the library's first voice.

    For a process that will hold this engine and answer with it, never for a
    library import and never for a command that renders once and exits, which
    has nowhere to hide the cost. Skipped when :data:`NO_WARM_ENV` is set.
    Nothing in it is fatal, the voice it reads included, because an
    optimisation that cannot run must not stop a process that can still
    serve. The line goes to stderr, where the MCP transport's protocol is
    not. See ``docs/design/transports.md``.
    """
    if os.environ.get(NO_WARM_ENV):
        return
    name = ""
    t0 = time.perf_counter()
    try:
        names = voices.names()
        if not names:
            return
        name = names[0]
        engine.warm(voices.load(name))
    except Exception as exc:  # whatever it was, serving is still on
        where = f"voice {name!r}" if name else str(voices.root)
        _LOG.warning("warm: %s failed (%s); the first render pays instead", where, exc)
        return
    _LOG.warning(
        "warm: first-use costs paid on voice '%s' (%.1fs)", name, time.perf_counter() - t0
    )


@dataclass(frozen=True, slots=True)
class Rendered:
    """One encoded synthesis, with the facts a caller needs to trust it.

    A class rather than a tuple, because ``hit_token_cap`` is the field a
    caller must not be able to drop by unpacking the first three. Truncation is
    not an error, the audio is real, it is just incomplete, so it travels as a
    field rather than an exception, and every transport is expected to forward
    it.
    """

    data: bytes
    """The encoded audio. WAV by default; see :data:`AudioFormat`.

    Named ``data`` rather than ``wav`` because it carries any of the six
    formats: a field called ``wav`` holding a FLAC is the kind of small lie
    that survives for years, since nothing ever asserts on a name.
    """

    duration: float
    n_tokens: int
    hit_token_cap: bool
    """True if generation stopped at the token cap rather than at a stop token.

    The utterance is cut off mid-sentence. Every transport must report it:
    silent truncation presented as complete audio reads as complete to an
    agent, which then moves on.
    """

    media_type: str = "audio/wav"
    """What to put in ``Content-Type``. Carried beside the bytes rather than
    re-derived at the route, so the two cannot disagree."""

    provenance: dict[str, object] | None = None
    """The unsigned loudkit provenance manifest for these bytes, if one was written.

    Always built; ``None`` only when the encoding cannot carry the trailing
    box (anything but WAV). Rides an HTTP header on every format, and the WAV
    carries the box itself, see :mod:`loudkit.provenance`.
    """

    continuation: tuple[int, ...] = ()
    """The tail to hand back as ``previous_tokens`` on the next request.

    The tail the engine itself would carry into the next window, six or seven
    ids depending on the decode recipe, rather than the whole token sequence:
    the tail is all the engine will use and the whole sequence is a few hundred
    integers a client would carry, log and send back for nothing. Small enough
    to ride an HTTP header.
    """


def _check_text(text: str) -> None:
    """The text cap, at the one place both render helpers can share.

    See ``docs/design/transports.md``.
    """
    if len(text) > _MAX_TEXT_LEN:
        raise ValueError(f"text too long: {len(text)} characters (max {_MAX_TEXT_LEN})")


def _check_encodable(audio_format: AudioFormat) -> None:
    """Refuse a format the loaded libsndfile cannot write, before the engine runs.

    Whether ``mp3`` and ``opus`` can be written depends on how that libsndfile
    was built, not on the request, and a refusal after the render has spent
    the render. See ``docs/design/transports.md``.
    """
    import soundfile as sf

    from .errors import UnsupportedFormatError

    fmt, subtype, _ = _ENCODINGS[audio_format]
    if fmt in sf.available_formats() and subtype in sf.available_subtypes(fmt):
        return
    raise UnsupportedFormatError(
        f"format {audio_format!r} needs libsndfile's {subtype} encoder, and the "
        f"libsndfile this server loads ({sf.__libsndfile_version__}) was built "
        "without it; wav and pcm16 always work"
    )


def _quantise(samples: NDArray[np.float32]) -> NDArray[np.int16]:
    """Float samples to int16 frames, once, by a rule of our own.

    See ``docs/design/transports.md``.
    """

    scaled = np.floor(np.asarray(samples, dtype=np.float64) * 32768.0)
    return np.clip(scaled, -32768.0, 32767.0).astype("<i2")


def _encode(
    frames: NDArray[np.int16], sample_rate: int, audio_format: AudioFormat
) -> tuple[bytes, str]:
    """Encode already-quantised frames, and say what they are.

    See ``docs/design/transports.md``.
    """
    import soundfile as sf

    fmt, subtype, media_type = _ENCODINGS[audio_format]
    buf = io.BytesIO()
    if fmt == "RAW":
        # RAW carries no header, so the byte order is not recorded anywhere in
        # the payload and has to be stated: little-endian, and the response says
        # so in a header because the bytes cannot.
        sf.write(buf, frames, sample_rate, format=fmt, subtype=subtype, endian="LITTLE")
    else:
        sf.write(buf, frames, sample_rate, format=fmt, subtype=subtype)
    return buf.getvalue(), media_type


def _continuation(engine: Engine, tokens: Sequence[int]) -> tuple[int, ...]:
    """The tail a client hands back as ``previous_tokens`` to continue this.

    :func:`~loudkit.window.carry_pair_aligned` and not a plain slice, and read
    off the engine's own ``chunking.prefix_tokens`` rather than a constant
    here. The slice is an algorithm value in two ways: how many ids, and where
    they start. Under ``fusion_mtp2`` the generator re-pairs a prefix from its
    own start, so a tail beginning on the second half of a pair fuses the right
    tokens with the wrong partners. Handing back the engine's own carry is what
    makes a client's join and an in-process join the same join.
    """
    wanted = engine.algorithm.chunking.prefix_tokens
    if wanted <= 0:
        return ()
    return tuple(carry_pair_aligned(engine.algorithm, tokens, wanted))


def fold_continuation(
    engine: Engine, tail: tuple[int, ...], chunk: tuple[int, ...]
) -> tuple[int, ...]:
    """The running continuation after one more streamed chunk.

    A stream's ``done`` event carries the passage's tail, not the last chunk's,
    because a final chunk shorter than the prefix carries fewer ids than the
    passage does. Here rather than in each transport so the fold obeys the same
    pair rule :func:`_continuation` obeys, in one place, for both streams.
    """
    wanted = engine.algorithm.chunking.prefix_tokens
    if wanted <= 0:
        return ()
    return tuple(carry_pair_aligned(engine.algorithm, tail + chunk, wanted))


def _provenance(
    engine: Engine,
    result: Result,
    voice: VoiceProfile,
    language: str,
    text: str,
    frames: NDArray[np.int16],
) -> dict[str, object]:
    """The unsigned loudkit provenance manifest for one rendered chunk.

    Built over the int16 frames every encoding carries, the server quantises
    once, so one hash binds all six formats to the same frames.
    """
    from ._version import package_version
    from .provenance import build_manifest

    version = package_version()
    return build_manifest(
        audio=frames.tobytes(),
        algorithm_fingerprint=engine.algorithm.fingerprint(),
        recipe_version=engine.algorithm.recipe_version,
        seed=result.seed,
        sample_rate=result.sample_rate,
        voice=voice.name,
        language=language or voice.language or "en",
        text=text,
        speed=result.speed,
        version=version,
        voice_sha256=voice.source_sha256,
        checkpoint_sha256=engine.checkpoint_sha256,
        backend=engine.backend,
        execution=engine.execution.describe(),
    )


def render_bytes(
    engine: Engine,
    text: str,
    voice: VoiceProfile,
    *,
    seed: int = 0,
    language: str | None = None,
    long_form: bool = True,
    speed: float = 1.0,
    previous_tokens: Sequence[int] | None = None,
    audio_format: AudioFormat = "wav",
    should_cancel: Callable[[], bool] | None = None,
) -> Rendered:
    """Synthesise and encode to ``audio_format``. The only place the server makes audio.

    Raises :class:`~loudkit.errors.CancelledError` when ``should_cancel``
    fired; each transport answers that in its own way. See
    ``docs/design/transports.md``.
    """
    _check_text(text)
    _check_encodable(audio_format)

    result = engine.synthesize(
        text,
        voice,
        seed=seed,
        language=language,
        speed=speed,
        previous_tokens=previous_tokens,
        single_window=not long_form,
        should_cancel=should_cancel,
    )
    return _rendered(engine, result, voice, language or "", text, audio_format)


def _rendered(
    engine: Engine,
    result: Result,
    voice: VoiceProfile,
    language: str,
    text: str,
    audio_format: AudioFormat,
) -> Rendered:
    """Encode a result and its provenance identically for every delivery mode."""
    frames = _quantise(result.audio)
    data, media_type = _encode(frames, result.sample_rate, audio_format)
    provenance = _provenance(engine, result, voice, language or "", text, frames)
    if audio_format == "wav":
        from .provenance import manifest_bytes

        data = data + manifest_bytes(provenance)
    return Rendered(
        data=data,
        media_type=media_type,
        duration=result.duration,
        n_tokens=len(result.tokens),
        hit_token_cap=result.hit_token_cap,
        provenance=provenance,
        continuation=_continuation(engine, result.tokens),
    )


def render_stream_chunks(
    engine: Engine,
    text: str,
    voice: VoiceProfile,
    *,
    seed: int = 0,
    language: str | None = None,
    speed: float = 1.0,
    previous_tokens: Sequence[int] | None = None,
    audio_format: AudioFormat = "wav",
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[Rendered]:
    """Synthesise chunk by chunk, yielding ``audio_format`` bytes as each becomes ready.

    See ``docs/design/transports.md``.
    """
    _check_text(text)
    _check_encodable(audio_format)

    for result in engine.stream(
        text,
        voice,
        seed=seed,
        language=language,
        speed=speed,
        previous_tokens=previous_tokens,
        should_cancel=should_cancel,
    ):
        chunk_text = result.chunks[0].text if result.chunks else text
        yield _rendered(engine, result, voice, language or "", chunk_text, audio_format)


def _first_exception(exc: BaseException) -> BaseException:
    """The real exception out of a (possibly nested) ExceptionGroup.

    See ``docs/design/transports.md``.
    """
    while True:
        nested = getattr(exc, "exceptions", None)
        if not nested:
            return exc
        exc = nested[0]
