"""The synthesis surface every transport shares.

``render_bytes`` is the only place in this package that turns an engine plus
a profile into encoded audio; the HTTP server, the MCP server and the gRPC
server are three adapters over it, and the conformance suite asserts that
each returns the bytes calling it directly would have returned. The request
limits here (text length, continuation length, queue wait) travel with it,
because a limit that one transport enforces and another does not is two
products wearing one name.

Everything in this module is transport-agnostic: no FastAPI, no MCP, no
grpcio. A transport that needs less than this module already requires is
importing too much of it.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .engine import Engine
from .errors import VoiceNotFoundError
from .voice import VoiceProfile

_MAX_TEXT_LEN = 10_000
"""Upper bound on a synthesis request's text, in characters.

There is no auth on the server and the MCP transport is meant for agents, so
a request with an unbounded text field is a memory/latency DoS: chunking and
generation scale with input length. Capped here, at the single place audio is
made, so both transports inherit it.
"""


_MAX_WAIT_S = 120.0
"""How long a request may wait for the single-flight engine before 503.

``_MAX_QUEUED`` bounds how many callers may be waiting; this bounds how
*long*. Without it, one synthesis that never returns — a backend wedged on a
driver, an ORT session deadlocked on a thread pool — holds the slot forever,
and every queued caller holds a connection open behind it with no answer
coming. Thirty
seconds of real synthesis is a long utterance, so this is four times the worst
honest wait and still a bounded one.

The render itself is deliberately **not** cancelled at this deadline. It runs in
a worker thread against an engine that is not reentrant and holds mutable
decoder state; abandoning it would let the next caller in while it is still
writing, which trades a hung request for a corrupted one. What the deadline
frees is the queue behind it: waiting callers get a truthful 503 instead of an
open socket, and ``/health`` reports the engine as stuck — see ``_started_at``.
"""


AudioFormat = Literal["wav", "pcm16", "flac", "ogg"]
"""What a synthesis can be encoded as.

Everything here is written by the ``soundfile`` the server already depends on;
none of it adds a package. Deliberately absent: **mp3 and opus**, which would
need an encoder this project does not ship and cannot ship everywhere — a format
that works on the maintainer's machine and fails on a user's is worse than one
that was never offered.
"""


_ENCODINGS: dict[str, tuple[str, str, str]] = {
    # format -> (soundfile format, soundfile subtype, media type)
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "ogg": ("OGG", "VORBIS", "audio/ogg"),
    # Header-less frames, little-endian, for a caller feeding a device or a
    # socket directly. `application/octet-stream` and an explicit rate header
    # rather than `audio/L16;rate=24000`: RFC 2586 defines L16 as **big**-endian,
    # and these frames are little-endian. Labelling them L16 would be a lie that
    # a conforming client would act on, and the byte order is not something the
    # payload can be inspected for.
    "pcm16": ("RAW", "PCM_16", "application/octet-stream"),
}


_MAX_PREVIOUS_TOKENS = 4096
"""Longest ``previous_tokens`` a request may carry.

The engine uses only the last ``chunking.prefix_tokens`` of it — six by default
— so anything beyond a previous utterance's worth is already pointless, and the
body bound above would not stop a caller from sending a megabyte of integers
that this server then parses into a list of Python ints. Bounded explicitly, and
generously: a whole window is 255 tokens, so 4096 is sixteen of them.
"""


_VOICE_CACHE_BYTES = 64 * 1024 * 1024
"""How much of the voice directory stays parsed in memory.

Loading a profile is a whole-file read, a SHA-256 over those bytes and a
safetensors parse; on the HTTP path that ran once per request, for a file that
had not changed, on the event loop. Bounded in bytes rather than in entries
because profiles differ by an order of magnitude in size (``MAX_VOICE_BYTES``
allows 8 MB, a real one is a few hundred KB), so an entry count is either a
memory promise it cannot keep or a cache that the shipped twenty-voice roster
evicts out of usefulness — the failure the generator's conditioning cache
already had.
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

    Insertion-ordered, and :meth:`load` reinserts on a hit, so the first key is
    the least recently *used* one — what :meth:`_evict` drops. Excluded from
    ``compare`` so two libraries over the same directory stay equal (and
    hashable) whatever either has read.

    Every read and write of this mapping happens under :attr:`_lock`.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    """Guards :attr:`_cache`, which several request threads share.

    The server serves this library from a thread pool, so the LRU bookkeeping
    is concurrent: ``pop`` then reinsert is two steps a second thread can land
    between, and :meth:`_evict` *iterates* the mapping while summing sizes,
    which another thread's insert turns into ``RuntimeError: dictionary
    changed size during iteration``. Atomic dict operations do not make a
    sequence of them atomic, which is what this class does.

    Held only across the bookkeeping, never across the file read — see
    :meth:`load`. Excluded from ``compare`` for the same reason ``_cache`` is.
    """

    def names(self) -> list[str]:
        root = self.root.resolve()
        return sorted(
            p.stem for p in self.root.glob("*.safetensors") if p.resolve().is_relative_to(root)
        )

    def load(self, name: str) -> VoiceProfile:
        from .voice import VoiceProfile

        # Reject separators outright rather than resolving and comparing: the
        # request supplies a name, and a name has no path in it.
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"not a voice name: {name!r}")
        # A name cannot escape the directory, but a symlink sitting inside it
        # can, and `glob` follows links: `voices/x.safetensors -> /elsewhere/y`
        # would otherwise hand any `.safetensors` on the host to an
        # unauthenticated caller. Resolve first, then confine — the same check
        # `hub.resolve_voice` makes, arriving here by a different door.
        root = self.root.resolve()
        path = (root / f"{name}.safetensors").resolve()
        if not path.is_relative_to(root) or not path.is_file():
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

        # Keyed on (mtime_ns, size) rather than on the name alone: a voice
        # re-enrolled under the same name has to be picked up on the next
        # request, and on nanoseconds rather than seconds because overwriting
        # within one second is exactly what an enrolment loop does.
        #
        # Two short critical sections rather than one around the whole method:
        # the parse below is a whole-file read plus a SHA-256, and holding
        # `_lock` across it would serialise every other name behind whichever
        # caller happened to miss. Two threads racing on the same cold voice
        # therefore both read the file, which costs a duplicate read and
        # produces the same profile either way — cheaper than making every hit
        # wait on someone else's miss, and cheaper than a per-name guard whose
        # own table would need the same locking as this one.
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

        The entry just stored is never dropped, even if it alone were over the
        budget: evicting what the current request is about to use would make the
        cache a pure cost.

        Callers must hold :attr:`_lock`: this walks the mapping, and an insert
        landing mid-walk is the ``dictionary changed size during iteration``
        the lock exists to stop.
        """
        total = sum(profile.n_bytes for _, profile in self._cache.values())
        while total > _VOICE_CACHE_BYTES and len(self._cache) > 1:
            _, dropped = self._cache.pop(next(iter(self._cache)))
            total -= dropped.n_bytes


@dataclass(frozen=True, slots=True)
class Rendered:
    """One encoded synthesis, with the facts a caller needs to trust it.

    A tuple was enough while there were three numbers; ``hit_token_cap`` is the
    fourth, and it is the one a caller must not be able to ignore by unpacking
    two of three. Truncation is not an error — the audio is real, it is just
    incomplete — so it travels as a field rather than an exception, and every
    transport is expected to forward it.
    """

    data: bytes
    """The encoded audio. WAV by default; see :data:`AudioFormat`.

    Named ``data`` rather than ``wav`` since it stopped always being one — a
    field called ``wav`` holding a FLAC is the kind of small lie that survives
    for years because nothing ever asserts on a name.
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
    """The C2PA claim-only manifest for these bytes, if one was written.

    Always built; ``None`` only when the encoding cannot carry the trailing
    box (anything but WAV). Rides an HTTP header on every format, and the WAV
    carries the box itself — see :mod:`loudkit.provenance`.
    """

    continuation: tuple[int, ...] = ()
    """The tail to hand back as ``previous_tokens`` on the next request.

    Exactly ``chunking.prefix_tokens`` ids — six by default — rather than the
    whole token sequence, because the tail is all the engine will use and the
    whole sequence is a few hundred integers a client would carry, log and send
    back for nothing. Small enough to ride an HTTP header.
    """


def _check_text(text: str) -> None:
    """The text cap, at the one place both render helpers can share.

    ``_MAX_TEXT_LEN`` says it is applied "at the single place audio is made, so
    both transports inherit it" — but only :func:`render_bytes` enforced it, and
    :func:`render_stream_chunks` is exported beside it in ``__all__``. Over HTTP
    pydantic covers the gap; an embedder calling the streaming helper directly
    had no bound at all, which is the caller least likely to have one of their
    own.
    """
    if len(text) > _MAX_TEXT_LEN:
        raise ValueError(f"text too long: {len(text)} characters (max {_MAX_TEXT_LEN})")


def _quantise(samples: Any) -> Any:
    """Float samples to int16 frames, once, by a rule of our own.

    This function exists because libsndfile does **not** apply one rule. Handed
    the same float array, its WAV writer floors and its FLAC writer rounds: at
    0.0020349235 the product is 66.68, WAV stores 66 and FLAC stores 67. On real
    engine audio that was 50 % of samples differing by one LSB between two
    formats both documented here as lossless — a claim that was false, and whose
    test could not see it because the fake vocoder renders silence and silence
    survives every codec identically.

    So the conversion happens here, once, and every container is handed the
    resulting int16 frames rather than the floats. wav, pcm16 and flac then carry
    identical samples *by construction* instead of by two encoders happening to
    agree.

    The rule is ``floor(x * 32768)``, clipped to the int16 range. Chosen rather
    than invented: it is bit-for-bit what libsndfile's WAV writer already did
    (verified over 40 009 samples including both rails and both signs), so the
    WAV bytes this server has always returned do not move. FLAC is the format
    that changes, by at most one LSB, toward what the WAV already said.

    Clipping rather than scaling by 32767: the engine's contract is samples in
    [-1, 1], the positive rail is one code short of the negative one in two's
    complement, and shrinking every sample to keep +1.0 representable would move
    every byte to accommodate a value the vocoder does not emit.
    """

    scaled = np.floor(np.asarray(samples, dtype=np.float64) * 32768.0)
    return np.clip(scaled, -32768.0, 32767.0).astype("<i2")


def _encode(frames: Any, sample_rate: int, audio_format: AudioFormat) -> tuple[bytes, str]:
    """Encode already-quantised frames, and say what they are.

    The samples are quantised to int16 by :func:`_quantise` before any encoder
    sees them, so wav, pcm16 and flac decode back to the *same* samples rather
    than to three encoders' opinions of the same floats. Ogg is lossy by
    construction and is the one format that does not round-trip.

    Every container is still written by ``soundfile``, ``pcm16`` included:
    hand-rolling the frame layout would be a second place for endianness and
    frame size to be decided.
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

    Read off the engine's own ``chunking.prefix_tokens`` rather than a constant
    here: the length is an algorithm value, it is in the fingerprint, and a
    server that shipped its own number would hand back the wrong amount of
    context the first time that recipe changed.
    """
    wanted = engine.algorithm.chunking.prefix_tokens
    return tuple(int(t) for t in tokens[-wanted:]) if wanted > 0 else ()


def _provenance(
    engine: Engine,
    result: Any,
    voice: VoiceProfile,
    language: str,
    text: str,
    frames: Any,
) -> dict[str, object]:
    """The C2PA claim-only manifest for one rendered chunk.

    Built over the int16 frames every encoding carries — the server quantises
    once, so one hash binds all four formats to the same bytes.
    """
    from .provenance import build_manifest

    try:
        from importlib.metadata import version as _pkg_version

        version = _pkg_version("loudkit")
    except Exception:  # pragma: no cover - metadata present in any install
        version = "0.1.0"
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
) -> Rendered:
    """Synthesise and encode to WAV. The only place the server makes audio.

    ``language`` is handed to the engine as given, ``None`` included: the engine
    owns the argument-then-voice-then-``"en"`` chain. ``speed`` and
    ``previous_tokens`` likewise: the range, the stretch and the prefix slice
    all live in the engine, and a transport that re-implements any of them is a
    second place for it to drift.
    """
    _check_text(text)

    result = (
        engine.synthesize_long(
            text,
            voice,
            seed=seed,
            language=language,
            speed=speed,
            previous_tokens=previous_tokens,
        )
        if long_form
        else engine.synthesize(
            text,
            voice,
            seed=seed,
            language=language,
            speed=speed,
            previous_tokens=previous_tokens,
        )
    )
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
    """Synthesise chunk by chunk, yielding WAV bytes as each becomes ready.

    The streaming half of :func:`render_bytes`, kept next to it so the two stay
    one path: a reader that streams and a reader that waits for the whole
    passage are the same synthesis with different delivery, not two engines.
    Uses :meth:`~loudkit.engine.Engine.stream` under the hood, so time to first
    audio is set by the first chunk rather than by the passage.

    ``should_cancel`` is handed to :meth:`~loudkit.engine.Engine.stream`, which
    polls it **on every decode step**, so a disconnected client stops costing
    GPU time within one forward pass rather than at the next chunk boundary.
    Passing it through matters more than it looks: a chunk is up to ~10 s of
    speech, so a callback checked only between chunks leaves a barge-in waiting
    seconds for audio no one wants. Polling per decode step is what makes the
    cancellation immediate.

    What cancelling does **not** do is recall chunks already yielded: over SSE
    those are events already on the wire, and stopping them playing is the
    client's job. ``docs/design/barge-in.md`` has the whole contract.

    Yields:
        One :class:`Rendered` per chunk, in order.
    """
    _check_text(text)

    for result in engine.stream(
        text,
        voice,
        seed=seed,
        language=language,
        speed=speed,
        previous_tokens=previous_tokens,
        should_cancel=should_cancel,
    ):
        frames = _quantise(result.audio)
        data, media_type = _encode(frames, result.sample_rate, audio_format)
        chunk_text = result.chunks[0].text if result.chunks else text
        provenance = _provenance(engine, result, voice, language or "", chunk_text, frames)
        if audio_format == "wav":
            from .provenance import manifest_bytes

            data = data + manifest_bytes(provenance)
        yield Rendered(
            data=data,
            media_type=media_type,
            duration=result.duration,
            n_tokens=len(result.tokens),
            hit_token_cap=result.hit_token_cap,
            provenance=provenance,
            continuation=_continuation(engine, result.tokens),
        )


def _first_exception(exc: BaseException) -> BaseException:
    """The real exception out of a (possibly nested) ExceptionGroup.

    An anyio task group reports whatever its child raised wrapped in an
    ExceptionGroup, whose own ``str`` is ``unhandled errors in a TaskGroup (1
    sub-exception)`` — true, and useless to the caller whose passage did not
    render.

    Unwrapped by duck-typing rather than ``except*`` or ``isinstance(...,
    BaseExceptionGroup)``: both are 3.11, this package supports 3.10, and on
    3.10 the group anyio raises comes from the ``exceptiongroup`` backport.
    """
    while True:
        nested = getattr(exc, "exceptions", None)
        if not nested:
            return exc
        exc = nested[0]
