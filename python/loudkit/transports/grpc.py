"""A gRPC transport, over the same synthesis path as everything else.

The third transport after HTTP and MCP, and like both of them it **builds
nothing and decides nothing**: every method resolves a voice and calls
:func:`~loudkit.server.render_bytes` or
:func:`~loudkit.server.render_stream_chunks`. That is the rule this library is
organised around — one place audio is made — and it is why a third transport
costs a file rather than a subsystem. A transport that re-implemented the speed
clamp, the language chain or the prefix slice would be a second place for any of
them to drift, which is the failure the whole project exists to prevent.

Why gRPC at all, when HTTP with SSE already streams: a typed schema and
backpressure. `proto/loudkit.proto` says what a request is, in a form a client
generator can read, and a slow consumer on `SynthesizeStream` stops the producer
instead of filling a buffer. Neither matters on loopback with one caller, which
is why this is an extra rather than a dependency.

The engine is single-flight — it holds mutable decoder state and is not
reentrant — so a lock serialises synthesis. The lock belongs to the *engine*,
not to this server: two `build_server` calls over one engine share one lock,
keyed per engine instance in `_lock_for`. What that does not cover is another
transport in the same process — `transports.http` keeps a lease of its own on
the event loop, and nothing in this module can reach it. Running the HTTP app
and a gRPC server over one engine in one process is therefore **not**
single-flight across the pair; one engine per transport process is the
supported shape until the arbiter lives in `loudkit.synthesis`, where both
transports can take it.

Waiting for the lock used to be unbounded — which is how a slow synthesis took
the whole server down rather than just itself. On this transport waiting costs
a *thread*: ten callers waiting hold ten threads, the pool is eight, and gRPC
then has nowhere to run anything. Measured, `ListVoices` — which reads a list
of filenames and touches no engine — answered DEADLINE_EXCEEDED after four
seconds.

So the wait is bounded three ways, because the bounds fail differently:

* **depth**, at half the pool. Past it a caller is refused immediately and its
  thread goes straight back, which is what keeps the engine-free methods
  answering while synthesis is saturated. This is the bound that fixes the
  measurement above; with it, the same ten callers leave `ListVoices` answering
  in 0.00 s.
* **time**, `_MAX_WAIT_S`. The depth bound does nothing for one render that
  never returns, and the callers within the depth would wait on it forever.
* **the caller's own deadline.** A waiter whose deadline passes while it queues
  has an RPC gRPC has already closed; acquiring anyway rendered a full reply
  for a caller that heard DEADLINE_EXCEEDED before the render began. The wait
  is capped at `time_remaining()`, and a successful acquisition is re-checked
  against `is_active()` before it counts.

A depth bound alone lets the admitted callers wait forever; a time bound alone
still fills the pool with waiters — 120 s of waiting is a wedged server just as
surely as an infinite one.

**Stopping a render is one flag, on both RPCs.** `Engine.stream` polls
`should_cancel` on every token decode step, so the things that end a call early
— the client cancelling, the client's deadline expiring, and `_MAX_STREAM_S` on
a stream — all set the same event, and the render stops inside one decode step
rather than at the next chunk boundary. The first two arrive on
`context.add_callback`; the cap on a timer, because a check read *between*
chunks cannot fire while the render it bounds is the thing that has not
returned. `Synthesize` reaches the flag through `_cancellable`, which threads
it into the engine's internal seams without adding a synthesis path.

The cancellation is cooperative, not preemptive. The flag is read where the
token decode loop reads it, which is where nearly all the time goes — but a
backend kernel already executing runs to the end of its call: a mel decode, a
vocoder pass or a time-stretch that has entered its backend cannot be
interrupted mid-kernel, and a cancel lands within one such step, never within
zero.

**A stream renders on a producer thread and delivers over a bounded queue.**
The producer owns the engine for the stream's whole life: it renders chunks,
puts them on a two-slot queue, and — whichever way the stream ends — reclaims
the engine's own threads and releases the lock itself. The gRPC worker only
drains the queue and writes the socket. The point of the split is the peer
that stays connected and stops *reading*: the write blocks, the queue fills,
the producer parks on a bounded `put` polling the cancel flag, and at
`_MAX_STREAM_S` the flag fires, the render stops and the engine comes back —
while the worker is still stuck in a write nothing on the server can unblock
(keepalive settles peers that vanished without closing). An engine held
hostage by a socket was a resource contract this transport did not keep; a
worker thread held by one is the cost of gRPC's synchronous API, and it is a
thread, not the engine.

The queue is two chunks deep. Deeper buys nothing — the consumer is a socket —
and each slot is one encoded chunk, so the depth is also the memory bound on a
stalled stream.

Whichever way a stream ends, the engine's own threads are reclaimed *before*
the lock is released — `close()` on the chunk generator runs `Engine.stream`'s
teardown, which cancels the renders nobody will read and joins the engine's
own producer thread. Releasing first and letting the generator be collected
afterwards leaves that thread inside a non-reentrant engine the next caller
has already entered, which is the same ordering the HTTP route's lease exists
to hold. Because the stream's producer does both, the reclaim-then-release
order lives on one thread and cannot interleave.

Install with ``pip install "loudkit[grpc]"``. Run with
``loudkit grpc --checkpoint <path> --voices <dir>``.
"""

from __future__ import annotations

import logging
import queue as queue_mod
import threading
import time
import weakref
from concurrent import futures
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_args

from ..errors import LoudkitError, UnsupportedLanguageError, VoiceNotFoundError, error_code
from ..frontend.chunking import CHARS_PER_TOKEN, estimate_tokens
from ..frontend.polish import speech_text
from ..models.timestretch import MAX_SPEED, MIN_SPEED
from ..synthesis import (
    _MAX_PREVIOUS_TOKENS,
    _MAX_TEXT_LEN,
    _MAX_WAIT_S,
    AudioFormat,
    Rendered,
    VoiceLibrary,
    render_bytes,
    render_stream_chunks,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from ..engine import Engine

__all__ = ["build_server", "serve"]

_MISSING_EXTRA = 'the gRPC server needs grpcio.\n  pip install "loudkit[grpc]"'

_LOG = logging.getLogger("loudkit.transports.grpc")
"""Where a defect's detail goes, since the client does not get it.

Same split the HTTP server makes: a caller can act on `invalid_request`, and a
`server_fault` is a filesystem path or a tensor shape handed to whoever can
reach the port. gRPC's default for an escaping exception is UNKNOWN carrying
`Exception calling application: <repr>`, which is that detail on the wire.
"""

_MAX_STREAM_S = 600.0
"""Wall-clock cap on one `SynthesizeStream`, counted from the moment it takes
the engine.

The HTTP stream needs no such cap because it polls `is_disconnected()` and the
event loop is never the thing being held. This one holds the engine lock for
its whole life, so a stream that never ends is an engine nobody else gets. Ten
minutes is far longer than any real passage and far shorter than forever.

Armed on a timer rather than read between chunks: the render is what fails to
return, so the check that bounds it cannot be a line the render has to reach.
The timer sets the cancel flag `Engine.stream` polls on every decode step, and
the stream's bounded `put` polls the same flag — so the cap frees the engine
whether the stream is stuck rendering or stuck delivering.
"""

_CLIENT_REPLY_LIMIT = 4 * 1024 * 1024
"""How large a `Synthesize` reply a default gRPC client will accept.

grpc's `max_receive_message_length` defaults to 4 MiB in every language, and a
reply past it is refused by the *client*, after the server has spent the whole
render producing it. This transport therefore refuses such a request before it
takes the engine, rather than turning an hour of audio into a RESOURCE_EXHAUSTED
the caller pays for twice.

The server cannot fix this by raising a limit of its own: the bound that bites
belongs to the peer, and a client generated from `loudkit.proto` in another
language has not been told to raise it. `SynthesizeStream` has no such ceiling —
each chunk is its own message — which is what the refusal points at.
"""

_REPLY_HEADROOM = 64 * 1024
"""Bytes reserved out of `_CLIENT_REPLY_LIMIT` for everything but the samples.

The client's 4 MiB bound is on the whole message, and the message is more than
frames: the WAV header, the trailing C2PA provenance manifest the WAV carries,
the other response fields and protobuf's own framing all count against it. A
preflight that budgets the samples to exactly the limit admits replies a few
kilobytes over it — refused by the client after the full render, which is the
precise failure the preflight exists to prevent. 64 KiB is over an order of
magnitude more than the manifest and headers measure, and costs under 0.7 s of
audio off the admissible length.
"""

_BYTES_PER_SAMPLE = 2
"""Mono int16, which is what `_quantise` produces for every container.

wav and pcm16 carry exactly this; flac and ogg carry less. The bound below uses
the uncompressed figure for all four because a compressed size is not knowable
from the text, and the two ways of being wrong are not symmetric: refusing a
compressible passage costs one clear error naming the streaming RPC, and
admitting one costs a full render the caller cannot receive.
"""

_QUEUE_CHUNKS = 2
"""Depth of the queue between a stream's producer and the gRPC write path.

One slot would make the producer lockstep with the socket, giving up the
overlap that lets chunk *k+1* render while *k* is in flight; anything deep is a
memory promise made on behalf of a peer that may never read. Two is the
smallest depth with overlap, and it is also the bound on what a stalled stream
can hold: two encoded chunks, and nothing more, however long the peer stalls.
"""

_QUEUE_POLL_S = 0.05
"""How often the two ends of the stream queue re-check their exit conditions.

The producer's `put` polls the cancel flag with it, so a stalled stream stops
rendering within this of the cap; the writer's `get` polls the producer's
liveness with it, so the end of a stream costs at most this in extra latency.
Small enough to be unnoticeable, large enough to cost nothing.
"""

_FORMATS: frozenset[str] = frozenset(get_args(AudioFormat))
"""The encodings `audio_format` may name, read off the type the encoder switches
on rather than restated here.

proto3 has no `Literal`, so what pydantic refuses for the HTTP server with a 422
listing the four that work arrives here as an arbitrary string. Checked at the
boundary for the same reason the speed range is: an unknown name reached the
encoder as a `KeyError`, which grpc reports as UNKNOWN with no error code — the
one verdict that tells a caller nothing about whose fault it is.
"""

_KEEPALIVE_OPTIONS: list[tuple[str, int]] = [
    # Detect a peer that vanished without closing: without these a half-open
    # connection holds its worker until the process restarts. (The engine it
    # cannot hold — the stream's producer frees that at `_MAX_STREAM_S` — but
    # a worker is still a worker.)
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.min_ping_interval_without_data_ms", 10_000),
    # A single client cannot occupy the whole pool by opening streams.
    ("grpc.max_concurrent_streams", 16),
    # grpc's default accepts a 4 MB request. This transport's largest honest
    # one is a 10 000 character string and a few thousand token ids -- under
    # 100 kB -- and the character cap is a check that runs *after* the message
    # has been deserialised, so the default let a caller spend four megabytes
    # of the process's memory to earn an INVALID_ARGUMENT. Refused at the
    # transport instead, before any of it is parsed. gRPC core answers that
    # refusal itself, as RESOURCE_EXHAUSTED with its own message: no servicer
    # frame exists yet, so no `loudkit-error-code` can ride along, and the
    # proto says so rather than promising metadata core cannot attach.
    ("grpc.max_receive_message_length", 256 * 1024),
]


_ENGINE_LOCKS: dict[int, threading.Lock] = {}
_ENGINE_LOCKS_GUARD = threading.Lock()


def _lock_for(engine: Engine) -> threading.Lock:
    """The single-flight lock for *this* engine, shared by every server over it.

    Keyed on identity rather than held per `build_server` call: the thing that
    is not reentrant is the engine, so a lock private to one server instance is
    a claim about the wrong object — two servers over one engine would each
    hold a lock the other never sees. Keyed on `id()` rather than the engine
    itself because `Engine` is a frozen dataclass: two engines loaded from one
    checkpoint compare equal, and equal keys sharing a lock would serialise
    independent engines. The finalizer drops the entry when the engine goes,
    so the map cannot outgrow the engines alive.
    """
    with _ENGINE_LOCKS_GUARD:
        key = id(engine)
        lock = _ENGINE_LOCKS.get(key)
        if lock is None:
            lock = _ENGINE_LOCKS[key] = threading.Lock()
            weakref.finalize(engine, _ENGINE_LOCKS.pop, key, None)
        return lock


class _cancellable:  # noqa: N801 - reads as a modifier at its call site
    """The engine with a cancel flag threaded into its render path.

    `render_bytes` takes no cancellation callback and neither do the engine's
    public entry points — only :meth:`Engine.stream` and the internal
    `_synthesize_one` accept one, and every render passes through one of those
    two seams. This facade forwards everything to the real engine and injects
    the flag at exactly those seams: `synthesize_long` and `synthesize` run the
    engine's own method bodies, bound here so their internal calls land back on
    this facade, and the flag rides `setdefault` so the engine's own internal
    wiring always wins. No audio is made here and no decision about audio is
    taken here, so the bytes stay the ones the conformance suite pins; the
    alternative — a cancellation parameter through `render_bytes` and
    `synthesize_long` — is an API change to modules a transport does not own.
    """

    def __init__(self, engine: Engine, should_cancel: Callable[[], bool]) -> None:
        self._inner = engine
        self._should_cancel = should_cancel

    def synthesize(self, *args: Any, **kwargs: Any) -> Any:
        from ..engine import Engine

        return Engine.synthesize(cast("Engine", self), *args, **kwargs)

    def synthesize_long(self, *args: Any, **kwargs: Any) -> Any:
        from ..engine import Engine

        return Engine.synthesize_long(cast("Engine", self), *args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("should_cancel", self._should_cancel)
        return self._inner.stream(*args, **kwargs)

    def _synthesize_one(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("should_cancel", self._should_cancel)
        return self._inner._synthesize_one(*args, **kwargs)  # noqa: SLF001 - the seam itself

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _load_grpc() -> Any:
    try:
        import grpc
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise ModuleNotFoundError(_MISSING_EXTRA) from exc
    return grpc


def _load_stubs() -> tuple[Any, Any]:
    try:
        from ..proto import loudkit_pb2, loudkit_pb2_grpc
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on generation
        raise ModuleNotFoundError(
            "the generated gRPC stubs are missing.\n  python tools/gen_proto.py"
        ) from exc
    return loudkit_pb2, loudkit_pb2_grpc


def build_server(  # noqa: PLR0915 - one server: its bounds and its four methods
    engine: Engine,
    voices: VoiceLibrary,
    *,
    max_workers: int = 8,
) -> Any:
    """A configured ``grpc.Server``, not yet started or bound.

    ``max_workers`` bounds gRPC's own pool, which is a queue depth rather than a
    parallelism setting: synthesis serialises behind the engine lock whatever
    this is, and the pool only decides how many callers may be waiting inside
    the process before the transport starts refusing.
    """
    grpc = _load_grpc()
    pb2, pb2_grpc = _load_stubs()

    # Same reason as the HTTP server's single-flight slot: the engine holds
    # mutable decoder state and a CUDA graph capture is not reentrant. Owned
    # by the engine rather than by this server — see `_lock_for`.
    synth_lock = _lock_for(engine)

    # How many callers may be *waiting* for the engine, as opposed to holding
    # it. Deliberately a fraction of the pool rather than a round number: on
    # this transport waiting costs a thread, so a queue as deep as the pool is
    # a queue that consumes the pool. Ten callers against eight workers left
    # nothing to answer `ListVoices` with, and `ListVoices` reads a list of
    # filenames. The HTTP server can afford `_MAX_QUEUED = 32` because waiting
    # there costs a coroutine.
    max_queued = max(1, max_workers // 2)
    queued = 0
    queue_lock = threading.Lock()
    # When the engine was taken, or None when it is free. Read by `Describe`,
    # which does not take the lock — a held-engine report that waits for the
    # engine reports nothing at the only moment it is wanted. A list because it
    # is written from the request threads and closed over, and `nonlocal` on a
    # float would be one more name to keep in step across two release sites.
    held_since: list[float | None] = [None]

    def _take_engine(context: Any) -> bool:
        """Acquire the engine, or refuse. Never queues past a bound.

        Three bounds, because they fail differently. The depth bound is what
        keeps the transport answering: past it a caller is turned away
        *immediately* and its thread goes back to the pool, so the methods
        that need no engine keep working while synthesis is saturated. The
        time bound is for the other shape — one render that never returns,
        holding callers who are within the depth. And the caller's own
        deadline caps the wait below both: a waiter whose deadline has passed
        has an RPC gRPC already closed, so acquiring for it renders audio for
        nobody — the engine is simply not taken.

        The wait is sliced rather than a single ``acquire(timeout=...)``,
        because a cancel is the one ending no timeout can see: a caller with
        no deadline that hung up still held its worker thread for the full
        ``_MAX_WAIT_S``: two minutes of pool for an RPC that ended in its
        first second. Each slice re-checks ``is_active()``, so a dead RPC
        gives its thread back within ``_QUEUE_POLL_S``. The re-check after a
        successful acquire covers the cancel that lands inside a slice.
        """
        nonlocal queued
        with queue_lock:
            if queued >= max_queued:
                _fail(
                    context,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    f"{queued} callers are already waiting for the engine",
                    error_code="busy",
                )
                return False
            queued += 1
        try:
            remaining = context.time_remaining()  # None when the RPC has no deadline
            wait = _MAX_WAIT_S if remaining is None else min(_MAX_WAIT_S, remaining)
            give_up_at = time.monotonic() + wait
            acquired = False
            while True:
                slice_s = min(_QUEUE_POLL_S, give_up_at - time.monotonic())
                if slice_s <= 0:
                    break
                if synth_lock.acquire(timeout=slice_s):
                    acquired = True
                    break
                if not context.is_active():
                    # Cancelled between slices. gRPC has already answered
                    # CANCELLED; there is no RPC left to carry a status, so
                    # the thread just goes back to the pool.
                    return False
            if acquired:
                if not context.is_active():
                    # Cancelled while waiting. The lock is held for nobody:
                    # give it straight back, and say nothing — there is no RPC
                    # left to carry a status.
                    synth_lock.release()
                    return False
                held_since[0] = time.monotonic()
                return True
            if not context.is_active():
                # The deadline (or a cancel) ended the RPC during the wait;
                # grpc has already answered DEADLINE_EXCEEDED or CANCELLED.
                return False
            _fail(
                context,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"waited {wait:.0f}s for the engine and never got it; "
                "another synthesis is still running",
                error_code="busy",
            )
            return False
        finally:
            with queue_lock:
                queued -= 1

    def _fail(
        context: Any, code: Any, message: str, *, error_code: str = "invalid_request"
    ) -> None:
        """Set the status, and name the condition in trailing metadata.

        ``loudkit-error-code`` carries the same frozen catalog the HTTP error
        bodies carry as ``"code"`` (see :mod:`loudkit.errors`), so a caller
        switching transports keeps the same vocabulary. gRPC status codes are
        coarser than the catalog — INVALID_ARGUMENT covers a dozen refusals —
        which is exactly why the metadata exists.
        """
        context.set_code(code)
        context.set_details(message)
        context.set_trailing_metadata((("loudkit-error-code", error_code),))

    def _resolve(  # noqa: PLR0911 - one refusal per line, each naming its own condition
        request: Any, context: Any
    ) -> Any | None:
        """The voice, or ``None`` with the status already set on ``context``."""
        if len(request.text) > _MAX_TEXT_LEN:
            _fail(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f"text is {len(request.text)} characters; the cap is {_MAX_TEXT_LEN}",
            )
            return None
        if not request.text.strip():
            _fail(context, grpc.StatusCode.INVALID_ARGUMENT, "text is empty")
            return None
        if len(request.previous_tokens) > _MAX_PREVIOUS_TOKENS:
            _fail(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f"previous_tokens has {len(request.previous_tokens)} entries; "
                f"the cap is {_MAX_PREVIOUS_TOKENS}",
            )
            return None
        # Speed and format, before the engine rather than inside it. Both are
        # questions about the request that nothing about the server's state can
        # change. Left to the engine they arrive as an exception no clause
        # here catches: an out-of-range speed as `validate_speed`'s ValueError,
        # an unknown format as a KeyError in the encoder. grpc turns an escaping
        # exception into UNKNOWN with no trailing metadata, which tells a caller
        # neither what was wrong nor whose fault it was, and tells it only after
        # the wait for the engine. The HTTP server refuses both in the request
        # model for the same reason.
        speed = request.speed
        if speed and not MIN_SPEED <= speed <= MAX_SPEED:
            _fail(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f"speed {speed} is outside [{MIN_SPEED}, {MAX_SPEED}]",
            )
            return None
        if request.audio_format and request.audio_format not in _FORMATS:
            _fail(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f"audio_format {request.audio_format!r} is not one of "
                f"{', '.join(sorted(_FORMATS))}",
            )
            return None
        try:
            return voices.load(request.voice)
        except VoiceNotFoundError as exc:
            _fail(context, grpc.StatusCode.NOT_FOUND, str(exc), error_code=error_code(exc))
        except ValueError as exc:  # a path, an empty string — not a voice *name*
            _fail(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                str(exc),
                error_code=error_code(exc),
            )
        return None

    def _kwargs(request: Any) -> dict[str, Any]:
        # proto3 cannot distinguish an absent scalar from a zero one, so the
        # zero values are read as "unset" for the two fields where zero is not
        # a value anybody means: 0.0 is not a speed and "" is not a format.
        return {
            "seed": int(request.seed),
            "language": request.language or None,
            "speed": request.speed or 1.0,
            "previous_tokens": list(request.previous_tokens) or None,
            "audio_format": request.audio_format or "wav",
        }

    def _refuse_oversize_reply(request: Any, context: Any, voice: Any) -> bool:
        """True when the unary reply would not fit a default client, said early.

        The size is bounded rather than measured, from numbers the engine
        already publishes: `estimate_tokens` is the chunker's own conservative
        *upper* estimate of the speech tokens a string produces, `token_rate_hz`
        turns tokens into seconds, and a slow `speed` stretches them. Nothing
        here renders anything, so the caller learns before the wait rather than
        after the work — which was the whole defect: an hour of audio, produced
        in full, then refused by the client's 4 MiB receive limit.

        Measured on the text the engine will *speak*, not the text the caller
        sent. The normalisation funnel runs before tokenising and it expands:
        digits become number words, dates become phrases, symbols become their
        names — a thousand characters of ``9`` normalise to five thousand of
        "nine", five times the audio the raw length promises — so a bound on
        the raw characters admits replies several times over the client's
        limit. The funnel is deterministic and cheap, so it runs here first
        and the estimate is of what will actually be rendered, under the same
        language chain the engine resolves (argument, then the voice's, then
        "en"), computed from the request and the voice this method already
        has.

        Conservative in the direction that costs nothing. The estimate over-
        counts (0.5 characters per token against 0.53 measured at the
        densest), treats every container as uncompressed, and reserves
        `_REPLY_HEADROOM` for the bytes around the samples — the WAV header,
        the provenance manifest, protobuf framing — so the passages turned
        away near the line would mostly have fitted. Each is turned away with
        the RPC that has no such ceiling in the message.
        """
        speed = request.speed or 1.0
        spoken = speech_text(request.text, request.language or voice.language or "en")
        rate = engine.algorithm.token_rate_hz
        seconds = estimate_tokens(spoken) / rate / speed
        size = int(seconds * engine.algorithm.sample_rate * _BYTES_PER_SAMPLE)
        budget = _CLIENT_REPLY_LIMIT - _REPLY_HEADROOM
        if size <= budget:
            return False
        fits = int(
            budget
            / _BYTES_PER_SAMPLE
            / engine.algorithm.sample_rate
            * rate
            * speed
            * CHARS_PER_TOKEN
        )
        _fail(
            context,
            grpc.StatusCode.INVALID_ARGUMENT,
            f"{len(request.text)} characters normalise to {len(spoken)} and render "
            f"up to {seconds:.0f}s of audio ({size / 1024 / 1024:.0f} MiB), and a "
            f"default gRPC client refuses a reply over "
            f"{_CLIENT_REPLY_LIMIT // 1024 // 1024} MiB. Use SynthesizeStream, "
            f"which sends one message per chunk, or send text that normalises to "
            f"at most about {fits} characters at this speed.",
            error_code="payload_too_large",
        )
        return True

    def _reclaim(chunks: Any) -> None:
        """Wait for the engine's own threads to stop, or record that they did not.

        `close()` raises GeneratorExit inside :meth:`~loudkit.engine.Engine.stream`,
        whose teardown cancels the renders nobody will read and joins the
        producer thread — so this returns only once the engine is idle or has
        marked itself unusable, which is the property the lock's next holder
        needs.

        It has to happen *before* the release. Letting the generator be
        collected after the lock is gone leaves that producer thread running
        inside a non-reentrant engine the next caller has already entered, which
        is a wrong answer rather than a slow one. The HTTP route reclaims in
        this order for the same reason.

        A failure here is not the next caller's to absorb: the stages cannot be
        shown to be free, so the engine is marked wedged rather than handed on.
        """
        try:
            chunks.close()
        except BaseException as exc:  # noqa: BLE001 — the verdict is the same for all
            _LOG.exception("could not reclaim the engine after a stream")
            engine._wedge(  # noqa: SLF001 — one package, one single-flight engine
                f"reclaiming an abandoned stream raised {type(exc).__name__}, "
                "so nothing here can show the token generator and the "
                "renderer were left idle"
            )

    class Speech(pb2_grpc.SpeechServicer):  # type: ignore[misc,name-defined]
        def Synthesize(  # noqa: N802, PLR0911 - one clause per named refusal
            self, request: Any, context: Any
        ) -> Any:
            voice = _resolve(request, context)
            if voice is None:
                return pb2.SynthesizeResponse()
            if _refuse_oversize_reply(request, context, voice):
                return pb2.SynthesizeResponse()
            # One flag for every way this call ends early, exactly as the
            # stream wires it: gRPC runs the callback when the RPC terminates,
            # which is what a client cancel and an expired deadline both are.
            # Without it the client's DEADLINE_EXCEEDED arrived on time and
            # the server rendered the whole reply regardless, holding the
            # engine for audio addressed to nobody.
            cancelled = threading.Event()
            if context.add_callback(cancelled.set) is False:
                # The RPC was already over before this line: grpc keeps no
                # callback list for a terminated call. `is False` rather than
                # `not`, so an implementation that returns nothing is read as
                # "registered" rather than as "cancel everything".
                cancelled.set()
            if not _take_engine(context):
                return pb2.SynthesizeResponse()
            try:
                rendered = render_bytes(
                    cast("Engine", _cancellable(engine, cancelled.is_set)),
                    request.text,
                    voice,
                    # Presence, not truthiness: an unmentioned `long_form`
                    # has to mean the library's default, not proto3's.
                    long_form=(request.long_form if request.HasField("long_form") else True),
                    **_kwargs(request),
                )
            except UnsupportedLanguageError as exc:
                _fail(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    str(exc),
                    error_code=error_code(exc),
                )
                return pb2.SynthesizeResponse()
            except LoudkitError as exc:
                _fail(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    str(exc),
                    error_code=error_code(exc),
                )
                return pb2.SynthesizeResponse()
            except ValueError as exc:
                # A refusal from a layer that has not earned a class yet. The
                # HTTP server answers 422 here and names the condition from the
                # same catalog; without this clause it left as UNKNOWN, which is
                # the status a caller can do least with.
                _fail(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    str(exc),
                    error_code=error_code(exc),
                )
                return pb2.SynthesizeResponse()
            except Exception:
                if cancelled.is_set():
                    # The cancel starves the render mid-pass, and the unwound
                    # call stack can surface as almost anything — the engine
                    # reports an emptied stream as its "returned nothing"
                    # RuntimeError, for one. The RPC is already closed, no
                    # status can reach the caller, and a cancel is not a
                    # defect, so it stays out of the defect log.
                    return pb2.SynthesizeResponse()
                # Not the caller's fault, and not the caller's detail: see
                # `_LOG`. INTERNAL rather than UNKNOWN so a client can tell "the
                # server broke" from "the server did not say".
                _LOG.exception("synthesis failed")
                _fail(
                    context,
                    grpc.StatusCode.INTERNAL,
                    "internal error",
                    error_code="server_fault",
                )
                return pb2.SynthesizeResponse()
            finally:
                held_since[0] = None
                synth_lock.release()
            return pb2.SynthesizeResponse(
                audio=rendered.data,
                media_type=rendered.media_type,
                duration_seconds=rendered.duration,
                token_count=rendered.n_tokens,
                truncated=rendered.hit_token_cap,
                continuation=list(rendered.continuation),
                fingerprint=engine.algorithm.fingerprint(),
                sample_rate=engine.algorithm.sample_rate,
            )

        def SynthesizeStream(  # noqa: N802, PLR0912, PLR0915 - one stream: producer, writer, verdict
            self, request: Any, context: Any
        ) -> Any:
            voice = _resolve(request, context)
            if voice is None:
                return
            fingerprint = engine.algorithm.fingerprint()
            sample_rate = engine.algorithm.sample_rate
            prefix_len = engine.algorithm.chunking.prefix_tokens
            # The lock spans the whole stream, not each chunk: the chunks of one
            # passage share the engine's carry between them, so letting a second
            # caller in mid-passage would interleave two readings.
            if not _take_engine(context):
                return
            # One flag for every way this stream ends early, because the engine
            # already takes one: `Engine.stream` polls `should_cancel` on every
            # decode step, so setting it stops a render inside one forward pass
            # instead of at the next chunk boundary — the same wiring the HTTP
            # route uses for a disconnected client.
            cancelled = threading.Event()
            timed_out = threading.Event()
            # The producer's failure, carried across threads for the writer to
            # turn into a status. One slot: the producer stops at its first.
            fault: list[BaseException] = []
            out: queue_mod.Queue[Rendered] = queue_mod.Queue(maxsize=_QUEUE_CHUNKS)

            def _expire() -> None:
                timed_out.set()
                cancelled.set()

            # A timer, not a check in a loop: the case the cap exists for is a
            # render (or a delivery) that has not returned, and a loop body is
            # only reached between chunks. The thread sleeps and sets a flag;
            # it never touches the engine.
            expiry = threading.Timer(_MAX_STREAM_S, _expire)
            expiry.daemon = True
            # Built before the producer starts, so the reclaim has something to
            # close on every path out, including a first `next()` that raises.
            chunks = render_stream_chunks(
                engine,
                request.text,
                voice,
                should_cancel=cancelled.is_set,
                **_kwargs(request),
            )

            def produce() -> None:
                """Render into the bounded queue, then give the engine back.

                This thread owns the engine from here: whichever way the
                stream ends — the passage finishing, a cancel, the cap, a
                synthesis failure — the reclaim and the release happen below,
                on this thread, in that order. Not on the gRPC worker, because
                the worker can be blocked in a socket write for as long as the
                peer cares to stall, and an engine whose release waits on a
                peer's read is an engine one silent client keeps from
                everyone.

                The `put` is bounded and polls the cancel flag: a full queue
                is a peer that is not reading, and the flag — a cancel, the
                deadline, the cap — is what turns "parked on a full queue"
                back into "engine free" within one poll.
                """
                try:
                    for chunk in chunks:
                        while True:
                            if cancelled.is_set():
                                return
                            try:
                                out.put(chunk, timeout=_QUEUE_POLL_S)
                                break
                            except queue_mod.Full:
                                continue
                except Exception as exc:
                    fault.append(exc)
                    if not isinstance(exc, (LoudkitError, ValueError)):
                        # A defect here; the detail goes to the operator's log
                        # (see `_LOG`), and the writer answers INTERNAL.
                        _LOG.exception("synthesis stream failed")
                finally:
                    expiry.cancel()
                    # Reclaim before release — see `_reclaim`. Both on this
                    # thread, so the ordering cannot interleave with anything.
                    _reclaim(chunks)
                    held_since[0] = None
                    synth_lock.release()

            producer = threading.Thread(target=produce, name="loudkit-grpc-stream", daemon=True)
            try:
                # Fires when the RPC terminates, which is what a cancel and an
                # expired client deadline both are. `is False`: grpc keeps no
                # callback list for a terminated call, and an implementation
                # that returns nothing must read as "registered", not as
                # "cancel everything".
                if context.add_callback(cancelled.set) is False:
                    cancelled.set()
                # The timer first: a producer that finishes before an unstarted
                # timer would cancel nothing, and the late timer would then
                # stamp `timed_out` on a stream that ended cleanly.
                expiry.start()
                producer.start()
            except BaseException:
                # The producer never ran, so nothing else will release.
                expiry.cancel()
                held_since[0] = None
                synth_lock.release()
                raise

            # The passage's tail so far, rebuilt chunk by chunk. A chunk's own
            # `continuation` is the tail of *that chunk's* tokens, and a chunk
            # shorter than the prefix carries fewer ids than the engine will
            # condition on — so the last chunk of "…long sentence. Go." hands
            # back two ids where `Synthesize` hands back six, and a client
            # chaining from it restarts most of its prosodic context. Folding
            # each chunk's tail onto the running one reproduces the passage
            # tail exactly, because the engine's own slice is the same
            # last-`prefix_len` rule this fold applies.
            tail: tuple[int, ...] = ()
            try:
                while True:
                    try:
                        item = out.get(timeout=_QUEUE_POLL_S)
                    except queue_mod.Empty:
                        if producer.is_alive():
                            if cancelled.is_set():
                                # The RPC is over or capped and nothing is
                                # buffered; the producer is mid-teardown and
                                # owns everything that remains.
                                break
                            continue
                        # The producer is done, so everything it will ever put
                        # is already in the queue; one racing item may have
                        # landed after the timeout above.
                        try:
                            item = out.get_nowait()
                        except queue_mod.Empty:
                            break
                    if prefix_len > 0:
                        tail = (tail + tuple(item.continuation))[-prefix_len:]
                    yield pb2.SynthesizeChunk(
                        audio=item.data,
                        media_type=item.media_type,
                        duration_seconds=item.duration,
                        token_count=item.n_tokens,
                        truncated=item.hit_token_cap,
                        continuation=list(tail),
                        fingerprint=fingerprint,
                        sample_rate=sample_rate,
                    )
                if fault:
                    exc = fault[0]
                    if isinstance(exc, (LoudkitError, ValueError)):
                        # LoudkitError covers the named refusals, unsupported
                        # language included; bare ValueError is a refusal from
                        # a layer that has not earned a class yet. Same split
                        # as `Synthesize`, arriving over the queue instead of
                        # the call stack.
                        _fail(
                            context,
                            grpc.StatusCode.INVALID_ARGUMENT,
                            str(exc),
                            error_code=error_code(exc),
                        )
                    else:
                        _fail(
                            context,
                            grpc.StatusCode.INTERNAL,
                            "internal error",
                            error_code="server_fault",
                        )
                elif timed_out.is_set():
                    # A cancelled RPC needs no status (grpc has already closed
                    # it with one), and a passage that simply ended must not
                    # be reported as late; `timed_out` is what tells the cap
                    # apart from both.
                    _fail(
                        context,
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                        f"stream exceeded {_MAX_STREAM_S:.0f}s holding the engine",
                        error_code="timeout",
                    )
            finally:
                # Every exit above is a `return` out of a generator, and a
                # generator abandoned by the framework is closed rather than
                # resumed — the flag is how that close reaches the producer.
                # The join keeps "this RPC is over" and "the engine is free"
                # in that order for anyone sequencing on the stream's end (the
                # next call, a Describe): it returns within one queue poll
                # plus the reclaim, because the flag just set is what every
                # wait in the producer polls.
                cancelled.set()
                producer.join()

        def Describe(self, request: Any, context: Any) -> Any:  # noqa: N802, ARG002
            from .. import __version__

            started = held_since[0]
            return pb2.DescribeResponse(
                algorithm=engine.algorithm.describe(),
                execution=engine.execution.describe(),
                fingerprint=engine.algorithm.fingerprint(),
                version=__version__,
                # Falls back to 0 at the end of every render, the abandoned
                # ones included: a cancel, an expired deadline, the stream cap
                # and a peer that stops reading all end at the cancel flag,
                # which the stream's producer polls even while parked on a
                # full queue. A value still growing past `_MAX_STREAM_S`
                # therefore names a render stuck inside a backend kernel — the
                # one place cooperative cancellation cannot reach.
                engine_held_seconds=0.0 if started is None else time.monotonic() - started,
            )

        def ListVoices(self, request: Any, context: Any) -> Any:  # noqa: N802, ARG002
            return pb2.ListVoicesResponse(voices=voices.names())

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=_KEEPALIVE_OPTIONS,
    )
    pb2_grpc.add_SpeechServicer_to_server(Speech(), server)
    return server


def serve(
    checkpoint: str,
    voices: str | Path | None = None,
    *,
    device: str | None = None,
    host: str = "127.0.0.1",
    port: int = 50051,
    first_chunk_tokens: int | None = None,
) -> None:
    """Load an engine and answer gRPC on ``host:port`` until interrupted.

    Loopback by default, for the same reason `loudkit serve` is: anyone who can
    reach the port can speak in every voice on the machine. Unlike the HTTP
    server this one has **no bearer token**, so a non-loopback bind is refused
    outright rather than being made defensible — adding auth to gRPC means
    interceptors and credentials, and shipping a half-guarded public port is
    worse than shipping none.

    ``first_chunk_tokens`` means what :func:`loudkit.server.serve` says: an
    opt-in budget on the first streamed chunk that buys time-to-first-audio at
    the price of a different algorithm fingerprint.
    """
    from .. import load

    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(
            f"refusing to bind {host}: the gRPC transport has no authentication. "
            "Put it behind something that does, or use `loudkit serve`, which "
            "requires a bearer token for a public bind."
        )

    from ..hub import backend_for_device, resolve_checkpoint

    # A repo id resolves to the checkpoint inside the snapshot the hub
    # returned, so the default voice directory is the snapshot's own `voices/`
    # (and `read_manifest` below gets a real file). A raw id handed to `Path`
    # computed `Path("org/repo").parent / "voices"`, and the server started
    # with the release's voices silently absent.
    # Normalised unconditionally, and by the backend the device needs. Three
    # things went wrong when this only ran for a repo id and always fetched
    # torch: a local *directory* kept its own name, so the voices below were
    # looked for one level above the release; and `device="onnx"` fetched a
    # snapshot holding no graphs, which the backend then could not run.
    # `resolve_checkpoint` answers all three shapes: a file is itself, a
    # directory yields the checkpoint inside it, a repo id fetches the set.
    ckpt = resolve_checkpoint(str(checkpoint), backend=backend_for_device(device))
    library = VoiceLibrary(Path(voices) if voices else ckpt.parent / "voices")
    algorithm = None
    if first_chunk_tokens is not None:
        import dataclasses

        from ..checkpoint import read_manifest
        from ..config import AlgorithmConfig

        base = AlgorithmConfig.from_manifest(read_manifest(ckpt))
        algorithm = dataclasses.replace(
            base,
            chunking=dataclasses.replace(
                base.chunking, first_chunk_max_tokens=first_chunk_tokens
            ),
        )
    engine = load(str(ckpt), device=device, algorithm=algorithm)
    names = library.names()
    if names:
        # First-use costs belong to startup, not the first caller — Engine.warm.
        engine.warm(library.load(names[0]))
    server = build_server(engine, library)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"loudkit gRPC on {host}:{port}  {engine.describe()}")
    print(f"voices: {', '.join(library.names()) or 'none in ' + str(library.root)}")
    server.wait_for_termination()
