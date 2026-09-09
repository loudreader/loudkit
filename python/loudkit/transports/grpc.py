"""A gRPC transport, over the same synthesis path as everything else.

Four methods (``Synthesize``, ``SynthesizeStream``, ``Describe``,
``ListVoices``) declared in ``proto/loudkit.proto``, for clients that want a
typed schema and backpressure. Every method resolves a voice and calls
:func:`~loudkit.synthesis.render_bytes` or its streaming twin; the request
caps and the bounds on the engine come from :mod:`.limits`, the same ones
HTTP and MCP apply. Loopback only: this transport has no authentication.
See ``docs/design/transports.md``.
"""

from __future__ import annotations

import logging
import queue as queue_mod
import threading
import time
import weakref
from concurrent import futures
from pathlib import Path
from typing import Any, get_args

from ..engine import Engine
from ..errors import (
    CancelledError,
    LoudkitError,
    UnsupportedLanguageError,
    VoiceNotFoundError,
    error_code,
)
from ..frontend.chunking import CHARS_PER_TOKEN, estimate_tokens
from ..frontend.speechtext import speech_text
from ..models.timestretch import MAX_SPEED, MIN_SPEED
from ..synthesis import (
    AudioFormat,
    Rendered,
    VoiceLibrary,
    fold_continuation,
    render_bytes,
    render_stream_chunks,
    warm_engine,
)
from .limits import (
    _MAX_STREAM_S,
    _MAX_WAIT_S,
    _host_is_loopback,
    _queue_depth_for,
    over_cap,
)
from .resolve import algorithm_override, open_release
from .schemas import _STREAMABLE

__all__ = ["build_server", "serve"]

_MISSING_EXTRA = 'the gRPC server needs grpcio.\n  pip install "loudkit[grpc]"'

_MISSING_PROTOBUF = 'the gRPC stubs need protobuf.\n  pip install "loudkit[grpc]"'
"""The remedy for a stub import that fails on `google.protobuf`.

The generated modules ship inside the package, so from an installed wheel they
are present and protobuf is the only thing under `..proto` that can be absent.
`tools/gen_proto.py` needs protobuf itself, so naming the generator here would
send the reader in a circle.
"""

_MISSING_STUBS = "the generated gRPC stubs are missing.\n  python tools/gen_proto.py"
"""The remedy in a source tree, where the generated modules can be absent."""

_LOG = logging.getLogger("loudkit.transports.grpc")
"""Where a defect's detail goes, since the client does not get it.

Same split the HTTP server makes: a caller can act on `invalid_request`, and a
`server_fault` is a filesystem path or a tensor shape handed to whoever can
reach the port. gRPC's default for an escaping exception is UNKNOWN carrying
`Exception calling application: <repr>`, which is that detail on the wire.
"""

_CLIENT_REPLY_LIMIT = 4 * 1024 * 1024
"""How large a `Synthesize` reply a default gRPC client will accept.

See ``docs/design/transports.md``.
"""

_REPLY_HEADROOM = 64 * 1024
"""Bytes reserved out of `_CLIENT_REPLY_LIMIT` for everything but the samples.

See ``docs/design/transports.md``.
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
"""The encodings `audio_format` may name, read off the type the encoder switches on rather
than restated here.

See ``docs/design/transports.md``.
"""

_SERVER_OPTIONS: list[tuple[str, int]] = [
    # Keepalive. Detect a peer that vanished without closing: without these a
    # half-open connection holds its worker until the process restarts. (The
    # engine it cannot hold, the stream's producer frees that at
    # `_MAX_STREAM_S`, but a worker is still a worker.)
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.min_ping_interval_without_data_ms", 10_000),
    # Bounds. A single client cannot occupy the whole pool by opening streams.
    ("grpc.max_concurrent_streams", 16),
    # grpc's default accepts a 4 MB request.
    ("grpc.max_receive_message_length", 256 * 1024),
]
"""Every channel option this server sets, keepalive and bounds both.

Named for what the list holds rather than for its first group: the last two
entries cap concurrency and request size and are not keepalive at all.
"""


_ENGINE_LOCKS: dict[int, threading.Lock] = {}
_ENGINE_LOCKS_GUARD = threading.Lock()


def _lock_for(engine: Engine) -> threading.Lock:
    """The single-flight lock for *this* engine, shared by every server over it.

    See ``docs/design/transports.md``.
    """
    with _ENGINE_LOCKS_GUARD:
        key = id(engine)
        lock = _ENGINE_LOCKS.get(key)
        if lock is None:
            lock = _ENGINE_LOCKS[key] = threading.Lock()
            weakref.finalize(engine, _ENGINE_LOCKS.pop, key, None)
        return lock


def _load_grpc() -> Any:
    try:
        import grpc
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise ModuleNotFoundError(_MISSING_EXTRA) from exc
    return grpc


def _load_stubs() -> tuple[Any, Any]:
    try:
        from ..proto import loudkit_pb2, loudkit_pb2_grpc
    except ModuleNotFoundError as exc:
        name = exc.name or ""
        missing_protobuf = name == "google" or name.startswith("google.")
        raise ModuleNotFoundError(
            _MISSING_PROTOBUF if missing_protobuf else _MISSING_STUBS
        ) from exc
    return loudkit_pb2, loudkit_pb2_grpc


_PUBLIC_BIND_REFUSAL = (
    "refusing to bind {target}: the gRPC transport has no authentication. "
    "Put it behind something that does, or serve HTTP (`loudkit serve` "
    "without `--grpc`), which requires a bearer token for a public bind."
)
"""One wording for the one rule, wherever a bind is decided."""


def _bind_is_local(target: str) -> bool:
    """Whether a gRPC bind target reaches no further than this machine.

    A Unix socket is scoped by the filesystem and has no network to be reached
    over. Everything else is a host and a port, decided by the function the
    HTTP guard and both ``serve`` entry points ask, so all of them mean the
    same thing by loopback.
    """
    if target.startswith(("unix:", "unix-abstract:")):
        return True
    return _host_is_loopback(target)


def _pin_to_loopback(server: Any) -> None:
    """Make a built server refuse a public bind, the way ``serve`` does.

    :func:`build_server` is exported and hands back a server nobody has bound
    yet, so the rule cannot live only in ``serve``: an embedder who starts one
    themselves never runs a line of it, and this transport has no
    authentication behind the port. Wrapped rather than checked once, because
    the address is not known until the caller supplies it.
    """
    bind = server.add_insecure_port

    def add_insecure_port(target: str) -> int:
        if not _bind_is_local(target):
            raise ValueError(_PUBLIC_BIND_REFUSAL.format(target=target))
        return int(bind(target))

    server.add_insecure_port = add_insecure_port


def build_server(  # noqa: PLR0915 - one server: its bounds and its four methods
    engine: Engine,
    voices: VoiceLibrary,
    *,
    max_workers: int = 8,
    warm: bool = False,
) -> Any:
    """A configured ``grpc.Server``, not yet started or bound.

    ``max_workers`` bounds gRPC's own pool, which is a queue depth rather than a
    parallelism setting: synthesis serialises behind the engine lock whatever
    this is, and the pool only decides how many callers may be waiting inside
    the process before the transport starts refusing.

    ``warm`` pays the first render's extra cost here, on the first voice in the
    library, before the server can be started. Off by default for the same
    reason as the HTTP app's: building a server is not by itself a decision to
    spend seconds on the GPU. :func:`serve` turns it on.
    """
    grpc = _load_grpc()
    pb2, pb2_grpc = _load_stubs()

    # After the extras are resolved: a missing one should fail before a render,
    # not after it.
    if warm:
        warm_engine(engine, voices)

    # Same reason as the HTTP server's single-flight slot: the engine holds
    # mutable decoder state and a CUDA graph capture is not reentrant. Owned
    # by the engine rather than by this server, see `_lock_for`.
    synth_lock = _lock_for(engine)

    # How many callers may be *waiting* for the engine, as opposed to holding it.
    max_queued = _queue_depth_for(max_workers)
    queued = 0
    queue_lock = threading.Lock()
    # When the engine was taken, or None when it is free. Read by `Describe`,
    # which does not take the lock, a held-engine report that waits for the
    # engine reports nothing at the only moment it is wanted. A list because it
    # is written from the request threads and closed over, and `nonlocal` on a
    # float would be one more name to keep in step across two release sites.
    held_since: list[float | None] = [None]

    def _take_engine(context: Any) -> bool:
        """Acquire the engine, or refuse. Never queues past a bound.

        See ``docs/design/transports.md``.
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
                    # give it straight back, and say nothing, there is no RPC
                    # left to carry a status.
                    synth_lock.release()
                    return False
                held_since[0] = time.monotonic()
                return True
            if not context.is_active():
                # The deadline (or a cancel) ended the RPC during the wait;
                # grpc has already answered DEADLINE_EXCEEDED or CANCELLED.
                return False
            # Whose bound ran out decides both the wording and the status.
            # `wait` is the smaller of `_MAX_WAIT_S` and the caller's own
            # `time_remaining()`, so when the caller's is smaller the thing
            # that ran out is the caller's deadline, not the server's patience,
            # and RESOURCE_EXHAUSTED would tell it to back off from congestion
            # it never met. `.3g` rather than `.0f`, which rounds a half-second
            # deadline to "waited 0s" -- a refusal with no wait in it at all --
            # while still spelling thirty seconds as 30. `busy` either way:
            # what it waited for was the engine, and the engine was.
            theirs = remaining is not None and remaining <= _MAX_WAIT_S
            _fail(
                context,
                grpc.StatusCode.DEADLINE_EXCEEDED
                if theirs
                else grpc.StatusCode.RESOURCE_EXHAUSTED,
                (
                    f"the engine was still busy when this call's {wait:.3g}s deadline ran out"
                    if theirs
                    else f"waited {wait:.3g}s for the engine and never got it"
                )
                + "; another synthesis is still running",
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
        coarser than the catalog, INVALID_ARGUMENT covers a dozen refusals -
        which is exactly why the metadata exists.
        """
        context.set_code(code)
        context.set_details(message)
        context.set_trailing_metadata((("loudkit-error-code", error_code),))

    def _resolve(  # one refusal per line, each naming its own condition
        request: Any, context: Any
    ) -> Any | None:
        """The voice, or ``None`` with the status already set on ``context``."""
        refusal = over_cap(request.text, request.previous_tokens)
        if refusal is not None:
            _fail(context, grpc.StatusCode.INVALID_ARGUMENT, refusal)
            return None
        # Speed and format, before the engine rather than inside it. Presence,
        # not truthiness: an unmentioned `speed` has to mean 1.0, and an
        # explicit 0.0 is a value outside the range that the other two doors
        # refuse by name.
        if request.HasField("speed") and not MIN_SPEED <= request.speed <= MAX_SPEED:
            _fail(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                f"speed {request.speed} is outside [{MIN_SPEED}, {MAX_SPEED}]",
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
        except ValueError as exc:  # a path, an empty string, not a voice *name*
            _fail(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                str(exc),
                error_code=error_code(exc),
            )
        return None

    def _kwargs(request: Any) -> dict[str, Any]:
        # `speed` reads its presence bit, because 0.0 is a value a caller can
        # mean to send and the answer to it is a refusal, not a default. The
        # rest are proto3 scalars whose zero is genuinely "unset": "" is not a
        # language, a format or a continuation.
        return {
            "seed": int(request.seed),
            "language": request.language or None,
            "speed": _speed(request),
            "previous_tokens": list(request.previous_tokens) or None,
            "audio_format": request.audio_format or "wav",
        }

    def _speed(request: Any) -> float:
        """The playback speed asked for, or the bypass when none was."""
        return float(request.speed) if request.HasField("speed") else 1.0

    def _refuse_oversize_reply(request: Any, context: Any, voice: Any) -> bool:
        """True when the unary reply would not fit a default client, said early.

        See ``docs/design/transports.md``.
        """
        speed = _speed(request)
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

        See ``docs/design/transports.md``.
        """
        try:
            chunks.close()
        except BaseException as exc:  # the verdict is the same for all
            _LOG.exception("could not reclaim the engine after a stream")
            engine._wedge(  # one package, one single-flight engine
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
            # One flag for every way this call ends early, exactly as the stream wires
            # it: gRPC runs the callback when the RPC terminates, which is what a client
            # cancelling or disconnecting looks like from here.
            cancelled = threading.Event()
            if context.add_callback(cancelled.set) is False:
                # The RPC is already over: grpc keeps no
                # callback list for a terminated call. `is False` rather than
                # `not`, so an implementation that returns nothing is read as
                # "registered" rather than as "cancel everything".
                cancelled.set()
            if not _take_engine(context):
                return pb2.SynthesizeResponse()
            try:
                rendered = render_bytes(
                    engine,
                    request.text,
                    voice,
                    # Presence, not truthiness: an unmentioned `long_form`
                    # has to mean the library's default, not proto3's.
                    long_form=(request.long_form if request.HasField("long_form") else True),
                    should_cancel=cancelled.is_set,
                    **_kwargs(request),
                )
            except CancelledError:
                # The client left. The RPC is already closed, so there is
                # no status to send and nothing to send it to.
                return pb2.SynthesizeResponse()
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
            except ValueError:
                # A bare `ValueError` out of the renderer is a defect, not a
                # refusal: `LoudkitError` does not derive from it, so the only
                # exceptions this clause can see are unclassified ones. Its
                # message may hold a checkpoint path, so it goes to the
                # operator's log and never to the caller.
                _LOG.exception("synthesis failed")
                _fail(
                    context,
                    grpc.StatusCode.INTERNAL,
                    "internal error",
                    error_code="server_fault",
                )
                return pb2.SynthesizeResponse()
            except Exception:
                if cancelled.is_set():
                    # The cancel starves the render mid-pass, and the unwound call stack
                    # can surface as almost anything; a cancelled call answers empty.
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
            # The streaming subset, which `_resolve` cannot check because the
            # unary RPC accepts every format. `proto/loudkit.proto` says this
            # contract mirrors the HTTP one field for field and that a
            # difference is a bug in that file: `/v1/synthesize/stream`
            # refuses ogg for a reason that is about the container, not the
            # door. An Ogg bitstream's state spans the whole stream, so a
            # per-chunk container names bytes that are not what they say.
            audio_format = request.audio_format or "wav"
            if audio_format not in _STREAMABLE:
                _fail(
                    context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"audio_format {audio_format!r} cannot be streamed: a container "
                    "is one continuous stream, not one payload per chunk. Streamable "
                    f"formats are {', '.join(sorted(_STREAMABLE))}; use Synthesize "
                    "for the rest.",
                )
                return
            fingerprint = engine.algorithm.fingerprint()
            sample_rate = engine.algorithm.sample_rate
            # The lock spans the whole stream, not each chunk: the chunks of one
            # passage share the engine's carry between them, so letting a second
            # caller in mid-passage would interleave two readings.
            if not _take_engine(context):
                return
            # One flag for every way this stream ends early, because the engine
            # already takes one: `Engine.stream` polls `should_cancel` on every
            # decode step, so setting it stops a render inside one forward pass
            # instead of at the next chunk boundary, the same wiring the HTTP
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

                See ``docs/design/transports.md``.
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
                    if not isinstance(exc, LoudkitError):
                        # A defect here; the detail goes to the operator's log
                        # (see `_LOG`), and the writer answers INTERNAL.
                        # `LoudkitError` is the whole line, as in `Synthesize`:
                        # anything else out of the renderer is unclassified, so
                        # it is logged rather than handed to the caller.
                        _LOG.exception("synthesis stream failed")
                finally:
                    expiry.cancel()
                    # Reclaim before release, see `_reclaim`. Both on this
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

            # The passage's tail so far, rebuilt chunk by chunk.
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
                    # The synthesis surface's own rule rather than a trailing
                    # slice here, so the fold ends where a pair does under
                    # `fusion_mtp2`, as one chunk's own continuation does.
                    tail = fold_continuation(engine, tail, tuple(item.continuation))
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
                    if isinstance(exc, LoudkitError):
                        # `LoudkitError` is the line, the same split as
                        # `Synthesize` and the other two doors, arriving over
                        # the queue instead of the call stack: a named refusal
                        # (unsupported language included) is the caller's
                        # fault and carries its own message, and everything
                        # else is a defect whose message may hold a checkpoint
                        # path and so is answered INTERNAL.
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
                # Every exit above is a `return` out of a generator, and a generator
                # abandoned by the framework is closed rather than resumed, so the flag is
                # set on every exit and the producer joined.
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
                # Falls back to 0 at the end of every render, the abandoned ones
                # included: a cancel, an expired deadline, the stream cap and a peer.
                engine_held_seconds=0.0 if started is None else time.monotonic() - started,
            )

        def ListVoices(self, request: Any, context: Any) -> Any:  # noqa: N802, ARG002
            return pb2.ListVoicesResponse(voices=voices.names())

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=_SERVER_OPTIONS,
    )
    pb2_grpc.add_SpeechServicer_to_server(Speech(), server)
    _pin_to_loopback(server)
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

    See ``docs/design/transports.md``.
    """
    from .. import load

    if not _host_is_loopback(host):
        # Said here as well as by the server the builder pins, so a bad host
        # is refused before an engine is loaded rather than after it.
        raise ValueError(_PUBLIC_BIND_REFUSAL.format(target=host))

    # The override below reads the manifest of the snapshot this returns.
    ckpt, library = open_release(checkpoint, voices, device)
    algorithm = algorithm_override(ckpt, first_chunk_tokens)
    engine = load(str(ckpt), device=device, algorithm=algorithm)
    # warm: the first render's extra cost is startup's, not the first caller's.
    server = build_server(engine, library, warm=True)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"loudkit gRPC on {host}:{port}  {engine.describe()}")
    print(f"voices: {', '.join(library.names()) or 'none in ' + str(library.root)}")
    server.wait_for_termination()
