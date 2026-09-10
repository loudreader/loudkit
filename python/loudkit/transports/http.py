"""A local synthesis server, because loading the model is the expensive part.

Holds one warm engine and answers ``/v1/voices``, ``/v1/synthesize``,
``/v1/synthesize/stream`` (Server-Sent Events) and OpenAI's
``/v1/audio/speech`` against it. It has no synthesis path of its own: every
route calls :func:`loudkit.synthesis.render_bytes` or its streaming twin and
returns what comes back. What it refuses and how it holds the engine live in
:mod:`.limits`; the wire shapes live in :mod:`.schemas`.
See ``docs/design/transports.md``.
"""

from __future__ import annotations

import logging
import secrets
import sys
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..engine import Engine, validate_speech_tokens
from ..errors import CancelledError, UnsupportedLanguageError, VoiceNotFoundError, error_code
from ..models.timestretch import MAX_SPEED, MIN_SPEED
from ..synthesis import (
    Rendered,
    VoiceLibrary,
    fold_continuation,
    render_bytes,
    render_stream_chunks,
    warm_engine,
)
from .limits import (
    _API_PREFIX,
    _MAX_QUEUED,
    _SLOW_RENDER_S,
    EngineBusyError,
    EngineSlot,
    _Guard,
    _host_is_loopback,
    _token_fault,
    stream_lease,
)
from .resolve import algorithm_override, open_release
from .schemas import (
    _CODE_BY_STATUS,
    _MISSING_EXTRA,
    _OPENAI_FORMATS,
    _OPENAI_UNSUPPORTED,
    _STREAMABLE,
    OpenAISpeechRequest,
    SpeakRequest,
    _audio_headers,
    _chunk_event,
    _done_event,
    _error_code,
    _error_kind,
    _first_message,
    _json_safe,
    _openai_error,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

try:
    from fastapi import Request
except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
    raise ModuleNotFoundError(_MISSING_EXTRA) from exc

# The two this transport owns. `render_bytes`, `render_stream_chunks` and
# `VoiceLibrary` are `synthesis`' names and were listed here too, which invited
# importing the facade through a door.
__all__ = ["build_app", "serve"]

_LOG = logging.getLogger("loudkit.transports.http")
"""Where a defect's detail goes. The caller gets the kind, which is all it can
act on; a path or a tensor shape belongs in the operator's log."""

_DISCONNECT_POLL_S = 0.05
"""How often a request re-asks whether its client is still there. The poll
runs on the event loop while the forward pass runs in a worker thread, which
is what lets a cancellation land mid-chunk."""


def build_app(  # noqa: PLR0915 - routes + a per-route lock; linear and explicit
    engine: Engine,
    voices: VoiceLibrary,
    *,
    token: str | None = None,
    allow_public: bool = False,
    warm: bool = False,
) -> FastAPI:
    """Wire the routes onto an engine.

    Separated from :func:`serve` so tests can exercise the app without binding
    a port. The token and public-bind rules are checked here rather than only
    in :func:`serve`, because an embedder wiring this app into their own ASGI
    stack never runs a line of ``serve``.

    ``warm`` pays the first render's extra cost here, before the app can be
    served, on the first voice in the library. Off by default: this returns an
    app, and building one is not by itself a decision to spend seconds on the
    GPU. :func:`serve` turns it on.
    """
    if allow_public and token is None:
        raise ValueError(
            "build_app(allow_public=True) needs a token: without one this app "
            "has no authentication, no Host pin and no rate limit, and anyone "
            "who can reach the port can synthesise in every voice on this "
            "machine. Pass token=secrets.token_urlsafe(32) (and keep it out of "
            "your logs), or drop allow_public for a loopback-only app."
        )
    fault = _token_fault(token)
    if fault is not None:
        raise ValueError(f"build_app(token=...) refused: {fault}")
    try:
        import anyio
        from fastapi import FastAPI, HTTPException
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse, Response, StreamingResponse
        from starlette.exceptions import HTTPException as StarletteHTTPException
        from starlette.types import Receive, Scope, Send
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise ModuleNotFoundError(_MISSING_EXTRA) from exc

    # After the extras are resolved: a missing one should fail before a render,
    # not after it.
    if warm:
        warm_engine(engine, voices)

    app = FastAPI(title="loudkit", version=_version())

    @app.exception_handler(StarletteHTTPException)
    async def _with_code(_request: Any, exc: StarletteHTTPException) -> JSONResponse:
        """Every error answer names its condition, not just its status.

        Registered for Starlette's class, not FastAPI's. Starlette dispatches
        on the raised exception's own MRO, and its router raises its own
        `HTTPException` -- which FastAPI's subclasses, so it does not match a
        handler registered for the subclass. A 404 and a 405 therefore answered
        with `detail` and no `code`, on a server that promises a `code` in
        every JSON error body. The base class covers both, and the routes'
        own `_refuse` raises the subclass, which is one of them.
        """
        code = getattr(exc, "loudkit_code", None) or _CODE_BY_STATUS.get(
            exc.status_code, "server_fault"
        )
        return JSONResponse(
            {"detail": exc.detail, "code": code},
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_with_code(_request: Any, exc: RequestValidationError) -> JSONResponse:
        """Schema violations speak the same vocabulary as every other refusal."""
        return JSONResponse(
            {"detail": _json_safe(exc.errors()), "code": "invalid_request"},
            status_code=422,
        )

    @app.exception_handler(EngineBusyError)
    async def _busy(_request: Any, exc: EngineBusyError) -> JSONResponse:
        return JSONResponse(
            {"detail": exc.detail, "code": _CODE_BY_STATUS[503]},
            status_code=503,
            headers={"Retry-After": str(exc.retry_after)},
        )

    def _refuse(
        status: int, exc: BaseException | str, *, code: str | None = None
    ) -> HTTPException:
        """An HTTPException that remembers which condition refused. A plain
        string and an explicit ``code`` are for a defect, whose message must
        not reach the caller."""
        if code is None:
            # Decided here, rather than left empty for `_with_code`'s `or` to
            # repair below. A string detail carries no exception to read a
            # catalog word off, so the status is what names it, and the empty
            # state in between existed only to be fixed up in another function.
            code = (
                _CODE_BY_STATUS.get(status, "server_fault")
                if isinstance(exc, str)
                else error_code(exc)
            )
        http = HTTPException(status_code=status, detail=str(exc))
        http.loudkit_code = code  # type: ignore[attr-defined]
        return http

    app.add_middleware(_Guard, token=token, allow_public=allow_public)

    # The engine is single-flight: it holds mutable decoder state and a CUDA
    # graph capture is not reentrant. Both routes take the slot on the loop,
    # before any worker thread, so a waiter never fills the thread pool.
    _engine_slot = EngineSlot(max_queued=_MAX_QUEUED)

    class _LeasedStream(StreamingResponse):
        """A stream that gives the engine slot back when the *response* is over.

        The body generator is not on every path out of a response: Starlette
        sends ``http.response.start`` before it pulls the first item, and a
        socket that dies there leaves the generator never started. ``__call__``
        is the one frame that exists on all of them, see
        :func:`~loudkit.transports.limits.stream_lease`.
        """

        def __init__(
            self,
            content: AsyncIterator[str],
            release: Callable[[], Awaitable[None]],
            *,
            media_type: str,
            headers: dict[str, str],
        ) -> None:
            super().__init__(content, media_type=media_type, headers=headers)
            self._release = release

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            async with stream_lease(self._release):
                await super().__call__(scope, receive, send)

    @app.get("/health", response_model=None)
    def health() -> Response:
        """The resolved algorithm and execution, 503 once the engine has
        wedged, and 503 while a render has held it longer than
        ``_SLOW_RENDER_S``."""
        body: dict[str, object] = {
            "status": "ok",
            "algorithm": engine.algorithm.describe(),
            "execution": engine.execution.describe(),
            "fingerprint": engine.algorithm.fingerprint(),
            "voices": voices.names(),
        }
        wedged = engine.wedged
        if wedged is not None:
            # Before the slow-render check, and terminal. A slow render is a
            # server that will answer; a wedged one answers every synthesis
            # with a fault and cannot be reclaimed, and reporting `ok` for it
            # keeps a load balancer routing to a process only a restart can
            # fix. No `Retry-After`: waiting is not the remedy.
            body["status"] = "wedged"
            body["detail"] = wedged
            return JSONResponse(body, status_code=503)
        started = _engine_slot.started_at
        if started is not None:
            stuck_for = time.monotonic() - started
            if stuck_for > _SLOW_RENDER_S:
                body["status"] = "stuck"
                body["synthesis_age_seconds"] = round(stuck_for, 1)
                return JSONResponse(body, status_code=503, headers={"Retry-After": "30"})
        return JSONResponse(body)

    @app.get(f"{_API_PREFIX}/voices")
    def list_voices() -> dict[str, list[str]]:
        return {"voices": voices.names()}

    @app.post(f"{_API_PREFIX}/synthesize", response_model=None)
    async def synthesize(  # noqa: PLR0915 - one linear route; the ladder is the contract
        req: SpeakRequest, request: Request
    ) -> Response:
        """Return the audio. Same text, voice and seed give the same bytes."""
        try:
            # Off the loop: a cold profile is a whole-file read and a hash.
            voice = await anyio.to_thread.run_sync(voices.load, req.voice)
        except VoiceNotFoundError as exc:
            raise _refuse(404, exc) from exc
        except ValueError as exc:
            raise _refuse(400, exc) from exc

        cancelled = threading.Event()

        def _render() -> Rendered:
            return render_bytes(
                engine,
                req.text,
                voice,
                seed=req.seed,
                language=req.language,
                long_form=req.long_form,
                speed=req.speed,
                previous_tokens=req.previous_tokens,
                audio_format=req.format,
                should_cancel=cancelled.is_set,
            )

        rendered_ok = False

        async def watch_disconnect() -> None:
            """Flip ``cancelled`` as soon as the client goes away, or we do.

            The ``finally`` covers the watcher being cancelled with the
            response task while ``to_thread.run_sync`` keeps rendering;
            ``rendered_ok`` tells "the render came back" from "torn down".
            """
            try:
                while not await request.is_disconnected():
                    await anyio.sleep(_DISCONNECT_POLL_S)
            finally:
                if not rendered_ok:
                    cancelled.set()

        try:
            rendered: Rendered | None = None
            render_error: BaseException | None = None
            async with _engine_slot, anyio.create_task_group() as tg:
                tg.start_soon(watch_disconnect)
                try:
                    rendered = await anyio.to_thread.run_sync(_render)
                except Exception as exc:  # re-raised below, unchanged
                    # Carried out by hand: an exception escaping the group
                    # arrives wrapped, and the ladder below would miss it.
                    render_error = exc
                finally:
                    rendered_ok = True
                    tg.cancel_scope.cancel()
            if render_error is not None:
                raise render_error
        except CancelledError:
            # The client is gone; 499 is nginx's spelling of "closed the request".
            return Response(status_code=499)
        except UnsupportedLanguageError as exc:
            raise _refuse(400, exc) from exc
        except ValueError as exc:
            # Classified, not assumed: a bare ValueError from the renderer is
            # a defect, and its message is not the caller's to read.
            if _error_kind(exc) != "bad_request":
                _LOG.exception("synthesis failed")
                raise _refuse(500, "internal error", code="server_fault") from exc
            raise _refuse(422, exc) from exc
        except (EngineBusyError, HTTPException):
            # Answers of their own, both with a `code` already: the busy one
            # through this app's handler for it, the refusal through
            # `_with_code`. Named before the clause below so that clause can
            # end the ladder without swallowing them.
            raise
        except Exception as exc:
            # The clause that closes the ladder: any other defect is logged
            # for the operator and answered with the same `server_fault` body
            # as the clause above, never echoed to the caller. Without it a
            # defect that is not a `ValueError` reaches Starlette's own
            # handler, which answers `text/plain` "Internal Server Error" with
            # no `code` and no log line, for a server that promises a `code`
            # in every JSON error body.
            if cancelled.is_set():
                # A cancel starves the render mid-pass and the unwound stack
                # can surface as almost anything; the client is already gone.
                return Response(status_code=499)
            _LOG.exception("synthesis failed")
            raise _refuse(500, "internal error", code="server_fault") from exc

        assert rendered is not None  # the render returned, or raised above
        return Response(
            content=rendered.data,
            media_type=rendered.media_type,
            headers=_audio_headers(
                rendered,
                sample_rate=engine.algorithm.sample_rate,
                fingerprint=engine.algorithm.fingerprint(),
            ),
        )

    @app.post(f"{_API_PREFIX}/audio/speech", response_model=None)
    async def openai_speech(req: OpenAISpeechRequest, request: Request) -> Response:
        """OpenAI's speech endpoint: a translation onto ``/v1/synthesize``, so
        the bytes are the bytes that route returns for the same request."""
        if req.response_format in _OPENAI_UNSUPPORTED:
            return _openai_error(
                400,
                f"response_format {req.response_format!r} is not available from this "
                f"server; it can return {', '.join(sorted(_OPENAI_FORMATS))}",
                "invalid_request_error",
            )
        audio_format = _OPENAI_FORMATS.get(req.response_format)
        if audio_format is None:
            return _openai_error(
                400,
                f"unknown response_format {req.response_format!r}; this server can "
                f"return {', '.join(sorted(_OPENAI_FORMATS))}",
                "invalid_request_error",
            )
        # Checked here rather than by `SpeakRequest`: their range is wider,
        # and "valid OpenAI, unsupported here" is the actionable message.
        if not MIN_SPEED <= req.speed <= MAX_SPEED:
            return _openai_error(
                400,
                f"speed {req.speed} is outside this engine's range [{MIN_SPEED}, {MAX_SPEED}]",
                "invalid_request_error",
            )

        try:
            return await synthesize(
                SpeakRequest(
                    text=req.input,
                    voice=req.voice,
                    speed=req.speed,
                    format=audio_format,
                ),
                # The same connection, so their client hanging up stops the
                # render for the same reason ours does.
                request,
            )
        except EngineBusyError as exc:
            return _openai_error(503, exc.detail, "server_error")
        except HTTPException as exc:
            # Re-dressed: an OpenAI client shows `detail` to nobody.
            kind = "invalid_request_error" if exc.status_code < 500 else "server_error"
            return _openai_error(exc.status_code, str(exc.detail), kind)

    @app.post(f"{_API_PREFIX}/synthesize/stream", response_model=None)
    async def synthesize_stream(  # noqa: PLR0915 - one linear route; the branches are the contract
        req: SpeakRequest, request: Request
    ) -> StreamingResponse:
        """Stream a passage chunk by chunk as Server-Sent Events.

        Everything that can decide a status code happens here, before a byte
        of the response is written: once Starlette has sent 200, a failure
        can only travel as the ``done`` event.
        """
        if not req.text.strip():
            # The schema's `min_length=1` admits whitespace, and the engine
            # would refuse it after the response has begun, which is a spent
            # 200 for a request that is knowably empty at the door. Text the
            # funnel later empties (an emoji, a bare symbol) is not knowable
            # here and does still arrive as the terminal event's error.
            raise _refuse(422, "nothing to speak", code="invalid_request")

        try:
            validate_speech_tokens(
                req.previous_tokens, limit=engine.algorithm.start_speech_token
            )
        except ValueError as exc:
            raise _refuse(422, exc) from exc

        if req.format not in _STREAMABLE:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"format {req.format!r} cannot be streamed: a container is one "
                    "continuous stream, not one payload per chunk. Streamable "
                    f"formats are {', '.join(sorted(_STREAMABLE))}; use "
                    f"{_API_PREFIX}/synthesize for the rest."
                ),
            )

        try:
            voice = await anyio.to_thread.run_sync(voices.load, req.voice)
        except VoiceNotFoundError as exc:
            raise _refuse(404, exc) from exc
        except ValueError as exc:
            raise _refuse(400, exc) from exc

        # Read before the acquire, with everything else the response needs:
        # both are pure reads of the engine's config, and the rule below leaves
        # no room for one of them to raise.
        fingerprint = engine.algorithm.fingerprint()
        sample_rate = engine.algorithm.sample_rate

        # From here the slot is given back by `_return_the_slot`, wired to the
        # response object; nothing that can fail may sit between the two.
        await _engine_slot.acquire()

        # Set by the disconnect watcher and by the reclaim, read by the decode
        # loop in the worker thread.
        cancelled = threading.Event()

        def produce() -> Generator[Rendered, None, None]:
            yield from render_stream_chunks(
                engine,
                req.text,
                voice,
                seed=req.seed,
                language=req.language,
                speed=req.speed,
                previous_tokens=req.previous_tokens,
                audio_format=req.format,
                should_cancel=cancelled.is_set,
            )

        # Built here so it can be closed on every path, including the ones
        # where `events()` never runs a line.
        it = produce()

        def _reclaim_the_engine() -> None:
            """Close the generator, which joins the engine's own threads, or
            mark the engine wedged when that cannot be shown to have happened."""
            try:
                it.close()
            except BaseException as exc:  # the verdict is the same for all
                _LOG.exception("could not reclaim the engine after a stream")
                engine._wedge(  # one package, one single-flight engine
                    f"reclaiming an abandoned stream raised {type(exc).__name__}, "
                    "so nothing here can show the token generator and the "
                    "renderer were left idle"
                )

        released = False

        async def _return_the_slot() -> None:
            """Give the slot back once, however the stream ended: flag, then
            reclaim, then release, in that order, and shielded because the
            usual way here is the response task being cancelled."""
            nonlocal released
            if released:
                return
            released = True
            cancelled.set()
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(_reclaim_the_engine)
            _engine_slot.release()

        async def events() -> AsyncIterator[str]:  # one linear stream loop
            # True only between "this chunk's render returned" and the end of
            # the task group that rendered it.
            chunk_ready = False

            async def watch_disconnect() -> None:
                try:
                    while not await request.is_disconnected():
                        await anyio.sleep(_DISCONNECT_POLL_S)
                finally:
                    if not chunk_ready:
                        cancelled.set()

            _done = object()
            truncated = False
            continuation: tuple[int, ...] = ()

            def _next_chunk() -> object:
                # StopIteration through a thread becomes a RuntimeError.
                try:
                    return next(it)
                except StopIteration:
                    return _done

            while True:
                if cancelled.is_set() or await request.is_disconnected():
                    # Closing the generator is the reclaim's job, off the loop.
                    return
                failure: str | None = None
                kind: str | None = None
                code: str | None = None
                try:
                    chunk_ready = False
                    async with anyio.create_task_group() as tg:
                        tg.start_soon(watch_disconnect)
                        chunk = await anyio.to_thread.run_sync(_next_chunk)
                        chunk_ready = True
                        tg.cancel_scope.cancel()
                except Exception as exc:
                    kind = _error_kind(exc)
                    code = _error_code(exc)
                    if kind == "server_fault":
                        _LOG.exception("synthesis stream failed")
                        failure = "internal error"
                    else:
                        failure = _first_message(exc)
                if failure is not None:
                    yield _done_event(
                        fingerprint,
                        truncated,
                        continuation,
                        error=failure,
                        kind=kind,
                        code=code,
                    )
                    return
                if cancelled.is_set():
                    return
                if chunk is _done:
                    break
                rendered = cast(Rendered, chunk)
                truncated = truncated or rendered.hit_token_cap
                # Folded chunk by chunk, as the gRPC stream folds it, and by
                # the synthesis surface's own rule rather than a trailing
                # slice here: a final chunk shorter than the prefix carries
                # fewer ids than the passage's tail, and the tail is what a
                # chaining client hands back as `previous_tokens`.
                continuation = fold_continuation(
                    engine, continuation, tuple(rendered.continuation)
                )
                yield _chunk_event(rendered, sample_rate=sample_rate, fingerprint=fingerprint)
            # The aggregate, so a client that reads only `done` still learns
            # that some chunk was cut off.
            yield _done_event(fingerprint, truncated, continuation)

        return _LeasedStream(
            events(),
            _return_the_slot,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                # The same two facts the one-shot reply carries, and the same
                # two every chunk event carries. Always: raw frames carry no
                # header to read the rate from, so a `pcm16` client reading
                # only the response would have nowhere to read it.
                "X-Loudkit-Sample-Rate": str(sample_rate),
                "X-Loudkit-Fingerprint": fingerprint,
            },
        )

    return app


def serve(
    checkpoint: str | Path,
    *,
    voices: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    device: str | None = None,
    allow_public: bool = False,
    token: str | None = None,
    first_chunk_tokens: int | None = None,
) -> None:
    """Load the engine once and answer requests until interrupted.

    A non-loopback bind is refused without ``allow_public``, and then requires
    a bearer token, generated and printed to stderr if none is given. On
    loopback a token is ignored: the operating system is the boundary.
    ``first_chunk_tokens`` caps the first streamed chunk so audio starts
    sooner; it changes the algorithm fingerprint, so it is opt-in.
    """
    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise ModuleNotFoundError(_MISSING_EXTRA) from exc

    # Refused before the load: a configuration mistake should not wait for
    # the model to be reported.
    public = not _host_is_loopback(host)
    if public and not allow_public:
        raise SystemExit(
            f"refusing to bind {host}: this server has no authentication and "
            "anyone who can reach the port can synthesise in every voice on "
            "this machine. Pass --allow-public only on a network you control."
        )
    if public:
        fault = _token_fault(token)
        if fault is not None:
            raise SystemExit(f"refusing the token given for this public bind: {fault}")

    from .. import load

    ckpt, library = open_release(checkpoint, voices, device)
    algorithm = algorithm_override(ckpt, first_chunk_tokens)
    engine = load(str(ckpt), device=device, algorithm=algorithm)
    print(f"loudkit {_version()}  {engine.describe()}")
    print(f"voices: {', '.join(library.names()) or 'none in ' + str(library.root)}")
    if not public:
        # Documented as ignored on loopback, so it is.
        token = None
    else:
        if token is None:
            token = secrets.token_urlsafe(32)
            # stderr, not stdout: a service log is not where a credential goes.
            print(
                f"generated an access token for this public bind:\n  {token}",
                file=sys.stderr,
            )
        print(
            f"binding {host}: every request must carry "
            "'Authorization: Bearer <token>'. Synthesis in every voice on this "
            "machine is available to anyone holding it."
        )
        print(
            "warning: this server speaks plain HTTP; the token and the audio "
            "cross the network in clear. Terminate TLS in front of it, see "
            "docs/guides/04-server-and-agents.md.",
            file=sys.stderr,
        )

    uvicorn.run(
        # The Host pin stays on a loopback bind whatever the flag said: DNS
        # rebinding does not care what the operator intended.
        # warm: the first render's extra cost is startup's, not the first
        # caller's, and this is the last thing before the port opens.
        build_app(
            engine,
            library,
            token=token,
            allow_public=allow_public and public,
            warm=True,
        ),
        host=host,
        port=port,
        log_level="info",
    )


def _version() -> str:
    from .. import __version__

    return __version__
