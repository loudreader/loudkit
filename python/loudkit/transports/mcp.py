"""An MCP server, so any MCP-aware agent can speak in a cloned voice.

Three tools over stdio: ``list_voices``, ``synthesize`` and ``describe``. The
server holds no synthesis path of its own; ``synthesize`` resolves a voice
and calls :func:`~loudkit.synthesis.render_bytes`, behind the request caps
and the wait bound in :mod:`.limits` that HTTP and gRPC apply too. The reply
carries the audio as base64 with the facts the HTTP route puts in headers.
See ``docs/design/transports.md``.
"""

from __future__ import annotations

import base64
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast, get_args

from ..engine import Engine
from ..errors import LoudkitError, UnsupportedLanguageError, VoiceNotFoundError, error_code
from ..models.timestretch import MAX_SPEED, MIN_SPEED
from ..synthesis import AudioFormat, Rendered, VoiceLibrary, warm_engine
from ..voice import VoiceProfile
from .limits import _MAX_WAIT_S, EngineBusyError, ThreadQueue, _queue_depth_for, over_cap
from .resolve import open_release

_AUDIO_FORMATS: frozenset[str] = frozenset(get_args(AudioFormat))
"""What ``synthesize`` accepts as ``format``, derived from the one
:data:`~loudkit.synthesis.AudioFormat` rather than repeated, so this transport
cannot offer an encoding the synthesis surface does not have."""

__all__ = ["build_server", "run_stdio"]

_MISSING_EXTRA = 'the MCP server needs the "mcp" package.\n  pip install "loudkit[mcp]"'


def _load_mcp() -> Any:
    """Import the MCP SDK, with a name that says what to install.

    ``Any``: the SDK is an optional extra, so its server class has no type here.
    """
    try:
        from mcp.server.mcpserver import MCPServer
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise ModuleNotFoundError(_MISSING_EXTRA) from exc
    return MCPServer


def build_server(
    checkpoint: str,
    voices: str | Path | None = None,
    *,
    device: str | None = None,
    engine: Engine | None = None,
    warm: bool = False,
) -> Any:
    """Build an MCP server backed by an engine and a voice library.

    ``warm`` pays the first render's extra cost here, on the first voice in the
    library, whether the engine was loaded here or handed in. Off by default,
    like the other two transports' builders; :func:`run_stdio` turns it on.

    The return is ``Any`` because it is the MCP SDK's own server object, from an
    optional extra that mypy does not see; everything this module owns is typed.

    See ``docs/design/transports.md``.
    """
    mcp_server_cls = _load_mcp()
    # The SDK's own scheduler, so the render runs on a worker thread the
    # framework can abandon. Imported here, after `_load_mcp`, because it
    # arrives with the SDK: a build without the extra refuses above rather
    # than at an import this module could not have satisfied either.
    import anyio.to_thread

    from .. import __version__, load

    if engine is not None and voices is not None:
        # Both questions the checkpoint answers have been answered by the caller: the
        # engine is loaded and the voice directory was named. So nothing is
        # resolved, and a repo id here fetches nothing.
        library = VoiceLibrary(Path(voices))
    else:
        ckpt, library = open_release(checkpoint, voices, device)
    if engine is None:
        engine = load(str(ckpt), device=device)
    if warm:
        warm_engine(engine, library)

    # The engine is single-flight (same as the HTTP server): a CUDA graph
    # capture is not reentrant, and torch modules are not thread-safe. MCP tool
    # calls can be dispatched concurrently by the host, so synthesis must
    # serialise behind a lock.
    # The wait is `_MAX_WAIT_S` under its own name, the same bound as the HTTP
    # server's queue wait: a wedged render must not hold every tool call open
    # indefinitely.
    _synth_lock = threading.Lock()

    # The depth bound, which is the half of the wait the time bound cannot
    # stand in for. A waiter here holds a worker thread for as long as it
    # waits, so past a bound the queue puts every thread on the lock and the
    # tools that need no engine stop answering: `list_voices` is a directory
    # listing and must not wait behind a render. Counted on the loop and
    # refused there, before a thread is taken.
    _engine_queue = ThreadQueue()

    server = mcp_server_cls(
        name="loudkit",
        title="loudkit TTS",
        description="Local text-to-speech in any voice the engine can clone.",
        version=__version__,
        instructions=(
            "Same text, voice and seed give the same audio every time. "
            "Voices are files in the library directory, resolved by name."
        ),
    )

    @server.tool(  # type: ignore[untyped-decorator]
        title="List voices",
        description="Names of every voice profile the server can speak in.",
    )
    def list_voices() -> list[str]:
        return library.names()

    @server.tool(  # type: ignore[untyped-decorator]
        title="Synthesize speech",
        description=(
            "Turn text into speech in a named voice. Returns the audio as "
            "base64 plus the audio duration and token count. `format` is "
            '"wav" by default; "flac" is the same samples, losslessly, at '
            "about a quarter the size: worth asking for when the reply is "
            'saved to a file rather than played. "mp3" and "opus" are lossy '
            "and smaller still, for a reply sent on to a chat or a phone. "
            "Same text, voice and seed give the same audio, and the same bytes "
            "in every format but ogg and opus, whose container carries a random "
            "stream serial. Omit `language` to read the text in the "
            "voice's own language; pass one only to read text in a language the "
            "voice was not enrolled in. `speed` is playback speed in [0.5, 2.0] "
            "with the pitch preserved: 1.0, the default, is an exact bypass. "
            "To read a long text as several calls without an audible restart at "
            "each join, pass the previous reply's `continuation` list back as "
            "`previous_tokens`. "
            "Check `truncated`: when true the "
            "utterance hit the token cap and the speech is cut off mid-sentence. "
            "A refusal comes back as `error` with `error_kind` "
            '"bad_request": something about this call to fix, and `supported` '
            "or `available` listing what would have worked, and `code` from "
            "the same frozen catalog the HTTP and gRPC doors name the "
            "condition with."
        ),
    )
    async def synthesize(  # noqa: PLR0911 - each error kind is its own answer
        text: str,
        voice: str,
        seed: int = 0,
        language: str | None = None,
        speed: float = 1.0,
        previous_tokens: list[int] | None = None,
        format: str = "wav",  # noqa: A002 - the HTTP surface's name for the same choice
    ) -> dict[str, Any]:
        refusal = over_cap(text, previous_tokens)
        if refusal is not None:
            # The same caps and the same code as the HTTP and gRPC doors, so
            # the host cannot reach the engine with a request they refuse.
            return {"error": refusal, "error_kind": "bad_request", "code": "invalid_request"}
        if format not in _AUDIO_FORMATS:
            return {
                "error": f"unknown format {format!r}",
                "error_kind": "bad_request",
                "code": "invalid_request",
                "supported": sorted(_AUDIO_FORMATS),
            }
        if not MIN_SPEED <= speed <= MAX_SPEED:
            # The same bound the HTTP and gRPC doors check before the engine;
            # out of range is something about this call to fix, not a defect.
            return {
                "error": f"speed {speed} is outside [{MIN_SPEED}, {MAX_SPEED}]",
                "error_kind": "bad_request",
                "code": "invalid_request",
            }
        try:
            profile = library.load(voice)
        except VoiceNotFoundError as exc:
            return {
                "error": str(exc),
                "error_kind": "bad_request",
                "code": error_code(exc),
                "available": exc.available,
            }
        except ValueError as exc:  # not a voice *name*, a path, an empty string
            return {"error": str(exc), "error_kind": "bad_request", "code": error_code(exc)}
        # One flag for the one way this call ends early, the wiring the HTTP
        # and gRPC doors already have: `render_bytes` polls it on every decode
        # step, so a cancel stops the render inside a forward pass instead of
        # at the end of the passage.
        cancelled = threading.Event()

        def take_engine_and_render() -> Rendered:
            """The blocking half, off the event loop.

            The acquire is here rather than beside the caps because it blocks
            for up to ``_MAX_WAIT_S``, and a loop stalled that long cannot
            deliver the very notification that would cancel this call.
            """
            if not _synth_lock.acquire(timeout=_MAX_WAIT_S):
                raise EngineBusyError(
                    "engine busy: another synthesis is holding the lock "
                    f"(waited {_MAX_WAIT_S:.0f}s)",
                    retry_after=30,
                )
            try:
                return _render(
                    engine,
                    profile,
                    text,
                    seed=seed,
                    language=language,
                    speed=speed,
                    previous_tokens=previous_tokens,
                    audio_format=cast(AudioFormat, format),
                    should_cancel=cancelled.is_set,
                )
            finally:
                _synth_lock.release()

        # The pool a waiter would hold a thread from, read here because the
        # limiter belongs to the running loop. Half of it may queue; the rest
        # answers `list_voices` and `describe` while synthesis is saturated.
        depth = _queue_depth_for(anyio.to_thread.current_default_thread_limiter().total_tokens)
        try:
            # `abandon_on_cancel`: a cancel returns here at once and leaves the
            # render on its own thread, which the `finally` below stops.
            # Waiting for it instead would sit out the whole passage being
            # cancelled, holding the lock against every other caller.
            with _engine_queue.admit(depth):
                rendered = await anyio.to_thread.run_sync(
                    take_engine_and_render, abandon_on_cancel=True
                )
        except EngineBusyError as exc:
            # Both bounds land here, the depth one from `admit` before a thread
            # is taken and the wait one from the render thread, and both are
            # the same answer with a different reason. `busy` is what HTTP
            # answers a full queue with, out of `_CODE_BY_STATUS[503]`, and
            # what gRPC sends beside RESOURCE_EXHAUSTED.
            return {"error": exc.detail, "error_kind": "busy", "code": "busy"}
        except UnsupportedLanguageError as exc:
            # UnsupportedLanguageError, not the builtin NotImplementedError.
            return {
                "error": str(exc),
                "error_kind": "bad_request",
                "code": error_code(exc),
                "supported": list(exc.supported),
            }
        except ValueError as exc:
            # `LoudkitError` is the line, as on the other two transports: a
            # bare `ValueError` out of the renderer is a defect, and returning
            # its message as `bad_request` both hands the agent whatever it
            # says and tells it to retry a request that was never wrong.
            if not isinstance(exc, LoudkitError):
                raise
            return {"error": str(exc), "error_kind": "bad_request", "code": error_code(exc)}
        finally:
            # Set on every exit. After a render that finished this is a no-op;
            # on the cancel path the framework unwinds this coroutine while the
            # render is still on its thread, and this is what ends it.
            cancelled.set()
        # Deliberately not caught: a bare NotImplementedError is a defect here,
        # not a question about the call. Swallowing it into the same
        # {"error": ...} shape told the agent its request was wrong and hid a
        # broken build from whoever could fix it. It escapes to the MCP
        # framework's own failure path, which is what a server fault looks like.
        return {
            "audio": base64.b64encode(rendered.data).decode(),
            # Beside the bytes, as everywhere else: a field of base64 does not
            # say what it decodes to, and the agent writing it to a file needs
            # the extension and the player needs the type.
            "format": format,
            "media_type": rendered.media_type,
            "duration": round(rendered.duration, 4),
            "tokens": rendered.n_tokens,
            "sample_rate": engine.algorithm.sample_rate,
            "fingerprint": engine.algorithm.fingerprint(),
            # Not an error: the audio is real, it is just incomplete. An agent
            # that cannot see this reads a cut-off sentence as a finished one.
            "truncated": rendered.hit_token_cap,
            # The tail to send back as `previous_tokens` next call, so a
            # multi-part reading does not restart its pitch contour at every
            # join. The tail rather than every token id: it is all the engine
            # uses, and a few hundred integers in a tool result is context an
            # agent pays for and cannot act on.
            "continuation": list(rendered.continuation),
        }

    @server.tool(  # type: ignore[untyped-decorator]
        title="Describe the engine",
        description=(
            "The resolved algorithm and execution configuration. Log this "
            "whenever a synthesis surprises you: it is the line that tells you "
            "which mode was active."
        ),
    )
    def describe() -> dict[str, str]:
        return {
            "algorithm": engine.algorithm.describe(),
            "execution": engine.execution.describe(),
            "fingerprint": engine.algorithm.fingerprint(),
            "device": engine.execution.resolved_device(),
        }

    return server


def _render(
    engine: Engine,
    profile: VoiceProfile,
    text: str,
    *,
    seed: int,
    language: str | None,
    speed: float = 1.0,
    previous_tokens: list[int] | None = None,
    audio_format: AudioFormat = "wav",
    should_cancel: Callable[[], bool] | None = None,
) -> Rendered:
    """The one place audio is made, shared with the HTTP server.

    Imported here rather than at module top so a missing ``soundfile`` (the
    ``audio`` extra) fails at tool-call time with its own message, not at
    import of a server the caller may only be inspecting.

    ``should_cancel`` reaches the decode loop, as it does from the HTTP and
    gRPC doors: a cancelled call that ran to the end would render a passage
    nobody will hear while holding ``_synth_lock`` against every other caller.
    """
    from ..synthesis import render_bytes

    return render_bytes(
        engine,
        text,
        profile,
        seed=seed,
        language=language,
        speed=speed,
        previous_tokens=previous_tokens,
        audio_format=audio_format,
        should_cancel=should_cancel,
    )


def run_stdio(
    checkpoint: str, voices: str | Path | None = None, *, device: str | None = None
) -> None:
    """Run the MCP server over stdio; what ``loudkit serve --mcp`` calls."""
    # warm: the first render's extra cost is startup's, not the first tool call's.
    server = build_server(checkpoint, voices, device=device, warm=True)
    server.run(transport="stdio")
