"""A local synthesis server, because loading the model is the expensive part.

The checkpoint is 747 MB and takes a few seconds to load. A CLI that pays
that per invocation is fine for a batch script and useless for anything
interactive, so the server exists to hold one warm :class:`~loudkit.engine.Engine`
and answer requests against it.

**It has no synthesis path of its own.** Every route builds nothing and decides
nothing; it calls the engine and returns what comes back. That is deliberate: a
second path is a second thing to keep in agreement, and this library exists
because two paths drifted. :func:`render_bytes` is the single place audio is
produced, and the test suite asserts its output is identical to calling the
engine directly.

Binds to localhost by default: it is a way to keep a model warm on your own
machine, not a service to expose. A non-loopback bind is **refused** unless
``--allow-public`` is passed, and then it **requires a bearer token** — one is
generated and printed if you do not supply it, because "on a network you
control" is a hope rather than a control, and anyone who can reach the port can
otherwise speak in every voice on the machine.

A loopback bind is not a private one, either: any page in a browser on this
machine can POST to it blind, so the write path refuses a request another
origin caused (``Sec-Fetch-Site``) and one that is not ``application/json``,
both before a route is reached.

Two other bounds exist for the same reason, and both are enforced rather than
documented: a request body is refused *before* it is read, at a bound derived
from the 10 000 character text cap (which pydantic applies only once the whole
body is already buffered), and at most 32 requests may queue for the
single-flight engine before the server answers 503 — an unbounded queue turns a
slow engine into unbounded memory with every client still holding a connection.

The API is under ``/v1``: ``/v1/voices``, ``/v1/synthesize``,
``/v1/synthesize/stream``. ``/health`` is not, because a liveness probe is
infrastructure and should not have to know which API version is current.

Install with ``pip install "loudkit[server]"``.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import secrets
import sys
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..errors import LoudkitError, UnsupportedLanguageError, VoiceNotFoundError, error_code
from ..models.timestretch import MAX_SPEED, MIN_SPEED

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

    from ..engine import Engine

from ..synthesis import (
    _MAX_PREVIOUS_TOKENS,
    _MAX_TEXT_LEN,
    _MAX_WAIT_S,
    AudioFormat,
    Rendered,
    VoiceLibrary,
    _first_exception,
    render_bytes,
    render_stream_chunks,
)

_LOG = logging.getLogger("loudkit.transports.http")
"""Where a defect's detail goes, since the client no longer gets it.

A stream's failures are delivered in the ``done`` event rather than a status
line, and only ``bad_request`` detail belongs there anyway: the caller asked
wrongly and needs to read why. For a ``server_fault`` it is a filesystem
path, a tensor shape or a stack frame's repr, handed to whoever can reach
the port. The detail belongs in the operator's log; the caller gets the
kind, which is all a caller can act on.
"""

_MISSING_EXTRA = (
    'the server needs fastapi, uvicorn and soundfile.\n  pip install "loudkit[server]"'
)

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
    raise ModuleNotFoundError(_MISSING_EXTRA) from exc

try:
    from fastapi import Request
except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
    raise ModuleNotFoundError(_MISSING_EXTRA) from exc

__all__ = ["serve", "build_app", "render_bytes", "render_stream_chunks", "VoiceLibrary"]


_API_PREFIX = "/v1"
"""Where the API lives. Every route that answers about *synthesis* is under it.

The version is in the path so a later shape change can be served beside this one
rather than instead of it — a client pinned to ``/v1`` keeps working while
``/v2`` exists. Added before the first release, so there are no aliases for the
unprefixed paths: nothing has ever spoken them.

``/health`` deliberately stays unversioned. It is infrastructure — a load
balancer, a container orchestrator or a person asking "is this the build I
meant" — and none of those track an API version. Versioning it would mean a
liveness probe has to be updated when the payload shape of synthesis changes,
which is exactly backwards.
"""


_MAX_BODY_BYTES = 12 * _MAX_TEXT_LEN + 4096
"""Largest request body the server will read at all.

``_MAX_TEXT_LEN`` is enforced by pydantic, which runs *after* Starlette has
already buffered the whole body in memory. This bound is checked against
``Content-Length`` before anything is read, and again while reading for a
chunked request that declares no length.

**Derived from the text cap rather than asserted to be larger than it.**
``json.dumps`` defaults to ``ensure_ascii=True`` — as does ``requests`` —
which spends 6 bytes per non-ASCII character (``\\uXXXX``) and 12 per astral
character, because a surrogate pair is two escapes. An in-cap request of
10 000 astral characters therefore encodes to 120 KB; twelve bytes per
character is the worst case, and the 4 KB covers the other fields.
"""

_MAX_QUEUED = 32
"""How many synthesis requests may be waiting for the single-flight engine.

The engine serialises, so concurrent requests queue — and an unbounded queue
turns a slow engine into unbounded memory and unbounded latency, with every
client still holding a connection open and waiting. Past this depth the server
answers 503 with ``Retry-After``, which is a truthful answer: it is busy, and
it will not get to you soon.
"""


_SLOW_RENDER_S = 120.0
"""When ``/health`` stops claiming the engine is fine.

A wedged synthesis cannot be preempted, so the one thing the server can still do
honestly is stop reporting ``ok`` while it is wedged. Past this, ``/health``
answers 503 with the age of the render that is stuck, which is what a load
balancer needs to take the instance out of rotation.
"""

_RATE_CAPACITY = 12
_RATE_REFILL_PER_S = 0.5
"""Token bucket for authenticated public binds: burst, then a steady trickle.

Only applied when the server binds somewhere other than loopback, which is the
same condition that forces a bearer token. On loopback the caller is already on
the machine and a limiter buys nothing. Twelve is a comfortable burst for a
client chunking a chapter; half a token per second is one synthesis every two
seconds sustained, which is faster than the engine can serve them anyway — the
bucket is there to stop a client that has stopped reading its own responses, not
to shape legitimate use.
"""


_OPENAI_FORMATS: dict[str, AudioFormat] = {
    # Their spelling on the left, ours on the right.
    "wav": "wav",
    "pcm": "pcm16",
    "flac": "flac",
    # Ours too, because a caller that knows it is talking to loudkit should not
    # have to look up which of its four formats OpenAI happens to have a name
    # for. `ogg` has no OpenAI spelling at all -- see `_OPENAI_UNSUPPORTED`.
    "pcm16": "pcm16",
    "ogg": "ogg",
}
"""OpenAI's `response_format` values this server can honour, mapped to ours."""

_OPENAI_UNSUPPORTED = ("mp3", "aac", "opus")
"""OpenAI formats this server will not produce, refused by name.

`mp3` and `aac` need an encoder this project does not ship. `opus` is the one
that looks available and is not: the `ogg` format here is Ogg **Vorbis**, which
shares `audio/ogg` with Ogg Opus and shares no bitstream with it, so answering
an `opus` request with it would hand back a container the client cannot decode
under a content type saying it can.
"""


_STREAMABLE: frozenset[str] = frozenset({"wav", "pcm16", "flac"})
"""Formats a chunk can be delivered in on its own.

Every event carries a *complete, playable* payload: one standalone file per
chunk, `media_type` riding beside it. Ogg stays excluded — its bitstream state
spans the whole stream, so a per-chunk container is a lie about what the bytes
are. FLAC is frame-based and each chunk encodes as a self-contained file, so it
streams honestly; it costs ~65% less on the wire than the base64 WAV for the
same samples (measured, `docs/benchmarks.md`), which is why a network client
would pick it. The trade is that chunks concatenate with a decoder rather than
with `+` — but so do WAV chunks, each with its own RIFF header, so nothing
about the contract changes.
"""


_SAFE_FETCH_SITES = frozenset({"same-origin", "none"})
"""``Sec-Fetch-Site`` values that are not another page driving the request.

``none`` is user-initiated — a typed URL, a bookmark, a client that is not a
browser tab at all; ``same-origin`` is a page this server served. Everything
else (``cross-site``, ``same-site``) means some other document caused the POST,
which here means a synthesis nobody asked for. Absent means the caller does not
speak the header, and every such caller is outside the CSRF model — see the
check in :class:`_Guard`.
"""


_DISCONNECT_POLL_S = 0.05
"""How often a streaming request re-asks whether its client is still there.

Fast enough that a closed tab stops the decode loop in well under a tenth of a
second, cheap enough to be free: ``is_disconnected()`` reads a receive channel,
it does not touch the socket. The poll runs on the event loop while the forward
pass runs in a worker thread, which is the only reason cancellation can land
mid-chunk at all.
"""


class SpeakRequest(BaseModel):
    """One synthesis request, shared by the one-shot and streaming routes.

    Defined at module level on purpose: under PEP 563 lazy annotations, a
    pydantic model defined inside ``build_app`` is not visible to FastAPI's route registration
    introspection, and the route silently stops accepting a body.
    """

    text: str = Field(min_length=1, max_length=_MAX_TEXT_LEN)
    voice: str
    seed: int = 0
    language: str | None = None
    """Omitted means "the voice's language".

    Passed through to the engine as ``None`` rather than resolved here: the
    chain (argument, then ``voice.language``, then ``"en"``) lives in
    :func:`loudkit.engine._resolve_language` and nowhere else, because a
    transport that reimplements it is a second place for it to drift.
    """

    long_form: bool = True

    speed: float = Field(default=1.0, ge=MIN_SPEED, le=MAX_SPEED)
    """Playback speed, pitch preserved. ``1.0`` is an exact bypass.

    Bounded by pydantic as well as by the engine so that an out-of-range value
    is refused before the request queues behind every other synthesis — the
    engine's own check would answer the same way, but only after the wait.
    """

    format: AudioFormat = "wav"
    """How to encode the audio. ``wav`` is what this server has always returned,
    byte for byte.

    A ``Literal``, so an unknown name is a 422 listing the four that work rather
    than a 200 carrying whatever the encoder produced.
    """

    previous_tokens: list[int] | None = Field(default=None, max_length=_MAX_PREVIOUS_TOKENS)
    """Speech tokens this request continues from, so the join is not audible.

    The tail is all that is used, and the tail is what the previous response
    handed back: ``X-Loudkit-Continuation`` on the one-shot route, the
    ``continuation`` field of the stream's ``done`` event. Longer input is
    accepted (the engine slices it) up to the bound; past that it is a 422.
    """


class OpenAISpeechRequest(BaseModel):
    """OpenAI's `/v1/audio/speech` body, so anything that speaks it can use this.

    Not an API this project designed — an API a great deal of tooling already
    emits. OpenClaw, for one, lets its `openai` provider be pointed at any
    `baseUrl`, so speaking this shape makes loudkit a drop-in TTS provider for
    it with no code on either side, only configuration. The same is true of
    every other client written against OpenAI's endpoint.

    It is a *fourth transport*, not a fourth synthesis path: the route below
    translates these fields into a `render_bytes` call and returns what comes
    back. Field names are theirs, including the ones this library would have
    spelled differently.
    """

    model: str = ""
    """Accepted and ignored. OpenAI requires it to choose between `tts-1` and
    `tts-1-hd`; this server has one engine, and `/health` says which. Refusing a
    request for naming a model would break clients that must send one."""

    input: str = Field(min_length=1, max_length=_MAX_TEXT_LEN)
    """Their name for the text. Ours is `text`; theirs wins here, because a
    transport that renames a field is a transport nothing can talk to."""

    voice: str
    """A voice *name* in this server's library. OpenAI's names (`alloy`, `nova`)
    mean nothing here, so a caller configures one of ours — `/v1/voices` lists
    them, and an unknown name comes back as a 404 naming what would have
    worked."""

    response_format: str = "wav"
    """Defaults to `wav`, and OpenAI's default is `mp3`.

    A deliberate deviation, because the alternative is worse. This server ships
    no mp3 encoder on purpose (see `AudioFormat`), so honouring their default
    would mean every unconfigured client got an error instead of audio. `mp3`,
    `opus` and `aac` are refused by name with the four formats that do work.
    """

    speed: float = 1.0
    """Playback speed. OpenAI allows 0.25–4.0 and this engine allows 0.5–2.0.

    Out-of-range is refused rather than clamped: a caller asking for 3x and
    silently getting 2x has been given audio that is not what it asked for, and
    nothing in the reply says so.
    """


def _continuation_header(continuation: tuple[int, ...]) -> dict[str, str]:
    """``X-Loudkit-Continuation``, or no header at all.

    The tail to send back as ``previous_tokens`` so the next request continues
    this one's prosody instead of restarting it. A header rather than a body
    field because the body is a WAV: the alternative was multipart, which every
    audio client would then have to learn in order to play a sound. Six small
    integers, comma-separated.

    **Omitted entirely when there is nothing to carry**, rather than sent empty.
    A recipe with ``prefix_tokens = 0`` produced `X-Loudkit-Continuation: ` and
    the obvious client parse — ``[int(t) for t in header.split(",")]`` — raises
    on it. A header a client must special-case is worse than one it can check
    for.
    """
    if not continuation:
        return {}
    return {"X-Loudkit-Continuation": ",".join(str(t) for t in continuation)}


class _Buckets:
    """One token bucket per client address, for the routes that cost real work.

    The queue depth and the wait deadline bound what a *reachable* server does
    under load, but neither costs the caller anything: a client that opens
    requests faster than the engine drains them still gets a slot each time one
    frees, ahead of everyone else, forever. That is the whole shape of the
    denial — no exploit, just a loop.

    Bounded in clients as well as in tokens. Keying by address means a caller
    behind a rotating source can mint entries, so the map is capped and the
    least recently seen entry is dropped: a full table costs a fixed amount of
    memory, and evicting the oldest can only ever *grant* tokens, never take
    them from a client that is still active.

    Not a general rate limiter and not a fairness scheme — see `_RATE_CAPACITY`.
    """

    __slots__ = ("_capacity", "_refill", "_max_clients", "_state")

    def __init__(self, capacity: int, refill_per_s: float, max_clients: int = 4096) -> None:
        self._capacity = float(capacity)
        self._refill = refill_per_s
        self._max_clients = max_clients
        # key -> (tokens, last seen). Insertion-ordered, so the first key is the
        # least recently added; `take` moves a key to the end on every hit,
        # which makes it least-recently-*used* as well.
        self._state: dict[str, tuple[float, float]] = {}

    def take(self, key: str) -> bool:
        now = time.monotonic()
        tokens, last = self._state.pop(key, (self._capacity, now))
        tokens = min(self._capacity, tokens + (now - last) * self._refill)
        allowed = tokens >= 1.0
        if allowed:
            tokens -= 1.0
        self._state[key] = (tokens, now)
        while len(self._state) > self._max_clients:
            self._state.pop(next(iter(self._state)))
        return allowed


def _host_is_loopback(host: str) -> bool:
    """Whether a ``Host`` header value names this machine's loopback.

    The port suffix is stripped the way RFC 3986 says, not by splitting on
    ``:`` — an IPv6 literal *contains* colons (``[::1]:8765``, or bare
    ``::1``), and a port split would leave those unreachable while the server
    legitimately binds them.
    """
    name = host.strip("[]").rsplit("]:", 1)[0] if host.startswith("[") else host
    # Bare IPv6 literals carry no port; everything else splits on the last colon.
    candidate = name if name.count(":") > 1 else name.rsplit(":", 1)[0]
    if candidate.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _cross_site_refusal(headers: dict[str, bytes]) -> tuple[int, dict[str, object]] | None:
    """Refuse a state-changing request that another page caused, or ``None``.

    Host pinning stops DNS rebinding from *reading* an answer; it does nothing
    about a blind write. A page on any origin can POST to
    ``http://127.0.0.1:8765`` without ever seeing the response, and the render
    it starts costs this engine exactly what a real request costs — on a server
    that synthesises one at a time, that is the whole denial.

    Two checks, because they close on different clients. ``Sec-Fetch-Site`` is
    sent by every current browser and names the offender outright. The JSON
    content type closes the path an older one would use: a form post or a
    ``no-cors`` fetch can only carry ``text/plain``, ``urlencoded`` or
    ``multipart``, and asking for ``application/json`` forces a preflight this
    server answers with no CORS headers at all. Non-browser callers send no
    ``Sec-Fetch-Site`` and already send the content type — ``requests``,
    ``httpx`` and every OpenAI client set it when they encode a JSON body.
    """
    site = headers.get("sec-fetch-site", b"").decode("latin-1").strip().lower()
    if site and site not in _SAFE_FETCH_SITES:
        return 403, {"detail": f"cross-site request from '{site}'", "code": "cross_site"}
    media = headers.get("content-type", b"").decode("latin-1").split(";", 1)[0]
    media = media.strip().lower()
    if media != "application/json" and not media.endswith("+json"):
        return 415, {
            "detail": f"expected Content-Type application/json, got {media or 'none'}",
            "code": "unsupported_media_type",
        }
    return None


class _Guard:
    """ASGI middleware: everything that must decide before the route exists.

    Host pinning, bearer auth, the cross-site refusal on the write path, and a
    body bound applied *before* the read.

    All of them have to sit above the route. Pydantic's ``max_length`` runs
    after Starlette has already buffered the whole body, so a 500 MB POST was
    read into memory in full and then refused for being 10 000 characters too
    long — the cap protected the engine and not the process. ``Content-Length``
    is checked first; a chunked request that declares no length is counted as
    it streams and cut off at the same bound.

    Written as raw ASGI rather than ``@app.middleware("http")`` because the
    HTTP-middleware form only sees a ``Request`` whose body is read on demand
    by the route below it — there is no point in that layer where the bytes can
    be refused before they arrive.
    """

    def __init__(
        self, app: Any, *, token: str | None = None, allow_public: bool = False
    ) -> None:
        self.app = app
        self.token = token
        self.allow_public = allow_public
        # Wherever this app is reachable from off the machine, which is what
        # `allow_public` and `token` each say in their own way. On the loopback
        # default the caller is already on the machine and a limiter buys
        # nothing but a way to lock yourself out; off it, a holder of the token
        # is not thereby entitled to fill the queue of a server that
        # synthesises one at a time. Both conditions are named rather than the
        # token alone, so that a caller who reaches this class directly cannot
        # arrange a public app with the limiter switched off.
        self.buckets = (
            _Buckets(_RATE_CAPACITY, _RATE_REFILL_PER_S)
            if token is not None or allow_public
            else None
        )

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v for k, v in scope.get("headers", [])}

        # Host pinning on the loopback default. A browser at an attacker's page
        # can resolve any hostname to 127.0.0.1 (DNS rebinding) and then read
        # responses from a server that trusts its loopback bind — the origin
        # model does not protect a raw API. Pinning the accepted Host to
        # loopback names closes it; public binds are authenticated already.
        if not self.allow_public:
            host = headers.get("host", b"").decode("latin-1")
            if not _host_is_loopback(host):
                await _send_json(
                    send,
                    403,
                    {"detail": f"host '{host}' not allowed", "code": "bad_host"},
                )
                return

        if self.token is not None:
            supplied = headers.get("authorization", b"")
            expected = f"Bearer {self.token}".encode()
            # compare_digest, not ==: a plain comparison short-circuits on the
            # first differing byte and leaks the matched prefix through timing.
            if len(supplied) != len(expected) or not hmac.compare_digest(supplied, expected):
                await _send_json(send, 401, {"detail": "unauthorized", "code": "unauthorized"})
                return

        declared = headers.get("content-length")
        # Length before value: CPython 3.11 refuses to parse an integer literal
        # past 4300 digits and raises, so a header of 5000 nines turned a body
        # that should be refused with 413 into an unhandled 500. A real length
        # is twenty digits at the outside; anything longer is over the cap by
        # inspection, without parsing it.
        if (
            declared is not None
            and declared.isdigit()
            and (len(declared) > 20 or int(declared) > _MAX_BODY_BYTES)
        ):
            await _send_json(
                send,
                413,
                {
                    "detail": f"request body exceeds {_MAX_BODY_BYTES} bytes",
                    "code": "payload_too_large",
                },
            )
            return

        # The write path, closed the way the read path already is — see
        # `_cross_site_refusal`. Same POST-under-/v1 condition as the bucket
        # below: those are the requests that cost a render.
        if scope.get("method") == "POST" and str(scope.get("path", "")).startswith(_API_PREFIX):
            refusal = _cross_site_refusal(headers)
            if refusal is not None:
                await _send_json(send, *refusal)
                return

        # Synthesis only, and that means POST: `GET /v1/voices` is a directory
        # read that costs nothing, and `/health` is what a load balancer polls —
        # rate-limiting either turns a busy server into an unreachable one at
        # exactly the wrong moment. The first version of this check said
        # "synthesis only" in a comment and then matched the whole `/v1` prefix.
        if (
            self.buckets is not None
            and scope.get("method") == "POST"
            and str(scope.get("path", "")).startswith(_API_PREFIX)
        ):
            client = scope.get("client")
            if not self.buckets.take(client[0] if client else "-"):
                await _send_json(
                    send,
                    429,
                    {
                        "detail": "too many requests; the engine synthesises one at a time",
                        "code": "rate_limited",
                    },
                )
                return

        seen = 0
        too_big = False
        sent_anything = False

        async def bounded_receive() -> Any:
            nonlocal seen, too_big
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > _MAX_BODY_BYTES:
                    # A chunked request declares no Content-Length, so the only
                    # place to stop it is here, mid-stream. The app is told the
                    # client hung up — there is nothing else in ASGI to tell it
                    # — and everything it says from that point is a reaction to
                    # a disconnect this middleware manufactured, not an answer
                    # to the request.
                    too_big = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Any) -> None:
            nonlocal sent_anything
            # Swallowed once the body is over the limit. Starlette turns the
            # manufactured disconnect into a ClientDisconnect and FastAPI
            # answers it with a 500 — so the caller who sent 200 MB was told
            # the *server* had failed, and this middleware's own 413 never got
            # out because a response had already started.
            if too_big:
                return
            sent_anything = True
            await send(message)

        await self.app(scope, bounded_receive, guarded_send)
        if too_big and not sent_anything:
            await _send_json(
                send,
                413,
                {
                    "detail": f"request body exceeds {_MAX_BODY_BYTES} bytes",
                    "code": "payload_too_large",
                },
            )


def _first_message(exc: BaseException) -> str:
    """What the stream's terminal event says, so it matches what the CLI says."""
    root = _first_exception(exc)
    return f"{type(root).__name__}: {root}" if str(root) else type(root).__name__


def _error_kind(exc: BaseException) -> str:
    """``bad_request`` or ``server_fault``, for a failure that missed the status line.

    ``/v1/synthesize`` answers 400 for a question about the request and 500 for
    a defect here. The stream cannot: by the time a chunk fails, Starlette has
    long since sent 200, so both arrived as the same ``{"done": true, "error":
    ...}`` and a client had nothing but prose to tell them apart. One is worth
    retrying with a different request; the other never is, and an agent that
    cannot see the difference will retry forever.

    :class:`~loudkit.errors.LoudkitError` is the line, which is what that
    hierarchy is for: everything loudkit refuses on purpose is under it, and
    everything else — a backend stub, a numpy failure, a bug here — is not.
    """
    return "bad_request" if isinstance(_first_exception(exc), LoudkitError) else "server_fault"


def _error_code(exc: BaseException) -> str:
    """The catalog code for a failure, from its root cause.

    The specific counterpart of :func:`_error_kind`: ``error_kind`` says which
    side owns the failure, this names the condition — ``voice_not_found``,
    ``window_overflow`` — using the same frozen catalog every transport speaks
    (see :mod:`loudkit.errors`). A failure that is not a deliberate refusal is
    ``server_fault``, matching the kind.
    """
    root = _first_exception(exc)
    return error_code(root) if isinstance(root, LoudkitError) else "server_fault"


_CODE_BY_STATUS = {
    400: "invalid_request",
    404: "invalid_request",
    413: "payload_too_large",
    422: "invalid_request",
    429: "rate_limited",
    503: "busy",
}
"""What an error answer says when the exception that caused it named nothing.

A refusal raised from a :class:`~loudkit.errors.LoudkitError` carries that
class's own code; these are the boundary conditions — full queue, oversized
body — that exist only here.
"""


async def _send_json(send: Any, status: int, payload: dict[str, object]) -> None:
    body = json.dumps(payload).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


_MIN_TOKEN_CHARS = 16
"""Shortest string this server will accept as a bearer token.

Sized against what it defends: one holder of the token can synthesise in every
voice on the machine, and nothing rate-limits *guesses* below the 12-request
bucket, so the secret has to be out of reach of a network guesser rather than
merely inconvenient to type. Sixteen characters of the alphabet this module
suggests (``secrets.token_urlsafe``, 64 symbols) is ~96 bits, which is not
searchable; the generated default is ``token_urlsafe(32)``, 43 characters, and
this floor sits well under it so an operator's own passphrase is still allowed.
A number rather than an entropy estimate because the string arrives as a string:
there is no way to tell a random 16 from a memorable one, and a length is the
only bound that holds for both.
"""


def _token_fault(token: str | None) -> str | None:
    """Why ``token`` cannot serve as a bearer credential, or ``None`` if it can.

    Every rejected shape here reaches :class:`_Guard` as a token that *is* set,
    which switches auth on and then compares ``Authorization`` against it. The
    empty string is the sharp one: the guard would demand exactly ``"Bearer "``
    from every caller, and a header holding nothing is not a secret anyone has
    to know. Whitespace-only is the same hole with a typo in it.

    Control characters are refused because the value is pasted into an HTTP
    header on the client side, where CR and LF end the header rather than sit
    inside it: a token containing one is a header-splitting primitive handed to
    whoever chose the token, and a token containing a tab or a space cannot
    survive the round trip intact anyway. The accepted set is printable ASCII
    with no spaces, which is what every token generator in this codebase emits.

    Returns a phrase naming the reason, so each caller can wrap it in the
    refusal its own layer speaks (``ValueError`` here, ``SystemExit`` in
    :func:`serve`) while the rule itself lives in one place.
    """
    if token is None:
        return None
    if not token.strip():
        return (
            "the token is empty or only whitespace, which is not a secret: "
            "the server would then accept the literal header 'Authorization: "
            "Bearer ' from anyone. Pass secrets.token_urlsafe(32), or pass no "
            "token at all for a loopback-only app."
        )
    bad = next((c for c in token if not (0x21 <= ord(c) <= 0x7E)), None)
    if bad is not None:
        return (
            f"the token contains {ord(bad):#04x}, which is not printable ASCII. "
            "The token is sent verbatim inside an HTTP header, where a newline "
            "or carriage return ends the header instead of belonging to it, and "
            "a space or tab does not survive the round trip. Use "
            "secrets.token_urlsafe(32)."
        )
    if len(token) < _MIN_TOKEN_CHARS:
        return (
            f"the token is {len(token)} characters, under the "
            f"{_MIN_TOKEN_CHARS}-character minimum. A short token is guessable "
            "by anyone who can reach the port, and what it protects is "
            "synthesis in every voice on this machine. Use "
            "secrets.token_urlsafe(32)."
        )
    return None


def build_app(  # noqa: PLR0915 - routes + a per-route lock; linear and explicit
    engine: Engine,
    voices: VoiceLibrary,
    *,
    token: str | None = None,
    allow_public: bool = False,
) -> FastAPI:
    """Wire the routes onto a warm engine.

    The engine is **single-flight**: synthesis routes serialise behind a lock,
    so concurrent requests queue rather than interleave. This is required by
    the CUDA-graph path (a graph capture is not reentrant) and by torch modules
    that are not thread-safe. Health and voice listing are read-only and stay
    lock-free.

    Args:
        token: when set, every request must carry ``Authorization: Bearer
            <token>``. **Required** whenever ``allow_public`` is set — a server
            reachable from the network with no auth lets anyone who can reach
            the port speak in every voice on the machine. Whatever the bind,
            a token that is set has to be usable as one: see
            :func:`_token_fault` for the shapes refused and why.
        allow_public: the app is going to be reachable from off this machine,
            so the Host pin is dropped (a public bind legitimately answers to
            its own hostname) and the bearer token becomes the boundary.

    Raises:
        ValueError: for ``allow_public=True`` with no token, and for a token
            that is present but cannot be a secret — empty, whitespace-only,
            carrying a character an HTTP header cannot hold, or shorter than
            ``_MIN_TOKEN_CHARS``. Each of those switches authentication on and
            then checks something a stranger can supply. The refusal lives
            here rather than only in :func:`serve` because this function is
            exported: an embedder who wires the app into their own uvicorn,
            gunicorn or ASGI stack never runs a line of ``serve``, and that
            path used to hand back an app with no Host pin, no token and no
            rate limit. Security that depends on which entry point the caller
            happened to use is not security.

    Separated from :func:`serve` so tests can exercise the app without binding a
    port, and so the assertion that matters — server bytes equal in-process
    bytes — is cheap to write.
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
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse, Response, StreamingResponse
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise ModuleNotFoundError(_MISSING_EXTRA) from exc

    app = FastAPI(title="loudkit", version=_version())

    @app.exception_handler(HTTPException)
    async def _with_code(_request: Any, exc: HTTPException) -> JSONResponse:
        """Every error answer names its condition, not just its status.

        ``code`` comes from the frozen catalog in :mod:`loudkit.errors`. A
        refusal raised through :func:`_refuse` carries the exact code of the
        exception that caused it; anything else falls back to what the status
        line already implied, so no error body ships without one.
        """
        code = getattr(exc, "loudkit_code", None) or _CODE_BY_STATUS.get(
            exc.status_code, "server_fault"
        )
        return JSONResponse(
            {"detail": exc.detail, "code": code},
            status_code=exc.status_code,
            headers=exc.headers,
        )

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def _validation_with_code(_request: Any, exc: RequestValidationError) -> JSONResponse:
        """Schema violations speak the same vocabulary as every other refusal.

        FastAPI's default 422 body is kept (``detail`` holds pydantic's error
        list); only ``code`` is added, so a client that branches on codes does
        not need a special case for "the request never reached a route".
        """
        return JSONResponse(
            {"detail": exc.errors(), "code": "invalid_request"}, status_code=422
        )

    def _refuse(status: int, exc: BaseException) -> HTTPException:
        """An HTTPException that remembers which condition refused.

        Raise sites say ``raise _refuse(404, exc) from exc`` instead of losing
        the class in ``str(exc)`` — the handler above turns the remembered
        code into the ``"code"`` field of the answer.
        """
        http = HTTPException(status_code=status, detail=str(exc))
        http.loudkit_code = error_code(exc)  # type: ignore[attr-defined]
        return http

    app.add_middleware(_Guard, token=token, allow_public=allow_public)

    # The engine is single-flight: it holds mutable CUDA state (a static KV
    # cache under --cuda-graphs, and torch modules that are not all reentrant).
    # FastAPI runs sync routes in a threadpool (40 threads by default), so N
    # concurrent requests would otherwise enter the same Engine at once — for
    # CUDA graphs that means two graph captures racing on the ambient stream.
    # One lock serialises synthesis. Health and /v1/voices are cheap and
    # read-only, so
    # they stay outside it.
    #
    # This lock is acquired on the event loop, by BOTH routes, *before* either
    # one enters the threadpool. Separate locks — a threading.Lock taken inside
    # the threadpool by /v1/synthesize, an anyio.Lock taken on the loop by the
    # stream — deadlock against each other: /v1/synthesize runs via FastAPI's
    # `run_in_threadpool`, which shares the same AnyIO
    # worker limiter as the stream's per-chunk `anyio.to_thread.run_sync`
    # calls. Enough concurrent /v1/synthesize requests fill that pool with
    # threads blocked on threading.Lock; once it is full, an in-flight
    # stream's next chunk pull has no worker left to run on and the whole
    # server — including /health — hangs. A single lock acquired on the loop
    # before either route touches a worker means a waiting request sits in the
    # loop, never occupying a worker it can't make progress on.
    #
    # A **Semaphore(1), not a Lock**, because the streaming route acquires and
    # releases from two different tasks: the route handler takes the slot before
    # the response starts, and the response is driven from a task of its own,
    # which is where the slot is given back. `anyio.Lock` is
    # owner-checked and refuses that release with "the current task is not
    # holding this lock" — mid-stream, after the 200. `max_value=1` keeps a
    # double release loud instead of silently admitting two callers into an
    # engine that is not reentrant.
    import anyio

    _engine_lock = anyio.Semaphore(1, max_value=1)
    _queued = 0
    # When the render currently holding the slot started, or None if the engine
    # is idle. A one-element list rather than a `nonlocal` float because
    # `/health` reads it from a different task than the one that writes it, and
    # a mutable cell makes that sharing explicit. See `_SLOW_RENDER_S`.
    _started_at: list[float | None] = [None]

    class _Queue:
        """Bound the depth of the queue in front of the single-flight engine.

        The engine serialises, so requests wait — and an unbounded wait turns a
        slow engine into unbounded memory and unbounded latency with every
        client still holding its connection. Past the bound the honest answer
        is 503 with Retry-After: busy, and not getting to you soon.

        Usable as an async context manager (``/v1/synthesize``, where the whole
        request is one ``await``) or as an explicit
        :meth:`acquire`/:meth:`release` pair (``/v1/synthesize/stream``, where the
        slot must be taken *before* the response starts and given back when the
        response is over and the engine has stopped — see the route).
        """

        async def acquire(self) -> None:
            nonlocal _queued
            if _queued >= _MAX_QUEUED:
                raise HTTPException(
                    status_code=503,
                    detail=f"{_queued} requests already queued for the engine",
                    headers={"Retry-After": "1"},
                )
            _queued += 1
            try:
                # See `_MAX_WAIT_S`: the depth bound alone lets a wedged render
                # hold every caller behind it open indefinitely.
                with anyio.fail_after(_MAX_WAIT_S):
                    await _engine_lock.acquire()
            except TimeoutError as exc:
                _queued -= 1
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"waited {_MAX_WAIT_S:.0f}s for the engine and never "
                        "reached it; a synthesis ahead of this one is stuck"
                    ),
                    headers={"Retry-After": "30"},
                ) from exc
            except BaseException:
                _queued -= 1
                raise
            _started_at[0] = time.monotonic()

        def release(self) -> None:
            nonlocal _queued
            _started_at[0] = None
            _queued -= 1
            _engine_lock.release()

        async def __aenter__(self) -> None:
            await self.acquire()

        async def __aexit__(self, *exc: object) -> None:
            self.release()

    _engine_slot = _Queue()

    from starlette.types import Receive, Scope, Send

    class _LeasedStream(StreamingResponse):
        """A stream that gives the engine slot back when the *response* is over.

        The slot cannot be released by the body generator, because the body
        generator is not on every path out of a response. Starlette sends
        ``http.response.start`` before it pulls the first item, and if that send
        fails — a client that hung up between the request and the first byte —
        the generator is never started, so no ``finally`` inside it ever runs.
        The slot was then held by nobody for the life of the process and every
        later request answered 503.

        ``__call__`` is the one frame that exists on all of them: normal
        completion, a client that disconnects mid-stream (Starlette cancels the
        streaming task and this returns), an exception on the way out, and the
        dead ``response.start``. Doing it here also keeps the release *after*
        the last byte, which the generator's own ``finally`` could not: that
        runs at the last ``yield``, with the send still pending.
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
            try:
                await super().__call__(scope, receive, send)
            finally:
                await self._release()

    @app.get("/health", response_model=None)
    def health() -> Response:
        """Enough to tell whether the thing answering is the thing you meant.

        Returns the resolved algorithm and execution configs, because the defect
        this library was built around survived an entire optimisation campaign
        for want of exactly that line.

        Answers 503 while a synthesis has been holding the single-flight slot
        longer than :data:`_SLOW_RENDER_S`. A wedged render cannot be preempted —
        it owns non-reentrant decoder state — so reporting it is the only honest
        move left, and it is the one a load balancer can act on. Under the old
        unconditional ``ok`` an instance that could no longer synthesise anything
        kept receiving traffic.
        """
        body: dict[str, object] = {
            "status": "ok",
            "algorithm": engine.algorithm.describe(),
            "execution": engine.execution.describe(),
            "fingerprint": engine.algorithm.fingerprint(),
            "voices": voices.names(),
        }
        started = _started_at[0]
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
    async def synthesize(req: SpeakRequest) -> Response:
        """Return a WAV. Same text, voice and seed give the same bytes."""
        try:
            # In a worker thread rather than on the loop: a cold profile is a
            # whole-file read, a SHA-256 over those bytes and a safetensors
            # parse, and the loop is what answers /health and polls every
            # in-flight stream for disconnects. `VoiceLibrary` caches the parse
            # (see `_VOICE_CACHE_BYTES`), so the warm path is one `stat` — this
            # hop keeps even that off the loop, and keeps a cold load from
            # stalling it.
            voice = await anyio.to_thread.run_sync(voices.load, req.voice)
        except VoiceNotFoundError as exc:
            raise _refuse(404, exc) from exc
        except ValueError as exc:
            raise _refuse(400, exc) from exc

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
            )

        try:
            # Acquired on the loop, before the worker thread — see the
            # comment on `_engine_lock` above.
            async with _engine_slot:
                rendered = await anyio.to_thread.run_sync(_render)
        # UnsupportedLanguageError, not NotImplementedError. Asking for a
        # language this build cannot preprocess is a question about the
        # request, so it is a 400, exactly as the CLI prints "unsupported:".
        #
        # Only this loudkit type is a 400. A builtin NotImplementedError —
        # an unfinished backend method, a stub left in a renderer — is a
        # server defect; reporting it as a bad request names the caller as the
        # cause, the one status code a client cannot act on. Anything else
        # escapes to FastAPI's handler and answers 500.
        except UnsupportedLanguageError as exc:
            raise _refuse(400, exc) from exc
        except ValueError as exc:  # over-window without chunking, bad config
            raise _refuse(422, exc) from exc

        return Response(
            content=rendered.data,
            media_type=rendered.media_type,
            headers={
                "X-Loudkit-Duration": f"{rendered.duration:.3f}",
                "X-Loudkit-Tokens": str(rendered.n_tokens),
                # Always, not only for `pcm16`. Raw frames carry no header to
                # read it from, and for the container formats it saves a client
                # parsing one to find out what it already asked for.
                "X-Loudkit-Sample-Rate": str(engine.algorithm.sample_rate),
                "X-Loudkit-Fingerprint": engine.algorithm.fingerprint(),
                # A truncated utterance is still a 200 — the audio is real,
                # just cut short. The header is the only way a client can tell,
                # and its absence was silent data loss.
                "X-Loudkit-Truncated": "true" if rendered.hit_token_cap else "false",
                # The C2PA claim-only manifest, out of band as well as in the
                # WAV trailer: the other three encodings cannot carry the box,
                # and a client that just wants the label should not have to
                # parse a container to read it.
                **(
                    {"X-Loudkit-Provenance": json.dumps(rendered.provenance, sort_keys=True)}
                    if rendered.provenance is not None
                    else {}
                ),
                **_continuation_header(rendered.continuation),
            },
        )

    def _openai_error(status: int, message: str, kind: str) -> JSONResponse:
        """An error in OpenAI's envelope rather than FastAPI's `detail`.

        Clients written against their API read `error.message`, so a bare
        `detail` surfaces to a user as a generic HTTP failure with the useful
        part -- which voice names exist, which formats work -- dropped.
        """
        return JSONResponse(
            {"error": {"message": message, "type": kind, "param": None, "code": None}},
            status_code=status,
        )

    @app.post(f"{_API_PREFIX}/audio/speech", response_model=None)
    async def openai_speech(req: OpenAISpeechRequest) -> Response:
        """OpenAI's speech endpoint, answered by this engine.

        A *transport*, not a second synthesis path: it builds a `SpeakRequest`
        and calls the route above, so the bytes are the bytes `/v1/synthesize`
        would have returned for the same text, voice and seed -- there is one
        funnel, one sampler and one encoder behind both, and a test pins the
        two responses byte for byte.

        Worth stating why this shape and not another: it is the one a great
        deal of existing tooling already emits. A client only has to be pointed
        at this server's `/v1` as its base URL, with no adapter in between and
        nothing to write on either side. The bearer token such clients send as
        their API key is the token `_Guard` already checks, so auth works
        without a second mechanism.

        Not implemented, and not planned: `stream_format: "sse"`. OpenAI's
        streaming envelope is a different framing from this server's own, and
        `/v1/synthesize/stream` is the route that streams. An unknown field is
        ignored by pydantic here, so a client asking for it gets the whole
        utterance in one response rather than an error -- correct audio,
        delivered less eagerly than requested.
        """
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
        # Checked here rather than left to pydantic's bounds on `SpeakRequest`,
        # because their range is wider than this engine's and the difference is
        # the whole message: "0.25 is valid OpenAI and unsupported here" is
        # actionable, and a 422 about a field the caller never sent is not.
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
                )
            )
        except HTTPException as exc:
            # Re-dressed, not re-raised: FastAPI would answer `{"detail": ...}`
            # and an OpenAI client would show the user nothing useful.
            kind = "invalid_request_error" if exc.status_code < 500 else "server_error"
            return _openai_error(exc.status_code, str(exc.detail), kind)

    @app.post(f"{_API_PREFIX}/synthesize/stream", response_model=None)
    async def synthesize_stream(  # noqa: PLR0915 - one linear route; the branches are the contract
        req: SpeakRequest, request: Request
    ) -> StreamingResponse:
        """Stream a passage chunk by chunk as Server-Sent Events.

        Each chunk is a full WAV, delivered as soon as it is rendered, so a
        reading app can start playing the first sentence while the rest is still
        being synthesised. This is the same synthesis as ``/v1/synthesize`` — the
        engine's :meth:`~loudkit.engine.Engine.stream` — with delivery attached;
        it is not a second synthesis path.

        Events carry the WAV in base64 plus the chunk's duration and token
        count; the final event is ``done`` with the aggregate fingerprint. A
        failure after the first chunk is a ``done`` event carrying ``error``,
        because by then the status line is long gone.

        **Everything that can decide a status code happens here, in the route,
        before a byte of the response is written.** Starlette sends
        ``http.response.start`` — status 200, ``text/event-stream`` — before it
        pulls the first item out of the body generator, so an ``HTTPException``
        raised inside that generator can never set a status: the queue bound's
        503 came back as a clean 200 with an empty body, which reads to a client
        exactly like a passage that had nothing to say. The engine slot is
        therefore taken here, before the response exists.

        It is given back by the response object rather than by the body
        generator, and only once the engine's own threads have stopped — see
        :class:`_LeasedStream` and ``_return_the_slot``. The generator is the
        wrong owner in both directions: it does not run at all when the socket
        dies on ``http.response.start``, and when it does run it finishes
        before the engine does.
        """
        # Before the voice, before the queue slot, before anything that costs:
        # these are questions about the request, and nothing about their answers
        # changes while the server runs.
        #
        # `previous_tokens` in particular has to be checked *here* rather than
        # left to the engine. This route takes the single-flight slot before the
        # response starts, so a list that was never usable would otherwise queue
        # behind every other synthesis in order to fail — and fail after the 200,
        # where a status code can no longer be sent and the refusal has to
        # travel as a `done` event instead.
        try:
            # Imported here rather than at module scope: `Engine` is only a
            # TYPE_CHECKING name in this file on purpose, so that importing the
            # server does not drag in the whole engine, and one early validator
            # is not a reason to give that up.
            from ..engine import validate_speech_tokens

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
            # Off the loop, as in `/v1/synthesize` above — and still before the
            # engine slot, so an unknown voice is a 404 rather than a `done`
            # event behind everyone else's queue.
            voice = await anyio.to_thread.run_sync(voices.load, req.voice)
        except VoiceNotFoundError as exc:
            raise _refuse(404, exc) from exc
        except ValueError as exc:
            raise _refuse(400, exc) from exc

        # Raises 503 when the queue is full, on the loop, before the response
        # exists. Nothing that can fail may sit between this and the response
        # below: from here the slot is given back by `_return_the_slot`, and
        # that is wired to the response object rather than to the route.
        await _engine_slot.acquire()

        # Set by the disconnect watcher inside `events()` and by the reclaim
        # below, read by the decode loop in the worker thread. A
        # threading.Event is the right primitive here precisely because the two
        # live on different threads: the loop sets it, the forward pass polls
        # it. It lives out here, with the generator it stops, because the
        # reclaim outlives `events()` — it runs on paths where that generator
        # was never started.
        cancelled = threading.Event()

        # The engine is single-flight: the sync route and the streaming route
        # must never be inside it at once. The streaming route cannot run a
        # forward pass on the loop (it would freeze /health and the disconnect
        # check), so `produce()` is a generator run in a worker thread, one
        # chunk per `next()` call.
        #
        # Built here rather than inside `events()` so that closing it is
        # possible on every path, including the ones where `events()` never
        # runs a line. Building a generator starts nothing, so this stays on
        # the "nothing that can fail" side of the acquire above.
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

        it = produce()

        def _reclaim_the_engine() -> None:
            """Get the stages back, or say they are gone.

            ``it.close()`` raises GeneratorExit inside
            :meth:`~loudkit.engine.Engine.stream`, whose own teardown stops the
            producer, cancels the renders nobody will read and joins the
            thread — and marks the engine wedged when that thread outlives the
            join. So this returns only once the engine is idle or has recorded
            that it is not, which is the property the slot's next holder needs.

            Off the loop, because that join is the one thing on this path that
            can take seconds.

            A failure here is not the caller's problem to absorb: the stages
            cannot be shown to be free, so the engine is marked unusable rather
            than handed to whoever is next in the queue. That is the same
            verdict the join timeout reaches, by a different route.
            """
            try:
                it.close()
            except BaseException as exc:  # noqa: BLE001 — the verdict is the same for all
                _LOG.exception("could not reclaim the engine after a stream")
                engine._wedge(  # noqa: SLF001 — one package, one single-flight engine
                    f"reclaiming an abandoned stream raised {type(exc).__name__}, "
                    "so nothing here can show the token generator and the "
                    "renderer were left idle"
                )

        released = False

        async def _return_the_slot() -> None:
            """Give the single-flight slot back, once, however the stream ended.

            Wired to the response object rather than to the body generator
            because the generator is not on every path. Starlette sends
            ``http.response.start`` before it pulls the first item, and a socket
            that dies on that send leaves the generator never started, so a
            ``finally`` inside it never runs and the slot is never returned:
            one dead client, and every later request is a 503 for the life of
            the process.

            Order is the whole point. The flag stops the decode loop, the
            reclaim waits for the engine to actually be out of its stages, and
            only then does the next caller get in. Releasing first and letting
            the engine wind down afterwards is what let a disconnected stream's
            producer thread run *inside* a non-reentrant engine that the next
            request had already entered.

            Shielded, because the common way to get here is the response task
            being cancelled, and a reclaim that is itself cancelled proves
            nothing about the threads it was supposed to be waiting for.
            """
            nonlocal released
            if released:
                return
            released = True
            cancelled.set()
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(_reclaim_the_engine)
            _engine_slot.release()

        def _terminal(
            truncated: bool,
            continuation: tuple[int, ...] = (),
            *,
            error: str | None = None,
            kind: str | None = None,
            code: str | None = None,
        ) -> str:
            """The one event every client waits for, success or not.

            ``error`` is absent on a clean finish and present when synthesis
            failed after the response had started. It rides on ``done`` rather
            than on an event of its own so that a client written against the
            documented shape — read until ``done`` — cannot miss it.

            ``error_kind`` travels with it — ``bad_request`` or
            ``server_fault`` — because this event is where a stream's failures
            land *instead of* a status code, and without it a client cannot
            tell a retryable mistake from one that will fail identically
            forever. ``error_code`` names the condition itself, from the frozen
            catalog in :mod:`loudkit.errors`. See :func:`_error_kind`.

            ``continuation`` is the passage's own tail — what to send back as
            ``previous_tokens`` so a following request continues this prosody
            rather than restarting it. On the terminal event rather than on
            every chunk because that is the one a chaining client needs: the
            tail of the *passage*, not of each piece of it. Empty when the
            stream ended before producing anything.
            """
            done: dict[str, object] = {
                "done": True,
                "fingerprint": engine.algorithm.fingerprint(),
                "truncated": truncated,
                "continuation": list(continuation),
            }
            if error is not None:
                done["error"] = error
                done["error_kind"] = kind
                done["error_code"] = code
            return f"data: {json.dumps(done)}\n\n"

        async def events() -> AsyncIterator[str]:  # noqa: PLR0915 — one linear stream loop
            import base64

            # True only between "this chunk's render returned" and the end of
            # the task group that rendered it — the one teardown of the watcher
            # that must *not* cancel the stream. Read by the watcher below.
            chunk_ready = False

            async def watch_disconnect() -> None:
                """Flip ``cancelled`` as soon as the client goes away — or we do.

                Runs concurrently with the worker thread that is rendering, so
                the flag is set *during* a chunk rather than after it. That is
                the whole point: the decode loop polls it every step, so a
                closed tab stops a 255-token render inside one forward pass.

                The ``finally`` is what makes that hold for a *cancelled* stream
                as well, and it is not defensive coding: cancelling the response
                task — server shutdown, Starlette's own disconnect handling, any
                scope above this one — cancels this watcher too, while
                ``to_thread.run_sync`` is never abandoned and keeps the chunk
                rendering to its end. Without the flag the watcher dies before
                the render it exists to outrace, and the cancellation lands
                whole seconds later, at the next chunk boundary. Cancellation is
                delivered here at a checkpoint, so the flag flips while the
                forward pass is still polling it.
                """
                try:
                    while not await request.is_disconnected():
                        await anyio.sleep(_DISCONNECT_POLL_S)
                finally:
                    if not chunk_ready:
                        cancelled.set()

            # The engine slot was taken by the route, on the loop, before any
            # worker thread — see the comment where the lock is created, and the
            # route's docstring for why it cannot be taken in here. Held for the
            # stream's whole lifetime, so /v1/synthesize and other streams wait
            # on the loop rather than occupying a worker they can't advance.
            #
            # Nothing in this generator gives it back. Every way out of a stream
            # — the last event, a disconnect, a failure, a caller that never
            # drains it, and a response that dies before the first `next()` —
            # ends in `_return_the_slot`, which is attached to the response
            # object above. A `finally` here would cover four of those five and
            # would run before the engine's own threads had stopped.
            _done = object()
            truncated = False
            continuation: tuple[int, ...] = ()

            def _next_chunk() -> object:
                # StopIteration is a control-flow exception; running it
                # through a thread (anyio) turns it into a RuntimeError.
                # Catch it here and return a sentinel instead.
                try:
                    return next(it)
                except StopIteration:
                    return _done

            while True:
                if cancelled.is_set() or await request.is_disconnected():
                    # Just stop. Closing the engine's generator is the reclaim's
                    # job, and it belongs there rather than here: `close()` runs
                    # the engine's teardown, which joins a producer thread, and
                    # that join has no business happening on the event loop that
                    # is meanwhile answering /health.
                    return
                # The task group is entered and left without yielding
                # across it — an async generator may not suspend inside a
                # cancel scope, so the watcher lives exactly as long as one
                # chunk's render and the `yield` happens outside it.
                # A synthesis failure here cannot be a status code: on every
                # chunk after the first the 200 is long gone, and on the
                # first one Starlette has already sent it. Without a
                # terminal event the client sees a stream that simply stops,
                # which is indistinguishable from a passage that ended — so
                # the failure is delivered as the `done` event, the one
                # event every client already waits for.
                #
                # `Exception`, not a named list: whatever went wrong, a
                # named end beats a truncated stream, and the message is
                # carried out of the clause rather than yielded inside it
                # because the yield must not happen while the group is
                # unwinding. Cancellation is a BaseException in anyio, so it
                # still propagates and still tears the stream down.
                #
                # The *kind* is carried alongside, because catching
                # everything is what makes a bad request and a server bug
                # look identical here — see `_error_kind`.
                failure: str | None = None
                kind: str | None = None
                code: str | None = None
                try:
                    chunk_ready = False
                    async with anyio.create_task_group() as tg:
                        tg.start_soon(watch_disconnect)
                        chunk = await anyio.to_thread.run_sync(_next_chunk)
                        # Before the cancel, not after: the watcher's
                        # teardown reads this to tell "the chunk is here"
                        # from "this stream is being torn down".
                        chunk_ready = True
                        tg.cancel_scope.cancel()
                except Exception as exc:
                    kind = _error_kind(exc)
                    code = _error_code(exc)
                    if kind == "server_fault":
                        # See `_LOG`: a defect's message is for the operator,
                        # not for an unauthenticated caller.
                        _LOG.exception("synthesis stream failed")
                        failure = "internal error"
                    else:
                        failure = _first_message(exc)
                if failure is not None:
                    yield _terminal(
                        truncated, continuation, error=failure, kind=kind, code=code
                    )
                    return
                if cancelled.is_set():
                    return
                if chunk is _done:
                    break
                rendered = cast(Rendered, chunk)
                truncated = truncated or rendered.hit_token_cap
                # Overwritten per chunk: what a chaining client needs is the
                # tail of the last chunk delivered, which is what this holds
                # by the time the terminal event is written.
                continuation = rendered.continuation
                payload = {
                    "audio": base64.b64encode(rendered.data).decode(),
                    "media_type": rendered.media_type,
                    "duration": rendered.duration,
                    "tokens": rendered.n_tokens,
                    "truncated": rendered.hit_token_cap,
                }
                yield f"data: {json.dumps(payload)}\n\n"
            # The terminal event carries the aggregate, because a client
            # that only reads `done` must still learn that some chunk was
            # cut off — matching Engine.synthesize_long, which ORs the flag
            # across chunks for exactly the same reason.
            yield _terminal(truncated, continuation)

        return _LeasedStream(
            events(),
            _return_the_slot,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
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

    Args:
        checkpoint: packed ``.safetensors``.
        voices: directory of voice profiles. Defaults to ``voices/`` beside the
            checkpoint.
        host: interface to bind. Localhost by default, and read the module
            docstring before changing it. A non-loopback bind is refused
            unless ``allow_public`` is set.
        port: TCP port.
        device: ``cpu`` / ``cuda`` / ``mps``, or ``None`` to pick.
        allow_public: opt into a non-loopback bind on a network you control.
        token: bearer token every request must carry. **Required** for a
            non-loopback bind, and generated and printed if one is not given:
            an authless server on a network lets anyone who can reach the port
            speak in every voice on the machine, and "on a network you control"
            is a hope rather than a control. Ignored (and unnecessary) on
            loopback, where the operating system is the boundary. On a public
            bind a supplied token has to be a real secret: empty,
            whitespace-only, non-printable or shorter than
            ``_MIN_TOKEN_CHARS`` is refused at startup rather than enforced.
        first_chunk_tokens: token budget for the *first* chunk of every
            streamed passage, or ``None`` for the checkpoint's own chunking.
            Time to first audio is the first chunk's generation plus render,
            both of which scale with its length; measured at 96 tokens this
            cuts first audio ~42% on an M3 Pro and ~19% on a 3090 with CUDA
            graphs. **Opt-in because it re-fingerprints**: where the first
            split falls is audible, so it is an algorithm value, and a server
            started this way reports a different fingerprint than the
            shipping recipe. Off by default so conformance and the ports stay
            on one recipe.
    """
    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise ModuleNotFoundError(_MISSING_EXTRA) from exc

    # Refuse a non-loopback bind BEFORE paying the 747 MB load: the
    # refusal is pure config and costs nothing, so a mistake should not make
    # you wait for the model before it is reported.
    if host not in ("127.0.0.1", "localhost", "::1") and not allow_public:
        raise SystemExit(
            f"refusing to bind {host}: this server has no authentication and "
            "anyone who can reach the port can synthesise in every voice on "
            "this machine. Pass --allow-public only on a network you control."
        )

    # The same shape rule build_app applies, run here so a token that cannot be
    # a credential is refused before the 747 MB load rather than after
    # it — and so the rule has one definition for the flag, the environment and
    # the embedder alike. Only on a public bind: a token on loopback is dropped
    # a few lines down, per this parameter's documented behaviour, so judging
    # the shape of a string this function then discards would refuse a startup
    # for a value nothing reads.
    if host not in ("127.0.0.1", "localhost", "::1"):
        fault = _token_fault(token)
        if fault is not None:
            raise SystemExit(f"refusing the token given for this public bind: {fault}")

    from .. import load
    from ..hub import backend_for_device, resolve_checkpoint

    # A repo id resolves to the checkpoint *inside the snapshot the hub
    # returned*, so the default voice directory below is the snapshot's own
    # `voices/`. Handing the raw id to `Path` instead computed
    # `Path("org/repo").parent / "voices"` — a directory that does not exist —
    # and the server started with the release's voices silently absent.
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

        # `ckpt` above, not a second resolve of the repo id: the hub can hand
        # two calls two different snapshots, and then the engine's checkpoint
        # and this manifest describe different releases of one name.
        base = AlgorithmConfig.from_manifest(read_manifest(ckpt))
        # Validation lives in ChunkConfig.__post_init__ via the replace: a
        # budget outside 1..max_tokens refuses here, at startup, not on the
        # first request.
        algorithm = dataclasses.replace(
            base,
            chunking=dataclasses.replace(
                base.chunking, first_chunk_max_tokens=first_chunk_tokens
            ),
        )

    engine = load(str(ckpt), device=device, algorithm=algorithm)
    print(f"loudkit {_version()}  {engine.describe()}")
    print(f"voices: {', '.join(library.names()) or 'none in ' + str(library.root)}")
    names = library.names()
    if names:
        # First-use costs (kernel autotune, graph capture, allocator pools)
        # belong to startup, not to the first caller — see Engine.warm.
        import time as time_mod

        t0 = time_mod.perf_counter()
        engine.warm(library.load(names[0]))
        print(f"warm: first-use costs paid at startup ({time_mod.perf_counter() - t0:.1f}s)")
    public = host not in ("127.0.0.1", "localhost", "::1")
    if not public and token is not None:
        # The parameter's own docstring says a token is "Ignored (and
        # unnecessary)" on loopback, and it was not — it was enforced. So an
        # operator who dropped `--allow-public` but left `--token` behind locked
        # themselves out of their own localhost, with the help text saying the
        # flag did nothing. Honour the documented behaviour.
        token = None
    if public:
        if token is None:
            # Generated rather than refused: the operator has already said
            # --allow-public deliberately, and an authless public bind is the
            # one outcome that must not be reachable by forgetting a flag.
            token = secrets.token_urlsafe(32)
            # stderr, not stdout: under systemd or any `>` redirect, stdout is
            # the service log — a file that outlives the process, is readable by
            # anyone who can read logs, and is shipped off the host by whatever
            # collects them. The operator reading a terminal sees this either
            # way; a log aggregator does not get a credential.
            print(
                f"generated an access token for this public bind:\n  {token}",
                file=sys.stderr,
            )
        print(
            f"binding {host}: every request must carry "
            "'Authorization: Bearer <token>'. Synthesis in every voice on this "
            "machine is available to anyone holding it."
        )
        # On every public bind, not only the ones where the token was generated
        # here: a supplied token is no less readable off the wire. This server
        # takes no certificate, so the credential and the audio both cross the
        # network in clear and one capture is enough to keep the token. Said on
        # stderr for the same reason the token is — a log aggregator has no
        # business collecting either.
        print(
            "warning: this server speaks plain HTTP; the token and the audio "
            "cross the network in clear. Terminate TLS in front of it — see "
            "docs/guides/04-server-and-agents.md.",
            file=sys.stderr,
        )

    uvicorn.run(
        # --allow-public on a loopback bind is the default plus a no-op flag:
        # keep the Host pin there, because DNS rebinding does not care what
        # the operator intended.
        build_app(
            engine,
            library,
            token=token,
            allow_public=allow_public and host not in ("127.0.0.1", "::1", "localhost"),
        ),
        host=host,
        port=port,
        log_level="info",
    )


def _version() -> str:
    from .. import __version__

    return __version__
