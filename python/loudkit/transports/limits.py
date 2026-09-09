"""What every transport enforces before it touches the engine.

The caps on a request, the bounds on waiting for and holding the single-flight
engine, the bearer-token rules and the ASGI guard the HTTP app runs in front
of its routes. One module with no web framework in it, so HTTP, gRPC and MCP
refuse the same request the same way and a limit cannot be raised on one door
and left on the others. See ``docs/design/transports.md``.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import math
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING

from ..synthesis import _MAX_PREVIOUS_TOKENS, _MAX_TEXT_LEN, _MAX_WAIT_S

if TYPE_CHECKING:  # pragma: no cover - typing only
    # The ASGI vocabulary `_Guard` speaks, named rather than left as `Any`.
    # Under `TYPE_CHECKING`, so the module's rule
    # holds: no web framework is imported at run time, and a build without the
    # server extra imports this module as before.
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Everything a transport reads from this module, underscore names included:
# they are private to the package, not to this file, and a list that omitted
# them would say a peer is reaching past the seam when it is not.
# `tests/test_import_graph.py` holds the two in
# step.
__all__ = [
    "EngineBusyError",
    "EngineSlot",
    "ThreadQueue",
    "over_cap",
    "stream_lease",
    # The caps live beside the render helpers in `synthesis`, beneath every
    # transport; this is where the transports read them from.
    "_MAX_PREVIOUS_TOKENS",
    "_MAX_TEXT_LEN",
    "_MAX_WAIT_S",
    # This module's own, shared with the doors that enforce them.
    "_API_PREFIX",
    "_MAX_QUEUED",
    "_MAX_STREAM_S",
    "_SLOW_RENDER_S",
    "_Guard",
    "_host_is_loopback",
    "_queue_depth_for",
    "_token_fault",
]

_LOG = logging.getLogger("loudkit.transports.limits")

_API_PREFIX = "/v1"
"""Where the HTTP API lives. ``/health`` stays outside it: a liveness probe is
infrastructure and should not track an API version."""

_MAX_BODY_BYTES = 12 * _MAX_TEXT_LEN + 13 * _MAX_PREVIOUS_TOKENS + 4096
"""Largest request body the HTTP server reads at all.

Every capped field at its worst encoding, so a request the caller was told it
may send is never refused for its size. ``json.dumps`` escapes an astral
character as two ``\\uXXXX`` sequences, twelve bytes. A continuation id travels
as the ``int32`` the proto declares, eleven characters at its widest, and
``json.dumps`` writes ``", "`` between entries by default, so thirteen. The
4 KB covers the other fields.

The continuation term is not optional: text at its cap and ``previous_tokens``
at its cap are one legal request, the multi-call reading the API advertises,
and a bound with no room for the second half answers it 413 naming a byte
limit the caller has never been shown.
"""

_MAX_QUEUED = 32
"""How many HTTP requests may hold or wait for the engine. The holder counts:
it enters ``queued`` before the acquire and leaves it at release, so 32 is 31
waiting behind one rendering. Past this the answer is 503 with ``Retry-After``;
an unbounded queue turns a slow engine into unbounded memory with every client
still holding a connection."""

_MAX_STREAM_S = 600.0
"""Wall-clock cap on one streamed response, counted from the moment it takes
the engine. A peer that stays connected and stops reading is not a disconnect
and no poll can see it; this is the bound that gives the engine back anyway.
Far longer than any real passage, far shorter than forever."""

_SLOW_RENDER_S = 120.0
"""When ``/health`` stops answering ``ok``. A wedged render cannot be
preempted, so reporting it is the one honest move left, and the one a load
balancer can act on."""

_RATE_CAPACITY = 12
_RATE_REFILL_PER_S = 0.5
"""Token bucket per client address on a public bind: a burst, then one
synthesis every two seconds sustained. There to stop a client that has stopped
reading its own responses, not to shape legitimate use."""

_SAFE_FETCH_SITES = frozenset({"same-origin", "none"})
"""``Sec-Fetch-Site`` values that are not another page driving the request."""

_MIN_TOKEN_CHARS = 16
"""Shortest bearer token accepted. Sixteen characters of ``token_urlsafe``'s
alphabet is ~96 bits, out of reach of a network guesser; the generated default
is 43."""


def _queue_depth_for(pool_size: float) -> int:
    """How many callers may wait for the engine where a waiter costs a thread.

    Half the pool, for gRPC and MCP, which both take the engine beside the
    render rather than on the loop. The other half is what keeps answering the
    calls that need no engine while synthesis is saturated, which is what the
    depth bound is for; a bound that let the whole pool queue leaves
    ``ListVoices`` and ``list_voices`` waiting behind renders they have nothing
    to do with. An unbounded pool falls back to the HTTP depth, where a waiter
    is a coroutine and the number is chosen rather than derived.
    """
    if not math.isfinite(pool_size):
        return _MAX_QUEUED
    return max(1, int(pool_size) // 2)


def over_cap(text: str, previous_tokens: Sequence[int] | None) -> str | None:
    """Why a request is refused at the door, or ``None`` if it may proceed.

    The one check gRPC and MCP run; HTTP applies the same caps in its request
    schema, which refuses before a route exists.
    """
    if not text.strip():
        return "text is empty"
    if len(text) > _MAX_TEXT_LEN:
        return f"text is {len(text)} characters; the cap is {_MAX_TEXT_LEN}"
    if previous_tokens is not None and len(previous_tokens) > _MAX_PREVIOUS_TOKENS:
        return (
            f"previous_tokens has {len(previous_tokens)} entries; "
            f"the cap is {_MAX_PREVIOUS_TOKENS}"
        )
    return None


class EngineBusyError(Exception):
    """The engine slot was not taken: the queue is full, or the wait ran out."""

    def __init__(self, detail: str, *, retry_after: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.retry_after = retry_after


class ThreadQueue:
    """The depth bound for a transport that waits for the engine on a thread.

    Holds the count and nothing else: the acquire itself belongs to the
    transport, which has its own answer for a cancelled caller. Past the bound
    :meth:`admit` raises rather than queues, so the caller's thread goes back
    to the pool instead of parking on the engine lock, and the calls that need
    no engine keep being answered. :class:`EngineSlot` counts inside itself
    instead, because an HTTP waiter is a coroutine and costs no thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.queued = 0

    @contextmanager
    def admit(self, max_queued: int) -> Iterator[None]:
        """Take a place in the queue, or raise :class:`EngineBusyError`."""
        with self._lock:
            if self.queued >= max_queued:
                raise EngineBusyError(
                    f"{self.queued} calls are already waiting for the engine, "
                    "which is as many as may wait",
                    retry_after=1,
                )
            self.queued += 1
        try:
            yield
        finally:
            with self._lock:
                self.queued -= 1


class EngineSlot:
    """The single-flight engine, with a bounded queue in front of it.

    Taken on the event loop before any worker thread, by every HTTP route, so
    a waiter never occupies a worker it cannot advance. A ``Semaphore(1)``
    rather than a ``Lock`` because the stream route takes the slot in the
    handler and gives it back from the response task; ``max_value=1`` keeps a
    double release loud instead of admitting two callers.
    """

    def __init__(self, *, max_queued: int) -> None:
        import anyio

        self._lock = anyio.Semaphore(1, max_value=1)
        self._max_queued = max_queued
        self.queued = 0
        self.started_at: float | None = None
        """When the render holding the slot started, or ``None`` while idle."""

    async def acquire(self) -> None:
        import anyio

        if self.queued >= self._max_queued:
            raise EngineBusyError(
                f"{self.queued} requests already queued for the engine", retry_after=1
            )
        self.queued += 1
        try:
            # `_MAX_WAIT_S`: the depth bound alone lets a wedged render hold
            # every caller behind it open indefinitely.
            with anyio.fail_after(_MAX_WAIT_S):
                await self._lock.acquire()
        except TimeoutError as exc:
            self.queued -= 1
            raise EngineBusyError(
                f"waited {_MAX_WAIT_S:.0f}s for the engine and never reached it; "
                "a synthesis ahead of this one is stuck",
                retry_after=30,
            ) from exc
        except BaseException:
            self.queued -= 1
            raise
        self.started_at = time.monotonic()

    def release(self) -> None:
        self.started_at = None
        self.queued -= 1
        self._lock.release()

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, *exc: object) -> None:
        self.release()


@asynccontextmanager
async def stream_lease(release: Callable[[], Awaitable[None]]) -> AsyncIterator[None]:
    """Hold the engine for one streamed response and give it back however it ends.

    ``fail_after`` rather than the cancel flag: a peer that stays connected and
    stops reading parks the response inside the framework's ``send``, where no
    flag is read again. The timeout cancels the task, the unwind reaches the
    ``finally`` here, and ``release`` runs on every exit, including a first
    send that dies before the body generator ever starts.
    """
    import anyio

    try:
        with anyio.fail_after(_MAX_STREAM_S):
            yield
    except TimeoutError:
        _LOG.warning(
            "a stream held the engine for %.0fs without finishing; the slot was taken back",
            _MAX_STREAM_S,
        )
    finally:
        await release()


class _Buckets:
    """One token bucket per client address, bounded in clients as well as tokens.

    The map is capped and the least recently used entry dropped, so a caller
    behind a rotating source cannot mint entries; evicting the oldest can only
    grant tokens, never take them from a client still active.
    """

    __slots__ = ("_capacity", "_refill", "_max_clients", "_state")

    def __init__(self, capacity: int, refill_per_s: float, max_clients: int = 4096) -> None:
        self._capacity = float(capacity)
        self._refill = refill_per_s
        self._max_clients = max_clients
        # key -> (tokens, last seen); `take` reinserts on every hit, so the
        # first key is the least recently used.
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

    The port is stripped the way RFC 3986 says, not by splitting on ``:``: an
    IPv6 literal contains colons (``[::1]:8765``, or bare ``::1``).
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

    Two checks because they close on different browsers: ``Sec-Fetch-Site``
    names the offender outright, and requiring ``application/json`` forces a
    preflight this server never answers. Non-browser clients send the content
    type already.
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


async def _send_json(send: Send, status: int, payload: dict[str, object]) -> None:
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


class _Guard:
    """ASGI middleware: everything that must decide before a route exists.

    Host pinning on the loopback default, bearer auth, the cross-site refusal
    on the write path, the rate bucket, and a body bound applied before the
    read. Raw ASGI rather than ``@app.middleware("http")`` because that layer
    only sees a body the route below reads on demand, so nowhere in it can the
    bytes be refused before they arrive.
    """

    def __init__(
        self, app: ASGIApp, *, token: str | None = None, allow_public: bool = False
    ) -> None:
        self.app = app
        self.token = token
        self.allow_public = allow_public
        # `allow_public` alone: it is the one flag that lifts the Host pin
        # below, and until it is lifted every request this app answers came
        # from this machine. A token says something else, since it is defence
        # in depth an embedder may want on loopback too, and a bucket there
        # only sheds the operator's own thirteenth call.
        self.buckets = _Buckets(_RATE_CAPACITY, _RATE_REFILL_PER_S) if allow_public else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v for k, v in scope.get("headers", [])}

        # A browser page can resolve any hostname to 127.0.0.1 (DNS rebinding)
        # and then read answers from a server that trusts its loopback bind.
        # Pinning the accepted Host closes it; public binds are authenticated.
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
            # compare_digest, not ==: a plain comparison leaks the matched
            # prefix through timing.
            if len(supplied) != len(expected) or not hmac.compare_digest(supplied, expected):
                await _send_json(send, 401, {"detail": "unauthorized", "code": "unauthorized"})
                return

        declared = headers.get("content-length")
        # Length before value: CPython refuses to parse an integer past 4300
        # digits. A real length is twenty digits at the outside.
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

        # POST under the prefix is what costs a render: the write path gets
        # the cross-site refusal and the bucket; `GET /v1/voices` and
        # `/health` are reads a load balancer polls.
        costly = scope.get("method") == "POST" and str(scope.get("path", "")).startswith(
            _API_PREFIX
        )
        if costly:
            refusal = _cross_site_refusal(headers)
            if refusal is not None:
                await _send_json(send, *refusal)
                return

        if self.buckets is not None and costly:
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

        async def bounded_receive() -> Message:
            nonlocal seen, too_big
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > _MAX_BODY_BYTES:
                    # A chunked request declares no Content-Length, so the
                    # only place to stop it is here, mid-stream.
                    too_big = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal sent_anything
            # Swallowed once the body is over the limit: everything the app
            # says after the manufactured disconnect is a reaction to it, and
            # the 413 below could not get out once a response had started.
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


def _token_fault(token: str | None) -> str | None:
    """Why ``token`` cannot serve as a bearer credential, or ``None`` if it can.

    Every rejected shape reaches :class:`_Guard` as a token that is set, which
    switches auth on and then checks something a stranger can supply: an empty
    header, a value an HTTP header cannot carry intact, or one short enough to
    guess. Returns a phrase so each caller can wrap it in its own refusal.
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
