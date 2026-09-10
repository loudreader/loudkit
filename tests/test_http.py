"""The HTTP surface's two guards: who may write, and when a stream may stop.

``test_server.py`` covers routing and stream shape against fake weights. This
file covers the layer above and beside them — the middleware that answers
before a route exists, and the cancellation path that has to outrace a render
already running in a worker thread.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from loudkit.config import AlgorithmConfig
from loudkit.transports.http import build_app
from loudkit.transports.limits import _Guard
from loudkit.voice import VoiceProfile

# The shared weight-free set, one definition for every suite that uses it.
from .conftest import _voices, chunking_engine

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


# --- the write path -------------------------------------------------------
#
# Deliberately without an engine: what is under test is the middleware, and a
# request that reaches the route at all has already passed it.


def _guarded() -> Any:
    """A stand-in app under the real ``_Guard``, on the loopback default."""
    from fastapi import FastAPI

    app = FastAPI()

    @app.post("/v1/synthesize")
    async def _synthesize(body: dict) -> dict:
        return {"reached": True}

    @app.get("/v1/voices")
    def _voice_names() -> dict:
        return {"voices": []}

    @app.get("/health")
    def _health() -> dict:
        return {"status": "ok"}

    app.add_middleware(_Guard, token=None, allow_public=False)
    return TestClient(app, base_url="http://127.0.0.1:8765")


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_a_cross_site_post_never_reaches_the_engine(site: str) -> None:
    """A page on another origin can POST here blind; it may not spend a render.

    It never sees the response — that is what host pinning and the absent CORS
    headers already cost it — but the synthesis it starts is as expensive as a
    real one, on a server that renders one at a time.
    """
    resp = _guarded().post(
        "/v1/synthesize", json={"text": "hi"}, headers={"Sec-Fetch-Site": site}
    )

    assert resp.status_code == 403
    assert resp.json()["code"] == "cross_site"


@pytest.mark.parametrize("site", ["same-origin", "none", "NONE"])
def test_a_first_party_post_is_untouched(site: str) -> None:
    """``none`` is a user-initiated request and ``same-origin`` is our own page."""
    resp = _guarded().post(
        "/v1/synthesize", json={"text": "hi"}, headers={"Sec-Fetch-Site": site}
    )

    assert resp.status_code == 200
    assert resp.json() == {"reached": True}


def test_a_post_without_sec_fetch_site_is_untouched() -> None:
    """Every non-browser caller sends no such header, and none is in the model."""
    assert _guarded().post("/v1/synthesize", json={"text": "hi"}).status_code == 200


@pytest.mark.parametrize("media", ["text/plain", "application/x-www-form-urlencoded", ""])
def test_a_post_that_is_not_json_is_refused(media: str) -> None:
    """The form and ``no-cors`` path, closed for the browsers that send no
    ``Sec-Fetch-Site``: those three media types are the only ones such a
    request can carry, and asking for JSON forces a preflight this server
    answers with no CORS headers at all."""
    headers = {"Content-Type": media} if media else {}
    resp = _guarded().post("/v1/synthesize", content=b'{"text": "hi"}', headers=headers)

    assert resp.status_code == 415
    assert resp.json()["code"] == "unsupported_media_type"


@pytest.mark.parametrize(
    "media", ["application/json", "application/json; charset=utf-8", "application/ld+json"]
)
def test_json_content_types_are_accepted(media: str) -> None:
    """Including the parameterised and ``+json`` spellings a client may send."""
    resp = _guarded().post(
        "/v1/synthesize", content=b'{"text": "hi"}', headers={"Content-Type": media}
    )

    assert resp.status_code == 200


@pytest.mark.parametrize("path", ["/health", "/v1/voices"])
def test_the_read_path_is_not_narrowed(path: str) -> None:
    """A liveness probe and a directory read cost nothing and stay reachable.

    Both are already closed against a browser reading them — host pinning above,
    and no CORS headers — and refusing them here would take a load balancer's
    probe out with the CSRF defence.
    """
    resp = _guarded().get(path, headers={"Sec-Fetch-Site": "cross-site"})

    assert resp.status_code == 200


# --- cancelling a stream mid-chunk ----------------------------------------


_RENDER_S = 4.0
"""How long the stalled fake takes to produce its first chunk.

Long enough that "the cancellation waited for the chunk" and "the cancellation
reached the decode loop" are seconds apart in the assertion below, short enough
that the failing case is not a hang.
"""


class _StalledEngine:
    """An engine whose first chunk takes :data:`_RENDER_S` to render.

    Polls ``should_cancel`` the way the real decode loop does — every step, in
    the worker thread — and records what it saw, which is the only place the
    difference between a watcher that lived and one that died is visible.
    """

    def __init__(self) -> None:
        self.algorithm = AlgorithmConfig()
        self.started = threading.Event()
        self.saw_cancel = threading.Event()
        self.ran_to_the_end = threading.Event()

    def stream(self, text: str, voice: VoiceProfile, **kwargs: Any) -> Any:
        should_cancel = kwargs.get("should_cancel")
        deadline = time.monotonic() + _RENDER_S
        while time.monotonic() < deadline:
            self.started.set()
            if should_cancel is not None and should_cancel():
                self.saw_cancel.set()
                break
            time.sleep(0.01)
        else:
            self.ran_to_the_end.set()
        yield from ()


def _stream_scope(body: bytes) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/synthesize/stream",
        "raw_path": b"/v1/synthesize/stream",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"127.0.0.1:8765"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 51000),
        "server": ("127.0.0.1", 8765),
    }


def test_cancelling_the_response_stops_the_render_mid_chunk(tmp_path: Path) -> None:
    """A cancelled stream must stop the chunk it is inside, not wait it out.

    The watcher that flips the cancellation flag is a child of the task group
    the render runs in, so cancelling the response task cancels the watcher too
    — while ``to_thread.run_sync`` is never abandoned and keeps the render
    alive to the end of the chunk. Without the flag set on the watcher's way
    out, the thing meant to outrace the render dies first and the teardown
    waits a whole chunk: up to ten seconds of speech, of GPU time, for a client
    that is already gone.

    Driven as raw ASGI rather than through ``TestClient`` because the thing
    under test *is* the cancellation of the response task, and the test client
    has no way to raise one.
    """
    import anyio

    engine = _StalledEngine()
    app = build_app(engine, _voices(tmp_path))
    body = json.dumps({"text": "hello. world.", "voice": "fake"}).encode()

    async def drive() -> None:
        delivered = False

        async def receive() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            # A connection that is open and has nothing more to say. The client
            # never hangs up: the cancellation comes from above, which is the
            # case the polled disconnect check cannot see.
            await anyio.sleep_forever()
            raise AssertionError("unreachable")

        async def send(message: dict[str, Any]) -> None:
            return None

        # Long enough for the render to be well inside its chunk, far short of
        # the chunk itself.
        with anyio.move_on_after(0.5):
            await app(_stream_scope(body), receive, send)

    began = time.monotonic()
    anyio.run(drive)
    elapsed = time.monotonic() - began

    assert engine.started.is_set(), "the render never began; the test proves nothing"
    assert engine.saw_cancel.is_set(), "cancellation never reached the decode loop"
    assert not engine.ran_to_the_end.is_set()
    assert elapsed < _RENDER_S / 2, f"teardown waited out the chunk: {elapsed:.2f}s"


# --- what a defect and a missing route answer with -------------------------
#
# Both classes were measured against the doors rather than read off them: a
# wedged engine driven through every route, and Starlette's own router asked
# for a path and a method it does not serve.


def _wedged_client(tmp_path: Path) -> Any:
    """A client on an engine that has already wedged.

    ``Engine._wedge`` is what ``stream.py`` and both streaming routes call when
    they cannot show the token generator and the renderer were left idle. The
    engine then raises a plain ``RuntimeError`` from every synthesis, which is
    the realistic unclassified defect: not a hypothetical, the one state the
    library records on purpose.
    """
    engine = chunking_engine()
    engine._wedge("the renderer raised while holding the stream")
    return TestClient(
        build_app(engine, _voices(tmp_path)),
        base_url="http://127.0.0.1:8765",
        # What a real client sees. The default re-raises the exception into
        # the test instead, which is how a bare 500 stayed invisible here.
        raise_server_exceptions=False,
    )


def _plain_client(tmp_path: Path) -> Any:
    """A client on a working engine."""
    return TestClient(
        build_app(chunking_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
    )


def test_an_unclassified_defect_is_json_with_a_code(tmp_path: Path) -> None:
    """``docs/reference/errors.md``: ``"code"`` in every JSON error body.

    The route's ladder caught ``CancelledError``, ``UnsupportedLanguageError``
    and ``ValueError``, so a ``RuntimeError`` -- the wedge's own type -- fell
    past it to Starlette, which answers ``text/plain`` "Internal Server Error"
    with no ``code`` and no log line. The streaming route classified the same
    failure correctly, so the two doors disagreed about the same engine.
    """
    resp = _wedged_client(tmp_path).post(
        "/v1/synthesize", json={"text": "one. two.", "voice": "fake"}
    )

    assert resp.status_code == 500
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json()["code"] == "server_fault"
    assert "wedge" not in resp.text, "the defect's own message reached the caller"


def test_the_openai_route_re_dresses_a_defect_too(tmp_path: Path) -> None:
    """An OpenAI client reads ``error.message``; it never sees ``detail``."""
    resp = _wedged_client(tmp_path).post(
        "/v1/audio/speech", json={"input": "one. two.", "voice": "fake"}
    )

    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "server_error"


def test_health_reports_a_wedged_engine(tmp_path: Path) -> None:
    """503, not ``ok``.

    ``/health`` read only how long the slot had been held, so an engine that
    answers every synthesis with a fault and cannot be reclaimed still
    reported ``{"status": "ok"}`` -- and a load balancer kept routing to a
    process only a restart can fix.
    """
    resp = _wedged_client(tmp_path).get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "wedged"
    assert "unusable" in body["detail"], "the probe does not say what is wrong"
    # A wedge is terminal: `Retry-After` would say waiting is the remedy.
    assert "retry-after" not in {k.lower() for k in resp.headers}


def test_health_is_ok_on_an_engine_that_has_not_wedged(tmp_path: Path) -> None:
    """The other half of the claim, so the test above can fail."""
    resp = _plain_client(tmp_path).get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [("get", "/v1/nope", 404), ("get", "/v1/synthesize", 405)],
)
def test_a_missing_route_carries_a_code(
    tmp_path: Path, method: str, path: str, status: int
) -> None:
    """Starlette's router raises its own ``HTTPException``, not FastAPI's.

    The handler was registered for FastAPI's subclass, and Starlette dispatches
    on the raised exception's own MRO, so a 404 and a 405 answered with
    ``detail`` and no ``code`` at all. Registering the base class covers both,
    because the routes' ``_refuse`` raises the subclass.
    """
    resp = getattr(_plain_client(tmp_path), method)(path)

    assert resp.status_code == status
    assert resp.json()["code"] == "invalid_request"


def test_a_nul_in_a_voice_name_is_refused_as_a_name(tmp_path: Path) -> None:
    """No path can hold a NUL, so the filesystem refused it -- in its words.

    ``lstat: embedded null character in path`` reached the caller verbatim on
    both doors: an OS-layer sentence naming neither the field it came from nor
    what to send instead.
    """
    resp = _plain_client(tmp_path).post(
        "/v1/synthesize", json={"text": "hi", "voice": "fa\x00ke"}
    )

    assert resp.status_code == 400
    assert resp.json()["detail"].startswith("not a voice name:")
    assert "lstat" not in resp.text
