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

import numpy as np
import pytest

from loudkit.config import AlgorithmConfig
from loudkit.synthesis import VoiceLibrary
from loudkit.transports.http import _Guard, build_app
from loudkit.voice import VoiceProfile

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


def _voice() -> VoiceProfile:
    return VoiceProfile(
        name="fake",
        speaker_embedding=np.full(256, 0.0625, np.float32),
        flow_embedding=np.full(192, 0.0625, np.float32),
        prompt_tokens=np.zeros(8, np.int64),
        prompt_mel=np.zeros((80, 16), np.float32),
        cond_prompt_tokens=np.zeros(8, np.int64),
    )


def _voices(tmp_path: Path) -> VoiceLibrary:
    _voice().save(tmp_path / "fake.safetensors")
    return VoiceLibrary(tmp_path)


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
