"""One request check for three transports.

``loudkit.transports.limits`` holds the caps every door applies. The claim
worth a test is not that each transport refuses an over-long request but that
all three refuse the *same* request with the *same* code, so a caller
switching transports keeps one vocabulary and a cap cannot be raised on one
door and left on the others.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from loudkit.transports.limits import _MAX_TEXT_LEN, over_cap

from .assets import requires_modules
from .conftest import _stub_checkpoint, _voices, chunking_engine

_TOO_LONG = "a" * (_MAX_TEXT_LEN + 1)
_CODE = "invalid_request"


def test_the_shared_check_names_the_cap() -> None:
    assert over_cap("a" * _MAX_TEXT_LEN, None) is None
    assert (
        over_cap(_TOO_LONG, None) == f"text is {_MAX_TEXT_LEN + 1} characters; the cap is 10000"
    )
    assert over_cap("   ", None) == "text is empty"
    assert over_cap("hi", [0] * 4097) == "previous_tokens has 4097 entries; the cap is 4096"


def _http_code(tmp_path) -> str:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from loudkit.transports.http import build_app

    client = TestClient(
        build_app(chunking_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
    )
    resp = client.post("/v1/synthesize", json={"text": _TOO_LONG, "voice": "fake"})
    assert resp.status_code == 422, resp.text
    return str(resp.json()["code"])


def _grpc_code(tmp_path) -> str:
    import grpc

    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc
    from loudkit.transports.grpc import build_server

    server = build_server(chunking_engine(), _voices(tmp_path))
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        with pytest.raises(grpc.RpcError) as info:
            stub.Synthesize(loudkit_pb2.SynthesizeRequest(text=_TOO_LONG, voice="fake"))
    finally:
        channel.close()
        server.stop(None)
    assert info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    codes = [str(v) for k, v in info.value.trailing_metadata() if k == "loudkit-error-code"]
    assert len(codes) == 1, codes
    return codes[0]


def _mcp_code(tmp_path) -> str:
    from loudkit.transports.mcp import build_server

    server = build_server(_mcp_holder(tmp_path), str(tmp_path), engine=chunking_engine())
    result = asyncio.run(server.call_tool("synthesize", {"text": _TOO_LONG, "voice": "fake"}))
    payload = json.loads(result.content[0].text)
    assert payload["error_kind"] == "bad_request", payload
    return str(payload["code"])


@requires_modules("fastapi", "grpc", "mcp.server.mcpserver")
def test_the_three_transports_refuse_the_same_request_with_the_same_code(tmp_path) -> None:
    """One text over the cap, three doors, one code.

    HTTP refuses it in the request schema and gRPC and MCP through
    :func:`over_cap`; all three name the refusal ``invalid_request``.

    The mark carries all three packages rather than each helper guarding
    itself: an `importorskip` inside `_grpc_code` reported this test skipped
    after its HTTP half had already passed, which reads as "two doors agree,
    the third was not asked" and is exactly the shape of a claim that stopped
    being tested. `requires_modules` also fails instead of skipping under
    `LOUDKIT_REQUIRE_ASSETS`.
    """
    _voices(tmp_path)
    assert (_http_code(tmp_path), _grpc_code(tmp_path), _mcp_code(tmp_path)) == (
        _CODE,
        _CODE,
        _CODE,
    )


def _http_voice_code(tmp_path) -> str:
    from fastapi.testclient import TestClient

    from loudkit.transports.http import build_app

    client = TestClient(
        build_app(chunking_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
    )
    resp = client.post("/v1/synthesize", json={"text": "hi", "voice": "nope"})
    assert resp.status_code == 404, resp.text
    return str(resp.json()["code"])


def _grpc_voice_code(tmp_path) -> str:
    import grpc

    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc
    from loudkit.transports.grpc import build_server

    server = build_server(chunking_engine(), _voices(tmp_path))
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        with pytest.raises(grpc.RpcError) as info:
            stub.Synthesize(loudkit_pb2.SynthesizeRequest(text="hi", voice="nope"))
    finally:
        channel.close()
        server.stop(None)
    assert info.value.code() == grpc.StatusCode.NOT_FOUND
    codes = [str(v) for k, v in info.value.trailing_metadata() if k == "loudkit-error-code"]
    assert len(codes) == 1, codes
    return codes[0]


def _mcp_voice_code(tmp_path) -> str:
    from loudkit.transports.mcp import build_server

    server = build_server(_mcp_holder(tmp_path), str(tmp_path), engine=chunking_engine())
    result = asyncio.run(server.call_tool("synthesize", {"text": "hi", "voice": "nope"}))
    payload = json.loads(result.content[0].text)
    assert payload["error_kind"] == "bad_request", payload
    return str(payload["code"])


@requires_modules("fastapi", "grpc", "mcp.server.mcpserver")
def test_the_three_transports_name_a_missing_voice_the_same_way(tmp_path) -> None:
    """The second condition all three doors can meet, and the second code.

    The cap test above passes with one code hardcoded at one door, which is
    what MCP did: `invalid_request` was written into the cap refusal by hand
    and nothing else there carried a code at all. A refusal whose code comes
    off the exception is the one that shows the catalog is actually being
    read, so this asks for a different word than the other test can produce.
    """
    _voices(tmp_path)
    assert (
        _http_voice_code(tmp_path),
        _grpc_voice_code(tmp_path),
        _mcp_voice_code(tmp_path),
    ) == ("voice_not_found", "voice_not_found", "voice_not_found")


def _warmed(monkeypatch) -> list[str]:
    """Every voice ``Engine.warm`` was handed, in order."""
    from loudkit.engine import Engine

    seen: list[str] = []
    monkeypatch.setattr(Engine, "warm", lambda _self, voice: seen.append(voice.name))
    return seen


def _mcp_holder(tmp_path):
    """The shared stub checkpoint as a string: MCP's builder takes a reference,
    not a path, because the argument may be a repo id."""
    return str(_stub_checkpoint(tmp_path))


def _build_http(tmp_path, **kw):
    pytest.importorskip("fastapi")
    from loudkit.transports.http import build_app

    return build_app(chunking_engine(), _voices(tmp_path), **kw)


def _build_grpc(tmp_path, **kw):
    pytest.importorskip("grpc")
    from loudkit.transports.grpc import build_server

    return build_server(chunking_engine(), _voices(tmp_path), **kw)


def _build_mcp(tmp_path, **kw):
    pytest.importorskip("mcp.server.mcpserver")
    from loudkit.transports.mcp import build_server

    _voices(tmp_path)  # MCP takes the directory, not the library the other two do
    return build_server(_mcp_holder(tmp_path), str(tmp_path), engine=chunking_engine(), **kw)


_BUILDERS = pytest.mark.parametrize("build", [_build_http, _build_grpc, _build_mcp])


@_BUILDERS
def test_no_door_warms_an_engine_it_was_merely_handed(tmp_path, monkeypatch, build) -> None:
    """Building a transport is not by itself a decision to spend seconds on the
    GPU: an embedder wiring one into their own process asks for the warm-up."""
    seen = _warmed(monkeypatch)
    build(tmp_path)
    assert seen == []


@_BUILDERS
def test_every_door_warms_on_the_first_roster_voice_when_asked(
    tmp_path, monkeypatch, build
) -> None:
    """One warm-up per door, on the library's first voice, so the first caller
    is not the one paying for kernel autotuning and graph capture."""
    seen = _warmed(monkeypatch)
    build(tmp_path, warm=True)
    assert seen == ["fake"]


@_BUILDERS
def test_the_environment_turns_the_warm_up_off_at_every_door(
    tmp_path, monkeypatch, build
) -> None:
    from loudkit.synthesis import NO_WARM_ENV

    monkeypatch.setenv(NO_WARM_ENV, "1")
    seen = _warmed(monkeypatch)
    build(tmp_path, warm=True)
    assert seen == []


@pytest.mark.parametrize("value", ["1", "0", "false", "no", " "])
def test_any_value_in_the_environment_turns_the_warm_up_off(
    tmp_path, monkeypatch, value
) -> None:
    """A switch, not a boolean. `LOUDKIT_NO_WARM=0` reads as "the variable is
    set", not as "warm": a deployer who wants the warm-up unsets it."""
    from loudkit.synthesis import NO_WARM_ENV

    monkeypatch.setenv(NO_WARM_ENV, value)
    seen = _warmed(monkeypatch)
    _build_http(tmp_path, warm=True)
    assert seen == []


def test_an_empty_value_is_not_a_setting(tmp_path, monkeypatch) -> None:
    """`LOUDKIT_NO_WARM=` is what an unset variable looks like to a shell that
    exported it empty, so it warms."""
    from loudkit.synthesis import NO_WARM_ENV

    monkeypatch.setenv(NO_WARM_ENV, "")
    seen = _warmed(monkeypatch)
    _build_http(tmp_path, warm=True)
    assert seen == ["fake"]


@_BUILDERS
def test_a_warm_up_that_throws_still_leaves_a_door_open(
    tmp_path, monkeypatch, caplog, build
) -> None:
    """The warm-up is an optimisation. One that cannot run must not stop a
    process that can still serve, so it is reported and the door still opens.

    Reported as a ``loudkit.synthesis`` record at WARNING, which an
    unconfigured process still prints to stderr; pytest's logging plugin
    handles it before `lastResort` can, so it is read here.
    """
    from loudkit.engine import Engine

    def _explode(_self, _voice) -> None:
        raise RuntimeError("no metal today")

    monkeypatch.setattr(Engine, "warm", _explode)
    assert build(tmp_path, warm=True) is not None
    assert "no metal today" in caplog.text


def test_a_door_that_could_not_warm_still_answers(tmp_path, monkeypatch) -> None:
    """Not merely built: the app whose warm-up failed serves the request the
    warm-up was meant to make faster."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from loudkit.engine import Engine

    def _explode(_self, _voice) -> None:
        raise RuntimeError("no metal today")

    monkeypatch.setattr(Engine, "warm", _explode)
    app = _build_http(tmp_path, warm=True)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert client.get("/v1/voices").json() == {"voices": ["fake"]}
        answer = client.post("/v1/synthesize", json={"text": "one.", "voice": "fake"})
        assert answer.status_code == 200


@_BUILDERS
def test_a_first_voice_that_will_not_load_is_not_fatal_either(
    tmp_path, monkeypatch, caplog, build
) -> None:
    """The voice the warm-up reads is part of the warm-up. A truncated profile
    in the library costs that one voice, not the whole server."""
    from loudkit.synthesis import VoiceLibrary

    monkeypatch.setattr(
        VoiceLibrary, "load", lambda _self, _name: (_ for _ in ()).throw(OSError("truncated"))
    )
    seen = _warmed(monkeypatch)
    assert build(tmp_path, warm=True) is not None
    assert seen == []
    assert "truncated" in caplog.text


@requires_modules("fastapi", "grpc", "mcp.server.mcpserver")
def test_nothing_renders_before_the_refusals_that_cost_nothing(tmp_path, monkeypatch) -> None:
    """A door that is going to refuse should refuse before it spends seconds on
    the GPU: a missing extra and a token that cannot serve are both known
    without rendering anything.

    Marked rather than guarded per door, for the reason above: a skip decided
    halfway through leaves the earlier doors' assertions counted as a skip.
    """
    seen = _warmed(monkeypatch)

    from loudkit.transports.http import build_app

    with pytest.raises(ValueError, match="refused"):
        build_app(chunking_engine(), _voices(tmp_path), token=" ", warm=True)

    import loudkit.transports.grpc as grpc_mod

    def _no_extra() -> None:
        raise ModuleNotFoundError("no grpcio")

    monkeypatch.setattr(grpc_mod, "_load_grpc", _no_extra)
    with pytest.raises(ModuleNotFoundError):
        grpc_mod.build_server(chunking_engine(), _voices(tmp_path), warm=True)

    import loudkit.transports.mcp as mcp_mod

    monkeypatch.setattr(mcp_mod, "_load_mcp", _no_extra)
    with pytest.raises(ModuleNotFoundError):
        mcp_mod.build_server(
            _mcp_holder(tmp_path), str(tmp_path), engine=chunking_engine(), warm=True
        )

    assert seen == []


def test_http_serve_asks_for_the_warm_up(tmp_path, monkeypatch) -> None:
    """`loudkit serve` reports ready on a warm engine, or the first caller pays."""
    pytest.importorskip("fastapi")
    import sys

    import loudkit
    import loudkit.transports.http as http_mod

    monkeypatch.setattr(loudkit, "load", lambda *_a, **_k: chunking_engine())
    asked: dict = {}
    monkeypatch.setattr(
        http_mod, "build_app", lambda _e, _l, **kw: asked.update(kw) or object()
    )
    monkeypatch.setitem(sys.modules, "uvicorn", type("_U", (), {"run": lambda *_a, **_k: None}))

    http_mod.serve(_stub_checkpoint(tmp_path), voices=tmp_path)
    assert asked["warm"] is True


def test_grpc_serve_asks_for_the_warm_up(tmp_path, monkeypatch) -> None:
    import loudkit
    import loudkit.transports.grpc as grpc_mod

    monkeypatch.setattr(loudkit, "load", lambda *_a, **_k: chunking_engine())
    asked: dict = {}

    class _Server:
        def add_insecure_port(self, _target) -> int:
            return 0

        def start(self) -> None: ...

        def wait_for_termination(self) -> None: ...

    monkeypatch.setattr(
        grpc_mod, "build_server", lambda _e, _l, **kw: asked.update(kw) or _Server()
    )
    grpc_mod.serve(str(_stub_checkpoint(tmp_path)), tmp_path)
    assert asked["warm"] is True


def test_mcp_run_stdio_asks_for_the_warm_up(tmp_path, monkeypatch) -> None:
    import loudkit.transports.mcp as mcp_mod

    asked: dict = {}

    class _Server:
        def run(self, *, transport: str) -> None: ...

    monkeypatch.setattr(
        mcp_mod, "build_server", lambda _c, _v, **kw: asked.update(kw) or _Server()
    )
    mcp_mod.run_stdio(str(_stub_checkpoint(tmp_path)), tmp_path)
    assert asked["warm"] is True
