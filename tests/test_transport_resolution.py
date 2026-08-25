"""What each transport asks the hub for, before it asks for anything else.

Three transports resolve a checkpoint argument, and each got it wrong the same
two ways. They passed no backend, so ``device="onnx"`` fetched the torch set: a
snapshot holding no graphs, which the backend then could not run on a release
that had downloaded perfectly. And they normalised only a repo id, so a local
*release directory* kept its own name and the default voices were looked for
one level above it.

The call under test is the one that decides where the voices live -- the one
before ``VoiceLibrary`` is built, not the later resolve inside ``load``. So the
resolver here records and answers, and the stand-in library is what ends the
call. Reading the wrong call is how an earlier version of this file passed
against the broken code: ``load`` resolves too, and correctly.

Nothing here starts a server, loads a model or touches the network.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

TRANSPORTS = ("http", "grpc", "mcp")
DEVICES = [
    ("cpu", "torch"),
    ("cuda:1", "torch"),
    ("mps", "torch"),
    ("onnx", "onnx"),
    ("coreml", "coreml"),
]


class _StopError(Exception):
    """Ends the call once both questions have been asked."""


class _Recorder:
    def __init__(self, answer: Path) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, ref: str, *, revision: str | None = None, backend: str = "torch"
    ) -> Path:
        self.calls.append({"ref": ref, "revision": revision, "backend": backend})
        return self.answer


def _release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    (root / "voices").mkdir(parents=True)
    (root / "voices" / "joe.safetensors").write_bytes(b"v")
    (root / "loudr-1.safetensors").write_bytes(b"c")
    return root


def _ask(
    transport: str, monkeypatch, ref: str, device: str | None, answer: Path
) -> tuple[_Recorder, Path | None]:
    """Drive one transport up to the moment it opens its voice library.

    Each transport imports the resolver from the hub inside its entry point, so
    the hub is where the stand-in has to sit for all three.
    """
    import loudkit.hub

    module = importlib.import_module(f"loudkit.transports.{transport}")
    recorder = _Recorder(answer)
    monkeypatch.setattr(loudkit.hub, "resolve_checkpoint", recorder)
    seen: dict[str, Path] = {}

    class _Library:
        def __init__(self, directory: Any) -> None:
            seen["voices"] = Path(directory)
            raise _StopError

    monkeypatch.setattr(module, "VoiceLibrary", _Library)
    entry = module.build_server if transport == "mcp" else module.serve
    with pytest.raises(_StopError):
        entry(ref, device=device)
    return recorder, seen.get("voices")


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize(("device", "backend"), DEVICES)
def test_the_device_decides_which_set_is_fetched(
    tmp_path: Path, monkeypatch, transport: str, device: str, backend: str
) -> None:
    root = _release(tmp_path)
    rec, _ = _ask(transport, monkeypatch, str(root), device, root / "loudr-1.safetensors")
    assert rec.calls, f"{transport} opened its voices without resolving anything"
    assert rec.calls[0]["backend"] == backend, rec.calls[0]


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("shape", ["file", "directory", "repo"])
def test_every_shape_goes_through_the_resolver(
    tmp_path: Path, monkeypatch, transport: str, shape: str
) -> None:
    """A file, a directory and a repo id all take the same door.

    Only a repo id used to, which is how a local release directory ended up
    looking for its voices one level above itself.
    """
    root = _release(tmp_path)
    ref = {
        "file": str(root / "loudr-1.safetensors"),
        "directory": str(root),
        "repo": "loudreader/loudr-1",
    }[shape]
    rec, voices = _ask(transport, monkeypatch, ref, "cpu", root / "loudr-1.safetensors")
    assert rec.calls, f"{transport} never resolved a {shape}"
    assert rec.calls[0]["ref"] == ref
    assert voices == root / "voices", voices


@pytest.mark.parametrize(("device", "backend"), DEVICES)
def test_a_pinned_revision_still_fetches_the_set_the_device_needs(
    monkeypatch, device: str, backend: str
) -> None:
    """The pin lives in the CLI, so the backend has to reach it there.

    No transport's ``serve`` takes a revision -- the CLI resolves the pin to a
    file and hands that over. Which means a pinned command asks the hub on its
    own, and asking for torch there turned ``--device onnx --revision`` into a
    download of a snapshot with no graphs in it. The unpinned form was fine,
    so pinning was the thing that broke it.
    """
    import argparse

    import loudkit.hub
    from loudkit.cli import _pinned_checkpoint

    recorder = _Recorder(Path("/tmp/loudr-1.safetensors"))
    monkeypatch.setattr(loudkit.hub, "resolve_checkpoint", recorder)
    _pinned_checkpoint(
        argparse.Namespace(checkpoint="loudreader/loudr-1", revision="v0.1", device=device)
    )
    assert recorder.calls[0]["revision"] == "v0.1", recorder.calls[0]
    assert recorder.calls[0]["backend"] == backend, recorder.calls[0]


def test_an_unpinned_checkpoint_is_left_for_the_transport(monkeypatch) -> None:
    """Without a revision the CLI must not resolve at all: that is the seam the
    transports own, and resolving twice would fetch twice."""
    import argparse

    import loudkit.hub
    from loudkit.cli import _pinned_checkpoint

    recorder = _Recorder(Path("/tmp/loudr-1.safetensors"))
    monkeypatch.setattr(loudkit.hub, "resolve_checkpoint", recorder)
    got = _pinned_checkpoint(
        argparse.Namespace(checkpoint="loudreader/loudr-1", revision=None, device="onnx")
    )
    assert got == "loudreader/loudr-1"
    assert recorder.calls == []


def test_a_ready_engine_and_named_voices_cost_no_download(monkeypatch, tmp_path) -> None:
    """`build_server` is what tests use to skip a 747 MB load.

    The checkpoint answers two questions: which weights to load, and where the
    voices are. Give an engine and a voices directory and it answers neither --
    but it was still resolved, so a repo id there fetched the release to compute
    a path that was then thrown away, and demanded the `hub` extra to do it.
    """
    import loudkit.hub
    import loudkit.transports.mcp as mcp_mod

    monkeypatch.setattr(
        loudkit.hub,
        "resolve_checkpoint",
        lambda *_a, **_k: pytest.fail("a repo id was fetched for a path nobody reads"),
    )
    voices = tmp_path / "already" / "local"
    voices.mkdir(parents=True)
    monkeypatch.setattr(mcp_mod, "_load_mcp", lambda: _StubMcpServer)
    mcp_mod.build_server("loudreader/loudr-1", str(voices), engine=object())


class _StubMcpServer:
    def __init__(self, *_a: Any, **_kw: Any) -> None:
        pass

    def tool(self, *_a: Any, **_kw: Any):  # type: ignore[no-untyped-def]
        return lambda fn: fn
