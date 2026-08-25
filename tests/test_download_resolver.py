"""Selective fetching, and servers that resolve voices from the snapshot.

One Hugging Face repository is the source of truth for every backend and every
port. What varies per backend is the *fetch*: the pattern sets under test here
are what `loudkit download --for ...` hands to ``snapshot_download``, asserted
without downloading anything.

The second half covers the server-side repo-id fix: a repo id given to
``serve`` or ``grpc`` must resolve its default voice directory from the
snapshot the hub returned, not from ``Path("org/repo").parent / "voices"`` —
a directory that does not exist, which started servers with the release's
voices silently absent.
"""

from __future__ import annotations

import sys

import pytest

from .assets import requires_modules
from .test_server import _engine

# ------------------------------------------------------------- pattern table


class TestReleasePatterns:
    def test_every_backend_carries_the_core_set(self) -> None:
        """All three backends read tensors from the packed checkpoint, and the
        tokenizer, the voices and the release manifest travel with it."""
        from loudkit.cli import _release_patterns

        for backend in ("torch", "onnx", "coreml"):
            allow, _ = _release_patterns(backend)
            for pattern in (
                "*.safetensors",
                "manifest.json",
                "tokenizer.json",
                "release.json",
                "voices/*",
                "SHA256SUMS",
            ):
                assert pattern in allow, (backend, pattern)

    def test_synthesis_ignores_exactly_the_two_cloning_files(self) -> None:
        """The two files a synthesis fetch must not move: the utterance voice
        encoder, and the enrollment artefact the packed checkpoint was split
        into. `*.safetensors` in the core set matches both, so the saving the
        split exists for is made here or not at all."""
        from loudkit.cli import _release_patterns

        _, ignore = _release_patterns("torch")
        assert ignore == ("ve.safetensors", "loudr-1-enrollment.safetensors")

    def test_cloning_is_what_fetches_the_enrollment_artefact(self) -> None:
        """A synthesis fetch never carries the enrollment weights.

        Cloning uncovers them for torch, which is the only backend whose
        enroller reads them; the graph ports get their enrollment graphs
        instead, which the next test covers.
        """
        from loudkit.cli import _release_patterns

        for backend in ("torch", "onnx", "coreml"):
            _, synthesis = _release_patterns(backend)
            assert "loudr-1-enrollment.safetensors" in synthesis, backend

        _, torch_cloning = _release_patterns("torch", cloning=True)
        assert torch_cloning == ()

    def test_an_unknown_backend_is_refused(self) -> None:
        """It used to be treated as ``torch`` — whose set is a strict subset of
        every other — so a typo answered with a plan that fetches no graphs at
        all, and the fetch then passed its own inventory check."""
        import loudkit.hub as hub_mod

        with pytest.raises(ValueError, match="not a backend"):
            hub_mod.release_patterns("tourch")

    def test_no_backend_fetches_another_backends_graphs(self) -> None:
        from loudkit.cli import _release_patterns

        torch_allow, _ = _release_patterns("torch", cloning=True)
        assert not any(p.startswith(("onnx/", "coreml/")) for p in torch_allow)
        onnx_allow, _ = _release_patterns("onnx", cloning=True)
        assert not any(p.startswith("coreml/") for p in onnx_allow)
        coreml_allow, _ = _release_patterns("coreml", cloning=True)
        assert not any(p.startswith("onnx/") for p in coreml_allow)

    def test_cloning_uncovers_the_torch_weights_for_torch_only(self) -> None:
        """The torch enrollment weights belong to the torch enroller.

        Python enrols through `build_torch_enroller` and nothing else, so torch
        with cloning needs them. The graph ports carry their own enrollers and
        clone through the enrollment graphs, so sending them 528 MB of torch
        weights is the overfetch the split exists to remove.
        """
        from loudkit.cli import _release_patterns
        from loudkit.hub import ENROLLMENT_NAME, VOICE_ENCODER_NAME

        _, torch_ignore = _release_patterns("torch", cloning=True)
        assert torch_ignore == ()

        for backend in ("onnx", "coreml"):
            _, ignore = _release_patterns(backend, cloning=True)
            assert ignore == (VOICE_ENCODER_NAME, ENROLLMENT_NAME), backend

    def test_the_enrollment_graphs_ride_only_with_cloning(self) -> None:
        from loudkit.cli import _release_patterns

        for backend, graphs in (
            ("onnx", ("onnx/s3_tokenizer.onnx", "onnx/camp.onnx", "onnx/voice_encoder.onnx")),
            (
                "coreml",
                (
                    "coreml/s3_tokenizer.mlpackage/*",
                    "coreml/camp.mlpackage/*",
                    "coreml/voice_encoder.mlpackage/*",
                ),
            ),
        ):
            synth, _ = _release_patterns(backend)
            with_cloning, _ = _release_patterns(backend, cloning=True)
            for graph in graphs:
                assert graph not in synth, (backend, graph)
                assert graph in with_cloning, (backend, graph)

    def test_the_hub_table_wins_when_it_exists(self, monkeypatch) -> None:
        """The canonical table belongs in ``loudkit.hub`` beside the resolver
        ``load()`` uses; the CLI's copy is the fallback until it lands, and
        must step aside the moment it does, so the two cannot drift."""
        import loudkit.hub as hub_mod
        from loudkit.cli import _release_patterns

        sentinel = (("everything",), ())

        def canonical(backend, *, cloning=False):
            del backend, cloning
            return sentinel

        monkeypatch.setattr(hub_mod, "release_patterns", canonical, raising=False)
        assert _release_patterns("torch") == sentinel

    def test_the_table_lives_in_hub(self) -> None:
        """The canonical table exists in ``loudkit.hub`` and the CLI answers
        with it verbatim — the moment `_release_patterns` was written for."""
        import loudkit.hub as hub_mod
        from loudkit.cli import _release_patterns

        for backend in ("torch", "onnx", "coreml"):
            for cloning in (False, True):
                assert _release_patterns(backend, cloning=cloning) == hub_mod.release_patterns(
                    backend, cloning=cloning
                ), (backend, cloning)


def _pack(path, *, role: str | None = None, assets: tuple[str, ...] = ()) -> None:
    """A file that reads as a loudkit checkpoint. ``role=None`` is pre-split."""
    import json

    import numpy as np
    from safetensors.numpy import save_file

    manifest: dict = {"format": "loudkit-checkpoint", "format_version": 1}
    if role is not None:
        manifest["artifact_role"] = role
    tensors = {"t3.dummy": np.zeros(2, np.float32)}
    for name in assets:
        tensors[f"assets.{name}"] = np.zeros(4, np.uint8)
    save_file(tensors, str(path), metadata={"manifest": json.dumps(manifest)})


def _core_set(root, *, omit: tuple[str, ...] = (), role: str | None = None) -> None:
    """What every backend's fetch must come back with, whatever else it holds.

    The checkpoint, the manifest and the tokenizer: the three pieces
    `verify_release_inventory` said nothing about, so a fetch that lost any of
    them printed as a usable set.
    """
    if "checkpoint" not in omit:
        _pack(root / "loudr-1.safetensors", role=role)
    for name in ("manifest.json", "tokenizer.json"):
        if name.split(".")[0] not in omit:
            (root / name).write_text("{}", encoding="utf-8")


def _mlpackage(root, stem: str, *, hollow: bool = False) -> None:
    """A CoreML package with the structure of one, or a hollow imitation."""
    package = root / "coreml" / f"{stem}.mlpackage"
    package.mkdir(parents=True)
    (package / "Manifest.json").write_text("{}", encoding="utf-8")
    if not hollow:
        data = package / "Data" / "com.apple.CoreML"
        data.mkdir(parents=True)
        (data / "model.mlmodel").write_bytes(b"m")


class TestTheInventoryChecksTheBasics:
    """A resolver may not accept a set nothing can run.

    `verify_release_inventory` judged only what varies per backend — the
    exported graphs, the voice encoder, the voices — and never asked whether
    the checkpoint, the tokenizer or the manifest had arrived. A fetch that
    came back without the weights passed here and was printed as usable.
    """

    def _torch_root(self, tmp_path, **kwargs):
        root = tmp_path / "snap"
        root.mkdir()
        (root / "voices").mkdir()
        (root / "voices" / "joe.safetensors").write_bytes(b"v")
        _core_set(root, **kwargs)
        return root

    def test_a_complete_torch_set_passes(self, tmp_path) -> None:
        import loudkit.hub as hub_mod

        root = self._torch_root(tmp_path)
        hub_mod.verify_release_inventory(root, "torch", require_voices=True)

    @pytest.mark.parametrize(
        ("omit", "named"),
        [
            ("checkpoint", "loudr-1.safetensors"),
            ("manifest", "manifest.json"),
            ("tokenizer", "tokenizer.json"),
        ],
    )
    def test_a_missing_basic_is_an_error_naming_it(self, tmp_path, omit, named) -> None:
        import loudkit.hub as hub_mod

        root = self._torch_root(tmp_path, omit=(omit,))
        with pytest.raises(FileNotFoundError, match=named):
            hub_mod.verify_release_inventory(root, "torch")

    def test_a_packed_tokenizer_needs_no_sibling(self, tmp_path) -> None:
        """A self-contained checkpoint carries `tokenizer.json` as an asset,
        which is the first branch of `Checkpoint.resolve_asset`. Requiring the
        sibling anyway would refuse a release that is complete."""
        import loudkit.hub as hub_mod

        root = self._torch_root(tmp_path, omit=("checkpoint", "tokenizer"))
        _pack(root / "loudr-1.safetensors", assets=("tokenizer.json",))
        hub_mod.verify_release_inventory(root, "torch")

    def test_a_hollow_mlpackage_is_not_a_package(self, tmp_path) -> None:
        """An ``.mlpackage`` is a directory tree with a shape: a Manifest.json
        beside a Data/ holding the model. The check was "a directory with
        anything in it", which passes for a package whose weights never came —
        the one shortfall a pattern fetch actually produces."""
        import loudkit.hub as hub_mod

        root = self._torch_root(tmp_path)
        for stem in ("flow_encoder", "flow_estimator"):
            _mlpackage(root, stem)
        _mlpackage(root, "vocoder", hollow=True)
        with pytest.raises(FileNotFoundError, match="coreml/vocoder.mlpackage"):
            hub_mod.verify_release_inventory(root, "coreml")

    def test_cloning_requires_the_enrollment_tensors(self, tmp_path) -> None:
        """`--with-cloning` against a release whose synthesis half says it is
        the synthesis half, and whose enrollment half did not come."""
        import loudkit.hub as hub_mod

        root = self._torch_root(tmp_path, omit=("checkpoint",))
        _pack(root / "loudr-1.safetensors", role="synthesis")
        (root / "ve.safetensors").write_bytes(b"ve")
        with pytest.raises(FileNotFoundError, match="loudr-1-enrollment.safetensors"):
            hub_mod.verify_release_inventory(root, "torch", cloning=True)

    def test_a_presplit_checkpoint_still_satisfies_cloning(self, tmp_path) -> None:
        """An older one-file release can clone and is not reported as short."""
        import loudkit.hub as hub_mod

        root = self._torch_root(tmp_path)
        (root / "ve.safetensors").write_bytes(b"ve")
        hub_mod.verify_release_inventory(root, "torch", cloning=True)

    def test_an_unknown_backend_is_refused_here_too(self, tmp_path) -> None:
        import loudkit.hub as hub_mod

        with pytest.raises(ValueError, match="not a backend"):
            hub_mod.verify_release_inventory(self._torch_root(tmp_path), "tourch")


class _StopResolveError(Exception):
    """Raised by a fake resolver so `load()` never builds an engine."""


@requires_modules("onnxruntime")
class TestLoadKnowsTheBackend:
    """`lk.load(repo, device="onnx")` must fetch the ONNX set.

    The resolver used to be generic: it fetched the torch set whatever the
    device, so the API path handed the ONNX backend a snapshot with no
    ``onnx/`` in it — the defect ``loudkit download --for`` was built to fix,
    still present one door over.
    """

    def _seen_backend(self, monkeypatch, device: str) -> str:
        import loudkit
        import loudkit.hub as hub_mod

        seen: dict = {}

        def fake_resolve(ref, *, revision=None, backend="torch"):
            seen["backend"] = backend
            raise _StopResolveError

        monkeypatch.setattr(hub_mod, "resolve_checkpoint", fake_resolve)
        with pytest.raises(_StopResolveError):
            loudkit.load("loudreader/loudr-1", device=device)
        return seen["backend"]

    def test_an_onnx_device_fetches_the_onnx_set(self, monkeypatch) -> None:
        assert self._seen_backend(monkeypatch, "onnx") == "onnx"

    def test_a_torch_device_fetches_the_torch_set(self, monkeypatch) -> None:
        assert self._seen_backend(monkeypatch, "cpu") == "torch"


class TestResolverFetchesAUsableSet:
    """`resolve_checkpoint` fetches by the plan and validates the receipt."""

    def _fake_snapshot(self, tmp_path, *, with_graphs: bool):
        root = tmp_path / "snap"
        root.mkdir()
        _core_set(root)
        if with_graphs:
            (root / "onnx").mkdir()
            for stem in (
                "t3_cond",
                "t3_prefill",
                "t3_step",
                "flow_encoder",
                "flow_estimator",
                "vocoder",
            ):
                (root / "onnx" / f"{stem}.onnx").write_bytes(b"graph")
        return root

    def _client(self, monkeypatch, root):
        import loudkit.hub as hub_mod

        calls: dict = {}

        class _Client:
            def snapshot_download(self, **kwargs):
                calls.update(kwargs)
                return str(root)

        client = _Client()
        monkeypatch.setattr(hub_mod, "_hub", lambda: client)
        return calls

    def test_the_backends_own_graphs_are_fetched(self, monkeypatch, tmp_path) -> None:
        import loudkit.hub as hub_mod

        root = self._fake_snapshot(tmp_path, with_graphs=True)
        calls = self._client(monkeypatch, root)
        # A third-party repo: no manifest, so the lenient path — what is under
        # test is the fetch plan and the inventory check, not the hashing.
        out = hub_mod.resolve_checkpoint("somebody/loudr-1", backend="onnx")
        assert out == root / "loudr-1.safetensors"
        assert "onnx/t3_step.onnx" in calls["allow_patterns"]
        assert calls["ignore_patterns"] == ["ve.safetensors", "loudr-1-enrollment.safetensors"]

    def test_a_fetch_short_of_the_plan_is_an_error(self, monkeypatch, tmp_path) -> None:
        """The repo held no graphs: the old resolver answered with a path that
        loads on torch and nothing else, and no one said a word."""
        import loudkit.hub as hub_mod

        root = self._fake_snapshot(tmp_path, with_graphs=False)
        self._client(monkeypatch, root)
        with pytest.raises(FileNotFoundError, match="onnx/t3_step.onnx"):
            hub_mod.resolve_checkpoint("somebody/loudr-1", backend="onnx")


# ------------------------------------------------- servers resolve snapshots


def _snapshot(tmp_path):
    """A resolved snapshot: a checkpoint file with a ``voices/`` beside it."""
    root = tmp_path / "snapshots" / "abc123"
    (root / "voices").mkdir(parents=True)
    ckpt = root / "loudr-1.safetensors"
    ckpt.write_bytes(b"never opened; load is faked")
    return ckpt


class _FakeUvicorn:
    captured: dict = {}

    @staticmethod
    def run(app, **_kwargs) -> None:
        _FakeUvicorn.captured["app"] = app


class TestHTTPServeResolvesARepoId:
    def test_the_voice_directory_comes_from_the_snapshot(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        import loudkit
        import loudkit.hub as hub_mod
        import loudkit.transports.http as server_mod

        ckpt = _snapshot(tmp_path)
        seen: dict = {}

        def fake_resolve(ref, **_kwargs):
            seen["ref"] = ref
            return ckpt

        monkeypatch.setattr(hub_mod, "resolve_checkpoint", fake_resolve)
        monkeypatch.setattr(
            loudkit, "load", lambda *a, **_k: seen.update(loaded=a) or _engine()
        )
        monkeypatch.setitem(sys.modules, "uvicorn", _FakeUvicorn)

        server_mod.serve("loudreader/loudr-1")

        assert seen["ref"] == "loudreader/loudr-1"
        # The engine loads the file the hub returned, not the raw id.
        assert seen["loaded"][0] == str(ckpt)
        # The empty snapshot voices/ is named in the banner: the library's root
        # is the snapshot's own directory, not Path("loudreader")/"voices".
        assert str(ckpt.parent / "voices") in capsys.readouterr().out

    def test_an_explicit_voices_flag_still_wins(self, tmp_path, monkeypatch, capsys) -> None:
        import loudkit
        import loudkit.hub as hub_mod
        import loudkit.transports.http as server_mod

        ckpt = _snapshot(tmp_path)
        mine = tmp_path / "mine"
        mine.mkdir()
        monkeypatch.setattr(hub_mod, "resolve_checkpoint", lambda *_a, **_k: ckpt)
        monkeypatch.setattr(loudkit, "load", lambda *_a, **_k: _engine())
        monkeypatch.setitem(sys.modules, "uvicorn", _FakeUvicorn)

        server_mod.serve("loudreader/loudr-1", voices=mine)

        assert str(mine) in capsys.readouterr().out


class _FakeGRPCServer:
    def add_insecure_port(self, address: str) -> None:
        pass

    def start(self) -> None:
        pass

    def wait_for_termination(self) -> None:
        pass


class TestGRPCServeResolvesARepoId:
    def test_the_voice_directory_comes_from_the_snapshot(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        import loudkit
        import loudkit.hub as hub_mod
        import loudkit.transports.grpc as grpc_mod

        ckpt = _snapshot(tmp_path)
        seen: dict = {}

        monkeypatch.setattr(
            hub_mod,
            "resolve_checkpoint",
            lambda ref, **_kw: seen.update(ref=ref) or ckpt,
        )
        monkeypatch.setattr(
            loudkit, "load", lambda *a, **_k: seen.update(loaded=a) or _engine()
        )
        monkeypatch.setattr(grpc_mod, "build_server", lambda *_a, **_k: _FakeGRPCServer())

        grpc_mod.serve("loudreader/loudr-1")

        assert seen["ref"] == "loudreader/loudr-1"
        assert seen["loaded"][0] == str(ckpt)
        assert str(ckpt.parent / "voices") in capsys.readouterr().out

    def test_a_missing_local_path_is_still_not_a_repo(self, monkeypatch) -> None:
        """`serve("unused.safetensors")` must not reach for the network.

        A path-shaped string is a path, however absent. It goes through the hub
        now -- every shape does, which is what stopped a local release directory
        from being read as a name -- so the guarantee is no longer "the hub is
        never consulted" but "the hub refuses it without a download". The
        failure says what was looked for, instead of the run continuing to a
        loader that fails about tensors.
        """
        import loudkit
        import loudkit.hub as hub_mod
        import loudkit.transports.grpc as grpc_mod

        def never(*_args, **_kwargs):
            raise AssertionError("a missing path was looked for on the network")

        monkeypatch.setattr(hub_mod, "snapshot_download", never, raising=False)
        monkeypatch.setattr(hub_mod, "hf_hub_download", never, raising=False)
        monkeypatch.setattr(
            loudkit, "load", lambda *_a, **_k: pytest.fail("a missing path reached the loader")
        )
        with pytest.raises(FileNotFoundError, match="unused.safetensors"):
            grpc_mod.serve("unused.safetensors")
