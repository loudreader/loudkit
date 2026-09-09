"""The package's public entry points: ``loudkit.load``, ``Engine.from_checkpoint``.

These are the two lines every user writes, so their *shape* is pinned here even
without weights: ``load`` returns an ``Engine`` and delegates to
``from_checkpoint``, ``from_checkpoint`` goes through the backend registry,
the registry refuses unknown devices before touching the file, and a missing
checkpoint fails loudly and named. The full weighted path — ``load`` returning
a speaking engine on each backend — is exercised by the parity, conformance
and onnx suites; this file pins the API contract that holds with no weights at
all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import loudkit
from loudkit.engine import Engine

from .assets import asset, requires
from .conftest import fake_engine, fake_voice

CKPT = asset("checkpoint")


class TestBestDevice:
    def test_returns_a_registered_device(self) -> None:
        assert loudkit.best_device() in ("cpu", "cuda", "mps")


class TestRegistryDispatch:
    def test_unknown_device_is_refused_before_the_file_is_touched(self) -> None:
        """A typo'd device must fail on the registry, not on a 747 MB file
        that was never going to be opened anyway."""
        with pytest.raises(ValueError, match="no backend for device"):
            loudkit.load("/definitely/not/a/checkpoint.safetensors", device="tpu")

    def test_missing_checkpoint_fails_loudly(self) -> None:
        with pytest.raises(FileNotFoundError):
            loudkit.load("/definitely/not/a/checkpoint.safetensors", device="cpu")

    def test_load_and_from_checkpoint_are_the_same_path(self, monkeypatch, tmp_path) -> None:
        """``loudkit.load`` is a thin, documented wrapper over
        ``Engine.from_checkpoint`` — the public surface has exactly one way to
        build an engine, so there is exactly one way to get it wrong.

        The checkpoint has to exist on disk now: ``load`` resolves its argument
        before building anything, so a path that is not there is refused before
        a backend is chosen rather than several layers in. Imported explicitly
        rather than leaning on `loudkit.backends` being populated as a side
        effect of an earlier test — which is what it used to do, and which broke
        the moment `load` started failing earlier.
        """
        import loudkit.backends

        checkpoint = tmp_path / "fake.safetensors"
        checkpoint.write_bytes(b"")
        calls: list[tuple[str, str]] = []

        def fake_build(path: str, *, device: str = "cpu", execution=None, algorithm=None):
            calls.append((path, device))
            return "engine"

        monkeypatch.setattr(loudkit.backends, "build_engine", fake_build)
        monkeypatch.setattr(loudkit, "_require_device", lambda _device: None)
        got = Engine.from_checkpoint(str(checkpoint), device="cpu")
        assert got == "engine"
        assert calls == [(str(checkpoint), "cpu")]

        # and loudkit.load routes through the same function
        calls.clear()
        got2 = loudkit.load(str(checkpoint), device="mps")
        assert got2 == "engine"
        assert calls == [(str(checkpoint), "mps")]


class TestLoadResolvesOneDevice:
    """`device=` and `execution.device` are two spellings of one decision.

    Both halves of getting that wrong have shipped. First the two were resolved
    independently, so naming both and disagreeing built an engine on one device
    and placed it on the other. Then reconciling them defaulted to `"cpu"`
    inside the helper, which is `build_engine`'s fallback and not `load`'s — so
    a caller passing an `ExecutionConfig` about something else lost the GPU
    without a word.
    """

    def _fake_build(self, monkeypatch, tmp_path):
        import loudkit.backends

        checkpoint = tmp_path / "fake.safetensors"
        checkpoint.write_bytes(b"")
        seen: list[str] = []

        def fake_build(path: str, *, device: str = "cpu", execution=None, algorithm=None):
            del path, execution, algorithm
            seen.append(device)
            return "engine"

        monkeypatch.setattr(loudkit.backends, "build_engine", fake_build)
        return checkpoint, seen

    def test_overrides_about_something_else_do_not_take_the_gpu_away(
        self, monkeypatch, tmp_path
    ) -> None:
        """`ExecutionConfig(num_threads=1)` names no device. That is the
        absence of a request, not a request for the default, which is the whole
        reason a field is `None` until named."""
        from loudkit.config import ExecutionConfig

        checkpoint, seen = self._fake_build(monkeypatch, tmp_path)
        monkeypatch.setattr(loudkit, "best_device", lambda: "cuda")
        loudkit.load(str(checkpoint), execution=ExecutionConfig(num_threads=1))
        assert seen == ["cuda"], "an override about threads must not choose the device"

    def test_a_device_named_only_in_execution_is_the_one_used(
        self, monkeypatch, tmp_path
    ) -> None:
        from loudkit.config import ExecutionConfig

        checkpoint, seen = self._fake_build(monkeypatch, tmp_path)
        monkeypatch.setattr(loudkit, "best_device", lambda: "cuda")
        loudkit.load(str(checkpoint), execution=ExecutionConfig(device="cpu"))
        assert seen == ["cpu"]

    def test_neither_spelling_takes_the_best_device(self, monkeypatch, tmp_path) -> None:
        checkpoint, seen = self._fake_build(monkeypatch, tmp_path)
        monkeypatch.setattr(loudkit, "best_device", lambda: "mps")
        loudkit.load(str(checkpoint))
        assert seen == ["mps"]

    def test_two_spellings_that_disagree_are_refused(self, monkeypatch, tmp_path) -> None:
        from loudkit.config import ExecutionConfig

        checkpoint, _seen = self._fake_build(monkeypatch, tmp_path)
        with pytest.raises(ValueError, match="disagree"):
            loudkit.load(
                str(checkpoint), device="cuda", execution=ExecutionConfig(device="cpu")
            )

    def test_the_runtime_check_reads_the_resolved_device(self, monkeypatch, tmp_path) -> None:
        """An ONNX-only install spelling its request through `execution`.

        `_require_runtime` asks for onnxruntime or torch depending on the
        device, and it used to read the `device=` argument alone — so
        `execution=ExecutionConfig(device="onnx")` on a machine with no
        torch was told to install torch.
        """
        from loudkit.config import ExecutionConfig

        checkpoint, _seen = self._fake_build(monkeypatch, tmp_path)
        asked: list[str] = []
        monkeypatch.setattr(
            loudkit, "_require_runtime", lambda device, **_: asked.append(device)
        )
        loudkit.load(str(checkpoint), execution=ExecutionConfig(device="onnx"))
        assert asked == ["onnx"]


class TestTheMissingRuntimeIsNamedBeforeTheDownload:
    """`_require_runtime` runs ahead of `resolve_checkpoint`, so an install
    without the runtime hears which extra it wants instead of paying for a
    release download and then a bare `ModuleNotFoundError` from inside the
    backend.
    """

    @staticmethod
    def _only(monkeypatch, present: str) -> None:
        """Report exactly one runtime as installed, whatever this machine has."""
        import importlib.util

        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, _package=None: object() if name == present else None,
        )

    def test_coreml_without_coremltools_names_the_coreml_extra(self, monkeypatch) -> None:
        self._only(monkeypatch, "torch")
        with pytest.raises(ModuleNotFoundError, match=r"loudkit\[coreml,audio\]"):
            loudkit._require_runtime("coreml")

    def test_coreml_asks_for_coremltools_and_not_for_torch(self, monkeypatch) -> None:
        self._only(monkeypatch, "coremltools")
        loudkit._require_runtime("coreml")
        with pytest.raises(ModuleNotFoundError):
            loudkit._require_runtime("cpu")


def _golden_fingerprint() -> str:
    """The one pin: the conformance fixture's, which every port reads too."""
    import json

    vectors = Path(__file__).resolve().parent / "data" / "conformance" / "vectors.json"
    return str(json.loads(vectors.read_text(encoding="utf-8"))["algorithm"]["fingerprint"])


@requires("checkpoint")
class TestWeightedPublicApi:
    @pytest.mark.slow
    def test_load_returns_a_speaking_engine(self) -> None:
        engine = loudkit.load(str(CKPT), device="cpu")
        assert isinstance(engine, Engine)
        # the engine speaks and the fingerprint is the conformance fixture's
        assert engine.algorithm.fingerprint() == _golden_fingerprint()

    @pytest.mark.slow
    def test_engine_describes_both_layers(self) -> None:
        engine = loudkit.load(str(CKPT), device="cpu")
        d = engine.describe()
        assert "algo[" in d
        assert "exec[" in d


class TestResolvingByName:
    """`loudkit.load("org/model")` should work on a machine that has never seen
    the model — the way every other library in this ecosystem works.

    The documented first step used to be a manual `hf download` into a directory
    the caller then had to name correctly: two commands and a path to get wrong,
    in front of the one thing a new user came to do.
    """

    def test_a_path_that_exists_is_never_a_repo_id(self, tmp_path, monkeypatch) -> None:
        """A local file wins, always. Nothing may reach for the network because
        a directory happened to be named like a repo."""
        from loudkit.hub import is_repo_id

        nested = tmp_path / "loudreader" / "loudr-1"
        nested.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        assert not is_repo_id("loudreader/loudr-1")

    def test_path_shaped_strings_are_paths(self) -> None:
        """`./model.safetensors` matches the `org/name` pattern perfectly well
        — as the repo `.` / `model.safetensors`. A misspelled filename must get
        "no such file", not a network round trip."""
        from loudkit.hub import is_repo_id

        for ref in ("./model.safetensors", "../x/y", "/abs/path", "~/model", "x.safetensors"):
            assert not is_repo_id(ref), ref

    def test_a_repo_id_is_recognised(self) -> None:
        from loudkit.hub import is_repo_id

        assert is_repo_id("loudreader/loudr-1")
        assert is_repo_id("some-org/some.model_v2")
        assert not is_repo_id("no-slash")
        assert not is_repo_id("too/many/slashes")

    def test_a_missing_path_says_so_rather_than_downloading(self) -> None:
        from loudkit.hub import resolve_checkpoint

        with pytest.raises(FileNotFoundError, match="no such file"):
            resolve_checkpoint("./definitely-not-here.safetensors")

    def test_a_directory_resolves_to_its_one_checkpoint(self, tmp_path) -> None:
        """A release is one packed checkpoint plus a voices/ directory. Voices
        are safetensors too, so the glob must not pick them up."""
        from loudkit.hub import resolve_checkpoint

        (tmp_path / "loudr-1.safetensors").write_bytes(b"")
        (tmp_path / "voices").mkdir()
        (tmp_path / "voices" / "en_klett.safetensors").write_bytes(b"")
        assert resolve_checkpoint(str(tmp_path)).name == "loudr-1.safetensors"

    def test_two_checkpoints_is_a_question_not_a_guess(self, tmp_path) -> None:
        from loudkit.hub import resolve_checkpoint

        (tmp_path / "a.safetensors").write_bytes(b"")
        (tmp_path / "b.safetensors").write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="name the one you mean"):
            resolve_checkpoint(str(tmp_path))


class TestErrorHierarchy:
    """Every loudkit error is also the builtin it replaced.

    The hierarchy exists so a boundary can classify — the HTTP server maps
    `UnsupportedLanguageError` to 400 and lets any other `NotImplementedError`
    be the 500 it is. That classification is only affordable if it costs
    existing callers nothing, so each class keeps the builtin base its raise
    site used before, and this pins that. A future class that forgets one would
    silently change what an embedder's `except ValueError` catches.
    """

    def test_every_error_is_exported(self) -> None:
        """Not a hand list. A hand list is what let two classes go missing.

        `NothingToSpeakError` and `ProvenanceError` were raised at public entry
        points, named in `docs/reference/errors.md`, and importable only from
        `loudkit.errors` — so the documented way to catch them did not work.
        Enumerating `loudkit.errors.__all__` means the next class is public the
        moment it exists, or this fails.
        """
        from loudkit import errors

        for name in errors.__all__:
            if not name.endswith("Error"):
                continue  # `error_code` is a function, not a class
            assert name in loudkit.__all__, f"{name} is not re-exported from loudkit"
            assert issubclass(getattr(loudkit, name), Exception)

    def test_the_builtin_bases_are_kept(self) -> None:
        assert issubclass(loudkit.UnsupportedLanguageError, NotImplementedError)
        assert issubclass(loudkit.VoiceNotFoundError, FileNotFoundError)
        assert issubclass(loudkit.WindowOverflowError, ValueError)
        assert issubclass(loudkit.NumberGrammarError, ValueError)

    def test_everything_is_catchable_as_one_thing(self) -> None:
        for cls in (
            loudkit.UnsupportedLanguageError,
            loudkit.VoiceNotFoundError,
            loudkit.WindowOverflowError,
            loudkit.NumberGrammarError,
        ):
            assert issubclass(cls, loudkit.LoudkitError), cls.__name__

    def test_the_window_overflow_carries_the_real_numbers(self) -> None:
        """Asserted at the raise site, not on a hand-built instance.

        Constructing the exception and reading back the keywords just passed to
        it proves the dataclass-shaped part of Python works. What is worth
        pinning is that the *engine* fills these in, and with the values a
        caller would otherwise have to parse out of the message.
        """
        from dataclasses import replace

        from loudkit.config import AlgorithmConfig

        algo = AlgorithmConfig()
        algo = algo.with_(
            window=type(algo.window)(max_speech_tokens=4),
            chunking=replace(algo.chunking, max_tokens=4, prefix_tokens=0),
            sampling=replace(algo.sampling, max_new_tokens=4),
        )
        with pytest.raises(loudkit.WindowOverflowError) as exc:
            fake_engine(algo).synthesize(
                "a b c d e f g h", fake_voice(), seed=1, single_window=True
            )
        # The fake generator emits one token per input word, so the count is
        # knowable rather than approximate.
        assert exc.value.n_tokens == 8
        assert exc.value.window == 4

    def test_an_unsupported_language_carries_the_roster(self) -> None:
        """Also at the raise site: `.supported` is what a refused client will
        retry with, so it has to be the roster the kit actually speaks."""
        from loudkit.frontend.numbers import supported_languages
        from loudkit.frontend.text import GraphemeTextFrontend

        from .assets import asset

        tokenizer = asset("tokenizer")
        if not tokenizer.exists():  # pragma: no cover - depends on assets
            pytest.skip("needs the shipped tokenizer")
        with pytest.raises(loudkit.UnsupportedLanguageError) as exc:
            GraphemeTextFrontend(str(tokenizer)).encode("Добър ден", "bg")
        assert exc.value.language == "bg"
        assert exc.value.supported == supported_languages()

    def test_a_missing_voice_names_what_was_there(self, tmp_path) -> None:
        """The one raise site where listing the alternatives is cheap."""
        from loudkit.synthesis import VoiceLibrary

        (tmp_path / "klett.safetensors").write_bytes(b"")
        (tmp_path / "savage.safetensors").write_bytes(b"")
        with pytest.raises(loudkit.VoiceNotFoundError) as exc:
            VoiceLibrary(tmp_path).load("nope")
        assert exc.value.ref == "nope"
        assert exc.value.available == ("klett", "savage")

    def test_a_typo_is_told_which_voice_was_meant(self, tmp_path) -> None:
        """The list is already in the message; the *answer* should not have to
        be read out of it. A voice name is long, lowercase and language-prefixed
        — the shape people retype wrong."""
        from loudkit.synthesis import VoiceLibrary

        (tmp_path / "en_klett.safetensors").write_bytes(b"")
        (tmp_path / "pl_zofia.safetensors").write_bytes(b"")
        with pytest.raises(loudkit.VoiceNotFoundError) as exc:
            VoiceLibrary(tmp_path).load("en_klet")
        assert "did you mean 'en_klett'?" in str(exc.value)

    def test_a_name_close_to_nothing_gets_no_guess(self, tmp_path) -> None:
        """A wrong guess is worse than none: the caller is already unsure which
        voices exist, and a confident suggestion of an unrelated one sends them
        further off."""
        from loudkit.synthesis import VoiceLibrary

        (tmp_path / "en_klett.safetensors").write_bytes(b"")
        with pytest.raises(loudkit.VoiceNotFoundError) as exc:
            VoiceLibrary(tmp_path).load("zzzzzzzz")
        assert "did you mean" not in str(exc.value)

    def test_every_error_survives_pickle_and_copy(self) -> None:
        """A worker's error has to reach the caller as itself.

        ``BaseException.__reduce__`` returns ``(cls, self.args)``, so unpickling
        calls ``cls(*args)`` — which none of these accept, because their
        diagnostics are required keyword-only arguments. Every one of them
        raised ``TypeError: __init__() missing 1 required keyword-only
        argument`` from inside pickle instead of arriving.

        That is precisely the ``ProcessPoolExecutor`` case the embedding
        tutorial points people at: errors come back from workers through
        pickle, so the message naming the bad voice was replaced by a message
        about pickle.
        """
        import copy
        import pickle

        for original in (
            loudkit.UnsupportedLanguageError("x", language="zh", supported=("en", "pl")),
            loudkit.VoiceNotFoundError("x", ref="nope", available=("a", "b")),
            loudkit.WindowOverflowError("x", n_tokens=26, window=8),
            loudkit.NumberGrammarError("x"),
            loudkit.LoudkitError("x"),
        ):
            for revived in (pickle.loads(pickle.dumps(original)), copy.copy(original)):
                assert type(revived) is type(original)
                assert str(revived) == str(original)
                assert revived.__dict__ == original.__dict__, type(original).__name__

    def test_the_number_grammar_error_is_one_class_under_two_names(self) -> None:
        """It is exported from `loudkit.frontend.numbers`, where callers know it, and
        defined in `loudkit.errors`, which `numbers` imports. Two names for one
        object — a copy would make `except NumberGrammarError` depend on which
        module the catcher imported from."""
        import loudkit.errors
        import loudkit.frontend.numbers

        assert (
            loudkit.errors.NumberGrammarError
            is loudkit.frontend.numbers.NumberGrammarError
            is loudkit.NumberGrammarError
        )


class TestDiscovery:
    """``loudkit.languages()`` and ``loudkit.voices()``: the two questions a
    reader of the README has before the second line of code, answerable without
    loading anything.

    No network here, ever. ``voices()`` is pointed at a directory laid out like
    a release, which is a shape the resolver supports for its own sake — a
    caller who unpacked a release by hand should not need a hub client to look
    inside it.
    """

    def _release(self, root, *names: str):
        (root / "loudr-1.safetensors").write_bytes(b"")
        voices = root / "voices"
        voices.mkdir()
        for name in names:
            (voices / f"{name}.safetensors").write_bytes(b"")
        return root

    def test_languages_is_the_numbers_roster(self) -> None:
        """One authority, not a copy. A grammar added or removed has to change
        this answer without anyone remembering to edit a list."""
        from loudkit.frontend.numbers import supported_languages

        assert loudkit.languages() == supported_languages()
        assert "languages" in loudkit.__all__

    def test_voices_lists_a_release_directory(self, tmp_path) -> None:
        self._release(tmp_path, "pl_zofia", "en_klett", "en_savage")
        assert loudkit.voices(repo=str(tmp_path)) == ("en_klett", "en_savage", "pl_zofia")

    def test_the_checkpoint_beside_the_voices_is_not_a_voice(self, tmp_path) -> None:
        """A release is one checkpoint and a ``voices/`` directory, and both are
        ``.safetensors``. Only the directory tells them apart."""
        self._release(tmp_path, "en_klett")
        assert loudkit.voices(repo=str(tmp_path)) == ("en_klett",)

    def test_an_empty_release_lists_nothing_rather_than_failing(self, tmp_path) -> None:
        self._release(tmp_path)
        assert loudkit.voices(repo=str(tmp_path)) == ()

    def test_no_repo_is_refused_with_an_instruction(self) -> None:
        """There is no default release, exactly as there is none for
        ``loudkit.voice()``."""
        with pytest.raises(ValueError, match="needs a repo"):
            loudkit.voices()

    def test_a_repo_id_reads_the_listing_not_the_files(self, monkeypatch) -> None:
        """Choosing between voices costs one request, not one download each.

        Also pins the filter: the checkpoint, the tokenizer and a nested path
        under ``voices/`` are all in a real release listing and none of them is
        a name ``loudkit.voice()`` would take back.
        """
        import loudkit.hub

        seen: list[tuple[str, str | None]] = []

        class FakeHub:
            @staticmethod
            def list_repo_files(*, repo_id: str, revision: str | None = None) -> list[str]:
                seen.append((repo_id, revision))
                return [
                    "loudr-1.safetensors",
                    "tokenizer.json",
                    "voices/en_savage.safetensors",
                    "voices/en_klett.safetensors",
                    "voices/archive/retired.safetensors",
                    "voices/README.md",
                ]

        monkeypatch.setattr(loudkit.hub, "_hub", lambda: FakeHub)
        assert loudkit.voices(repo="loudreader/loudr-1", revision="v1") == (
            "en_klett",
            "en_savage",
        )
        assert seen == [("loudreader/loudr-1", "v1")]

    def test_a_repo_that_is_a_file_never_reaches_the_network(
        self, monkeypatch, tmp_path
    ) -> None:
        """ "Anything that exists on disk is a path, always" — including when it
        is the wrong kind of path.

        A `repo=` pointing at an existing *file* used to fall through to the
        remote branch and ask a hub about it: the one failure mode that leaves
        the machine, reached by a caller who obviously meant something local.
        """
        import loudkit.hub

        self._release(tmp_path, "en_klett")
        monkeypatch.setattr(
            loudkit.hub,
            "_hub",
            lambda: pytest.fail("a local path reached the network"),
        )
        with pytest.raises(FileNotFoundError, match="not a release directory"):
            loudkit.voices(repo=str(tmp_path / "loudr-1.safetensors"))

    def test_a_voice_name_cannot_climb_out_of_the_release(self, monkeypatch, tmp_path) -> None:
        """A name is a name, not a path.

        The server's `VoiceLibrary` has always refused separators, because a
        request naming a filesystem path would read any `.safetensors` on the
        machine. This branch arrives at the same hole by a different door — it
        builds its path by joining rather than by looking a name up in a listing
        — so it needs the same refusal.
        """
        import loudkit.hub

        self._release(tmp_path, "en_klett")
        outside = tmp_path.parent / "OUTSIDE.safetensors"
        outside.write_bytes(b"")
        monkeypatch.setattr(
            loudkit.hub,
            "_hub",
            lambda: pytest.fail("a traversal reached the network"),
        )
        with pytest.raises(loudkit.VoiceNotFoundError, match="named, not addressed"):
            loudkit.hub.resolve_voice("../OUTSIDE", repo=str(tmp_path))

    def test_a_voice_name_carrying_nul_is_refused_by_name(self, tmp_path) -> None:
        """No path can hold a NUL, so `resolve()` inside the tree branch raises
        `lstat: embedded null character in path` — an OS-layer sentence naming
        neither the field it came from nor what to send instead. The same
        refusal `VoiceLibrary.load` makes, at the door that joins rather than
        looks up.
        """
        import loudkit.hub

        self._release(tmp_path, "en_klett")
        with pytest.raises(loudkit.VoiceNotFoundError, match="named, not addressed"):
            loudkit.hub.resolve_voice("en_kl\x00ett", repo=str(tmp_path))

    def test_a_local_release_resolves_a_voice_by_name(self, tmp_path) -> None:
        """The listing and the resolver agree about the layout, which is the
        only reason a name from one can be handed to the other."""
        from loudkit.hub import resolve_voice

        self._release(tmp_path, "en_klett")
        got = resolve_voice("en_klett", repo=str(tmp_path))
        assert got == tmp_path / "voices" / "en_klett.safetensors"


class TestAnEngineKnowsItsOwnRelease:
    """`lk.load("org/model")` then `lk.voice("joe", repo="org/model")` names the
    release twice, and the second one is the one people get wrong.

    An engine already holds the checkpoint it was built from, and a release is
    that file's directory — the same shape `loudkit.voices(repo=...)` accepts
    for an unpacked release, and the shape a hub snapshot has, because `load`
    fetches the backend's whole file set rather than one file. So the answer is
    local and costs nothing.
    """

    def _release(self, root, *names: str, checkpoint: str = "loudr-1.safetensors"):
        (root / checkpoint).write_bytes(b"")
        voices = root / "voices"
        voices.mkdir()
        for name in names:
            (voices / f"{name}.safetensors").write_bytes(b"")
        return root / checkpoint

    def _engine(self, checkpoint):
        from dataclasses import replace

        return replace(fake_engine(), checkpoint_path=str(checkpoint))

    def test_it_lists_the_voices_beside_its_checkpoint(self, tmp_path) -> None:
        engine = self._engine(self._release(tmp_path, "joe", "gosia"))
        assert engine.voices() == ("gosia", "joe")

    def test_a_name_it_does_not_have_is_refused_by_name(self, tmp_path) -> None:
        engine = self._engine(self._release(tmp_path, "joe"))
        with pytest.raises(loudkit.VoiceNotFoundError):
            engine.voice("nope")

    def test_the_checkpoint_may_be_named_anything(self, tmp_path) -> None:
        """The release is the checkpoint's directory, not a fixed filename.

        A turbo release ships a differently named checkpoint beside the same
        `voices/`, and a `voice()` that reached for `loudr-1.safetensors` would
        answer for the wrong release or for none.
        """
        ckpt = self._release(tmp_path, "joe", checkpoint="loudr-1-turbo.safetensors")
        assert self._engine(ckpt).voices() == ("joe",)

    def test_an_engine_built_by_hand_says_it_cannot_know(self, tmp_path) -> None:
        """Components assembled in a test or a notebook came from no release.
        The message has to name the call that does work, not just fail."""
        with pytest.raises(FileNotFoundError, match="loudkit.voice"):
            fake_engine().voices()


class TestCloningReachesItsWeights:
    """``loudkit.enroll`` must find the utterance voice encoder by itself.

    The encoder is not inside the packed checkpoint — it is a derived work with
    its own attribution, so a release ships it at the root beside the
    checkpoint. ``enroll`` used to build the enroller without it, so *every*
    call to the public cloning entry point raised the enroller's
    ``pass voice_encoder_weights=...`` — an argument ``enroll`` did not accept.
    The README's third example, and the feature the project leads with, could
    not be reached through the API that exists to reach it. Worse, the suite was
    green: a test asserted the RuntimeError, pinning the dead end as correct.

    No weights needed for any of this: what is under test is which path is
    resolved, which is decided before a byte is read.
    """

    def _release(self, root):
        (root / "loudr-1.safetensors").write_bytes(b"")
        (root / "ve.safetensors").write_bytes(b"")
        return root

    def test_the_encoder_is_found_beside_the_checkpoint(self, tmp_path) -> None:
        from loudkit.hub import resolve_voice_encoder

        tree = self._release(tmp_path)
        got = resolve_voice_encoder(str(tree / "loudr-1.safetensors"))
        assert got == tree / "ve.safetensors"

    def test_a_release_directory_resolves_too(self, tmp_path) -> None:
        from loudkit.hub import resolve_voice_encoder

        tree = self._release(tmp_path)
        assert resolve_voice_encoder(str(tree)) == tree / "ve.safetensors"

    def test_a_synthesis_only_release_says_so(self, tmp_path) -> None:
        """The remedy has to be one the caller can act on. The old message named
        an argument the public function did not take."""
        from loudkit.hub import resolve_voice_encoder

        (tmp_path / "loudr-1.safetensors").write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="ve.safetensors"):
            resolve_voice_encoder(str(tmp_path / "loudr-1.safetensors"))

    def test_enroll_takes_an_explicit_override(self) -> None:
        """A dev tree keeps the encoder somewhere else; the parameter exists so
        that is not a dead end either."""
        import inspect

        import loudkit

        assert "voice_encoder_weights" in inspect.signature(loudkit.enroll).parameters


class TestADeviceTheMachineLacksIsRefusedFirst:
    """Torch fails later and deeper, with an assertion from inside ``.to()``.
    ``load`` says it in one sentence before the checkpoint is touched."""

    def test_cuda_without_cuda(self, monkeypatch, tmp_path) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        with pytest.raises(ValueError, match="device 'cuda' is not available on this machine"):
            loudkit.load(str(tmp_path / "nowhere.safetensors"), device="cuda")

    def test_an_index_past_the_last_gpu(self, monkeypatch, tmp_path) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        with pytest.raises(ValueError, match="device 'cuda:1' is not available"):
            loudkit.load(str(tmp_path / "nowhere.safetensors"), device="cuda:1")

    def test_mps_without_mps(self, monkeypatch, tmp_path) -> None:
        import torch

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        with pytest.raises(ValueError, match="device 'mps' is not available"):
            loudkit.load(str(tmp_path / "nowhere.safetensors"), device="mps")

    def test_the_sentence_names_the_device_this_machine_has(
        self, monkeypatch, tmp_path
    ) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        with pytest.raises(ValueError, match=f"best device here is {loudkit.best_device()!r}"):
            loudkit.load(str(tmp_path / "nowhere.safetensors"), device="cuda")
