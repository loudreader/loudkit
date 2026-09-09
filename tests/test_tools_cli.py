"""Command lines of the scripts in tools/ and research/, and the one download
they perform.

Three benchmark scripts used to read ``sys.argv`` by index. That parser shipped
a real bug — ``--seed N`` puts the flag at ``argv[5]`` and the value at
``argv[6]``, so the guard that read ``argv[5]`` was always False and every run
used seed 7 while the command line said otherwise — and nothing failed, because
a hand-rolled parser has no error path to test. These cases pin the parsers:
the defaults, and a refusal for each value the old code would have swallowed.

The fetch cases pin the other half: a lexicon download that cannot stall
forever, and cannot leave a partial or wrong file under the final name.
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
from pathlib import Path

import pytest

from .conftest import tool

REPO = Path(__file__).resolve().parent.parent


def _script_prose() -> list[tuple[str, str]]:
    """Every comment and docstring under tools/ and research/, with where it lives.

    String literals are excluded on purpose: several of them are fixture notes
    that ship inside `tests/data/conformance/speechtext.json`, and a prose rule
    must not move an artefact's bytes.
    """
    import ast
    import io
    import tokenize

    out: list[tuple[str, str]] = []
    for root in ("tools", "research"):
        for path in sorted((REPO / root).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            rel = path.relative_to(REPO)
            for node in ast.walk(ast.parse(source)):
                if isinstance(
                    node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                ):
                    doc = ast.get_docstring(node)
                    if doc:
                        out.append((f"{rel}:{getattr(node, 'name', '<module>')}", doc))
            for tok in tokenize.generate_tokens(io.StringIO(source).readline):
                if tok.type == tokenize.COMMENT:
                    out.append((f"{rel}:{tok.start[0]}", tok.string))
    return out


class TestTheScriptsAreHeldToTheProseRules:
    """`tests/test_hygiene.py` walks `python/loudkit` only.

    Nothing held the scripts that build what ships to the same rules, and 155
    em dashes and a pointer to a page that does not exist had accumulated
    there. These two cases are the gate.
    """

    def test_no_comment_or_docstring_carries_an_em_dash(self) -> None:
        """The owner's style rule, and prose the owner ships."""
        hits = [where for where, text in _script_prose() if "—" in text]
        assert not hits, "em dashes in tools/ or research/ prose:\n  " + "\n  ".join(hits)

    def test_every_documentation_page_a_script_names_exists(self) -> None:
        """A pointer a reader cannot follow is worse than no pointer.

        `amend_manifest.py` named `docs/design/parity.md`, which has never been
        in this tree.
        """
        import re

        pattern = re.compile(r"docs/[\w./-]+\.md")
        missing = sorted(
            {
                f"{where}: {name}"
                for where, text in _script_prose()
                for name in pattern.findall(text)
                if not (REPO / name).is_file()
            }
        )
        assert not missing, "documentation pages named in tools/ or research/ prose:\n  " + (
            "\n  ".join(missing)
        )


class TestBenchParsers:
    def test_seed_is_read_wherever_the_flag_sits(self) -> None:
        parser = tool("bench_cuda_box")._parser()
        args = parser.parse_args(["ckpt", "voice", "cuda:1", "out", "--seed", "11"])
        assert args.seed == 11
        assert args.device == "cuda:1"
        assert args.cuda_graphs is False

    def test_bench_defaults(self) -> None:
        args = tool("bench_cuda_box")._parser().parse_args(["c", "v", "cpu", "o"])
        assert (args.seed, args.cuda_graphs) == (7, False)
        assert args.outdir == Path("o")

    @pytest.mark.parametrize(
        "argv",
        [
            ["c", "v", "cpu", "o", "--seed"],  # trailing flag, no value
            ["c", "v", "cpu", "o", "--seed", "half"],
            ["c", "v", "cpu"],  # one positional short
        ],
    )
    def test_bench_refuses_bad_command_lines(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit):
            tool("bench_cuda_box")._parser().parse_args(argv)

    def test_batch_list_default_and_parse(self) -> None:
        mod = tool("bench_batch")
        parser = mod._parser()
        assert parser.parse_args(["c", "v", "cuda", "o"]).batches == list(mod.DEFAULT_BATCHES)
        assert parser.parse_args(["c", "v", "cuda", "o", "1,4,16"]).batches == [1, 4, 16]

    def test_batch_list_refuses_a_non_integer(self) -> None:
        with pytest.raises(SystemExit):
            tool("bench_batch")._parser().parse_args(["c", "v", "cuda", "o", "1,2,x"])

    # Both scripts build their parser from research/_bench.py, but each passes
    # its own DEFAULT_BATCHES and device help in. Held to the same two cases as
    # bench_batch above, so neither wiring can be broken without a red test.
    def test_render_batch_list_default_and_parse(self) -> None:
        mod = tool("bench_render")
        parser = mod._parser()
        assert parser.parse_args(["c", "v", "mps", "o"]).batches == list(mod.DEFAULT_BATCHES)
        assert parser.parse_args(["c", "v", "cpu", "o", "1,4,16"]).batches == [1, 4, 16]

    def test_render_batch_list_refuses_a_non_integer(self) -> None:
        with pytest.raises(SystemExit):
            tool("bench_render")._parser().parse_args(["c", "v", "cpu", "o", "1,2,x"])


class TestRoundtripParser:
    def test_score_takes_two_paths(self) -> None:
        args = tool("eval_roundtrip")._parser().parse_args(["--score", "out", "t.json"])
        assert args.score == [Path("out"), Path("t.json")]

    def test_render_takes_three_positionals(self) -> None:
        args = tool("eval_roundtrip")._parser().parse_args(["ckpt", "voice", "out"])
        assert (args.checkpoint, args.voice, args.out_dir) == ("ckpt", "voice", Path("out"))

    def test_render_with_a_missing_positional_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["eval_roundtrip.py", "ckpt", "voice"])
        with pytest.raises(SystemExit):
            tool("eval_roundtrip").main()


class TestDevicePack:
    def test_out_is_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No default: the app it was written for is not in this repo, and a
        default pointing at it staged 356 MB into a path only one machine has."""
        monkeypatch.setattr(sys, "argv", ["make_device_pack.py", "--checkpoint", "x"])
        with pytest.raises(SystemExit):
            tool("make_device_pack").main()


class TestAcceptance:
    def test_speak_defaults_to_the_readme_runtime_extras(self) -> None:
        mod = tool("acceptance")
        assert mod.requested_extras("", speak=True) == "torch,audio,hub"
        assert mod.requested_extras("onnx,audio,hub", speak=True) == ("onnx,audio,hub")
        assert mod.requested_extras("", speak=False) == ""

    def test_speak_gate_names_the_downloaded_release_and_a_real_voice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = tool("acceptance")
        calls: list[list[str]] = []

        def fake_run(
            cmd: list[str],
            *,
            cwd: Path,
            env_note: str = "",
        ) -> str:
            calls.append(cmd)
            if len(cmd) > 1 and cmd[1] == "speak":
                output = Path(cmd[cmd.index("--out") + 1])
                output.write_bytes(b"x" * 1025)
            return ""

        monkeypatch.setattr(mod, "run", fake_run)
        bindir = tmp_path / "bin"
        repo = "loudreader/loudr-1"
        mod.check_speaks(bindir, tmp_path, repo)

        exe = str(bindir / ("loudkit.exe" if sys.platform == "win32" else "loudkit"))
        assert calls == [
            [exe, "download", repo],
            [
                exe,
                "speak",
                "--checkpoint",
                repo,
                "--voice",
                "joe",
                "--out",
                str(tmp_path / "acceptance.wav"),
                "The clean room speaks.",
            ],
            [exe, "verify", str(tmp_path / "acceptance.wav")],
        ]


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._buf = io.BytesIO(payload)

    def read(self, size: int) -> bytes:
        return self._buf.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture
def serving(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    def serve(payload: bytes) -> dict[str, object]:
        def fake_urlopen(url: str, timeout: float | None = None) -> _FakeResponse:
            calls["url"], calls["timeout"] = url, timeout
            return _FakeResponse(payload)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        return calls

    return serve


class TestFetchLexicons:
    def test_the_download_carries_a_timeout(
        self, tmp_path, serving, capsys, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """20-100 MB over one socket with no timeout has no failure mode except
        the operator noticing."""
        mod = tool("fetch_nst_lexicons")
        payload = b"lexicon" * 100
        calls = serving(payload)
        monkeypatch.setattr(mod, "MIN_BYTES", 8)
        mod.fetch("sv", tmp_path)

        assert calls["timeout"] == mod.TIMEOUT_S
        assert (tmp_path / "nst_sv.tar.gz").read_bytes() == payload
        assert not list(tmp_path.glob("*.part"))
        assert hashlib.sha256(payload).hexdigest() in capsys.readouterr().out

    def test_an_error_page_never_lands_under_the_lexicon_name(self, tmp_path, serving) -> None:
        """An HTTP error page or captive-portal splash arrives with a 200 and is
        kilobytes; the smallest real lexicon is ~20 MB."""
        mod = tool("fetch_nst_lexicons")
        serving(b"<html>404</html>")
        with pytest.raises(SystemExit, match="too small"):
            mod.fetch("da", tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_a_digest_that_misses_the_pin_is_discarded(
        self, tmp_path, serving, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = tool("fetch_nst_lexicons")
        serving(b"not the lexicon")
        monkeypatch.setattr(mod, "MIN_BYTES", 8)
        monkeypatch.setitem(mod.SOURCES, "no", (mod.SOURCES["no"][0], "0" * 64))
        with pytest.raises(SystemExit, match="does not match the pin"):
            mod.fetch("no", tmp_path)
        assert list(tmp_path.iterdir()) == []


class TestTheEstimatorSwapCannotChangeTheAlgorithm:
    """`export_coreml.py --estimator-ckpt` traces distilled weights against a
    checkpoint it was handed, and the Euler step count comes from *that*
    checkpoint's manifest.

    The recipe is explicit that a step-distilled estimator run at any other K
    is a different algorithm rather than the same one at a different speed. So
    pointing the flag at a stock `loudr-1` (K=2) with a K=1 estimator exported
    a package that ran the distilled estimator twice, printed `euler=2`, and
    said nothing — and the CoreML package carries no K of its own to catch it
    downstream, because the graph is one step and the count lives in the
    engine's config.
    """

    @staticmethod
    def _guard(blob, euler, declared_k=None):
        import argparse
        from dataclasses import replace

        from loudkit.config import AlgorithmConfig

        export_coreml = tool("export_coreml")

        algo = replace(AlgorithmConfig(), euler_steps=euler)
        args = argparse.Namespace(
            estimator_k=declared_k, estimator_ckpt="best.pt", checkpoint="ckpt.safetensors"
        )
        return export_coreml._check_estimator_k(blob, algo, args)

    def test_a_k_mismatch_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="distilled for K=1"):
            self._guard({"cfg": {"k_student": 1}}, euler=2)

    def test_a_match_is_accepted(self) -> None:
        self._guard({"cfg": {"k_student": 1}}, euler=1)
        self._guard({"k_student": 2}, euler=2)

    def test_a_silent_checkpoint_must_be_declared(self) -> None:
        """Refused rather than guessed, the same shape as `pack_turbo.py`
        taking `n_cfm_timesteps` from a declared recipe."""
        with pytest.raises(SystemExit, match="does not record the Euler step count"):
            self._guard({}, euler=2)
        self._guard({}, euler=1, declared_k=1)
        with pytest.raises(SystemExit, match="distilled for K=1"):
            self._guard({}, euler=2, declared_k=1)

    def test_the_declaration_may_not_contradict_the_file(self) -> None:
        with pytest.raises(SystemExit, match="contradicts the checkpoint"):
            self._guard({"k_student": 1}, euler=1, declared_k=2)


class TestOneParserAnswersHowManyStepsAnEstimatorWasDistilledFor:
    """`pack_turbo.py` and `export_coreml.py` both swap in a distilled
    estimator, and both have to know its K before they do.

    They used to answer differently, which is how the hole opened. The exporter
    read the checkpoint file and refused a mismatch; the packer did not read it
    at all and trusted `--n-cfm-timesteps`, whose default is 2. A K=1 estimator
    packed that way produced a checkpoint declaring K=2, and every later guard
    believed the manifest — including the exporter's, because by then the
    estimator was inside the checkpoint and there was nothing left to
    cross-check it against.
    """

    @staticmethod
    def _parser():
        return tool("distill_meta")

    def test_the_step_count_is_found_wherever_a_training_run_parks_it(self) -> None:
        """Several spellings, because the training side is a research tree whose
        checkpoints predate anyone needing to read this back."""
        k = self._parser().estimator_k
        assert k({"k_student": 3}) == 3
        assert k({"cfg": {"n_cfm_timesteps": 4}}) == 4
        assert k({"hparams": {"euler_steps": 5}}) == 5
        assert k({"step": 900, "score": 0.99}) is None

    def test_a_step_count_that_is_not_a_step_count_is_refused_not_skipped(self) -> None:
        """Skipping a malformed declaration is how a contradiction got through.

        `True` is an `int` in Python and is not a step count, and `0` is not one
        either. Both used to be stepped over, so a file carrying `k_student = 1`
        beside a nested `k_student = 0` answered 1 -- the self-contradiction the
        function above exists to refuse, arriving through the one branch that
        did not look. A declaration that is present and unreadable is refused.
        """
        k = self._parser().estimator_k
        for blob in ({"k": True}, {"k_student": 0}, {"cfg": {"euler_steps": -1}}):
            with pytest.raises(ValueError, match="not a positive integer"):
                k(blob)
        with pytest.raises(ValueError, match="not a positive integer"):
            k({"k_student": 1, "cfg": {"k_student": 0}})

    def test_a_file_that_contradicts_itself_has_no_step_count(self) -> None:
        """Returning the first hit made the answer depend on search order.

        A checkpoint carrying `k = 1` at the top level and `cfg.k_student = 2`
        inside resolved to whichever the walk reached first, so a file that
        disagreed with itself answered as though it did not.
        """
        k = self._parser().estimator_k
        with pytest.raises(ValueError, match="more than one Euler step count"):
            k({"k": 1, "cfg": {"k_student": 2}})
        # Agreeing twice is not contradicting.
        assert k({"k": 1, "cfg": {"k_student": 1}}) == 1

    def test_the_packer_refuses_an_estimator_that_does_not_belong_at_its_k(self) -> None:
        """The path with no recipe, which is the one that was unguarded.

        `tools/recipes/loudr-1-turbo.json` pins K=1 and hashes every input, so
        the recipe path was always safe. `--estimator ... --n-cfm-timesteps 2`
        was not, and it is still a supported way to build.
        """
        import types

        pack_turbo = tool("pack_turbo")

        # A stand-in for torch: the guard runs before any tensor is touched.
        loaded = {"ema_sd": {}, "cfg": {"k_student": 1}}
        fake_torch = types.SimpleNamespace(load=lambda *_a, **_kw: loaded)
        with pytest.raises(SystemExit, match="distilled for K=1"):
            pack_turbo.build_s3gen_tensors(Path("s3gen"), Path("best.pt"), fake_torch, 2)

    def test_the_packer_refuses_a_step_count_below_one(self) -> None:
        """`--n-cfm-timesteps 0` wrote a checkpoint nothing could load.

        The number goes into the manifest, and `AlgorithmConfig` refuses
        `euler_steps < 1` at load time — so the build succeeded and the failure
        arrived at whoever downloaded it, where it cannot be fixed. Checked
        before any weight is read, like the two refusals above.
        """
        import types

        pack_turbo = tool("pack_turbo")

        fake_torch = types.SimpleNamespace(load=lambda *_a, **_kw: {"ema_sd": {}})
        for bad in (0, -1):
            with pytest.raises(SystemExit, match="must be at least 1"):
                pack_turbo.build_s3gen_tensors(Path("s3gen"), Path("best.pt"), fake_torch, bad)

    def test_the_packer_refuses_to_guess_a_k_the_estimator_does_not_record(self) -> None:
        """The hole a default left open, which is worse than the one above.

        `--n-cfm-timesteps` defaulted to 2. An estimator whose file records
        nothing — a research checkpoint from a tree that predates anyone
        needing to read this back — therefore packed as K=2 whatever it was,
        and the manifest is what every later guard believes, including the
        CoreML exporter's own. A K=1 estimator declared K=2 renders a one-step
        estimator integrated twice, and nothing downstream can see it, because
        the checkpoint agrees with itself.

        `export_coreml.py` already refused this case. There is no default here
        now, and stating the number is how a metadata-less estimator gets
        packed.
        """
        import types

        pack_turbo = tool("pack_turbo")

        silent = {"ema_sd": {}}  # no k anywhere
        fake_torch = types.SimpleNamespace(load=lambda *_a, **_kw: silent)
        with pytest.raises(SystemExit, match="does not record the Euler step count"):
            pack_turbo.build_s3gen_tensors(Path("s3gen"), Path("best.pt"), fake_torch, None)

    def test_the_packer_refuses_a_build_with_no_estimator_and_no_k(self) -> None:
        """The manifest declares the step count and there is no file to read
        it from, so there is nothing to write."""
        import types

        pack_turbo = tool("pack_turbo")

        fake_torch = types.SimpleNamespace(load=lambda *_a, **_kw: {})
        with pytest.raises(SystemExit, match="required when no --estimator"):
            pack_turbo.build_s3gen_tensors(Path("s3gen"), None, fake_torch, None)

    def test_the_flag_has_no_default(self) -> None:
        """A default is what made the guess possible, so its absence is pinned
        rather than left to a reading of the argparse block."""
        source = (REPO / "tools" / "pack_turbo.py").read_text(encoding="utf-8")
        assert '"--n-cfm-timesteps",\n        type=int,\n        default=None,' in source


class TestARendererStateDictThatDoesNotFitIsRefused:
    """`pack_turbo.py` loads the stock renderer with `strict=False`.

    The two key lists it returns were discarded, so a tensor absent from
    `s3gen.safetensors` under the expected name kept the module's random
    initialisation and packed as weights. Nothing downstream reads weights for
    correctness: the release gate checks determinism, and the conformance
    fixture is generated from the same packed file, so all five ports would
    agree on it. The two checkpoints that were actually built were compared
    tensor by tensor against their sources and nothing was random, so these
    cases are about what the tool allows, not about what shipped.
    """

    @staticmethod
    def _packer():
        return tool("pack_turbo")

    def test_a_missing_tensor_is_refused_and_named(self) -> None:
        """Named, because "the file does not fit" is not actionable and the
        key is what says which s3gen file this is."""
        with pytest.raises(SystemExit, match=r"1 missing \(flow\.decoder\.estimator\.mid"):
            self._packer().refuse_partial_load(
                Path("s3gen.safetensors"), ["flow.decoder.estimator.mid.weight"], []
            )

    def test_an_unexpected_tensor_is_refused_too(self) -> None:
        """A source carrying names this renderer does not have is not the
        renderer the build thinks it is packing."""
        with pytest.raises(SystemExit, match=r"1 unexpected \(flow\.decoder\.spare"):
            self._packer().refuse_partial_load(
                Path("s3gen.safetensors"), [], ["flow.decoder.spare.weight"]
            )

    def test_a_long_list_is_counted_rather_than_printed_in_full(self) -> None:
        """A whole-file mismatch is hundreds of keys; the refusal stays a
        message, and the count is the part that says which failure it is."""
        with pytest.raises(SystemExit, match=r"9 missing \(.*\+4 more\)"):
            self._packer().refuse_partial_load(
                Path("s3gen.safetensors"), [f"k{i}" for i in range(9)], []
            )

    def test_a_state_dict_that_fits_is_not_refused(self) -> None:
        """The passing side, so the guard cannot be satisfied by refusing
        every build."""
        assert self._packer().refuse_partial_load(Path("s3gen.safetensors"), [], []) is None

    def test_the_packer_refuses_an_s3gen_file_that_is_short_a_tensor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wiring, not just the helper: `build_s3gen_tensors` has to look at
        what `load_state_dict` returned before it folds and packs.

        `chatterbox` is a research dependency that is not importable here, so
        the renderer and the safetensors reader are stood in for. The stand-in
        reports one missing key, which is the case that used to pack the
        module's random initialisation as weights.
        """
        import types

        packer = self._packer()

        class _Renderer:
            def load_state_dict(self, state, strict=True):
                return types.SimpleNamespace(
                    missing_keys=["flow.decoder.estimator.mid.weight"], unexpected_keys=[]
                )

        s3gen_mod = types.ModuleType("chatterbox.models.s3gen.s3gen")
        s3gen_mod.S3Token2Wav = _Renderer
        for name in ("chatterbox", "chatterbox.models", "chatterbox.models.s3gen"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "chatterbox.models.s3gen.s3gen", s3gen_mod)

        import safetensors.torch

        monkeypatch.setattr(safetensors.torch, "load_file", lambda _p: {})

        # No estimator, so the K guards pass on `--n-cfm-timesteps` alone and
        # the renderer load is the only thing left between here and the pack.
        fake_torch = types.SimpleNamespace(load=lambda *_a, **_kw: {})
        with pytest.raises(SystemExit, match="is not the state dict this renderer expects"):
            packer.build_s3gen_tensors(Path("s3gen.safetensors"), None, fake_torch, 1)


class TestASubsetPackDoesNotStrandADigest:
    """`pack_assets.py --only` re-packs some of the assets, not all of them.

    Every `assets.*` tensor was dropped from the copy and only the selected
    ones were put back, while the manifest kept the previous run's digest for
    the dropped ones. The output then declared a `pl_en_respell_sha256` for
    bytes it no longer carried, which is what the loader checks a sibling copy
    against, and the opposite of the idempotence the comment claimed.
    """

    @staticmethod
    def _checkpoint(tmp_path: Path, assets: dict[str, bytes]) -> Path:
        """A minimal packed checkpoint, its manifest carrying each digest."""
        import json

        import numpy as np
        from safetensors.numpy import save_file

        from loudkit.checkpoint import ASSET_PREFIX

        tensors = {"t3.dummy": np.zeros(2, np.float32)}
        manifest: dict[str, object] = {"format": "loudkit-checkpoint", "format_version": 1}
        keys = {
            "tokenizer.json": "tokenizer_sha256",
            "pl_en_respell.json": "pl_en_respell_sha256",
        }
        for name, payload in assets.items():
            tensors[f"{ASSET_PREFIX}{name}"] = np.frombuffer(payload, dtype=np.uint8)
            manifest[keys[name]] = hashlib.sha256(payload).hexdigest()
        manifest["packed_assets"] = sorted(assets)
        path = tmp_path / "packed.safetensors"
        save_file(tensors, str(path), metadata={"manifest": json.dumps(manifest)})
        return path

    def test_an_asset_only_did_not_select_is_carried_through_with_its_digest(
        self, tmp_path: Path
    ) -> None:
        """The lexicon is not named on the command line, so it comes out
        unchanged and its digest still describes what is in the file."""
        packer = tool("pack_assets")
        source = self._checkpoint(
            tmp_path, {"tokenizer.json": b"old", "pl_en_respell.json": b'{"respell":{}}'}
        )
        (tmp_path / "tokenizer.json").write_bytes(b"new")

        out = tmp_path / "repacked.safetensors"
        packer.pack(source, out, only={"tokenizer.json"})

        from loudkit.checkpoint import Checkpoint

        final = Checkpoint.open(out)
        assert final.asset("tokenizer.json") == b"new", "the named asset was not re-packed"
        assert final.asset("pl_en_respell.json") == b'{"respell":{}}', (
            "an asset --only did not name was dropped from the copy"
        )
        assert (
            final.manifest["pl_en_respell_sha256"]
            == hashlib.sha256(b'{"respell":{}}').hexdigest()
        ), "the manifest declares a digest for bytes the file does not carry"
        assert final.manifest["packed_assets"] == ["pl_en_respell.json", "tokenizer.json"], (
            "packed_assets lists what this run wrote rather than what the file holds"
        )


class TestACoreMLPackageSetCameFromOneExport:
    """Three `.mlpackage` directories in one folder look like a set.

    `--stages` takes a list, so a run naming two stages leaves the third as it
    was; and it took *any* word, so a typo skipped its stage and said nothing.
    Downstream, the runtime and the release gate checked that three names
    exist, which a mixed folder satisfies. A turbo encoder and vocoder beside a
    stale K=2 estimator would ship, load, and integrate a one-step estimator
    twice — audible, and reported by nothing.
    """

    @staticmethod
    def _exporter():
        return tool("export_coreml")

    def test_an_unknown_stage_is_refused_rather_than_skipped(self) -> None:
        module = self._exporter()
        parser_stages = set(module.STAGE_PACKAGES)
        assert parser_stages == {
            "cond",
            "prefill",
            "step",
            "pair_step",
            "head2",
            "encoder",
            "estimator",
            "vocoder",
        }

        import argparse

        for spelling in ("encodr", "encoder,vocodor", "", "encoder,"):
            args = argparse.Namespace(stages=spelling)
            stages = {name.strip() for name in args.stages.split(",")}
            unknown = sorted(stages - parser_stages)
            assert unknown or not stages - {""}, (
                f"{spelling!r} would have been accepted, and a stage that is not "
                f"exported leaves whatever was already in the directory"
            )

    def test_the_record_names_every_package_a_run_wrote(self, tmp_path) -> None:
        import json
        from dataclasses import replace
        from types import SimpleNamespace

        from loudkit.backends.coreml_backend import EXPORT_PROVENANCE
        from loudkit.config import AlgorithmConfig

        module = self._exporter()
        algo = replace(AlgorithmConfig(), euler_steps=1)
        ckpt = SimpleNamespace(path=tmp_path / "loudr-1-turbo.safetensors")
        ckpt.path.write_bytes(b"weights")
        args = SimpleNamespace(estimator_ckpt=None, estimator_k=None)

        # The runtime's own name for the record, so the writer and the reader
        # cannot drift onto two filenames.
        module._write_provenance(tmp_path, {"encoder", "vocoder"}, ckpt, algo, args, None)
        record = json.loads((tmp_path / EXPORT_PROVENANCE).read_text(encoding="utf-8"))
        assert set(record["packages"]) == {module.ENCODER_NAME, module.VOCODER_NAME}

        # A second, partial run merges rather than replaces, which is exactly
        # what makes a mixed set visible instead of tidying it away.
        stale = replace(AlgorithmConfig(), euler_steps=2)
        module._write_provenance(tmp_path, {"estimator"}, ckpt, stale, args, None)
        record = json.loads((tmp_path / EXPORT_PROVENANCE).read_text(encoding="utf-8"))
        assert len(record["packages"]) == 3
        steps = {e["euler_steps"] for e in record["packages"].values()}
        assert steps == {1, 2}, "the mixed set has to be legible in the record"


class _ExportSet:
    """One backend's export-provenance record, as the refusal tests read it.

    CoreML writes ``packages`` and ONNX writes ``graphs``, and past that the two
    records are the same document under two names: a mapping of stage name to
    the checkpoint digest, algorithm fingerprint and step count the stage was
    traced from. The law they are held to is the same law, so it is written
    once and parametrized rather than transcribed twice, which is how the two
    suites drifted into nine near-identical tests apiece.
    """

    def __init__(self, module: str, record_key: str, format_name: str) -> None:
        self.module = module
        self.record_key = record_key
        self.format_name = format_name

    @property
    def backend(self):
        import importlib

        return importlib.import_module(f"loudkit.backends.{self.module}")

    def names(self) -> tuple[str, ...]:
        """Every stage a complete set records, in export order."""
        b = self.backend
        if self.record_key == "packages":
            return (
                "t3_cond.mlpackage",
                "t3_prefill.mlpackage",
                "t3_step.mlpackage",
                b.ENCODER_PACKAGE,
                b.ESTIMATOR_PACKAGE,
                b.HIFT_PACKAGE,
            )
        return tuple(b._SESSIONS)

    def entry(self, digest="abc", fingerprint="f" * 16, euler=1) -> dict:
        return {
            "checkpoint_sha256": digest,
            "algorithm_fingerprint": fingerprint,
            "euler_steps": euler,
        }

    def matching_entry(self, **kw) -> dict:
        """An entry that agrees with the algorithm ``_check`` compares against."""
        kw.setdefault("fingerprint", self.algorithm().fingerprint())
        return self.entry(**kw)

    @staticmethod
    def algorithm(euler: int = 1):
        from dataclasses import replace

        from loudkit.config import AlgorithmConfig

        return replace(AlgorithmConfig(), euler_steps=euler)

    def write(self, assets, entries) -> None:
        self.write_payload(assets, {self.record_key: entries})

    def write_payload(self, assets, payload) -> None:
        """A record written verbatim, for the shapes that are not records."""
        import json

        (assets / self.backend.EXPORT_PROVENANCE).write_text(
            json.dumps({"format": self.format_name, **payload}), encoding="utf-8"
        )

    def check(self, assets, digest="abc", euler=1, manifest=None):
        from types import SimpleNamespace

        ckpt = SimpleNamespace(
            file_digest=digest, path=Path("loudr-1.safetensors"), manifest=manifest or {}
        )
        # The entries carry a fingerprint literal, so the config's own value is
        # what the check compares against.
        return self.backend._check_provenance(assets, ckpt, self.algorithm(euler))


_COREML_SET = _ExportSet("coreml_backend", "packages", "loudkit-coreml-export")
_ONNX_SET = _ExportSet("onnx_backend", "graphs", "loudkit-onnx-export")


@pytest.mark.parametrize(
    "exported",
    [pytest.param(_COREML_SET, id="coreml"), pytest.param(_ONNX_SET, id="onnx")],
)
class TestTheRuntimeRefusesAMixedSet:
    """The reading half of the export record, for both backends.

    The ONNX folder holds six graphs, three of them the token generator, and
    ``--stages`` exports a subset. Re-exporting the renderer against a new
    checkpoint while three stale T3 graphs sit beside it produced a folder that
    passed every gate the project had: ``_assets_dir`` checks that six *names*
    exist. What loads is one checkpoint's tokens through another checkpoint's
    renderer, and nothing said so. CoreML's package set has the same hole and
    the same fix.
    """

    def test_a_consistent_set_that_matches_the_checkpoint_loads(
        self, exported: _ExportSet, tmp_path
    ) -> None:
        exported.write(tmp_path, dict.fromkeys(exported.names(), exported.matching_entry()))
        exported.check(tmp_path)

    def test_a_set_from_another_checkpoint_is_refused(
        self, exported: _ExportSet, tmp_path
    ) -> None:
        entry = exported.matching_entry(digest="somebody-elses")
        exported.write(tmp_path, dict.fromkeys(exported.names(), entry))
        with pytest.raises(ValueError, match="different engine"):
            exported.check(tmp_path)

    def test_a_stage_the_record_does_not_name_is_refused(
        self, exported: _ExportSet, tmp_path
    ) -> None:
        names = exported.names()
        exported.write(tmp_path, dict.fromkeys(names[:-1], exported.matching_entry()))
        with pytest.raises(ValueError, match="does not record"):
            exported.check(tmp_path)

    def test_a_set_traced_from_a_foreign_estimator_is_refused(
        self, exported: _ExportSet, tmp_path
    ) -> None:
        """The digest that was written and never read.

        `--estimator-ckpt` swaps a distilled estimator in before tracing and
        records its sha256. Nothing compared it, so a set traced from a
        *different* estimator with the same K agreed with itself on every field
        the check looked at, and described a renderer the checkpoint's
        fingerprint does not cover.
        """
        entry = exported.matching_entry()
        entry["estimator_sha256"] = "the-one-that-was-traced"
        exported.write(tmp_path, dict.fromkeys(exported.names(), entry))
        packed = {"sources": {"best.pt": {"role": "estimator", "sha256": "the-one-packed"}}}
        with pytest.raises(ValueError, match="traced with an estimator the checkpoint"):
            exported.check(tmp_path, manifest=packed)

    def test_re_exporting_one_stage_from_another_estimator_is_a_mixed_set(
        self, exported: _ExportSet, tmp_path
    ) -> None:
        """The narrower form: only the estimator stage re-traced, so the other
        entries keep the old digest and the record disagrees with itself."""
        old = exported.matching_entry()
        old["estimator_sha256"] = "the-first-one"
        fresh = dict(old, estimator_sha256="a-different-one")
        names = exported.names()
        record = dict.fromkeys(names, old)
        record[names[-2]] = fresh
        exported.write(tmp_path, record)
        with pytest.raises(ValueError, match="mixed"):
            exported.check(tmp_path)

    def test_a_record_of_the_wrong_shape_is_refused_not_crashed_on(
        self, exported: _ExportSet, tmp_path
    ) -> None:
        """Parses as JSON, is not a record.

        An empty list, and an entry that is a number, both reached `.items()`
        and `.get()` and came out as `AttributeError` from inside a load --
        naming neither the file nor the remedy. A malformed record gets the
        same answer as a wrong one: refuse, and say to re-export.
        """
        for bad in ([], {"a": 3}, "nope"):
            exported.write(tmp_path, bad)
            with pytest.raises(ValueError, match="is not a mapping of"):
                exported.check(tmp_path)

    def test_a_field_one_level_down_is_checked_too(
        self, exported: _ExportSet, tmp_path
    ) -> None:
        """The guard above looked at the entry and not into it.

        `{"a": {"euler_steps": []}}` is a mapping of name to mapping, so it
        passed, and the tuple built from it reached `set(...)` as
        `TypeError: unhashable type: 'list'` -- the same raw crash one layer
        deeper. `True` is checked separately because it is an `int` in Python
        and is not a step count.
        """
        for bad in ([], {"a": 1}, True):
            entry = exported.entry()
            entry["euler_steps"] = bad
            exported.write(tmp_path, {"a": entry})
            with pytest.raises(ValueError, match="is not a mapping of"):
                exported.check(tmp_path)

    def test_an_absent_record_warns_rather_than_refuses(
        self, exported: _ExportSet, tmp_path
    ) -> None:
        """Every export written before the record exists without one, and
        refusing those would strand working assets to gain nothing this cannot
        say in a sentence. `build_release.py` requires it outright, because a
        release is where re-exporting is possible."""
        with pytest.warns(RuntimeWarning, match="came from one export"):
            exported.check(tmp_path)


class TestTheMixedSetEachBackendReachesItsOwnWay:
    """The two cases the law above is the same for but the run that causes it
    is not: CoreML mixes on the step count, ONNX on the checkpoint digest."""

    def test_a_mixed_set_is_refused(self, tmp_path) -> None:
        estimator = _COREML_SET.names()[-2]
        good = _COREML_SET.matching_entry()
        stale = _COREML_SET.matching_entry(euler=2)
        _COREML_SET.write(
            tmp_path, dict.fromkeys(_COREML_SET.names(), good) | {estimator: stale}
        )
        with pytest.raises(ValueError, match="mixed package set"):
            _COREML_SET.check(tmp_path)

    def test_a_renderer_re_exported_beside_stale_t3_graphs_is_refused(self, tmp_path) -> None:
        """The failure this check exists for, spelled as the run that causes
        it: `--stages encoder,estimator,vocoder` against a new checkpoint."""
        from loudkit.backends import onnx_backend as ob

        fresh = _ONNX_SET.matching_entry(digest="abc")
        stale = _ONNX_SET.matching_entry(digest="the-previous-checkpoint")
        _ONNX_SET.write(
            tmp_path,
            {
                ob.COND_GRAPH: stale,
                ob.PREFILL_GRAPH: stale,
                ob.STEP_GRAPH: stale,
                ob.ENCODER_GRAPH: fresh,
                ob.ESTIMATOR_GRAPH: fresh,
                ob.HIFT_GRAPH: fresh,
            },
        )
        with pytest.raises(ValueError, match="mixed graph set"):
            _ONNX_SET.check(tmp_path)

    def test_the_exporter_refuses_a_stage_it_does_not_export(self) -> None:
        from tools.export_generator import validate_stages

        with pytest.raises(SystemExit, match="do not belong"):
            validate_stages("single", {"encodr"})


class TestTurboCarriesItsOwnPostprocessBlock:
    """`pack_turbo.py` writes the preset out in full and labels its origin.

    Inheriting the key verbatim shipped loudr-1's thresholds under a manifest
    that did not say so, and left the fourteen code-only knobs unreachable
    from any manifest. The block is turbo's own now: every non-census knob,
    loudr-1's values, `calibrated_on` naming where they came from, and the
    runtime reads it back to exactly the preset it would have used.
    """

    def test_every_knob_is_written_and_reads_back(self) -> None:
        from dataclasses import fields

        from loudkit.config import AlgorithmConfig
        from loudkit.postprocess import PostprocessConfig

        block = tool("pack_turbo").turbo_postprocess_block()
        assert block["calibrated_on"] == "loudr-1"
        knobs = {
            f.name for f in fields(PostprocessConfig) if not f.name.endswith("_render_ids")
        }
        assert set(block) - {"calibrated_on"} == knobs
        assert "postprocess" not in tool("pack_turbo").INHERITED_KEYS
        read = AlgorithmConfig.from_manifest({"postprocess": block}).postprocess
        assert read == PostprocessConfig()


class TestBothRewritersRefuseAnUnvouchedPayload:
    """`split_checkpoint.py` and `amend_manifest.py` rewrite the same file.

    They refused differently. `amend_manifest`'s `if recorded and ...` let a
    checkpoint whose manifest records no digest through, which is the one case
    where nothing at all is being checked: it would then stamp a fresh,
    confident manifest onto bytes nobody vouched for. Both go through
    `payload_refusal` now, so the rule is stated once.
    """

    @staticmethod
    def _rule():
        return tool("split_checkpoint").payload_refusal

    @pytest.mark.parametrize(
        "recorded",
        [None, "", "a" * 63, "A" * 64, "g" * 64, 12345, ["a" * 64]],
        ids=["absent", "empty", "short", "uppercase", "not-hex", "integer", "list"],
    )
    def test_a_digest_that_is_not_a_sha256_is_refused(self, recorded) -> None:
        problem = self._rule()({"tensor_payload_sha256": recorded}, "b" * 64)
        assert problem is not None
        assert "not a sha256" in problem

    def test_a_digest_that_does_not_match_the_bytes_is_refused(self) -> None:
        problem = self._rule()({"tensor_payload_sha256": "a" * 64}, "b" * 64)
        assert problem is not None
        assert "payload hash mismatch" in problem

    def test_the_matching_digest_is_the_only_thing_that_passes(self) -> None:
        assert self._rule()({"tensor_payload_sha256": "a" * 64}, "a" * 64) is None

    def test_both_tools_ask_the_same_question(self) -> None:
        import ast

        for name in ("split_checkpoint", "amend_manifest"):
            tree = ast.parse((REPO / "tools" / f"{name}.py").read_text(encoding="utf-8"))
            called = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            assert "payload_refusal" in called, f"{name} rewrites the file without the check"

    def test_both_rewriters_read_through_the_runtime(self) -> None:
        """The other half of the same rule: one parse of the container.

        `amend_manifest` indexed the safetensors metadata unguarded, so a file
        with no metadata died with a `TypeError` traceback out of
        `meta["manifest"]` where its sibling refuses with a sentence.
        """
        import ast

        tree = ast.parse((REPO / "tools" / "amend_manifest.py").read_text(encoding="utf-8"))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "read_manifest" in called

    def test_a_file_that_is_not_a_checkpoint_is_a_sentence(self, tmp_path) -> None:
        import subprocess

        path = tmp_path / "loudr-1.safetensors"
        path.write_bytes(b"not a safetensors container")
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "tools" / "amend_manifest.py"),
                "--checkpoint",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "Traceback" not in result.stderr, result.stderr
        assert str(path) in result.stderr


class TestTheSplitIsNamedByTheModelNotTheFile:
    """Which model a checkpoint is, is a question for its manifest.

    The packer's ``--out`` writes whatever it was told, so a turbo checkpoint
    arrives under whatever name its operator chose. Naming the synthesis half
    after the *input file* wrote `loudr-1.safetensors` for a fusion checkpoint,
    recorded a `split.roles.synthesis` saying so, and the release then refused
    the turbo bundle built from it: `check_bundle` asks for
    `loudr-1-turbo.safetensors`, and the two never met.
    """

    @staticmethod
    def _name(manifest: dict) -> str:
        name: str = tool("split_checkpoint").synthesis_filename(manifest)
        return name

    def test_a_fusion_manifest_names_the_turbo_half(self) -> None:
        assert self._name({"decode": {"mode": "fusion_mtp2"}}) == "loudr-1-turbo.safetensors"

    def test_a_single_token_manifest_names_the_base_half(self) -> None:
        assert self._name({"decode": {"mode": "single"}}) == "loudr-1.safetensors"

    def test_no_decode_block_is_the_base_model(self) -> None:
        """Absent means single-token, which is how the runtime reads it."""
        assert self._name({}) == "loudr-1.safetensors"

    def test_the_input_file_name_does_not_decide(self) -> None:
        """The case the old rule got wrong, and the only one worth a test."""
        import inspect

        source = inspect.getsource(tool("split_checkpoint").synthesis_filename)
        assert "path.name" not in source
        assert "args.checkpoint" not in source


def test_one_safetensors_header_parse_serves_every_reader() -> None:
    """Three copies of eight-bytes-then-JSON, down to the error strings.

    ``verify._read_header`` and ``make_device_pack.read_header``
    re-implemented the parse that ``loudkit.checkpoint.payload_sha256``
    already contained, including the 100 MiB cap. A container the runtime
    accepts and a tool does not is a release that refuses a file which loads,
    or ships one that does not.
    """
    import importlib

    from loudkit.checkpoint import read_header

    assert tool("make_device_pack").read_header is read_header

    sys.path.insert(0, str(REPO / "tools"))
    try:
        verify = importlib.import_module("release.verify")
    finally:
        sys.path.remove(str(REPO / "tools"))
    assert verify.read_header is read_header
