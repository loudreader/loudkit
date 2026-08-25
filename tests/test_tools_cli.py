"""The tools' own command lines, and the one download they perform.

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

REPO = Path(__file__).resolve().parent.parent


def _tool(name: str):
    sys.path.insert(0, str(REPO / "tools"))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


class TestBenchParsers:
    def test_seed_is_read_wherever_the_flag_sits(self) -> None:
        parser = _tool("bench_cuda_box")._parser()
        args = parser.parse_args(["ckpt", "voice", "cuda:1", "out", "--seed", "11"])
        assert args.seed == 11
        assert args.device == "cuda:1"
        assert args.cuda_graphs is False

    def test_bench_defaults(self) -> None:
        args = _tool("bench_cuda_box")._parser().parse_args(["c", "v", "cpu", "o"])
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
            _tool("bench_cuda_box")._parser().parse_args(argv)

    def test_batch_list_default_and_parse(self) -> None:
        mod = _tool("bench_batch")
        parser = mod._parser()
        assert parser.parse_args(["c", "v", "cuda", "o"]).batches == list(mod.DEFAULT_BATCHES)
        assert parser.parse_args(["c", "v", "cuda", "o", "1,4,16"]).batches == [1, 4, 16]

    def test_batch_list_refuses_a_non_integer(self) -> None:
        with pytest.raises(SystemExit):
            _tool("bench_batch")._parser().parse_args(["c", "v", "cuda", "o", "1,2,x"])


class TestRoundtripParser:
    def test_score_takes_two_paths(self) -> None:
        args = _tool("eval_roundtrip")._parser().parse_args(["--score", "out", "t.json"])
        assert args.score == [Path("out"), Path("t.json")]

    def test_render_takes_three_positionals(self) -> None:
        args = _tool("eval_roundtrip")._parser().parse_args(["ckpt", "voice", "out"])
        assert (args.checkpoint, args.voice, args.out_dir) == ("ckpt", "voice", Path("out"))

    def test_render_with_a_missing_positional_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["eval_roundtrip.py", "ckpt", "voice"])
        with pytest.raises(SystemExit):
            _tool("eval_roundtrip").main()


class TestDevicePack:
    def test_out_is_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No default: the app it was written for is not in this repo, and a
        default pointing at it staged 356 MB into a path only one machine has."""
        monkeypatch.setattr(sys, "argv", ["make_device_pack.py", "--checkpoint", "x"])
        with pytest.raises(SystemExit):
            _tool("make_device_pack").main()


class TestAcceptance:
    def test_speak_defaults_to_the_readme_runtime_extras(self) -> None:
        mod = _tool("acceptance")
        assert mod.requested_extras("", speak=True) == "torch,audio,hub"
        assert mod.requested_extras("onnx,audio,hub", speak=True) == ("onnx,audio,hub")
        assert mod.requested_extras("", speak=False) == ""

    def test_speak_gate_names_the_downloaded_release_and_a_real_voice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _tool("acceptance")
        calls: list[list[str]] = []

        def fake_run(
            cmd: list[str],
            *,
            cwd: Path,
            env_note: str = "",  # noqa: ARG001
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

        exe = str(bindir / "loudkit")
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
        mod = _tool("fetch_nst_lexicons")
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
        mod = _tool("fetch_nst_lexicons")
        serving(b"<html>404</html>")
        with pytest.raises(SystemExit, match="too small"):
            mod.fetch("da", tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_a_digest_that_misses_the_pin_is_discarded(
        self, tmp_path, serving, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _tool("fetch_nst_lexicons")
        serving(b"not the lexicon")
        monkeypatch.setattr(mod, "MIN_BYTES", 8)
        monkeypatch.setitem(mod.SOURCES, "no", (mod.SOURCES["no"][0], "0" * 64))
        with pytest.raises(SystemExit, match="does not match the pin"):
            mod.fetch("no", tmp_path)
        assert list(tmp_path.iterdir()) == []
