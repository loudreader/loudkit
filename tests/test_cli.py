"""The CLI grammar and dispatch, without weights.

Two layers, asserted separately so a regression names the layer.

**The grammar** is ``build_parser()``: every subcommand exists, carries the
arguments the README promises, and validates the ones that should be validated
(``--device`` choices, required ``--checkpoint``/``--voice``). A future
subcommand that is documented but not wired, or wired with a different flag,
fails here before any engine is loaded.

**Dispatch** runs ``main`` with the engine and voice replaced by fakes (the
same deterministic ones the server tests use). The commands' job is to
construct an :class:`~loudkit.engine.Engine` and call it — never to invent a
second synthesis path — so a fake engine plus a captured call proves the wiring
without a 1.27 GB checkpoint. The error paths (missing files, missing optional
dependency) are asserted as exit codes and stderr text.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from loudkit.engine import Engine
from loudkit.errors import UnsupportedLanguageError
from loudkit.voice import VoiceProfile

from .assets import asset, requires_modules
from .conftest import fake_engine, fake_voice

CKPT = asset("checkpoint")


@pytest.fixture
def fake_ckpt(tmp_path) -> object:
    """A path that exists so the CLI's file-existence gate passes; nothing
    ever opens it because ``loudkit.load`` is monkeypatched."""
    p = tmp_path / "fake.safetensors"
    p.write_bytes(b"not a real checkpoint, never opened")
    return p


@pytest.fixture
def fake_voice_file(tmp_path) -> object:
    voice = fake_voice()
    voice.save(tmp_path / "fake.safetensors")
    return tmp_path / "fake.safetensors"


def _fake_load_engine(*args, **kwargs) -> Engine:
    del args, kwargs
    return fake_engine()


def _fake_load_voice(path: object) -> VoiceProfile:
    del path
    return fake_voice()


# ------------------------------------------------------------------ grammar


class TestParserGrammar:
    COMMANDS = {"speak", "text", "clone", "doctor", "download", "voices", "verify", "serve"}

    def test_subcommands_exist(self) -> None:
        from loudkit.cli import build_parser

        parser = build_parser()
        sub = parser._subparsers._group_actions[0].choices
        assert set(sub) == self.COMMANDS

    @pytest.mark.parametrize(
        "device", ["cpu", "cuda", "cuda:0", "cuda:1", "mps", "coreml", "onnx"]
    )
    def test_device_choices(self, device: str) -> None:
        from loudkit.cli import build_parser

        args = build_parser().parse_args(["doctor", "--device", device])
        assert args.device == device

    def test_the_device_roster_is_the_literal_and_is_written_once(self) -> None:
        """`--device` is derived from `Device`, not restated beside it.

        A backend added to the Literal used to need a second edit in the CLI's
        own tuple and a third in each `--device` help line, and the tuple and
        the Literal had already drifted into different orders. Asserting the
        joined roster, not the membership, is what catches a fourth copy: a
        restatement that happens to hold the same names passes a membership
        check and fails this one.
        """
        import argparse
        from typing import get_args

        from loudkit.cli import _device_arg, build_parser
        from loudkit.execution import Device

        roster = ", ".join(get_args(Device))
        for name in get_args(Device):
            assert build_parser().parse_args(["doctor", "--device", name]).device == name
        with pytest.raises(argparse.ArgumentTypeError, match=roster):
            _device_arg("tpu")

    def test_the_device_help_spells_the_roster_the_refusal_spells(self) -> None:
        """Both `--device` help lines come off the same tuple, so neither can
        name a backend the parser refuses or omit one it takes."""
        from typing import get_args

        from loudkit.cli import build_parser
        from loudkit.execution import Device

        roster = ", ".join(get_args(Device))
        commands = build_parser()._subparsers._group_actions[0].choices
        for command in ("speak", "clone"):
            (action,) = [
                a for a in commands[command]._actions if "--device" in a.option_strings
            ]
            assert roster in action.help

    def test_indexed_cuda_any_gpu_index(self) -> None:
        """Multi-GPU boxes must be reachable from the CLI, not just the API:
        the registry splits on ':' and torch.device('cuda:N') is valid."""
        from loudkit.cli import build_parser

        args = build_parser().parse_args(["speak", "--voice", "y", "--device", "cuda:3", "hi"])
        assert args.device == "cuda:3"

    def test_unknown_device_rejected(self) -> None:
        from loudkit.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["doctor", "--device", "tpu"])

    def test_indexed_cuda_must_be_numeric(self) -> None:
        """cuda:<index> requires a numeric index; 'cuda:abc' is garbage."""
        from loudkit.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["doctor", "--device", "cuda:abc"])

    def test_checkpoint_defaults_to_the_release(self) -> None:
        """`--checkpoint` is not required every time: the release is the default,
        and `$LOUDKIT_CHECKPOINT` overrides it."""
        from loudkit.cli import DEFAULT_CHECKPOINT, build_parser

        for argv in (
            ["speak", "--voice", "v", "hi"],
            ["clone", "me.wav", "--name", "n", "--language", "en"],
            ["serve"],
            ["doctor"],
        ):
            assert build_parser().parse_args(argv).checkpoint == DEFAULT_CHECKPOINT
        assert build_parser().parse_args(["voices"]).repo == DEFAULT_CHECKPOINT

    def test_speak_requires_voice_and_text(self) -> None:
        from loudkit.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["speak", "--checkpoint", "x"])
        with pytest.raises(SystemExit):
            build_parser().parse_args(["speak", "--checkpoint", "x", "--voice", "v"])

    def test_defaults(self) -> None:
        from loudkit.cli import build_parser

        args = build_parser().parse_args(
            ["speak", "--checkpoint", "x", "--voice", "v", "hello"]
        )
        assert str(args.output) == "out.wav"
        assert args.seed == 0
        # None, not "en": an omitted --language means the voice's own language,
        # and the engine resolves the chain.
        assert args.language is None
        assert args.device is None
        # 1.0, not None: the default is documented as an exact bypass, and a
        # sentinel here would put the decision in two places.
        assert args.speed == 1.0
        assert args.verbose is False

    def test_no_command_prints_help(self, capsys) -> None:
        from loudkit.cli import main

        assert main([]) == 1
        out = capsys.readouterr().out
        assert "usage:" in out


# ------------------------------------------------------------------ dispatch


class TestDispatch:
    def test_version(self, capsys) -> None:
        # Against the package's own `__version__`, not a literal: a literal
        # here fails at every release for no reason, and the four manifests
        # are already held to one value by tests/test_release.py.
        import loudkit
        from loudkit.cli import main

        assert main(["--version"]) == 0
        assert capsys.readouterr().out.strip() == loudkit.__version__

    def test_missing_checkpoint_is_named(self, tmp_path, capsys) -> None:
        from loudkit.cli import main

        rc = main(
            ["speak", "--checkpoint", str(tmp_path / "nope.safetensors"), "--voice", "v", "hi"]
        )
        assert rc == 1
        assert "checkpoint not found" in capsys.readouterr().err

    def test_oversized_stdin_is_refused_before_the_checkpoint_is_read(
        self, fake_ckpt, fake_voice_file, monkeypatch
    ) -> None:
        """The free refusals come first, or the user waits for one of them.

        Reading the checkpoint is 747 MB off disk, or a download for a repo id.
        `speak` used to load it, then the voice, then stdin, so a paste over the
        cap was answered minutes after the command was typed, by which time the
        answer was worth nothing. `loudkit.load` here fails the test if it is
        reached at all, which is the assertion that keeps the order from
        drifting back.
        """
        import io

        import loudkit
        from loudkit.cli import _MAX_STDIN_CHARS, main

        def refuse_to_load(*_a, **_k):
            raise AssertionError("the engine loaded before stdin was read")

        monkeypatch.setattr(loudkit, "load", refuse_to_load)
        monkeypatch.setattr("sys.stdin", io.StringIO("x" * (_MAX_STDIN_CHARS + 1)))
        with pytest.raises(SystemExit, match="stdin is over"):
            main(
                [
                    "speak",
                    "--checkpoint",
                    str(fake_ckpt),
                    "--voice",
                    str(fake_voice_file),
                    "-",
                ]
            )

    def test_an_unreadable_voice_is_refused_before_the_checkpoint_is_read(
        self, fake_ckpt, tmp_path, monkeypatch, capsys
    ) -> None:
        """The second free refusal, same reason: a voice file that is not a
        voice costs one open to discover and used to cost a whole model load
        first."""
        import loudkit
        from loudkit.cli import main

        def refuse_to_load(*_a, **_k):
            raise AssertionError("the engine loaded before the voice was read")

        voice = tmp_path / "not-a-voice.safetensors"
        voice.write_bytes(b"not a voice profile either")
        monkeypatch.setattr(loudkit, "load", refuse_to_load)
        assert main(["speak", "--checkpoint", str(fake_ckpt), "--voice", str(voice), "hi"]) == 1

    def test_missing_voice_is_named(self, tmp_path, capsys) -> None:
        from loudkit.cli import main

        ckpt = tmp_path / "ckpt.safetensors"
        ckpt.write_bytes(b"unused")
        rc = main(
            [
                "speak",
                "--checkpoint",
                str(ckpt),
                "--voice",
                str(tmp_path / "nope.safetensors"),
                "hi",
            ]
        )
        assert rc == 1
        assert "voice not found" in capsys.readouterr().err

    def test_download_survives_the_path_pre_check(self) -> None:
        """A repo id is not a missing file, and download's arguments hold no
        path the pre-check should trip over."""
        from loudkit.cli import _missing_path, build_parser

        args = build_parser().parse_args(["download", "loudreader/loudr-1"])
        assert _missing_path(args) is None

    def test_the_hinted_bare_name_against_a_repo_id_runs(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """`speak --checkpoint <repo> --voice kathleen` is the exact command
        doctor and download print as the next step. It has to reach the
        resolver: the path pre-check used to answer "voice not found:
        kathleen" first, which made the printed hint a dead end."""
        import loudkit
        import loudkit.hub as hub_mod

        profile = tmp_path / "kathleen.safetensors"
        fake_voice().save(profile)
        asked: list[tuple[str, str | None]] = []

        def fake_resolve(ref, *, repo=None, revision=None):
            asked.append((ref, repo))
            return profile

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)
        monkeypatch.setattr(hub_mod, "resolve_voice", fake_resolve)
        from loudkit.cli import main

        out = tmp_path / "out.wav"
        rc = main(
            [
                "speak",
                "--checkpoint",
                "loudreader/loudr-1",
                "--voice",
                "kathleen",
                "-o",
                str(out),
                "hello",
            ]
        )
        assert rc == 0
        assert asked == [("kathleen", "loudreader/loudr-1")]
        assert out.exists()

    def test_the_hinted_bare_name_against_a_local_release_runs(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """The same shape with the release unpacked on disk — no network, and
        no monkeypatched resolver: `voices/` beside the checkpoint is what
        `hub.resolve_voice` reads."""
        import loudkit

        release = tmp_path / "loudr-1"
        (release / "voices").mkdir(parents=True)
        (release / "loudr-1.safetensors").write_bytes(b"unused")
        fake_voice().save(release / "voices" / "kathleen.safetensors")

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)
        from loudkit.cli import main

        out = tmp_path / "out.wav"
        rc = main(
            [
                "speak",
                "--checkpoint",
                str(release),
                "--voice",
                "kathleen",
                "-o",
                str(out),
                "hello",
            ]
        )
        assert rc == 0
        assert out.exists()

    def test_a_bare_name_resolves_beside_the_checkpoint_file_too(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """A release is a checkpoint beside `voices/`, so naming the file is
        naming the release — `resolve_voice_encoder` already reads it that
        way, and a user who tab-completed the checkpoint gets the same answer
        as one who named its directory."""
        import loudkit

        release = tmp_path / "loudr-1"
        (release / "voices").mkdir(parents=True)
        checkpoint = release / "loudr-1.safetensors"
        checkpoint.write_bytes(b"unused")
        fake_voice().save(release / "voices" / "kathleen.safetensors")

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)
        from loudkit.cli import main

        out = tmp_path / "out.wav"
        rc = main(
            [
                "speak",
                "--checkpoint",
                str(checkpoint),
                "--voice",
                "kathleen",
                "-o",
                str(out),
                "hello",
            ]
        )
        assert rc == 0
        assert out.exists()

    def test_a_bare_name_with_no_release_still_says_voice_not_found(
        self, tmp_path, capsys
    ) -> None:
        """Nothing above may swallow the plain case: a checkpoint that names
        no release leaves a bare name a missing file, reported as one."""
        from loudkit.cli import main

        rc = main(
            [
                "speak",
                "--checkpoint",
                str(tmp_path / "nope.safetensors"),
                "--voice",
                "kathleen",
                "hi",
            ]
        )
        assert rc == 1
        assert "checkpoint not found" in capsys.readouterr().err

    def test_doctor_describe_prints_the_engine(self, fake_ckpt, capsys, monkeypatch) -> None:
        """The fingerprint left `speak`'s default output; `doctor --describe` and
        `speak --verbose` are where it is printed."""
        import loudkit

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)
        from loudkit.cli import main

        assert main(["doctor", "--describe", "--checkpoint", str(fake_ckpt)]) == 0
        out = capsys.readouterr().out
        assert "algo[" in out
        assert "single_path" in out

    def test_speak_is_quiet_about_the_fingerprint_unless_asked(
        self, fake_ckpt, fake_voice_file, tmp_path, capsys, monkeypatch
    ) -> None:
        import loudkit

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)
        monkeypatch.setattr(loudkit.VoiceProfile, "load", _fake_load_voice, raising=False)
        from loudkit.cli import main

        argv = ["speak", "--checkpoint", str(fake_ckpt), "--voice", str(fake_voice_file)]
        argv += ["-o", str(tmp_path / "a.wav"), "hello"]
        assert main(argv) == 0
        assert "algo[" not in capsys.readouterr().err
        assert main(["--debug", *argv, "--verbose"]) == 0
        assert "algo[" in capsys.readouterr().err

    def test_speak_writes_a_wav(
        self, fake_ckpt, fake_voice_file, tmp_path, capsys, monkeypatch
    ) -> None:
        import loudkit

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)
        monkeypatch.setattr(loudkit.VoiceProfile, "load", _fake_load_voice, raising=False)

        from loudkit.cli import main

        out = tmp_path / "out.wav"
        rc = main(
            [
                "speak",
                "--checkpoint",
                str(fake_ckpt),
                "--voice",
                str(fake_voice_file),
                "-o",
                str(out),
                "hello there",
            ]
        )
        assert rc == 0
        assert out.exists()
        assert out.stat().st_size > 0
        # the summary names the output
        assert str(out) in capsys.readouterr().err

    def test_speak_says_how_to_hear_the_file(
        self, fake_ckpt, fake_voice_file, tmp_path, capsys, monkeypatch
    ) -> None:
        """The first run must not end at a filename. One platform-appropriate
        command, printed with the summary — a hint, not a played sound."""
        import platform

        import loudkit

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)
        monkeypatch.setattr(loudkit.VoiceProfile, "load", _fake_load_voice, raising=False)

        from loudkit.cli import main

        out = tmp_path / "out.wav"
        rc = main(
            [
                "speak",
                "--checkpoint",
                str(fake_ckpt),
                "--voice",
                str(fake_voice_file),
                "-o",
                str(out),
                "hello there",
            ]
        )
        assert rc == 0
        err = capsys.readouterr().err
        player = {"Darwin": "afplay", "Windows": "Start-Process"}.get(
            platform.system(), "aplay"
        )
        assert player in err
        assert str(out) in err.split("hear it:")[1]

    @pytest.mark.parametrize(
        ("system", "expected"),
        [
            ("Darwin", "afplay '/tmp/my recordings/out.wav'"),
            ("Windows", 'Start-Process "/tmp/my recordings/out.wav"'),
            ("Linux", "aplay '/tmp/my recordings/out.wav'"),
        ],
    )
    def test_the_hint_is_a_command_that_runs_for_a_path_with_a_space(
        self, system, expected, monkeypatch
    ) -> None:
        """A hint the reader has to repair is not a hint.

        Asserted on the whole line rather than on the path being somewhere in
        it, which is true of a hint that names two arguments where the shell
        needs one.
        """
        import platform

        from loudkit.cli import _play_hint

        monkeypatch.setattr(platform, "system", lambda: system)
        assert _play_hint(Path("/tmp/my recordings/out.wav")) == expected

    def test_speak_splits_a_paragraph_instead_of_refusing(
        self, fake_ckpt, fake_voice_file, tmp_path, monkeypatch
    ) -> None:
        """The first interface most people try must not drop a paragraph.

        It did, twice over: `speak` chose between two engine methods, first by
        catching an exception that could not reach it, then by asking
        `needs_long_form` and re-reading on a cap hit. Now it calls
        `synthesize` once and the splitting is the engine's, so there is no
        routing left in the CLI to get wrong. The assertion is that the
        paragraph came back as more than one chunk — a CLI that quietly asked
        for a single window would return one, and would refuse.
        """
        import loudkit

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)
        monkeypatch.setattr(loudkit.VoiceProfile, "load", _fake_load_voice, raising=False)

        seen: list[object] = []
        real = loudkit.Engine.synthesize

        def record(self, text, voice, **kw):
            assert not kw.get("single_window"), "speak must not ask for one window"
            result = real(self, text, voice, **kw)
            seen.append(result)
            return result

        monkeypatch.setattr(loudkit.Engine, "synthesize", record)

        from loudkit.cli import main

        out = tmp_path / "long.wav"
        text = "One sentence here. " * 60  # far past one window's budget
        rc = main(
            [
                "speak",
                "--checkpoint",
                str(fake_ckpt),
                "--voice",
                str(fake_voice_file),
                "-o",
                str(out),
                text,
            ]
        )
        assert rc == 0
        assert out.exists()
        assert out.stat().st_size > 0
        assert len(seen) == 1, "speak must call synthesize once and nothing else"
        assert len(seen[0].chunks) > 1, "the paragraph was not split"

    def test_a_sentence_takes_the_same_path_as_a_paragraph(
        self, fake_ckpt, fake_voice_file, tmp_path, monkeypatch
    ) -> None:
        """One path, whatever the length.

        The old CLI routed by length, so `speak` on a sentence and `speak` on a
        paragraph could diverge in ways nothing here would notice. A single
        call with no `single_window` is the whole contract now.
        """
        import loudkit

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)
        monkeypatch.setattr(loudkit.VoiceProfile, "load", _fake_load_voice, raising=False)

        calls: list[dict] = []
        real = loudkit.Engine.synthesize

        def record(self, text, voice, **kw):
            calls.append(kw)
            return real(self, text, voice, **kw)

        monkeypatch.setattr(loudkit.Engine, "synthesize", record)

        from loudkit.cli import main

        rc = main(
            [
                "speak",
                "--checkpoint",
                str(fake_ckpt),
                "--voice",
                str(fake_voice_file),
                "-o",
                str(tmp_path / "short.wav"),
                "One sentence here.",
            ]
        )
        assert rc == 0
        assert len(calls) == 1
        assert "single_window" not in calls[0]

    def test_speak_never_warms_the_engine(
        self, fake_ckpt, fake_voice_file, tmp_path, monkeypatch
    ) -> None:
        """A warm-up moves the first render's extra cost somewhere nobody waits.
        A command that renders and exits has no such place: the whole run is the
        wait, and a warm-up only adds a second render to it.

        Measured on this machine's CPU, `speak` on a two-window passage: 51.6s
        and 52.2s without, 73.6s with a warm-up that cost 22.7s and took
        nothing off the render (mel 42.3s against 42.4s). Re-measured on a
        three-chunk passage: 79.4s without, 24.8s of warm-up plus 76.3s of
        render with. On MPS the same passage takes 6.7s either way, so the
        best case for warming a command that exits is that it changes nothing.
        The servers warm, because startup is a place nobody waits; this does
        not. See ``docs/design/transports.md``.
        """
        import loudkit

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)
        monkeypatch.setattr(loudkit.VoiceProfile, "load", _fake_load_voice, raising=False)

        # Recorded, not raised: `warm_engine` swallows what a warm-up throws,
        # so a sentinel that raises would be caught and the test would pass on
        # a `speak` that warms.
        warmed: list[str] = []
        monkeypatch.setattr(loudkit.Engine, "warm", lambda _s, voice: warmed.append(voice.name))

        from loudkit.cli import main

        for text in ("One sentence here.", "One sentence here. " * 60):
            out = tmp_path / "spoken.wav"
            assert (
                main(
                    [
                        "speak",
                        "--checkpoint",
                        str(fake_ckpt),
                        "--voice",
                        str(fake_voice_file),
                        "-o",
                        str(out),
                        text,
                    ]
                )
                == 0
            )
            assert out.stat().st_size > 0
            assert warmed == [], "speak warmed the engine"

    def test_serve_starts_with_engine(self, fake_ckpt, monkeypatch) -> None:
        """serve must hand the constructed engine to the server, not construct
        a second one — the server is forbidden a synthesis path of its own."""
        import loudkit

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)

        captured = {}

        import loudkit.transports.http as server_mod

        def fake_serve(
            ckpt, voices, host, port, device, allow_public, token=None, first_chunk_tokens=None
        ):
            captured["host"], captured["port"], captured["allow_public"] = (
                host,
                port,
                allow_public,
            )

        monkeypatch.setattr(server_mod, "serve", fake_serve)

        from loudkit.cli import main

        assert (
            main(
                [
                    "serve",
                    "--checkpoint",
                    str(fake_ckpt),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9000",
                ]
            )
            == 0
        )
        assert captured == {"host": "127.0.0.1", "port": 9000, "allow_public": False}

    def test_serve_allow_public_flag_flows_through(self, fake_ckpt, monkeypatch) -> None:
        """--allow-public must reach the server's allow_public parameter."""
        import loudkit

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)

        captured = {}

        import loudkit.transports.http as server_mod

        def fake_serve(
            ckpt, voices, host, port, device, allow_public, token=None, first_chunk_tokens=None
        ):
            captured["host"], captured["allow_public"] = host, allow_public

        monkeypatch.setattr(server_mod, "serve", fake_serve)

        from loudkit.cli import main

        assert (
            main(
                [
                    "serve",
                    "--checkpoint",
                    str(fake_ckpt),
                    "--host",
                    "0.0.0.0",
                    "--allow-public",
                    "--port",
                    "9000",
                ]
            )
            == 0
        )
        assert captured == {"host": "0.0.0.0", "allow_public": True}

    def test_mcp_starts_stdio(self, fake_ckpt, monkeypatch) -> None:
        import loudkit

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)

        called = {}
        import loudkit.transports.mcp as mcp_mod

        def fake_stdio(ckpt, voices, *, device=None):
            called["ckpt"] = ckpt
            called["device"] = device

        monkeypatch.setattr(mcp_mod, "run_stdio", fake_stdio)

        from loudkit.cli import main

        assert main(["serve", "--mcp", "--checkpoint", str(fake_ckpt)]) == 0
        # A str, not a Path: `--checkpoint` may name a repo id, which only a
        # string can be asked about.
        assert called == {"ckpt": str(fake_ckpt), "device": None}

    def test_mcp_device_reaches_the_server(self, fake_ckpt, monkeypatch) -> None:
        """`--device` is parsed for every subcommand and this one dropped it.

        `loudkit serve --mcp --device cuda:3` was accepted and ignored, so the agent
        host got a different device, memory profile and speed from the one the
        operator asked for, with nothing said about it.
        """
        import loudkit.transports.mcp as mcp_mod

        seen = {}
        monkeypatch.setattr(
            mcp_mod,
            "run_stdio",
            lambda _ckpt, _voices, *, device=None: seen.update(device=device),
        )

        from loudkit.cli import main

        assert (
            main(["serve", "--mcp", "--checkpoint", str(fake_ckpt), "--device", "cuda:3"]) == 0
        )
        assert seen == {"device": "cuda:3"}

    @pytest.mark.parametrize(
        ("transport", "flag", "value"),
        [
            ("--mcp", "--host", "127.0.0.2"),
            ("--mcp", "--port", "9000"),
            ("--mcp", "--allow-public", None),
            ("--mcp", "--token", "sekrit"),
            ("--mcp", "--first-chunk-tokens", "8"),
            ("--grpc", "--allow-public", None),
            ("--grpc", "--token", "sekrit"),
        ],
    )
    def test_serve_refuses_a_flag_the_transport_would_drop(
        self, fake_ckpt, monkeypatch, capsys, transport, flag, value
    ) -> None:
        """The same defect `--device` had, on five flags and two doors.

        Each of these parsed and evaporated: the server came up, said nothing,
        and ran without a setting the operator believes is in force. That is
        worse than either honouring it or refusing it, because the operator
        has no way to find out. `--token` was the dangerous one, since the
        person passing it thinks the port is authenticated.

        Exit 2, argparse's own code for a command line that cannot be run, and
        the flag is named in the sentence: `serve --mcp cannot honour` alone
        would leave the reader to guess which of the eight they typed is
        wrong.
        """
        import loudkit.transports.grpc as grpc_mod
        import loudkit.transports.mcp as mcp_mod
        from loudkit.cli import main

        def _unreachable(*_a: object, **_kw: object) -> None:
            raise AssertionError("the transport was started for a refused command line")

        monkeypatch.setattr(mcp_mod, "run_stdio", _unreachable)
        monkeypatch.setattr(grpc_mod, "serve", _unreachable)

        argv = ["serve", transport, "--checkpoint", str(fake_ckpt), flag]
        if value is not None:
            argv.append(value)
        assert main(argv) == 2
        err = capsys.readouterr().err
        assert flag in err, err
        assert transport in err, err

    def test_a_flag_the_transport_reads_is_not_refused(self, fake_ckpt, monkeypatch) -> None:
        """The other half of the claim: gRPC keeps the two MCP cannot take.

        A refusal table is only useful if it is a table and not a blanket, so
        the flags gRPC honours must still reach it.
        """
        import loudkit.transports.grpc as grpc_mod
        from loudkit.cli import main

        seen: dict[str, object] = {}
        monkeypatch.setattr(
            grpc_mod, "serve", lambda _ckpt, _voices, **kwargs: seen.update(kwargs)
        )
        argv = [
            "serve",
            "--grpc",
            "--checkpoint",
            str(fake_ckpt),
            "--host",
            "127.0.0.2",
            "--port",
            "7003",
            "--first-chunk-tokens",
            "8",
        ]
        assert main(argv) == 0
        assert seen["host"] == "127.0.0.2"
        assert seen["port"] == 7003
        assert seen["first_chunk_tokens"] == 8

    def test_an_unmentioned_host_is_left_to_the_transport(self, fake_ckpt, monkeypatch) -> None:
        """`--host` states no default of its own any more.

        It had one, `"127.0.0.1"`, which is also both transports' signature
        default -- three copies of one number, and the copy in the CLI is what
        made "did the operator ask for a host" unanswerable, since a dropped
        `--host` and an absent one looked the same.
        """
        import inspect

        import loudkit.transports.http as http_mod
        from loudkit.cli import main

        assert inspect.signature(http_mod.serve).parameters["host"].default == "127.0.0.1"

        seen: dict[str, object] = {}
        monkeypatch.setattr(http_mod, "serve", lambda _ckpt, **kwargs: seen.update(kwargs))
        assert main(["serve", "--checkpoint", str(fake_ckpt)]) == 0
        assert "host" not in seen

        seen.clear()
        assert main(["serve", "--checkpoint", str(fake_ckpt), "--host", "127.0.0.2"]) == 0
        assert seen["host"] == "127.0.0.2"


# ---------------------------------------------------------------- error paths


class TestErrorPaths:
    def test_backend_runtime_error_is_one_line_not_a_traceback(
        self, fake_ckpt, fake_voice_file, monkeypatch, capsys
    ) -> None:
        """The backends raise RuntimeError as well as ValueError.

        Only ValueError was caught, so half the backends' refusals reached the
        user as a multi-line interpreter traceback — noise in a terminal, and
        considerably worse in one being read aloud by a screen reader.
        """
        import loudkit

        def explode(*_a, **_k):
            raise RuntimeError("backend exploded")

        monkeypatch.setattr(loudkit, "load", explode)

        from loudkit.cli import main

        code = main(
            ["speak", "--checkpoint", str(fake_ckpt), "--voice", str(fake_voice_file), "hi"]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "backend exploded" in err
        assert "Traceback" not in err

    def test_debug_keeps_the_traceback(self, fake_ckpt, fake_voice_file, monkeypatch) -> None:
        """A diagnosis that cannot be reached is not a diagnosis."""
        import loudkit

        def explode(*_a, **_k):
            raise RuntimeError("backend exploded")

        monkeypatch.setattr(loudkit, "load", explode)

        from loudkit.cli import main

        with pytest.raises(RuntimeError, match="backend exploded"):
            main(
                [
                    "--debug",
                    "speak",
                    "--checkpoint",
                    str(fake_ckpt),
                    "--voice",
                    str(fake_voice_file),
                    "hi",
                ]
            )

    @pytest.mark.parametrize(
        "failure",
        [
            ModuleNotFoundError("No module named 'soundfile'", name="soundfile"),
            FileNotFoundError(2, "No such file or directory", "/nowhere/loudr-1.safetensors"),
            UnsupportedLanguageError("kl", language="kl", supported=("en",)),
            NotImplementedError("no such path on this backend"),
            RuntimeError("backend exploded"),
            OSError("the file is present but could not be read"),
        ],
        ids=lambda failure: type(failure).__name__,
    )
    def test_debug_keeps_the_traceback_whichever_clause_would_have_caught_it(
        self, failure, fake_ckpt, fake_voice_file, monkeypatch
    ) -> None:
        """One case per clause of the ladder, because the flag is one rule.

        ``--debug`` is documented as re-raising, and a reader who passes it to
        diagnose a missing extra or a missing file has to get the traceback
        that names which import or which path. A guard written per clause is a
        guard a clause can be written without, so every clause is asserted.
        """
        import loudkit

        def explode(*_a, **_k):
            raise failure

        monkeypatch.setattr(loudkit, "load", explode)

        from loudkit.cli import main

        with pytest.raises(BaseException) as caught:  # the identity is the assertion
            main(
                [
                    "--debug",
                    "speak",
                    "--checkpoint",
                    str(fake_ckpt),
                    "--voice",
                    str(fake_voice_file),
                    "hi",
                ]
            )
        assert caught.value is failure

    def _speak_raising(self, exc, fake_ckpt, fake_voice_file, monkeypatch, capsys) -> str:
        """Run `speak` against a `loudkit.load` that raises, return stderr."""
        import loudkit

        def explode(*_a, **_k):
            raise exc

        monkeypatch.setattr(loudkit, "load", explode)

        from loudkit.cli import main

        assert (
            main(
                ["speak", "--checkpoint", str(fake_ckpt), "--voice", str(fake_voice_file), "hi"]
            )
            == 1
        )
        return capsys.readouterr().err

    def test_clone_refuses_a_language_this_build_cannot_read(self, tmp_path, capsys) -> None:
        """Before the ten seconds and before the file, not after both.

        `--language` is what the engine reads text as whenever a call names
        none, so a value with no grammar behind it makes the profile unusable
        at its own default. The refusal used to arrive at `speak` time: one
        enrollment and one saved voice too late.
        """
        import loudkit
        from loudkit.cli import main

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")
        # Present but never opened: the refusal has to land before anything
        # reads the checkpoint, which is the whole point of it landing early.
        checkpoint = tmp_path / "loudr-1.safetensors"
        checkpoint.write_bytes(b"")
        assert (
            main(
                [
                    "clone",
                    str(audio),
                    "--checkpoint",
                    str(checkpoint),
                    "--name",
                    "v",
                    "--language",
                    "klingon",
                ]
            )
            == 1
        )
        err = capsys.readouterr().err
        assert "klingon" in err
        # The roster, not just a refusal: the reader has to be told what to
        # pass instead.
        for language in loudkit.languages():
            assert language in err

    def test_clone_refuses_a_language_the_way_text_does(self, tmp_path) -> None:
        """One refusal, one implementation.

        `clone` used to print its own `unsupported: ...` sentence and return 1,
        re-spelling by hand what `text` gets by raising the error class the
        dispatcher already prints. Two implementations of one refusal drift;
        this pins that `clone` raises the class rather than imitating it.
        """
        from loudkit.cli import main
        from loudkit.errors import UnsupportedLanguageError

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"")
        checkpoint = tmp_path / "loudr-1.safetensors"
        checkpoint.write_bytes(b"")
        with pytest.raises(UnsupportedLanguageError) as caught:
            main(
                [
                    "--debug",
                    "clone",
                    str(audio),
                    "--checkpoint",
                    str(checkpoint),
                    "--name",
                    "v",
                    "--language",
                    "klingon",
                ]
            )
        assert caught.value.language == "klingon"

    def test_unsupported_language_is_a_message_not_a_traceback(
        self, fake_ckpt, fake_voice_file, monkeypatch, capsys
    ) -> None:
        """`GraphemeTextFrontend` refuses a language off the roster with
        `UnsupportedLanguageError`, which is neither ValueError nor
        FileNotFoundError and used to reach the user raw — for what is really a
        supported-input question."""
        from loudkit.errors import UnsupportedLanguageError

        err = self._speak_raising(
            UnsupportedLanguageError(
                "language 'zh' needs model-based preprocessing",
                language="zh",
                supported=("en", "pl"),
            ),
            fake_ckpt,
            fake_voice_file,
            monkeypatch,
            capsys,
        )
        assert "unsupported" in err
        assert "Traceback" not in err

    def test_a_backend_stub_is_not_reported_as_unsupported_input(
        self, fake_ckpt, fake_voice_file, monkeypatch, capsys
    ) -> None:
        """The CLI's half of the 400-vs-500 distinction.

        Catching the builtin `NotImplementedError` printed "unsupported: ..."
        for a stub method in a backend — telling the user to change a command
        that was never wrong, and hiding the defect from whoever could fix it.
        Only `UnsupportedLanguageError` is a question about the input now;
        everything else is reported as a bug in loudkit, with the traceback one
        flag away.
        """
        err = self._speak_raising(
            NotImplementedError("mel decoder for this backend is a stub"),
            fake_ckpt,
            fake_voice_file,
            monkeypatch,
            capsys,
        )
        assert "internal error" in err
        assert "bug in loudkit" in err
        assert "unsupported" not in err
        assert "Traceback" not in err

    def test_missing_optional_dependency_explains_the_extra(self) -> None:
        from loudkit.cli import _explain_missing

        exc = ModuleNotFoundError("No module named 'torch'")
        exc.name = "torch"
        msg = _explain_missing(exc)
        assert "loudkit[torch]" in msg

        exc = ModuleNotFoundError("No module named 'onnxruntime'")
        exc.name = "onnxruntime"
        assert "loudkit[onnx]" in _explain_missing(exc)

    def test_unknown_dependency_is_named_without_a_fix(self) -> None:
        from loudkit.cli import _explain_missing

        exc = ModuleNotFoundError("No module named 'zorp'")
        exc.name = "zorp"
        assert "zorp" in _explain_missing(exc)
        assert "[" not in _explain_missing(exc)

    def test_a_nameless_import_failure_names_no_empty_package(self) -> None:
        """`exc.name` is optional, and most raisers here do not set it.

        `hub._hub` raises with a message and no name, so the name-shaped
        sentence rendered as `the '' package` — a package with no name, which
        the reader cannot install and cannot look up.
        """
        from loudkit.cli import _explain_missing

        msg = _explain_missing(ModuleNotFoundError())

        assert "''" not in msg
        assert '""' not in msg

    def test_a_written_message_reaches_the_user(self) -> None:
        """A raiser that wrote instructions keeps them.

        The message `hub` raises names the pip command and the alternative to
        installing anything. Replacing it with a sentence rebuilt from the
        module name threw away both, so the careful message never arrived.
        """
        from loudkit.cli import _explain_missing
        from loudkit.hub import _MISSING_HUB

        written = _MISSING_HUB.format(ref="a repo id")
        msg = _explain_missing(ModuleNotFoundError(written))

        assert msg == written
        assert "loudkit[hub]" in msg

    def test_a_name_survives_a_written_message_being_absent(self) -> None:
        """An import failure carrying only a name still names its extra."""
        from loudkit.cli import _explain_missing

        msg = _explain_missing(ModuleNotFoundError(name="mcp"))

        assert "mcp" in msg
        assert "loudkit[mcp]" in msg


# --------------------------------------------------------- doctor and verify


class TestDoctor:
    def test_doctor_runs_and_exits_zero(self, capsys) -> None:
        """A diagnosis is not a failure: doctor reads state, changes nothing,
        and exits 0 whatever it finds on disk. A ``--checkpoint`` that is not
        there is refused before dispatch, like every other command's."""
        from loudkit.cli import main

        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "backends:" in out
        assert "loudkit" in out

    def test_doctor_looks_where_checkpoint_points(
        self, fake_voice_file, capsys, monkeypatch, tmp_path
    ) -> None:
        """``--checkpoint`` is documented as a release directory; doctor scans
        it, not the working directory it was run from."""
        from loudkit.cli import main

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert main(["doctor", "--checkpoint", str(fake_voice_file.parent)]) == 0
        out = capsys.readouterr().out
        assert f"{fake_voice_file}  (voice)" in out
        assert "under the current directory" not in out

    def test_a_device_this_machine_lacks_is_one_line(
        self, fake_ckpt, tmp_path, capsys, monkeypatch
    ) -> None:
        """``--device cuda`` on a box without CUDA is a diagnosis, not a torch
        traceback; the check runs before the checkpoint is opened.

        The voice is a real profile file so that the device refusal is the
        first thing here that can fail: `speak` reads the text and the voice
        before it reads the checkpoint, and a voice that is not there would
        answer first.
        """
        import torch

        from loudkit.cli import main

        voice = fake_voice().save(tmp_path / "device-test-voice.safetensors")
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        assert (
            main(
                [
                    "speak",
                    "--checkpoint",
                    str(fake_ckpt),
                    "--voice",
                    str(voice),
                    "--device",
                    "cuda",
                    "hi",
                ]
            )
            == 1
        )
        err = capsys.readouterr().err
        assert "error: device 'cuda' is not available on this machine" in err
        assert "Traceback" not in err


class TestVerify:
    def test_a_valid_voice_profile_verifies(self, tmp_path, capsys) -> None:
        from loudkit.cli import main

        path = fake_voice().save(tmp_path / "v.safetensors")
        assert main(["verify", str(path)]) == 0
        out = capsys.readouterr().out
        assert "a voice profile, and a valid one" in out
        assert "sha256:" in out

    def test_a_wav_without_provenance_says_so(self, tmp_path, capsys) -> None:
        import numpy as np
        import soundfile as sf

        from loudkit.cli import main

        wav = tmp_path / "plain.wav"
        sf.write(str(wav), np.zeros(2400, dtype=np.float32), 24_000)
        assert main(["verify", str(wav)]) == 1
        assert "no provenance" in capsys.readouterr().out

    def test_a_provenanced_wav_verifies_and_prints_its_identity(self, tmp_path, capsys) -> None:
        import numpy as np

        from loudkit.cli import main
        from loudkit.engine import Provenance, Result, StageTimings

        result = Result(
            audio=np.zeros(2400, dtype=np.float32),
            tokens=[],
            mel=np.zeros((80, 4), dtype=np.float32),
            seed=1,
            sample_rate=24_000,
            timings=StageTimings(0.0, 0.0, 0.0),
            provenance=Provenance(
                algorithm_fingerprint="ab" * 8,
                recipe_version="loudkit-1",
                voice="kathleen",
                checkpoint_sha256="dd" * 32,
                backend="torch",
            ),
        )
        wav = tmp_path / "prov.wav"
        result.save(str(wav))
        assert main(["verify", str(wav)]) == 0
        out = capsys.readouterr().out
        assert "provenance verified" in out
        assert "dd" * 32 in out

    def test_an_unrelated_safetensors_is_named_not_guessed(self, tmp_path, capsys) -> None:
        import numpy as np
        from safetensors.numpy import save_file

        from loudkit.cli import main

        path = tmp_path / "other.safetensors"
        save_file({"weights": np.zeros(4, dtype=np.float32)}, str(path))
        assert main(["verify", str(path)]) == 1
        assert "neither a loudkit checkpoint nor a voice profile" in capsys.readouterr().out


class TestVoicesCommand:
    def test_lists_a_local_release_tree(self, tmp_path, capsys) -> None:
        from loudkit.cli import main

        (tmp_path / "voices").mkdir()
        fake_voice().save(tmp_path / "voices" / "kathleen.safetensors")
        fake_voice().save(tmp_path / "voices" / "gosia.safetensors")
        assert main(["voices", str(tmp_path)]) == 0
        assert capsys.readouterr().out.splitlines() == ["gosia", "kathleen"]

    def test_an_empty_release_is_an_error_not_silence(self, tmp_path, capsys) -> None:
        from loudkit.cli import main

        (tmp_path / "voices").mkdir()
        assert main(["voices", str(tmp_path)]) == 1
        err = capsys.readouterr().err
        # The message has to name what was looked for. "no voices" was true and
        # unusable: a mistyped repo id, a release built without its voice
        # directory and a revision that predates one all produced it.
        assert "voices/" in err
        assert ".safetensors" in err
        assert str(tmp_path) in err

    def test_a_name_the_console_cannot_spell_still_prints(self, tmp_path, monkeypatch) -> None:
        """A voice name outside the locale's encoding must not kill the listing.

        Nine languages ship voices and cp1252 — a Windows console's default —
        cannot represent most of them. Python encodes stdout with the locale's
        encoding, so `print` raised UnicodeEncodeError on the name itself: the
        release was intact, the voice was found, and the command died reporting
        it. The stream here is a real cp1252 writer, not a mock, because the
        bug lived in the encoder rather than in anything loudkit called.
        """
        import io
        import sys

        from loudkit.cli import main

        (tmp_path / "voices").mkdir()
        fake_voice().save(tmp_path / "voices" / "kathleen.safetensors")
        fake_voice().save(tmp_path / "voices" / "pl_gałczyński.safetensors")

        raw = io.BytesIO()
        monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
        assert main(["voices", str(tmp_path)]) == 0
        sys.stdout.flush()

        assert raw.getvalue().decode("utf-8").splitlines() == [
            "kathleen",
            "pl_gałczyński",
        ]


# ------------------------------------------------------------- --provider


class TestProviderFlag:
    """``--provider`` names the onnxruntime execution provider, in the same
    five spellings the Rust and Go CLIs take. It reaches the engine as the one
    named field of an ``ExecutionConfig``, so a run can be asked for one
    provider and can never answer on another."""

    @pytest.mark.parametrize("provider", ["auto", "cpu", "cuda", "coreml", "directml"])
    def test_the_five_spellings_are_accepted(self, provider: str) -> None:
        from loudkit.cli import build_parser

        args = build_parser().parse_args(
            ["speak", "--checkpoint", "x", "--voice", "v", "--provider", provider, "hi"]
        )
        assert args.provider == provider

    def test_omitted_is_none(self) -> None:
        """None, not "auto": unset and "asked for auto" are different requests,
        and only the first leaves the build's own choice alone."""
        from loudkit.cli import build_parser

        args = build_parser().parse_args(["speak", "--checkpoint", "x", "--voice", "v", "hi"])
        assert args.provider is None

    def test_unknown_provider_names_all_five(self, capsys) -> None:
        from loudkit.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["speak", "--checkpoint", "x", "--voice", "v", "--provider", "metal", "hi"]
            )
        err = capsys.readouterr().err
        for name in ("auto", "cpu", "cuda", "coreml", "directml"):
            assert name in err

    @pytest.mark.parametrize("command", ["speak", "doctor", "serve"])
    def test_every_engine_building_command_takes_it(self, command: str) -> None:
        from loudkit.cli import build_parser

        argv = ["--checkpoint", "x", "--device", "onnx", "--provider", "cuda"]
        if command == "speak":
            argv += ["hi", "--voice", "v"]
        assert build_parser().parse_args([command, *argv]).provider == "cuda"

    def test_provider_reaches_execution(self, fake_ckpt, fake_voice_file, monkeypatch) -> None:
        """The flag has to arrive as an override on the config the engine is
        built from. A flag that parses and then evaporates is worse than none:
        the run reports a provider it never used."""
        import loudkit

        captured = {}

        def capture_load(*args, **kwargs):
            captured["execution"] = kwargs.get("execution")
            return fake_engine()

        monkeypatch.setattr(loudkit, "load", capture_load)
        monkeypatch.setattr(loudkit.VoiceProfile, "load", _fake_load_voice, raising=False)

        from loudkit.cli import main

        rc = main(
            [
                "speak",
                "--checkpoint",
                str(fake_ckpt),
                "--voice",
                str(fake_voice_file),
                "--device",
                "onnx",
                "--provider",
                "cuda",
                "-o",
                str(fake_voice_file.parent / "out.wav"),
                "hello",
            ]
        )
        assert rc == 0
        execution = captured["execution"]
        assert execution.onnx_provider == "cuda"
        # Only the provider is named: --provider is not a request for graphs,
        # for a compile, or for a device the user did not type.
        from dataclasses import fields

        named = {f.name for f in fields(execution) if getattr(execution, f.name) is not None}
        assert named == {"onnx_provider"}

    def test_override_lands_on_the_config(self) -> None:
        """The override the CLI builds must produce that provider on a real
        ExecutionConfig, which is what the ONNX backend reads."""
        from loudkit.cli import _execution, build_parser
        from loudkit.config import ExecutionConfig

        args = build_parser().parse_args(
            [
                "speak",
                "--checkpoint",
                "x",
                "--voice",
                "v",
                "--device",
                "onnx",
                "--provider",
                "coreml",
                "hi",
            ]
        )
        execution = _execution(args)
        assert execution is not None
        assert execution.resolved(ExecutionConfig(device="onnx")).onnx_provider == "coreml"

    @pytest.mark.parametrize("device", [None, "cpu", "cuda", "mps", "coreml"])
    def test_provider_without_the_onnx_backend_is_refused(
        self, fake_ckpt, fake_voice_file, capsys, monkeypatch, device
    ) -> None:
        """A provider on a torch device is a conflict, not a preference. The
        alternative is a run asked for cuda that answers on the torch CPU
        backend, which is a wrong number carrying a right label."""
        import loudkit

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)
        monkeypatch.setattr(loudkit.VoiceProfile, "load", _fake_load_voice, raising=False)

        from loudkit.cli import main

        argv = [
            "speak",
            "--checkpoint",
            str(fake_ckpt),
            "--voice",
            str(fake_voice_file),
            "--provider",
            "cuda",
            "hi",
        ]
        if device is not None:
            argv += ["--device", device]
        assert main(argv) == 1
        err = capsys.readouterr().err
        assert "--provider cuda needs --device onnx" in err


# ------------------------------------------------------------------ pinning


class TestRevisionFlag:
    """``--revision`` pins what a repo id resolves to, on every subcommand that
    takes one.

    Guide 1 tells production to pin the revision, because a repo id without one
    follows the default branch and the same command can return different
    weights a week later. The flag lived on ``download`` and ``voices`` only,
    so every subcommand that *runs* a checkpoint — the ones where the moving
    weights would actually speak — had no spelling for it.
    """

    ENGINE_COMMANDS = ("speak", "doctor", "serve")

    @pytest.mark.parametrize("command", ENGINE_COMMANDS)
    def test_every_subcommand_that_takes_a_repo_id_takes_a_revision(self, command: str) -> None:
        from loudkit.cli import build_parser

        argv = [command, "--checkpoint", "loudreader/loudr-1", "--revision", "a1b2c3d"]
        if command == "speak":
            argv += ["hi", "--voice", "v"]
        assert build_parser().parse_args(argv).revision == "a1b2c3d"

    @pytest.mark.parametrize("command", ENGINE_COMMANDS)
    def test_it_is_in_the_help_a_reader_greps(self, command: str) -> None:
        """The guide says "pin the revision"; ``--help`` is where a reader looks
        for how. A flag that parses but is undocumented is not reachable."""
        from loudkit.cli import build_parser

        sub = build_parser()._subparsers._group_actions[0].choices
        assert "--revision" in sub[command].format_help()

    @pytest.mark.parametrize("command", ENGINE_COMMANDS)
    def test_omitted_is_none(self, command: str) -> None:
        from loudkit.cli import build_parser

        argv = [command, "--checkpoint", "loudreader/loudr-1"]
        if command == "speak":
            argv += ["hi", "--voice", "v"]
        assert build_parser().parse_args(argv).revision is None

    def test_speak_pins_both_the_checkpoint_and_the_voice(self, monkeypatch, tmp_path) -> None:
        """A pinned checkpoint read by a voice from the moving branch is pinned
        in name only: the pair is what reproduces, not the checkpoint alone."""
        import loudkit
        import loudkit.hub

        seen: dict[str, object] = {}

        def capture_load(checkpoint, **kwargs):
            seen["checkpoint_revision"] = kwargs.get("revision")
            return fake_engine()

        def capture_voice(name, *, repo=None, revision=None):
            seen["voice_repo"] = repo
            seen["voice_revision"] = revision
            return tmp_path / "unused.safetensors"

        monkeypatch.setattr(loudkit, "load", capture_load)
        monkeypatch.setattr(loudkit.hub, "resolve_voice", capture_voice)
        monkeypatch.setattr(loudkit.VoiceProfile, "load", _fake_load_voice, raising=False)

        from loudkit.cli import main

        rc = main(
            [
                "speak",
                "--checkpoint",
                "loudreader/loudr-1",
                "--revision",
                "a1b2c3d",
                "--voice",
                "kathleen",
                "-o",
                str(tmp_path / "out.wav"),
                "hello",
            ]
        )
        assert rc == 0
        assert seen["checkpoint_revision"] == "a1b2c3d"
        assert seen["voice_repo"] == "loudreader/loudr-1"
        assert seen["voice_revision"] == "a1b2c3d"

    def test_a_server_resolves_the_pin_before_handing_it_over(self, monkeypatch) -> None:
        """No transport's ``serve`` takes a revision, so the CLI resolves the
        repo id itself and hands over the file the pin names. A flag that
        parsed and then evaporated would leave the deployment believing it had
        pinned something."""
        import loudkit.hub
        from loudkit.transports import http

        seen: dict[str, object] = {}

        def capture_resolve(ref, *, revision=None, backend="torch"):
            seen["ref"] = ref
            seen["revision"] = revision
            seen["backend"] = backend
            return Path("/cache/snapshots/a1b2c3d/loudr-1.safetensors")

        monkeypatch.setattr(loudkit.hub, "resolve_checkpoint", capture_resolve)
        monkeypatch.setattr(http, "serve", lambda ckpt, **_kw: seen.update(served=ckpt))

        from loudkit.cli import main

        assert (
            main(
                [
                    "serve",
                    "--checkpoint",
                    "loudreader/loudr-1",
                    "--revision",
                    "a1b2c3d",
                ]
            )
            == 0
        )
        assert seen["ref"] == "loudreader/loudr-1"
        assert seen["revision"] == "a1b2c3d"
        assert seen["served"] == str(Path("/cache/snapshots/a1b2c3d/loudr-1.safetensors"))

    def test_a_path_is_passed_through_untouched(self, fake_ckpt) -> None:
        """A path names its bytes already. Passed through rather than refused,
        because ``loudkit.load`` answers the same pairing the same way and one
        flag must not mean two things."""
        from loudkit.cli import _pinned_checkpoint, build_parser

        args = build_parser().parse_args(
            ["serve", "--checkpoint", str(fake_ckpt), "--revision", "a1b2c3d"]
        )
        assert _pinned_checkpoint(args) == str(fake_ckpt)

    def test_without_a_revision_the_repo_id_reaches_the_transport_unresolved(self) -> None:
        """The unpinned path is unchanged: the transport still receives the repo
        id and still resolves it itself."""
        from loudkit.cli import _pinned_checkpoint, build_parser

        args = build_parser().parse_args(["serve", "--checkpoint", "loudreader/loudr-1"])
        assert _pinned_checkpoint(args) == "loudreader/loudr-1"


class TestDoctorReportsOnlyLoudkitArtefacts:
    """``doctor`` output is what people paste into bug reports.

    It listed every repo in the shared Hugging Face cache and every
    ``.safetensors`` under the working directory, most of which belongs to
    other libraries entirely. That is noise in a diagnostic and a disclosure in
    a bug report: which models a machine has downloaded is not loudkit's to
    publish. What answers "why did loading fail" is which *loudkit* artefacts
    are here, and that is what stays.
    """

    def test_a_strangers_cached_repo_is_not_named(self, tmp_path, monkeypatch, capsys) -> None:
        from loudkit.cli import main

        cache = tmp_path / "hub"
        ours = cache / "models--loudreader--loudr-1" / "snapshots" / "a1b2c3d"
        theirs = cache / "models--someone--private-llm" / "snapshots" / "deadbee"
        ours.mkdir(parents=True)
        theirs.mkdir(parents=True)
        _pack_checkpoint(ours / "loudr-1.safetensors")
        fake_voice().save(theirs / "model.safetensors")

        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache), raising=False)
        monkeypatch.chdir(tmp_path)
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "loudreader/loudr-1" in out
        assert "someone" not in out
        assert "private-llm" not in out

    def test_an_unrelated_local_safetensors_is_not_listed(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """A voice and a checkpoint are named, and are named as what they are.
        Somebody else's tensors in the same directory are not loudkit's to
        report on."""
        from loudkit.cli import main

        fake_voice().save(tmp_path / "kathleen.safetensors")
        _pack_checkpoint(tmp_path / "loudr-1.safetensors")
        _write_foreign_safetensors(tmp_path / "someones-lora.safetensors")

        monkeypatch.chdir(tmp_path)
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "kathleen.safetensors  (voice)" in out
        assert "loudr-1.safetensors  (checkpoint)" in out
        assert "someones-lora" not in out

    def test_a_truncated_checkpoint_is_named_rather_than_dropped(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The one file `doctor` most needed to mention was the one it hid.

        A header that will not parse and a file belonging to another library
        were given the same answer: not ours, do not mention. So a download
        that died halfway left a 700 MB file with the right name in the working
        directory and `doctor` reported "no loudkit checkpoint or voice under
        the current directory".
        """
        from loudkit.cli import main

        (tmp_path / "loudr-1.safetensors").write_bytes(b"\x00" * 64)
        monkeypatch.chdir(tmp_path)
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "loudr-1.safetensors  (unreadable)" in out

    def test_the_listing_says_how_many_it_left_out(self, tmp_path, monkeypatch, capsys) -> None:
        """A silent cap is a listing that lies by omission.

        Both sections stopped at a bare `[:8]`, and eight entries is what a
        directory holding exactly eight also prints — so a ninth voice, quite
        possibly the one being asked about, vanished from a diagnostic without
        a word.
        """
        from loudkit.cli import _DOCTOR_LISTING_CAP, main

        for i in range(_DOCTOR_LISTING_CAP + 3):
            fake_voice().save(tmp_path / f"v{i:02d}.safetensors")
        monkeypatch.chdir(tmp_path)
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert f"and 3 more (listing stops at {_DOCTOR_LISTING_CAP})" in out

    def test_a_listing_inside_the_cap_says_nothing_about_it(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        from loudkit.cli import main

        fake_voice().save(tmp_path / "kathleen.safetensors")
        monkeypatch.chdir(tmp_path)
        assert main(["doctor"]) == 0
        assert "listing stops at" not in capsys.readouterr().out

    def test_a_bare_machine_says_so_rather_than_listing_nothing(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The reader ran this because loading failed. "Nothing here" is the
        answer to that question and has to be printed, not implied by silence."""
        from loudkit.cli import main

        cache = tmp_path / "hub"
        cache.mkdir()
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache), raising=False)
        monkeypatch.chdir(tmp_path)
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "no loudkit checkpoint or voice under the current directory" in out
        assert "no loudkit release in the hub cache" in out


class TestTopLevelSurface:
    """``loudkit --help`` lists the supported commands.

    These cover the path from a bare machine to a cloned voice
    speaking. Repo tooling (bench, profile) lives in ``tools/``, and the other
    two transports are flags on ``serve``.
    """

    COMMANDS = ["speak", "text", "clone", "voices", "download", "serve", "verify", "doctor"]

    def test_help_lists_exactly_the_commands(self, capsys) -> None:
        from loudkit.cli import COMMANDS, main

        assert list(COMMANDS) == self.COMMANDS
        assert main([]) == 1
        out = capsys.readouterr().out
        listed = [
            line.strip().split()[0]
            for line in out.splitlines()
            if line.startswith("    ") and not line.startswith("     ") and line.strip()
        ]
        assert listed == self.COMMANDS

    def test_the_usage_line_names_only_the_commands(self, capsys) -> None:
        from loudkit.cli import main

        assert main([]) == 1
        usage = capsys.readouterr().out.split("\n\n")[0]
        assert "{" + ",".join(self.COMMANDS) + "}" in usage
        for name in ("bench", "profile", "describe", "mcp", "grpc"):
            assert name not in usage

    def test_the_other_transports_are_flags_on_serve(self) -> None:
        from loudkit.cli import build_parser

        sub = build_parser()._subparsers._group_actions[0].choices
        help_text = sub["serve"].format_help()
        assert "--grpc" in help_text
        assert "--mcp" in help_text
        with pytest.raises(SystemExit):
            build_parser().parse_args(["serve", "--grpc", "--mcp"])

    def test_serve_grpc_hands_the_engine_to_the_grpc_transport(
        self, fake_ckpt, monkeypatch
    ) -> None:
        """Without ``--port`` the CLI states no port at all.

        The number belongs to the transport's own signature, and the CLI
        repeating it is how the two drift apart. What is asserted here is
        therefore the absence of the argument plus the default it falls to.
        """
        import inspect

        import loudkit.transports.grpc as grpc_mod
        from loudkit.cli import main

        assert inspect.signature(grpc_mod.serve).parameters["port"].default == 50051

        seen: dict[str, object] = {}

        def fake_serve(ckpt, voices, **kwargs):
            seen.update(ckpt=ckpt, **kwargs)

        monkeypatch.setattr(grpc_mod, "serve", fake_serve)
        assert main(["serve", "--grpc", "--checkpoint", str(fake_ckpt)]) == 0
        assert seen["ckpt"] == str(fake_ckpt)
        assert "port" not in seen

    def test_serve_grpc_passes_the_port_it_was_given(self, fake_ckpt, monkeypatch) -> None:
        import loudkit.transports.grpc as grpc_mod
        from loudkit.cli import main

        seen: dict[str, object] = {}

        def fake_serve(ckpt, voices, **kwargs):
            seen.update(kwargs)

        monkeypatch.setattr(grpc_mod, "serve", fake_serve)
        assert main(["serve", "--grpc", "--checkpoint", str(fake_ckpt), "--port", "7001"]) == 0
        assert seen["port"] == 7001

    def test_serve_http_leaves_its_port_to_the_transport(self, fake_ckpt, monkeypatch) -> None:
        """The HTTP half of the same rule, and the default it falls to."""
        import inspect

        import loudkit.transports.http as http_mod
        from loudkit.cli import main

        assert inspect.signature(http_mod.serve).parameters["port"].default == 8765

        seen: dict[str, object] = {}

        def fake_serve(ckpt, **kwargs):
            seen.update(kwargs)

        monkeypatch.setattr(http_mod, "serve", fake_serve)
        assert main(["serve", "--checkpoint", str(fake_ckpt)]) == 0
        assert "port" not in seen

        seen.clear()
        assert main(["serve", "--checkpoint", str(fake_ckpt), "--port", "7002"]) == 0
        assert seen["port"] == 7002


def _fake_enroll(monkeypatch, seen: dict) -> None:
    """Replace ``loudkit.enroll`` with a recorder returning a real profile.

    A real :class:`VoiceProfile` because the command writes it and the tests
    below read the bytes and the mode back; a recorder because what the command
    owes the library is the arguments it was given, unchanged.
    """
    import dataclasses

    import loudkit

    def enroll(audio, checkpoint, **kwargs):
        seen["audio"] = audio
        seen["checkpoint"] = checkpoint
        seen.update(kwargs)
        return dataclasses.replace(
            fake_voice(), name=kwargs.get("name", ""), language=kwargs.get("language", "en")
        )

    monkeypatch.setattr(loudkit, "enroll", enroll)


@pytest.fixture
def recording(tmp_path):
    """A WAV that exists. Nothing reads it — ``enroll`` is faked."""
    path = tmp_path / "me.wav"
    path.write_bytes(b"RIFF....WAVE not really, never opened")
    return path


class TestCloneGrammar:
    def test_every_flag_of_the_contract(self, tmp_path) -> None:
        from loudkit.cli import build_parser

        args = build_parser().parse_args(
            [
                "clone",
                "me.wav",
                "--checkpoint",
                "loudreader/loudr-1",
                "--name",
                "mine",
                "--language",
                "pl",
                "-o",
                str(tmp_path / "out.safetensors"),
                "--revision",
                "v1",
                "--device",
                "cuda:1",
                "--force",
            ]
        )
        assert args.audio == "me.wav"
        assert args.checkpoint == "loudreader/loudr-1"
        assert args.name == "mine"
        assert args.language == "pl"
        assert str(args.output) == str(tmp_path / "out.safetensors")
        assert args.revision == "v1"
        assert args.device == "cuda:1"
        assert args.force is True

    def test_audio_name_and_language_are_all_required(self) -> None:
        """The three that identify the clone are explicit; the checkpoint defaults."""
        from loudkit.cli import build_parser

        base = ["clone", "me.wav", "--checkpoint", "c", "--name", "n", "--language", "en"]
        for drop in ("--name", "--language"):
            argv = list(base)
            i = argv.index(drop)
            del argv[i : i + 2]
            with pytest.raises(SystemExit):
                build_parser().parse_args(argv)
        with pytest.raises(SystemExit):
            build_parser().parse_args(base[:1] + base[2:])

    def test_defaults(self) -> None:
        from loudkit.cli import build_parser

        args = build_parser().parse_args(
            ["clone", "me.wav", "--checkpoint", "c", "--name", "n", "--language", "en"]
        )
        assert args.output is None
        assert args.revision is None
        assert args.device is None
        assert args.force is False

    @pytest.mark.parametrize(
        "device", ["cpu", "cuda", "cuda:0", "cuda:3", "mps", "onnx", "coreml"]
    )
    def test_enrollment_devices_are_accepted(self, device: str) -> None:
        from loudkit.cli import build_parser

        args = build_parser().parse_args(
            [
                "clone",
                "me.wav",
                "--checkpoint",
                "c",
                "--name",
                "n",
                "--language",
                "en",
                "--device",
                device,
            ]
        )
        assert args.device == device

    @pytest.mark.parametrize("device", ["tpu", "cuda:x"])
    def test_unknown_devices_are_refused(self, device: str) -> None:
        from loudkit.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    "clone",
                    "me.wav",
                    "--checkpoint",
                    "c",
                    "--name",
                    "n",
                    "--language",
                    "en",
                    "--device",
                    device,
                ]
            )


@requires_modules("torch", "torchaudio", "librosa")
class TestCloneCommand:
    def test_it_writes_the_default_path_and_passes_the_flags_through(
        self, recording, fake_ckpt, tmp_path, monkeypatch, capsys
    ) -> None:
        from loudkit.cli import main

        seen: dict = {}
        _fake_enroll(monkeypatch, seen)
        monkeypatch.chdir(tmp_path)

        code = main(
            [
                "clone",
                str(recording),
                "--checkpoint",
                str(fake_ckpt),
                "--name",
                "mine",
                "--language",
                "pl",
                "--device",
                "cpu",
                "--revision",
                "v1",
            ]
        )
        assert code == 0
        written = tmp_path / "voices" / "mine.safetensors"
        assert written.is_file()
        assert seen["audio"] == str(recording)
        assert seen["checkpoint"] == str(fake_ckpt)
        assert seen["name"] == "mine"
        assert seen["language"] == "pl"
        assert seen["device"] == "cpu"
        assert seen["revision"] == "v1"
        # The relative path the user gets back is the one they can hand to `speak`.
        assert "voices/mine.safetensors" in capsys.readouterr().out

    def test_the_language_tag_is_read_in_either_case(
        self, recording, fake_ckpt, tmp_path, monkeypatch
    ) -> None:
        """``--language EN`` is the same request as ``--language en``.

        `text` lower-cases the tag before it looks it up, so a tag refused
        here while `text` accepts it is one release answering the same request
        two ways. The normalised tag is what reaches `enroll`, so the profile
        records the tag the engine looks up.
        """
        from loudkit.cli import main

        seen: dict = {}
        _fake_enroll(monkeypatch, seen)
        monkeypatch.chdir(tmp_path)

        code = main(
            [
                "clone",
                str(recording),
                "--checkpoint",
                str(fake_ckpt),
                "--name",
                "mine",
                "--language",
                "EN",
            ]
        )
        assert code == 0
        assert seen["language"] == "en"

    @pytest.mark.parametrize("flag", [[], ["--no-end-in-silence"]])
    def test_the_pause_cut_is_the_default_and_the_flag_turns_it_off(
        self, recording, fake_ckpt, tmp_path, monkeypatch, flag
    ) -> None:
        """The command's default is the library's ``end_in_silence=True``.

        A shipped voice speaks from the edge of silence because its prompt was
        cut at a pause and padded; a clone made at the command line gets the
        same treatment unless the user asks for the clip as given.
        """
        from loudkit.cli import main

        seen: dict = {}
        _fake_enroll(monkeypatch, seen)
        monkeypatch.chdir(tmp_path)

        code = main(
            [
                "clone",
                str(recording),
                "--checkpoint",
                str(fake_ckpt),
                "--name",
                "mine",
                "--language",
                "en",
                "--device",
                "cpu",
                *flag,
            ]
        )
        assert code == 0
        assert seen["end_in_silence"] is (flag == [])

    def test_the_profile_it_writes_reads_back(
        self, recording, fake_ckpt, tmp_path, monkeypatch
    ) -> None:
        from loudkit.cli import main

        _fake_enroll(monkeypatch, {})
        out = tmp_path / "elsewhere" / "v.safetensors"
        assert (
            main(
                [
                    "clone",
                    str(recording),
                    "--checkpoint",
                    str(fake_ckpt),
                    "--name",
                    "mine",
                    "--language",
                    "pl",
                    "-o",
                    str(out),
                ]
            )
            == 0
        )
        profile = VoiceProfile.load(out)
        assert profile.name == "mine"
        assert profile.language == "pl"

    def test_output_flag_overrides_the_default(
        self, recording, fake_ckpt, tmp_path, monkeypatch
    ) -> None:
        from loudkit.cli import main

        _fake_enroll(monkeypatch, {})
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "chosen.safetensors"
        assert (
            main(
                [
                    "clone",
                    str(recording),
                    "--checkpoint",
                    str(fake_ckpt),
                    "--name",
                    "mine",
                    "--language",
                    "en",
                    "-o",
                    str(out),
                ]
            )
            == 0
        )
        assert out.is_file()
        assert not (tmp_path / "voices").exists()

    @pytest.mark.skipif(os.name != "posix", reason="0600 is a POSIX mode, not a Windows ACL")
    def test_the_written_profile_is_owner_only(
        self, recording, fake_ckpt, tmp_path, monkeypatch
    ) -> None:
        """A voice is derived from a recording of a person. Anything group- or
        world-readable has wider reach than the consent that covered it."""
        import stat

        from loudkit.cli import main

        _fake_enroll(monkeypatch, {})
        out = tmp_path / "v.safetensors"
        assert (
            main(
                [
                    "clone",
                    str(recording),
                    "--checkpoint",
                    str(fake_ckpt),
                    "--name",
                    "mine",
                    "--language",
                    "en",
                    "-o",
                    str(out),
                ]
            )
            == 0
        )
        assert stat.S_IMODE(out.stat().st_mode) == 0o600

    def test_it_refuses_to_overwrite_without_force(
        self, recording, fake_ckpt, tmp_path, monkeypatch, capsys
    ) -> None:
        from loudkit.cli import main

        _fake_enroll(monkeypatch, {})
        out = tmp_path / "v.safetensors"
        out.write_bytes(b"someone else's voice")

        argv = [
            "clone",
            str(recording),
            "--checkpoint",
            str(fake_ckpt),
            "--name",
            "mine",
            "--language",
            "en",
            "-o",
            str(out),
        ]
        assert main(argv) == 1
        assert out.read_bytes() == b"someone else's voice"
        assert "--force" in capsys.readouterr().err

        assert main([*argv, "--force"]) == 0
        assert VoiceProfile.load(out).name == "mine"

    def test_a_missing_recording_is_named_before_any_work(
        self, fake_ckpt, tmp_path, capsys
    ) -> None:
        from loudkit.cli import main

        missing = tmp_path / "nope.wav"
        assert (
            main(
                [
                    "clone",
                    str(missing),
                    "--checkpoint",
                    str(fake_ckpt),
                    "--name",
                    "mine",
                    "--language",
                    "en",
                ]
            )
            == 1
        )
        err = capsys.readouterr().err
        assert "audio not found" in err
        assert str(missing) in err

    def test_only_wav_and_flac(self, fake_ckpt, tmp_path, monkeypatch, capsys) -> None:
        """0.1 reads two lossless containers. A lossy file makes the clone worse
        than the person who ran the command can see."""
        from loudkit.cli import main

        _fake_enroll(monkeypatch, {})
        mp3 = tmp_path / "me.mp3"
        mp3.write_bytes(b"not really an mp3")
        assert (
            main(
                [
                    "clone",
                    str(mp3),
                    "--checkpoint",
                    str(fake_ckpt),
                    "--name",
                    "mine",
                    "--language",
                    "en",
                ]
            )
            == 1
        )
        assert "WAV or a FLAC" in capsys.readouterr().err

    def test_a_flac_is_accepted(self, fake_ckpt, tmp_path, monkeypatch) -> None:
        from loudkit.cli import main

        _fake_enroll(monkeypatch, {})
        flac = tmp_path / "me.flac"
        flac.write_bytes(b"fLaC not really, never opened")
        out = tmp_path / "v.safetensors"
        assert (
            main(
                [
                    "clone",
                    str(flac),
                    "--checkpoint",
                    str(fake_ckpt),
                    "--name",
                    "mine",
                    "--language",
                    "en",
                    "-o",
                    str(out),
                ]
            )
            == 0
        )

    @pytest.mark.parametrize("name", ["a/b", "../escape", ".hidden", "a\\b"])
    def test_a_name_that_is_not_a_filename_is_refused(
        self, name: str, recording, fake_ckpt, tmp_path, monkeypatch, capsys
    ) -> None:
        """``--name`` becomes a filename and a voice key. Refused rather than
        sanitised: a name silently rewritten is a voice the user cannot find
        again under the name they chose."""
        from loudkit.cli import main

        _fake_enroll(monkeypatch, {})
        monkeypatch.chdir(tmp_path)
        assert (
            main(
                [
                    "clone",
                    str(recording),
                    "--checkpoint",
                    str(fake_ckpt),
                    "--name",
                    name,
                    "--language",
                    "en",
                ]
            )
            == 1
        )
        assert "--name" in capsys.readouterr().err

    def test_a_failed_enrollment_leaves_no_partial_file(
        self, recording, fake_ckpt, tmp_path, monkeypatch
    ) -> None:
        """Written beside the target and moved onto it, so a run that dies
        halfway leaves the previous file or no file — never a truncated profile
        that ``VoiceProfile.load`` refuses much later."""
        import loudkit
        from loudkit.cli import main

        def boom(*args, **kwargs):
            raise RuntimeError("the encoder fell over")

        monkeypatch.setattr(loudkit, "enroll", boom)
        monkeypatch.chdir(tmp_path)
        assert (
            main(
                [
                    "clone",
                    str(recording),
                    "--checkpoint",
                    str(fake_ckpt),
                    "--name",
                    "mine",
                    "--language",
                    "en",
                ]
            )
            == 1
        )
        assert list((tmp_path / "voices").glob("*")) == [] or not (tmp_path / "voices").exists()

    def test_the_temp_name_comes_from_the_os(self, tmp_path) -> None:
        """The name is made by ``mkstemp`` in the destination directory, so no
        two calls can pick the same one. It is hidden, it says ``.partial``,
        and it is gone once the profile is under its own name."""
        from loudkit.cli import _save_voice_atomically

        written: list = []

        class _Profile:
            def save(self, path) -> None:
                written.append(path)
                path.write_bytes(b"profile")

        out = tmp_path / "voices" / "mine.safetensors"
        assert _save_voice_atomically(_Profile(), out) == out
        assert _save_voice_atomically(_Profile(), out) == out
        first, second = written
        assert first != second
        for tmp in (first, second):
            assert tmp.parent == out.parent
            assert tmp.name.startswith(".")
            assert ".partial" in tmp.name
            assert not tmp.exists()
        assert out.read_bytes() == b"profile"
        assert sorted(p.name for p in out.parent.iterdir()) == ["mine.safetensors"]

    def test_two_threads_cloning_at_once_never_see_each_other_s_bytes(self, tmp_path) -> None:
        """Two threads of one process writing the same output must each keep
        their own temp file.

        A name built in the process — even one carrying the pid — is shared by
        every thread in it, and then one thread's ``save`` overwrites the
        other's half-written temp, the first ``os.replace`` takes the name away
        from the second, and the published profile is whichever bytes lost the
        race.

        Deterministic, not timing-dependent: a barrier holds both threads
        between writing their temp and reading it back, so both temps exist at
        once on every run. Sharing one name is then a certain failure, not a
        likely one.
        """
        import threading

        from loudkit.cli import _save_voice_atomically

        out = tmp_path / "voices" / "mine.safetensors"
        both_written = threading.Barrier(2, timeout=30)
        seen: dict[bytes, bytes] = {}

        class _Profile:
            def __init__(self, payload: bytes) -> None:
                self.payload = payload

            def save(self, path) -> None:
                path.write_bytes(self.payload)
                both_written.wait()
                # What this thread's own temp holds once the other thread has
                # written its own. Anything but `self.payload` means the two
                # threads shared a file.
                seen[self.payload] = path.read_bytes()

        payloads = (b"the first voice", b"the second voice")
        failures: list[BaseException] = []

        def clone(payload: bytes) -> None:
            try:
                _save_voice_atomically(_Profile(payload), out)
            except BaseException as exc:  # the defect, reported by the assert below
                failures.append(exc)
                both_written.abort()

        threads = [threading.Thread(target=clone, args=(payload,)) for payload in payloads]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive()

        assert failures == []
        assert seen == {payload: payload for payload in payloads}
        # One of the two is published whole, and neither temp is left behind.
        assert out.read_bytes() in payloads
        assert sorted(p.name for p in out.parent.iterdir()) == ["mine.safetensors"]


class TestTheMcpServerResolvesARepoId:
    """`build_server` computed the voice directory beside the raw argument.

    For a repo id that is `Path("org/repo").parent / "voices"`, a local
    directory that does not exist, so the server came up with the release's
    voices silently absent. The other two transports already resolved first.
    """

    def test_a_repo_id_takes_its_voices_from_the_snapshot(self, tmp_path, monkeypatch) -> None:
        from pathlib import Path

        import loudkit.transports.mcp as mcp_mod
        import loudkit.transports.resolve as resolve_mod

        snapshot = tmp_path / "snap"
        (snapshot / "voices").mkdir(parents=True)
        (snapshot / "voices" / "joe.safetensors").write_bytes(b"v")
        ckpt = snapshot / "loudr-1.safetensors"
        ckpt.write_bytes(b"c")

        seen: dict = {}

        class _Library:
            def __init__(self, directory) -> None:
                seen["directory"] = Path(directory)

            def names(self):
                return []

        # On the shared resolver, which is where all three doors build one now.
        monkeypatch.setattr(resolve_mod, "VoiceLibrary", _Library)
        monkeypatch.setattr(mcp_mod, "_load_mcp", lambda: _StubMcp)

        def _resolved(_ref: str, **_kw: object) -> Path:
            return ckpt

        monkeypatch.setattr("loudkit.hub.resolve_checkpoint", _resolved)

        mcp_mod.build_server("loudreader/loudr-1", engine=object())
        assert seen["directory"] == snapshot / "voices", seen


class _StubMcp:
    """Enough of the MCP server surface for build_server to finish."""

    def __init__(self, *a, **kw) -> None:
        pass

    def tool(self, *a, **kw):
        return lambda fn: fn


@requires_modules("torch", "torchaudio", "librosa")
class TestCloneObtainsTheEnrollmentArtefact:
    """A release is two files, and the enrollment tensors are in the second.

    ``clone`` asks the hub which file that is and gets it before enrolling, so
    a release that cannot clone is refused in a second rather than after ten
    of model loading, or after a gigabyte of download.
    """

    def test_it_asks_the_hub_and_names_what_it_got(
        self, recording, fake_ckpt, tmp_path, monkeypatch, capsys
    ) -> None:
        from loudkit import hub
        from loudkit.cli import main

        artefact = tmp_path / "fake-enrollment.safetensors"
        artefact.write_bytes(b"the enrollment half")
        asked: dict = {}

        def resolve_enrollment_checkpoint(ref, *, revision=None):
            asked["ref"] = ref
            asked["revision"] = revision
            return artefact

        monkeypatch.setattr(hub, "resolve_enrollment_checkpoint", resolve_enrollment_checkpoint)
        _fake_enroll(monkeypatch, {})
        monkeypatch.chdir(tmp_path)
        assert (
            main(
                [
                    "clone",
                    str(recording),
                    "--checkpoint",
                    str(fake_ckpt),
                    "--name",
                    "mine",
                    "--language",
                    "en",
                    "--revision",
                    "v1",
                ]
            )
            == 0
        )
        assert asked == {"ref": str(fake_ckpt), "revision": "v1"}
        assert str(artefact) in capsys.readouterr().err
        assert (tmp_path / "voices" / "mine.safetensors").is_file()

    def test_a_release_with_no_enrollment_half_is_refused_before_the_work(
        self, recording, fake_ckpt, tmp_path, monkeypatch, capsys
    ) -> None:
        import loudkit
        from loudkit import hub
        from loudkit.cli import main

        def resolve_enrollment_checkpoint(ref, *, revision=None):
            raise FileNotFoundError(
                f"{ref} is a synthesis-only release, with no enrollment artefact to clone with."
            )

        def never(*args, **kwargs):
            raise AssertionError("enrolled from a release that cannot clone")

        monkeypatch.setattr(hub, "resolve_enrollment_checkpoint", resolve_enrollment_checkpoint)
        monkeypatch.setattr(loudkit, "enroll", never)
        monkeypatch.chdir(tmp_path)
        assert (
            main(
                [
                    "clone",
                    str(recording),
                    "--checkpoint",
                    str(fake_ckpt),
                    "--name",
                    "mine",
                    "--language",
                    "en",
                ]
            )
            == 1
        )
        assert "synthesis-only release" in capsys.readouterr().err
        assert not (tmp_path / "voices").exists()

    def test_a_pre_split_checkpoint_answers_for_itself(
        self, recording, tmp_path, monkeypatch, capsys
    ) -> None:
        """The real resolver, on the file that is on disk and published today:
        one packed checkpoint carrying both halves, which is its own enrollment
        artefact. No second file is looked for and the clone goes through."""
        from loudkit.cli import main

        checkpoint = tmp_path / "loudr-1.safetensors"
        _pack_checkpoint(checkpoint, enrollment=True)
        _fake_enroll(monkeypatch, {})
        monkeypatch.chdir(tmp_path)
        assert (
            main(
                [
                    "clone",
                    str(recording),
                    "--checkpoint",
                    str(checkpoint),
                    "--name",
                    "mine",
                    "--language",
                    "en",
                ]
            )
            == 0
        )
        assert f"enrollment tensors: {checkpoint}" in capsys.readouterr().err
        assert (tmp_path / "voices" / "mine.safetensors").is_file()

    def test_a_split_release_on_disk_resolves_to_the_enrollment_artefact(
        self, recording, tmp_path, monkeypatch, capsys
    ) -> None:
        """The real resolver again, on the two files a split release unpacks
        to: the synthesis half is named on the command line, the enrollment
        half is what a clone reads."""
        from loudkit.cli import main

        checkpoint = tmp_path / "loudr-1.safetensors"
        _pack_checkpoint(checkpoint, role="synthesis")
        artefact = tmp_path / "loudr-1-enrollment.safetensors"
        _pack_enrollment(artefact)
        _fake_enroll(monkeypatch, {})
        monkeypatch.chdir(tmp_path)
        assert (
            main(
                [
                    "clone",
                    str(recording),
                    "--checkpoint",
                    str(checkpoint),
                    "--name",
                    "mine",
                    "--language",
                    "en",
                ]
            )
            == 0
        )
        assert f"enrollment tensors: {artefact}" in capsys.readouterr().err

    def test_a_synthesis_only_release_on_disk_is_refused(
        self, recording, tmp_path, monkeypatch, capsys
    ) -> None:
        """A checkpoint that declares itself the synthesis half, with no
        enrollment artefact beside it, cannot clone — and says so before the
        model loading rather than from inside the enroller."""
        import loudkit
        from loudkit.cli import main

        checkpoint = tmp_path / "loudr-1.safetensors"
        _pack_checkpoint(checkpoint, role="synthesis")

        def never(*args, **kwargs):
            raise AssertionError("enrolled from a release that cannot clone")

        monkeypatch.setattr(loudkit, "enroll", never)
        monkeypatch.chdir(tmp_path)
        assert (
            main(
                [
                    "clone",
                    str(recording),
                    "--checkpoint",
                    str(checkpoint),
                    "--name",
                    "mine",
                    "--language",
                    "en",
                ]
            )
            == 1
        )
        err = capsys.readouterr().err
        assert "loudr-1-enrollment.safetensors" in err
        assert not (tmp_path / "voices").exists()


class TestCloneMissingExtras:
    """The missing-extra path names every extra the invocation needs, at once.

    Enrollment needs ``enroll``; a repo id needs ``hub`` on top of it. Learning
    about the second one after installing the first costs a 1.27 GB download to
    discover.
    """

    def test_a_repo_id_names_both_extras(self, recording, monkeypatch, capsys) -> None:
        import importlib.util

        from loudkit.cli import main

        real = importlib.util.find_spec

        def blind(name, *args, **kwargs):
            if name in ("torch", "torchaudio", "librosa", "huggingface_hub"):
                return None
            return real(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", blind)
        assert (
            main(
                [
                    "clone",
                    str(recording),
                    "--checkpoint",
                    "loudreader/loudr-1",
                    "--name",
                    "mine",
                    "--language",
                    "en",
                ]
            )
            == 1
        )
        err = capsys.readouterr().err
        assert "loudkit[enroll,hub]" in err
        assert "pip install" in err

    def test_a_local_checkpoint_needs_only_enroll(
        self, recording, fake_ckpt, monkeypatch, capsys
    ) -> None:
        """A path names its bytes, so nothing has to be fetched for it."""
        import importlib.util

        from loudkit.cli import main

        real = importlib.util.find_spec

        def blind(name, *args, **kwargs):
            if name in ("torch", "torchaudio", "librosa"):
                return None
            return real(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", blind)
        assert (
            main(
                [
                    "clone",
                    str(recording),
                    "--checkpoint",
                    str(fake_ckpt),
                    "--name",
                    "mine",
                    "--language",
                    "en",
                ]
            )
            == 1
        )
        err = capsys.readouterr().err
        assert "loudkit[enroll]" in err
        assert "hub" not in err

    def test_it_refuses_before_touching_the_output(
        self, recording, tmp_path, monkeypatch
    ) -> None:
        import importlib.util

        from loudkit.cli import main

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **k: None if name == "librosa" else real(name, *a, **k),
        )
        monkeypatch.chdir(tmp_path)
        assert (
            main(
                [
                    "clone",
                    str(recording),
                    "--checkpoint",
                    "loudreader/loudr-1",
                    "--name",
                    "mine",
                    "--language",
                    "en",
                ]
            )
            == 1
        )
        assert not (tmp_path / "voices").exists()


class TestDoctorReportsCloneReadiness:
    """``doctor`` answers "can this machine clone" as three questions.

    Each has its own remedy: the extra is a pip command, the encoder is a file
    the release either ships or does not, and the enrollment tensors are ~40%
    of the weights, in their own artefact that a synthesis-only release does
    not publish. One yes/no sent people reinstalling extras to fix a checkpoint.

    The tensors have two possible homes — the enrollment artefact beside the
    checkpoint, or the checkpoint itself for a release packed before the split
    — so the line names the file it found.
    """

    def test_it_has_a_cloning_section(self, capsys) -> None:
        from loudkit.cli import main

        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "cloning:" in out
        assert "loudkit clone" in out

    def test_the_extra_is_reported(self, monkeypatch, capsys) -> None:
        import importlib.util

        from loudkit.cli import main

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **k: None if name == "torchaudio" else real(name, *a, **k),
        )
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "missing torchaudio" in out
        assert "loudkit[enroll]" in out

    def test_a_pre_split_checkpoint_reads_as_cloning_capable(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """What is on disk and published today: one packed file carrying both
        halves. It still answers yes, and says the tensors are in it."""
        from loudkit.cli import main

        _pack_checkpoint(tmp_path / "loudr-1.safetensors", enrollment=True)
        (tmp_path / "ve.safetensors").write_bytes(b"the utterance voice encoder")
        monkeypatch.chdir(tmp_path)
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "enrollment tensors: in this file" in out
        assert "ve.safetensors" in out

    def test_a_split_release_names_the_enrollment_artefact(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """After the split the checkpoint holds no enrollment tensors and the
        artefact beside it does, so the answer is that file's name."""
        from loudkit.cli import main

        _pack_checkpoint(tmp_path / "loudr-1.safetensors")
        _pack_enrollment(tmp_path / "loudr-1-enrollment.safetensors")
        (tmp_path / "ve.safetensors").write_bytes(b"the utterance voice encoder")
        monkeypatch.chdir(tmp_path)
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "enrollment tensors: loudr-1-enrollment.safetensors" in out
        assert "enrollment tensors: none" not in out
        # The enrollment artefact is the answer for the checkpoint beside it,
        # not a checkpoint with a cloning line of its own.
        assert "loudr-1-enrollment.safetensors  enrollment tensors:" not in out

    def test_an_enrollment_artefact_short_of_its_tensors_is_not_a_yes(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The name is not the evidence. A file that is there and holds the
        wrong tensors would otherwise be found by the enroller, ten seconds
        into a clone."""
        from loudkit.cli import main

        _pack_checkpoint(tmp_path / "loudr-1.safetensors")
        _pack_checkpoint(tmp_path / "loudr-1-enrollment.safetensors")
        monkeypatch.chdir(tmp_path)
        assert main(["doctor"]) == 0
        assert "enrollment tensors: none" in capsys.readouterr().out

    def test_a_synthesis_only_release_says_which_half_is_missing(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        from loudkit.cli import main

        _pack_checkpoint(tmp_path / "loudr-1.safetensors")
        monkeypatch.chdir(tmp_path)
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "enrollment tensors: none" in out
        assert "loudr-1-enrollment.safetensors" in out
        assert "encoder: missing" in out

    def test_no_checkpoint_here_is_said_plainly(self, tmp_path, monkeypatch, capsys) -> None:
        from loudkit.cli import main

        monkeypatch.chdir(tmp_path)
        assert main(["doctor"]) == 0
        assert "no checkpoint under the current directory" in capsys.readouterr().out


def _pack_checkpoint(path, *, enrollment: bool = False, role: str | None = None) -> None:
    """A file that reads as a loudkit checkpoint: the embedded manifest is what
    says so, and nothing here loads the tensors.

    ``enrollment=True`` adds the two tensor groups a clone reads — the speaker
    encoder and the speech tokenizer — under the names the packed checkpoint
    uses, which is a checkpoint from before the two artefacts were split. It is
    what is on disk and published today, so it has to keep reading as
    cloning-capable from the safetensors header alone.

    ``role="synthesis"`` is the other half of that: the manifest of a file
    built after the split, which claims to carry synthesis and nothing else.
    The absence of the claim is what marks a pre-split file, so the default is
    to make none.
    """

    from safetensors.numpy import save_file

    tensors = {"t3.dummy": np.zeros(2, np.float32)}
    if enrollment:
        tensors["s3gen.speaker_encoder.dummy"] = np.zeros(2, np.float32)
        tensors["s3gen.tokenizer.dummy"] = np.zeros(2, np.float32)
    manifest: dict = {"format": "loudkit-checkpoint", "format_version": 1}
    if role is not None:
        manifest["artifact_role"] = role
    save_file(tensors, str(path), metadata={"manifest": json.dumps(manifest)})


def _pack_enrollment(path) -> None:
    """The enrollment half of a split release: the speaker encoder and the
    speech tokenizer, and none of the synthesis weights.

    Its manifest carries ``artifact_role: enrollment``, so a reader that asks
    the file what it is gets an answer, and ``doctor`` still recognises it by
    the name it sits under beside the checkpoint.
    """

    from safetensors.numpy import save_file

    save_file(
        {
            "s3gen.speaker_encoder.dummy": np.zeros(2, np.float32),
            "s3gen.tokenizer.dummy": np.zeros(2, np.float32),
        },
        str(path),
        metadata={
            "manifest": json.dumps(
                {
                    "format": "loudkit-checkpoint",
                    "format_version": 1,
                    "artifact_role": "enrollment",
                }
            )
        },
    )


def _write_foreign_safetensors(path) -> None:
    """Valid safetensors, no loudkit metadata — somebody else's file."""
    from safetensors.numpy import save_file

    save_file({"weight": np.zeros(2, np.float32)}, str(path))


# ------------------------------------------------------------------ download


class _FakeHubClient:
    """Records what ``snapshot_download`` was asked for and returns a prepared
    snapshot directory — the patterns are the contract, and asserting them
    needs no network and no 1.2 GB."""

    def __init__(self, root) -> None:
        self.root = root
        self.calls: list[dict] = []

    def list_repo_files(self, **kwargs):
        from pathlib import Path

        root = Path(self.root)
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())

    def hf_hub_download(self, *, filename: str, **kwargs):
        from pathlib import Path

        from huggingface_hub.errors import EntryNotFoundError

        path = Path(self.root) / filename
        if not path.is_file():
            raise EntryNotFoundError(f"404 {filename}")
        return str(path)

    def snapshot_download(self, **kwargs):
        self.calls.append(kwargs)
        return str(self.root)


_CORE_PATTERNS = [
    "*.safetensors",
    "manifest.json",
    "tokenizer.json",
    "release.json",
    "voices/*",
    "SHA256SUMS",
]
_CLONING_ONLY_FILES = ["ve.safetensors", "loudr-1-enrollment.safetensors"]
"""What a synthesis fetch leaves behind: the utterance voice encoder and the
enrollment artefact. They are the point of the split — a caller who never
clones does not move the ~40% of the weights only a clone reads."""

_ONNX_SYNTH_PATTERNS = [
    "onnx/t3_cond.onnx",
    "onnx/t3_prefill.onnx",
    "onnx/t3_step.onnx",
    "onnx/flow_encoder.onnx",
    "onnx/flow_estimator.onnx",
    "onnx/vocoder.onnx",
    # The record that says the six came from one export. Fetched with them:
    # a record shipped and not downloaded leaves the backend's mixed-set check
    # dead for everyone who got their assets the ordinary way.
    "onnx/export.json",
    "onnx/t3_pair_step.onnx",
    "onnx/t3_head2.onnx",
]
_ONNX_ENROLL_PATTERNS = [
    "onnx/s3_tokenizer.onnx",
    "onnx/camp.onnx",
    "onnx/voice_encoder.onnx",
]
_COREML_SYNTH_PATTERNS = [
    "coreml/flow_encoder.mlpackage/*",
    "coreml/flow_estimator.mlpackage/*",
    "coreml/vocoder.mlpackage/*",
    "coreml/export.json",  # see the ONNX list
    "coreml/t3_*.mlpackage/*",
]
_COREML_ENROLL_PATTERNS = [
    "coreml/s3_tokenizer.mlpackage/*",
    "coreml/camp.mlpackage/*",
    "coreml/voice_encoder.mlpackage/*",
]


class TestDownloadCommand:
    """``loudkit download`` fetches what one backend needs and nothing more.

    The old command fetched the torch set whatever the user ran, so Python on
    ONNX downloaded 1.2 GB and then had no ``onnx/``, and the ports' guides had
    to reach for the Hugging Face CLI. The patterns are asserted against what
    reaches ``snapshot_download`` — nothing here downloads anything.
    """

    def _run(self, tmp_path, monkeypatch, argv, *, omit: tuple[str, ...] = ()):
        """Run ``argv`` against a fake snapshot holding the full inventory.

        ``omit`` names pieces the fake repo does not hold — ``"voices"``,
        ``"encoder"``, ``"enrollment"``, ``"onnx"``, ``"coreml"`` — for the
        shortfall tests: a fetch that comes back without them must be an error,
        not a warning.

        Omitting the enrollment artefact also makes the checkpoint declare the
        synthesis role, because that claim is what makes the set synthesis-only:
        a checkpoint that declares nothing is a pre-split file, and a pre-split
        file carries the enrollment tensors itself.
        """
        import loudkit.hub as hub_mod
        from loudkit.cli import main

        root = tmp_path / "snapshot"
        root.mkdir()
        _pack_checkpoint(root / "loudr-1.safetensors", role="synthesis")
        if "enrollment" not in omit:
            _pack_enrollment(root / "loudr-1-enrollment.safetensors")
        # The two files every backend's set is short without.
        (root / "manifest.json").write_text("{}", encoding="utf-8")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        if "voices" not in omit:
            (root / "voices").mkdir()
            (root / "voices" / "joe.safetensors").write_bytes(b"v")
            (root / "voices" / "kathleen.safetensors").write_bytes(b"v")
        if "encoder" not in omit:
            (root / "ve.safetensors").write_bytes(b"ve")
        if "onnx" not in omit:
            (root / "onnx").mkdir()
            for stem in (
                "t3_cond",
                "t3_prefill",
                "t3_step",
                "flow_encoder",
                "flow_estimator",
                "vocoder",
                "s3_tokenizer",
                "camp",
                "voice_encoder",
            ):
                (root / "onnx" / f"{stem}.onnx").write_bytes(b"g")
        if "coreml" not in omit:
            for stem in (
                "flow_encoder",
                "flow_estimator",
                "vocoder",
                "s3_tokenizer",
                "camp",
                "voice_encoder",
            ):
                package = root / "coreml" / f"{stem}.mlpackage"
                (package / "Data").mkdir(parents=True)
                (package / "Manifest.json").write_text("{}", encoding="utf-8")
                (package / "Data" / "model.mlmodel").write_bytes(b"m")
        client = _FakeHubClient(root)
        monkeypatch.setattr(hub_mod, "_hub", lambda: client)
        verified: list = []
        monkeypatch.setattr(
            hub_mod, "_verify_sha256sums", lambda r, *, repo=None: verified.append((r, repo))
        )
        # These tests are about the patterns handed to the snapshot; the
        # judgement that runs before it (release.json, the listing) is
        # pinned in tests/test_hub.py and stubbed here like the checksum pass.
        monkeypatch.setattr(hub_mod, "_refuse_before_fetching", lambda *_a, **_k: None)
        rc = main(argv)
        return rc, client, verified

    def test_the_default_is_torch_synthesis_only(self, tmp_path, monkeypatch, capsys) -> None:
        rc, client, verified = self._run(
            tmp_path, monkeypatch, ["download", "loudreader/loudr-1"]
        )
        assert rc == 0
        (call,) = client.calls
        assert call["repo_id"] == "loudreader/loudr-1"
        assert call["revision"] is None
        assert call["local_dir"] is None
        assert call["allow_patterns"] == _CORE_PATTERNS
        # Synthesis only: the utterance voice encoder and the enrollment
        # artefact are the cloning pieces, and neither is fetched.
        assert call["ignore_patterns"] == _CLONING_ONLY_FILES
        assert verified, "the download must be checked against SHA256SUMS"
        out = capsys.readouterr().out
        assert "loudr-1.safetensors" in out
        assert "voices      2" in out

    def test_for_onnx_adds_the_six_synthesis_graphs(self, tmp_path, monkeypatch) -> None:
        rc, client, _ = self._run(
            tmp_path, monkeypatch, ["download", "loudreader/loudr-1", "--for", "onnx"]
        )
        assert rc == 0
        (call,) = client.calls
        assert call["allow_patterns"] == _CORE_PATTERNS + _ONNX_SYNTH_PATTERNS
        assert call["ignore_patterns"] == _CLONING_ONLY_FILES

    def test_for_coreml_adds_the_three_synthesis_packages(self, tmp_path, monkeypatch) -> None:
        rc, client, _ = self._run(
            tmp_path, monkeypatch, ["download", "loudreader/loudr-1", "--for", "coreml"]
        )
        assert rc == 0
        (call,) = client.calls
        assert call["allow_patterns"] == _CORE_PATTERNS + _COREML_SYNTH_PATTERNS
        assert call["ignore_patterns"] == _CLONING_ONLY_FILES

    @pytest.mark.parametrize(
        ("backend", "extra"),
        [
            ("torch", []),
            ("onnx", _ONNX_SYNTH_PATTERNS + _ONNX_ENROLL_PATTERNS),
            ("coreml", _COREML_SYNTH_PATTERNS + _COREML_ENROLL_PATTERNS),
        ],
    )
    def test_with_cloning_adds_the_enrollment_pieces(
        self, tmp_path, monkeypatch, backend, extra
    ) -> None:
        """The graph backends add their enrollment graphs; torch adds none,
        because Python enrols on torch straight from the enrollment checkpoint
        plus the encoder.

        Only torch uncovers those two torch-side files. Sending 528 MB of torch
        weights to a caller who asked for ONNX or CoreML is the overfetch the
        split exists to remove, and their own enrollers read the graphs above
        instead.
        """
        from loudkit.hub import ENROLLMENT_NAME, VOICE_ENCODER_NAME

        rc, client, _ = self._run(
            tmp_path,
            monkeypatch,
            ["download", "loudreader/loudr-1", "--for", backend, "--with-cloning"],
        )
        assert rc == 0
        (call,) = client.calls
        assert call["allow_patterns"] == _CORE_PATTERNS + extra
        if backend == "torch":
            assert call["ignore_patterns"] is None
        else:
            assert call["ignore_patterns"] == [VOICE_ENCODER_NAME, ENROLLMENT_NAME]

    def test_a_missing_graph_set_is_an_error_not_a_warning(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The repo held no ``onnx/``: the command used to print the path to a
        directory that does not exist and exit 0, which every script and every
        port's setup guide read as success."""
        rc, _, _ = self._run(
            tmp_path,
            monkeypatch,
            ["download", "loudreader/loudr-1", "--for", "onnx"],
            omit=("onnx",),
        )
        assert rc != 0
        err = capsys.readouterr().err
        assert "onnx/t3_step.onnx" in err

    def test_a_missing_coreml_package_is_an_error(self, tmp_path, monkeypatch, capsys) -> None:
        rc, _, _ = self._run(
            tmp_path,
            monkeypatch,
            ["download", "loudreader/loudr-1", "--for", "coreml"],
            omit=("coreml",),
        )
        assert rc != 0
        assert "coreml/vocoder.mlpackage" in capsys.readouterr().err

    def test_a_missing_encoder_with_cloning_is_an_error(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """`--with-cloning` was answered with a warning and exit 0 when the
        encoder did not come; the one thing the flag asked for is the one
        thing whose absence must fail the command."""
        rc, _, _ = self._run(
            tmp_path,
            monkeypatch,
            ["download", "loudreader/loudr-1", "--with-cloning"],
            omit=("encoder",),
        )
        assert rc != 0
        assert "ve.safetensors" in capsys.readouterr().err

    def test_a_missing_enrollment_artefact_with_cloning_is_an_error(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The other half of what ``--with-cloning`` asks for. The checkpoint
        came, so the fetch looks finished; the tensors a clone reads are in the
        file that did not come, and nothing later would say so."""
        rc, _, _ = self._run(
            tmp_path,
            monkeypatch,
            ["download", "loudreader/loudr-1", "--with-cloning"],
            omit=("enrollment",),
        )
        assert rc != 0
        assert "loudr-1-enrollment.safetensors" in capsys.readouterr().err

    def test_a_voiceless_fetch_is_an_error(self, tmp_path, monkeypatch, capsys) -> None:
        """The printed next step is ``--voice <name>``; zero voices cannot be
        a success."""
        rc, _, _ = self._run(
            tmp_path,
            monkeypatch,
            ["download", "loudreader/loudr-1"],
            omit=("voices",),
        )
        assert rc != 0
        assert "voices" in capsys.readouterr().err

    def test_a_revision_is_passed_to_the_fetch(self, tmp_path, monkeypatch) -> None:
        rc, client, _ = self._run(
            tmp_path,
            monkeypatch,
            ["download", "loudreader/loudr-1", "--revision", "a1b2c3d"],
        )
        assert rc == 0
        assert client.calls[0]["revision"] == "a1b2c3d"

    def test_a_local_dir_is_passed_and_named_in_the_hint(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        target = tmp_path / "release"
        rc, client, _ = self._run(
            tmp_path,
            monkeypatch,
            ["download", "loudreader/loudr-1", "--local-dir", str(target)],
        )
        assert rc == 0
        assert client.calls[0]["local_dir"] == str(target)
        assert str(target) in capsys.readouterr().err

    def test_the_with_cloning_default_is_off(self) -> None:
        from loudkit.cli import build_parser

        args = build_parser().parse_args(["download", "loudreader/loudr-1"])
        assert args.backend == "torch"
        assert args.with_cloning is False
        assert args.local_dir is None
        assert args.revision is None

    @pytest.mark.parametrize("backend", ["torch", "onnx", "coreml"])
    def test_for_accepts_exactly_the_three_backends(self, backend: str) -> None:
        from loudkit.cli import build_parser

        args = build_parser().parse_args(["download", "loudreader/loudr-1", "--for", backend])
        assert args.backend == backend

    def test_a_language_name_is_not_a_backend(self) -> None:
        """No per-language variants: Rust, Go and JS read the onnx set and
        Swift reads the coreml set. The flag refuses five SDK names."""
        from loudkit.cli import build_parser

        for wrong in ("rust", "go", "js", "swift", "python", "all"):
            with pytest.raises(SystemExit):
                build_parser().parse_args(["download", "loudreader/loudr-1", "--for", wrong])

    def test_voice_is_gone_from_the_grammar_and_the_help(self) -> None:
        """0.1 removes ``download --voice`` rather than shipping a flag whose
        help promises one voice while the command fetches all of them."""
        from loudkit.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["download", "loudreader/loudr-1", "--voice", "kathleen"])
        sub = build_parser()._subparsers._group_actions[0].choices
        help_text = sub["download"].format_help()
        assert "--voice" not in help_text
        for flag in ("--for", "--with-cloning", "--revision", "--local-dir"):
            assert flag in help_text

    # ---- a printed path has to be a path to something -------------------
    # `--with-cloning` used to print `encoder <root>/ve.safetensors` whatever
    # the backend, while the plan fetches that file for torch alone. So an
    # ONNX or CoreML caller was handed the path of a file the download had
    # deliberately left behind.

    def test_torch_names_the_two_files_it_fetched(self, tmp_path, monkeypatch, capsys) -> None:
        rc, _, _ = self._run(
            tmp_path,
            monkeypatch,
            ["download", "loudreader/loudr-1", "--for", "torch", "--with-cloning"],
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "encoder" in out
        assert "ve.safetensors" in out
        # Both files, not just the encoder: torch enrols from the pair.
        assert "enrollment" in out
        assert "loudr-1-enrollment.safetensors" in out

    @pytest.mark.parametrize("backend", ["onnx", "coreml"])
    def test_a_graph_backend_never_names_the_torch_encoder(
        self, tmp_path, monkeypatch, backend, capsys
    ) -> None:
        rc, _, _ = self._run(
            tmp_path,
            monkeypatch,
            ["download", "loudreader/loudr-1", "--for", backend, "--with-cloning"],
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "ve.safetensors" not in out, out
        assert f"{backend} graphs" in out, out

    @pytest.mark.parametrize("backend", ["onnx", "coreml"])
    def test_a_graph_backend_clones_without_the_torch_files(
        self, tmp_path, monkeypatch, backend
    ) -> None:
        """The inventory must ask for what the plan fetched, not more.

        The repo is made to hold neither torch file, which is the point: a
        graph backend never fetches them, so requiring them turned a correct
        download into an error naming what the caller was right not to have.
        Without the omit this test passes against the bug, because the fake
        snapshot holds everything.
        """
        rc, _, _ = self._run(
            tmp_path,
            monkeypatch,
            ["download", "loudreader/loudr-1", "--for", backend, "--with-cloning"],
            omit=("encoder", "enrollment"),
        )
        assert rc == 0


class TestServedCheckpoint:
    """``mcp`` hands its checkpoint over the way the other two transports do.

    A helper here used to resolve a repo id on the transport's behalf, because
    the transport defaulted its voice directory by path arithmetic and a raw
    repo id became ``Path("org/repo").parent / "voices"`` -- no voices at all.
    The transport resolves every shape itself now, so the helper had become a
    second resolution of a path already resolved, and the CLI passes through.
    """

    @pytest.mark.parametrize("ref", ["loudreader/loudr-1", None])
    def test_the_checkpoint_reaches_the_transport_verbatim(
        self, monkeypatch, fake_ckpt, ref
    ) -> None:
        import loudkit.hub as hub_mod
        import loudkit.transports.mcp as mcp_mod
        from loudkit.cli import main

        wanted = ref or str(fake_ckpt)
        monkeypatch.setattr(
            hub_mod,
            "resolve_checkpoint",
            lambda *_a, **_k: pytest.fail("the CLI resolved what the transport resolves"),
        )
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            mcp_mod, "run_stdio", lambda ckpt, *_a, **_k: seen.update(ckpt=ckpt)
        )
        assert main(["serve", "--mcp", "--checkpoint", wanted]) == 0
        assert seen["ckpt"] == wanted

    def test_a_pin_is_still_resolved_here(self, monkeypatch, tmp_path) -> None:
        """No transport takes a revision, so the pin has to be spent in the CLI."""
        import loudkit.hub as hub_mod
        import loudkit.transports.mcp as mcp_mod
        from loudkit.cli import main

        snapshot = tmp_path / "snapshots" / "abc" / "loudr-1.safetensors"
        monkeypatch.setattr(hub_mod, "resolve_checkpoint", lambda *_a, **_k: snapshot)
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            mcp_mod, "run_stdio", lambda ckpt, *_a, **_k: seen.update(ckpt=ckpt)
        )
        main(["serve", "--mcp", "--checkpoint", "loudreader/loudr-1", "--revision", "abc"])
        assert seen["ckpt"] == str(snapshot)


class TestTextPreview:
    @pytest.mark.parametrize(
        "text",
        [
            "Dr. Smith paid $12 on 2024-01-02. [12]",
            "Hello",
            "",
            "[12]",
            "Cześć, 42!",
            pytest.param("hello " * 2000 + "12", id="repeated-words"),
        ],
    )
    def test_preview_uses_the_funnel_without_loading(self, text, capsys, monkeypatch):
        import loudkit
        from loudkit.cli import main
        from loudkit.frontend.speechtext import speech_text

        monkeypatch.setattr(loudkit, "load", lambda *_a, **_k: pytest.fail("model loaded"))
        assert main(["text", text, "--language", "en"]) == 0
        captured = capsys.readouterr()
        assert captured.out == speech_text(text, "en") + "\n"
        if text == "Hello":
            assert captured.err == ""
        if text == "[12]":
            assert "removed '[12]'" in captured.err
        if "Smith" in text:
            assert "replaced" in captured.err
            assert "twelve" in captured.err

    def test_stdin_and_language(self, capsys, monkeypatch):
        import io

        from loudkit.cli import main
        from loudkit.frontend.speechtext import speech_text

        monkeypatch.setattr("sys.stdin", io.StringIO("Mam 12 kotów."))
        assert main(["text", "-", "--language", "PL"]) == 0
        assert capsys.readouterr().out == speech_text("Mam 12 kotów.", "pl") + "\n"

    def test_unknown_language(self, capsys):
        from loudkit.cli import main

        assert main(["text", "hello", "--language", "xx"]) == 1
        assert "unsupported" in capsys.readouterr().err


class TestPlayback:
    @pytest.mark.parametrize(
        ("system", "available", "expected"),
        [
            ("Darwin", {"afplay"}, "afplay"),
            ("Linux", {"paplay"}, "paplay"),
            ("Linux", {"aplay", "paplay"}, "aplay"),
            ("Windows", {"powershell"}, "powershell"),
        ],
    )
    def test_platform_player_receives_path_as_data(
        self, system, available, expected, tmp_path, monkeypatch
    ):
        from loudkit.cli import _play_audio

        path = tmp_path / "quote' $(touch unsafe); audio.wav"
        monkeypatch.setattr("platform.system", lambda: system)
        monkeypatch.setattr(
            "shutil.which", lambda name: f"/bin/{name}" if name in available else None
        )
        calls = []
        monkeypatch.setattr("subprocess.run", lambda argv, **kw: calls.append((argv, kw)))
        _play_audio(path)
        argv, kwargs = calls[0]
        assert argv[0] == f"/bin/{expected}"
        assert kwargs["check"] is True
        assert not kwargs.get("shell")
        if system == "Windows":
            assert str(path) not in argv[-1]
            assert kwargs["env"]["LOUDKIT_PLAY_WAV"] == str(path.resolve())
        else:
            assert argv[1:] == [str(path.resolve())]

    def test_missing_player_is_actionable(self, tmp_path, monkeypatch):
        from loudkit.cli import _play_audio

        monkeypatch.setattr("platform.system", lambda: "Linux")
        monkeypatch.setattr("shutil.which", lambda _name: None)
        with pytest.raises(RuntimeError, match="install alsa-utils or pulseaudio-utils"):
            _play_audio(tmp_path / "out.wav")

    def test_player_failure_names_saved_file(self, tmp_path, monkeypatch):
        import subprocess

        from loudkit.cli import _play_audio

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/afplay")

        def fail(argv, **kwargs):
            raise subprocess.CalledProcessError(1, argv)

        monkeypatch.setattr("subprocess.run", fail)
        with pytest.raises(RuntimeError, match="playback failed with afplay; WAV saved at"):
            _play_audio(tmp_path / "out.wav")

    @pytest.mark.parametrize("play", [False, True])
    def test_speak_saves_before_optional_playback(
        self, play, fake_ckpt, fake_voice_file, tmp_path, capsys, monkeypatch
    ):
        import loudkit
        from loudkit.cli import main

        monkeypatch.setattr(loudkit, "load", _fake_load_engine)
        monkeypatch.setattr(loudkit.VoiceProfile, "load", _fake_load_voice)
        calls = []

        def record(path):
            assert path.is_file()
            calls.append(path)

        monkeypatch.setattr("loudkit.cli._play_audio", record)
        output = tmp_path / "out.wav"
        argv = [
            "speak",
            "--checkpoint",
            str(fake_ckpt),
            "--voice",
            str(fake_voice_file),
            "hello",
            "-o",
            str(output),
        ]
        if play:
            argv.append("--play")
        assert main(argv) == 0
        assert calls == ([output] if play else [])
        assert ("hear it:" in capsys.readouterr().err) is not play


@pytest.mark.parametrize("device", ["onnx", "coreml"])
def test_graph_clone_does_not_require_torch(
    device, recording, fake_ckpt, tmp_path, monkeypatch
):
    import importlib.util

    import loudkit
    from loudkit.cli import main

    def available(name):
        assert name not in ("torch", "torchaudio", "librosa")
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", available)
    monkeypatch.setattr(
        "loudkit.hub.resolve_enrollment_checkpoint",
        lambda *_a, **_k: pytest.fail("torch enrollment preflight"),
    )
    seen = []

    def enroll(*_args, **kwargs):
        seen.append(kwargs["device"])
        return fake_voice()

    monkeypatch.setattr(loudkit, "enroll", enroll)
    assert (
        main(
            [
                "clone",
                str(recording),
                "--checkpoint",
                str(fake_ckpt),
                "--device",
                device,
                "--name",
                "mine",
                "--language",
                "en",
                "-o",
                str(tmp_path / "mine.safetensors"),
            ]
        )
        == 0
    )
    assert seen == [device]
