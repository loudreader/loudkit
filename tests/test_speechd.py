"""The speech-dispatcher module's protocol, as a transcript.

This integration is how loudkit reaches Orca, Firefox and every ``spd-say``
caller on Linux — an accessibility path, where a module that misbehaves does
not produce a stack trace, it produces a screen reader that talks over itself.
It had no tests at all, and four separate protocol defects: ``SPEAK`` answered
``207`` where the spec says ``202``; the events were ``702 BEGIN``/``703 END``
where the codes are ``701``/``702``/``703``; ``BEGIN`` was emitted *after*
playback finished; and ``STOP`` was answered synchronously, which the protocol
forbids, while cancelling nothing that was still being synthesised.

The module is driven here exactly as speech-dispatcher drives it — lines in,
lines out — with the HTTP render and the audio player replaced by fakes. No
server, no sound, no daemon.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import threading
import time
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "integrations"
    / "speech-dispatcher"
    / "loudkit-speechd.py"
)


@pytest.fixture
def speechd():
    """Import the module by path — its filename is not an identifier."""
    spec = importlib.util.spec_from_file_location("loudkit_speechd", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: @dataclass resolves its annotations through
    # sys.modules[cls.__module__], which is not there yet for a module loaded
    # from a path.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


class _FakePlayer:
    """Stands in for ``paplay``: records what it was fed, never makes sound."""

    def __init__(self, block: threading.Event | None = None) -> None:
        self.written = bytearray()
        self.terminated = False
        self.killed = False
        self.reaped = False
        self.closed = False
        self._block = block
        self.stdin = self

    # -- the Popen surface the module uses --
    def write(self, data: bytes) -> None:
        self.written.extend(data)

    def close(self) -> None:
        self.closed = True

    def wait(self, timeout: float | None = None) -> int:
        """``timeout`` because ``_reap`` passes one: a player that ignores the
        signal must not hold the module for longer than a stop is worth."""
        if self._block is not None:
            self._block.wait(timeout=timeout if timeout is not None else 5)
        self.reaped = True
        return 0

    def poll(self) -> int | None:
        return None if not self.terminated else 0

    def terminate(self) -> None:
        self.terminated = True
        if self._block is not None:
            self._block.set()

    def kill(self) -> None:
        self.killed = True
        self.terminate()


def _drive(speechd, module, lines: list[str]) -> list[str]:
    """Feed `lines` through `main`'s loop and collect what it wrote."""
    out = io.StringIO()
    stdin, stdout = sys.stdin, sys.stdout
    sys.stdin, sys.stdout = io.StringIO("\n".join([*lines, ""])), out
    try:
        speechd.run_loop(module)
    finally:
        sys.stdin, sys.stdout = stdin, stdout
    return [line for line in out.getvalue().splitlines() if line]


def test_speak_handshake_uses_the_protocols_codes(speechd, monkeypatch) -> None:
    """202 to receive, then 200 OK SPEAKING — not 207.

    speech-dispatcher dispatches on these numbers. 207 is not a code it expects
    from a module at this point in the exchange.
    """
    module = speechd.Module()
    monkeypatch.setattr(module, "_render", lambda _text, **_kw: b"RIFF" + b"\0" * 40)
    player = _FakePlayer()
    monkeypatch.setattr(speechd.subprocess, "Popen", lambda *_a, **_k: player)

    replies = _drive(speechd, module, ["SPEAK", "hello", ".", "QUIT"])
    assert replies[0] == "202 OK RECEIVE DATA"
    assert "200 OK SPEAKING" in replies


def test_events_are_701_702_and_begin_comes_first(speechd, monkeypatch) -> None:
    """BEGIN before the samples, END after them, in the protocol's numbering.

    Emitting 702 for BEGIN made speech-dispatcher read the start of an
    utterance as its end; sending BEGIN after playback made the event useless
    even with the right number.
    """
    module = speechd.Module()
    order: list[str] = []
    monkeypatch.setattr(module, "_render", lambda _text, **_kw: b"RIFF" + b"\0" * 40)

    class _Recording(_FakePlayer):
        def write(self, data: bytes) -> None:
            order.append("audio")
            super().write(data)

    player = _Recording()
    monkeypatch.setattr(speechd.subprocess, "Popen", lambda *_a, **_k: player)
    monkeypatch.setattr(
        speechd, "_reply", lambda line: order.append(line) if line.startswith("7") else None
    )

    module.speak("hello")
    _wait_until(lambda: "702 END" in order)
    assert order == ["701 BEGIN", "audio", "702 END"], order


def test_stop_is_not_answered_and_reports_703(speechd, monkeypatch) -> None:
    """The protocol forbids a reply to STOP; 703 STOP is the answer."""
    module = speechd.Module()
    monkeypatch.setattr(module, "_render", lambda _text, **_kw: b"RIFF" + b"\0" * 40)
    monkeypatch.setattr(speechd.subprocess, "Popen", lambda *_a, **_k: _FakePlayer())

    replies = _drive(speechd, module, ["STOP", "QUIT"])
    assert replies[0] == "703 STOP", replies
    assert "200 OK" not in replies[:1], "STOP must not be answered"


def test_stop_during_synthesis_prevents_playback(speechd, monkeypatch) -> None:
    """A stop that lands while the HTTP render is in flight must cancel it.

    `stop()` used to kill only the player process, which does not exist yet at
    that point — so the worker returned seconds later and started speaking an
    utterance the user had already moved past. For a screen reader that is the
    worst available behaviour: the wrong thing, late, over the top of whatever
    is being read now.
    """
    module = speechd.Module()
    started = threading.Event()
    release = threading.Event()

    def slow_render(text: str, speed: float = 1.0) -> bytes:
        started.set()
        release.wait(timeout=5)
        return b"RIFF" + b"\0" * 40

    monkeypatch.setattr(module, "_render", slow_render)
    made_players: list[_FakePlayer] = []

    def popen(*_a, **_k):
        player = _FakePlayer()
        made_players.append(player)
        return player

    monkeypatch.setattr(speechd.subprocess, "Popen", popen)

    module.speak("a long sentence")
    assert started.wait(timeout=5), "render never started"
    module.stop()
    release.set()
    # Waited for, not slept through. The worker returns its slot in a `finally`,
    # so taking one is a join; a sleep asserts nothing here, because "no player
    # was made" is equally true of a worker that has not reached that line yet,
    # and the test would pass on a machine slow enough to break it.
    assert module._renders.acquire(timeout=5), "the render worker never finished"

    assert made_players == [], "a cancelled utterance still reached the player"


def test_in_range_rate_is_rendered_by_the_engine(speechd, monkeypatch) -> None:
    """A rate the engine can render travels as the request's ``speed`` field.

    The engine's speed control is pitch-preserving (WSOLA); the header rewrite
    is not. So for the whole of speech-dispatcher's upper range — every rate
    from 0 upward maps into the engine's 0.5–2.0 — the audio must reach the
    player exactly as the server made it, with the rate in the request rather
    than applied after the fact.
    """
    module = speechd.Module()
    header = _wav_header(24_000)
    speeds: list[float] = []

    def render(text: str, speed: float = 1.0) -> bytes:
        speeds.append(speed)
        return header

    monkeypatch.setattr(module, "_render", render)
    player = _FakePlayer()
    monkeypatch.setattr(speechd.subprocess, "Popen", lambda *_a, **_k: player)
    monkeypatch.setattr(speechd, "_reply", lambda _line: None)

    module.set_rate(100)  # the maximum speech-dispatcher sends → 2.0, in range
    module.speak("hello")
    _wait_until(lambda: bool(player.written))

    assert speeds == [2.0]
    assert bytes(player.written) == header, "in-range rate must not touch the header"


def test_below_range_rate_falls_back_to_header_resample(speechd, monkeypatch) -> None:
    """Below the engine's 0.5 floor, the header rewrite still applies.

    Speech Dispatcher's scale reaches down to -100 (a 0.0 multiplier); the
    engine stops at 0.5. Sending such a rate as ``speed`` would earn a 422 and
    silence, and clamping it would render a rate the user did not ask for — so
    the request goes out at 1.0 and the old resample carries the difference,
    pitch shift and all.
    """
    module = speechd.Module()
    header = _wav_header(24_000)
    speeds: list[float] = []

    def render(text: str, speed: float = 1.0) -> bytes:
        speeds.append(speed)
        return header

    monkeypatch.setattr(module, "_render", render)
    player = _FakePlayer()
    monkeypatch.setattr(speechd.subprocess, "Popen", lambda *_a, **_k: player)
    monkeypatch.setattr(speechd, "_reply", lambda _line: None)

    module.set_rate(-75)  # 0.25, below the engine's floor
    module.speak("hello")
    _wait_until(lambda: bool(player.written))

    assert speeds == [1.0], "an out-of-range rate must not reach the server"
    assert _sample_rate(bytes(player.written)) == 6_000


def test_header_resample_still_changes_the_rate(speechd) -> None:
    """The fallback itself: a real speed change via the declared sample rate."""
    header = _wav_header(24_000)
    assert speechd._resample_header(header, 1.0) == header
    assert _sample_rate(speechd._resample_header(header, 0.25)) == 6_000


def test_config_file_is_actually_read(speechd, tmp_path) -> None:
    """The shipped loudkit.conf configured nothing at all.

    It defined LoudkitServer/LoudkitVoice/LoudkitPlayer and the module read
    only environment variables, so a user editing the official configuration
    file got the defaults and no indication why.
    """
    conf = tmp_path / "loudkit.conf"
    conf.write_text(
        "# comment\n"
        'LoudkitServer "http://192.0.2.10:9000"\n'
        'LoudkitVoice "pl_reader"\n'
        'LoudkitPlayer "aplay"\n',
        encoding="utf-8",
    )
    config = speechd.load_config(str(conf))
    assert config.server == "http://192.0.2.10:9000"
    assert config.voice == "pl_reader"
    assert config.player == "aplay"


def test_set_block_is_one_command_not_five(speechd, monkeypatch) -> None:
    """Multi-line SET must be collected, like SPEAK.

    Only SPEAK was, so each line of a SET block was parsed as its own command
    and answered with its own 200 — a settings handshake turning into a burst
    of stray replies the server has to make sense of.
    """
    module = speechd.Module()
    replies = _drive(speechd, module, ["SET", "RATE=50", "PITCH=10", ".", "QUIT"])
    assert replies.count("203 OK SETTINGS RECEIVED") == 1
    assert module._rate == pytest.approx(1.5)


def _wav_header(rate: int) -> bytes:
    import struct

    return (
        b"RIFF"
        + struct.pack("<I", 36)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", 0)
    )


def _sample_rate(wav: bytes) -> int:
    import struct

    return int(struct.unpack_from("<I", wav, 24)[0])


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for the module")


def test_add_voice_lines_become_a_selectable_voice_list(speechd, tmp_path) -> None:
    """`AddVoice` in loudkit.conf was parsed by nobody and reachable by no one.

    The shipped conf carries `AddVoice "en" "MALE1" "kathleen"`.
    `load_config` recognised four keys and dropped this one; `LIST VOICES` fell
    through to the catch-all `200 OK` with no list; and `SET SELF VOICE MALE1`
    got `203 OK SETTINGS RECEIVED` and changed nothing. So the voice was
    whatever `LoudkitVoice` said, permanently, and a screen-reader user had no
    way to discover or change it — while the configuration file advertised
    otherwise.
    """
    conf = tmp_path / "loudkit.conf"
    conf.write_text(
        'LoudkitVoice "kathleen"\n'
        'AddVoice "en" "MALE1" "kathleen"\n'
        'AddVoice "pl" "FEMALE1" "pl_reader2"\n',
        encoding="utf-8",
    )
    module = speechd.Module(speechd.load_config(str(conf)))

    assert module.config.voices == (
        ("en", "MALE1", "kathleen"),
        ("pl", "FEMALE1", "pl_reader2"),
    )
    assert module.voice_list() == [("MALE1", "en", "none"), ("FEMALE1", "pl", "none")]

    # Selecting one changes the profile the server is actually asked for.
    assert module.set_voice("female1"), "the symbolic name must be case-insensitive"
    assert module._voice == "pl_reader2"
    # And an unknown name is refused rather than silently ignored.
    assert not module.set_voice("NOBODY")
    assert module._voice == "pl_reader2"


def test_the_protocol_answers_list_voices_and_set_voice(speechd, tmp_path) -> None:
    """The two commands a client actually uses to pick a voice."""
    conf = tmp_path / "loudkit.conf"
    conf.write_text('AddVoice "en" "MALE1" "kathleen"\n', encoding="utf-8")
    module = speechd.Module(speechd.load_config(str(conf)))

    out = _drive(
        speechd,
        module,
        ["LIST VOICES", "SET SELF VOICE MALE1", "SET SELF VOICE NOPE", "QUIT"],
    )
    joined = "\n".join(out)

    assert "200-MALE1\ten\tnone" in joined, joined
    assert "200 OK VOICE LIST SENT" in joined, joined
    assert "203 OK SETTINGS RECEIVED" in joined, joined
    assert "300 ERR UNKNOWN VOICE NOPE" in joined, joined


def test_a_refused_speak_leaves_the_current_utterance_speaking(speechd, monkeypatch) -> None:
    """The bound refuses the new utterance, not the one being spoken.

    Bumping the generation stale-marks every render in flight and
    ``_kill_player`` silences what is playing, so running either before the
    semaphore is tried answers "try again" to a user who now has nothing
    playing and nothing about to play. On a screen reader that is a sentence
    cut off mid-word by a request that was refused.

    The shape that reaches the branch: one render held in flight, one
    utterance held at the player, both slots taken.
    """
    module = speechd.Module()
    rendering = threading.Event()
    hold_render = threading.Event()
    speaking = threading.Event()
    hold_player = threading.Event()

    def render(text: str, speed: float = 1.0) -> bytes:
        if text == "held in flight":
            rendering.set()
            hold_render.wait(timeout=5)
        return b"RIFF" + b"\0" * 40

    monkeypatch.setattr(module, "_render", render)

    players: list[_FakePlayer] = []

    def popen(*_a, **_k):
        player = _FakePlayer(hold_player)
        players.append(player)
        speaking.set()
        return player

    monkeypatch.setattr(speechd.subprocess, "Popen", popen)

    replies: list[str] = []
    monkeypatch.setattr(speechd, "_reply", replies.append)

    try:
        module.speak("held in flight")
        assert rendering.wait(timeout=5), "the first render never started"
        module.speak("held at the player")
        assert speaking.wait(timeout=5), "the second utterance never reached the player"
        generation = module._generation

        module.speak("one too many")

        assert any("still rendering" in line for line in replies), replies
        assert module._generation == generation, "the refusal bumped the generation"
        assert not players[0].terminated, "the refusal silenced what was being spoken"
    finally:
        hold_render.set()
        hold_player.set()


@pytest.mark.parametrize("command", ["SPEAK", "SET", "AUDIO", "LOGLEVEL"])
def test_no_block_accumulates_past_the_cap(speechd, monkeypatch, command: str) -> None:
    """ "Bounded while it accumulates" was true of ``SPEAK`` alone.

    ``speak`` re-checks the joined text, so a runaway ``SPEAK`` was caught one
    buffer late but caught. ``SET``, ``AUDIO`` and ``LOGLEVEL`` have no
    downstream check at all, and the block loop tested the cap only when it was
    collecting a ``SPEAK``. Measured before this: a 200000-line ``SET`` block
    accumulated all 1.4 MB.
    """
    handed: list[int] = []
    module = speechd.Module()
    monkeypatch.setattr(module, "_render", lambda _text, **_kw: b"RIFF" + b"\0" * 40)
    monkeypatch.setattr(speechd.subprocess, "Popen", lambda *_a, **_k: _FakePlayer())
    monkeypatch.setattr(module, "speak", lambda text: handed.append(len(text)))
    monkeypatch.setattr(
        speechd,
        "_parse_setting",
        lambda lines, _m: handed.append(sum(len(x) + 1 for x in lines)),
    )

    line = "RATE=0"
    over = speechd.MAX_TEXT_CHARS // len(line) * 4
    _drive(speechd, module, [command, *([line] * over), ".", "QUIT"])

    assert handed, "the block never reached its handler"
    # One line's slack: the cap is tested before a line is appended, so the
    # line that crosses it is kept and the next is the first one dropped.
    assert handed[0] <= speechd.MAX_TEXT_CHARS + len(line) + 1, (
        f"{command} accumulated {handed[0]} characters past a cap of {speechd.MAX_TEXT_CHARS}"
    )


def test_a_cancelled_player_is_closed_and_waited_for(speechd, monkeypatch) -> None:
    """``terminate`` signals and returns; it does not reap.

    Without the ``wait`` the child stays a zombie until some later ``Popen``
    happens to reap it, and without closing stdin a player blocked on its
    input never sees the end of it -- so a player that traps SIGTERM to flush
    its device waits forever on a pipe nothing will close.
    """
    playing = threading.Event()
    player = _FakePlayer(block=threading.Event())
    module = speechd.Module()
    monkeypatch.setattr(module, "_render", lambda _text, **_kw: b"RIFF" + b"\0" * 40)

    def popen(*_a, **_k):
        playing.set()
        return player

    monkeypatch.setattr(speechd.subprocess, "Popen", popen)
    monkeypatch.setattr(speechd, "_reply", lambda _line: None)

    module.speak("hello")
    assert playing.wait(timeout=5), "the utterance never reached the player"
    module.stop()

    assert player.terminated, "the player was never signalled"
    assert player.closed, "the player's stdin was left open"
    assert player.reaped, "the player was signalled but never waited for"


def test_reaping_twice_is_not_an_error(speechd) -> None:
    """The natural-end path and ``stop`` can both reach the same player."""
    player = _FakePlayer()

    speechd._reap(player)
    speechd._reap(player)

    assert player.terminated
