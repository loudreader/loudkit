#!/usr/bin/env python3
"""A speech-dispatcher module, so loudkit speaks wherever Linux speaks.

Speech Dispatcher is the layer that sits between applications and speech
synthesisers on Linux. Firefox reads pages through it, Orca reads screens
through it, and any application that calls ``spd-say`` reaches it. Registering
here means loudkit becomes available to all of them at once, which is a wider
audience than a Python API reaches.

It talks the ``ssip`` module protocol over stdin/stdout: the server writes
commands, the module writes numeric responses, and audio goes out through the
module itself.

**It does not synthesise.** It forwards to a running ``loudkit serve``, which
holds the warm engine. A module that loaded a 747 MB checkpoint per utterance
would make every menu item take six seconds.

Four things about this module are load-bearing, in ways that only show up in
a real screen reader:

* **The reply and event codes are the protocol's**, not approximations.
  ``SPEAK`` answers ``202 OK RECEIVE DATA``; the events are ``701 BEGIN``,
  ``702 END`` and ``703 STOP``. Emitting ``702`` for BEGIN makes Speech
  Dispatcher read the start of an utterance as its end and the end as a stop —
  its queue and its idea of what is speaking are wrong from the first word.
* **BEGIN is sent before the first samples**, not after the last. It is the
  event a reader uses to know speech has started.
* **STOP is not answered.** The protocol forbids a reply to it; what it
  requires is a later ``703 STOP`` once the utterance has actually stopped.
* **STOP cancels the synthesis, not just the player.** A stop that only
  reaches the player lands after the HTTP request returns, and the worker
  starts playing an utterance the user has already moved on from. Every
  utterance carries a generation number, and a worker whose generation is
  stale plays nothing.

Rate rides the engine's own ``speed`` control when it can. A rate that maps
into the engine's 0.5–2.0 range is sent as the ``speed`` field of the
synthesis request, so faster speech keeps its pitch (WSOLA, applied
server-side). Only a rate below 0.5 — Speech Dispatcher's scale reaches down
to 0.0, the engine's does not — falls back to rewriting the WAV header's
sample rate, which is a genuine speed change that also shifts pitch, exactly
like ``sox speed``. The fallback is kept rather than clamped away because a
module that accepts ``SET SELF RATE`` and quietly renders a different rate
leaves a user adjusting a slider that lies to them.

Install:

    pip install "loudkit[server]"
    loudkit serve --checkpoint …/loudr-1.safetensors &
    cp integrations/speech-dispatcher/loudkit.conf ~/.config/speech-dispatcher/modules/
    cp integrations/speech-dispatcher/loudkit-speechd.py ~/.local/bin/
    # then add `AddModule "loudkit" "loudkit-speechd.py" "loudkit.conf"` to speechd.conf
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import struct
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# ssip module replies, from the module protocol. The numbers are not
# interchangeable: speech-dispatcher dispatches on them.
RECEIVE_DATA = "202 OK RECEIVE DATA"
MAX_TEXT_CHARS = 10_000
"""The server's own text cap, checked here so a hopeless request never leaves.

Speech Dispatcher hands over whatever a client selected, and a screen reader
told to read a whole document will happily hand over a megabyte. Sent, that is
a megabyte across a socket to earn a 422; refused here it costs nothing and the
message says the actual limit instead of a status code.
"""

MAX_IN_FLIGHT = 2
"""How many renders may be running at once.

Every SPEAK used to start a thread with nothing bounding how many, and a client
that sends SPEAK faster than the engine renders — key-repeat on a screen
reader's "read next line" is exactly that — opened one socket and one thread
per keystroke. Two, not one, because a render already in flight is usually
about to be superseded and there should be room for its replacement to start
before it finishes.

The work is not lost by refusing: only the newest generation is ever played, so
a render started behind a burst was going to be discarded anyway.
"""

MIN_SPEED = 0.5
MAX_SPEED = 2.0
"""The engine's pitch-preserving speed range, from ``loudkit.models.timestretch``.

Duplicated rather than imported: this module runs standalone under
speech-dispatcher and must not require the loudkit package on its path. A rate
inside the range is rendered by the engine (as the request's ``speed`` field);
outside it, the WAV-header fallback below still applies.
"""

SPEAKING = "200 OK SPEAKING"
SETTINGS_RECEIVED = "203 OK SETTINGS RECEIVED"
DONE = "200 OK"
ERR = "300 ERROR"

EVENT_BEGIN = "701 BEGIN"
EVENT_END = "702 END"
EVENT_STOP = "703 STOP"


@dataclass(frozen=True)
class Config:
    """Where to synthesise, in which voice, through which player."""

    server: str = "http://127.0.0.1:8765"
    voice: str = "kathleen"
    player: str = "paplay"
    token: str = ""
    voices: tuple[tuple[str, str, str], ...] = ()
    """``AddVoice`` entries: (language, symbolic name, loudkit profile).

    Without a table, ``LIST VOICES`` answers bare ``200 OK`` and ``SET SELF
    VOICE`` has nothing to select from — the voice is whatever ``LoudkitVoice``
    says, permanently, and a screen-reader user has no way to change it.
    Empty means "no table": the configured voice is the only one.
    """


def load_config(path: str | None) -> Config:
    """Read the ``loudkit.conf`` speech-dispatcher hands us as argv[1].

    The shipped conf defined ``LoudkitServer``/``LoudkitVoice``/``LoudkitPlayer``
    and this module read only the environment, so editing the official
    configuration file changed nothing: the module still talked to the default
    server, in the default voice, through the default player. Environment
    variables still win, because that is how the file documented itself.
    """
    settings = {
        "LoudkitServer": Config.server,
        "LoudkitVoice": Config.voice,
        "LoudkitPlayer": Config.player,
        "LoudkitToken": Config.token,
    }
    voices: list[tuple[str, str, str]] = []
    if path and Path(path).is_file():
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(" ")
            if key in settings:
                settings[key] = value.strip().strip('"')
            elif key == "AddVoice":
                # `AddVoice "en" "MALE1" "en_clarke_holmes"` — language, the symbolic
                # name a client asks for, and the loudkit profile to speak it
                # with. A malformed line is skipped rather than fatal: one bad
                # entry should not stop the module loading.
                parts = shlex.split(value)
                if len(parts) == 3:
                    voices.append((parts[0], parts[1].upper(), parts[2]))

    return Config(
        server=os.environ.get("LOUDKIT_SERVER", settings["LoudkitServer"]),
        voice=os.environ.get("LOUDKIT_VOICE", settings["LoudkitVoice"]),
        player=os.environ.get("LOUDKIT_PLAYER", settings["LoudkitPlayer"]),
        token=os.environ.get("LOUDKIT_TOKEN", settings["LoudkitToken"]),
        voices=tuple(voices),
    )


def _resample_header(wav: bytes, rate: float) -> bytes:
    """Rewrite a WAV header's sample rate, so the player speaks faster.

    The fallback for rates the engine's ``speed`` field cannot render
    (below ``MIN_SPEED``): changing the declared rate is a genuine speed
    change that also shifts pitch, which is what ``sox speed`` does and what
    this module documents. Returned unchanged when the rate is 1.0 or the
    header is not the canonical 44-byte PCM one — guessing at an unexpected
    layout would corrupt the audio.
    """
    if abs(rate - 1.0) < 1e-6 or len(wav) < 44 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        return wav
    if wav[12:16] != b"fmt " or struct.unpack_from("<I", wav, 16)[0] != 16:
        return wav
    sample_rate = struct.unpack_from("<I", wav, 24)[0]
    channels = struct.unpack_from("<H", wav, 22)[0]
    bits = struct.unpack_from("<H", wav, 34)[0]
    new_rate = max(1, int(round(sample_rate * rate)))
    out = bytearray(wav)
    struct.pack_into("<I", out, 24, new_rate)
    struct.pack_into("<I", out, 28, new_rate * channels * bits // 8)  # byte rate
    return bytes(out)


class Module:
    """One utterance at a time, cancellable while it is still being made.

    Speech Dispatcher expects STOP to take effect promptly — a screen reader
    that keeps talking after the user has moved on is worse than one that is
    slow. "Promptly" has to include the synthesis: an utterance spends most of
    its life inside the HTTP request, and a stop that only kills the player
    lets a cancelled sentence start speaking seconds later.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._rate = 1.0
        # The profile currently selected, which starts as the configured one
        # and moves when a client picks a symbolic name from the AddVoice
        # table. Before this the table was dropped at load and the voice could
        # never change.
        self._voice = self.config.voice
        # The generation an utterance belongs to. `stop()` bumps it; a worker
        # whose generation is stale produces no sound and no events.
        self._generation = 0
        self._renders = threading.Semaphore(MAX_IN_FLIGHT)

    # -- speech-dispatcher calls these ------------------------------------

    def speak(self, text: str) -> None:
        if len(text) > MAX_TEXT_CHARS:
            _reply(
                f"{ERR} text is {len(text)} characters; this module and the "
                f"server both cap it at {MAX_TEXT_CHARS}. Split it and send "
                "the pieces."
            )
            return
        with self._lock:
            self._generation += 1
            generation = self._generation
        self._kill_player()
        # Bounded, and non-blocking about it: a SPEAK arriving while two
        # renders are already running is answered rather than queued, because
        # the queue is what turned a key-repeat into a thread per keystroke.
        if not self._renders.acquire(blocking=False):
            _reply(f"{ERR} still rendering the previous utterance; try again")
            return
        threading.Thread(
            target=self._render_and_play, args=(text, generation), daemon=True
        ).start()

    def cancel(self) -> None:
        """Abandon the current utterance without announcing anything.

        Bumping the generation is what makes this work during the HTTP render:
        the request still completes (there is no way to abort it from here
        without a second connection), but its result is discarded and never
        reaches the player.
        """
        with self._lock:
            self._generation += 1
        self._kill_player()

    def stop(self) -> None:
        """Cancel and report it, which is what a STOP command means.

        Separate from :meth:`cancel` so that shutting down does not emit a
        stray ``703 STOP`` for an utterance nobody was listening to — the
        server would take it as an event about speech that never existed.
        """
        self.cancel()
        _reply(EVENT_STOP)

    def set_rate(self, spd_rate: int) -> None:
        """Speech Dispatcher sends -100..100; map it to a playback multiplier."""
        self._rate = 1.0 + (max(-100, min(100, spd_rate)) / 100.0)

    def set_voice(self, symbolic: str) -> bool:
        """Select a voice by the symbolic name an ``AddVoice`` line gave it.

        Returns False for a name the table does not carry, so the caller can
        answer 300 rather than silently keeping the old voice — which is what
        happened to every ``SET SELF VOICE`` before this: a 203 OK and no change.
        """
        wanted = symbolic.strip().upper()
        for _language, name, profile in self.config.voices:
            if name == wanted:
                self._voice = profile
                return True
        return False

    def voice_list(self) -> list[tuple[str, str, str]]:
        """``(name, language, variant)`` triples for ``LIST VOICES``.

        Falls back to the configured voice when the conf declares no table, so
        a client always sees at least the voice it is going to get.
        """
        if self.config.voices:
            return [(name, language, "none") for language, name, _ in self.config.voices]
        # The fallback used to advertise `self.config.voice` under the name the
        # conf gives it, which `set_voice` then refused because it is not in the
        # (empty) table: a client read the list, chose the only entry, and was
        # told no. Advertising nothing is the honest answer to "what tables do
        # you have" when there are none — the configured voice is still what
        # every synthesis uses.
        return []

    # -- internals ---------------------------------------------------------

    def _kill_player(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
            self._proc = None

    def _current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation

    def _render_and_play(self, text: str, generation: int) -> None:
        try:
            self._render_and_play_inner(text, generation)
        finally:
            self._renders.release()

    def _render_and_play_inner(self, text: str, generation: int) -> None:
        # Read once, so the request and the fallback see the same rate even if
        # a SET lands mid-render. In range, the engine renders the rate itself
        # (pitch preserved); below MIN_SPEED — speech-dispatcher's scale goes
        # to 0.0, the engine's stops at 0.5 — the header fallback still works.
        rate = self._rate
        engine_renders_rate = MIN_SPEED <= rate <= MAX_SPEED
        try:
            wav = self._render(text, speed=rate if engine_renders_rate else 1.0)
        except urllib.error.HTTPError as exc:
            # The server answered. "Unreachable" was the message for a 400 about
            # the request, a 404 for a voice that is not there and a 429 for
            # going too fast — three fixable things reported as a network
            # outage, which is the one explanation that suggests doing nothing.
            if self._current(generation):
                detail = ""
                with contextlib.suppress(Exception):
                    detail = exc.read().decode("utf-8", "replace")[:200]
                _reply(
                    f"{ERR} loudkit server refused the request "
                    f"({exc.code}): {detail or exc.reason}"
                )
            return
        except (urllib.error.URLError, OSError) as exc:
            if self._current(generation):
                _reply(f"{ERR} loudkit server unreachable at {self.config.server}: {exc}")
            return
        # The stop may have arrived while the request was in flight. Checked
        # here, before a single sample is played: this is the whole point of
        # the generation number.
        if not self._current(generation):
            return

        if not engine_renders_rate:
            wav = _resample_header(wav, rate)
        try:
            proc = subprocess.Popen(  # noqa: S603 - configured argv, no shell
                [self.config.player], stdin=subprocess.PIPE
            )
        except FileNotFoundError:
            if self._current(generation):
                _reply(f"{ERR} audio player {self.config.player!r} not found")
            return
        with self._lock:
            if generation != self._generation:
                proc.terminate()
                return
            self._proc = proc

        # BEGIN before the first samples: it is the event a reader uses to know
        # speech has started, and it used to be sent after playback finished.
        _reply(EVENT_BEGIN)
        if proc.stdin:
            try:
                proc.stdin.write(wav)
                proc.stdin.close()
                proc.wait()
            except BrokenPipeError:
                pass  # stopped mid-utterance, which is normal
        # A stopped utterance already reported 703 STOP; only a natural
        # ending is an END.
        if self._current(generation):
            _reply(EVENT_END)

    def _render(self, text: str, speed: float = 1.0) -> bytes:
        body = json.dumps(
            {
                "text": text,
                "voice": self._voice,
                "seed": 0,
                "long_form": True,
                "speed": speed,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        req = urllib.request.Request(  # noqa: S310 - configured URL, no user input
            # /v1: the server's API is versioned, /health is not. `server` is
            # the origin, so the version belongs here rather than in the
            # configured value — otherwise every loudkit.conf on every machine
            # has to be edited to follow a version bump.
            f"{self.config.server}/v1/synthesize",
            data=body,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            return bytes(resp.read())


def _reply(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _parse_setting(lines: list[str], module: Module) -> None:
    """Apply the settings in a ``SET`` block. Unknown keys are ignored.

    A module that errors on an unknown SET makes the whole voice unavailable,
    and most of them are cosmetic — but RATE is not, and it used to be ignored
    along with the rest.
    """
    for line in lines:
        key, _, value = line.partition("=")
        if key.strip().upper() == "RATE":
            # A malformed rate keeps the previous speed: the utterance still
            # speaks, which beats refusing the whole SET block over one field.
            with contextlib.suppress(ValueError):
                module.set_rate(int(value.strip()))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    module = Module(load_config(args[0] if args else None))
    _reply("299-loudkit\n299 OK LOADED MODULE")
    return run_loop(module)


def run_loop(module: Module) -> int:  # noqa: PLR0912, PLR0915 - one branch per ssip
    # command; the protocol's shape is the function's shape and splitting it hides that
    """The protocol state machine, over stdin/stdout.

    Split from :func:`main` so a test can drive it with a fake module and a
    string for stdin. This integration is an accessibility path, and "we
    followed the spec" is a claim that has to be checkable without a daemon, a
    sound card and a screen reader.
    """
    # The protocol's multi-line commands (SPEAK, SET, AUDIO, LOGLEVEL) all end
    # with a lone ".", and a leading ".." in the body is an escaped ".". Only
    # SPEAK used to be collected this way, so the lines of a SET block were
    # each parsed as commands and answered individually — which is how a
    # "settings received" handshake turns into five stray 200s.
    block: list[str] = []
    block_chars = 0
    collecting: str | None = None

    for raw in sys.stdin:
        line = raw.rstrip("\r\n")

        if collecting is not None:
            if line == ".":
                # The body is captured before `block` is rebound: a tuple
                # assignment that resets it in the same statement hands the
                # handler an empty list, which is a silent no-op for both
                # SPEAK and SET.
                body, command = block, collecting
                block, collecting = [], None
                block_chars = 0
                if command == "SPEAK":
                    module.speak("\n".join(body))
                    _reply(SPEAKING)
                else:
                    _parse_setting(body, module)
                    _reply(SETTINGS_RECEIVED)
            else:
                # Bounded while it accumulates, not after. `speak` checks the
                # cap on the joined text, which is one buffer too late: a client
                # sending a gigabyte inside one SPEAK block had all of it in
                # memory before anything looked at the length. The line is
                # dropped rather than the connection closed, because SSIP has no
                # way to refuse mid-block — the reply comes when the terminator
                # does, and `speak` says the same sentence about the same cap.
                if collecting == "SPEAK" and block_chars > MAX_TEXT_CHARS:
                    continue
                block.append(line[1:] if line.startswith("..") else line)
                block_chars += len(block[-1]) + 1
            continue

        cmd = line.strip().upper()
        if cmd == "SPEAK":
            collecting = "SPEAK"
            block = []
            block_chars = 0
            _reply(RECEIVE_DATA)
        elif cmd in {"SET", "AUDIO", "LOGLEVEL"}:
            collecting = cmd
            block = []
            _reply(RECEIVE_DATA)
        elif cmd in {"STOP", "CANCEL"}:
            # No reply: the protocol forbids one here. The 703 STOP that
            # `stop()` emits is the answer.
            module.stop()
        elif cmd == "QUIT":
            module.cancel()
            _reply("210 OK QUIT")
            return 0
        elif cmd.startswith("SET SELF RATE"):
            # The single-line form some clients still send.
            with contextlib.suppress(ValueError, IndexError):
                module.set_rate(int(line.split()[-1]))
            _reply(DONE)
        elif cmd in {"LIST VOICES", "LIST SYNTHESIS_VOICES"}:
            # Answered, not swallowed by the `else` below. A client that asks
            # what voices exist got a bare `200 OK` with no list, so the
            # AddVoice lines in loudkit.conf were unreachable from every
            # client — the voice was whatever LoudkitVoice said, permanently.
            voices = module.voice_list()
            body = "\r\n".join(f"200-{n}\t{lang}\t{var}" for n, lang, var in voices)
            _reply(f"{body}\r\n200 OK VOICE LIST SENT" if body else "200 OK VOICE LIST SENT")
        elif cmd.startswith(("SET SELF VOICE", "SET SELF SYNTHESIS_VOICE")):
            name = line.split()[-1]
            if module.set_voice(name):
                # 203, the code for a setting that was accepted — the same one
                # the multi-line SET block answers with.
                _reply(SETTINGS_RECEIVED)
            else:
                # 300 rather than 203: silently keeping the old voice is how a
                # user ends up convinced the setting does nothing.
                _reply(f"300 ERR UNKNOWN VOICE {name}")
        else:
            _reply(DONE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
