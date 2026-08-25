"""The server's two jobs, without weights: route correctness and stream shape.

The load-bearing assertion of the server is that it holds **no synthesis path
of its own** — ``render_bytes`` is the only place audio is made, and the suite
asserts the streaming path returns the same audio per chunk as the whole-passage
path. If those ever diverge, the library's founding promise (one algorithm, one
path) is broken at the HTTP layer.

These use the same deterministic fakes as test_bench, plus FastAPI's
TestClient. The stream assertion checks structure and event order rather than
audio bytes: the fake vocoder emits zeros, which are as good a fingerprint as
anything, and the point is that chunked delivery matches sequential delivery.
"""

from __future__ import annotations

import contextlib
import json
import secrets
from pathlib import Path

import numpy as np
import pytest

from loudkit.config import AlgorithmConfig
from loudkit.contracts import Mel, Sampler, SpeechTokens, Waveform
from loudkit.engine import Engine
from loudkit.errors import (
    UnsupportedLanguageError,
    VoiceNotFoundError,
    WindowOverflowError,
)
from loudkit.models.timestretch import MAX_SPEED, MIN_SPEED
from loudkit.synthesis import VoiceLibrary, render_bytes, render_stream_chunks
from loudkit.transports.http import _MIN_TOKEN_CHARS, build_app
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


class _SplitFrontend:
    """Splits on full stops so streaming has more than one chunk."""

    def encode(self, text: str, language: str = "en") -> np.ndarray:
        words = text.replace(".", " .").split()
        return np.arange(len(words), dtype=np.int64)


class _FakeGenerator:
    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config

    def generate(
        self,
        text_tokens: np.ndarray,
        voice: VoiceProfile,
        *,
        sampler: Sampler,
        max_new_tokens: int | None = None,
        prefix: SpeechTokens = (),
        should_cancel=None,
    ) -> SpeechTokens:
        n = max(1, len(text_tokens))
        return [*range(n), self.config.stop_speech_token]

    def teacher_forced_logits(
        self, text_tokens: np.ndarray, voice: VoiceProfile, forced: SpeechTokens
    ) -> np.ndarray:
        return np.zeros((len(forced) + 1, self.config.speech_vocab_size), np.float32)


class _FakeMelDecoder:
    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config

    def decode(self, tokens: SpeechTokens, voice: VoiceProfile, *, seed: int) -> Mel:
        return np.full((80, max(1, len(tokens)) * 2), float(seed % 97), np.float32)


class _FakeVocoder:
    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config

    def synthesize(self, mel: Mel, voice: VoiceProfile, *, seed: int) -> Waveform:
        return np.zeros(mel.shape[1] * 256, np.float32)


def _stub_checkpoint(tmp_path) -> Path:
    """A checkpoint file that exists but is never read.

    These tests inject an engine, so nothing loads the weights -- but `serve`
    hands its checkpoint argument to the hub whatever shape it has, and the hub
    will not hand back a path to nothing. It sits in a directory of its own so a
    test that also passes ``voices=tmp_path`` does not find it offered as a
    voice.
    """
    holder = tmp_path / "checkpoint"
    holder.mkdir(exist_ok=True)
    path = holder / "ckpt.safetensors"
    path.write_bytes(b"x")
    return path


def _engine() -> Engine:
    from dataclasses import replace

    # max_tokens=2 makes the fake split "one. two. three." into small chunks;
    # prefix_tokens=0 keeps it from tripping the prefix < max_tokens check.
    chunking = replace(AlgorithmConfig().chunking, max_tokens=2, prefix_tokens=0)
    algo = AlgorithmConfig().with_(chunking=chunking)
    return Engine(
        frontend=_SplitFrontend(),
        token_generator=_FakeGenerator(algo),
        mel_decoder=_FakeMelDecoder(algo),
        vocoder=_FakeVocoder(algo),
        algorithm=algo,
    )


def _voices(tmp_path) -> VoiceLibrary:
    """A voice library with one voice named 'fake', written to disk."""
    voice = _voice()
    voice.save(tmp_path / "fake.safetensors")
    return VoiceLibrary(tmp_path)


def _client(tmp_path):
    return TestClient(build_app(_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765")


def test_public_host_refused_without_flag(tmp_path, monkeypatch) -> None:
    """A non-loopback bind must fail fast: no auth on this server."""
    import loudkit
    import loudkit.transports.http as server_mod

    monkeypatch.setattr(loudkit, "load", lambda *_a, **_k: _engine())

    with pytest.raises(SystemExit, match="refusing to bind 0.0.0.0"):
        server_mod.serve(_stub_checkpoint(tmp_path), host="0.0.0.0")


def test_public_host_allowed_with_flag(tmp_path, monkeypatch) -> None:
    """--allow-public is the explicit opt-in for a non-loopback bind."""
    import sys

    import loudkit
    import loudkit.transports.http as server_mod

    monkeypatch.setattr(loudkit, "load", lambda *_a, **_k: _engine())

    started = {}

    def fake_uvicorn(*args, host, port, **kwargs):
        started["host"], started["port"] = host, port

    monkeypatch.setitem(sys.modules, "uvicorn", type("U", (), {"run": fake_uvicorn}))

    server_mod.serve(_stub_checkpoint(tmp_path), host="0.0.0.0", allow_public=True)
    assert started == {"host": "0.0.0.0", "port": 8765}


def test_synthesize_returns_wav(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post("/v1/synthesize", json={"text": "hello", "voice": "fake"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    assert b"RIFF" in resp.content  # WAV magic
    assert "X-Loudkit-Tokens" in resp.headers


def test_concurrent_synthesis_is_serialised(tmp_path) -> None:
    """The engine is single-flight: concurrent requests must not interleave.
    FastAPI runs sync routes in a threadpool, so without the lock N requests
    enter the same Engine at once — a real hazard under --cuda-graphs. Every
    response must still be a complete, valid WAV.

    Uses ``with client:`` so every thread shares one portal, i.e. one event
    loop — that's what a real uvicorn worker gives concurrent requests.
    Without it, each ``client.post`` from a fresh OS thread spins its own
    portal/event loop, and ``_engine_lock`` (an ``anyio.Lock``, bound to the
    loop that first acquires it) hangs forever when a second loop awaits it —
    a TestClient artifact, not a server bug, but one that hides the real
    concurrency behaviour if the portal isn't shared."""
    from concurrent.futures import ThreadPoolExecutor

    with _client(tmp_path) as client, ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(
                client.post,
                "/v1/synthesize",
                json={"text": f"request {i}", "voice": "fake", "seed": 7},
            )
            for i in range(16)
        ]
        for f in futures:
            resp = f.result()
            assert resp.status_code == 200, resp.text
            assert resp.headers["content-type"].startswith("audio/wav")
            assert b"RIFF" in resp.content


class _CountingGenerator(_FakeGenerator):
    """A generator with a real decode loop, so cancellation depth is observable.

    The plain fake returns a token list in one go, which cannot distinguish
    "stopped at the chunk boundary" from "stopped inside the forward pass" —
    and that distinction is the whole feature. This one polls ``should_cancel``
    per step exactly as the torch generator does, and counts the steps it ran.
    """

    def __init__(self, config: AlgorithmConfig) -> None:
        super().__init__(config)
        self.steps = 0

    def generate(
        self,
        text_tokens: np.ndarray,
        voice: VoiceProfile,
        *,
        sampler: Sampler,
        max_new_tokens: int | None = None,
        prefix: SpeechTokens = (),
        should_cancel=None,
    ) -> SpeechTokens:
        out: list[int] = []
        for i in range(max(1, len(text_tokens))):
            if should_cancel is not None and should_cancel():
                return out
            self.steps += 1
            out.append(i)
        out.append(self.config.stop_speech_token)
        return out


def _counting_engine() -> tuple[Engine, _CountingGenerator]:
    """An engine whose chunks are *long*, so mid-chunk cancellation is visible.

    The shared ``_engine()`` uses ``max_tokens=2`` to force many chunks out of a
    short string, which is exactly wrong here: with one token per chunk, every
    cancellation looks like a boundary cancellation and the test cannot fail.
    """
    from dataclasses import replace

    chunking = replace(AlgorithmConfig().chunking, max_tokens=255, prefix_tokens=0)
    algo = AlgorithmConfig().with_(chunking=chunking)
    gen = _CountingGenerator(algo)
    return (
        Engine(
            frontend=_SplitFrontend(),
            token_generator=gen,
            mel_decoder=_FakeMelDecoder(algo),
            vocoder=_FakeVocoder(algo),
            algorithm=algo,
        ),
        gen,
    )


def test_cancel_before_the_stream_starts_runs_no_forward_pass() -> None:
    """A cancel that is already true must cost nothing at all.

    ``render_stream_chunks`` used to accept ``should_cancel`` and never hand it
    to the engine, checking it only after a chunk came back — so even a stream
    that was cancelled before it began rendered one full chunk first. On a
    255-token window that is seconds of GPU time for output nobody receives.
    """
    engine, gen = _counting_engine()
    chunks = list(
        render_stream_chunks(
            engine, "one. two. three.", _voice(), seed=7, should_cancel=lambda: True
        )
    )
    assert chunks == []
    assert gen.steps == 0, f"cancelled stream still ran {gen.steps} decode steps"


def test_cancel_lands_inside_the_chunk_not_at_its_boundary() -> None:
    """Barge-in is token-level: the interrupt is honoured within one step.

    A chunk is up to ~10 s of speech, so a callback consulted only between
    chunks makes an interruption wait for work the listener has already
    rejected. The engine polls per decode step; this pins that it still does,
    end to end through the server's own streaming helper.
    """
    text = "one two three four five six seven eight nine ten eleven twelve."

    # The uncancelled run first, so the budget below is known to be well inside
    # a single chunk rather than accidentally past its end.
    baseline_engine, baseline = _counting_engine()
    baseline_chunks = list(render_stream_chunks(baseline_engine, text, _voice(), seed=7))
    assert len(baseline_chunks) == 1, "text should be one chunk at max_tokens=255"
    assert baseline.steps > 4, "one chunk should carry several decode steps"

    engine, gen = _counting_engine()
    budget = 3

    list(
        render_stream_chunks(
            engine, text, _voice(), seed=7, should_cancel=lambda: gen.steps >= budget
        )
    )
    # Stopped inside the chunk: a boundary-only check would have run every step
    # the baseline ran before noticing.
    assert gen.steps == budget, f"ran {gen.steps} steps for a budget of {budget}"
    assert gen.steps < baseline.steps


class _CappedGenerator(_FakeGenerator):
    """Runs to the token cap without ever emitting a stop token.

    That is what a broken EOS path looks like from the engine's side, and it is
    the only way to produce ``hit_token_cap`` deterministically without weights.
    """

    def generate(
        self,
        text_tokens: np.ndarray,
        voice: VoiceProfile,
        *,
        sampler: Sampler,
        max_new_tokens: int | None = None,
        prefix: SpeechTokens = (),
        should_cancel=None,
    ) -> SpeechTokens:
        cap = max_new_tokens or self.config.sampling.max_new_tokens
        return list(range(cap))


def _capped_engine() -> Engine:
    from dataclasses import replace

    algo = AlgorithmConfig()
    algo = algo.with_(
        sampling=replace(algo.sampling, max_new_tokens=4),
        chunking=replace(algo.chunking, max_tokens=255, prefix_tokens=0),
    )
    return Engine(
        frontend=_SplitFrontend(),
        token_generator=_CappedGenerator(algo),
        mel_decoder=_FakeMelDecoder(algo),
        vocoder=_FakeVocoder(algo),
        algorithm=algo,
    )


def test_truncation_is_reported_over_http(tmp_path) -> None:
    """A capped utterance is a 200 with cut-off audio; the client must be told.

    The engine has always computed ``hit_token_cap`` and the CLI has always
    warned about it. HTTP returned the truncated WAV with no indication at all,
    so a caller could not distinguish a finished sentence from a severed one.
    """
    client = TestClient(
        build_app(_capped_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
    )
    resp = client.post("/v1/synthesize", json={"text": "one two three", "voice": "fake"})
    assert resp.status_code == 200
    assert resp.headers["X-Loudkit-Truncated"] == "true"

    # And the negative case, so the header is not simply always "true".
    ok = TestClient(build_app(_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765")
    resp = ok.post("/v1/synthesize", json={"text": "one. two.", "voice": "fake"})
    assert resp.headers["X-Loudkit-Truncated"] == "false"


def test_truncation_is_reported_in_the_stream(tmp_path) -> None:
    """Both per chunk and in the terminal event.

    A client that only reads ``done`` must still learn that some chunk was cut
    off, which is why the aggregate is ORed across chunks exactly as
    ``Engine.synthesize_long`` does it.
    """
    client = TestClient(
        build_app(_capped_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
    )
    with client.stream(
        "POST", "/v1/synthesize/stream", json={"text": "one two three", "voice": "fake"}
    ) as resp:
        events = [
            json.loads(line[len("data: ") :])
            for line in resp.iter_lines()
            if line.startswith("data: ")
        ]

    chunks = [e for e in events if not e.get("done")]
    assert chunks
    assert all(e["truncated"] for e in chunks)
    assert events[-1] == {
        "done": True,
        "fingerprint": events[-1]["fingerprint"],
        "truncated": True,
        # The tail to chain from, always present so a client can read it
        # unconditionally. Empty here only if the recipe carries no prefix.
        "continuation": events[-1]["continuation"],
    }


def test_synthesize_rejects_unknown_voice(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post("/v1/synthesize", json={"text": "hello", "voice": "nope"})
    assert resp.status_code == 404
    # The condition's name from the frozen catalog rides in the body: a status
    # is one of five numbers, the code says *which* refusal this was.
    assert resp.json()["code"] == "voice_not_found"


def test_synthesize_rejects_oversized_text(tmp_path) -> None:
    """A text field with no upper bound is a memory DoS: chunking + generation
    scale with input length, and a localhost-only server is still reachable by
    an MCP agent. Cap the request, not the engine."""
    client = _client(tmp_path)
    resp = client.post(
        "/v1/synthesize",
        json={"text": "a" * 10_001, "voice": "fake"},
    )
    assert resp.status_code == 422


def test_stream_yields_one_event_per_chunk_then_done(tmp_path) -> None:
    client = _client(tmp_path)
    with client.stream(
        "POST", "/v1/synthesize/stream", json={"text": "one. two. three.", "voice": "fake"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = [line[6:] for line in resp.iter_lines() if line.startswith("data: ")]

    chunks = [json.loads(e) for e in events]
    assert chunks[-1]["done"] is True
    audio_events = chunks[:-1]
    assert len(audio_events) >= 2, "multi-sentence text must produce more than one chunk"
    for ev in audio_events:
        assert "audio" in ev, "each chunk carries base64 WAV bytes"
        assert ev["audio"]
        assert ev["duration"] > 0
        assert ev["tokens"] > 0
        assert b"RIFF" in __import__("base64").b64decode(ev["audio"])


def test_stream_same_audio_as_whole_passage(tmp_path) -> None:
    """The streaming path must deliver exactly what /v1/synthesize would produce.

    This is the server's version of the one-path rule: streaming is delivery,
    not a second synthesis.
    """
    engine = _engine()
    voice = _voice()
    text = "one. two. three."

    whole = render_bytes(engine, text, voice, seed=7).data
    chunks = [r.data for r in render_stream_chunks(engine, text, voice, seed=7)]

    # WAV files have headers; concatenating raw bodies would break them. What is
    # compared is the decoded PCM, not the containers.
    import io

    import soundfile as sf

    whole_pcm = sf.read(io.BytesIO(whole), dtype="float32")[0]
    chunked_pcm = np.concatenate([sf.read(io.BytesIO(c), dtype="float32")[0] for c in chunks])
    assert len(chunked_pcm) == len(whole_pcm)
    assert np.array_equal(chunked_pcm, whole_pcm)


def test_health_reports_fingerprint(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "fingerprint" in body
    assert "voices" in body
    assert "fake" in body["voices"]


def test_voice_library_rejects_path_traversal(tmp_path) -> None:
    """A voice name is a name, not a path: separators and dot-prefixes must not
    resolve outside the library directory. A request naming a filesystem path
    would let anyone who can reach the port read any ``.safetensors`` on the
    machine — this is the guard that keeps a name a name."""
    lib = _voices(tmp_path)
    # A real file outside the library, to prove a traversal would have hit it.
    outside = tmp_path.parent / "secret.safetensors"
    _voice().save(outside)

    for bad in (
        "../secret",
        "..%2fsecret",
        "sub/voice",
        r"sub\voice",
        ".hidden",
        "/etc/passwd",
    ):
        with pytest.raises(ValueError, match="not a voice name"):
            lib.load(bad)

    # A plain name still loads.
    assert lib.load("fake") is not None


def test_voice_library_rejects_symlink_out_of_the_library(tmp_path) -> None:
    """A bare name cannot escape the directory, but a symlink sitting in it can.

    ``glob`` follows links, so a link planted in the voices directory — by a
    careless deploy, an unpacked archive, or anyone with write access there —
    would otherwise hand any ``.safetensors`` on the host to an unauthenticated
    caller, through a name that passes every character check. The traversal test
    above guards the name; this one guards the file the name resolves to."""
    lib = _voices(tmp_path)
    outside = tmp_path.parent / "secret.safetensors"
    _voice().save(outside)
    try:
        (tmp_path / "leak.safetensors").symlink_to(outside)
    except OSError as exc:  # pragma: no cover - platform-dependent
        # Windows gates symlink creation behind Developer Mode or an elevated
        # process, and a developer without either would otherwise see this fail
        # as though the library had stopped refusing the link. Skipping names
        # the reason; the guard it covers is platform-independent and CI's
        # Linux and macOS runners still exercise it.
        pytest.skip(f"cannot create a symlink on this machine: {exc}")

    # The name is spotless, so it survives the character checks and is refused
    # only on where it lands.
    with pytest.raises(VoiceNotFoundError):
        lib.load("leak")
    # And it is not advertised either: a listing that names it invites the call.
    assert "leak" not in lib.names()
    assert lib.load("fake") is not None


class _SlowEngine:
    """Engine whose stream yields chunks with a real delay, so a streaming
    request actually holds the lock for a while."""

    def __init__(self) -> None:
        import time
        from dataclasses import replace

        self._t = time
        algo = AlgorithmConfig().with_(
            chunking=replace(AlgorithmConfig().chunking, max_tokens=2, prefix_tokens=0)
        )
        self.algorithm = algo
        self.execution = type("_Exec", (), {"describe": staticmethod(lambda: "test")})()
        self.backend = "fake"
        self.checkpoint_sha256 = ""
        self.slices = 0
        self.languages: list[str | None] = []
        """Every `language` this engine was handed, verbatim.

        Recorded rather than ignored because the server's whole share of the
        language chain is *not resolving it*: `SpeakRequest.language` defaults
        to None and hands None on, so the engine can consult the voice. A fake
        that discards the argument cannot tell that apart from a server that
        hardcodes "en", and for a while nothing could — reverting the default
        left every server test green.
        """

    SLICES_PER_CHUNK = 10
    CHUNKS = 3

    def stream(
        self,
        text,
        voice,
        *,
        seed=0,
        language=None,
        speed=1.0,
        previous_tokens=None,
        should_cancel=None,
    ):
        self.languages.append(language)
        # three chunks, 100 ms each — enough for a concurrent request to try.
        # `should_cancel` is polled per slice rather than per chunk, mirroring
        # the real engine's per-decode-step poll, so a disconnect test here
        # measures the same latency a client would see. `slices` counts the
        # work actually done, which is what a cancellation test must assert on:
        # a stream that stops delivering but keeps computing has not cancelled.
        for _ in range(self.CHUNKS):
            for _ in range(self.SLICES_PER_CHUNK):
                if should_cancel is not None and should_cancel():
                    return
                self.slices += 1
                self._t.sleep(0.01)
            yield _FakeResult()

    def synthesize(
        self, text, voice, *, seed=0, language=None, speed=1.0, previous_tokens=None
    ):
        self.languages.append(language)
        return _FakeResult()

    def synthesize_long(
        self, text, voice, *, seed=0, language=None, speed=1.0, previous_tokens=None
    ):
        self.languages.append(language)
        return _FakeResult()


class _FakeResult:
    @property
    def audio(self):
        import numpy as np

        return np.zeros(1000, dtype=np.float32)

    @property
    def sample_rate(self):
        return 24_000

    @property
    def duration(self):
        return 0.04

    @property
    def tokens(self):
        return [1, 2]

    @property
    def mel(self):
        import numpy as np

        return np.zeros((80, 10), dtype=np.float32)

    @property
    def timings(self):
        return None

    @property
    def seed(self):
        return 7

    @property
    def hit_token_cap(self):
        return False

    @property
    def speed(self):
        return 1.0

    @property
    def chunks(self):
        return []

    @property
    def algorithm_fingerprint(self):
        return "test"

    @property
    def describe(self):
        return "test"


def test_stream_does_not_freeze_health(tmp_path) -> None:
    """A streaming request that holds the single-flight lock must not freeze the
    event loop: /health must still answer while a stream is mid-flight.

    This is the regression test for the deadlock N1: the async stream used to
    take a blocking threading.Lock on the event loop, so a second request
    waiting on the lock would freeze the loop and starve the first stream of
    its chance to release it."""
    from concurrent.futures import ThreadPoolExecutor

    engine = _SlowEngine()
    voices = _voices(tmp_path)
    client = TestClient(build_app(engine, voices), base_url="http://127.0.0.1:8765")  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=2) as pool:
        stream_fut = pool.submit(
            lambda: client.stream(
                "POST",
                "/v1/synthesize/stream",
                json={"text": "one. two. three.", "voice": "fake"},
            ).__enter__()
        )
        import time

        time.sleep(0.05)  # let the stream start and take the lock
        t0 = time.time()
        resp = client.get("/health")
        elapsed = time.time() - t0
        assert resp.status_code == 200, resp.text
        assert elapsed < 1.0, f"/health blocked by in-flight stream: {elapsed:.2f}s"
        stream_fut.result().close()


def test_disconnect_stops_the_render_it_does_not_just_stop_sending(tmp_path) -> None:
    """A closed connection must stop the GPU, not just the socket.

    The disconnect check used to run only between chunks, so a client that
    vanished during a 255-token window paid for the whole window anyway. A
    watcher now polls ``is_disconnected()`` on the event loop *while* the
    forward pass runs in a worker thread, and flips the flag the decode loop
    reads. Asserted on work done (`slices`), not on events delivered: a stream
    that stops sending while still computing has not cancelled anything.

    Driven as raw ASGI rather than through ``TestClient``: the test client
    never emits ``http.disconnect`` for an abandoned response, so a test written
    against it would pass whether or not the watcher exists. Here the disconnect
    is delivered explicitly, which is the event the server actually reacts to.
    """
    import anyio

    engine = _SlowEngine()
    app = build_app(engine, _voices(tmp_path))  # type: ignore[arg-type]
    total = _SlowEngine.CHUNKS * _SlowEngine.SLICES_PER_CHUNK
    body = json.dumps({"text": "one. two. three.", "voice": "fake"}).encode()

    async def drive() -> None:
        sent: list[dict] = []
        disconnect_after = anyio.Event()

        async def receive() -> dict:
            if not sent:
                sent.append({})
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnect_after.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            # Hang up once the first chunk's event has actually been delivered,
            # so the disconnect lands while the *second* chunk is mid-render in
            # its worker thread. Hanging up at response.start would cancel
            # before any work began and prove nothing about mid-chunk latency.
            if message["type"] == "http.response.body" and message.get("body"):
                disconnect_after.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.1"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/v1/synthesize/stream",
            "raw_path": b"/v1/synthesize/stream",
            "query_string": b"",
            "root_path": "",
            "scheme": "http",
            "headers": [
                (b"host", b"127.0.0.1:8765"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("testclient", 50000),
            "server": ("127.0.0.1", 8765),
        }
        with anyio.move_on_after(5):
            await app(scope, receive, send)

    anyio.run(drive)

    per_chunk = _SlowEngine.SLICES_PER_CHUNK
    assert engine.slices >= per_chunk, "the first chunk should have been delivered"
    assert engine.slices < total, (
        f"engine ran all {total} slices after the client left — the disconnect "
        "stopped delivery but not synthesis"
    )
    # Sharper than "stopped eventually": it stopped *inside* the second chunk,
    # which is only possible if the flag is observed during a forward pass
    # rather than at the boundary after it.
    assert engine.slices < 2 * per_chunk, (
        f"ran {engine.slices} slices — cancellation waited for a chunk boundary"
    )


def _stream_scope(body: bytes) -> dict:
    """The ASGI scope for one POST to the streaming route.

    Shared by the tests that drive the app directly. They have to: every
    failure they cover is about the transport dying at a particular instant,
    and ``TestClient`` never delivers ``http.disconnect`` for an abandoned
    response nor lets a send fail, so a test written against it would pass
    whether or not the server handles either.
    """
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.1"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/synthesize/stream",
        "raw_path": b"/v1/synthesize/stream",
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "headers": [
            (b"host", b"127.0.0.1:8765"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("testclient", 50000),
        "server": ("127.0.0.1", 8765),
    }


class _CountedStreamEngine(_SlowEngine):
    """Reports whether two callers were ever inside `stream()` at once.

    The engine is single-flight because it is not reentrant, so "the slot came
    back" and "the engine is free" are different claims and only the second one
    matters. This fake makes the difference observable: it counts open
    `stream()` frames and remembers if a second one ever opened while the first
    was still there, which is what the slot exists to prevent.
    """

    def __init__(self) -> None:
        super().__init__()
        self.open_streams = 0
        self.overlapped = False

    def stream(self, text, voice, **kwargs):  # type: ignore[no-untyped-def]
        self.open_streams += 1
        self.overlapped = self.overlapped or self.open_streams > 1
        try:
            yield from super().stream(text, voice, **kwargs)
        finally:
            # Runs when the generator is closed, which is the only signal the
            # engine gets that an abandoned stream is over.
            self.open_streams -= 1


def test_a_disconnect_gives_the_slot_back_only_once_the_engine_is_free(tmp_path) -> None:
    """A client that hangs up mid-stream must not hand a live engine away.

    The slot used to be released by the body generator's `finally`, which runs
    the moment that generator is torn down — while the engine's own producer
    and render threads are still inside the stages, because nothing had told
    them to stop and nothing had closed the iterator they live in. The next
    request then took the slot and walked into a non-reentrant engine
    alongside them, and neither side was wrong about holding it.

    Asserted on the engine, not on the slot: `open_streams` is what the next
    caller actually collides with, and a slot that is free while a stream is
    still open is the defect however tidy the bookkeeping looks.
    """
    import anyio

    engine = _CountedStreamEngine()
    app = build_app(engine, _voices(tmp_path))  # type: ignore[arg-type]
    body = json.dumps({"text": "one. two. three.", "voice": "fake"}).encode()
    open_when_next_began: list[int] = []

    async def drive() -> None:
        async def abandon() -> None:
            """Take the first chunk, then vanish."""
            sent: list[dict] = []
            gone = anyio.Event()

            async def receive() -> dict:
                if not sent:
                    sent.append({})
                    return {"type": "http.request", "body": body, "more_body": False}
                await gone.wait()
                return {"type": "http.disconnect"}

            async def send(message: dict) -> None:
                # After the first chunk is on the wire, not at response.start:
                # the failure is about a stream torn down between chunks, where
                # the engine is mid-passage rather than not yet started.
                if message["type"] == "http.response.body" and message.get("body"):
                    gone.set()

            with anyio.move_on_after(10):
                await app(_stream_scope(body), receive, send)

        async def follow() -> None:
            """The request that inherits the slot, drained to the end."""
            sent: list[dict] = []
            never = anyio.Event()

            async def receive() -> dict:
                if not sent:
                    sent.append({})
                    return {"type": "http.request", "body": body, "more_body": False}
                await never.wait()
                return {"type": "http.disconnect"}

            async def send(message: dict) -> None:
                if message["type"] == "http.response.start":
                    open_when_next_began.append(engine.open_streams)

            with anyio.move_on_after(10):
                await app(_stream_scope(body), receive, send)

        await abandon()
        await follow()

    anyio.run(drive)

    assert engine.open_streams == 0, "a stream was left open after both requests"
    assert open_when_next_began == [0], (
        f"the next request started with {open_when_next_began} stream(s) still "
        "open — the slot came back before the engine did"
    )
    assert not engine.overlapped, "two callers were inside the single-flight engine at once"


def test_a_socket_that_dies_at_response_start_does_not_leak_the_slot(
    tmp_path, monkeypatch
) -> None:
    """The one path where the body generator never runs a line.

    Starlette sends `http.response.start` before it pulls the first item, so a
    connection that dies on that send leaves the generator unstarted: its
    `finally` never runs, and with the release living there the slot was held
    by nobody for the rest of the process. Every later synthesis answered 503,
    and nothing in the log said why.

    `_MAX_QUEUED` is 1 so the leak is visible immediately rather than after the
    full engine wait: with the slot lost, the queue never empties and the very
    next request is over the bound.
    """
    import anyio

    import loudkit.transports.http as server_mod

    monkeypatch.setattr(server_mod, "_MAX_QUEUED", 1)
    engine = _CountedStreamEngine()
    app = build_app(engine, _voices(tmp_path))  # type: ignore[arg-type]
    body = json.dumps({"text": "one. two. three.", "voice": "fake"}).encode()
    statuses: list[int] = []

    def a_receive():  # type: ignore[no-untyped-def]
        """One request's `receive`. A fresh one per request: the body is
        delivered exactly once, and a reused closure would starve the second
        request of the body it is waiting for."""
        sent: list[dict] = []
        never = anyio.Event()

        async def receive() -> dict:
            if not sent:
                sent.append({})
                return {"type": "http.request", "body": body, "more_body": False}
            await never.wait()
            return {"type": "http.disconnect"}

        return receive

    async def drive() -> None:
        async def dead_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                raise OSError("connection reset by peer")

        with anyio.move_on_after(10), contextlib.suppress(Exception):
            await app(_stream_scope(body), a_receive(), dead_send)

        async def send(message: dict) -> None:
            if message["type"] == "http.response.start":
                statuses.append(message["status"])

        with anyio.move_on_after(10):
            await app(_stream_scope(body), a_receive(), send)

    anyio.run(drive)

    assert statuses == [200], (
        f"the request after a dead socket answered {statuses} — the slot the "
        "first one took was never given back"
    )


def test_stream_and_sync_synthesis_do_not_deadlock(tmp_path) -> None:
    """Regression for N2: a stream in flight must not be starved of a worker
    thread by concurrent /v1/synthesize requests.

    Both routes used to serialise on separate locks: /v1/synthesize took a
    threading.Lock *inside* the threadpool, the stream took an anyio.Lock on
    the loop but pulled each chunk via anyio.to_thread.run_sync — the same
    worker pool. Enough concurrent /v1/synthesize requests fill that pool with
    threads parked on the threading.Lock, and the in-flight stream then has no
    worker left to advance its next chunk: the whole server hangs. Both routes
    must now wait on one lock taken on the loop, so a waiting request never
    occupies a worker it can't make progress on."""
    from concurrent.futures import ThreadPoolExecutor

    engine = _SlowEngine()
    voices = _voices(tmp_path)
    # `with client:` shares one portal (one event loop) across every thread
    # below — what a real uvicorn worker gives concurrent requests. Without
    # it, each call spins its own portal/loop and `_engine_lock` (bound to
    # whichever loop first acquires it) hangs when awaited from another.
    client = TestClient(build_app(engine, voices), base_url="http://127.0.0.1:8765")  # type: ignore[arg-type]

    # More concurrent /v1/synthesize calls than AnyIO's default worker limiter
    # (40), to actually exhaust the pool the way the deadlock needed.
    n_sync = 48

    with client, ThreadPoolExecutor(max_workers=n_sync + 1) as pool:
        stream_fut = pool.submit(
            lambda: client.stream(
                "POST",
                "/v1/synthesize/stream",
                json={"text": "one. two. three.", "voice": "fake"},
            ).__enter__()
        )
        import time

        time.sleep(0.05)  # let the stream start and take the lock

        sync_futs = [
            pool.submit(client.post, "/v1/synthesize", json={"text": "hi", "voice": "fake"})
            for _ in range(n_sync)
        ]

        # Generous on purpose. The failure this guards against is a deadlock,
        # which is infinite — so a long ceiling costs nothing and a short one
        # only measures how contended the machine is. At 5 s this failed about
        # one run in three on an M1 with 49 threads live, which is a flake
        # reporting scheduling noise, not starvation.
        stream_ctx = stream_fut.result(timeout=60.0)
        stream_ctx.close()

        # 200 or 503, but never a hang: the queue in front of the engine is
        # bounded, so past `_MAX_QUEUED` waiters the honest answer is "busy".
        # What this test is about is that every request gets *an* answer.
        codes = [f.result(timeout=60.0).status_code for f in sync_futs]
        assert set(codes) <= {200, 503}, codes
        assert 200 in codes, "every request was shed; the engine never ran"


def test_oversized_body_is_refused_before_it_is_read(tmp_path) -> None:
    """The text cap runs after Starlette has buffered the whole body.

    Pydantic's `max_length` protected the engine, not the process: a 500 MB
    POST was read into memory in full and then refused for being 10 000
    characters too long. The bound now applies to the body itself, from the
    Content-Length header, before anything is read.
    """
    from loudkit.transports.http import _MAX_BODY_BYTES

    client = _client(tmp_path)
    resp = client.post(
        "/v1/synthesize",
        json={"text": "a" * (_MAX_BODY_BYTES * 2), "voice": "fake"},
    )
    assert resp.status_code == 413
    assert "body exceeds" in resp.text


def test_oversized_chunked_body_is_also_refused(tmp_path) -> None:
    """A chunked request declares no Content-Length, so the header check misses it.

    The only place to stop it is mid-stream, and the only thing ASGI offers
    there is to tell the app the client hung up. Starlette turns that into a
    ClientDisconnect and FastAPI answers with a **500** — so a caller who sent
    200 MB was told the server had failed, and the middleware's own 413 never
    got out because a response had already started. The app's output is now
    swallowed once the limit is passed, because everything it says after that
    point is a reaction to a disconnect this middleware invented.
    """
    from loudkit.transports.http import _MAX_BODY_BYTES

    client = _client(tmp_path)
    payload = json.dumps({"text": "a" * (_MAX_BODY_BYTES * 2), "voice": "fake"}).encode()

    def chunks():
        # An iterable body makes httpx use Transfer-Encoding: chunked, with no
        # Content-Length for the cheap check to catch.
        for i in range(0, len(payload), 8192):
            yield payload[i : i + 8192]

    resp = client.post(
        "/v1/synthesize", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 413, f"got {resp.status_code}: {resp.text[:200]}"
    assert "body exceeds" in resp.text


def test_a_full_queue_answers_busy_rather_than_growing(tmp_path, monkeypatch) -> None:
    """An unbounded queue turns a slow engine into unbounded memory.

    Every waiter still holds a connection, so the failure is not just latency:
    it is memory and descriptors climbing with no ceiling. Past the bound the
    server says 503 with Retry-After, which is true and actionable.

    The bound is lowered for the test so this stays a few hundred milliseconds
    rather than a minute of real serialised synthesis.
    """
    from concurrent.futures import ThreadPoolExecutor

    import loudkit.transports.http as server_mod

    monkeypatch.setattr(server_mod, "_MAX_QUEUED", 2)

    engine = _SlowEngine()
    client = TestClient(build_app(engine, _voices(tmp_path)), base_url="http://127.0.0.1:8765")  # type: ignore[arg-type]
    n = 8

    # `with client` matters: outside its context TestClient spins a fresh
    # event loop per request, and the app's anyio.Lock belongs to the loop it
    # was created in — the requests then wait on a lock nobody can release.
    with client, ThreadPoolExecutor(max_workers=n) as pool:
        futures = [
            pool.submit(client.post, "/v1/synthesize", json={"text": "one.", "voice": "fake"})
            for _ in range(n)
        ]
        codes = [f.result(timeout=60.0).status_code for f in futures]

    assert 503 in codes, f"queue never shed anything: {sorted(set(codes))}"
    assert 200 in codes, "everything was shed; the engine never ran"


def test_a_token_locks_the_server(tmp_path) -> None:
    """`--allow-public` used to expose every voice on the machine with no auth.

    A bearer token is the smallest thing that makes a non-loopback bind
    defensible; `serve` requires one for any non-loopback host.
    """
    app = build_app(_engine(), _voices(tmp_path), token="s3cret-real-token")
    client = TestClient(app, base_url="http://127.0.0.1:8765")

    assert client.get("/health").status_code == 401
    speak = client.post("/v1/synthesize", json={"text": "hi", "voice": "fake"})
    assert speak.status_code == 401

    ok = client.get("/health", headers={"Authorization": "Bearer s3cret-real-token"})
    assert ok.status_code == 200
    assert client.get("/health", headers={"Authorization": "Bearer wrong!"}).status_code == 401


def test_a_public_bind_rate_limits_synthesis_but_never_health(tmp_path) -> None:
    """Holding the token is not a licence to keep the engine busy forever.

    The queue bound and the wait deadline shape what the server does under load,
    but neither costs the caller anything: a client looping on `/v1/synthesize`
    takes every slot as it frees. The bucket puts a price on the call.

    `/health` stays outside it on purpose — it is what a load balancer polls, and
    limiting it would take an instance out of rotation for being busy, which is
    the moment its health matters most.
    """
    from loudkit.transports.http import _RATE_CAPACITY

    app = build_app(_engine(), _voices(tmp_path), token="s3cret-real-token")
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    auth = {"Authorization": "Bearer s3cret-real-token"}

    # The bucket starts full, so the first `_RATE_CAPACITY` calls are admitted —
    # 404 because the voice name is nonsense, which is the router answering and
    # therefore proof the guard let it through.
    for _ in range(_RATE_CAPACITY):
        r = client.post("/v1/synthesize", json={"text": "hi", "voice": "nope"}, headers=auth)
        assert r.status_code != 429

    drained = client.post("/v1/synthesize", json={"text": "hi", "voice": "nope"}, headers=auth)
    assert drained.status_code == 429

    # Health is unaffected, with the bucket empty.
    assert client.get("/health", headers=auth).status_code == 200


def test_loopback_is_not_rate_limited(tmp_path) -> None:
    """No token means loopback, and a limiter there only locks the operator out.

    The caller is already on the machine: anything the bucket would stop, they
    could do by calling the library directly.
    """
    client = TestClient(
        build_app(_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
    )
    for _ in range(40):
        r = client.post("/v1/synthesize", json={"text": "hi", "voice": "nope"})
        assert r.status_code == 404


def test_public_bind_generates_a_token_rather_than_running_open(
    tmp_path, monkeypatch, capsys
) -> None:
    """`--allow-public` used to mean "no auth, on a network you control".

    "A network you control" is a hope, not a control, and an authless public
    bind is the one outcome that must not be reachable by forgetting a flag.
    The token is generated and printed when the operator does not supply one.
    """
    import loudkit
    import loudkit.transports.http as server_mod

    monkeypatch.setattr(loudkit, "load", lambda *_a, **_k: _engine())

    captured = {}
    monkeypatch.setattr(
        server_mod,
        "uvicorn",
        None,
        raising=False,
    )

    def fake_run(app, **kwargs):
        captured["app"] = app

    fake_uvicorn = type("_U", (), {"run": staticmethod(fake_run)})
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", fake_uvicorn)

    server_mod.serve(
        _stub_checkpoint(tmp_path), voices=tmp_path, host="0.0.0.0", allow_public=True
    )
    streams = capsys.readouterr()
    # The token goes to stderr, not stdout: under systemd or any `>` redirect
    # stdout is the service log, and a credential does not belong in a file that
    # outlives the process and gets shipped to whatever collects logs. The
    # advice about the header is not a secret and stays on stdout with the rest
    # of the banner.
    assert "generated an access token" in streams.err
    assert "generated an access token" not in streams.out
    assert "Bearer <token>" in streams.out

    # And the app it handed uvicorn really does refuse an unauthenticated call.
    client = TestClient(captured["app"])
    assert client.get("/health").status_code == 401


class _RefusingEngine(_SlowEngine):
    """An engine that will not speak the language it was asked for.

    Mirrors ``GraphemeTextFrontend.encode``, which raises
    ``UnsupportedLanguageError`` for zh/ja/he/ko/ru: those need model-based
    preprocessing this build does not carry. The refusal is deliberate and its
    message names the alternative, so every transport's job is to deliver that
    sentence rather than a stack trace.
    """

    UNSUPPORTED = ("zh", "ja", "he", "ko", "ru")

    def _check(self, language: str | None) -> None:
        # No fallback of its own. The real frontend is handed a language the
        # engine has already resolved, so a fake that quietly substitutes "en"
        # for None is a fake that would pass whether or not the server forwards
        # what it was given.
        if language is None:
            return
        lang = language.lower()
        if lang in self.UNSUPPORTED:
            raise UnsupportedLanguageError(
                f"language {lang!r} needs model-based text preprocessing",
                language=lang,
                supported=("en", "pl"),
            )

    def synthesize_long(
        self, text, voice, *, seed=0, language=None, speed=1.0, previous_tokens=None
    ):
        self._check(language)
        return super().synthesize_long(text, voice, seed=seed, language=language)

    def synthesize(
        self, text, voice, *, seed=0, language=None, speed=1.0, previous_tokens=None
    ):
        self._check(language)
        return super().synthesize(text, voice, seed=seed, language=language)

    def stream(
        self,
        text,
        voice,
        *,
        seed=0,
        language=None,
        speed=1.0,
        previous_tokens=None,
        should_cancel=None,
    ):
        self._check(language)
        yield from super().stream(
            text, voice, seed=seed, language=language, should_cancel=should_cancel
        )


class _BuggyEngine(_SlowEngine):
    """A backend with a method nobody finished.

    The whole point of :class:`~loudkit.errors.UnsupportedLanguageError`. A
    ``NotImplementedError`` raised anywhere in a backend — a stub renderer, an
    unwritten branch — used to reach the client as ``400``, because the route
    caught the builtin and read every one of them as "the caller asked for a
    language we do not have". A caller cannot act on that: the request was
    fine.
    """

    def synthesize_long(
        self, text, voice, *, seed=0, language=None, speed=1.0, previous_tokens=None
    ):
        raise NotImplementedError("mel decoder for this backend is a stub")

    def synthesize(
        self, text, voice, *, seed=0, language=None, speed=1.0, previous_tokens=None
    ):
        raise NotImplementedError("mel decoder for this backend is a stub")


class _FailsAfterOneChunk(_SlowEngine):
    """Delivers a chunk, then raises the way an over-window passage does.

    ``Engine._strip_specials`` raises ``WindowOverflowError`` when a chunk's
    tokens exceed the render window — deliberately loud, because the
    alternative is speech going missing. On the streaming path that lands
    *after* audio has been delivered, which is the case this fake exists to
    reproduce.

    The loudkit type, not a bare ``ValueError``: it is what the engine raises,
    and it is what puts ``error_kind`` at ``bad_request``.
    """

    def stream(
        self,
        text,
        voice,
        *,
        seed=0,
        language=None,
        speed=1.0,
        previous_tokens=None,
        should_cancel=None,
    ):
        yield _FakeResult()
        raise WindowOverflowError(
            "26 speech tokens exceed the 8-token window by 18", n_tokens=26, window=8
        )


class _FailsWithABugAfterOneChunk(_SlowEngine):
    """Delivers a chunk, then fails the way a defect here does.

    The counterpart of :class:`_FailsAfterOneChunk`. Both used to produce a
    byte-identical terminal event, because the route catches everything and had
    only prose to distinguish them — so a client could not tell "ask for
    something else" from "this will fail the same way forever".
    """

    def stream(
        self,
        text,
        voice,
        *,
        seed=0,
        language=None,
        speed=1.0,
        previous_tokens=None,
        should_cancel=None,
    ):
        yield _FakeResult()
        raise RuntimeError("CUDA graph replay failed")


def _events(body: bytes) -> list[dict]:
    """Parse an SSE body into its JSON payloads."""
    return [
        json.loads(line[len("data: ") :])
        for line in body.decode().splitlines()
        if line.startswith("data: ")
    ]


def test_a_full_queue_answers_busy_on_the_stream_too(tmp_path, monkeypatch) -> None:
    """The 503 is documented for the server, not for one of its two routes.

    `/v1/synthesize/stream` could not produce it at all: Starlette sends
    `http.response.start` — 200, text/event-stream — before it pulls the first
    item out of the body generator, and the queue check lived inside that
    generator. The refusal became a clean 200 with an empty body, which to a
    client is indistinguishable from a passage that had nothing to say. The
    admission decision now happens in the route, before the response exists.
    """
    import loudkit.transports.http as server_mod

    # Zero rather than a race: every request is over the bound, so this asserts
    # the refusal itself rather than winning a scheduling coin flip.
    monkeypatch.setattr(server_mod, "_MAX_QUEUED", 0)
    client = TestClient(
        build_app(_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
    )
    body = {"text": "one. two.", "voice": "fake"}

    with client:
        for route in ("/v1/synthesize", "/v1/synthesize/stream"):
            resp = client.post(route, json=body)
            assert resp.status_code == 503, f"{route}: got {resp.status_code}"
            assert resp.headers.get("Retry-After") == "1", f"{route}: no Retry-After"
            assert "queued for the engine" in resp.text, f"{route}: {resp.text[:120]}"


def test_the_queue_slot_is_given_back_when_a_stream_ends(tmp_path, monkeypatch) -> None:
    """Taking the slot in the route means the route must not leak it.

    The slot is now acquired before the response and released in the
    generator's `finally`. If those ever come apart, the first stream wins and
    every request after it is a 503 forever — a failure that only shows up on
    the *second* request, which is why it gets its own test.
    """
    import loudkit.transports.http as server_mod

    monkeypatch.setattr(server_mod, "_MAX_QUEUED", 1)
    client = TestClient(
        build_app(_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
    )
    body = {"text": "one. two.", "voice": "fake"}

    with client:
        for attempt in range(3):
            resp = client.post("/v1/synthesize/stream", json=body)
            assert resp.status_code == 200, f"attempt {attempt}: {resp.status_code}"
            assert _events(resp.content)[-1]["done"] is True
        # And the sync route can still get in, so the lock came back too.
        assert client.post("/v1/synthesize", json=body).status_code == 200


def test_an_omitted_language_reaches_the_engine_as_none(tmp_path) -> None:
    """The server's whole share of the language chain is not having one.

    `SpeakRequest.language` defaults to None and is handed on unchanged, so the
    engine can consult `voice.language`. If the server substituted "en" — as
    the field's default used to — a Polish voice would read Polish text in
    English over HTTP while the in-process API got it right, and the two
    transports would disagree about the same call.

    Asserted on what the engine received, because that is the only place the
    difference is visible: both spellings return 200 with identical bytes from
    a fake that ignores language.
    """
    engine = _SlowEngine()
    client = TestClient(build_app(engine, _voices(tmp_path)), base_url="http://127.0.0.1:8765")  # type: ignore[arg-type]
    with client:
        body = {"text": "hi", "voice": "fake"}
        assert client.post("/v1/synthesize", json=body).status_code == 200
        client.post("/v1/synthesize/stream", json=body).raise_for_status()
    assert engine.languages == [None, None], engine.languages


def test_an_explicit_language_passes_through_verbatim(tmp_path) -> None:
    """The other half: a named language must not be re-resolved or normalised
    on the way through. The server is a transport, not a second frontend."""
    engine = _SlowEngine()
    client = TestClient(build_app(engine, _voices(tmp_path)), base_url="http://127.0.0.1:8765")  # type: ignore[arg-type]
    with client:
        body = {"text": "hi", "voice": "fake", "language": "pl"}
        assert client.post("/v1/synthesize", json=body).status_code == 200
        client.post("/v1/synthesize/stream", json=body).raise_for_status()
    assert engine.languages == ["pl", "pl"], engine.languages


def test_a_backends_own_not_implemented_error_is_a_server_fault(tmp_path) -> None:
    """The distinction the exception hierarchy exists for.

    The route used to catch the *builtin* `NotImplementedError` and answer 400.
    That is right for the one case it was written for — a language this build
    cannot preprocess — and wrong for every other `NotImplementedError` in the
    process: a stub method in a backend, an unwritten branch in a renderer. The
    client was told its request was bad when the request was fine and the
    server was broken, which is the one diagnosis it can do nothing with.

    Now only `UnsupportedLanguageError` is a 400. A bare
    `NotImplementedError` escapes to FastAPI and answers 500, and this test is
    the whole reason `loudkit.errors` exists.

    `raise_server_exceptions=False` because TestClient re-raises unhandled
    server exceptions by default, which would show the traceback instead of the
    status a real client over a real socket receives.
    """
    client = TestClient(
        build_app(_BuggyEngine(), _voices(tmp_path)),  # type: ignore[arg-type]
        raise_server_exceptions=False,
        base_url="http://127.0.0.1:8765",
    )
    with client:
        resp = client.post("/v1/synthesize", json={"text": "hello.", "voice": "fake"})
    assert resp.status_code == 500, f"got {resp.status_code}: {resp.text[:200]}"


def test_an_unsupported_language_is_the_callers_problem(tmp_path) -> None:
    """A language this build cannot preprocess is a 400, not a 500.

    `NotImplementedError` is a subclass of `RuntimeError`, so the routes'
    `except ValueError` never saw it and it escaped to FastAPI's 500 handler:
    a caller who asked a question about their own request was told the server
    had failed. The CLI has printed `unsupported: ...` since 30626c7; the two
    agent-facing transports had not caught up.

    The refusal is now a named type rather than the builtin — see
    :func:`test_a_backends_own_not_implemented_error_is_a_server_fault` for the
    case that forced the split.
    """
    client = TestClient(
        build_app(_RefusingEngine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
    )  # type: ignore[arg-type]
    body = {"text": "你好。", "voice": "fake", "language": "zh"}

    with client:
        resp = client.post("/v1/synthesize", json=body)
        assert resp.status_code == 400, f"got {resp.status_code}: {resp.text[:200]}"
        assert "preprocessing" in resp.text

        # The stream cannot answer 400 — the refusal happens inside the engine,
        # after Starlette has sent the status line — so it must say so in the
        # one event a client is documented to wait for.
        stream = client.post("/v1/synthesize/stream", json=body)
        assert stream.status_code == 200
        events = _events(stream.content)
        assert events, "the stream said nothing at all"
        assert events[-1]["done"] is True
        assert "preprocessing" in events[-1].get("error", "")
        # The status code the stream could not send, carried as a field: this
        # is the caller's to fix, so a client may usefully retry differently.
        assert events[-1]["error_kind"] == "bad_request"
        # And the specific condition, same catalog as the HTTP bodies.
        assert events[-1]["error_code"] == "unsupported_language"


def test_a_mid_stream_failure_is_named_rather_than_truncated(tmp_path) -> None:
    """A stream that stops is indistinguishable from a passage that ended.

    Once the first chunk is out, a synthesis failure cannot be a status code.
    Before this it was nothing at all: the connection carried two chunks and
    then closed, and a client reading until `done` waited for an event that
    never came — or, worse, treated the audio it had as the whole passage.
    """
    client = TestClient(
        build_app(_FailsAfterOneChunk(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
    )  # type: ignore[arg-type]

    with client:
        resp = client.post(
            "/v1/synthesize/stream", json={"text": "one. two. three.", "voice": "fake"}
        )
        assert resp.status_code == 200
        events = _events(resp.content)
        assert len(events) == 2, f"expected one chunk then a terminal event: {events}"
        assert "audio" in events[0], "the chunk that did render was lost"
        assert events[-1]["done"] is True, "no terminal event: the client waits forever"
        assert "window" in events[-1].get("error", ""), events[-1]
        assert events[-1]["error_kind"] == "bad_request", events[-1]

    # A defect here produces the same event shape, and used to be
    # indistinguishable from the above. `error_kind` is the whole difference:
    # one is worth retrying with a different request, the other never is.
    buggy = TestClient(
        build_app(_FailsWithABugAfterOneChunk(), _voices(tmp_path)),
        base_url="http://127.0.0.1:8765",
    )  # type: ignore[arg-type]
    with buggy:
        resp = buggy.post(
            "/v1/synthesize/stream", json={"text": "one. two. three.", "voice": "fake"}
        )
        assert resp.status_code == 200
        events = _events(resp.content)
        assert events[-1]["done"] is True
        assert events[-1]["error_kind"] == "server_fault", events[-1]


def test_a_legal_text_fits_the_body_bound_however_the_client_encodes_it(tmp_path) -> None:
    """The two bounds were reasoned about, not computed against each other.

    `_MAX_TEXT_LEN` is 10 000 *characters*; `_MAX_BODY_BYTES` was a flat 64 KB
    with a comment claiming the text cap "fits many times over". `json.dumps`
    defaults to `ensure_ascii=True` — so does `requests` — which spends six
    bytes per non-ASCII character and twelve per astral one, because a
    surrogate pair is two `\\uXXXX` escapes. A perfectly legal request of
    10 000 emoji encoded to 120 KB and was refused, naming a byte limit the
    caller has never been shown; 10 000 Polish characters sat at 92 % of the
    old bound, so a longer voice name was enough to tip it.

    httpx happens to encode with `ensure_ascii=False`, which is why every
    existing test missed this: the bodies are written the way the standard
    library and `requests` write them.
    """
    from loudkit.transports.http import _MAX_TEXT_LEN

    client = TestClient(
        build_app(_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
    )
    with client:
        # An astral character costs twelve bytes under `ensure_ascii`: a
        # surrogate pair is two escapes. A CJK Extension B ideograph rather
        # than an emoji, because the speech funnel scrubs symbols — an emoji
        # passage is legal at the transport and empty by the time it reaches
        # the splitter, which would test the funnel rather than the bound.
        for label, ch in (("ascii", "a"), ("polish", "ą"), ("astral", "𠀋")):
            body = json.dumps({"text": ch * _MAX_TEXT_LEN, "voice": "fake"}).encode()
            resp = client.post(
                "/v1/synthesize", content=body, headers={"content-type": "application/json"}
            )
            assert resp.status_code == 200, f"{label}: {len(body)} B -> {resp.status_code}"


def test_the_streaming_helper_enforces_the_text_cap_too(tmp_path) -> None:
    """`_MAX_TEXT_LEN` claims both transports inherit it "at the single place
    audio is made". `render_bytes` enforced it; `render_stream_chunks`, exported
    beside it in `__all__`, did not. Over HTTP pydantic covers the gap — an
    embedder calling the helper directly had no bound at all.
    """
    from loudkit.transports.http import _MAX_TEXT_LEN

    engine = _engine()
    voice = _voice()
    over = "a " * _MAX_TEXT_LEN

    with pytest.raises(ValueError, match="text too long"):
        render_bytes(engine, over, voice)
    with pytest.raises(ValueError, match="text too long"):
        next(iter(render_stream_chunks(engine, over, voice)))


class TestSpeed:
    """`speed` reaches the engine from both routes, and its bounds are the
    engine's bounds — a transport that re-implements the range is a second
    place for it to drift."""

    def test_the_default_is_a_bypass(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Byte-identical to the request that never mentioned speed, which is
        the promise every existing client is holding."""
        client = _client(tmp_path)
        plain = client.post("/v1/synthesize", json={"text": "one. two.", "voice": "fake"})
        explicit = client.post(
            "/v1/synthesize", json={"text": "one. two.", "voice": "fake", "speed": 1.0}
        )
        assert plain.content == explicit.content

    def test_faster_is_shorter(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        client = _client(tmp_path)
        plain = client.post("/v1/synthesize", json={"text": "one. two.", "voice": "fake"})
        fast = client.post(
            "/v1/synthesize", json={"text": "one. two.", "voice": "fake", "speed": 2.0}
        )
        assert float(fast.headers["X-Loudkit-Duration"]) == pytest.approx(
            float(plain.headers["X-Loudkit-Duration"]) / 2, rel=0.02
        )

    def test_out_of_range_is_a_422_naming_the_range(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        client = _client(tmp_path)
        resp = client.post(
            "/v1/synthesize", json={"text": "hello", "voice": "fake", "speed": 4.0}
        )
        assert resp.status_code == 422
        body = json.dumps(resp.json())
        assert "speed" in body
        assert "2" in body, "the refusal has to name the bound that was crossed"

    def test_the_stream_route_stretches_too(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        client = _client(tmp_path)

        def durations(payload: dict[str, object]) -> float:
            resp = client.post("/v1/synthesize/stream", json=payload)
            assert resp.status_code == 200
            total = 0.0
            for line in resp.text.splitlines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if not event.get("done"):
                    total += float(event["duration"])
            return total

        base = {"text": "one. two. three.", "voice": "fake"}
        assert durations({**base, "speed": 2.0}) == pytest.approx(durations(base) / 2, rel=0.02)


class TestCrossRequestContext:
    """A passage read as several requests should not restart its prosody at
    every one. The server's whole share of that is carrying two values.

    Its own engine because the module-level fake runs with ``prefix_tokens=0``
    — chunks deliberately independent — and the whole point here is the tail
    that a non-zero prefix produces.
    """

    def _client(self, tmp_path):  # type: ignore[no-untyped-def]
        from dataclasses import replace

        chunking = replace(AlgorithmConfig().chunking, max_tokens=4, prefix_tokens=2)
        algo = AlgorithmConfig().with_(chunking=chunking)
        engine = Engine(
            frontend=_SplitFrontend(),
            token_generator=_FakeGenerator(algo),
            mel_decoder=_FakeMelDecoder(algo),
            vocoder=_FakeVocoder(algo),
            algorithm=algo,
        )
        return TestClient(
            build_app(engine, _voices(tmp_path)), base_url="http://127.0.0.1:8765"
        )

    def test_the_reply_hands_back_what_to_send_next(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A header rather than a body field: the body is a WAV, and the
        alternative was multipart — which every audio client would then have to
        learn in order to play a sound."""
        client = self._client(tmp_path)
        resp = client.post("/v1/synthesize", json={"text": "one. two.", "voice": "fake"})
        tail = resp.headers["X-Loudkit-Continuation"]
        assert tail
        assert all(part.lstrip("-").isdigit() for part in tail.split(","))
        assert len(tail.split(",")) == 2

    def test_nothing_to_carry_means_no_header_at_all(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """An empty `X-Loudkit-Continuation: ` breaks the obvious client parse,
        `[int(t) for t in header.split(",")]`. A header a client must
        special-case is worse than one it can check for — so a recipe that
        carries no prefix sends none.

        The module-level fake runs with `prefix_tokens = 0`, which is exactly
        that case.
        """
        client = _client(tmp_path)
        resp = client.post("/v1/synthesize", json={"text": "one. two.", "voice": "fake"})
        assert resp.status_code == 200
        assert "X-Loudkit-Continuation" not in resp.headers

    def test_what_comes_back_is_accepted_going_out(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The round trip is the feature: a client should be able to feed the
        header straight back without knowing anything about token ids."""
        client = self._client(tmp_path)
        first = client.post("/v1/synthesize", json={"text": "one. two.", "voice": "fake"})
        tail = [int(t) for t in first.headers["X-Loudkit-Continuation"].split(",")]
        second = client.post(
            "/v1/synthesize",
            json={"text": "three. four.", "voice": "fake", "previous_tokens": tail},
        )
        assert second.status_code == 200
        assert second.headers["X-Loudkit-Continuation"]

    def test_the_stream_carries_the_tail_on_the_done_event(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """On ``done`` rather than on every chunk: what a chaining client needs
        is the tail of the passage, not of each piece of it."""
        client = self._client(tmp_path)
        resp = client.post(
            "/v1/synthesize/stream", json={"text": "one. two. three.", "voice": "fake"}
        )
        events = [
            json.loads(line[len("data: ") :])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events[-1]["done"] is True
        assert len(events[-1]["continuation"]) == 2

    def test_an_oversized_history_is_a_422(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The body bound would not stop a megabyte of integers that this
        process then parses into a list of Python ints."""
        client = self._client(tmp_path)
        resp = client.post(
            "/v1/synthesize",
            json={"text": "hello", "voice": "fake", "previous_tokens": list(range(5000))},
        )
        assert resp.status_code == 422

    def test_a_token_outside_the_codebook_is_a_422(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        client = self._client(tmp_path)
        resp = client.post(
            "/v1/synthesize",
            json={"text": "hello", "voice": "fake", "previous_tokens": [10**9]},
        )
        assert resp.status_code == 422
        assert "acoustic speech token" in json.dumps(resp.json())


class _SignalVocoder(_FakeVocoder):
    """A vocoder that renders something a codec has to actually encode.

    The module's ``_FakeVocoder`` emits zeros, which is right for the streaming
    assertions — chunked delivery matching sequential delivery does not care
    what the samples are — and useless for asserting that two encoders agree,
    because **silence round-trips identically through every codec**. The format
    tests below passed against it while WAV and FLAC disagreed on half of every
    real utterance; the test could not fail.

    Deterministic: fixed frequencies plus a fixed-seed noise floor, so a
    threshold measured here means the same thing tomorrow. The noise matters —
    pure sines are unusually easy for a lossy codec, and an Ogg tolerance
    calibrated on them would be calibrated on the best case.
    """

    def synthesize(self, mel: Mel, voice: VoiceProfile, *, seed: int) -> Waveform:
        n = mel.shape[1] * 256
        t = np.arange(n, dtype=np.float64) / 24_000
        wave = (
            0.5 * np.sin(2 * np.pi * 220 * t)
            + 0.25 * np.sin(2 * np.pi * 437 * t)
            + 0.1 * np.sin(2 * np.pi * 1310 * t)
            + 0.02 * np.random.default_rng(0).standard_normal(n)
        )
        return wave.astype(np.float32)


class TestOutputFormats:
    """Four encodings of one synthesis. The assertion that matters is that they
    are four encodings of *one* synthesis — decode any of them and the samples
    come back."""

    def _client(self, tmp_path):  # type: ignore[no-untyped-def]
        """An engine whose vocoder is audible. See ``_SignalVocoder``."""
        from dataclasses import replace

        chunking = replace(AlgorithmConfig().chunking, max_tokens=2, prefix_tokens=0)
        algo = AlgorithmConfig().with_(chunking=chunking)
        engine = Engine(
            frontend=_SplitFrontend(),
            token_generator=_FakeGenerator(algo),
            mel_decoder=_FakeMelDecoder(algo),
            vocoder=_SignalVocoder(algo),
            algorithm=algo,
        )
        return TestClient(
            build_app(engine, _voices(tmp_path)), base_url="http://127.0.0.1:8765"
        )

    def _decode(self, body: bytes, media_type: str):  # type: ignore[no-untyped-def]
        import io

        import soundfile as sf

        if media_type == "application/octet-stream":
            # No header to read: the frames are what the response headers said
            # they were, little-endian 16-bit at the advertised rate.
            return np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
        data, _ = sf.read(io.BytesIO(body), dtype="float32")
        return np.asarray(data, dtype=np.float32)

    def test_wav_is_byte_identical_to_before(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The default must not have moved. Every existing client is holding
        these exact bytes."""
        client = self._client(tmp_path)
        body = {"text": "one. two.", "voice": "fake"}
        assert (
            client.post("/v1/synthesize", json=body).content
            == client.post("/v1/synthesize", json={**body, "format": "wav"}).content
        )

    @pytest.mark.parametrize(
        ("fmt", "media_type"),
        [
            ("wav", "audio/wav"),
            ("pcm16", "application/octet-stream"),
            ("flac", "audio/flac"),
            ("ogg", "audio/ogg"),
        ],
    )
    def test_each_format_is_labelled_and_decodes(self, tmp_path, fmt, media_type) -> None:  # type: ignore[no-untyped-def]
        client = self._client(tmp_path)
        resp = client.post(
            "/v1/synthesize", json={"text": "one. two.", "voice": "fake", "format": fmt}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith(media_type)
        assert resp.headers["X-Loudkit-Sample-Rate"] == "24000"
        samples = self._decode(resp.content, media_type)
        assert samples.size > 0

    def test_the_lossless_formats_carry_the_same_samples(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """WAV, raw PCM and FLAC are three containers around one quantisation.

        Equal, not merely close — and only because the server quantises to
        int16 itself, once, before any encoder sees the floats. Left to
        libsndfile the two disagreed: its WAV writer floors and its FLAC writer
        rounds, which on real engine audio was 50 % of samples differing by one
        LSB between two formats both documented here as lossless.

        Driven by an audible vocoder on purpose. The zero-rendering fake this
        module uses elsewhere made this assertion unfalsifiable — silence
        survives every codec identically.
        """
        client = self._client(tmp_path)
        body = {"text": "one. two. three.", "voice": "fake"}
        got = {}
        for fmt, media_type in (
            ("wav", "audio/wav"),
            ("pcm16", "application/octet-stream"),
            ("flac", "audio/flac"),
        ):
            resp = client.post("/v1/synthesize", json={**body, "format": fmt})
            got[fmt] = self._decode(resp.content, media_type)
        assert np.array_equal(got["wav"], got["pcm16"])
        assert np.array_equal(got["wav"], got["flac"])

    def test_ogg_is_the_same_length_and_close(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Vorbis is lossy: same frames, not the same numbers. Asserted with a
        tolerance rather than skipped, because a silent failure to encode would
        otherwise look exactly like a pass.

        The bounds are stated as measurements with headroom, not as the numbers
        that happened to pass. On this signal — sines plus a fixed-seed noise
        floor, which is harder for a codec than sines alone — the worst sample
        lands at **0.0635** and the RMS error at **0.0149**, against the 0.15
        and 0.05 asserted here.

        The previous version of this test used 0.05 against a measured 0.04962:
        a 0.8 % margin nobody had ever exercised, one libsndfile release away
        from a red suite that would have meant nothing. Peak *and* RMS because
        peak alone is one bad sample away from noise, and RMS alone would not
        notice a single catastrophic one.
        """
        client = self._client(tmp_path)
        body = {"text": "one. two. three.", "voice": "fake"}
        wav = self._decode(
            client.post("/v1/synthesize", json={**body, "format": "wav"}).content, "audio/wav"
        )
        ogg = self._decode(
            client.post("/v1/synthesize", json={**body, "format": "ogg"}).content, "audio/ogg"
        )
        assert ogg.size == wav.size
        error = ogg - wav
        assert float(np.max(np.abs(error))) < 0.15
        assert float(np.sqrt(np.mean(np.square(error.astype(np.float64))))) < 0.05

    def test_an_unknown_format_is_a_422_listing_the_real_ones(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        client = self._client(tmp_path)
        resp = client.post(
            "/v1/synthesize", json={"text": "hi", "voice": "fake", "format": "mp3"}
        )
        assert resp.status_code == 422
        body = json.dumps(resp.json())
        for name in ("wav", "pcm16", "flac", "ogg"):
            assert name in body, body

    def test_the_stream_takes_pcm16(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        client = self._client(tmp_path)
        resp = client.post(
            "/v1/synthesize/stream",
            json={"text": "one. two. three.", "voice": "fake", "format": "pcm16"},
        )
        assert resp.status_code == 200
        events = [
            json.loads(line[len("data: ") :])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        chunks = [e for e in events if not e.get("done")]
        assert chunks
        import base64

        for event in chunks:
            assert event["media_type"] == "application/octet-stream"
            # Raw frames concatenate with `+`; that is the whole reason this
            # format is streamable and the containers are not.
            assert len(base64.b64decode(event["audio"])) % 2 == 0

    def test_the_stream_takes_flac_as_self_contained_files(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Every FLAC event is a complete file: header, frames, end-of-stream.
        Decoded alone it yields the same samples the one-shot route encodes,
        which is what "complete, playable payload" means here — and why FLAC
        streams honestly where Ogg does not."""
        import base64
        import io

        import soundfile as sf

        client = self._client(tmp_path)
        text = "one. two. three."
        streamed = client.post(
            "/v1/synthesize/stream",
            json={"text": text, "voice": "fake", "format": "flac"},
        )
        assert streamed.status_code == 200
        events = [
            json.loads(line[len("data: ") :])
            for line in streamed.text.splitlines()
            if line.startswith("data: ")
        ]
        chunks = [e for e in events if not e.get("done")]
        assert len(chunks) > 1
        for event in chunks:
            assert event["media_type"] == "audio/flac"
            raw = base64.b64decode(event["audio"])
            data, sr = sf.read(io.BytesIO(raw), dtype="float32")
            assert sr == 24000
            assert np.asarray(data).size > 0

    @pytest.mark.parametrize("fmt", ["ogg"])
    def test_the_stream_refuses_containers_with_the_allowed_set(self, tmp_path, fmt) -> None:  # type: ignore[no-untyped-def]
        """Ogg's bitstream state spans the whole stream, so a per-chunk
        container mislabels the bytes. Refused before the queue slot, because
        nothing about it will change while the server runs. FLAC was refused
        here once for the same reason and no longer is: each chunk encodes as
        a self-contained FLAC file, `media_type` rides beside every event, and
        it costs ~65% less on the wire than base64 WAV — the trade (chunks
        concatenate with a decoder, not with `+`) was already true of WAV."""
        client = self._client(tmp_path)
        resp = client.post(
            "/v1/synthesize/stream",
            json={"text": "one. two.", "voice": "fake", "format": fmt},
        )
        assert resp.status_code == 422
        detail = json.dumps(resp.json())
        assert "pcm16" in detail
        assert "wav" in detail


class TestABadTokenListIsTheCallersProblem:
    """The streaming route maps exception *types* to `bad_request` or
    `server_fault`, and everything outside the loudkit hierarchy is a fault by
    definition. `_carry_from` raised a bare ValueError, so one wrong integer in
    a request body was reported to the client as "the server is broken" — the
    one verdict a client cannot act on, and exactly backwards. The one-shot
    route was already right, which is how the two came to disagree.
    """

    def _client(self, tmp_path):  # type: ignore[no-untyped-def]
        from dataclasses import replace

        chunking = replace(AlgorithmConfig().chunking, max_tokens=4, prefix_tokens=2)
        algo = AlgorithmConfig().with_(chunking=chunking)
        engine = Engine(
            frontend=_SplitFrontend(),
            token_generator=_FakeGenerator(algo),
            mel_decoder=_FakeMelDecoder(algo),
            vocoder=_FakeVocoder(algo),
            algorithm=algo,
        )
        return TestClient(
            build_app(engine, _voices(tmp_path)), base_url="http://127.0.0.1:8765"
        ), algo

    def test_the_stream_calls_it_a_bad_request(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        client, algo = self._client(tmp_path)
        resp = client.post(
            "/v1/synthesize/stream",
            json={
                "text": "one. two.",
                "voice": "fake",
                "previous_tokens": [algo.stop_speech_token],
            },
        )
        # Refused before the response starts, so it is a status code rather than
        # a `done` event carrying a verdict — which is the better of the two
        # answers, and the reason the check was hoisted above the engine slot.
        assert resp.status_code == 422
        assert "acoustic speech token" in json.dumps(resp.json())

    def test_it_is_refused_before_a_queue_slot_is_taken(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A list that was never usable should not wait behind every other
        synthesis in order to fail. Asserted by the engine never being asked:
        the generator records nothing."""
        client, algo = self._client(tmp_path)
        resp = client.post(
            "/v1/synthesize/stream",
            json={"text": "one. two.", "voice": "nope", "previous_tokens": [-1]},
        )
        # The voice is bogus too, and the token check still wins: it is the
        # cheaper question and it is settled first.
        assert resp.status_code == 422

    def test_the_classifier_agrees_even_if_it_reaches_the_stream_body(self) -> None:
        """The hoisted check makes this unreachable through `previous_tokens`,
        but the classification is the actual defect and is pinned on its own:
        any later raise site validating caller-supplied ids inherits it."""
        from loudkit.errors import InvalidTokensError
        from loudkit.transports.http import _error_kind

        assert _error_kind(InvalidTokensError("x", token=9, limit=8)) == "bad_request"

    def test_the_error_carries_the_offending_id_and_the_bound(self) -> None:
        """A caller filtering a long sequence needs to know which entry to look
        at, not which list."""
        import loudkit
        from loudkit.engine import validate_speech_tokens

        algo = AlgorithmConfig()
        with pytest.raises(loudkit.InvalidTokensError) as exc:
            validate_speech_tokens([1, 2, 99_999], limit=algo.start_speech_token)
        assert exc.value.token == 99_999
        assert exc.value.limit == algo.start_speech_token


class TestTheOpenAICompatibleRoute:
    """`/v1/audio/speech`, so existing tooling can use this server unmodified.

    The point of the route is that a client written against OpenAI's API needs
    no adapter — only a base URL. These tests are therefore mostly about the
    edges where the two APIs do *not* line up: the formats this server will not
    encode, the speed range it will not honour, and the default OpenAI picked
    that this server cannot.
    """

    def test_it_returns_exactly_what_the_native_route_returns(self, tmp_path) -> None:
        """The claim that makes this a transport and not a second engine."""
        client = _client(tmp_path)
        native = client.post("/v1/synthesize", json={"text": "hello", "voice": "fake"})
        compat = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "input": "hello", "voice": "fake"},
        )
        assert native.status_code == compat.status_code == 200
        assert compat.content == native.content
        # The loudkit headers ride along too — an OpenAI client ignores what it
        # does not know, and one that does know gains the fingerprint.
        assert (
            compat.headers["X-Loudkit-Fingerprint"] == (native.headers["X-Loudkit-Fingerprint"])
        )

    def test_the_model_field_is_accepted_and_ignored(self, tmp_path) -> None:
        # Their API requires it and this server has one engine. Refusing a
        # request for naming a model would break every conforming client.
        client = _client(tmp_path)
        for model in ("tts-1", "tts-1-hd", "gpt-4o-mini-tts", ""):
            resp = client.post(
                "/v1/audio/speech",
                json={"model": model, "input": "hi", "voice": "fake"},
            )
            assert resp.status_code == 200, model

    def test_an_unset_format_is_wav_not_mp3(self, tmp_path) -> None:
        """The one deliberate deviation from their API, pinned here.

        OpenAI defaults `response_format` to mp3 and this server ships no mp3
        encoder, so honouring the default would mean every client that did not
        configure a format got an error instead of audio.
        """
        resp = _client(tmp_path).post("/v1/audio/speech", json={"input": "hi", "voice": "fake"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("audio/wav")
        assert resp.content[:4] == b"RIFF"

    @pytest.mark.parametrize("fmt", ["mp3", "aac", "opus"])
    def test_a_format_this_server_cannot_encode_is_refused_by_name(
        self, tmp_path, fmt: str
    ) -> None:
        # `opus` is the one that matters: `ogg` here is Ogg Vorbis, which shares
        # a media type with Ogg Opus and no bitstream, so answering with it
        # would be a decode failure inside a 200.
        resp = _client(tmp_path).post(
            "/v1/audio/speech",
            json={"input": "hi", "voice": "fake", "response_format": fmt},
        )
        assert resp.status_code == 400
        message = resp.json()["error"]["message"]
        assert fmt in message
        # The refusal has to say what *would* have worked.
        assert "wav" in message
        assert "flac" in message

    def test_an_unknown_format_is_refused_the_same_way(self, tmp_path) -> None:
        resp = _client(tmp_path).post(
            "/v1/audio/speech",
            json={"input": "hi", "voice": "fake", "response_format": "aiff"},
        )
        assert resp.status_code == 400
        assert "aiff" in resp.json()["error"]["message"]

    @pytest.mark.parametrize(
        ("asked", "media_type"),
        [
            ("wav", "audio/wav"),
            ("flac", "audio/flac"),
            ("pcm", "application/octet-stream"),
            # This server's own spellings are accepted as well, so a caller who
            # knows what it is talking to need not translate.
            ("pcm16", "application/octet-stream"),
            ("ogg", "audio/ogg"),
        ],
    )
    def test_the_formats_that_do_work(self, tmp_path, asked: str, media_type: str) -> None:
        resp = _client(tmp_path).post(
            "/v1/audio/speech",
            json={"input": "hi", "voice": "fake", "response_format": asked},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith(media_type)

    def test_their_pcm_is_this_servers_pcm16(self, tmp_path) -> None:
        client = _client(tmp_path)
        compat = client.post(
            "/v1/audio/speech",
            json={"input": "hi", "voice": "fake", "response_format": "pcm"},
        )
        native = client.post(
            "/v1/synthesize", json={"text": "hi", "voice": "fake", "format": "pcm16"}
        )
        assert compat.content == native.content

    @pytest.mark.parametrize("speed", [0.25, 4.0])
    def test_a_speed_their_api_allows_and_this_engine_does_not(
        self, tmp_path, speed: float
    ) -> None:
        """Refused, not clamped, and refused with the range that applies.

        A caller asking for 4x and silently receiving 2x has been handed audio
        that is not what it requested, with nothing in the reply saying so.
        """
        resp = _client(tmp_path).post(
            "/v1/audio/speech",
            json={"input": "hi", "voice": "fake", "speed": speed},
        )
        assert resp.status_code == 400
        message = resp.json()["error"]["message"]
        assert str(speed) in message
        assert str(MIN_SPEED) in message
        assert str(MAX_SPEED) in message

    def test_a_speed_both_agree_on_still_works(self, tmp_path) -> None:
        resp = _client(tmp_path).post(
            "/v1/audio/speech",
            json={"input": "hi", "voice": "fake", "speed": 1.5},
        )
        assert resp.status_code == 200

    def test_an_unknown_voice_answers_in_their_envelope(self, tmp_path) -> None:
        # A 404 whose body a conforming client can actually read: it looks for
        # `error.message`, and FastAPI's own `detail` would surface as a blank
        # HTTP failure with the list of real voice names thrown away.
        resp = _client(tmp_path).post(
            "/v1/audio/speech", json={"input": "hi", "voice": "nobody"}
        )
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" not in body
        assert "nobody" in body["error"]["message"]
        assert body["error"]["type"] == "invalid_request_error"

    def test_empty_input_is_refused(self, tmp_path) -> None:
        resp = _client(tmp_path).post("/v1/audio/speech", json={"input": "", "voice": "fake"})
        assert resp.status_code == 422

    def test_an_unknown_field_does_not_fail_the_request(self, tmp_path) -> None:
        # `stream_format` is the live case: this route does not implement their
        # SSE envelope, and a client asking for it should get the whole
        # utterance rather than an error.
        resp = _client(tmp_path).post(
            "/v1/audio/speech",
            json={"input": "hi", "voice": "fake", "stream_format": "sse"},
        )
        assert resp.status_code == 200

    def test_it_is_behind_the_same_bearer_token(self, tmp_path) -> None:
        """An OpenAI client's API key is this server's token, with no wiring.

        Worth a test rather than an assumption: the guard is ASGI middleware
        over the whole app, so a route added later is covered by construction —
        and this is the test that fails if that ever stops being true.
        """
        client = TestClient(
            build_app(_engine(), _voices(tmp_path), token="s3cret-real-token"),
            base_url="http://127.0.0.1:8765",
        )
        body = {"input": "hi", "voice": "fake"}
        assert client.post("/v1/audio/speech", json=body).status_code == 401
        ok = client.post(
            "/v1/audio/speech", json=body, headers={"Authorization": "Bearer s3cret-real-token"}
        )
        assert ok.status_code == 200

    def test_the_real_openai_sdk_talks_to_it_unmodified(self, tmp_path) -> None:
        """The claim this route exists for, checked against their own client.

        Everything above drives the wire format directly, which is the contract
        — but the contract is only worth having if the clients people actually
        run accept it. This one builds an `OpenAI`, points its base URL at this
        server, hands it the TestClient as its transport, and asks for speech.

        Skipped rather than required: `openai` is not a dependency of this
        project and adding one so that a compatibility claim can be tested
        would be a strange trade. Where it is installed, this runs.
        """
        openai = pytest.importorskip("openai")

        client = TestClient(build_app(_engine(), _voices(tmp_path), token="s3cret-real-token"))
        # The API key is the server's bearer token — no second mechanism.
        sdk = openai.OpenAI(
            api_key="s3cret-real-token", base_url="http://127.0.0.1:8765/v1", http_client=client
        )

        spoken = sdk.audio.speech.create(
            model="tts-1", voice="fake", input="hello", response_format="wav"
        )
        native = client.post(
            "/v1/synthesize",
            json={"text": "hello", "voice": "fake"},
            headers={"Authorization": "Bearer s3cret-real-token"},
        )
        assert spoken.content == native.content

        # And the refusals arrive as the typed errors the SDK raises, carrying
        # the message rather than an empty HTTP failure.
        with pytest.raises(openai.BadRequestError, match="mp3"):
            sdk.audio.speech.create(
                model="tts-1", voice="fake", input="hi", response_format="mp3"
            )
        with pytest.raises(openai.NotFoundError, match="nobody"):
            sdk.audio.speech.create(
                model="tts-1", voice="nobody", input="hi", response_format="wav"
            )


def test_the_openapi_schema_generates(tmp_path) -> None:
    """`/openapi.json` — and therefore `/docs` — must not be a 500.

    It was. Every route returning a `Response` subclass is annotated as such,
    and under PEP 563 those annotations are strings resolved at module scope —
    where `Response` does not exist, because fastapi is imported inside
    `build_app` so that importing this module does not require it. FastAPI then
    tried to build a response model out of an unresolvable forward reference and
    raised on every request for the schema.

    `response_model=None` on those routes is the documented way to say "this
    returns a response, not a model". Pinned here because nothing else asks for
    the schema: the failure was invisible to the entire test suite while being
    the first thing a person browsing the server hits.
    """
    client = _client(tmp_path)
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert set(resp.json()["paths"]) == {
        "/health",
        "/v1/voices",
        "/v1/synthesize",
        "/v1/synthesize/stream",
        "/v1/audio/speech",
    }


def test_a_request_whose_words_the_funnel_strips_is_refused(tmp_path) -> None:
    """Emoji and bare symbols are legal at the transport and gone by the funnel.

    Both entry points must refuse identically: a clean empty success teaches
    the caller that silence was the request, which is the one failure they
    cannot see in the audio.
    """
    client = TestClient(
        build_app(_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
    )
    with client:
        body = {"text": "🎉🎉®™", "voice": "fake"}
        whole = client.post("/v1/synthesize", json=body)
        assert whole.status_code == 422
        assert "nothing to speak" in whole.text

        # The stream answers in band: headers are gone by the time the funnel
        # empties the text, so the refusal arrives as the terminal event's
        # explicit error — never as a clean done.
        streamed = client.post("/v1/synthesize/stream", json=body)
        assert streamed.status_code == 200
        events = [line for line in streamed.text.splitlines() if line.startswith("data:")]
        assert events, "the stream produced no events at all"
        terminal = json.loads(events[-1][len("data: ") :])
        # The stream's terminal message carries the exception class prefix, as
        # every transport error does; kind is the machine-readable half.
        assert "nothing to speak" in terminal.get("error", "")
        assert terminal.get("error_kind") == "bad_request"


class TestAnEmbedderCannotBuildAnOpenServer:
    """`build_app` is exported, so it is where the refusal has to live.

    `serve` refuses a non-loopback bind without `--allow-public` and generates
    a token when it is given one, but an embedder mounting the app in their own
    uvicorn, gunicorn or ASGI stack never executes a line of it. The same
    argument pair passed straight to `build_app` produced an app with no Host
    pin (`allow_public` switches it off), no token to check, and no rate limit
    — reachable from the network, speaking in every voice on the machine.
    """

    def test_public_without_a_token_is_refused_at_construction(self, tmp_path) -> None:
        with pytest.raises(ValueError) as caught:
            build_app(_engine(), _voices(tmp_path), allow_public=True)
        message = str(caught.value)
        # The message has to name the cause and the remedy: this is raised at
        # import-adjacent wiring time, far from any request that would show it.
        assert "no authentication" in message
        assert "token=" in message

    def test_public_with_a_token_builds(self, tmp_path) -> None:
        app = build_app(
            _engine(), _voices(tmp_path), token="s3cret-real-token", allow_public=True
        )
        client = TestClient(app, base_url="http://loudkit.example:8765")
        body = {"text": "hi", "voice": "fake"}
        # No Host pin off loopback, by design — the token is the boundary.
        assert client.post("/v1/synthesize", json=body).status_code == 401
        ok = client.post(
            "/v1/synthesize", json=body, headers={"Authorization": "Bearer s3cret-real-token"}
        )
        assert ok.status_code == 200

    def test_the_loopback_default_still_needs_nothing(self, tmp_path) -> None:
        """The developer path: no token, no flag, and the Host pin still on."""
        client = TestClient(
            build_app(_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
        )
        body = {"text": "hi", "voice": "fake"}
        assert client.post("/v1/synthesize", json=body).status_code == 200
        pinned = TestClient(
            build_app(_engine(), _voices(tmp_path)), base_url="http://evil.example:8765"
        )
        assert pinned.get("/v1/voices").status_code == 403

    def test_a_public_app_is_rate_limited(self, tmp_path) -> None:
        """Off the machine, a token holder is not entitled to the whole queue.

        The limiter used to be conditioned on the token alone, with a comment
        arguing that on loopback it only buys a way to lock yourself out. That
        argument is about loopback: it says nothing about a bind where the
        callers are strangers and the engine synthesises one at a time.
        """
        from loudkit.transports.http import _RATE_CAPACITY, _Guard

        assert _Guard(None, token=None, allow_public=False).buckets is None
        assert _Guard(None, token="t", allow_public=True).buckets is not None

        client = TestClient(
            build_app(
                _engine(), _voices(tmp_path), token="s3cret-real-token", allow_public=True
            ),
            base_url="http://loudkit.example:8765",
        )
        head = {"Authorization": "Bearer s3cret-real-token"}
        body = {"text": "hi", "voice": "fake"}
        codes = [
            client.post("/v1/synthesize", json=body, headers=head).status_code
            for _ in range(_RATE_CAPACITY + 2)
        ]
        assert codes.count(429) >= 1
        # The reads stay open: a limiter that answers /health with 429 makes a
        # busy server look dead to whatever is watching it.
        assert client.get("/health", headers=head).status_code == 200


class TestATokenHasToBeUsableAsOne:
    """A token that is set switches authentication on. It must be a secret.

    `build_app` refused only `None` under `allow_public`, so `token=""` built a
    public, "authenticated" server whose check was `Authorization: Bearer ` —
    a header anyone can send, on a boundary that reports itself as closed.
    Whitespace is that same hole with a typo in it, a control character is a
    header-splitting primitive travelling in a credential, and a one-character
    token is a boundary that can be guessed faster than it can be typed.
    """

    GOOD = secrets.token_urlsafe(32)

    @pytest.mark.parametrize(
        ("token", "reason"),
        [
            ("", "empty"),
            ("   ", "empty"),
            # A lone tab is whitespace-only, so the emptiness rule catches it
            # first; the header rule is for control characters inside a token
            # that is otherwise the right size.
            ("\t", "empty"),
            ("abcdefgh\nijklmnop", "printable"),
            ("abcdefgh\x07ijklmnop", "printable"),
            ("abcdefgh ijklmnop", "printable"),
            ("short", "minimum"),
            ("x" * (_MIN_TOKEN_CHARS - 1), "minimum"),
        ],
    )
    def test_a_token_that_is_not_a_secret_is_refused_by_name(
        self, tmp_path, token, reason
    ) -> None:
        with pytest.raises(ValueError) as caught:
            build_app(_engine(), _voices(tmp_path), token=token, allow_public=True)
        # Named, not just refused: this is raised at wiring time, where the
        # only diagnosis the operator gets is the string.
        assert reason in str(caught.value)

    def test_the_same_rule_holds_on_a_loopback_app(self, tmp_path) -> None:
        """`allow_public` is not what makes an empty token dangerous.

        The guard enforces a set token on either bind, so an empty one on
        loopback is the same non-check with a smaller audience.
        """
        with pytest.raises(ValueError, match="empty"):
            build_app(_engine(), _voices(tmp_path), token="")

    def test_a_real_token_is_accepted_and_enforced(self, tmp_path) -> None:
        app = build_app(_engine(), _voices(tmp_path), token=self.GOOD, allow_public=True)
        client = TestClient(app, base_url="http://loudkit.example:8765")
        assert client.get("/health").status_code == 401
        head = {"Authorization": f"Bearer {self.GOOD}"}
        assert client.get("/health", headers=head).status_code == 200

    def test_the_minimum_is_a_boundary_not_a_gesture(self, tmp_path) -> None:
        """Exactly `_MIN_TOKEN_CHARS` passes; one character less does not."""
        assert (
            build_app(_engine(), _voices(tmp_path), token="a" * _MIN_TOKEN_CHARS).title
            == "loudkit"
        )
        with pytest.raises(ValueError, match="minimum"):
            build_app(_engine(), _voices(tmp_path), token="a" * (_MIN_TOKEN_CHARS - 1))

    def test_the_loopback_default_still_builds_with_no_token(self, tmp_path) -> None:
        """The developer path takes no credential and must not acquire one."""
        client = TestClient(
            build_app(_engine(), _voices(tmp_path)), base_url="http://127.0.0.1:8765"
        )
        assert client.get("/health").status_code == 200

    @pytest.mark.parametrize("token", ["", "   ", "short", "abcdefgh\nijklmnop"])
    def test_serve_refuses_a_hand_supplied_token_before_the_load(
        self, tmp_path, monkeypatch, token
    ) -> None:
        """One rule, whichever entry point the operator reached for.

        `serve` is where `--token` and an environment variable arrive, and it
        hands its token to `build_app` only after loading 747 MB of weights.
        The refusal has to land before that, so the check is repeated at the
        top rather than left to the constructor at the bottom.
        """
        import loudkit
        import loudkit.transports.http as server_mod

        def refuse_to_load(*_a, **_k):
            raise AssertionError("the weights were loaded before the token was judged")

        monkeypatch.setattr(loudkit, "load", refuse_to_load)

        with pytest.raises(SystemExit, match="public bind"):
            server_mod.serve(
                _stub_checkpoint(tmp_path),
                host="0.0.0.0",
                allow_public=True,
                token=token,
            )

    def test_serve_still_takes_a_real_token(self, tmp_path, monkeypatch) -> None:
        import sys

        import loudkit
        import loudkit.transports.http as server_mod

        monkeypatch.setattr(loudkit, "load", lambda *_a, **_k: _engine())
        captured = {}
        monkeypatch.setitem(
            sys.modules,
            "uvicorn",
            type(
                "_U",
                (),
                {"run": staticmethod(lambda app, **_kw: captured.update(app=app))},
            ),
        )

        server_mod.serve(
            _stub_checkpoint(tmp_path),
            voices=tmp_path,
            host="0.0.0.0",
            allow_public=True,
            token=self.GOOD,
        )
        client = TestClient(captured["app"])
        assert client.get("/health").status_code == 401
        assert (
            client.get("/health", headers={"Authorization": f"Bearer {self.GOOD}"}).status_code
            == 200
        )
