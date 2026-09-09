"""The MCP server's surface, exercised without weights.

Same rule as the HTTP server's tests: the MCP server must hold no synthesis
path of its own, so these assert that its tools resolve through the engine and
the voice library — a fake engine injected via ``build_server(engine=...)`` —
and that tool *calls* return the shapes an agent host expects. The WAV bytes
come from :func:`~loudkit.synthesis.render_bytes`, which the HTTP tests already
pin to the engine; what is new here is the transport, not the synthesis.

The ``mcp`` SDK is an extra, so the whole module skips when it is absent —
the same discipline the server's fastapi import uses.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time

import numpy as np
import pytest

from loudkit.config import AlgorithmConfig
from loudkit.contracts import Mel, Sampler, SpeechTokens, Waveform
from loudkit.engine import Engine
from loudkit.errors import CancelledError
from loudkit.voice import VoiceProfile

# The shared weight-free set, one definition for every suite that uses it.
from .conftest import (
    FakeGenerator,
    FakeMelDecoder,
    FakeVocoder,
    _SplitFrontend,
    _stub_checkpoint,
    _voice,
)

pytest.importorskip("mcp.server.mcpserver")

from loudkit.transports.mcp import build_server


def _engine() -> Engine:
    """Local, not shared: the HTTP suite's engine caps chunking so its stream
    splits, and this one wants the library's own defaults."""
    algo = AlgorithmConfig()
    return Engine(
        frontend=_SplitFrontend(),
        token_generator=FakeGenerator(algo),
        mel_decoder=FakeMelDecoder(algo),
        vocoder=FakeVocoder(algo),
        algorithm=algo,
    )


def _server(tmp_path, engine: Engine | None = None) -> object:
    _voice().save(tmp_path / "fake.safetensors")
    return build_server(
        str(_stub_checkpoint(tmp_path)),
        str(tmp_path),
        engine=engine or _engine(),
    )


def _run(coro):
    return asyncio.run(coro)


def test_lists_voices(tmp_path) -> None:
    result = _run(_server(tmp_path).call_tool("list_voices", {}))
    text = result.content[0].text
    assert "fake" in text


def test_describe_reports_fingerprint(tmp_path) -> None:
    server = _server(tmp_path)
    result = _run(server.call_tool("describe", {}))
    body = result.content[0].text
    assert "fingerprint" in body
    assert "algo[" in body


def test_synthesize_returns_base64_wav(tmp_path) -> None:
    server = _server(tmp_path)
    result = _run(
        server.call_tool("synthesize", {"text": "one. two.", "voice": "fake", "seed": 7})
    )
    body = result.content[0].text
    payload = json.loads(body)
    assert payload["duration"] > 0
    assert payload["tokens"] > 0
    wav = base64.b64decode(payload["audio"])
    assert wav[:4] == b"RIFF"
    # The reply says what the base64 decodes to, so an agent writing a file
    # knows the extension without sniffing magic bytes.
    assert payload["format"] == "wav"
    assert payload["media_type"] == "audio/wav"


def test_synthesize_format_flac_returns_flac(tmp_path) -> None:
    """`format` reaches the one synthesis path; the reply names what came back.

    FLAC is the format that matters on this transport: the bytes land base64'd
    in a model's context, and a quarter the size is a quarter the tokens.
    """
    server = _server(tmp_path)
    result = _run(
        server.call_tool(
            "synthesize",
            {"text": "one. two.", "voice": "fake", "seed": 7, "format": "flac"},
        )
    )
    payload = json.loads(result.content[0].text)
    assert base64.b64decode(payload["audio"])[:4] == b"fLaC"
    assert payload["format"] == "flac"
    assert payload["media_type"] == "audio/flac"


def test_the_tool_returns_the_bytes_the_library_returns(tmp_path) -> None:
    """The claim the transport exists to keep, on MCP.

    The two tests above check the magic bytes, which a transport that re-encoded
    or dropped the provenance trailer would still satisfy. `render_bytes` is the
    only place this library makes audio, so what the tool base64s has to be
    exactly what a direct call returns. Both formats, because the encoder is
    chosen per call.

    `test_grpc_returns_the_same_bytes_as_the_library` and
    `test_the_route_returns_the_bytes_the_library_returns` are the same
    assertion on the other two doors.
    """
    from loudkit.synthesis import VoiceLibrary, render_bytes

    engine = _engine()
    server = _server(tmp_path, engine=engine)
    voice = VoiceLibrary(tmp_path).load("fake")
    text = "one. two."

    for audio_format in ("wav", "flac"):
        direct = render_bytes(engine, text, voice, seed=7, audio_format=audio_format)
        result = _run(
            server.call_tool(
                "synthesize",
                {"text": text, "voice": "fake", "seed": 7, "format": audio_format},
            )
        )
        payload = json.loads(result.content[0].text)
        assert base64.b64decode(payload["audio"]) == direct.data, (
            f"{audio_format}: the tool made its own audio"
        )
        assert payload["media_type"] == direct.media_type


def test_synthesize_unknown_format_is_refused_with_the_supported_list(tmp_path) -> None:
    server = _server(tmp_path)
    result = _run(
        server.call_tool(
            "synthesize",
            {"text": "one. two.", "voice": "fake", "format": "mp3"},
        )
    )
    payload = json.loads(result.content[0].text)
    assert payload["error_kind"] == "bad_request"
    assert "mp3" in payload["error"]
    assert "flac" in payload["supported"]


class _CappedGenerator(FakeGenerator):
    """Runs to the cap without a stop token — a broken EOS path, deterministically."""

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
        return list(range(max_new_tokens or self.config.sampling.max_new_tokens))


def test_synthesize_reports_truncation(tmp_path) -> None:
    """An agent must be able to tell a finished sentence from a severed one.

    The tool used to return the truncated WAV with duration and token count and
    nothing else, so a cut-off utterance was indistinguishable from a complete
    one — the failure mode an autonomous caller is least able to notice.
    """
    from dataclasses import replace

    algo = AlgorithmConfig()
    algo = algo.with_(sampling=replace(algo.sampling, max_new_tokens=4))
    capped = Engine(
        frontend=_SplitFrontend(),
        token_generator=_CappedGenerator(algo),
        mel_decoder=FakeMelDecoder(algo),
        vocoder=FakeVocoder(algo),
        algorithm=algo,
    )

    result = _run(
        _server(tmp_path, capped).call_tool(
            "synthesize", {"text": "one two three", "voice": "fake"}
        )
    )
    assert json.loads(result.content[0].text)["truncated"] is True

    # The uncapped fake stops on its stop token, so the flag must be false —
    # otherwise this asserts nothing but the key's existence.
    ok = _run(_server(tmp_path).call_tool("synthesize", {"text": "one. two.", "voice": "fake"}))
    assert json.loads(ok.content[0].text)["truncated"] is False


def test_synthesize_unknown_voice_returns_error(tmp_path) -> None:
    result = _run(_server(tmp_path).call_tool("synthesize", {"text": "hi", "voice": "nope"}))
    assert "error" in result.content[0].text


def test_synthesize_oversized_text_returns_error(tmp_path) -> None:
    """The MCP tool inherits the shared render guard: a text longer than the
    cap is refused rather than fed to chunking (memory/latency DoS)."""
    result = _run(
        _server(tmp_path).call_tool("synthesize", {"text": "a" * 10_001, "voice": "fake"})
    )
    assert "error" in result.content[0].text


def test_unknown_tool_is_rejected(tmp_path) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    server = _server(tmp_path)
    with pytest.raises(ToolError):
        _run(server.call_tool("no_such_tool", {}))


def test_synthesize_unsupported_language_returns_error(tmp_path) -> None:
    """A language this build cannot preprocess is an answer, not a crash.

    ``GraphemeTextFrontend.encode`` raises ``UnsupportedLanguageError`` for a
    language off the roster, and the message names what would have worked. The
    tool caught only ``(FileNotFoundError, ValueError)``, so the exception
    escaped the tool call and the agent host saw a transport failure instead of
    the sentence explaining which languages work, which the CLI prints.

    The fake raises the loudkit type rather than the builtin, because that is
    now the whole distinction: see
    :func:`test_a_backend_stub_is_not_reported_as_the_agents_mistake`.
    """
    from loudkit.errors import UnsupportedLanguageError

    class _RefusingFrontend:
        def encode(self, text: str, language: str = "en") -> np.ndarray:
            if language.lower() in ("zh", "ja", "he", "ko", "ru"):
                raise UnsupportedLanguageError(
                    f"language {language.lower()!r} needs model-based text preprocessing",
                    language=language.lower(),
                    supported=("en", "pl"),
                )
            return np.arange(len(text.split()), dtype=np.int64)

    engine = _engine()
    server = _server(
        tmp_path,
        engine=Engine(
            frontend=_RefusingFrontend(),
            token_generator=engine.token_generator,
            mel_decoder=engine.mel_decoder,
            vocoder=engine.vocoder,
            algorithm=engine.algorithm,
        ),
    )
    result = _run(
        server.call_tool("synthesize", {"text": "你好。", "voice": "fake", "language": "zh"})
    )
    assert "preprocessing" in result.content[0].text


class _RecordingEngine:
    """Wraps the fake engine and remembers the language it was handed.

    The MCP tool's whole share of the language chain is *not resolving it*:
    `language` defaults to None and is passed on, so the engine can consult
    `voice.language`. A fake that ignores the argument cannot tell that apart
    from a tool that hardcodes "en" — and for a while nothing could, because
    reverting the default left every MCP test green.
    """

    def __init__(self) -> None:
        self._inner = _engine()
        self.algorithm = self._inner.algorithm
        self.execution = self._inner.execution
        self.backend = self._inner.backend
        self.checkpoint_sha256 = self._inner.checkpoint_sha256
        self.languages: list[str | None] = []

    def synthesize(
        self,
        text,
        voice,
        *,
        seed=0,
        language=None,
        speed=1.0,
        previous_tokens=None,
        single_window=False,
        should_cancel=None,
    ):
        self.languages.append(language)
        return self._inner.synthesize(text, voice, seed=seed, language=language)


def test_an_omitted_language_reaches_the_engine_as_none(tmp_path) -> None:
    """Omitting `language` must mean "the voice's own", over MCP as in process.

    If the tool substituted "en", an agent calling `synthesize(text, voice)`
    with a Polish voice would get an English read of Polish text, while the
    same call through the Python API got it right.
    """
    engine = _RecordingEngine()
    server = _server(tmp_path, engine=engine)
    _run(server.call_tool("synthesize", {"text": "one. two.", "voice": "fake"}))
    assert engine.languages == [None], engine.languages


def test_an_explicit_language_passes_through_verbatim(tmp_path) -> None:
    engine = _RecordingEngine()
    server = _server(tmp_path, engine=engine)
    _run(
        server.call_tool("synthesize", {"text": "one. two.", "voice": "fake", "language": "pl"})
    )
    assert engine.languages == ["pl"], engine.languages


def test_a_backend_stub_is_not_reported_as_the_agents_mistake(tmp_path) -> None:
    """The MCP half of the 400-vs-500 distinction.

    The tool caught the builtin `NotImplementedError` and returned it as
    `{"error": ...}` — the same shape as "you asked for Chinese". An agent
    reading that learns its request was wrong, so it rewrites a request that
    was never the problem, while the broken build stays invisible to the person
    who could fix it.

    Only `UnsupportedLanguageError` is an answer now. A bare
    `NotImplementedError` escapes to the framework's failure path, which is
    what a server fault looks like on this transport.
    """

    from mcp.server.mcpserver.exceptions import ToolError

    class _StubFrontend:
        def encode(self, text: str, language: str = "en") -> np.ndarray:
            raise NotImplementedError("mel decoder for this backend is a stub")

    inner = _engine()
    server = _server(
        tmp_path,
        engine=Engine(
            frontend=_StubFrontend(),
            token_generator=inner.token_generator,
            mel_decoder=inner.mel_decoder,
            vocoder=inner.vocoder,
            algorithm=inner.algorithm,
        ),
    )
    with pytest.raises((ToolError, NotImplementedError)):
        _run(server.call_tool("synthesize", {"text": "hello.", "voice": "fake"}))

    # And the refusal that *is* the caller's stays an answer, with the kind and
    # the alternatives an agent needs to retry into something that works.
    result = _run(server.call_tool("synthesize", {"text": "hello.", "voice": "nope"}))
    payload = json.loads(result.content[0].text)
    assert payload["error_kind"] == "bad_request"
    assert payload["available"] == ["fake"]


def test_synthesize_hands_back_a_continuation_and_takes_it_again(tmp_path) -> None:
    """An agent reading a chapter in pieces should not have to know what a
    speech token is — it copies one field from the last reply into the next
    call, and the join stops being audible.

    Returned as the tail rather than every id: a few hundred integers in a tool
    result is context the agent pays for and cannot act on.
    """
    server = _server(tmp_path)
    first = _run(
        server.call_tool("synthesize", {"text": "Part one.", "voice": "fake", "seed": 1})
    )
    tail = json.loads(first.content[0].text)["continuation"]
    assert tail
    assert all(isinstance(t, int) for t in tail)
    # At most the prefix length — a short utterance simply has fewer tokens
    # than the recipe would carry, and the tail is what exists.
    assert len(tail) <= AlgorithmConfig().chunking.prefix_tokens

    second = _run(
        server.call_tool(
            "synthesize",
            {"text": "Part two.", "voice": "fake", "seed": 2, "previous_tokens": tail},
        )
    )
    assert "audio" in json.loads(second.content[0].text)


def test_synthesize_refuses_a_token_outside_the_codebook(tmp_path) -> None:
    """A bad id is a question about the call, so it comes back as a
    bad_request rather than as a transport failure."""
    server = _server(tmp_path)
    result = _run(
        server.call_tool(
            "synthesize", {"text": "hi", "voice": "fake", "previous_tokens": [10**9]}
        )
    )
    payload = json.loads(result.content[0].text)
    assert payload["error_kind"] == "bad_request"
    assert "acoustic speech token" in payload["error"]


def test_synthesize_speed_reaches_the_engine(tmp_path) -> None:
    server = _server(tmp_path)
    plain = json.loads(
        _run(_server(tmp_path).call_tool("synthesize", {"text": "one. two.", "voice": "fake"}))
        .content[0]
        .text
    )
    fast = json.loads(
        _run(
            server.call_tool("synthesize", {"text": "one. two.", "voice": "fake", "speed": 2.0})
        )
        .content[0]
        .text
    )
    assert fast["duration"] == pytest.approx(plain["duration"] / 2, rel=0.02)


def test_synthesize_refuses_a_speed_outside_the_range_as_bad_request(tmp_path) -> None:
    """The tool description promises `bad_request` for something about the
    call to fix. A speed outside the stretcher's range is exactly that, and it
    used to escape as a bare ValueError and an `isError` reply with no shape."""

    from loudkit.models.timestretch import MAX_SPEED, MIN_SPEED

    server = _server(tmp_path)
    result = _run(
        server.call_tool("synthesize", {"text": "one. two.", "voice": "fake", "speed": 3.0})
    )
    payload = json.loads(result.content[0].text)
    assert payload["error_kind"] == "bad_request"
    assert "speed" in payload["error"]
    assert str(MIN_SPEED) in payload["error"]
    assert str(MAX_SPEED) in payload["error"]


_REFUSALS = [
    # One entry per refusal `synthesize` can return without a purpose-built
    # engine, each with the catalog code the other two doors answer the same
    # request with. A refusal added without a code fails here rather than
    # reaching an agent that cannot branch on it.
    pytest.param({"text": "a" * 10_001, "voice": "fake"}, "invalid_request", id="over-cap"),
    pytest.param(
        {"text": "hi", "voice": "fake", "format": "mp3"}, "invalid_request", id="format"
    ),
    pytest.param({"text": "hi", "voice": "fake", "speed": 3.0}, "invalid_request", id="speed"),
    pytest.param({"text": "hi", "voice": "nope"}, "voice_not_found", id="unknown-voice"),
    pytest.param({"text": "hi", "voice": "a/b"}, "invalid_request", id="not-a-voice-name"),
    pytest.param(
        {"text": "hi", "voice": "fake", "previous_tokens": [10**9]},
        "invalid_tokens",
        id="renderer-refusal",
    ),
]


@pytest.mark.parametrize(("arguments", "code"), _REFUSALS)
def test_every_refusal_names_its_condition_from_the_catalog(tmp_path, arguments, code) -> None:
    """One vocabulary across three doors, on every refusal and not just one.

    HTTP puts `code` in every error body and gRPC sends `loudkit-error-code`
    on every failed call. Here only the cap refusal carried one, so an agent
    that had learned to branch on `code` had to fall back to matching the
    English of `error` for everything else, which is the string this project
    is free to reword.
    """
    result = _run(_server(tmp_path).call_tool("synthesize", arguments))
    payload = json.loads(result.content[0].text)
    assert payload["error_kind"] == "bad_request", payload
    assert payload["code"] == code, payload


def test_an_unsupported_language_names_its_condition_too(tmp_path) -> None:
    """The seventh refusal, which needs a frontend that refuses.

    Same claim as the parametrised set above; it is separate only because
    reaching it takes an engine built for it.
    """
    from loudkit.errors import UnsupportedLanguageError

    class _RefusingFrontend:
        def encode(self, text: str, language: str = "en") -> np.ndarray:
            raise UnsupportedLanguageError(
                f"language {language!r} needs model-based text preprocessing",
                language=language,
                supported=("en", "pl"),
            )

    inner = _engine()
    server = _server(
        tmp_path,
        engine=Engine(
            frontend=_RefusingFrontend(),
            token_generator=inner.token_generator,
            mel_decoder=inner.mel_decoder,
            vocoder=inner.vocoder,
            algorithm=inner.algorithm,
        ),
    )
    result = _run(
        server.call_tool("synthesize", {"text": "你好。", "voice": "fake", "language": "zh"})
    )
    payload = json.loads(result.content[0].text)
    assert payload["code"] == "unsupported_language", payload


def test_a_busy_engine_names_its_condition_too(tmp_path, monkeypatch) -> None:
    """The eighth refusal: the wait bound expired, so no render was attempted.

    `busy` is the one refusal here that is not the caller's fault, and the one
    an agent most needs to tell apart by code: the answer is to retry the
    identical request, where every other refusal means change it. HTTP answers
    a full queue with `busy` out of `_CODE_BY_STATUS[503]` and gRPC sends the
    same word beside RESOURCE_EXHAUSTED.

    Two threads and two events rather than a sleep: the second call has to
    land while the first is inside the render holding the lock, and a duration
    cannot state that.
    """
    import threading

    import loudkit.transports.mcp as mcp_mod

    inside = threading.Event()
    let_go = threading.Event()
    inner = _engine()

    class _BlockingVocoder:
        def __init__(self, config: AlgorithmConfig) -> None:
            self.config = config

        def synthesize(self, mel: Mel, voice: VoiceProfile, *, seed: int) -> Waveform:
            inside.set()
            assert let_go.wait(10), "the second call never answered"
            return np.zeros(mel.shape[1] * 256, np.float32)

    server = _server(
        tmp_path,
        engine=Engine(
            frontend=inner.frontend,
            token_generator=inner.token_generator,
            mel_decoder=inner.mel_decoder,
            vocoder=_BlockingVocoder(inner.algorithm),
            algorithm=inner.algorithm,
        ),
    )
    # Read off the module at call time, so the waiter gives up while the
    # holder is still inside the render rather than after a real minute.
    monkeypatch.setattr(mcp_mod, "_MAX_WAIT_S", 0.05)

    held = threading.Thread(
        target=lambda: _run(
            server.call_tool("synthesize", {"text": "one. two.", "voice": "fake"})
        ),
        daemon=True,
    )
    held.start()
    try:
        assert inside.wait(10), "the first call never reached the engine"
        result = _run(server.call_tool("synthesize", {"text": "three.", "voice": "fake"}))
    finally:
        let_go.set()
        held.join(10)
    payload = json.loads(result.content[0].text)
    assert payload["error_kind"] == "busy", payload
    assert payload["code"] == "busy", payload


def test_a_saturated_engine_still_answers_list_voices(tmp_path) -> None:
    """The depth bound, which the wait bound cannot stand in for.

    A waiter here holds a worker thread from the SDK's pool for as long as it
    waits, so a queue deeper than the pool parks every thread on the engine
    lock and the tools that need no engine stop answering. `list_voices` reads
    a directory. Raising `_MAX_WAIT_S` does not help it; only refusing past a
    depth below the pool size does, which is the same arithmetic gRPC applies
    to its own pool.

    Asserted as liveness, not latency. Every admitted render is held until this
    test lets it go, so a `list_voices` that comes back at all proves a thread
    was free for it; nothing here is a number a loaded runner can move.
    """
    import anyio.to_thread

    from loudkit.transports.limits import _queue_depth_for

    held = threading.Event()
    inner = _engine()

    class _HeldVocoder:
        def __init__(self, config: AlgorithmConfig) -> None:
            self.config = config

        def synthesize(self, mel: Mel, voice: VoiceProfile, *, seed: int) -> Waveform:
            assert held.wait(60), "a render was never released"
            return np.zeros(mel.shape[1] * 256, np.float32)

    server = _server(
        tmp_path,
        engine=Engine(
            frontend=inner.frontend,
            token_generator=inner.token_generator,
            mel_decoder=inner.mel_decoder,
            vocoder=_HeldVocoder(inner.algorithm),
            algorithm=inner.algorithm,
        ),
    )

    async def drive() -> None:
        pool = anyio.to_thread.current_default_thread_limiter().total_tokens
        admitted = _queue_depth_for(pool)
        # More callers than the pool has threads, which is the state the bound
        # exists for: without it every thread is on the lock.
        callers = int(pool) + 20
        async with asyncio.TaskGroup() as group:
            calls = [
                group.create_task(
                    server.call_tool("synthesize", {"text": "one. two.", "voice": "fake"})
                )
                for _ in range(callers)
            ]
            try:
                # The refused ones answer without taking a thread, so they are
                # the first to finish and waiting for them is waiting for the
                # bound to engage rather than for a duration.
                refused = []
                pending = set(calls)
                while len(refused) < callers - admitted:
                    done, pending = await asyncio.wait(
                        pending, timeout=60, return_when=asyncio.FIRST_COMPLETED
                    )
                    assert done, "no call was refused; the depth bound is not engaged"
                    refused += [json.loads(c.result().content[0].text) for c in done]

                names = await asyncio.wait_for(server.call_tool("list_voices", {}), timeout=60)
            finally:
                held.set()

        assert len(refused) == callers - admitted, (
            f"{len(refused)} of {callers} callers were refused; {admitted} may wait"
        )
        for payload in refused:
            assert payload["error_kind"] == "busy", payload
            assert payload["code"] == "busy", payload
        assert "fake" in names.content[0].text

    _run(drive())


class _BlockingGenerator(FakeGenerator):
    """Holds inside the decode loop until the cancel flag arrives.

    A render that finished before the cancel would prove nothing: the flag has
    to be observed from inside the pass the host asked to stop.
    """

    def __init__(self, config: AlgorithmConfig) -> None:
        super().__init__(config)
        self.entered = threading.Event()
        self.saw_cancel = threading.Event()

    def generate(
        self,
        text_tokens,
        voice,
        *,
        sampler,
        max_new_tokens=None,
        prefix=(),
        should_cancel=None,
    ):
        if self.saw_cancel.is_set():
            # The cancel has been proved; later calls in the same test are
            # ordinary renders, and blocking them would only cost the suite ten
            # seconds each.
            return super().generate(
                text_tokens,
                voice,
                sampler=sampler,
                max_new_tokens=max_new_tokens,
                prefix=prefix,
            )
        self.entered.set()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if should_cancel is not None and should_cancel():
                self.saw_cancel.set()
                raise CancelledError("cancelled: should_cancel returned true")
            time.sleep(0.005)
        # The flag never came. Fall through so the test fails on its assert
        # rather than hanging here.
        return super().generate(
            text_tokens, voice, sampler=sampler, max_new_tokens=max_new_tokens, prefix=prefix
        )


def test_a_cancelled_tool_call_stops_the_render(tmp_path) -> None:
    """The other two doors pass a `should_cancel`; this one rendered the whole
    passage and threw the audio away, holding the engine lock while it did.

    A host cancels by sending `notifications/cancelled`, which the SDK applies
    by cancelling the handler's scope, so the tool has to be a coroutine with
    the render on a thread the framework can leave behind. Cancelling the
    scope here is what that notification does.
    """
    import anyio

    algo = AlgorithmConfig()
    generator = _BlockingGenerator(algo)
    engine = Engine(
        frontend=_SplitFrontend(),
        token_generator=generator,
        mel_decoder=FakeMelDecoder(algo),
        vocoder=FakeVocoder(algo),
        algorithm=algo,
    )
    server = _server(tmp_path, engine)

    async def drive() -> None:
        async with anyio.create_task_group() as tg:

            async def call() -> None:
                await server.call_tool("synthesize", {"text": "one two", "voice": "fake"})

            tg.start_soon(call)
            await anyio.to_thread.run_sync(lambda: generator.entered.wait(10))
            tg.cancel_scope.cancel()

    anyio.run(drive)

    assert generator.entered.is_set(), "the render never reached the generator"
    assert generator.saw_cancel.wait(10), "the cancel never reached the decode loop"

    # And the lock came back: the abandoned render released it on its way out,
    # so the next call is served rather than answered `busy`.
    result = _run(server.call_tool("synthesize", {"text": "three.", "voice": "fake"}))
    assert "audio" in json.loads(result.content[0].text)
