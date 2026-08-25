"""The MCP server's surface, exercised without weights.

Same rule as the HTTP server's tests: the MCP server must hold no synthesis
path of its own, so these assert that its tools resolve through the engine and
the voice library — a fake engine injected via ``build_server(engine=...)`` —
and that tool *calls* return the shapes an agent host expects. The WAV bytes
come from :func:`~loudkit.server.render_bytes`, which the HTTP tests already
pin to the engine; what is new here is the transport, not the synthesis.

The ``mcp`` SDK is an extra, so the whole module skips when it is absent —
the same discipline the server's fastapi import uses.
"""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest

from loudkit.config import AlgorithmConfig
from loudkit.contracts import Mel, Sampler, SpeechTokens, Waveform
from loudkit.engine import Engine
from loudkit.voice import VoiceProfile

pytest.importorskip("mcp.server.mcpserver")

from loudkit.transports.mcp import build_server  # noqa: E402


class _SplitFrontend:
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


def _engine() -> Engine:
    algo = AlgorithmConfig()
    return Engine(
        frontend=_SplitFrontend(),
        token_generator=_FakeGenerator(algo),
        mel_decoder=_FakeMelDecoder(algo),
        vocoder=_FakeVocoder(algo),
        algorithm=algo,
    )


def _voice() -> VoiceProfile:
    return VoiceProfile(
        name="fake",
        speaker_embedding=np.full(256, 0.0625, np.float32),
        flow_embedding=np.full(192, 0.0625, np.float32),
        prompt_tokens=np.zeros(8, np.int64),
        prompt_mel=np.zeros((80, 16), np.float32),
        cond_prompt_tokens=np.zeros(8, np.int64),
    )


def _server(tmp_path, engine: Engine | None = None) -> object:
    voice = _voice()
    voice.save(tmp_path / "fake.safetensors")
    # Present but never read: an injected engine means nothing loads it, and the
    # voice directory is given explicitly below. It has to exist all the same,
    # because the checkpoint argument now goes through the hub whatever shape it
    # has, and the hub will not hand back a path to nothing. It sits outside the
    # voice directory, or the library would offer it as a voice called "unused".
    holder = tmp_path / "checkpoint"
    holder.mkdir(exist_ok=True)
    (holder / "unused.safetensors").write_bytes(b"x")
    return build_server(
        str(holder / "unused.safetensors"),
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
    import json

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


class _CappedGenerator(_FakeGenerator):
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
        mel_decoder=_FakeMelDecoder(algo),
        vocoder=_FakeVocoder(algo),
        algorithm=algo,
    )

    result = _run(
        _server(tmp_path, capped).call_tool(
            "synthesize", {"text": "one two three", "voice": "fake"}
        )
    )
    import json

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
    the sentence explaining which languages work — the CLI has printed it since
    30626c7.

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
        self, text, voice, *, seed=0, language=None, speed=1.0, previous_tokens=None
    ):
        self.languages.append(language)
        return self._inner.synthesize(text, voice, seed=seed, language=language)

    def synthesize_long(
        self, text, voice, *, seed=0, language=None, speed=1.0, previous_tokens=None
    ):
        self.languages.append(language)
        return self._inner.synthesize_long(text, voice, seed=seed, language=language)


def test_an_omitted_language_reaches_the_engine_as_none(tmp_path) -> None:
    """Omitting `language` must mean "the voice's own", over MCP as in process.

    If the tool substituted "en", an agent calling `synthesize(text, voice)`
    with a Polish voice would get an English read of Polish text, while the
    same call through the Python API got it right.
    """
    engine = _RecordingEngine()
    server = _server(tmp_path, engine=engine)  # type: ignore[arg-type]
    _run(server.call_tool("synthesize", {"text": "one. two.", "voice": "fake"}))
    assert engine.languages == [None], engine.languages


def test_an_explicit_language_passes_through_verbatim(tmp_path) -> None:
    engine = _RecordingEngine()
    server = _server(tmp_path, engine=engine)  # type: ignore[arg-type]
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
    import json

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
    import json

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
    import json

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
    import json

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
