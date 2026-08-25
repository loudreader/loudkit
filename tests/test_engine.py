"""The engine's own logic, exercised without weights.

These exist because of a specific embarrassment: ``synthesize_long`` was written
and committed in a state where it could not run at all — it referenced ``np`` and
``Iterator`` without importing them. Every test passed, because every test that
would have called it needs a 1.27 GB checkpoint and skips without one.

So the engine gets fakes. They compute nothing meaningful; they exist so the
sequencing, the seeding, the chunk stitching and the refusal paths are executed
by a suite that runs anywhere, in two seconds, with no assets. A component's
arithmetic is somebody else's test.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from loudkit.config import AlgorithmConfig, ChunkConfig, SamplingConfig
from loudkit.contracts import Mel, Sampler, SpeechTokens, Waveform
from loudkit.engine import Engine
from loudkit.errors import InvalidTokensError
from loudkit.voice import VoiceProfile


def _voice(language: str = "en") -> VoiceProfile:
    return VoiceProfile(
        name="fake",
        speaker_embedding=np.full(256, 0.0625, np.float32),
        flow_embedding=np.full(192, 0.0625, np.float32),
        prompt_tokens=np.zeros(8, np.int64),
        prompt_mel=np.zeros((80, 16), np.float32),
        cond_prompt_tokens=np.zeros(8, np.int64),
        language=language,
    )


class FakeFrontend:
    """Records the language it was asked for; that is the whole assertion in
    ``TestLanguageComesFromTheVoice``."""

    def __init__(self) -> None:
        self.languages: list[str] = []

    def encode(self, text: str, language: str = "en") -> NDArray[np.int64]:
        self.languages.append(language)
        return np.arange(len(text.split()), dtype=np.int64)


class FakeGenerator:
    """Emits a token per input word, then stops. Records what it was given."""

    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        text_tokens: NDArray[np.int64],
        voice: VoiceProfile,
        *,
        sampler: Sampler,
        max_new_tokens: int | None = None,
        prefix: SpeechTokens = (),
        should_cancel=None,
    ) -> SpeechTokens:
        self.calls.append({"n_text": len(text_tokens), "prefix": list(prefix)})
        n = max(1, len(text_tokens))
        return [*range(n), self.config.stop_speech_token]

    def teacher_forced_logits(
        self, text_tokens: NDArray[np.int64], voice: VoiceProfile, forced: SpeechTokens
    ) -> NDArray[np.float32]:
        return np.zeros((len(forced) + 1, self.config.speech_vocab_size), np.float32)


class FakeMelDecoder:
    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config
        self.seeds: list[int] = []

    def decode(self, tokens: SpeechTokens, voice: VoiceProfile, *, seed: int) -> Mel:
        self.seeds.append(seed)
        return np.full((80, max(1, len(tokens)) * 2), float(seed % 97), np.float32)


class FakeVocoder:
    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config
        self.seeds: list[int] = []

    def synthesize(self, mel: Mel, voice: VoiceProfile, *, seed: int) -> Waveform:
        self.seeds.append(seed)
        return np.zeros(mel.shape[1] * 256, np.float32)


def _engine(algorithm: AlgorithmConfig | None = None) -> Engine:
    algo = algorithm or AlgorithmConfig()
    return Engine(
        frontend=FakeFrontend(),
        token_generator=FakeGenerator(algo),
        mel_decoder=FakeMelDecoder(algo),
        vocoder=FakeVocoder(algo),
        algorithm=algo,
    )


class TestOneAlgorithm:
    def test_mismatched_component_is_rejected(self) -> None:
        """The only enforcement this library has. It must fire."""
        algo = AlgorithmConfig()
        other = algo.with_(euler_steps=4)
        with pytest.raises(ValueError, match="different algorithm config"):
            Engine(
                frontend=FakeFrontend(),
                token_generator=FakeGenerator(other),
                mel_decoder=FakeMelDecoder(algo),
                vocoder=FakeVocoder(algo),
                algorithm=algo,
            )

    def test_the_error_names_both_configs(self) -> None:
        algo = AlgorithmConfig()
        with pytest.raises(ValueError) as exc:
            Engine(
                frontend=FakeFrontend(),
                token_generator=FakeGenerator(
                    algo.with_(guidance="cfg_dual_path", guidance_rate=0.7)
                ),
                mel_decoder=FakeMelDecoder(algo),
                vocoder=FakeVocoder(algo),
                algorithm=algo,
            )
        message = str(exc.value)
        assert "single_path" in message, "the error should name the engine's mode"
        assert "cfg@0.7" in message, "the error should name the component's mode"

    def test_engine_is_frozen(self) -> None:
        """A mutable engine could be re-pointed after the check ran."""
        engine = _engine()
        with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError
            engine.algorithm = AlgorithmConfig(euler_steps=3)  # type: ignore[misc]


class TestSynthesize:
    def test_produces_audio_and_intermediates(self) -> None:
        engine = _engine()
        result = engine.synthesize("one two three", _voice(), seed=7)
        assert result.audio.size > 0
        assert len(result.tokens) == 3  # stop token stripped
        assert result.mel.shape[0] == 80
        assert result.algorithm_fingerprint == engine.algorithm.fingerprint()

    def test_stages_get_different_seeds(self) -> None:
        """Shared seeds would let a change in one stage's consumption shift
        another stage's stream."""
        engine = _engine()
        engine.synthesize("one two", _voice(), seed=5)
        mel_seed = engine.mel_decoder.seeds[0]  # type: ignore[attr-defined]
        voc_seed = engine.vocoder.seeds[0]  # type: ignore[attr-defined]
        assert mel_seed != voc_seed != 5

    def test_same_seed_same_result(self) -> None:
        a = _engine().synthesize("one two three", _voice(), seed=3)
        b = _engine().synthesize("one two three", _voice(), seed=3)
        np.testing.assert_array_equal(a.audio, b.audio)
        assert list(a.tokens) == list(b.tokens)

    def test_over_window_is_loud(self) -> None:
        """Silent truncation in a reading tool means text vanishes while the
        audio still sounds fine.

        The chunk and sampling budgets are narrowed alongside the window
        because ``AlgorithmConfig`` now refuses a config whose budgets exceed
        it — a generator allowed to produce more speech than the renderer
        accepts fails *after* generating, and on the streaming path after
        delivering. The refusal under test here is the runtime one: this fake
        generator ignores its cap, which is exactly the backend bug the window
        check exists to catch.
        """
        algo = AlgorithmConfig()
        algo = algo.with_(
            window=type(algo.window)(max_speech_tokens=4),
            chunking=replace(algo.chunking, max_tokens=4, prefix_tokens=0),
            sampling=replace(algo.sampling, max_new_tokens=4),
        )
        engine = _engine(algo)
        with pytest.raises(ValueError, match="exceed the 4-token window"):
            engine.synthesize("a b c d e f g h", _voice(), seed=1)

    def test_a_config_whose_budgets_outrun_the_window_does_not_load(self) -> None:
        """The failure must arrive at the door, not mid-passage.

        ``chunking.max_tokens``, ``sampling.max_new_tokens`` and
        ``window.max_speech_tokens`` are three independent manifest blocks with
        three independent validators, so nothing asked whether they agreed. A
        combination where they do not is not a bad utterance — it is a config
        that *guarantees* every long passage dies after audio has already been
        delivered and played, under a fingerprint that faithfully records the
        broken recipe.
        """
        algo = AlgorithmConfig()
        window = type(algo.window)(max_speech_tokens=8, static_length=8)

        with pytest.raises(ValueError, match="chunking.max_tokens"):
            algo.with_(window=window, chunking=replace(algo.chunking, max_tokens=64))
        with pytest.raises(ValueError, match="max_new_tokens"):
            algo.with_(
                window=window,
                chunking=replace(algo.chunking, max_tokens=8, prefix_tokens=0),
                sampling=replace(algo.sampling, max_new_tokens=64),
            )


class TestLanguageComesFromTheVoice:
    """The obvious call must not be the wrong one.

    ``engine.synthesize("Cześć", polish_voice)`` used to run Polish text through
    the English frontend, because ``language`` defaulted to ``"en"`` and a
    profile's own ``language`` — recorded at enrollment — was never consulted.
    The chain is now argument, then voice, then ``"en"``, and these three tests
    are the three links.

    Asserted at the frontend rather than on the audio: the fakes compute
    nothing, so the language id reaching ``encode`` *is* the observable
    behaviour, and it is the one thing every downstream stage keys off.
    """

    def test_a_polish_voice_reads_polish_by_default(self) -> None:
        engine = _engine()
        result = engine.synthesize("jeden dwa", _voice(language="pl"), seed=1)
        assert engine.frontend.languages == ["pl"]  # type: ignore[attr-defined]
        assert result.language == "pl"

    def test_an_explicit_language_overrides_the_profile(self) -> None:
        """Cross-lingual synthesis: a Polish voice reading English text."""
        engine = _engine()
        result = engine.synthesize("one two", _voice(language="pl"), seed=1, language="en")
        assert engine.frontend.languages == ["en"]  # type: ignore[attr-defined]
        assert result.language == "en"

    def test_a_profile_without_a_language_falls_back_to_english(self) -> None:
        """A hand-built profile can carry an empty language, and an empty
        language id is not a language — it would tag the text ``[]``.

        Only hand-built or hand-edited ones: a *missing* header key loads as
        ``"en"``, so a file written before the field was read back inherits
        nothing rather than falling through here.
        """
        engine = _engine()
        result = engine.synthesize("one two", _voice(language=""), seed=1)
        assert engine.frontend.languages == ["en"]  # type: ignore[attr-defined]
        assert result.language == "en"

    def test_the_chain_reaches_the_streaming_path_too(self) -> None:
        """``stream`` resolves once, before splitting, so every chunk of a
        passage is read in the same language — and ``synthesize_long`` is
        ``stream`` concatenated, so it inherits the same resolution."""
        engine = _engine()
        engine.synthesize_long("jeden. dwa. trzy.", _voice(language="pl"), seed=1)
        assert set(engine.frontend.languages) == {"pl"}  # type: ignore[attr-defined]

        other = _engine()
        list(other.stream("one. two.", _voice(language="pl"), seed=1, language="de"))
        assert set(other.frontend.languages) == {"de"}  # type: ignore[attr-defined]


class TestStreamingAndLongForm:
    """The methods that were once committed in a state where they could not run.

    Nothing here checks audio quality. It checks that the code executes, that
    chunks are stitched in order, and that the prefix reaches the generator.
    """

    def _long_engine(self, prefix_tokens: int = 0) -> Engine:
        algo = AlgorithmConfig().with_(
            chunking=ChunkConfig(max_tokens=20, prefix_tokens=prefix_tokens),
            sampling=SamplingConfig(max_new_tokens=64),
        )
        return _engine(algo)

    def test_warm_renders_and_discards(self) -> None:
        """warm() must run every stage (or its first-use cost survives to the
        first request) and must not leak a Result."""
        engine = self._long_engine()
        assert engine.warm(_voice()) is None

    def test_pipelined_stream_is_byte_identical_to_the_serial_phases(self) -> None:
        """`stream` renders window k while generating k+1; the audio must not
        know that. This composes the two phases serially — the exact loop the
        pipeline replaced — and asserts hash equality against `stream`.

        With `prefix_tokens` non-zero, so the carry (the one value that crosses
        between windows) is exercised, not just the seeds.
        """
        import hashlib

        import numpy as np

        from loudkit.engine import _STREAM_CHUNK, _derive

        engine = self._long_engine(prefix_tokens=4)
        voice = _voice()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."

        from loudkit.frontend.chunking import split_text
        from loudkit.frontend.polish import speech_text

        chunks = split_text(speech_text(text, "en"), engine.algorithm.chunking)
        assert len(chunks) > 2, "the fixture must span several windows"
        prefix_len = engine.algorithm.chunking.prefix_tokens
        carry: list[int] = []
        serial = []
        for i, chunk in enumerate(chunks):
            window = engine._generate_window(
                chunk,
                voice,
                seed=_derive(1, _STREAM_CHUNK + i),
                language="en",
                prefix=carry,
                prepared=True,
                is_terminal=i == len(chunks) - 1,
            )
            assert window is not None
            if prefix_len:
                carry = list(window.speech[-prefix_len:])
            serial.append(engine._render_window(window, voice, speed=1.0))

        streamed = list(engine.stream(text, voice, seed=1))
        assert len(streamed) == len(serial)
        a = np.concatenate([p.audio for p in serial])
        b = np.concatenate([p.audio for p in streamed])
        assert (
            hashlib.sha256(a.tobytes()).hexdigest() == hashlib.sha256(b.tobytes()).hexdigest()
        )

    def test_the_hash_gate_detects_an_injected_fault(self) -> None:
        """Negative control for the equality test above: a gate that cannot
        fail is not measuring. One injected least-significant wobble in the
        render must change the hash. (Seeds are no use here — this suite's
        fake vocoder emits silence whatever the seed.)"""
        import hashlib

        import numpy as np

        engine = self._long_engine(prefix_tokens=4)
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        a = np.concatenate([p.audio for p in engine.stream(text, _voice(), seed=1)])

        original = engine._render_window

        def faulty(window, voice, *, speed, should_cancel=None):  # type: ignore[no-untyped-def]
            from dataclasses import replace as _replace

            result = original(window, voice, speed=speed, should_cancel=should_cancel)
            audio = result.audio.copy()
            audio[0] += np.float32(1e-6)
            return _replace(result, audio=audio)

        object.__setattr__(engine, "_render_window", faulty)
        b = np.concatenate([p.audio for p in engine.stream(text, _voice(), seed=1)])
        assert (
            hashlib.sha256(a.tobytes()).hexdigest() != hashlib.sha256(b.tobytes()).hexdigest()
        )

    def test_a_render_failure_reaches_the_consumer(self) -> None:
        """An exception in the render worker must surface from the generator,
        not hang the queue or die on a background thread."""
        import pytest

        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."

        original = engine._render_window
        calls = {"n": 0}

        def failing(window, voice, *, speed, should_cancel=None):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("vocoder exploded")
            return original(window, voice, speed=speed, should_cancel=should_cancel)

        object.__setattr__(engine, "_render_window", failing)
        with pytest.raises(RuntimeError, match="vocoder exploded"):
            list(engine.stream(text, _voice(), seed=1))

    def test_an_abandoned_stream_stops_the_producer(self) -> None:
        """A consumer that walks away after one chunk must not leave the
        producer speaking the rest of the passage to nobody."""
        import threading

        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        gen = engine.stream(text, _voice(), seed=1)
        next(gen)
        gen.close()
        # The producer thread is named; none may remain alive after close().
        assert not [t for t in threading.enumerate() if t.name.startswith("loudkit-generate")]

    def test_a_producer_that_outlives_the_join_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A wedged render keeps the producer alive past the join. The engine
        it still holds is the one the caller is about to reuse, so the timeout
        has to be said out loud rather than swallowed by the finally."""
        import logging
        import threading

        import loudkit.engine as engine_mod

        monkeypatch.setattr(engine_mod, "_PRODUCER_JOIN_TIMEOUT", 0.05)
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."

        release = threading.Event()
        running = threading.Event()
        original = engine._render_window
        calls = {"n": 0}

        def wedged(window, voice, *, speed, should_cancel=None):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] > 1:
                # Only the windows the abandoned consumer never asks for; the
                # first must return, or next(gen) below never yields.
                running.set()
                release.wait(30.0)
            return original(window, voice, speed=speed, should_cancel=should_cancel)

        object.__setattr__(engine, "_render_window", wedged)
        gen = engine.stream(text, _voice(), seed=1)
        try:
            next(gen)
            # Wait for the stuck render to be *running*, not merely queued.
            # A render still sitting in the pool's queue is cancelled by the
            # teardown and holds nothing, which is the whole point of the
            # cancellation there; only a thread already inside the renderer
            # can outlive the join, and that is what this asserts about.
            assert running.wait(30.0), "the second render never started"
            with caplog.at_level(logging.ERROR, logger="loudkit.engine"):
                gen.close()
            assert "loudkit-generate" in caplog.text
        finally:
            release.set()
            for thread in threading.enumerate():
                if thread.name.startswith("loudkit-generate"):
                    thread.join(timeout=30.0)

    def test_a_producer_that_outlives_the_join_makes_the_engine_unusable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reporting the stuck thread and carrying on is not enough.

        The producer that outlived the join is still inside the token generator
        and the renderer, so the next call does not fail — it contends, with a
        thread nobody is waiting on, and comes back slow or wrong depending on
        how much state the two share. Every public entry refuses instead, and
        the message names the cause rather than leaving it to be inferred from
        a log line nobody read.
        """
        import threading

        import numpy as np

        import loudkit.engine as engine_mod

        monkeypatch.setattr(engine_mod, "_PRODUCER_JOIN_TIMEOUT", 0.05)
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."

        release = threading.Event()
        running = threading.Event()
        original = engine._render_window
        calls = {"n": 0}

        def wedged(window, voice, *, speed, should_cancel=None):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] > 1:
                running.set()
                release.wait(30.0)
            return original(window, voice, speed=speed, should_cancel=should_cancel)

        object.__setattr__(engine, "_render_window", wedged)
        gen = engine.stream(text, _voice(), seed=1)
        try:
            next(gen)
            # Running, not merely queued: a queued render is cancelled by the
            # teardown, and only a thread already inside the renderer can
            # outlive the join.
            assert running.wait(30.0), "the second render never started"
            gen.close()

            voice = _voice()
            # Every entry, not only the streaming one: the stages a stuck
            # render holds are the same stages each of these walks into.
            for call in (
                lambda: engine.synthesize("Hello.", voice, seed=1),
                lambda: engine.synthesize_long(text, voice, seed=1),
                lambda: list(engine.stream(text, voice, seed=1)),
                lambda: engine.synthesize_tokens(np.zeros(4, np.int64), voice, seed=1),
                lambda: engine.warm(voice),
            ):
                with pytest.raises(RuntimeError, match="unusable"):
                    call()
        finally:
            release.set()
            for thread in threading.enumerate():
                if thread.name.startswith("loudkit-generate"):
                    thread.join(timeout=30.0)

    def test_a_wedged_engine_does_not_wedge_the_next_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal is one engine's fate, not the class's. A long-running
        process replaces the engine and keeps serving; that is the whole
        remedy, and it has to work."""
        import threading

        import loudkit.engine as engine_mod

        monkeypatch.setattr(engine_mod, "_PRODUCER_JOIN_TIMEOUT", 0.05)
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."

        release = threading.Event()
        running = threading.Event()
        original = engine._render_window
        calls = {"n": 0}

        def wedged(window, voice, *, speed, should_cancel=None):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] > 1:
                running.set()
                release.wait(30.0)
            return original(window, voice, speed=speed, should_cancel=should_cancel)

        object.__setattr__(engine, "_render_window", wedged)
        gen = engine.stream(text, _voice(), seed=1)
        try:
            next(gen)
            # Running, not merely queued: a queued render is cancelled by the
            # teardown, and only a thread already inside the renderer can
            # outlive the join.
            assert running.wait(30.0), "the second render never started"
            gen.close()
            assert self._long_engine().synthesize("Hello.", _voice(), seed=1).duration > 0
        finally:
            release.set()
            for thread in threading.enumerate():
                if thread.name.startswith("loudkit-generate"):
                    thread.join(timeout=30.0)

    def test_a_slow_render_is_not_a_wedged_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wedging is for a thread that outlived its join, not a slow renderer.

        An abandoned stream leaves the pipeline holding renders the consumer
        will never read: one inside the vocoder and up to ``_PIPELINE_DEPTH``
        more queued behind it. ``ThreadPoolExecutor.__exit__`` drains its own
        queue, so every one of those used to run to completion before the
        producer could exit, and the join was really waiting on ``depth x
        render`` rather than on one render. A renderer merely slower than that
        arithmetic was then reported as a thread that would never come back,
        and the engine refused every call after it for the life of the process.

        Scaled down so the numbers fit a test: renders of ``slow_s`` against a
        join of ``join_s``, with ``join_s`` comfortably longer than one render
        and shorter than three. Nothing here is stuck; the only question is
        whether the teardown takes back the work nobody asked for.
        """
        import threading
        import time

        import loudkit.engine as engine_mod

        slow_s = 0.2
        join_s = 0.5

        monkeypatch.setattr(engine_mod, "_PRODUCER_JOIN_TIMEOUT", join_s)
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."

        original = engine._render_window
        calls = {"n": 0}

        def slow(window, voice, *, speed, should_cancel=None):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] > 1:
                # The first must return promptly or next(gen) never yields.
                # Every one after it is a render the abandoned consumer will
                # not read, which is exactly the work the teardown has to drop.
                time.sleep(slow_s)
            return original(window, voice, speed=speed, should_cancel=should_cancel)

        generated = threading.Semaphore(0)
        original_generate = engine._generate_window

        def counting(*args, **kwargs):  # type: ignore[no-untyped-def]
            window = original_generate(*args, **kwargs)
            generated.release()
            return window

        object.__setattr__(engine, "_render_window", slow)
        object.__setattr__(engine, "_generate_window", counting)

        gen = engine.stream(text, _voice(), seed=1)
        next(gen)
        # Four windows generated means three renders were handed to the pool
        # after the one already delivered: one running and two queued behind
        # it. Waiting for that rather than sleeping is what makes the pipeline
        # provably full at the moment the consumer walks away.
        for _ in range(4):
            assert generated.acquire(timeout=30.0), "the producer never filled the pipeline"
        started = time.monotonic()
        gen.close()
        elapsed = time.monotonic() - started
        after_close = calls["n"]

        assert engine._wedged is None, engine._wedged
        assert elapsed < join_s, (
            f"the teardown took {elapsed:.2f}s, longer than the {join_s:.2f}s join — "
            "it waited for renders nobody was going to read"
        )
        # And it really is the renders that were dropped, not the clock that
        # was generous: nothing more may start once the consumer has gone.
        time.sleep(slow_s * 2)
        assert calls["n"] == after_close, (
            f"{calls['n'] - after_close} more renders ran after the stream closed"
        )
        assert engine.synthesize("Hello.", _voice(), seed=1).duration > 0

    def test_a_healthy_engine_is_never_wedged(self) -> None:
        """A stream that drains normally leaves the engine usable, which is the
        case that must not be caught by the guard above."""
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        assert list(engine.stream(text, _voice(), seed=1))
        assert engine._wedged is None
        assert engine.synthesize("Hello.", _voice(), seed=1).duration > 0

    def test_stream_yields_one_result_per_chunk(self) -> None:
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        parts = list(engine.stream(text, _voice(), seed=1))
        assert len(parts) > 1
        assert all(p.audio.size > 0 for p in parts)

    def test_synthesize_long_runs_and_concatenates(self) -> None:
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        streamed = list(engine.stream(text, _voice(), seed=1))
        joined = engine.synthesize_long(text, _voice(), seed=1)
        assert joined.audio.size == sum(p.audio.size for p in streamed)
        assert list(joined.tokens) == [t for p in streamed for t in p.tokens]
        assert joined.mel.shape[1] == sum(p.mel.shape[1] for p in streamed)

    def test_short_text_takes_the_single_chunk_path(self) -> None:
        engine = self._long_engine()
        result = engine.synthesize_long("Two words.", _voice(), seed=1)
        assert result.audio.size > 0

    def test_prefix_reaches_the_generator(self) -> None:
        """Chunks generated independently restart their pitch contour, which is
        audible at every join."""
        engine = self._long_engine(prefix_tokens=2)
        list(engine.stream("One. Two. Three. Four. Five. Six.", _voice(), seed=1))
        calls = engine.token_generator.calls  # type: ignore[attr-defined]
        assert len(calls) > 1
        assert calls[0]["prefix"] == []
        assert len(calls[1]["prefix"]) == 2  # type: ignore[arg-type]

    def test_no_prefix_when_disabled(self) -> None:
        engine = self._long_engine(prefix_tokens=0)
        list(engine.stream("One. Two. Three. Four. Five. Six.", _voice(), seed=1))
        calls = engine.token_generator.calls  # type: ignore[attr-defined]
        assert all(c["prefix"] == [] for c in calls)

    def test_stream_barge_in_stops_between_chunks(self) -> None:
        """A voice agent interrupts mid-speech: should_cancel must stop the
        stream at the next chunk boundary as well as inside a chunk.

        This one measures the boundary: the callback only turns true once two
        chunks have been yielded, so nothing is cancelled mid-generation and
        the loop must simply stop asking for more. The mid-generation half —
        where the partial chunk is discarded rather than rendered — is
        `test_a_cancelled_chunk_is_never_rendered` below.
        """
        engine = self._long_engine()

        yielded = []
        for part in engine.stream(
            "One. Two. Three. Four. Five. Six.",
            _voice(),
            seed=1,
            should_cancel=lambda: len(yielded) >= 2,
        ):
            yielded.append(part)
        assert len(yielded) == 2, f"expected 2 chunks before cancel, got {len(yielded)}"

    def test_the_reported_seed_does_not_depend_on_the_chunk_count(self) -> None:
        """`Result.seed` is documented as what you need to reproduce the audio.

        The multi-chunk branch built a fresh `Result` with `seed=seed`; the
        single-chunk branch returned the chunk's own `Result` verbatim, and a
        chunk carries `derive(seed, 16 + index)`. So the same call reported two
        different numbers depending only on how the text happened to split —
        from the field `__repr__` prints and a user would quote in a bug report.
        """
        engine = self._long_engine()
        short = engine.synthesize_long("Two words.", _voice(), seed=7)
        long = engine.synthesize_long(
            "One. Two. Three. Four. Five. Six. Seven. Eight.", _voice(), seed=7
        )
        assert len(long.tokens) > len(short.tokens), "the long text must actually split"
        assert short.seed == 7, f"single-chunk long-form reported seed {short.seed}"
        assert long.seed == 7, f"multi-chunk long-form reported seed {long.seed}"

    def test_a_cancelled_chunk_is_never_rendered(self) -> None:
        """Half the barge-in cost is the renderer, and it is pure waste.

        When the interrupt lands mid-generation the partial tokens belong to
        speech the listener has already stopped wanting. Running them through
        the mel decoder and the vocoder anyway adds the whole render to the
        latency the cancellation exists to remove — on an edge device the
        larger half of it.
        """
        engine = self._long_engine()
        decoder = engine.mel_decoder
        vocoder = engine.vocoder
        decoder.calls = 0  # type: ignore[attr-defined]
        vocoder.calls = 0  # type: ignore[attr-defined]

        inner_decode = decoder.decode
        inner_synth = vocoder.synthesize

        def counting_decode(*a, **kw):  # type: ignore[no-untyped-def]
            decoder.calls += 1  # type: ignore[attr-defined]
            return inner_decode(*a, **kw)

        def counting_synth(*a, **kw):  # type: ignore[no-untyped-def]
            vocoder.calls += 1  # type: ignore[attr-defined]
            return inner_synth(*a, **kw)

        decoder.decode = counting_decode  # type: ignore[method-assign]
        vocoder.synthesize = counting_synth  # type: ignore[method-assign]

        # Cancel from the very first poll: generation stops immediately and
        # nothing should reach the renderer at all.
        parts = list(
            engine.stream(
                "One. Two. Three. Four.", _voice(), seed=1, should_cancel=lambda: True
            )
        )
        assert parts == []
        assert decoder.calls == 0, "a cancelled chunk was still decoded to mel"  # type: ignore[attr-defined]
        assert vocoder.calls == 0, "a cancelled chunk was still vocoded"  # type: ignore[attr-defined]

    def test_a_cancel_during_the_mel_decode_stops_before_the_vocoder(self) -> None:
        """A cancel that lands inside a render stage stops the next stage.

        A kernel already executing cannot be interrupted; what must not happen
        is the vocoder *starting* after the caller is gone. A deadline that
        expired during the mel decode used to buy the caller a full vocoder
        pass on an RPC that had already ended, measured over gRPC as
        DEADLINE_EXCEEDED with the vocoder starting after the cancel.
        """
        engine = self._long_engine()
        decoder = engine.mel_decoder
        vocoder = engine.vocoder
        vocoder.calls = 0  # type: ignore[attr-defined]
        cancelled = {"flag": False}

        inner_decode = decoder.decode
        inner_synth = vocoder.synthesize

        def decode_then_expire(*a, **kw):  # type: ignore[no-untyped-def]
            # The deadline expires while the mel decoder is running: the stage
            # completes (it cannot be preempted) and the flag is up by the
            # time the engine decides whether to start the vocoder.
            mel = inner_decode(*a, **kw)
            cancelled["flag"] = True
            return mel

        def counting_synth(*a, **kw):  # type: ignore[no-untyped-def]
            vocoder.calls += 1  # type: ignore[attr-defined]
            return inner_synth(*a, **kw)

        decoder.decode = decode_then_expire  # type: ignore[method-assign]
        vocoder.synthesize = counting_synth  # type: ignore[method-assign]

        parts = list(
            engine.stream(
                "Two words.", _voice(), seed=1, should_cancel=lambda: cancelled["flag"]
            )
        )
        assert parts == [], "a window cancelled mid-render was still yielded"
        assert vocoder.calls == 0, "the vocoder started after the cancel"  # type: ignore[attr-defined]

    def test_chunk_audio_does_not_depend_on_how_many_came_before(self) -> None:
        """Each chunk gets its own derived seed, so stopping a stream early
        cannot change what the earlier chunks were."""
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six."
        full = list(engine.stream(text, _voice(), seed=9))
        partial = []
        for i, part in enumerate(self._long_engine().stream(text, _voice(), seed=9)):
            partial.append(part)
            if i == 0:
                break
        np.testing.assert_array_equal(full[0].audio, partial[0].audio)

    def test_empty_text_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nothing to speak"):
            self._long_engine().synthesize_long("   ", _voice(), seed=1)

    def test_every_entry_point_refuses_empty_text_the_same_way(self) -> None:
        """One input, one answer, whichever door it arrives through.

        `stream` and `synthesize_long` raised; `synthesize` generated against an
        empty prompt and returned about a tenth of a second of near-silence. A
        caller batching titles, or reading a field that happened to be blank,
        got audio back and no reason to look twice — which is the failure a
        refusal exists to prevent.
        """
        engine = self._long_engine()
        for call in (
            lambda: engine.synthesize("   ", _voice(), seed=1),
            lambda: engine.synthesize_long("   ", _voice(), seed=1),
            lambda: list(engine.stream("   ", _voice(), seed=1)),
        ):
            with pytest.raises(ValueError, match="nothing to speak"):
                call()


class TestSynthesizeTokens:
    def test_renders_a_given_sequence(self) -> None:
        """The diagnostic that takes the generator out of a comparison."""
        engine = _engine()
        result = engine.synthesize_tokens([1, 2, 3], _voice(), seed=4)
        assert list(result.tokens) == [1, 2, 3]
        assert result.timings.tokens == 0.0


class CondemnedThenCleanGenerator(FakeGenerator):
    """Produces a dropout-shaped row (far fewer tokens than text) until the
    sampler's seed changes, then a healthy one — the retry ladder's happy path."""

    def generate(
        self, text_tokens, voice, *, sampler, max_new_tokens=None, prefix=(), should_cancel=None
    ):
        self.calls.append({"seed": sampler._seed if hasattr(sampler, "_seed") else None})
        n = len(text_tokens)
        if len(self.calls) == 1:
            # Dropout shape: 40 text tokens ask for speech; 5 arrive, ended.
            return [*range(5), self.config.stop_speech_token]
        return [*range(n * 2), self.config.stop_speech_token]


class AlwaysCondemnedGenerator(FakeGenerator):
    def generate(
        self, text_tokens, voice, *, sampler, max_new_tokens=None, prefix=(), should_cancel=None
    ):
        self.calls.append({})
        return [*range(5), self.config.stop_speech_token]


class TestSelectiveReroll:
    """A condemned window is regenerated from a derived seed; a clean one costs
    exactly one pass. The ladder is a pure function of the caller's seed."""

    def _engine_with(self, generator_cls):
        algo = AlgorithmConfig()
        gen = generator_cls(algo)
        engine = Engine(
            frontend=FakeFrontend(),
            token_generator=gen,
            mel_decoder=FakeMelDecoder(algo),
            vocoder=FakeVocoder(algo),
            algorithm=algo,
        )
        return engine, gen

    def test_a_clean_window_costs_one_pass(self) -> None:
        engine, gen = self._engine_with(FakeGenerator)
        engine.synthesize("one two three four five", _voice(), seed=7)
        assert len(gen.calls) == 1

    def test_a_dropout_window_is_rerolled_and_the_second_roll_ships(self) -> None:
        engine, gen = self._engine_with(CondemnedThenCleanGenerator)
        text = " ".join(["word"] * 40)
        result = engine.synthesize(text, _voice(), seed=7)
        assert len(gen.calls) == 2, "the condemned first roll must trigger exactly one retry"
        assert result.inspections[0].reason == "clean"

    def test_the_ladder_is_bounded(self) -> None:
        engine, gen = self._engine_with(AlwaysCondemnedGenerator)
        text = " ".join(["word"] * 40)
        result = engine.synthesize(text, _voice(), seed=7)
        assert len(gen.calls) == 1 + engine.algorithm.postprocess.retry_max_attempts
        # Nothing could be fixed; the last attempt ships, and the verdict says so.
        assert result.inspections[0].reason == "dropout"

    def test_zero_disables_retries(self) -> None:
        from loudkit.postprocess import PostprocessConfig

        algo = AlgorithmConfig(postprocess=PostprocessConfig(retry_max_attempts=0))
        gen = AlwaysCondemnedGenerator(algo)
        engine = Engine(
            frontend=FakeFrontend(),
            token_generator=gen,
            mel_decoder=FakeMelDecoder(algo),
            vocoder=FakeVocoder(algo),
            algorithm=algo,
        )
        engine.synthesize(" ".join(["word"] * 40), _voice(), seed=7)
        assert len(gen.calls) == 1

    def test_same_seed_same_ladder(self) -> None:
        a_engine, a_gen = self._engine_with(CondemnedThenCleanGenerator)
        b_engine, b_gen = self._engine_with(CondemnedThenCleanGenerator)
        text = " ".join(["word"] * 40)
        a = a_engine.synthesize(text, _voice(), seed=7)
        b = b_engine.synthesize(text, _voice(), seed=7)
        assert list(a.tokens) == list(b.tokens)
        assert len(a_gen.calls) == len(b_gen.calls)


class TestCrossRequestContext:
    """`previous_tokens` is the chunk prefix, exposed across calls.

    The assertions are on what reaches the generator, not on audio: the fakes
    compute nothing, so the conditioning context handed to `generate` *is* the
    observable behaviour — and it is the one thing every downstream stage keys
    off.
    """

    def _long_engine(self, prefix_tokens: int = 6) -> Engine:
        algo = AlgorithmConfig().with_(
            chunking=ChunkConfig(max_tokens=20, prefix_tokens=prefix_tokens),
            sampling=SamplingConfig(max_new_tokens=64),
        )
        return _engine(algo)

    def test_the_first_chunk_is_conditioned_on_the_tail(self) -> None:
        engine = self._long_engine(prefix_tokens=3)
        engine.synthesize("one two", _voice(), seed=1, previous_tokens=[10, 11, 12, 13, 14])
        calls = engine.token_generator.calls  # type: ignore[attr-defined]
        assert calls[0]["prefix"] == [12, 13, 14]

    def test_a_long_history_is_sliced_rather_than_refused(self) -> None:
        """Chaining should be `previous_tokens=result.tokens`, with no arithmetic
        at the call site: a caller who had to know the prefix length would be
        keeping a copy of an algorithm value."""
        engine = self._long_engine(prefix_tokens=2)
        engine.synthesize("one two", _voice(), seed=1, previous_tokens=list(range(200)))
        assert engine.token_generator.calls[0]["prefix"] == [198, 199]  # type: ignore[attr-defined]

    def test_none_is_todays_behaviour(self) -> None:
        engine = self._long_engine()
        engine.synthesize("one two", _voice(), seed=1)
        assert engine.token_generator.calls[0]["prefix"] == []  # type: ignore[attr-defined]

    def test_zero_prefix_tokens_means_no_context_not_all_of_it(self) -> None:
        """`tokens[-0:]` is the whole list. At the setting that means "chunks
        are independent", that would condition on the entire previous
        utterance — the exact opposite."""
        engine = self._long_engine(prefix_tokens=0)
        engine.synthesize("one two", _voice(), seed=1, previous_tokens=[1, 2, 3])
        assert engine.token_generator.calls[0]["prefix"] == []  # type: ignore[attr-defined]

    def test_only_the_first_chunk_takes_it(self) -> None:
        """Every chunk after the first is conditioned on the one before, as
        always — the caller's history seeds the carry, it does not replace it."""
        engine = self._long_engine(prefix_tokens=2)
        list(
            engine.stream(
                "One. Two. Three. Four. Five. Six.",
                _voice(),
                seed=1,
                previous_tokens=[90, 91],
            )
        )
        calls = engine.token_generator.calls  # type: ignore[attr-defined]
        assert len(calls) > 1
        assert calls[0]["prefix"] == [90, 91]
        assert calls[1]["prefix"] != [90, 91]
        assert len(calls[1]["prefix"]) == 2  # type: ignore[arg-type]

    def test_long_form_passes_it_to_the_first_chunk(self) -> None:
        engine = self._long_engine(prefix_tokens=2)
        engine.synthesize_long(
            "One. Two. Three. Four.", _voice(), seed=1, previous_tokens=[7, 8, 9]
        )
        assert engine.token_generator.calls[0]["prefix"] == [8, 9]  # type: ignore[attr-defined]

    def test_the_same_history_gives_the_same_bytes(self) -> None:
        """Determinism is the whole contract; a new input must not weaken it."""
        first = self._long_engine().synthesize(
            "one two three", _voice(), seed=5, previous_tokens=[3, 4, 5, 6, 7, 8]
        )
        second = self._long_engine().synthesize(
            "one two three", _voice(), seed=5, previous_tokens=[3, 4, 5, 6, 7, 8]
        )
        assert np.array_equal(first.audio, second.audio)
        assert list(first.tokens) == list(second.tokens)

    def test_a_token_outside_the_codebook_is_refused(self) -> None:
        """An id the renderer cannot look up would index off the embedding
        table. Named at the boundary rather than three stages in."""
        engine = self._long_engine()
        with pytest.raises(ValueError, match="not an acoustic speech token"):
            engine.synthesize(
                "one two",
                _voice(),
                seed=1,
                previous_tokens=[engine.algorithm.stop_speech_token],
            )
        with pytest.raises(ValueError, match="not an acoustic speech token"):
            engine.synthesize("one two", _voice(), seed=1, previous_tokens=[-1])

    def test_a_result_chains_straight_into_the_next_call(self) -> None:
        """The documented usage, end to end, with nothing in between."""
        engine = self._long_engine(prefix_tokens=3)
        first = engine.synthesize("one two three four", _voice(), seed=1)
        engine.synthesize("five six", _voice(), seed=2, previous_tokens=first.tokens)
        calls = engine.token_generator.calls  # type: ignore[attr-defined]
        assert calls[-1]["prefix"] == list(first.tokens[-3:])


class TestSynthesizeTokensBoundsWhatItIsHanded:
    """The public token entry, against ids no renderer can look up.

    `_strip_specials` drops ids at or above `start_speech_token` and says
    nothing about the bottom, so `-1` reached an embedding lookup: in torch a
    negative index reads from the *end* of the table and returns a plausible
    vector, on the ONNX path it is an out-of-bounds read. Either way the caller
    gets audio rather than an error.

    The check already existed and was wired into the streaming route only, for
    `previous_tokens`. A public render entry has to bound what it is handed
    rather than assume the caller derived it from a previous render.
    """

    @pytest.mark.parametrize("bad", [[-1], [0, -1, 1], [10**9], [-(10**9)]])
    def test_out_of_range_ids_are_refused(self, bad: list[int]) -> None:
        engine = _engine()
        with pytest.raises(InvalidTokensError, match="tokens contains"):
            engine.synthesize_tokens(np.array(bad, dtype=np.int64), _voice())

    def test_the_message_names_this_field_not_the_streaming_one(self) -> None:
        # It used to say "previous_tokens" whatever called it, which sends a
        # caller looking for a field they never passed.
        engine = _engine()
        with pytest.raises(InvalidTokensError) as caught:
            engine.synthesize_tokens(np.array([-1], dtype=np.int64), _voice())
        assert "previous_tokens" not in str(caught.value)
