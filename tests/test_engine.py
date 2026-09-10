"""The engine's own logic, exercised without weights.

These exist because of a specific embarrassment: the long-form path was written
and committed in a state where it could not run at all — it referenced ``np`` and
``Iterator`` without importing them. Every test passed, because every test that
would have called it needs a 1.27 GB checkpoint and skips without one.

So the engine gets fakes. They compute nothing meaningful; they exist so the
sequencing, the seeding, the chunk stitching and the refusal paths are executed
by a suite that runs anywhere, in two seconds, with no assets. A component's
arithmetic is somebody else's test.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from loudkit.config import AlgorithmConfig, ChunkConfig, SamplingConfig
from loudkit.engine import Engine
from loudkit.errors import InvalidTokensError

from .conftest import (
    FakeFrontend,
    FakeGenerator,
    FakeMelDecoder,
    FakeVocoder,
    fake_engine,
    fake_voice,
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
        engine = fake_engine()
        with pytest.raises(FrozenInstanceError):
            engine.algorithm = AlgorithmConfig(euler_steps=3)


class TestTheRateItRendersAt:
    """`token_rate_hz` is a manifest field; the geometry under it is not.

    A speech token is two mel frames of 480 samples in every backend, so at
    24 kHz the renderer produces 25 Hz and nothing else. A manifest free to
    declare 12.5 Hz would have the engine announce the 255-token window as
    20.4 s of speech and render 10.2 s of it, with a fingerprint that says the
    two engines are different and nothing that says which one is wrong.
    """

    def test_a_rate_the_renderer_cannot_produce_is_refused(self) -> None:
        with pytest.raises(ValueError, match="this renderer produces 25.0 Hz"):
            fake_engine(AlgorithmConfig(token_rate_hz=12.5))

    def test_a_sample_rate_that_moves_with_it_is_accepted(self) -> None:
        """The rule ties the two numbers together; it does not pin 24 kHz."""
        engine = fake_engine(AlgorithmConfig(sample_rate=12_000, token_rate_hz=12.5))
        assert engine.algorithm.token_rate_hz == 12.5

    def test_the_error_names_the_geometry_it_read(self) -> None:
        with pytest.raises(ValueError) as exc:
            fake_engine(AlgorithmConfig(sample_rate=16_000))
        message = str(exc.value)
        assert "2 mel frames of 480 samples at 16000 Hz" in message


class TestSynthesize:
    def test_produces_audio_and_intermediates(self) -> None:
        engine = fake_engine()
        result = engine.synthesize("one two three", fake_voice(), seed=7)
        assert result.audio.size > 0
        assert len(result.tokens) == 3  # stop token stripped
        assert result.mel.shape[0] == 80
        assert result.provenance.algorithm_fingerprint == engine.algorithm.fingerprint()

    def test_stages_get_different_seeds(self) -> None:
        """Shared seeds would let a change in one stage's consumption shift
        another stage's stream."""
        engine = fake_engine()
        engine.synthesize("one two", fake_voice(), seed=5)
        mel_seed = engine.mel_decoder.seeds[0]
        voc_seed = engine.vocoder.seeds[0]
        assert mel_seed != voc_seed != 5

    def test_same_seed_same_result(self) -> None:
        a = fake_engine().synthesize("one two three", fake_voice(), seed=3)
        b = fake_engine().synthesize("one two three", fake_voice(), seed=3)
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
        engine = fake_engine(algo)
        with pytest.raises(ValueError, match="exceed the 4-token window"):
            engine.synthesize("a b c d e f g h", fake_voice(), seed=1, single_window=True)

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
        engine = fake_engine()
        result = engine.synthesize("jeden dwa", fake_voice(language="pl"), seed=1)
        assert engine.frontend.languages == ["pl"]
        assert result.provenance.language == "pl"

    def test_an_explicit_language_overrides_the_profile(self) -> None:
        """Cross-lingual synthesis: a Polish voice reading English text."""
        engine = fake_engine()
        result = engine.synthesize("one two", fake_voice(language="pl"), seed=1, language="en")
        assert engine.frontend.languages == ["en"]
        assert result.provenance.language == "en"

    def test_a_profile_without_a_language_falls_back_to_english(self) -> None:
        """A hand-built profile can carry an empty language, and an empty
        language id is not a language — it would tag the text ``[]``.

        Only hand-built or hand-edited ones: a *missing* header key loads as
        ``"en"``, so a file written before the field was read back inherits
        nothing rather than falling through here.
        """
        engine = fake_engine()
        result = engine.synthesize("one two", fake_voice(language=""), seed=1)
        assert engine.frontend.languages == ["en"]
        assert result.provenance.language == "en"

    def test_the_chain_reaches_the_streaming_path_too(self) -> None:
        """``stream`` resolves once, before splitting, so every chunk of a
        passage is read in the same language — and ``synthesize`` is ``stream``
        concatenated, so it inherits the same resolution."""
        engine = fake_engine()
        engine.synthesize("jeden. dwa. trzy.", fake_voice(language="pl"), seed=1)
        assert set(engine.frontend.languages) == {"pl"}

        other = fake_engine()
        list(other.stream("one. two.", fake_voice(language="pl"), seed=1, language="de"))
        assert set(other.frontend.languages) == {"de"}


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
        return fake_engine(algo)

    def test_the_carry_starts_where_a_pair_starts_under_fusion_decode(self) -> None:
        """A join must hand the generator the pairing it generated.

        Under ``fusion_mtp2`` two tokens share one KV slot, and the generator
        re-pairs a prefix from the prefix's own start. A plain tail slice whose
        first token was the *second* half of a pair therefore fuses the right
        tokens with the wrong partners — silently, and only on the joins where
        the previous chunk ended on an odd count, which is why it needs a test
        rather than a reading.
        """
        algo = AlgorithmConfig().with_(chunking=ChunkConfig(max_tokens=20, prefix_tokens=4))
        single = fake_engine(algo)
        fused = fake_engine(algo.with_(decode_mode="fusion_mtp2"))
        even = list(range(10))  # pairs (0,1) (2,3) ... — tail of 4 starts at 6
        odd = list(range(11))  # pairs (0,1) ... (8,9), 10 alone — tail starts at 7

        assert single._carry_from(even) == [6, 7, 8, 9]
        assert single._carry_from(odd) == [7, 8, 9, 10]
        # Fusion ends on the last complete pair and starts on a pair boundary.
        # Token 10 of the odd sequence was the first half of a pair that never
        # finished, so it never occupied a slot and cannot be carried as one.
        assert fused._carry_from(even) == [6, 7, 8, 9]
        assert fused._carry_from(odd) == [6, 7, 8, 9]

    def test_an_odd_prefix_length_is_carried_one_token_long(self) -> None:
        """The branch the even default never reaches.

        `_carry_pair_aligned` ends the slice on a pair boundary, so the start
        lands off one exactly when `prefix_tokens` is odd — and the shipped
        default is 6. Every case above therefore skipped the `start % 2`
        branch, and a comment beside it claimed it preserved the requested
        length, which it cannot: aligning the start means handing over one
        token more than was asked for. That is the intended trade. An aligned
        prefix of `wanted + 1` re-pairs the way it was generated; a
        correctly-sized one starting mid-pair does not.

        Verified by drift: dropping `start -= 1` returns a five-token prefix
        beginning at an odd index, and this test goes red.
        """
        algo = AlgorithmConfig().with_(chunking=ChunkConfig(max_tokens=20, prefix_tokens=5))
        single = fake_engine(algo)
        fused = fake_engine(algo.with_(decode_mode="fusion_mtp2"))
        tokens = list(range(10))  # pairs (0,1) (2,3) (4,5) (6,7) (8,9)

        assert single._carry_from(tokens) == [5, 6, 7, 8, 9]
        # Six tokens for a prefix of five, starting where pair (4,5) starts.
        assert fused._carry_from(tokens) == [4, 5, 6, 7, 8, 9]
        # The trailing unpaired token of an odd sequence is still dropped
        # first, so both lengths land on the same aligned slice.
        assert fused._carry_from(list(range(11))) == [4, 5, 6, 7, 8, 9]

    def test_the_generator_drops_the_newest_token_of_an_odd_carry(self) -> None:
        """The helper above is only half the fix, and the other half undid it.

        `TorchTokenGenerator.generate` re-pairs whatever prefix it is handed,
        starting at that prefix's first token. It used to drop the *oldest*
        token of an odd prefix, which moves the pairing back onto the odd index
        the helper had just moved it off — so the engine's alignment reached the
        generator and was thrown away, on exactly the joins it was written for.
        Asserting on `_carry_from` alone could not see that.

        The assertion is on which token the generator marked as seen, not on
        which tokens it produced: at this size the weights are random, the
        logits are flat and the sampler's noise decides everything, so two
        different contexts sample identically and a token comparison would pass
        no matter what the code did.
        """
        torch = pytest.importorskip("torch")
        from loudkit.models.generator import TorchTokenGenerator

        llama = {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 128,
            "head_dim": 32,
        }
        algo = AlgorithmConfig().with_(decode_mode="fusion_mtp2")
        torch.manual_seed(0)
        gen = TorchTokenGenerator(algo, llama, attention="eager")
        gen.eval()
        for p in gen.parameters():
            p.requires_grad_(False)

        marked: list[np.ndarray] = []

        class Recorder:
            def __call__(self, logits, *, step, seen):
                if not marked:
                    marked.append(seen.copy())
                return int(np.argmax(logits))

        with torch.inference_mode():
            gen.generate(
                np.array([10, 20, 30], dtype=np.int64),
                fake_voice(),
                sampler=Recorder(),
                max_new_tokens=2,
                prefix=[100, 101, 102, 103, 104, 105, 106],
            )
        seen = marked[0]
        assert seen[100], "the oldest token of the carry must survive: it sets the pairing"
        assert not seen[106], "the newest token is the orphan half-pair and is the one to drop"

    def test_warm_renders_and_discards(self) -> None:
        """warm() must run every stage (or its first-use cost survives to the
        first request) and must not leak a Result."""
        engine = self._long_engine()
        assert engine.warm(fake_voice()) is None

    def test_pipelined_stream_is_byte_identical_to_the_serial_phases(self) -> None:
        """`stream` renders window k while generating k+1; the audio must not
        know that. This composes the two phases serially — the exact loop the
        pipeline replaced — and asserts hash equality against `stream`.

        With `prefix_tokens` non-zero, so the carry (the one value that crosses
        between windows) is exercised, not just the seeds.
        """
        import hashlib

        import numpy as np

        from loudkit.window import _STREAM_CHUNK, _derive

        engine = self._long_engine(prefix_tokens=4)
        voice = fake_voice()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."

        from loudkit.frontend.chunking import split_text
        from loudkit.frontend.speechtext import speech_text

        chunks = split_text(speech_text(text, "en"), engine.algorithm.chunking)
        assert len(chunks) > 2, "the fixture must span several windows"
        prefix_len = engine.algorithm.chunking.prefix_tokens
        carry: list[int] = []
        serial = []
        for i, chunk in enumerate(chunks):
            window = engine._generate_window(
                chunk,
                voice,
                # Chunk 0 draws the caller's seed itself; later chunks derive.
                seed=1 if i == 0 else _derive(1, _STREAM_CHUNK + i),
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
        # The fakes render the same bytes whatever the seed, so the seeds are
        # compared on their own: a window reports the seed it drew.
        assert [p.seed for p in streamed] == [p.seed for p in serial]
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
        a = np.concatenate([p.audio for p in engine.stream(text, fake_voice(), seed=1)])

        original = engine._render_window

        def faulty(window, voice, *, speed, should_cancel=None):
            from dataclasses import replace as _replace

            result = original(window, voice, speed=speed, should_cancel=should_cancel)
            audio = result.audio.copy()
            audio[0] += np.float32(1e-6)
            return _replace(result, audio=audio)

        object.__setattr__(engine, "_render_window", faulty)
        b = np.concatenate([p.audio for p in engine.stream(text, fake_voice(), seed=1)])
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

        def failing(window, voice, *, speed, should_cancel=None):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("vocoder exploded")
            return original(window, voice, speed=speed, should_cancel=should_cancel)

        object.__setattr__(engine, "_render_window", failing)
        with pytest.raises(RuntimeError, match="vocoder exploded"):
            list(engine.stream(text, fake_voice(), seed=1))

    def test_an_abandoned_stream_stops_the_producer(self) -> None:
        """A consumer that walks away after one chunk must not leave the
        producer speaking the rest of the passage to nobody."""
        import threading

        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        gen = engine.stream(text, fake_voice(), seed=1)
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

        import loudkit.stream as engine_mod

        monkeypatch.setattr(engine_mod, "_PRODUCER_JOIN_TIMEOUT", 0.05)
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."

        release = threading.Event()
        running = threading.Event()
        original = engine._render_window
        calls = {"n": 0}

        def wedged(window, voice, *, speed, should_cancel=None):
            calls["n"] += 1
            if calls["n"] > 1:
                # Only the windows the abandoned consumer never asks for; the
                # first must return, or next(gen) below never yields.
                running.set()
                release.wait(30.0)
            return original(window, voice, speed=speed, should_cancel=should_cancel)

        object.__setattr__(engine, "_render_window", wedged)
        gen = engine.stream(text, fake_voice(), seed=1)
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

        import loudkit.stream as engine_mod

        monkeypatch.setattr(engine_mod, "_PRODUCER_JOIN_TIMEOUT", 0.05)
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."

        release = threading.Event()
        running = threading.Event()
        original = engine._render_window
        calls = {"n": 0}

        def wedged(window, voice, *, speed, should_cancel=None):
            calls["n"] += 1
            if calls["n"] > 1:
                running.set()
                release.wait(30.0)
            return original(window, voice, speed=speed, should_cancel=should_cancel)

        object.__setattr__(engine, "_render_window", wedged)
        gen = engine.stream(text, fake_voice(), seed=1)
        try:
            next(gen)
            # Running, not merely queued: a queued render is cancelled by the
            # teardown, and only a thread already inside the renderer can
            # outlive the join.
            assert running.wait(30.0), "the second render never started"
            gen.close()

            voice = fake_voice()
            # Every entry, not only the streaming one: the stages a stuck
            # render holds are the same stages each of these walks into.
            for call in (
                lambda: engine.synthesize("Hello.", voice, seed=1),
                lambda: engine.synthesize(text, voice, seed=1),
                lambda: engine.synthesize("Hello.", voice, seed=1, single_window=True),
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

        import loudkit.stream as engine_mod

        monkeypatch.setattr(engine_mod, "_PRODUCER_JOIN_TIMEOUT", 0.05)
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."

        release = threading.Event()
        running = threading.Event()
        original = engine._render_window
        calls = {"n": 0}

        def wedged(window, voice, *, speed, should_cancel=None):
            calls["n"] += 1
            if calls["n"] > 1:
                running.set()
                release.wait(30.0)
            return original(window, voice, speed=speed, should_cancel=should_cancel)

        object.__setattr__(engine, "_render_window", wedged)
        gen = engine.stream(text, fake_voice(), seed=1)
        try:
            next(gen)
            # Running, not merely queued: a queued render is cancelled by the
            # teardown, and only a thread already inside the renderer can
            # outlive the join.
            assert running.wait(30.0), "the second render never started"
            gen.close()
            assert self._long_engine().synthesize("Hello.", fake_voice(), seed=1).duration > 0
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

        import loudkit.stream as engine_mod

        slow_s = 0.2
        join_s = 0.5

        monkeypatch.setattr(engine_mod, "_PRODUCER_JOIN_TIMEOUT", join_s)
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."

        original = engine._render_window
        calls = {"n": 0}

        def slow(window, voice, *, speed, should_cancel=None):
            calls["n"] += 1
            if calls["n"] > 1:
                # The first must return promptly or next(gen) never yields.
                # Every one after it is a render the abandoned consumer will
                # not read, which is exactly the work the teardown has to drop.
                time.sleep(slow_s)
            return original(window, voice, speed=speed, should_cancel=should_cancel)

        generated = threading.Semaphore(0)
        original_windows = engine._windows_for_chunk

        def counting(*args, **kwargs):
            windows = original_windows(*args, **kwargs)
            for _ in windows or ():
                generated.release()
            return windows

        object.__setattr__(engine, "_render_window", slow)
        object.__setattr__(engine, "_windows_for_chunk", counting)

        gen = engine.stream(text, fake_voice(), seed=1)
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
        assert engine.synthesize("Hello.", fake_voice(), seed=1).duration > 0

    def test_a_healthy_engine_is_never_wedged(self) -> None:
        """A stream that drains normally leaves the engine usable, which is the
        case that must not be caught by the guard above."""
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        assert list(engine.stream(text, fake_voice(), seed=1))
        assert engine._wedged is None
        assert engine.synthesize("Hello.", fake_voice(), seed=1).duration > 0

    def test_stream_yields_one_result_per_chunk(self) -> None:
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        parts = list(engine.stream(text, fake_voice(), seed=1))
        assert len(parts) > 1
        assert all(p.audio.size > 0 for p in parts)

    def test_synthesize_is_stream_concatenated(self) -> None:
        """The identity the two delivery shapes are held to: same seed, same
        tokens, whichever one the caller asked for."""
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        streamed = list(engine.stream(text, fake_voice(), seed=1))
        assert len(streamed) > 1, "the text has to actually split"
        joined = engine.synthesize(text, fake_voice(), seed=1)
        assert joined.audio.size == sum(p.audio.size for p in streamed)
        assert list(joined.tokens) == [t for p in streamed for t in p.tokens]
        assert joined.mel.shape[1] == sum(p.mel.shape[1] for p in streamed)
        assert len(joined.chunks) == len(streamed)

    def test_a_short_text_agrees_with_stream_too(self) -> None:
        """The identity holds for one window as well as for many. It did not
        before: `synthesize` seeded the single window with the raw seed while
        `stream` derived a chunk seed, so the same seed read a short text two
        ways depending on which method was called.

        Asserted on the seed the renderer was handed, not only on the bytes:
        the fakes here emit the same silence whatever the seed, so bytes alone
        would pass with the two laws still disagreeing.
        """
        text = "Two words."
        a = self._long_engine()
        streamed = list(a.stream(text, fake_voice(), seed=1))
        assert len(streamed) == 1, "the text has to fit one window"

        b = self._long_engine()
        joined = b.synthesize(text, fake_voice(), seed=1)
        assert joined.audio.size > 0
        assert list(joined.tokens) == list(streamed[0].tokens)
        np.testing.assert_array_equal(joined.audio, streamed[0].audio)
        assert b.mel_decoder.seeds == a.mel_decoder.seeds
        assert b.vocoder.seeds == a.vocoder.seeds

    def test_the_single_window_keyword_is_the_same_seed_law(self) -> None:
        """`single_window` changes what is refused, never what is read.

        It used to be a second law: chunk 0 drew `derive(seed, 16)` while a
        single window drew the raw seed, so a short text read one way through
        `synthesize` and another through `stream` — and Python read it
        differently from the four ports, whose one-window call has always used
        the raw seed. Chunk 0 draws the raw seed everywhere now, and the
        `end_to_end` conformance vectors and the reference parity dumps hold on
        the default path because of it.
        """
        text = "Two words."
        a = self._long_engine()
        streamed = list(a.stream(text, fake_voice(), seed=1))
        b = self._long_engine()
        one = b.synthesize(text, fake_voice(), seed=1, single_window=True)
        assert b.mel_decoder.seeds == a.mel_decoder.seeds
        assert b.vocoder.seeds == a.vocoder.seeds
        assert list(one.tokens) == list(streamed[0].tokens)

    def test_a_multi_window_text_is_not_refused(self) -> None:
        """The refusal is `single_window`'s job now, and only its job."""
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        result = engine.synthesize(text, fake_voice(), seed=1)
        assert len(result.chunks) > 1, "the text has to actually split"
        assert result.audio.size > 0

    def test_prefix_reaches_the_generator(self) -> None:
        """Chunks generated independently restart their pitch contour, which is
        audible at every join."""
        engine = self._long_engine(prefix_tokens=2)
        list(engine.stream("One. Two. Three. Four. Five. Six.", fake_voice(), seed=1))
        calls = engine.token_generator.calls
        assert len(calls) > 1
        assert calls[0]["prefix"] == []
        assert len(calls[1]["prefix"]) == 2

    def test_no_prefix_when_disabled(self) -> None:
        engine = self._long_engine(prefix_tokens=0)
        list(engine.stream("One. Two. Three. Four. Five. Six.", fake_voice(), seed=1))
        calls = engine.token_generator.calls
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
            fake_voice(),
            seed=1,
            should_cancel=lambda: len(yielded) >= 2,
        ):
            yielded.append(part)
        assert len(yielded) == 2, f"expected 2 chunks before cancel, got {len(yielded)}"

    def test_the_reported_seed_does_not_depend_on_the_chunk_count(self) -> None:
        """`Result.seed` is documented as what you need to reproduce the audio.

        The multi-chunk branch builds a fresh `Result` with `seed=seed`; the
        single-chunk branch returns the chunk's own `Result` with its seed set
        to the caller's (`stream.joined`). Both are asserted, so
        the same call cannot report two different numbers depending only on how
        the text happened to split — from the field `__repr__` prints and a user
        would quote in a bug report.
        """
        engine = self._long_engine()
        short = engine.synthesize("Two words.", fake_voice(), seed=7)
        long = engine.synthesize(
            "One. Two. Three. Four. Five. Six. Seven. Eight.", fake_voice(), seed=7
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
        decoder.calls = 0
        vocoder.calls = 0

        inner_decode = decoder.decode
        inner_synth = vocoder.synthesize

        def counting_decode(*a, **kw):
            decoder.calls += 1
            return inner_decode(*a, **kw)

        def counting_synth(*a, **kw):
            vocoder.calls += 1
            return inner_synth(*a, **kw)

        decoder.decode = counting_decode
        vocoder.synthesize = counting_synth

        # Cancel from the very first poll: generation stops immediately and
        # nothing should reach the renderer at all.
        parts = list(
            engine.stream(
                "One. Two. Three. Four.", fake_voice(), seed=1, should_cancel=lambda: True
            )
        )
        assert parts == []
        assert decoder.calls == 0, "a cancelled chunk was still decoded to mel"
        assert vocoder.calls == 0, "a cancelled chunk was still vocoded"

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
        vocoder.calls = 0
        cancelled = {"flag": False}

        inner_decode = decoder.decode
        inner_synth = vocoder.synthesize

        def decode_then_expire(*a, **kw):
            # The deadline expires while the mel decoder is running: the stage
            # completes (it cannot be preempted) and the flag is up by the
            # time the engine decides whether to start the vocoder.
            mel = inner_decode(*a, **kw)
            cancelled["flag"] = True
            return mel

        def counting_synth(*a, **kw):
            vocoder.calls += 1
            return inner_synth(*a, **kw)

        decoder.decode = decode_then_expire
        vocoder.synthesize = counting_synth

        parts = list(
            engine.stream(
                "Two words.", fake_voice(), seed=1, should_cancel=lambda: cancelled["flag"]
            )
        )
        assert parts == [], "a window cancelled mid-render was still yielded"
        assert vocoder.calls == 0, "the vocoder started after the cancel"

    def test_chunk_audio_does_not_depend_on_how_many_came_before(self) -> None:
        """Each chunk gets its own derived seed, so stopping a stream early
        cannot change what the earlier chunks were."""
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six."
        full = list(engine.stream(text, fake_voice(), seed=9))
        partial = []
        for i, part in enumerate(self._long_engine().stream(text, fake_voice(), seed=9)):
            partial.append(part)
            if i == 0:
                break
        np.testing.assert_array_equal(full[0].audio, partial[0].audio)

    def test_empty_text_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nothing to speak"):
            self._long_engine().synthesize("   ", fake_voice(), seed=1)

    def test_every_entry_point_refuses_empty_text_the_same_way(self) -> None:
        """One input, one answer, whichever door it arrives through.

        `stream` raised; `synthesize` generated against an empty prompt and
        returned about a tenth of a second of near-silence. A caller batching
        titles, or reading a field that happened to be blank, got audio back
        and no reason to look twice — which is the failure a refusal exists to
        prevent.
        """
        engine = self._long_engine()
        for call in (
            lambda: engine.synthesize("   ", fake_voice(), seed=1),
            lambda: engine.synthesize("   ", fake_voice(), seed=1, single_window=True),
            lambda: list(engine.stream("   ", fake_voice(), seed=1)),
        ):
            with pytest.raises(ValueError, match="nothing to speak"):
                call()


class TestSynthesizeTokens:
    def test_renders_a_given_sequence(self) -> None:
        """The diagnostic that takes the generator out of a comparison."""
        engine = fake_engine()
        result = engine.synthesize_tokens([1, 2, 3], fake_voice(), seed=4)
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
    exactly one pass. The ladder is a pure function of the caller's seed.

    Asserted through ``single_window=True``: the ladder is one window's
    property, and a text that split would multiply the call counts by the
    chunk count and measure nothing.
    """

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
        engine.synthesize("one two three four five", fake_voice(), seed=7, single_window=True)
        assert len(gen.calls) == 1

    def test_a_dropout_window_is_rerolled_and_the_second_roll_ships(self) -> None:
        engine, gen = self._engine_with(CondemnedThenCleanGenerator)
        text = " ".join(["word"] * 40)
        result = engine.synthesize(text, fake_voice(), seed=7, single_window=True)
        assert len(gen.calls) == 2, "the condemned first roll must trigger exactly one retry"
        assert result.inspections[0].reason == "clean"

    def test_a_starved_desperation_cut_is_rerolled(self) -> None:
        """The da0028 failure: a cap-hit row whose desperation trim keeps
        under `desperation_min_keep_per_text_token` per text token is
        condemned, not shipped — the trim was 1.44 s of near-silence through
        ids no census lists, and the same window renders clean on a retry."""
        algo = AlgorithmConfig(sampling=SamplingConfig(silence_token_ids=(3,)))

        class StarvedThenCleanGenerator(FakeGenerator):
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
                self.calls.append({})
                if len(self.calls) == 1:
                    # 40 text tokens cap at 200; the seam cut lands at 20,
                    # under the 68-token floor (40 x 1.7). No stop token:
                    # the row burned to the ceiling.
                    return [*range(100, 120), *([3] * 12), *range(200, 368)]
                return [*range(80), self.config.stop_speech_token]

        gen = StarvedThenCleanGenerator(algo)
        engine = Engine(
            frontend=FakeFrontend(),
            token_generator=gen,
            mel_decoder=FakeMelDecoder(algo),
            vocoder=FakeVocoder(algo),
            algorithm=algo,
        )
        result = engine.synthesize(
            " ".join(["word"] * 40), fake_voice(), seed=7, single_window=True
        )
        assert len(gen.calls) == 2, "the starved trim must trigger exactly one retry"
        assert result.inspections[0].reason == "clean"
        assert not result.suspect

    def test_an_exhausted_starved_ladder_ships_the_trim_flagged(self) -> None:
        """When every attempt is starved, the verdict's keep still holds the
        cut, so what ships is today's trimmed audio — flagged `suspect` —
        rather than the untrimmed cap-hit babble."""
        algo = AlgorithmConfig(sampling=SamplingConfig(silence_token_ids=(3,)))

        class AlwaysStarvedGenerator(FakeGenerator):
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
                self.calls.append({})
                return [*range(100, 120), *([3] * 12), *range(200, 368)]

        gen = AlwaysStarvedGenerator(algo)
        engine = Engine(
            frontend=FakeFrontend(),
            token_generator=gen,
            mel_decoder=FakeMelDecoder(algo),
            vocoder=FakeVocoder(algo),
            algorithm=algo,
        )
        result = engine.synthesize(
            " ".join(["word"] * 40), fake_voice(), seed=7, single_window=True
        )
        assert len(gen.calls) == 1 + algo.postprocess.retry_max_attempts
        assert list(result.tokens) == list(range(100, 120)), (
            "the exhausted ladder ships the trim, not the whole cap-hit row"
        )
        assert result.inspections[0].reason == "desperation"
        assert result.suspect, "the caller has to be told the trim is a fallback"

    def test_the_ladder_is_bounded(self) -> None:
        engine, gen = self._engine_with(AlwaysCondemnedGenerator)
        text = " ".join(["word"] * 40)
        result = engine.synthesize(text, fake_voice(), seed=7, single_window=True)
        assert len(gen.calls) == 1 + engine.algorithm.postprocess.retry_max_attempts
        # Nothing could be fixed; the best attempt ships (here they are all
        # identical), and the verdict says so.
        assert result.inspections[0].reason == "dropout"

    def test_the_retry_ladder_headroom_is_the_gap_the_engine_leaves(self) -> None:
        """The bound in `postprocess` is a copy; this is what it copies.

        `PostprocessConfig` cannot import the engine's stream constants without
        a cycle, so `RETRY_LADDER_HEADROOM` restates the gap by hand. A hand
        copy of a number in another module is the shape that goes stale, so it
        is pinned here rather than trusted.
        """
        from loudkit.postprocess import RETRY_LADDER_HEADROOM
        from loudkit.window import _STREAM_CHUNK, _STREAM_RETRY

        assert RETRY_LADDER_HEADROOM == _STREAM_CHUNK - _STREAM_RETRY

    def test_an_attempt_count_that_would_reach_the_chunk_streams_is_refused(self) -> None:
        """The comment beside `_STREAM_RETRY` says the ladder stays clear of
        the chunk streams. Nothing made that true: attempt 8 draws stream 16,
        which is `_STREAM_CHUNK`, so a long-form chunk seed and a deep retry
        seed collided off the same base. The configuration is now refused
        where every other out-of-range constant is."""
        from loudkit.postprocess import RETRY_LADDER_HEADROOM, PostprocessConfig

        PostprocessConfig(retry_max_attempts=RETRY_LADDER_HEADROOM - 1)  # the last legal rung
        with pytest.raises(ValueError, match="retry_max_attempts must be in"):
            PostprocessConfig(retry_max_attempts=RETRY_LADDER_HEADROOM)

    def test_each_rung_of_the_ladder_draws_the_stream_it_documents(self) -> None:
        """Which seed each attempt samples from, asserted rather than read.

        The first attempt uses the caller's seed untouched and attempt *n* uses
        `derive(seed, _STREAM_RETRY + n)`. Both halves matter: a first attempt
        that derived would change every clean render's audio, and a ladder that
        derived from the wrong base would still look reproducible while
        colliding with another stage's stream.
        """
        from loudkit.postprocess import PostprocessConfig
        from loudkit.window import _STREAM_RETRY, _derive

        algo = AlgorithmConfig(postprocess=PostprocessConfig(retry_max_attempts=3))
        gen = AlwaysCondemnedGenerator(algo)
        engine = Engine(
            frontend=FakeFrontend(),
            token_generator=gen,
            mel_decoder=FakeMelDecoder(algo),
            vocoder=FakeVocoder(algo),
            algorithm=algo,
        )
        seeds: list[int] = []

        original = gen.generate

        def recording(*args, **kwargs):
            seeds.append(kwargs["sampler"]._seed)
            return original(*args, **kwargs)

        gen.generate = recording
        engine.synthesize(" ".join(["word"] * 40), fake_voice(), seed=7, single_window=True)

        assert seeds == [7, *(_derive(7, _STREAM_RETRY + n) for n in (1, 2, 3))]
        assert len(set(seeds)) == len(seeds), "a repeated seed would re-roll the same row"

    def test_a_tie_on_the_ladder_keeps_the_earlier_attempt(self) -> None:
        """Which of two equally good attempts ships, made observable.

        The selection is `silence_count < best[0]` — strict, so a later attempt
        that merely ties does not displace an earlier one. That keeps the
        shipped row a function of the seed alone. Loosening it to `<=` is
        invisible in every other test here, because none of them produces two
        attempts with the same silence count and different tokens: the
        worsening case has counts 3, 1, 2, and the always-condemned case
        returns one identical row.

        Verified by drift: with `<=` the third attempt ships and this goes red.
        """
        algo = AlgorithmConfig(sampling=SamplingConfig(silence_token_ids=(3,)))

        class TyingGenerator(FakeGenerator):
            # Three dropout-shaped rows, silence counts 2, 1, 1 — attempts two
            # and three tie for best and hold different tokens.
            rows = ([3, 3, 5], [3, 5, 7], [3, 5, 9])

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
                row = self.rows[len(self.calls)]
                self.calls.append({})
                return [*row, self.config.stop_speech_token]

        gen = TyingGenerator(algo)
        engine = Engine(
            frontend=FakeFrontend(),
            token_generator=gen,
            mel_decoder=FakeMelDecoder(algo),
            vocoder=FakeVocoder(algo),
            algorithm=algo,
        )
        result = engine.synthesize(
            " ".join(["word"] * 40), fake_voice(), seed=7, single_window=True
        )
        assert len(gen.calls) == 1 + algo.postprocess.retry_max_attempts
        assert list(result.tokens) == [3, 5, 7], "the earlier of two tied attempts ships"

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
        engine.synthesize(" ".join(["word"] * 40), fake_voice(), seed=7, single_window=True)
        assert len(gen.calls) == 1

    def test_same_seed_same_ladder(self) -> None:
        a_engine, a_gen = self._engine_with(CondemnedThenCleanGenerator)
        b_engine, b_gen = self._engine_with(CondemnedThenCleanGenerator)
        text = " ".join(["word"] * 40)
        a = a_engine.synthesize(text, fake_voice(), seed=7, single_window=True)
        b = b_engine.synthesize(text, fake_voice(), seed=7, single_window=True)
        assert list(a.tokens) == list(b.tokens)
        assert len(a_gen.calls) == len(b_gen.calls)

    def test_an_exhausted_ladder_ships_the_best_attempt_not_the_last(self) -> None:
        """Measured: on the worst voices 30% of condemned fires exhaust the
        ladder, and the last attempt can be worse than the first. "Best" is
        fewest tokens in the true-silence set — integer and portable."""
        algo = AlgorithmConfig(sampling=SamplingConfig(silence_token_ids=(3,)))

        class WorseningGenerator(FakeGenerator):
            # Every attempt is a dropout-shaped row; their silence counts are
            # 3, 1, 2 — the middle attempt is the best and must ship.
            rows = ([3, 3, 3, 5], [5, 3, 5], [3, 3, 5])

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
                row = self.rows[len(self.calls)]
                self.calls.append({})
                return [*row, self.config.stop_speech_token]

        gen = WorseningGenerator(algo)
        engine = Engine(
            frontend=FakeFrontend(),
            token_generator=gen,
            mel_decoder=FakeMelDecoder(algo),
            vocoder=FakeVocoder(algo),
            algorithm=algo,
        )
        result = engine.synthesize(
            " ".join(["word"] * 40), fake_voice(), seed=7, single_window=True
        )
        assert len(gen.calls) == 1 + algo.postprocess.retry_max_attempts
        assert list(result.tokens) == [5, 3, 5], (
            "the attempt with the fewest true-silence tokens ships, not the last"
        )
        assert result.inspections[0].reason == "dropout"


class TestTheStageSeedIsWhatSeparatesTheStreams:
    """The sub-stream id is not the separation, and reading it as one is the
    trap: sampling, the flow prior and the vocoder phase all draw sub-stream 0.
    What keeps them apart is the per-stage seed, so renumbering an id to "fix"
    a collision that is not there moves every sample the model draws.
    """

    def test_the_three_sub_stream_ids_are_the_same_number(self) -> None:
        from loudkit.models.windowing import FLOW_NOISE_STREAM, VOCODER_PHASE_STREAM
        from loudkit.sampler import _STREAM_SAMPLING

        assert (_STREAM_SAMPLING, FLOW_NOISE_STREAM, VOCODER_PHASE_STREAM) == (0, 0, 0)

    def test_one_user_seed_reaches_each_stage_as_a_different_number(self) -> None:
        from loudkit.window import _STREAM_CHUNK, _STREAM_RESPLIT, _STREAM_RETRY, _derive

        stages = [_derive(7, s) for s in (_STREAM_RETRY, _STREAM_CHUNK, _STREAM_RESPLIT)]
        assert len(set(stages)) == len(stages)
        assert 7 not in stages


class TestSynthesizeRefusesRatherThanTruncating:
    """`synthesize` shipped a truncated window and said so only in a flag.

    `WindowOverflowError` exists for exactly this and its docstring calls the
    alternative unacceptable: text goes missing while the audio still sounds
    fine, and only a listener who knows the passage would notice. The four
    ports' READMEs and guides have promised this error since before 0.1.1 and
    none of the five raised one.
    """

    def _engine_with_window(self, window: int, tokens: int):
        from loudkit.config import ChunkConfig, SamplingConfig, WindowConfig

        # The chunk budget has to fit the window, which `AlgorithmConfig`
        # refuses to let it exceed — the same refusal, one layer up.
        algo = AlgorithmConfig().with_(
            window=WindowConfig(max_speech_tokens=window),
            sampling=SamplingConfig(max_new_tokens=window),
            chunking=ChunkConfig(max_tokens=window, prefix_tokens=0),
        )

        class Filling(FakeGenerator):
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
                self.calls.append({})
                return list(range(tokens))  # no stop token: the cap is what ends it

        gen = Filling(algo)
        return Engine(
            frontend=FakeFrontend(),
            token_generator=gen,
            mel_decoder=FakeMelDecoder(algo),
            vocoder=FakeVocoder(algo),
            algorithm=algo,
        )

    def test_a_window_that_filled_is_refused(self) -> None:
        from loudkit.errors import WindowOverflowError

        engine = self._engine_with_window(window=40, tokens=40)
        with pytest.raises(WindowOverflowError, match="did not fit one 40-token window"):
            engine.synthesize(" ".join(["word"] * 8), fake_voice(), seed=1, single_window=True)

    def test_the_message_names_the_remedy(self) -> None:
        from loudkit.errors import WindowOverflowError

        engine = self._engine_with_window(window=40, tokens=40)
        with pytest.raises(WindowOverflowError) as caught:
            engine.synthesize(" ".join(["word"] * 8), fake_voice(), seed=1, single_window=True)
        assert "synthesize() splits at sentence boundaries" in str(caught.value)
        assert caught.value.window == 40

    def test_the_refusal_lands_before_the_renderer(self) -> None:
        """Cheap where it costs the most.

        The refusal used to be read off a finished `Result`, so the mel decoder
        and the vocoder had already run on the one input guaranteed to be a
        full window -- the most expensive render this engine performs, thrown
        away, and then performed again by a caller who dropped
        `single_window`. The other four ports refuse between the two phases.
        """
        from loudkit.errors import WindowOverflowError

        engine = self._engine_with_window(window=40, tokens=40)
        with pytest.raises(WindowOverflowError):
            engine.synthesize(" ".join(["word"] * 8), fake_voice(), seed=1, single_window=True)
        assert engine.mel_decoder.seeds == [], "the mel decoder ran on a refused window"
        assert engine.vocoder.seeds == [], "the vocoder ran on a refused window"

    def _engine_that_repeats(self, window: int, tokens: int):
        """A filled window whose content postprocess wants to cut back."""
        from loudkit.config import ChunkConfig, SamplingConfig, WindowConfig

        algo = AlgorithmConfig().with_(
            window=WindowConfig(max_speech_tokens=window),
            sampling=SamplingConfig(max_new_tokens=window),
            chunking=ChunkConfig(max_tokens=window, prefix_tokens=0),
        )

        class Repeating(FakeGenerator):
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
                self.calls.append({})
                # One token, over and over, to the cap: a run the trim cuts.
                return [7] * tokens

        return Engine(
            frontend=FakeFrontend(),
            token_generator=Repeating(algo),
            mel_decoder=FakeMelDecoder(algo),
            vocoder=FakeVocoder(algo),
            algorithm=algo,
        )

    def test_a_trim_cannot_hide_a_filled_window(self) -> None:
        """The refusal read the length postprocess left, not the one generated.

        A window that filled and was then trimmed back came out of
        `_generate_window` looking like a short render: `len(window.speech)`
        was under the window, the guard's own `n_tokens < window` test passed,
        and `synthesize` returned audio for a fraction of the text with the
        rest never spoken. Postprocess removed the evidence, not the overflow.
        """
        from loudkit.errors import WindowOverflowError

        engine = self._engine_that_repeats(window=40, tokens=40)
        text = " ".join(["word"] * 8)
        # The case only tests anything if the trim really does cut it back.
        window = engine._generate_window(text, fake_voice(), seed=1, language="en")
        assert window is not None
        assert window.n_generated == 40, "the case needs a window that filled"
        assert len(window.speech) < 40, "the case needs postprocess to have trimmed it"

        with pytest.raises(WindowOverflowError, match="did not fit one 40-token window"):
            engine.synthesize(text, fake_voice(), seed=1, single_window=True)

    def test_a_trim_cannot_hide_a_filled_window_from_the_resplit_either(self) -> None:
        """The same fact, read by the other reader.

        `_windows_for_chunk` re-splits a chunk that filled its window, and it
        asked the same trimmed length: a chunk whose window filled and was then
        cut back was handed on unsplit, which is the long-form half of the same
        silent truncation.
        """
        engine = self._engine_that_repeats(window=40, tokens=40)
        text = " ".join(["word"] * 8)
        windows = engine._windows_for_chunk(
            text,
            fake_voice(),
            index=0,
            seed=1,
            language="en",
            prefix=(),
            is_terminal=True,
            should_cancel=None,
        )
        assert windows is not None
        assert len(windows) > 1, "a filled window was handed on without a re-split"

    def test_a_ceiling_hit_is_not_refused(self) -> None:
        """The other half of `hit_token_cap`, and the half that must not raise.

        The postprocess length ceiling stops a *short* text that ran away. That
        text fitted; the model babbled. There is nothing for the caller to
        split, so refusing would hand them an error they can do nothing about —
        and `_windows_for_chunk` declines to re-split it for the same reason.
        """
        # Past the length ceiling (4 speech tokens per text token, +40) and
        # well short of the 255-token window: a runaway on a one-word text.
        engine = self._engine_with_window(window=255, tokens=200)
        result = engine.synthesize("word", fake_voice(), seed=1)
        assert result is not None
        assert result.hit_token_cap, "the case needs a render the ceiling stopped"

    def test_long_form_still_repairs_rather_than_refusing(self) -> None:
        """Plain `synthesize` splits; only `single_window=True` refuses."""
        engine = self._engine_with_window(window=40, tokens=40)
        result = engine.synthesize(" ".join(["word"] * 8), fake_voice(), seed=1)
        assert result is not None


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
        return fake_engine(algo)

    def test_the_first_chunk_is_conditioned_on_the_tail(self) -> None:
        engine = self._long_engine(prefix_tokens=3)
        engine.synthesize("one two", fake_voice(), seed=1, previous_tokens=[10, 11, 12, 13, 14])
        calls = engine.token_generator.calls
        assert calls[0]["prefix"] == [12, 13, 14]

    def test_a_long_history_is_sliced_rather_than_refused(self) -> None:
        """Chaining should be `previous_tokens=result.tokens`, with no arithmetic
        at the call site: a caller who had to know the prefix length would be
        keeping a copy of an algorithm value."""
        engine = self._long_engine(prefix_tokens=2)
        engine.synthesize("one two", fake_voice(), seed=1, previous_tokens=list(range(200)))
        assert engine.token_generator.calls[0]["prefix"] == [198, 199]

    def test_none_is_todays_behaviour(self) -> None:
        engine = self._long_engine()
        engine.synthesize("one two", fake_voice(), seed=1)
        assert engine.token_generator.calls[0]["prefix"] == []

    def test_zero_prefix_tokens_means_no_context_not_all_of_it(self) -> None:
        """`tokens[-0:]` is the whole list. At the setting that means "chunks
        are independent", that would condition on the entire previous
        utterance — the exact opposite."""
        engine = self._long_engine(prefix_tokens=0)
        engine.synthesize("one two", fake_voice(), seed=1, previous_tokens=[1, 2, 3])
        assert engine.token_generator.calls[0]["prefix"] == []

    def test_only_the_first_chunk_takes_it(self) -> None:
        """Every chunk after the first is conditioned on the one before, as
        always — the caller's history seeds the carry, it does not replace it."""
        engine = self._long_engine(prefix_tokens=2)
        list(
            engine.stream(
                "One. Two. Three. Four. Five. Six.",
                fake_voice(),
                seed=1,
                previous_tokens=[90, 91],
            )
        )
        calls = engine.token_generator.calls
        assert len(calls) > 1
        assert calls[0]["prefix"] == [90, 91]
        assert calls[1]["prefix"] != [90, 91]
        assert len(calls[1]["prefix"]) == 2

    def test_long_form_passes_it_to_the_first_chunk(self) -> None:
        engine = self._long_engine(prefix_tokens=2)
        engine.synthesize(
            "One. Two. Three. Four.", fake_voice(), seed=1, previous_tokens=[7, 8, 9]
        )
        assert engine.token_generator.calls[0]["prefix"] == [8, 9]

    def test_the_same_history_gives_the_same_bytes(self) -> None:
        """Determinism is the whole contract; a new input must not weaken it."""
        first = self._long_engine().synthesize(
            "one two three", fake_voice(), seed=5, previous_tokens=[3, 4, 5, 6, 7, 8]
        )
        second = self._long_engine().synthesize(
            "one two three", fake_voice(), seed=5, previous_tokens=[3, 4, 5, 6, 7, 8]
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
                fake_voice(),
                seed=1,
                previous_tokens=[engine.algorithm.stop_speech_token],
            )
        with pytest.raises(ValueError, match="not an acoustic speech token"):
            engine.synthesize("one two", fake_voice(), seed=1, previous_tokens=[-1])

    def test_a_result_chains_straight_into_the_next_call(self) -> None:
        """The documented usage, end to end, with nothing in between."""
        engine = self._long_engine(prefix_tokens=3)
        first = engine.synthesize("one two three four", fake_voice(), seed=1)
        engine.synthesize("five six", fake_voice(), seed=2, previous_tokens=first.tokens)
        calls = engine.token_generator.calls
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
        engine = fake_engine()
        with pytest.raises(InvalidTokensError, match="tokens contains"):
            engine.synthesize_tokens(np.array(bad, dtype=np.int64), fake_voice())

    def test_the_message_names_this_field_not_the_streaming_one(self) -> None:
        # It used to say "previous_tokens" whatever called it, which sends a
        # caller looking for a field they never passed.
        engine = fake_engine()
        with pytest.raises(InvalidTokensError) as caught:
            engine.synthesize_tokens(np.array([-1], dtype=np.int64), fake_voice())
        assert "previous_tokens" not in str(caught.value)


class CappedGenerator(FakeGenerator):
    """Never emits a stop token, so every window runs into whatever cap it has.

    The only way to reach `hit_token_cap` deterministically without weights,
    and therefore the only way to exercise `cap_resplit` here. Records the
    prefix and the text-token count of every call, which is what the carry and
    the terminality assertions read.
    """

    def generate(
        self, text_tokens, voice, *, sampler, max_new_tokens=None, prefix=(), should_cancel=None
    ):
        self.calls.append({"n_text": len(text_tokens), "prefix": list(prefix)})
        default_cap = self.config.sampling.max_new_tokens
        cap = max_new_tokens if max_new_tokens is not None else default_cap
        return list(range(cap))


class TestCapResplit:
    """The engine's side of `chunking.cap_resplit`.

    The shared conformance fixture pins the law against real weights, but a
    valid config gives a chunk at most `window * CHARS_PER_TOKEN` characters,
    so a half is a quarter of the window and can never overrun. Everything
    that needs a controllable cap lives here. Three mutations to
    `_windows_for_chunk` -- the second half taking the pre-split prefix, the
    second half losing its terminality, and the window gate removed -- once
    survived the entire suite.
    """

    def fake_engine(self, *, window: int = 40, budget_tokens: int = 40, law: str = "word"):
        base = AlgorithmConfig()
        algo = base.with_(
            window=replace(base.window, max_speech_tokens=window, static_length=window),
            sampling=replace(base.sampling, max_new_tokens=window),
            chunking=replace(base.chunking, max_tokens=budget_tokens, cap_resplit=law),
        )
        gen = CappedGenerator(algo)
        engine = Engine(
            frontend=FakeFrontend(),
            token_generator=gen,
            mel_decoder=FakeMelDecoder(algo),
            vocoder=FakeVocoder(algo),
            algorithm=algo,
        )
        return engine, gen, algo

    def _text(self, algo) -> str:
        # Two chunks at least, so there is a join to carry across. The tail is
        # short words rather than another `wordNN`, so the LAST chunk has a word
        # boundary: several cases only discriminate when the terminal window is
        # itself a second half.
        from loudkit.frontend.chunking import split_in_half, split_text

        text = " ".join(f"word{i}" for i in range(36)) + " an ox at a bay"
        chunks = split_text(text, algo.chunking)
        assert len(chunks) >= 2
        assert split_in_half(chunks[-1]) is not None
        return text

    def test_a_capped_chunk_becomes_two_windows(self) -> None:
        from loudkit.frontend.chunking import split_in_half, split_text

        engine, _, algo = self.fake_engine()
        text = self._text(algo)
        chunks = split_text(text, algo.chunking)
        out = list(engine.stream(text, fake_voice(), seed=7))
        # The generator never stops, so every chunk caps; a chunk with no word
        # boundary is still shipped whole, so the expectation is built from
        # what `split_in_half` actually returns rather than from a doubling.
        want = []
        for c in chunks:
            halves = split_in_half(c)
            want.extend(halves if halves is not None else [c])
        assert [" ".join(s.text for s in r.chunks) for r in out] == want
        assert len(want) > len(chunks), "the case must produce at least one split"

    def test_the_second_half_carries_the_first_half_tail(self) -> None:
        """Not the prefix the pre-split window received.

        This is the join the repair creates, and the prefix is the whole reason
        a join does not restart the pitch contour. Handing the second half the
        chunk's incoming prefix instead leaves the seam the carry exists to
        remove, and no other test in the suite noticed.
        """
        engine, gen, algo = self.fake_engine()
        list(engine.stream(self._text(algo), fake_voice(), seed=7))
        n = algo.chunking.prefix_tokens
        # calls: [chunk0 (discarded), half A, half B, chunk1 (discarded), ...]
        first_a, first_b = gen.calls[1], gen.calls[2]
        assert first_b["prefix"] != first_a["prefix"], "half B reused half A's prefix"
        assert first_b["prefix"] == list(range(algo.sampling.max_new_tokens))[-n:]

    def test_the_next_chunk_carries_the_second_half(self) -> None:
        engine, gen, algo = self.fake_engine()
        list(engine.stream(self._text(algo), fake_voice(), seed=7))
        n = algo.chunking.prefix_tokens
        # The discarded generation for chunk 1 is call 3; its prefix must come
        # off half B, not off the window that was thrown away.
        assert gen.calls[3]["prefix"] == list(range(algo.sampling.max_new_tokens))[-n:]

    def test_terminality_goes_to_the_second_half(self, monkeypatch) -> None:
        import loudkit.window as eng

        seen: list[bool] = []
        real = eng.inspect

        def spy(*a, **kw):
            seen.append(bool(kw.get("is_terminal")))
            return real(*a, **kw)

        monkeypatch.setattr(eng, "inspect", spy)
        engine, _, algo = self.fake_engine()
        text = self._text(algo)
        from loudkit.frontend.chunking import split_in_half, split_text

        # The case only discriminates if the LAST chunk splits: otherwise the
        # terminal window is an undivided chunk and the second half's flag is
        # never read. An earlier version missed that and passed against a
        # mutation that stripped terminality from every second half.
        chunks = split_text(text, algo.chunking)
        assert split_in_half(chunks[-1]) is not None, "the case needs a split at the end"

        out = list(engine.stream(text, fake_voice(), seed=7))
        assert len(seen) == len(out) + sum(1 for c in chunks if split_in_half(c) is not None), (
            "one inspection per generated window, discarded ones included"
        )
        # The last chunk splits, so the final three inspections are the window
        # that overran, then half A, then half B. Terminality belongs to the
        # window that ends the passage: the discarded parent had it, half A must
        # not, and half B must inherit it.
        assert seen[-3:] == [True, False, True], seen[-6:]
        assert sum(seen) == 2, f"only the last chunk is terminal: {sum(seen)}"

    def test_the_cap_flag_survives_to_the_passage(self) -> None:
        """A half that still overruns must reach the caller.

        The joined path ORs the flag across windows. Nothing else asserts the
        True side, so a port returning a constant False passed.
        """
        engine, _, algo = self.fake_engine()
        result = engine.synthesize(self._text(algo), fake_voice(), seed=7)
        assert result.hit_token_cap is True

    def test_one_chunk_that_overruns_is_repaired_by_public_synthesize(self) -> None:
        """The half of the truncation the chunk plan cannot see.

        The plan is a character budget against a constant, and whether a voice
        fits inside it is a fact about the voice and the text together. A text
        that plans as *one* chunk can still run to the window cap, and until
        `synthesize` covered every length the CLI carried the repair by
        catching `WindowOverflowError` and calling a second method. Nothing
        else here asserts that the repair reaches a caller who never mentioned
        chunking: `stream` covers the multi-chunk case, and the single-chunk
        case went out with those CLI tests.
        """
        from loudkit.errors import WindowOverflowError
        from loudkit.frontend.chunking import split_in_half, split_text

        engine, _, algo = self.fake_engine()
        text = "one two three four"
        assert len(split_text(text, algo.chunking)) == 1, "this text must plan as one chunk"
        halves = split_in_half(text)
        assert halves is not None

        result = engine.synthesize(text, fake_voice(), seed=7)
        assert [c.text for c in result.chunks] == list(halves), (
            "a one-chunk text that overran the window must come back as both halves"
        )
        with pytest.raises(WindowOverflowError):
            engine.synthesize(text, fake_voice(), seed=7, single_window=True)

    def test_the_length_ceiling_does_not_trigger_a_split(self) -> None:
        """Only the window may. Halving a runaway gives two runaways.

        `_generate_window` caps at min(max_new_tokens, ceiling_for(...)), and
        the second fires on short text. Three of 54 cap hits on the roster were
        this, including a seven-character chunk with a space in it.
        """
        from loudkit.postprocess import ceiling_for

        engine, _, algo = self.fake_engine(window=255, budget_tokens=255)
        text = "one two"
        ceiling = ceiling_for(
            len(engine.frontend.encode(text, "en")),
            config=algo.postprocess,
            window=algo.window.max_speech_tokens,
        )
        assert ceiling < algo.window.max_speech_tokens, "the case needs the ceiling to bite"
        out = list(engine.stream(text, fake_voice(), seed=7))
        assert len(out) == 1, "a ceiling hit was split"
        assert out[0].hit_token_cap is True

    def test_a_half_that_still_overruns_is_not_split_again(self) -> None:
        """The generator here never stops, so both halves cap too.

        A recursing implementation would keep halving until the text ran out of
        word boundaries. The count is exactly one split per splittable chunk.
        """
        from loudkit.frontend.chunking import split_in_half, split_text

        engine, _, algo = self.fake_engine()
        text = self._text(algo)
        chunks = split_text(text, algo.chunking)
        expected = sum(2 if split_in_half(c) is not None else 1 for c in chunks)
        out = list(engine.stream(text, fake_voice(), seed=7))
        assert all(r.hit_token_cap for r in out), "the case needs every half to cap"
        assert len(out) == expected, "a half was split a second time"

    def test_an_unsplittable_capped_chunk_ships_as_one(self) -> None:
        engine, _, algo = self.fake_engine()
        out = list(engine.stream("unbrokenrun", fake_voice(), seed=7))
        assert len(out) == 1
        assert out[0].hit_token_cap is True

    def test_off_ships_the_truncation(self) -> None:
        from loudkit.frontend.chunking import split_text

        engine, _, algo = self.fake_engine(law="off")
        text = self._text(algo)
        out = list(engine.stream(text, fake_voice(), seed=7))
        assert len(out) == len(split_text(text, algo.chunking))
        assert all(r.hit_token_cap for r in out)


@pytest.mark.parametrize("single_window", [False, True])
def test_fractional_carry_is_refused_before_generation(single_window: bool) -> None:
    engine = fake_engine()
    with pytest.raises(InvalidTokensError, match="whole number"):
        engine.synthesize(
            "Hello world.",
            fake_voice(),
            previous_tokens=[1.9, 2.1],
            single_window=single_window,
        )
    assert engine.token_generator.calls == []
