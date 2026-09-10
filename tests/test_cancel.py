"""The cancellation contract, held without weights.

``synthesize`` returns the whole passage or raises ``CancelledError``; it never
hands back the chunks that finished. ``stream`` yields the chunks that finished
before the cancel and nothing after, and raises nothing: the caller flipped
the flag. The five implementations hold the same contract; each port's cancel
test cancels at a decode step and asserts the same two things.
"""

from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest

from loudkit.config import AlgorithmConfig, SamplingConfig
from loudkit.contracts import Sampler, SpeechTokens
from loudkit.engine import Engine
from loudkit.errors import CancelledError, LoudkitError, error_code
from loudkit.voice import VoiceProfile

from .conftest import FakeFrontend, FakeGenerator, FakeMelDecoder, FakeVocoder, fake_voice

_PASSAGE = "one two three four. five six seven eight. nine ten eleven twelve."


class _SteppingGenerator(FakeGenerator):
    """Polls ``should_cancel`` once per token, as the real decode loops do, and
    counts the steps it ran.

    ``gate``, when set, holds every window after the first until the test
    opens it: the pipeline generates window 1 while window 0 renders, and a
    fake that finishes instantly would otherwise race the consumer for it.
    """

    def __init__(self, config: AlgorithmConfig) -> None:
        super().__init__(config)
        self.steps = 0
        self.gate: threading.Event | None = None

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
        self.calls.append({"n_text": len(text_tokens), "prefix": list(prefix)})
        if len(self.calls) > 1 and self.gate is not None:
            assert self.gate.wait(timeout=5.0), "the consumer never took window 0"
        out: list[int] = []
        for i in range(max(1, len(text_tokens))):
            if should_cancel is not None and should_cancel():
                # Raised, not returned short, because that is what the real
                # loops and the four ports do. A fake that returns the partial
                # models a contract nothing implements, and would keep this
                # suite green if a loop went back to it.
                raise CancelledError(f"at decode step {i}: cancelled")
            self.steps += 1
            out.append(i)
        out.append(self.config.stop_speech_token)
        return out


def _engine() -> tuple[Engine, _SteppingGenerator]:
    """Four chunks of three or four tokens from ``_PASSAGE``, no carry between them."""
    algo = AlgorithmConfig().with_(
        chunking=replace(AlgorithmConfig().chunking, max_tokens=40, prefix_tokens=0),
        sampling=SamplingConfig(max_new_tokens=64),
    )
    gen = _SteppingGenerator(algo)
    engine = Engine(
        frontend=FakeFrontend(),
        token_generator=gen,
        mel_decoder=FakeMelDecoder(algo),
        vocoder=FakeVocoder(algo),
        algorithm=algo,
    )
    return engine, gen


def _cancel_at(gen: _SteppingGenerator, step: int):
    return lambda: gen.steps >= step


class TestTheError:
    def test_it_is_a_loudkit_error_with_a_code(self) -> None:
        """Transports map by ``code``; ``cancelled`` is the word they send."""
        assert issubclass(CancelledError, LoudkitError)
        assert error_code(CancelledError("x")) == "cancelled"


class TestAFlagThatIsTrueOnlyOnce:
    """The refusal must land where the cancel was seen, not at a later poll.

    A caller whose flag is consumed by its first reader is ordinary: a toggle
    set by a barge-in and cleared when something reads it. If the decode loop
    returned its partial row instead of raising, the only thing standing
    between that row and the caller was a second poll after the loop, which
    such a flag does not survive. The truncated row then reached the
    detectors, the retry ladder fired, and the call returned different speech
    rather than refusing.
    """

    def test_a_one_shot_cancel_refuses_rather_than_speaking_something_else(self) -> None:
        engine, gen = _engine()
        seen = []

        def once() -> bool:
            if gen.steps >= 2 and not seen:
                seen.append(True)
                return True
            return False

        with pytest.raises(CancelledError):
            engine.synthesize(_PASSAGE, fake_voice(), should_cancel=once)
        assert seen, "the flag was never read"


class TestSynthesize:
    def test_a_cancel_at_a_decode_step_raises_and_returns_nothing(self) -> None:
        """Cancelled inside chunk 1: chunk 0 finished, and it is not the answer.

        Drift: joining the finished chunks turns this into a ``Result`` with
        fewer chunks than the text has.
        """
        engine, gen = _engine()
        whole = engine.synthesize(_PASSAGE, fake_voice(), seed=1)
        assert len(whole.chunks) > 1, "the case needs a passage that splits"
        step = whole.chunks[0].tokens + 2

        engine, gen = _engine()
        with pytest.raises(CancelledError):
            engine.synthesize(
                _PASSAGE, fake_voice(), seed=1, should_cancel=_cancel_at(gen, step)
            )
        assert gen.steps == step, "the cancel did not land at the step it was asked for"

    def test_a_cancel_already_true_costs_no_decode_step(self) -> None:
        engine, gen = _engine()
        with pytest.raises(CancelledError):
            engine.synthesize(_PASSAGE, fake_voice(), seed=1, should_cancel=lambda: True)
        assert gen.steps == 0

    def test_one_window_raises_the_same(self) -> None:
        engine, gen = _engine()
        with pytest.raises(CancelledError):
            engine.synthesize(
                "one two three",
                fake_voice(),
                seed=1,
                single_window=True,
                should_cancel=_cancel_at(gen, 2),
            )

    def test_a_cancel_during_the_render_raises_too(self) -> None:
        """A flag that turns true after generation, while the mel decodes:
        the vocoder is not started and nothing is returned."""
        engine, _ = _engine()
        flag = {"on": False}
        inner = engine.mel_decoder.decode

        def decode_then_cancel(*a, **kw):
            mel = inner(*a, **kw)
            flag["on"] = True
            return mel

        engine.mel_decoder.decode = decode_then_cancel
        with pytest.raises(CancelledError):
            engine.synthesize(
                "one two three",
                fake_voice(),
                seed=1,
                single_window=True,
                should_cancel=lambda: flag["on"],
            )

    def test_a_caller_that_does_not_cancel_gets_the_whole_passage(self) -> None:
        engine, _ = _engine()
        assert len(engine.synthesize(_PASSAGE, fake_voice(), seed=1).chunks) == 4


class TestStream:
    def test_the_chunks_before_the_step_arrive_and_nothing_after(self) -> None:
        """Cancelled inside chunk 1: chunk 0 is yielded as it was, chunk 1 is
        never rendered, chunk 2 is never generated.

        Drift: rendering the partial chunk, or raising from ``stream``, turns
        this red.
        """
        engine, _ = _engine()
        full = list(engine.stream(_PASSAGE, fake_voice(), seed=1))
        assert len(full) == 4
        step = len(full[0].tokens) + 2

        engine, gen = _engine()
        gen.gate = threading.Event()
        decoder = engine.mel_decoder
        got = []
        for chunk in engine.stream(
            _PASSAGE, fake_voice(), seed=1, should_cancel=_cancel_at(gen, step)
        ):
            got.append(chunk)
            gen.gate.set()
        assert len(got) == 1, (
            f"{len(got)} chunks arrived; only the one before the cancel should"
        )
        assert got[0].tokens == full[0].tokens
        np.testing.assert_array_equal(got[0].audio, full[0].audio)
        assert gen.steps == step
        assert len(decoder.seeds) == 1, "the partial chunk was rendered"

    def test_a_cancel_already_true_yields_nothing(self) -> None:
        engine, gen = _engine()
        assert (
            list(engine.stream(_PASSAGE, fake_voice(), seed=1, should_cancel=lambda: True))
            == []
        )
        assert gen.steps == 0


@pytest.mark.parametrize("mode", ["single_window", "synthesize", "stream"])
def test_cancel_during_final_vocoder_discards_inflight_result(mode: str) -> None:
    engine, _ = _engine()
    flag = {"on": False}
    original = engine.vocoder.synthesize

    def vocode_then_cancel(*args, **kwargs):
        audio = original(*args, **kwargs)
        flag["on"] = True
        return audio

    engine.vocoder.synthesize = vocode_then_cancel
    if mode == "stream":
        assert (
            list(engine.stream("Hello world.", fake_voice(), should_cancel=lambda: flag["on"]))
            == []
        )
    else:
        with pytest.raises(CancelledError):
            engine.synthesize(
                "Hello world.",
                fake_voice(),
                single_window=mode == "single_window",
                should_cancel=lambda: flag["on"],
            )
