"""WSOLA: the properties, not the samples.

Nothing here pins a waveform. A time-stretch has no golden output that survives
a change of compiler, and the four ports sum the same floats in their own order
— so what is asserted is what a listener would notice if it broke: the length,
the pitch, the loudness, and the fact that ``speed=1.0`` is not a stretch at all
but a bypass.

The same four properties are asserted in Go, Rust, TypeScript and Swift. A
shared byte-level fixture was considered and rejected: the alignment search
picks between candidates whose cross-correlations can differ in the last bit
across languages, so one offset chosen differently would move every sample after
it, and the fixture would fail for a reason that is not a defect.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from loudkit.models.timestretch import (
    MAX_SPEED,
    MIN_SPEED,
    stretched_length,
    time_stretch,
    validate_speed,
)

SAMPLE_RATE = 24_000
SPEEDS = (0.5, 0.75, 0.9, 1.25, 1.5, 2.0)


def _signal(seconds: float = 1.0, f0: float = 220.0) -> np.ndarray:
    """A voiced-ish test signal: a low fundamental, a harmonic, and a sweep.

    Deterministic by construction — no RNG anywhere in this module, because the
    stretcher has none either and a flaky DSP test is worse than no DSP test.
    """
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float64) / SAMPLE_RATE
    wave = (
        0.5 * np.sin(2 * math.pi * f0 * t)
        + 0.25 * np.sin(2 * math.pi * 2 * f0 * t)
        + 0.15 * np.sin(2 * math.pi * 600 * t * (1 + t))
    )
    return wave.astype(np.float32)


def _pitch_hz(x: np.ndarray) -> float:
    """Fundamental by autocorrelation. Enough to catch a chipmunk."""
    window = x[SAMPLE_RATE // 4 : SAMPLE_RATE // 4 + 4096].astype(np.float64)
    window = window - window.mean()
    ac = np.correlate(window, window, "full")[len(window) - 1 :]
    lo, hi = int(SAMPLE_RATE / 500), int(SAMPLE_RATE / 80)
    return SAMPLE_RATE / (lo + int(np.argmax(ac[lo:hi])))


class TestUnitySpeedIsABypass:
    def test_the_same_array_comes_back(self) -> None:
        """Identity, not equality. The engine's default must not depend on a
        DSP path being lossless — it must not enter the DSP path at all."""
        x = _signal(0.3)
        assert time_stretch(x, sample_rate=SAMPLE_RATE, speed=1.0) is x

    def test_bit_identical_through_the_engine(self) -> None:
        """The claim that matters: every existing caller keeps their bytes."""
        from .test_engine import _engine, _voice

        engine = _engine()
        without = engine.synthesize("One two three.", _voice(), seed=3)
        explicit = engine.synthesize("One two three.", _voice(), seed=3, speed=1.0)
        assert np.array_equal(without.audio, explicit.audio)
        assert without.speed == explicit.speed == 1.0


class TestLength:
    @pytest.mark.parametrize("speed", SPEEDS)
    def test_the_output_is_exactly_as_long_as_asked(self, speed: float) -> None:
        x = _signal(1.0)
        got = time_stretch(x, sample_rate=SAMPLE_RATE, speed=speed)
        assert got.shape == (stretched_length(len(x), speed),)

    @pytest.mark.parametrize(
        ("n", "speed", "expected"),
        [
            (5, 2.0, 3),  # 2.5 -> 3, not 2: half-up, not half-even
            (3, 2.0, 2),  # 1.5 -> 2, where half-even would also give 2
            (7, 2.0, 4),  # 3.5 -> 4, where half-even would give 4
            (9, 2.0, 5),  # 4.5 -> 5, where half-even would give 4
            (5, 0.5, 10),
            (3, 0.5, 6),
            (24_000, 1.25, 19_200),
            (24_000, 1.5, 16_000),
            (1_000, 0.8, 1_250),
            (1, 2.0, 1),  # 0.5 -> 1, where half-even would give 0
        ],
    )
    def test_the_length_formula_is_pinned_by_hand(
        self, n: int, speed: float, expected: int
    ) -> None:
        """Hand-computed, not computed by the function under test.

        The length assertions elsewhere in this file compare `got.shape` against
        `stretched_length(...)` — the same function the implementation calls, so
        a wrong formula would agree with itself and pass. This table is the
        independent pin, and it covers the halves in both directions: Python
        rounds them to even and Go, Rust, Swift and JavaScript do not, which is
        why the formula is written as `floor(x + 0.5)` in all five and never as
        the language's `round()`.
        """
        assert stretched_length(n, speed) == expected

    def test_a_sample_rate_too_low_to_have_a_hop_does_not_hang(self) -> None:
        """The guard that no implementation tested, which is how two of the five
        shipped without it.

        Below ~60 Hz the derived frame is one sample, so the hop — frame // 2 —
        is zero, and the overlap-add loop advances by it. TypeScript and Swift
        computed the hop *after* the degenerate-shape guard and never tested it,
        so both looped forever on an input Python returned from in microseconds;
        nothing was red, because nothing asked. Now all five ask.

        Asserted with a wall-clock bound as well as a length: a regression here
        is a hang, and a hang is a suite that never finishes rather than a suite
        that fails.
        """
        started = time.perf_counter()
        got = time_stretch(np.zeros(64, np.float32), sample_rate=40, speed=1.5)
        assert time.perf_counter() - started < 1.0, "the overlap-add loop did not terminate"
        assert got.shape == (stretched_length(64, 1.5),) == (43,)

    def test_a_fragment_shorter_than_a_frame_is_still_the_right_length(self) -> None:
        """No overlap to align, so it is cut or padded. At 24 kHz this is under
        25 ms — below anything the engine renders, and the alternative is a
        crash on the degenerate case."""
        tiny = np.ones(64, np.float32)
        assert time_stretch(tiny, sample_rate=SAMPLE_RATE, speed=2.0).shape == (32,)
        assert time_stretch(tiny, sample_rate=SAMPLE_RATE, speed=0.5).shape == (128,)


class TestItIsAStretchAndNotAResample:
    @pytest.mark.parametrize("speed", SPEEDS)
    def test_pitch_is_preserved(self, speed: float) -> None:
        """The entire point. A resampler would move the fundamental by exactly
        ``speed``; this must not move it at all."""
        x = _signal(2.0)
        got = time_stretch(x, sample_rate=SAMPLE_RATE, speed=speed)
        assert _pitch_hz(got) == pytest.approx(_pitch_hz(x), rel=0.03)

    @pytest.mark.parametrize("speed", SPEEDS)
    def test_loudness_survives(self, speed: float) -> None:
        """Overlap-add with a window that does not sum to one is the classic
        way to get a 6 dB drop or a comb filter. The periodic Hann at 50 %
        overlap sums to one, and the denominator corrects the ends."""
        x = _signal(1.0)
        got = time_stretch(x, sample_rate=SAMPLE_RATE, speed=speed)
        rms_in = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
        rms_out = float(np.sqrt(np.mean(np.square(got.astype(np.float64)))))
        assert rms_out == pytest.approx(rms_in, rel=0.15)

    @pytest.mark.parametrize("speed", SPEEDS)
    def test_nothing_clips_or_goes_non_finite(self, speed: float) -> None:
        got = time_stretch(_signal(1.0), sample_rate=SAMPLE_RATE, speed=speed)
        assert np.all(np.isfinite(got))
        assert float(np.max(np.abs(got))) <= 1.2

    def test_silence_stays_silent(self) -> None:
        """The correlation search divides by candidate energy; a silent frame
        is where that division has to not happen."""
        silence = np.zeros(SAMPLE_RATE, np.float32)
        got = time_stretch(silence, sample_rate=SAMPLE_RATE, speed=1.5)
        assert np.all(got == 0.0)


class TestDeterminism:
    def test_two_calls_agree_bit_for_bit(self) -> None:
        """No RNG, no adaptivity, no wall-clock. Same in, same out, forever."""
        x = _signal(1.0)
        first = time_stretch(x, sample_rate=SAMPLE_RATE, speed=1.4)
        second = time_stretch(x, sample_rate=SAMPLE_RATE, speed=1.4)
        assert np.array_equal(first, second)


class TestValidation:
    @pytest.mark.parametrize("speed", [0.49, 2.01, 0.0, -1.0, 10.0])
    def test_out_of_range_is_refused_not_clamped(self, speed: float) -> None:
        """A caller who asked for 3x and silently got 2x has a bug only a
        stopwatch finds."""
        with pytest.raises(ValueError, match=r"outside \[0.5, 2.0\]"):
            validate_speed(speed)

    def test_nan_is_refused_before_the_range_check(self) -> None:
        """``nan`` compares false against both bounds, so a naive range test
        would let it through and produce an empty waveform."""
        with pytest.raises(ValueError, match="finite"):
            validate_speed(float("nan"))

    @pytest.mark.parametrize("speed", [MIN_SPEED, 1.0, MAX_SPEED])
    def test_the_bounds_themselves_are_allowed(self, speed: float) -> None:
        assert validate_speed(speed) == speed

    def test_the_engine_refuses_before_it_generates(self) -> None:
        """Six seconds of generation should not happen to discover a typo in a
        keyword argument."""
        from .test_engine import _engine, _voice

        engine = _engine()
        with pytest.raises(ValueError, match="outside"):
            engine.synthesize("One two.", _voice(), seed=1, speed=4.0)
        assert engine.token_generator.calls == []  # type: ignore[attr-defined]


class TestThroughTheEngine:
    def test_the_result_records_what_was_asked_for(self) -> None:
        from .test_engine import _engine, _voice

        result = _engine().synthesize("One two three four.", _voice(), seed=1, speed=1.5)
        assert result.speed == 1.5
        assert "speed=1.5x" in repr(result)

    def test_timings_describe_the_stretched_waveform(self) -> None:
        """Computed after the stretch, on the audio the caller receives — so
        there is no ``1/speed`` correction to apply, and applying one would
        double-count."""
        from .test_engine import _engine, _voice

        result = _engine().synthesize("One two three four.", _voice(), seed=1, speed=2.0)
        assert result.chunks[-1].end == result.duration
        assert result.chunks[-1].words[-1].end == pytest.approx(result.duration)

    def test_long_form_stretches_every_chunk(self) -> None:
        """Per chunk, like the seeds and the prefix: a chunk's audio must not
        depend on how many came before it."""
        from dataclasses import replace

        from loudkit.config import AlgorithmConfig, ChunkConfig, SamplingConfig

        from .test_engine import _engine, _voice

        algo = AlgorithmConfig().with_(
            chunking=ChunkConfig(max_tokens=20, prefix_tokens=0),
            sampling=SamplingConfig(max_new_tokens=64),
        )
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        plain = _engine(algo).synthesize_long(text, _voice(), seed=1)
        fast = _engine(algo).synthesize_long(text, _voice(), seed=1, speed=2.0)
        assert len(fast.chunks) == len(plain.chunks) > 1
        assert fast.duration == pytest.approx(plain.duration / 2, rel=0.01)
        assert fast.speed == 2.0
        # And the join is still exact, on the stretched spans.
        for left, right in zip(fast.chunks, fast.chunks[1:], strict=False):
            assert left.end == right.start
        assert replace(fast, speed=2.0).speed == 2.0
