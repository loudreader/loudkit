"""bench and profile — the measuring commands, exercised without weights.

These commands wrap ``StageTimings`` and the determinism check, so the logic
worth testing without a 1.27 GB checkpoint is the accounting: samples are
collected per passage, medians over several runs are computed and are medians
rather than means, the determinism verdict reflects the bytes actually produced,
and the JSON round-trips. The fakes here are the same ones test_engine uses —
they compute nothing meaningful, but they are deterministic, which is exactly
what the determinism check needs to be exercised.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from loudkit.bench import Sample, bench, render_table, run_bench, to_json
from loudkit.config import AlgorithmConfig
from loudkit.contracts import Mel, Sampler, SpeechTokens, Waveform
from loudkit.engine import Engine
from loudkit.profile import profile_passage
from loudkit.voice import VoiceProfile


def _voice() -> VoiceProfile:
    return VoiceProfile(
        name="fake",
        speaker_embedding=np.full(256, 0.0625, np.float32),
        flow_embedding=np.full(192, 0.0625, np.float32),
        prompt_tokens=np.zeros(8, np.int64),
        prompt_mel=np.zeros((80, 16), np.float32),
        cond_prompt_tokens=np.zeros(8, np.int64),
    )


class _FakeFrontend:
    def encode(self, text: str, language: str = "en") -> np.ndarray:
        return np.arange(len(text.split()), dtype=np.int64)


class _FakeGenerator:
    step_delay_s = 0.002
    """Cost of one decode step.

    A real forward pass takes milliseconds; a fake that returns instantly makes
    "wait for the next poll" free, and a cancel measurement that skips that
    wait would still look correct. Small enough not to slow the suite, large
    enough to be well clear of timer noise."""

    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config
        self.polls = 0

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
        # Polls per token, exactly as the torch generator does. A fake that
        # ignores should_cancel would make every cancellation look like a
        # boundary cancellation and hide the case this file measures.
        n = max(1, len(text_tokens))
        out: list[int] = []
        for i in range(n):
            if should_cancel is not None and should_cancel():
                return out
            self.polls += 1
            time.sleep(self.step_delay_s)  # a forward pass is not free
            out.append(i)
        out.append(self.config.stop_speech_token)
        return out

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
        frontend=_FakeFrontend(),
        token_generator=_FakeGenerator(algo),
        mel_decoder=_FakeMelDecoder(algo),
        vocoder=_FakeVocoder(algo),
        algorithm=algo,
    )


def test_run_bench_collects_one_sample_per_text() -> None:
    engine = _engine()
    result = run_bench(engine, _voice(), texts=["one two three", "four five six seven"], seed=7)
    assert len(result.samples) == 2
    assert all(isinstance(s, Sample) for s in result.samples)
    assert all(s.rtf > 0 for s in result.samples)
    assert result.deterministic is True
    assert result.fingerprint == engine.algorithm.fingerprint()


def test_determinism_check_detects_drift() -> None:
    """A generator whose output changes between identical calls must fail it."""

    class Drifting:
        def __init__(self, inner) -> None:
            self._inner = inner
            self._count = 0

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

        def synthesize(self, *a, **kw):  # type: ignore[no-untyped-def]
            from dataclasses import replace

            r = self._inner.synthesize(*a, **kw)
            self._count += 1
            if self._count == 2:  # the first determinism re-check drifts
                return replace(r, audio=r.audio + 1e-6)
            return r

    result = run_bench(Drifting(_engine()), _voice(), texts=["hello"], seed=7)
    assert result.deterministic is False


def test_bench_records_load_time_and_peak_rss() -> None:
    engine = _engine()
    result = bench(engine, _voice(), texts=["hello"], load_s=1.25)
    assert result.load_s == 1.25
    assert result.peak_rss > 0
    assert result.rss_unit in ("bytes", "kB")


def test_bench_preserves_every_field_run_bench_set() -> None:
    """`bench()` must not drop fields on its way to adding load_s.

    It used to rebuild BenchResult field by field, naming all twelve. A field
    added to the dataclass then either broke this path loudly (no default) or,
    worse, reverted to its default here while `run_bench` returned it
    correctly — a divergence no test could see. `host` was the field that
    caught it. Comparing the whole record rather than one name means the next
    field added is covered without editing this test."""
    from dataclasses import replace

    engine = _engine()
    plain = run_bench(engine, _voice(), texts=["hello"], seed=7)
    loaded = bench(engine, _voice(), texts=["hello"], load_s=1.25)
    # Everything except the three fields that legitimately differ: load_s is the
    # point of bench(), samples carry wall-clock timings from a second run, and
    # peak_rss is a measurement of the process rather than of the result. The
    # second run allocates, so the peak can only have risen by the time it is
    # read — 249552896 to 249561088 bytes, two pages, on the Windows run that
    # first showed this. Pinning it to a constant is what keeps the comparison
    # about dropped fields, which is the whole subject of this test.
    normalise = {"load_s": 0.0, "samples": [], "peak_rss": 0}
    assert replace(loaded, **normalise) == replace(plain, **normalise)


def test_host_names_the_machine_not_just_the_device() -> None:
    """A row must say which computer produced it.

    `device: cpu` names an abstraction: out/rows/ and out/rows_s1/ each held a
    cpu row, one an Apple M3 Pro and one a 2016 i7, and nothing in either file
    distinguished them — the attribution lived only in a markdown table."""
    result = run_bench(_engine(), _voice(), texts=["hello"], seed=7)
    assert result.host, "no host recorded"
    # Hostname, OS and CPU brand at minimum, so two machines cannot collide.
    assert len(result.host.split(" | ")) >= 3, result.host
    assert json.loads(to_json(result))["host"] == result.host


def test_json_round_trips() -> None:
    result = run_bench(_engine(), _voice(), texts=["one two"], seed=7)
    blob = json.loads(to_json(result))
    assert blob["fingerprint"] == result.fingerprint
    assert len(blob["samples"]) == 1
    assert blob["samples"][0]["n_tokens"] == 2  # fake emits one token per word
    # TTFA is recorded per sample (the streaming-path time to first audio).
    assert "ttfa_s" in blob["samples"][0]
    assert blob["samples"][0]["ttfa_s"] >= 0.0


def test_cancel_latency_is_measured_mid_generation() -> None:
    """The barge-in number must time an interrupt, not an empty loop.

    ``should_cancel=lambda: True`` is caught by ``Engine.stream``'s pre-chunk
    check before a single token is decoded, so the old measurement timed loop
    entry and exit and reported ~0 s for every device — a number that looked
    excellent precisely because nothing had happened. The flag now stays false
    long enough for the interrupt to land inside the decode loop.
    """
    from loudkit.bench import _CANCEL_AFTER_POLLS

    engine = _engine()
    long_text = " ".join(f"word{i}" for i in range(_CANCEL_AFTER_POLLS * 3))
    result = run_bench(engine, _voice(), texts=[long_text], seed=7)

    sample = result.samples[0]
    assert sample.cancel_latency_s is not None, (
        "no interrupt was timed — the passage never reached the flip"
    )
    # The generator really decoded before the cancel landed: more polls than
    # the passage has chunks, which is all a pre-chunk-only cancel would show.
    assert engine.token_generator.polls > _CANCEL_AFTER_POLLS  # type: ignore[attr-defined]

    # The interrupt is armed on one poll and honoured on the next, so at least
    # one decode step sits inside the number. An implementation that flips the
    # flag and returns true in the same poll times only the loop unwinding —
    # microseconds, ~30x under the truth, and wrong in the flattering
    # direction. Anchor against the step the fake generator actually takes.
    step_s = engine.token_generator.step_delay_s  # type: ignore[attr-defined]
    assert sample.cancel_latency_s >= step_s, (
        f"cancel latency {sample.cancel_latency_s:.6f}s is under one decode step "
        f"({step_s:.6f}s) — the wait for the next poll is missing from the measurement"
    )


def test_render_table_contains_reproduce_line() -> None:
    result = run_bench(_engine(), _voice(), texts=["one"], seed=7)
    table = render_table(result)
    assert "reproduce:" in table
    assert "RTF" in table


def test_profile_reports_medians_and_warmup_separately() -> None:
    engine = _engine()
    result = profile_passage(engine, _voice(), "one two three", seed=7, runs=4)
    assert result.n_runs == 4
    # Medians must be actual medians, not means. The fake vocoder is instant, so
    # the interesting assert is structural: warm timing is a single sample and
    # the median is computed over the timed runs.
    assert result.median_total_s > 0
    assert result.warm_audio_s >= 0
    # A warm-up run and the timed runs are separate measurements — no ordering
    # between them is claimed, and the `... or True` that used to stand here
    # asserted nothing at all while looking like it did. What *is* claimable is
    # that the warm sample is a real measurement of the same work, not a
    # leftover zero.
    assert result.warm_tokens_s > 0
    assert result.median_tokens_s > 0


def test_profile_total_is_a_median_of_totals_not_a_sum_of_medians() -> None:
    """Those differ, and the sum can name a total no run ever took.

    The runs below take 3 s, 12 s and 12 s, so the median run took 12 s. Each
    stage's median is 1 s, because no stage is slow in the *same* run as
    another — so the sum of per-stage medians is 3 s, a total that only one run
    ever had and not the middle one. The sum also cannot be reconciled with
    median_rtf, which is computed per run.
    """
    import statistics

    from loudkit.engine import Result, StageTimings

    engine = _engine()
    voice = _voice()
    real = engine.synthesize("one two", voice, seed=7)

    # Three runs with deliberately anti-correlated stage times.
    timings = [(1.0, 1.0, 1.0), (10.0, 1.0, 1.0), (1.0, 10.0, 1.0)]
    scripted = iter(
        Result(
            audio=real.audio,
            tokens=real.tokens,
            mel=real.mel,
            seed=7,
            sample_rate=real.sample_rate,
            timings=StageTimings(*t),
            algorithm_fingerprint=real.algorithm_fingerprint,
        )
        # The warm-up consumes one before the timed runs begin.
        for t in [(0.0, 0.0, 0.0), *timings]
    )

    class _Scripted:
        algorithm = engine.algorithm
        execution = engine.execution

        def synthesize(self, *_a, **_k):
            return next(scripted)

    result = profile_passage(_Scripted(), voice, "one two", seed=7, runs=len(timings))
    assert result.median_total_s == statistics.median(sum(t) for t in timings)
    sum_of_medians = sum(statistics.median(stage) for stage in zip(*timings, strict=True))
    assert result.median_total_s != sum_of_medians


def test_profile_rejects_zero_runs() -> None:
    """`statistics.median([])` raises a StatisticsError that names neither this
    function nor the argument that was wrong."""
    with pytest.raises(ValueError, match="runs must be at least 1"):
        profile_passage(_engine(), _voice(), "one", runs=0)


def test_profile_json_round_trips() -> None:
    from loudkit.profile import to_json as profile_to_json

    result = profile_passage(_engine(), _voice(), "hello world", runs=3)
    blob = json.loads(profile_to_json(result))
    assert blob["n_runs"] == 3
    assert blob["text"] == "hello world"
