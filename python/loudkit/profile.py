"""Profile one passage stage by stage, on this machine.

Where ``loudkit bench`` measures the whole engine on several passages, ``loudkit
profile`` drills into one passage and reports each stage — token generator, mel
decoder, vocoder — separately, as the median of several runs. The two answer
different questions. Bench answers "what will this hardware do"; profile answers
"*where* does the time go", which is the question you ask when a number is worse
than expected and you need to know whether the autoregressive generator, the
flow render, or the vocoder is the bottleneck.

Medians rather than means, because a synthesis has a warm-up run and occasional
scheduling hiccups, and the median is what survives both. The first run is
reported separately (``warm``) precisely because it is not representative — model
warm-up, allocator growth, and first-call dispatch are real but not steady-state.

The load time is measured too: for anything interactive it is the difference
between "fine" and "six seconds of blank", and it is the reason the server exists.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["profile_passage", "ProfileResult", "to_json", "render_table"]


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Per-stage timings for one passage, with load and warm-up."""

    loudkit_version: str
    command: str
    device: str
    execution: str
    algorithm: str
    fingerprint: str
    text: str
    load_s: float
    n_runs: int
    warm_tokens_s: float
    warm_mel_s: float
    warm_audio_s: float
    """The **discarded warm-up run's** per-stage times, reported rather than
    thrown away.

    The warm-up is the run where the allocator, the kernel autotuner and the
    graph capture all pay once. It is not part of the median population —
    reporting it next to a median is the only cold-start information this table
    carries.
    """

    median_tokens_s: float
    median_mel_s: float
    median_audio_s: float
    median_total_s: float
    """Median of each run's **total**, not the sum of the per-stage medians.

    Those differ whenever the stages' slow runs are not the same runs, and the
    sum has no run behind it: it can name a total that never happened, and it
    cannot be reconciled with ``median_rtf``, which is computed per run.
    """

    median_rtf: float
    duration_s: float
    n_tokens: int


def profile_passage(
    engine: Any,
    voice: Any,
    text: str,
    *,
    seed: int = 7,
    runs: int = 5,
    load_s: float = 0.0,
    command: str = "",
) -> ProfileResult:
    """Synthesise ``text`` ``runs`` times and summarise per-stage timing.

    Args:
        engine: a loaded :class:`~loudkit.engine.Engine`.
        voice: a :class:`~loudkit.voice.VoiceProfile`.
        text: the passage to profile.
        seed: seed for every run. Same seed each time on purpose — a stage that
            is slow on this seed is slow; a stage that is fast on a different
            seed was never the bottleneck to begin with.
        runs: how many timed runs after the warm-up. Must be at least 1. The
            warm-up run is always performed and never counted, so ``runs`` is
            the sample size of the medians.
        load_s: the measured model load time, reported through.
    """
    if runs < 1:
        # statistics.median of an empty list raises StatisticsError, whose
        # message names neither this function nor the argument that was wrong.
        raise ValueError(f"runs must be at least 1, got {runs}")

    # The warm-up run: not counted in the medians, but reported as the `warm_*`
    # numbers, because a cold first pass is a real cost a caller may need to
    # plan for. Timing it is the only way the "warm" column means anything
    # other than "one more sample from the same population".
    warm = engine.synthesize(text, voice, seed=seed)

    tokens_s: list[float] = []
    mel_s: list[float] = []
    audio_s: list[float] = []
    total_s: list[float] = []
    duration_s = 0.0
    n_tokens = 0
    rtf: list[float] = []

    for _ in range(runs):
        r = engine.synthesize(text, voice, seed=seed)
        tokens_s.append(r.timings.tokens)
        mel_s.append(r.timings.mel)
        audio_s.append(r.timings.audio)
        total_s.append(r.timings.total)
        rtf.append(r.timings.rtf(r.duration))
        duration_s = r.duration
        n_tokens = len(r.tokens)

    from . import __version__

    med = statistics.median
    return ProfileResult(
        loudkit_version=__version__,
        command=command,
        device=engine.execution.device,
        execution=engine.execution.describe(),
        algorithm=engine.algorithm.describe(),
        fingerprint=engine.algorithm.fingerprint(),
        text=text,
        load_s=load_s,
        n_runs=runs,
        warm_tokens_s=warm.timings.tokens,
        warm_mel_s=warm.timings.mel,
        warm_audio_s=warm.timings.audio,
        median_tokens_s=med(tokens_s),
        median_mel_s=med(mel_s),
        median_audio_s=med(audio_s),
        median_total_s=med(total_s),
        median_rtf=med(rtf),
        duration_s=duration_s,
        n_tokens=n_tokens,
    )


def to_json(result: ProfileResult) -> str:
    """Serialise to the JSON that lands in a profile artefact."""
    import json

    return json.dumps(asdict(result), indent=2, sort_keys=True)


def render_table(result: ProfileResult) -> str:
    """Human-readable report from a :class:`ProfileResult`."""
    header = (
        f"loudkit {result.loudkit_version}  {result.device}\n"
        f"  {result.algorithm}\n"
        f"  {result.execution}\n"
        f"  load {result.load_s:.2f}s  {result.n_runs} timed runs "
        f"(median), warm = the discarded first pass\n"
    )
    rows = [
        ("token generator", result.median_tokens_s, result.warm_tokens_s),
        ("mel decoder", result.median_mel_s, result.warm_mel_s),
        ("vocoder", result.median_audio_s, result.warm_audio_s),
        (
            "total",
            result.median_total_s,
            result.warm_tokens_s + result.warm_mel_s + result.warm_audio_s,
        ),
    ]
    lines = [header, f"{'stage':<16}{'median':>10}{'warm':>10}"]
    for name, median, warm in rows:
        lines.append(f"{name:<16}{median:>10.3f}s{warm:>10.3f}s")
    lines.append("")
    lines.append(
        f"{result.duration_s:.2f}s audio @ RTF {result.median_rtf:.2f}x  "
        f"({result.n_tokens} tokens)"
    )
    lines.append(f"reproduce: {result.command}")
    return "\n".join(lines)
