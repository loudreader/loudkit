"""Postprocess: deciding where a generated chunk actually ended.

This is the third layer of ``preprocess -> tts -> postprocess``, and it is a
**detector**, not a filter. It looks at the speech tokens a chunk produced and
answers one question — *where did the sentence really stop?* — then hands back a
verdict. It never touches a sample of audio.

Why the token domain and not the audio domain
---------------------------------------------

The artifact this layer exists to remove is not spectral. It is *generated*.

The decoder is free-running: nothing in the model guarantees it stops when the
sentence is over. Silence tokens are exempt from both the repetition penalty and
the ``min_p`` cutoff (see :class:`~loudkit.config.SamplingConfig` — a reader
pauses repeatedly, and penalising silence measurably removes pauses), so once
the sentence is genuinely finished those tokens keep probability mass
indefinitely. The decoder free-runs silence, and **any step where a non-silence
token survives the cutoff becomes a hallucinated word**. What a listener reports
is "it finished the sentence, then a long gap, then one random word".

That failure has a shape in tokens — a run of silence with a short burst behind
it — and no reliable shape in the spectrum. A denoiser cannot find it; a
threshold on energy cannot tell a hallucinated word from a real one. So the
evidence is read where the evidence is.

It also makes the layer portable. Token counts and set membership are integers,
so five implementations can agree exactly rather than to a tolerance.

Where the failure comes from
----------------------------

Field traces on the reading app (2026-08-08) found both reported artifact
sentences rendered clean interactively and broke **only in the batched preload
pass**, where a short text rides padded to its longest neighbour. Batching is
where this bites hardest, which is exactly where a server puts it.

The rules
---------

Six evidence rules, each from a specific failure, applied in a fixed
precedence. See ``docs/reference/postprocess.md`` for the full provenance of every
constant; the short version lives on each function here.

Precedence, and why it is this one:

1. ``repetition`` — first, because it is the only rule that knows *exactly*
   where the failure began. Every other anchor is inferred from a signal that
   might mean something else; an exactly repeated cycle is not.
2. ``silence_tail`` — the peak-anchored rescue, on a row that never said it was
   finished.
3. ``terminal_echo`` — the other peak-anchored rescue, for a tail with no
   silence seam to anchor on.
4. ``desperation`` — the length-anchored one, applied last because it is the
   bluntest, and applied to *ended* rows too: a model that babbles past its
   sentence and only then samples a stop token has forfeited the trust that
   stopping implies.
5. ``ended_tail`` — only when nothing above fired and the row ended cleanly.
   An ended row keeps its own decision; only stranded dead air comes off.

A row that survives all five but is still impossibly long for its text is
reported as ``suspect``. It is not cut — no token anchor agreed on where — but
the caller is told, because silently shipping a row that is 4.5x its plausible
length is how the artifact reached listeners in the first place.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "PostprocessConfig",
    "Inspection",
    "Reason",
    "PostprocessMode",
    "ceiling_for",
    "is_trailing_filler",
    "pacing_outliers",
    "repetition_cut",
    "desperation_cut",
    "ended_tail_trim",
    "terminal_echo_cut",
    "inspect",
]

Reason = Literal[
    "clean",
    "dropout",
    "repetition",
    "silence_tail",
    "terminal_echo",
    "desperation",
    "ended_tail",
]
"""Which rule fired. ``clean`` means none did."""

PostprocessMode = Literal["off", "report", "trim"]
"""What the engine does with a verdict.

``trim`` applies the cut, which changes the audio and is therefore part of the
algorithm — it travels in the fingerprint like every other audible decision.
``report`` runs the detectors and attaches the verdict without acting on it, for
a caller that would rather hear the artifact than risk a wrong cut. ``off``
skips the detectors entirely.
"""


_FINITE_FIELDS: tuple[str, ...] = (
    "ceiling_speech_per_text_token",
    "trailing_filler_threshold",
    "filler_min_eos_probability",
    "desperation_band_ratio",
    "desperation_speech_per_text_token",
    "echo_strong_eos_probability",
    "echo_weak_eos_probability",
    "pacing_tolerance",
)
"""Every float on :class:`PostprocessConfig`.

Listed rather than derived so that adding a float and forgetting it here
is a visible omission in this file, not a silent gap in the validator —
the same reasoning the ports' hand-written manifest walls now carry a
test for."""


@dataclass(frozen=True, slots=True)
class PostprocessConfig:
    """Constants for the artifact detectors. Algorithm layer.

    Every value here was settled against a device trace or a regression, not
    chosen. They are config rather than module constants for the same reason
    the sampling law is: a port computing with a different number produces
    different audio, and the fingerprint is the only thing that catches a field
    no test covers.
    """

    mode: PostprocessMode = "trim"

    ceiling_speech_per_text_token: float = 4.0
    """Hard stop for generation, as a multiple of the text-token count.

    A guard against a three-word sentence decoding for ten seconds, and nothing
    finer. Device trace of the showcase render::

        t3.overrun  gen=92 ceiling=92 bestEOS=74@0.003 floor=31

    A chunk of ~26 text tokens stopped only because it hit the ceiling, with the
    model's confidence in stopping at 0.003 — mid-sentence, already at 3.5
    speech tokens per text token. Four is comfortably past anything measured.

    NOT the chunker's 2.6. That number is the conservative end of a measured
    1.75–2.35 and is conservative *for budgeting a chunk*, where guessing high
    only wastes window. Here it is the opposite: guessing low cuts a sentence
    off.
    """

    ceiling_slack_tokens: int = 40
    """Slack above the proportional bound, in speech tokens (1.6 s of audio).
    Carries the very short texts, where a ratio alone is unsafe."""

    trailing_filler_threshold: float = 0.7
    """The share of a tail that must be silence before it counts as one."""

    trailing_silence_run_tokens: int = 12
    """An unbroken silence run that marks a structural boundary (~0.5 s at
    25 Hz).

    A hallucinated word at the very end sits *behind* such a seam — silence,
    then a burst of speech tokens. The burst lowers the silence share below
    the share test's threshold, which is why this unbroken-run test exists:
    without it, the audible tails are exactly the ones the rescue refuses to
    cut.
    """

    desperation_band_ratio: float = 2.6
    """Top of the stop-peak acceptance band, as a multiple of the text-token
    count.

    The desperation rescue's fallback anchor cuts at the stop peak only if the
    peak sits where a real read could have ended. Measured reads run
    1.75–2.35 speech tokens per text token, so ``int(ratio * n)`` reaches past
    every legitimate ending while staying well under the 4.5x garbage
    threshold — a row whose best stop lands far below its own length has no
    honest end in sight, and the seam rule handles that case instead.
    """

    desperation_band_floor: int = 12
    """Slack added above the proportional band, in speech tokens (~0.5 s).

    Carries the short texts, where ``ratio * n`` alone would close the band on
    endings a legitimate read had already reached."""

    filler_min_eos_probability: float = 0.05
    """How confident the model's best stop must be before the share/run test is
    even consulted. From the EOS-defence bench (``bench_eos_stats.py``,
    variant B): a peak worth trusting."""

    desperation_speech_per_text_token: float = 4.5
    """Past this ratio the row certainly contains garbage, whatever its EOS
    confidence said.

    "It was as he expected." — 14 text tokens — came back as 96 speech tokens of
    sentence-then-dense-babble, with the stop peak at the right *place* (45) but
    confidence 0.000, so every probability-gated rescue refused. Measured real
    speech runs 1.75–2.35 speech tokens per text token; 4.5x is unreachable by
    any legitimate read, so past it the question is no longer whether to cut but
    where.
    """

    desperation_min_text_tokens: int = 10
    """Tiny texts are exempt: fixed overheads (initial breath, final pause) give
    a clean "No!" a ratio of 6+ all by itself."""

    ended_tail_silence_run: int = 6
    """Silence before a blip counts as stranding it (~0.24 s)."""

    ended_tail_blip_max: int = 2
    """<= 80 ms of "speech" is a click, not a word. A real word is never 1–2
    tokens, so speech is untouchable above this."""

    ended_tail_word_max: int = 10
    """~0.4 s: a word, not a clause. A stray word behind a full silence seam on
    a *terminal* chunk is cut with it — prose does not resume after that much
    dead air with a single word. Continuation chunks keep their tails; their
    pauses are the sentence's rhythm and their "end" is not an end."""

    filler_max_speech_after_run: int = 10
    """~0.4 s: how much speech may follow a silence seam and still count as a
    hallucinated word rather than a continuing clause.

    Deliberately a separate field from ``ended_tail_word_max`` despite holding
    the same number and meaning the same duration. They govern different rows —
    this one every row with a seam, that one terminal ended rows — and were
    settled separately. Folding them into one field would mean loosening the
    trim on terminal chunks silently loosened the filler test everywhere.
    """

    ended_tail_keep: int = 5
    """~0.2 s of pause left in place after trimming."""

    echo_strong_eos_probability: float = 0.10
    echo_strong_max_tail: int = 30
    echo_strong_min_position_pct: int = 68
    """The ordinary terminal echo: a stop the model was confident about, late in
    the row, with at most ~1.2 s — two words — after it. The position rule is
    what keeps a real comma or clause pause from being read as an ending."""

    echo_weak_eos_probability: float = 0.003
    echo_weak_max_tail: int = 16
    echo_weak_min_position_pct: int = 85
    """The narrow second path, for one measured regression ("...but a brigand.
    Pass. Four.": ``gen=124/124, bestEOS=109@0.004``). Confidence this weak is
    untrustworthy on its own and is accepted only with every corroborator at
    once: a terminal chunk, a real ceiling overrun, and a tail inside the last
    15% of the row."""

    retry_max_attempts: int = 2
    """How many re-rolls a condemned window may get before shipping as is.

    The literature's measured shape: catastrophic-failure rates drop from 5.8%
    to zero at a single retry — on the failing rows. Retrying every row costs
    N× compute; retrying only the rows the detectors condemned costs ~1.1× at
    the measured fire rates, which is the whole argument for having detectors
    that report rather than guess.

    Only the verdicts nothing can trim retry: ``dropout`` (the content is
    missing) and ``suspect`` (certainly wrong, nowhere to cut). A trimmed
    verdict already has its fix. Each attempt draws from a **derived** seed, so
    the whole ladder is a pure function of the caller's seed — reproducibility
    is why this is config rather than a loop someone writes around the engine.

    Zero disables it.
    """

    pacing_tolerance: float = 1.6
    """How far a chunk's pace may drift from the passage's median before it is
    flagged (as a multiplicative factor, both directions).

    Pace is speech tokens per text token — the same integer-exact quantity the
    length rules use. A long passage is rendered chunk by chunk, and a chunk
    whose pace lands far from its neighbours' is rushing or dragging: the
    long-form drift the literature reports as prosody breakdown past the
    training window. Report-only by design — a pace is a property of a healthy
    render too, and cutting on it would be guessing.

    1.6 is calibrated from the nine-language probe: healthy per-chunk ratios
    run 1.69–2.79 (a 1.65x spread across *languages*), so a chunk drifting
    1.6x from its own passage's median is outside anything ordinary prose
    produced in any of the twelve.
    """

    repetition_max_period: int = 12
    """Longest cycle, in tokens (~0.5 s), that counts as a stuck decoder.

    Above this a repeated block is a *phrase*, and a repeated phrase is rhetoric
    — "no, no, no", a stammer, a refrain. Below it the model is emitting the same
    fragment over and over because its own output has become its context, which
    is a different event with a different fix.
    """

    repetition_min_cycles: int = 3
    """How many consecutive identical cycles before it can be a loop.

    Two is a repeated phrase, which is ordinary speech. Three is the smallest
    count that is not, and it is a necessary condition rather than a sufficient
    one — `repetition_min_span` is what actually separates the classes.
    """

    dropout_min_tokens: int = 25
    """Below this a row is too short to be the sentence it was asked for (~1.0 s).

    The failure is early truncation, and it is the most damaging one in the set
    because content is *missing*: no trim recovers it, and unlike a hallucinated
    tail the listener cannot tell that anything went wrong. The published
    criterion for a catastrophic neural-codec TTS failure uses exactly this
    shape — a speech-token count under 25, or an ASR transcript of at most one
    word — so the threshold is borrowed from a measurement rather than guessed.

    It is a *floor on the row*, not on the text: a genuinely short line is
    exempt, because the shortest legitimate reads measured across nine languages
    run 35 tokens and up, and the rule only fires when the text asked for
    materially more than it got.
    """

    repetition_min_span: int = 24
    """How many tokens the repeating region must cover (~1.0 s).

    Cycle count alone does not separate a stuck decoder from healthy speech:
    this model winds down nearly every row with a short repeated tail token, so
    "does it repeat" is true of almost all real speech. What separates a stuck
    decoder is that the repetition *does not stop*.

    Measured across 27 renders, nine languages, one voice: the longest naturally
    repeating span in a healthy row is 10 tokens (0.4 s), median 7. The one row
    in the set whose decoder genuinely ran away — a Spanish three-word phrase
    that never emitted a stop token — repeats for 44 tokens (1.76 s). 24 sits
    between them with 2.4x margin over the healthy maximum.
    """

    def __post_init__(self) -> None:
        if self.mode not in ("off", "report", "trim"):
            raise ValueError(f"unknown postprocess mode: {self.mode!r}")
        self._validate_ranges()

    def _validate_ranges(self) -> None:  # noqa: PLR0912 - one branch per constant;
        # a validator that groups its fields to satisfy a branch count reads as if
        # the groupings meant something, and they do not.
        # Finite first, because every check below is a comparison and NaN loses
        # all of them: `nan < 0` is False, `nan > 0` is False, so a NaN walked
        # through the whole validator untouched. It then surfaced far away —
        # `int(nan)` raising inside the ceiling, or a pacing tolerance that
        # silently disabled outlier detection because nothing ever exceeded it.
        # The manifest is data from outside the process, so this is where it
        # has to be caught.
        for name in _FINITE_FIELDS:
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number: {value!r}")
        if self.retry_max_attempts < 0:
            raise ValueError(f"retry_max_attempts must be >= 0: {self.retry_max_attempts}")
        if self.repetition_min_cycles < 2:
            # One cycle is not a repetition and two is the definition of one; a
            # threshold below two would cut every row that says a word twice.
            raise ValueError(
                f"repetition_min_cycles must be at least 2: {self.repetition_min_cycles}"
            )
        if self.repetition_max_period < 1:
            raise ValueError(
                f"repetition_max_period must be positive: {self.repetition_max_period}"
            )
        if self.repetition_min_span < self.repetition_min_cycles:
            # A span shorter than the cycle count is unreachable: the shortest
            # qualifying loop is min_cycles copies of a one-token cycle.
            raise ValueError(
                f"repetition_min_span ({self.repetition_min_span}) must be at least "
                f"repetition_min_cycles ({self.repetition_min_cycles})"
            )
        if self.ceiling_speech_per_text_token <= 0.0:
            raise ValueError(
                "ceiling_speech_per_text_token must be positive: "
                f"{self.ceiling_speech_per_text_token}"
            )
        if self.desperation_speech_per_text_token <= self.ceiling_speech_per_text_token:
            # The desperation rule exists for rows the ceiling let through. If
            # it triggered at or below the ceiling it would fire on every
            # ceiling-stopped row, including the ones the ceiling stopped
            # correctly, and "certainly broken" would stop meaning anything.
            raise ValueError(
                f"desperation_speech_per_text_token "
                f"({self.desperation_speech_per_text_token}) must exceed "
                f"ceiling_speech_per_text_token ({self.ceiling_speech_per_text_token}): "
                "below it, the rule that means 'certainly broken' fires on rows "
                "the ceiling stopped correctly"
            )
        if not 0.0 < self.trailing_filler_threshold <= 1.0:
            raise ValueError(
                f"trailing_filler_threshold must be in (0, 1]: {self.trailing_filler_threshold}"
            )
        if not 0.0 <= self.filler_min_eos_probability < 1.0:
            raise ValueError(
                f"filler_min_eos_probability out of range: {self.filler_min_eos_probability}"
            )
        for name in (
            "ceiling_slack_tokens",
            "trailing_silence_run_tokens",
            "desperation_min_text_tokens",
            "ended_tail_silence_run",
            "ended_tail_blip_max",
            "ended_tail_word_max",
            "filler_max_speech_after_run",
            "ended_tail_keep",
            "echo_strong_max_tail",
            "echo_weak_max_tail",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0: {value}")
        for name in ("echo_strong_min_position_pct", "echo_weak_min_position_pct"):
            pct = getattr(self, name)
            if not 0 <= pct <= 100:
                raise ValueError(f"{name} is a percentage: {pct}")


@dataclass(frozen=True, slots=True)
class Inspection:
    """What the detectors concluded about one chunk."""

    keep: int
    """How many leading tokens survive. Equal to the input length when nothing
    fired, so a caller can always slice by it without branching."""

    reason: Reason = "clean"

    suspect: bool = False
    """The row is impossibly long for its text and no anchor agreed where to cut.

    Not an error and not a cut — a report. It is the honest outcome for a row
    that is certainly wrong in a way this layer cannot locate, and it exists
    because the alternative (shipping it silently) is how the artifact reached
    listeners.
    """

    @property
    def cut(self) -> bool:
        """Whether anything was removed."""
        return self.reason != "clean"

    def __repr__(self) -> str:
        flag = ", suspect" if self.suspect else ""
        return f"Inspection({self.reason}, keep={self.keep}{flag})"


def ceiling_for(text_token_count: int, *, config: PostprocessConfig, window: int) -> int:
    """Speech tokens at which the decoder is stopped whatever it thinks.

    Applied during generation rather than after it, because the tokens past the
    ceiling cost real time on a device and are certain to be discarded. Clamped
    to ``window``: the renderer refuses anything above it, so promising more is
    promising work that will be thrown away.
    """
    proportional = int(text_token_count * config.ceiling_speech_per_text_token)
    # The `+ 15` slack is gone: `frame_windows` refuses anything past
    # `max_speech_tokens`, so those fifteen tokens could never be rendered — a
    # row allowed to reach 270 was stopped at 270 and then rejected at 255,
    # which is real time spent to produce something thrown away. Changed in all
    # five together with the `funnel-2` bump, because a ceiling change moves
    # audio and has to be visible in the fingerprint.
    return min(window, proportional + config.ceiling_slack_tokens)


def _silence_flags(tokens: Sequence[int], silence: Collection[int]) -> list[bool]:
    ids = frozenset(silence)
    return [t in ids for t in tokens]


def is_trailing_filler(  # noqa: PLR0911, PLR0912 — run-collection reads linearly
    tokens: Sequence[int],
    index: int,
    *,
    silence: Collection[int],
    config: PostprocessConfig,
) -> bool:
    """Whether what follows ``index`` is a trailing tail rather than more sentence.

    The overrun rescue cuts back to where the model came closest to stopping,
    and that peak is a hint, not a verdict. Trusting it alone truncated whole
    sentences: the same showcase script that runs 10.3 s in one narrator came
    back at 3.2 s in another, because a voice reading a language its tag does
    not match may never commit to stopping, so its best moment of hesitation
    lands a third of the way in.

    So the peak has to be corroborated by *what it proposes to discard*. Two
    forms of corroboration, either one suffices:

    * the tail is mostly silence by share, or
    * the tail contains a long unbroken silence run **and** what follows that
      run is a stray word rather than a continuing clause.

    The second half of that second condition is not decoration. Without it a
    rhetorical pause mid-tail (25 silent tokens, then 80 tokens of speech)
    matched the run rule and the rescue cut the rest of the sentence off —
    caught by ``TrailingFillerTests.aPauseFollowedByMoreSentenceIsNotFiller``.
    """
    if index < 0 or index >= len(tokens):
        return False
    flags = _silence_flags(tokens[index:], silence)

    silent = 0
    run = 0
    longest_run = 0
    for is_silent in flags:
        if is_silent:
            silent += 1
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 0
    if silent / len(flags) >= config.trailing_filler_threshold:
        return True
    if longest_run < config.trailing_silence_run_tokens:
        return False

    # Collect qualifying runs, then require every gap of speech BETWEEN them
    # (and after the last) to be a stray word or less. [seam][real
    # sentence][seam][word] fails: the 80 tokens between the two seams are the
    # sentence itself, not filler trailing the first boundary.
    runs: list[tuple[int, int]] = []  # (start, end) of qualifying runs
    scan_run = 0
    scan_start = 0
    for i, is_silent in enumerate(flags):
        if is_silent:
            if scan_run == 0:
                scan_start = i
            scan_run += 1
            if scan_run == config.trailing_silence_run_tokens:
                runs.append((scan_start, i + 1))
        else:
            scan_run = 0
    if not runs:
        return False

    # A qualifying run buried behind substantial speech is not a trailing
    # boundary — the tail contains the rest of the sentence, and cutting back
    # to the peak would eat it. The run must sit within a stray word or two of
    # the tail's start.
    if runs[0][0] > config.filler_max_speech_after_run:
        return False

    speech_after_run = len(flags) - runs[-1][1]
    if speech_after_run > config.filler_max_speech_after_run:
        return False
    for (_, prev_end), (next_start, _) in zip(runs, runs[1:], strict=False):
        if next_start - prev_end > config.filler_max_speech_after_run:
            return False
    return True


def pacing_outliers(ratios: Sequence[float], *, config: PostprocessConfig) -> list[int]:
    """Indices of chunks whose pace drifts past the tolerance from the median.

    The long-form drift signal, in the same integer-derived domain as the rest
    of the layer: each ratio is speech tokens over text tokens for one chunk.
    Report-only — the caller is told which chunks to listen to, and nothing is
    cut, because a strange pace is evidence of *something* without saying what.

    The median rather than the mean, so one broken chunk cannot drag the
    baseline toward itself and hide.
    """
    if len(ratios) < 3:
        # One chunk has no neighbours; two cannot say which of them drifted.
        return []
    ordered = sorted(ratios)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    if median <= 0:
        return []
    return [
        i
        for i, ratio in enumerate(ratios)
        if ratio > median * config.pacing_tolerance or ratio < median / config.pacing_tolerance
    ]


def repetition_cut(
    tokens: Sequence[int],
    *,
    silence: Collection[int],
    config: PostprocessConfig,
) -> int | None:
    """Where a stuck decoder started looping, or ``None``.

    The one failure this layer did not cover, and the one the literature puts
    first or second in every ranking of what goes wrong with autoregressive
    speech models. The mechanism is the same one behind the trailing hallucinated
    word — the model's own output is its context — but it strikes *inside* the
    row rather than after it, so no tail rule can see it. VALL-E's authors
    describe greedy search "continually generating silence codec codes"; the
    Very Attentive Tacotron stress test produced 52 repetitions of a phrase that
    was supposed to occur nine times.

    Unlike the other rules here, this one cuts **mid-sequence**, so it is
    deliberately hard to trigger: a short cycle, repeated many times, exactly.
    Approximate repetition is left alone. A decoder that has genuinely locked up
    emits the same tokens, not similar ones, and a fuzzy match on a signal this
    destructive would be a way to truncate real speech.

    A cycle that is entirely silence is never a loop: silence repeating is what
    silence *is*, and a pause is already judged by the tail rules against the
    place it sits in. A cycle mixing silence with speech still counts — a
    word-then-pause stutter is one of the shapes this failure takes.

    Returns the index one full cycle past the loop's start: the first instance is
    plausibly the word the sentence actually wanted, and everything after it is
    the model reading its own tail.

    **Provenance is different from every other rule on this page** and is stated
    rather than buried: the two constants come from the published parameters of
    inline repetition guards (VALL-E 2's repetition-aware sampling uses a window
    of 10 tokens; MSpoofTTS scans at segment lengths 10/25/50), not from a device
    trace of this model. They are calibrated to be *safe* rather than sensitive,
    and the measured firing rate on real renders is reported in the docs.
    """
    n = len(tokens)
    if n < config.repetition_min_span:
        return None
    quiet = _silence_flags(tokens, silence)

    # Earliest loop wins: a row that locks up twice locked up first at the first
    # one, and everything after it is already inside the failure.
    best: int | None = None
    longest_period = min(config.repetition_max_period, n // config.repetition_min_cycles)
    for period in range(1, longest_period + 1):
        start = 0
        while start + period * config.repetition_min_cycles <= n:
            # How many consecutive copies of tokens[start:start+period] follow it.
            cycles = 1
            at = start + period
            while at + period <= n and all(
                tokens[at + i] == tokens[start + i] for i in range(period)
            ):
                cycles += 1
                at += period
            if (
                cycles >= config.repetition_min_cycles
                and cycles * period >= config.repetition_min_span
                and not all(quiet[start + i] for i in range(period))
            ):
                candidate = start + period
                if best is None or candidate < best:
                    best = candidate
                break
            # Nothing anchored here; the next possible start is one token on.
            start += 1
    return best


def desperation_cut(
    tokens: Sequence[int],
    *,
    text_token_count: int,
    min_tokens: int,
    eos_peak_at: int,
    silence: Collection[int],
    config: PostprocessConfig,
    peak_allowed: bool = True,
) -> int | None:
    """The rescue for rows whose *length* is the evidence.

    Past ``desperation_speech_per_text_token`` the row is certainly broken, so
    the question is where to cut, not whether:

    * at the first long silence run that starts past the floor — a structural
      boundary, and on a certainly-broken row what follows it needs no further
      corroboration. A run straddling the floor belongs to the sentence, not to
      the tail, which is why the run's *start* is tested rather than its end;
    * else at the stop peak, if it sits in a band a real read could have ended
      in. The band is what protects the mislabeled-language case (92 generated /
      26 text = 3.5x, below the ratio guard) — a row of that kind must never be
      cut at a peak landing a third of the way in, and here such a peak fails
      the floor.

    ``peak_allowed`` is false for a continuation chunk: it has no sentence end,
    so its stop peak means nothing.
    """
    if text_token_count < config.desperation_min_text_tokens:
        return None
    if len(tokens) < text_token_count * config.desperation_speech_per_text_token:
        return None

    earliest = max(min_tokens, 10)
    flags = _silence_flags(tokens, silence)

    run_start = -1
    run = 0
    for i, is_silent in enumerate(flags):
        if is_silent:
            if run == 0:
                run_start = i
            run += 1
            if run >= config.trailing_silence_run_tokens and run_start >= earliest:
                return run_start
        else:
            run = 0

    # No seam — the babble is dense; fall back to the model's own best stop, if
    # it lands where a real read could have ended.
    if not peak_allowed:
        return None
    band_top = (
        int(config.desperation_band_ratio * text_token_count) + config.desperation_band_floor
    )
    if earliest <= eos_peak_at <= band_top and eos_peak_at < len(tokens):
        return eos_peak_at
    return None


def ended_tail_trim(
    tokens: Sequence[int],
    *,
    silence: Collection[int],
    config: PostprocessConfig,
    is_terminal: bool = False,
) -> int | None:
    """Dead air past the sentence on a row that stopped when it meant to.

    An ended row is trusted to have stopped where it meant to — but three tail
    shapes still ship dead air, walked backward as
    ``[sentence][r1 silence][burst][r2 silence]``:

    * a bare silence run half a second long, tightened to a natural pause;
    * a silence run with a 1–2 token blip right before the stop (a 40–80 ms
      click after a pause; the device specimen ended ``.......#``);
    * on a *terminal* chunk only, a stray word up to ``ended_tail_word_max``
      behind a full silence seam.
    """
    flags = _silence_flags(tokens, silence)
    j = len(tokens) - 1

    r2 = 0
    while j >= 0 and flags[j]:
        r2 += 1
        j -= 1
    if j < 0:
        return None
    if r2 >= config.trailing_silence_run_tokens:
        new_count = j + 1 + min(r2, config.ended_tail_keep)
        return new_count if new_count < len(tokens) else None

    burst = 0
    while j >= 0 and not flags[j]:
        burst += 1
        j -= 1
    r1 = 0
    while j >= 0 and flags[j]:
        r1 += 1
        j -= 1
    if j < 0:
        return None  # the "burst" was the sentence

    stranded_click = burst <= config.ended_tail_blip_max and r1 >= config.ended_tail_silence_run
    stranded_word = (
        is_terminal
        and burst <= config.ended_tail_word_max
        and r1 >= config.trailing_silence_run_tokens
    )
    if not (stranded_click or stranded_word):
        return None
    new_count = j + 1 + min(r1, config.ended_tail_keep)
    return new_count if new_count < len(tokens) else None


def terminal_echo_cut(
    *,
    token_count: int,
    eos_peak_at: int,
    eos_peak_prob: float,
    min_tokens: int,
    is_terminal: bool,
    hit_ceiling: bool,
    config: PostprocessConfig,
) -> int | None:
    """A terminal chunk that ended correctly and then free-ran an extra word.

    No silence seam here, so :func:`is_trailing_filler` has nothing to anchor
    on. Instead the earlier stop candidate must be strong, late, and followed by
    a short tail. The late-position rule is what protects real comma and clause
    pauses.

    The second acceptance path is narrower and exists for one measured
    regression ("...but a brigand. Pass. Four.": ``gen=124/124,
    bestEOS=109@0.004``). The model never sampled a stop token, but its best —
    very weak — stop was 15 tokens before the hard ceiling. Weak confidence is
    not trustworthy in general; it is trustworthy only with all three
    corroborators together: a terminal chunk, an actual ceiling overrun, and a
    very short tail occupying the last 15% of the row.
    """
    if not is_terminal:
        return None
    if not (max(min_tokens, 10) < eos_peak_at < token_count):
        return None

    tail = token_count - eos_peak_at
    strong_peak = (
        eos_peak_prob >= config.echo_strong_eos_probability
        and tail <= config.echo_strong_max_tail
        and eos_peak_at * 100 >= token_count * config.echo_strong_min_position_pct
    )
    weak_late_peak_at_ceiling = (
        hit_ceiling
        and eos_peak_prob >= config.echo_weak_eos_probability
        and tail <= config.echo_weak_max_tail
        and eos_peak_at * 100 >= token_count * config.echo_weak_min_position_pct
    )
    if strong_peak or weak_late_peak_at_ceiling:
        return eos_peak_at
    return None


def _is_dropout(token_count: int, text_token_count: int, config: PostprocessConfig) -> bool:
    """Whether the row is too short to be the text it was asked for.

    Two conditions, both required. The absolute floor catches a row that stopped
    almost immediately whatever the text was. The proportional one is what keeps
    a genuinely short line exempt: a row is only suspect when the text asked for
    materially more speech than arrived, and the chunker's own conservative
    estimate of "materially more" is the floor the sampler already generated
    under.
    """
    if token_count >= config.dropout_min_tokens:
        return False
    # `min_tokens_floor` is the sampler's own bound and is not visible here, so
    # the comparison uses the same conservative ratio the chunker budgets with:
    # a read that produced less than one speech token per text token has not
    # said the text under any pronunciation.
    return text_token_count > 0 and token_count < text_token_count


def inspect(
    tokens: Sequence[int],
    *,
    text_token_count: int,
    min_tokens: int,
    eos_peak_at: int,
    eos_peak_prob: float,
    ended: bool,
    is_terminal: bool,
    hit_ceiling: bool,
    silence: Collection[int],
    config: PostprocessConfig,
) -> Inspection:
    """Run every detector in precedence order and return one verdict.

    The reading app grew five entry points, one per field bug, and left the
    ordering to each call site. Here they are one resolver with the precedence
    written down, because an order that lives in a caller is an order the next
    caller gets wrong.

    Args:
        tokens: the chunk's speech tokens, specials already stripped.
        text_token_count: how many *text* tokens produced them. The denominator
            of every ratio rule.
        min_tokens: the EOS floor this row was generated under.
        eos_peak_at: step index at which the stop token was most probable, or a
            negative number if it was never observed.
        eos_peak_prob: that probability. Observational — it never feeds back
            into sampling — but it *does* gate two rules, so it is pinned by the
            conformance fixture like any other audible value.
        ended: whether generation stopped at the stop token rather than a cap.
        is_terminal: whether this chunk ends the passage. A continuation chunk
            has no sentence end, so its stop peak means nothing and its pauses
            are rhythm rather than dead air.
        hit_ceiling: whether generation was stopped by the length ceiling.
        silence: the silence token ids.
    """
    if config.mode == "off" or not tokens:
        return Inspection(keep=len(tokens))

    # Early truncation, before anything else is considered.
    #
    # It is reported and never cut, because there is nothing to cut: the row is
    # already too short, and the missing content cannot be recovered by removing
    # more. This is the only verdict in the layer that says "what you have is
    # incomplete" rather than "the end of what you have is wrong", and it is the
    # most damaging failure in the set precisely because a listener cannot hear
    # that anything is absent.
    #
    # The floor is on the row against what the text asked for, not on the row
    # alone: a three-word line legitimately renders short, and the shortest
    # healthy reads measured across nine languages run 35 tokens.
    if _is_dropout(len(tokens), text_token_count, config):
        return Inspection(keep=len(tokens), reason="dropout", suspect=True)

    filler_cut = (
        # Terminal chunks only, like its three siblings. `is_terminal` is
        # documented above as meaning a continuation chunk's stop peak means
        # nothing and its pauses are rhythm rather than dead air — and this rule
        # reads exactly those two signals, so it was trimming mid-passage chunks
        # on evidence the contract says is not evidence. `terminal_echo_cut`
        # guards with `if not is_terminal`, `desperation_cut` through
        # `peak_allowed`, `ended_tail_trim` by construction; this one did not.
        is_terminal
        and not ended
        and eos_peak_prob > config.filler_min_eos_probability
        and eos_peak_at > max(min_tokens, 10)
        and eos_peak_at < len(tokens)
        and is_trailing_filler(tokens, eos_peak_at, silence=silence, config=config)
    )

    cut: int | None = None
    reason: Reason = "clean"
    # First, because it is the only rule that knows *exactly* where the failure
    # began. Every other anchor here is inferred — a stop peak the model was
    # unsure about, a silence run that might be a pause, a ratio that says
    # something is wrong without saying where. An exact repeated cycle is
    # evidence of a different quality, so it outranks all of them.
    looped = repetition_cut(tokens, silence=silence, config=config)
    if looped is not None:
        cut, reason = looped, "repetition"
    elif filler_cut:
        cut, reason = eos_peak_at, "silence_tail"
    else:
        echo = terminal_echo_cut(
            token_count=len(tokens),
            eos_peak_at=eos_peak_at,
            eos_peak_prob=eos_peak_prob,
            min_tokens=min_tokens,
            is_terminal=is_terminal,
            hit_ceiling=hit_ceiling,
            config=config,
        )
        if echo is not None:
            cut, reason = echo, "terminal_echo"
        else:
            desperate = desperation_cut(
                tokens,
                text_token_count=text_token_count,
                min_tokens=min_tokens,
                eos_peak_at=eos_peak_at,
                silence=silence,
                config=config,
                peak_allowed=is_terminal,
            )
            if desperate is not None:
                cut, reason = desperate, "desperation"

    if cut is None and ended:
        trimmed = ended_tail_trim(
            tokens, silence=silence, config=config, is_terminal=is_terminal
        )
        if trimmed is not None:
            cut, reason = trimmed, "ended_tail"

    keep = len(tokens) if cut is None else cut
    # A condemned row that dodged every token anchor. Reported, never cut: no
    # rule could say where, and cutting at a guess is how the rescue truncated
    # whole sentences before the corroboration rules were added.
    suspect = (
        cut is None
        and text_token_count >= config.desperation_min_text_tokens
        and len(tokens) >= text_token_count * config.desperation_speech_per_text_token
    )
    return Inspection(keep=keep, reason=reason, suspect=suspect)
