"""Postprocess: deciding where a generated chunk actually ended.

See ``docs/design/postprocess-detectors.md``.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = ["Inspection", "PostprocessMode", "Reason", "inspect"]

Reason = Literal[
    "clean",
    "dropout",
    "stall",
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
algorithm, it travels in the fingerprint like every other audible decision.
``report`` runs the detectors and attaches the verdict without acting on it, for
a caller that would rather hear the artifact than risk a wrong cut. ``off``
skips the detectors entirely.
"""

RepetitionResume = Literal["condemn", "cut"]
"""What a qualifying loop the decoder *resumed from* receives."""

RepetitionSilence = Literal["acoustic", "sampling"]
"""Which silence family the all-silence-cycle exemption reads."""


_FINITE_FIELDS: tuple[str, ...] = (
    "ceiling_speech_per_text_token",
    "trailing_filler_threshold",
    "filler_min_eos_probability",
    "desperation_band_ratio",
    "desperation_speech_per_text_token",
    "desperation_min_keep_per_text_token",
    "echo_strong_eos_probability",
    "echo_weak_eos_probability",
    "pacing_tolerance",
)
"""Every float on :class:`PostprocessConfig`, listed so a NaN cannot walk through the
validator."""


RETRY_LADDER_HEADROOM = 8
"""How many retry streams the seed ladder has room for: the gap between the
engine's retry stream and its chunk streams, restated here because the engine
cannot be imported without a cycle. Pinned to the originals by a test."""

_EARLIEST_CUT_TOKENS = 10
"""No tail rule cuts earlier than this, whatever the floor allows.

Ten tokens is about 0.4 s, shorter than any spoken word plus the breath before
it, so nothing a cut above this bound removes was ever a whole word. All three
tail detectors share it. It is part of the law the four ports reimplement, so
the number moves in five places or in none.
"""


def _earliest_cut(min_tokens: int) -> int:
    """The earliest index any tail rule may cut at.

    The floor and the bound are both lower limits, so the later one wins.
    """
    return max(min_tokens, _EARLIEST_CUT_TOKENS)


@dataclass(frozen=True, slots=True)
class PostprocessConfig:
    """The detectors' preset: every knob was measured on one checkpoint, not chosen.

    Read from the checkpoint's manifest ``postprocess`` block; ``mode`` is the
    one knob a caller changes. Hashed into the fingerprint, because a port
    computing with a different number produces different audio.
    """

    mode: PostprocessMode = "trim"

    ceiling_speech_per_text_token: float = 4.0
    """Hard stop for generation, as a multiple of the text-token count.

    See ``docs/design/postprocess-detectors.md``.
    """

    ceiling_slack_tokens: int = 40
    """Slack above the proportional bound, in speech tokens (1.6 s of audio).
    Carries the very short texts, where a ratio alone is unsafe."""

    trailing_filler_threshold: float = 0.7
    """The share of a tail that must be silence before it counts as one."""

    trailing_silence_run_tokens: int = 12
    """An unbroken silence run that marks a structural boundary (~0.5 s at 25 Hz).

    See ``docs/design/postprocess-detectors.md``.
    """

    desperation_band_ratio: float = 2.6
    """Top of the stop-peak acceptance band, as a multiple of the text-token count.

    See ``docs/design/postprocess-detectors.md``.
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
    """Past this ratio the row certainly contains garbage, whatever its EOS confidence
    said.

    See ``docs/design/postprocess-detectors.md``.
    """

    desperation_min_text_tokens: int = 10
    """Tiny texts are exempt: fixed overheads (initial breath, final pause) give
    a clean "No!" a ratio of 6+ all by itself."""

    desperation_min_keep_per_text_token: float = 1.7
    """A cap-hit row whose desperation cut keeps fewer speech tokens than this many per
    text token is condemned into the retry ladder instead of shipping the trim.

    See ``docs/design/postprocess-detectors.md``.
    """

    ended_tail_silence_run: int = 6
    """Silence before a blip counts as stranding it (~0.24 s)."""

    ended_tail_blip_max: int = 2
    """<= 80 ms of "speech" is a click, not a word. A real word is never 1 to 2
    tokens, so speech is untouchable above this."""

    ended_tail_word_max: int = 10
    """~0.4 s: a word, not a clause. A stray word behind a full silence seam on
    a *terminal* chunk is cut with it, prose does not resume after that much
    dead air with a single word. Continuation chunks keep their tails; their
    pauses are the sentence's rhythm and their "end" is not an end."""

    filler_max_speech_after_run: int = 10
    """~0.4 s: how much speech may follow a silence seam and still count as a hallucinated
    word rather than a continuing clause.

    See ``docs/design/postprocess-detectors.md``.
    """

    ended_tail_keep: int = 5
    """~0.2 s of pause left in place after trimming."""

    echo_strong_eos_probability: float = 0.10
    echo_strong_max_tail: int = 30
    echo_strong_min_position_pct: int = 68
    """The ordinary terminal echo: a stop the model was confident about, late in
    the row, with at most ~1.2 s, two words, after it. The position rule is
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

    See ``docs/design/postprocess-detectors.md``.
    """

    pacing_tolerance: float = 1.6
    """How far a chunk's pace may drift from the passage's median before it is flagged (as
    a multiplicative factor, both directions).

    See ``docs/design/postprocess-detectors.md``.
    """

    repetition_max_period: int = 12
    """Longest cycle, in tokens (~0.5 s), that counts as a stuck decoder.

    Above this a repeated block is a *phrase*, and a repeated phrase is rhetoric:
    "no, no, no", a stammer, a refrain. Below it the model is emitting the same
    fragment over and over because its own output has become its context, which
    is a different event with a different fix.
    """

    repetition_min_cycles: int = 3
    """How many consecutive identical cycles before it can be a loop.

    Two is a repeated phrase, which is ordinary speech. Three is the smallest
    count that is not, and it is a necessary condition rather than a sufficient
    one, `repetition_min_span` is what actually separates the classes.
    """

    dropout_min_tokens: int = 25
    """Below this a row is too short to be the sentence it was asked for (~1.0 s).

    See ``docs/design/postprocess-detectors.md``.
    """

    stall_run_tokens: int = 25
    """A non-tail dead-air run this long condemns the row (~1.0 s at 25 Hz).

    See ``docs/design/postprocess-detectors.md``.
    """

    silence_render_ids: tuple[int, ...] = ()
    """Token ids that render as true digital silence.

    See ``docs/design/postprocess-detectors.md``.
    """

    quiet_render_ids: tuple[int, ...] = ()
    """The contextually-quiet family: breath and decay ids.

    See ``docs/design/postprocess-detectors.md``.
    """

    repetition_min_span: int = 24
    """How many tokens the repeating region must cover (~1.0 s).

    See ``docs/design/postprocess-detectors.md``.
    """

    repetition_resume: RepetitionResume = "condemn"
    """What a qualifying loop the decoder *resumed from* receives.

    See ``docs/design/postprocess-detectors.md``.
    """

    repetition_silence: RepetitionSilence = "acoustic"
    """Which silence family the all-silence-cycle exemption reads.

    See ``docs/design/postprocess-detectors.md``.
    """

    def __post_init__(self) -> None:
        if self.mode not in ("off", "report", "trim"):
            raise ValueError(f"unknown postprocess mode: {self.mode!r}")
        if self.repetition_resume not in ("condemn", "cut"):
            raise ValueError(f"unknown repetition_resume: {self.repetition_resume!r}")
        if self.repetition_silence not in ("acoustic", "sampling"):
            raise ValueError(f"unknown repetition_silence: {self.repetition_silence!r}")
        self._validate_ranges()

    def _validate_ranges(self) -> None:  # noqa: PLR0912 - one branch per constant;
        # a validator that groups its fields to satisfy a branch count reads as if the
        # groupings meant something, and they do not.
        for name in _FINITE_FIELDS:
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number: {value!r}")
        if not 0 <= self.retry_max_attempts < RETRY_LADDER_HEADROOM:
            raise ValueError(
                f"retry_max_attempts must be in [0, {RETRY_LADDER_HEADROOM}): "
                f"{self.retry_max_attempts}. Above that the ladder's derived "
                f"seeds run into the streams the chunk seeds use."
            )
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
        if self.stall_run_tokens < 1:
            # At zero every row with a single silence token before speech is a
            # stall, and "condemned" stops meaning anything.
            raise ValueError(f"stall_run_tokens must be positive: {self.stall_run_tokens}")
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
        if self.desperation_min_keep_per_text_token < 0.0:
            raise ValueError(
                "desperation_min_keep_per_text_token must be >= 0: "
                f"{self.desperation_min_keep_per_text_token}"
            )
        if self.desperation_min_keep_per_text_token > self.desperation_band_ratio:
            # The band top is where a real read could still have ended.
            # Demanding a keep above it condemns cuts landing exactly where
            # the band admits them, and "starved" stops meaning anything.
            raise ValueError(
                f"desperation_min_keep_per_text_token "
                f"({self.desperation_min_keep_per_text_token}) must not exceed "
                f"desperation_band_ratio ({self.desperation_band_ratio})"
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
            # `desperation_band_floor` and `dropout_min_tokens` are counts like
            # every other name here and were the two this list omitted, so a
            # negative one loaded: the band closed instead of opening, and no
            # row was ever short enough to be a dropout. Rust holds its counts
            # as `usize` and cannot represent either, so leaving them unnamed
            # here meant that port refusing a manifest the reference accepts.
            "desperation_band_floor",
            "desperation_min_text_tokens",
            "ended_tail_silence_run",
            "ended_tail_blip_max",
            "ended_tail_word_max",
            "filler_max_speech_after_run",
            "ended_tail_keep",
            "echo_strong_max_tail",
            "echo_weak_max_tail",
            "dropout_min_tokens",
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
    """The row is certainly wrong in a way no cut can fix.

    See ``docs/design/postprocess-detectors.md``.
    """

    @property
    def cut(self) -> bool:
        """Whether a rule fired, which is not the same as whether tokens came off.

        See ``docs/design/postprocess-detectors.md``.
        """
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
    # A ceiling change moves audio, so it moves the fingerprint: change this
    # line in all five ports together, with the recipe version.
    return min(window, proportional + config.ceiling_slack_tokens)


def _silence_flags(tokens: Sequence[int], silence: Collection[int]) -> list[bool]:
    ids = frozenset(silence)
    return [t in ids for t in tokens]


def is_trailing_filler(  # noqa: PLR0911, PLR0912, run-collection reads linearly
    tokens: Sequence[int],
    index: int,
    *,
    silence: Collection[int],
    config: PostprocessConfig,
) -> bool:
    """Whether what follows ``index`` is a trailing tail rather than more sentence.

    See ``docs/design/postprocess-detectors.md``.
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
    # boundary, the tail contains the rest of the sentence, and cutting back
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

    See ``docs/design/postprocess-detectors.md``.
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

    See ``docs/design/postprocess-detectors.md``.
    """
    found = _loop_candidate(tokens, silence=silence, config=config)
    return None if found is None else found[0]


def _loop_candidate(
    tokens: Sequence[int],
    *,
    silence: Collection[int],
    config: PostprocessConfig,
) -> tuple[int, bool] | None:
    """The earliest qualifying loop: ``(cut index, decoder resumed)``.

    See ``docs/design/postprocess-detectors.md``.
    """
    n = len(tokens)
    if n < config.repetition_min_span:
        return None
    # The exemption's family, not the run rules': the tail rules keep reading
    # the sampler list they were calibrated against. Resolved here rather than
    # by the caller for the same reason `is_stalled` reads its censuses off the
    # config, a family that lives in a caller is a family the next caller
    # feeds wrong. Without censuses the union is the sampler list, unchanged.
    family: Collection[int] = silence
    if config.repetition_silence == "acoustic":
        family = frozenset(silence).union(config.silence_render_ids, config.quiet_render_ids)
    quiet = _silence_flags(tokens, family)

    # Earliest loop wins: a row that locks up twice locked up first at the first
    # one, and everything after it is already inside the failure.
    best: tuple[int, bool] | None = None
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
                if best is None or candidate < best[0]:
                    best = (candidate, n - at >= period)
                break
            # Nothing anchored here; the next possible start is one token on.
            start += 1
    return best


def is_stalled(
    tokens: Sequence[int],
    *,
    hit_ceiling: bool,
    silence: Collection[int],
    config: PostprocessConfig,
) -> bool:
    """Whether the decoder spent this row trapped in silence.

    See ``docs/design/postprocess-detectors.md``.
    """
    if not tokens:
        return False
    census = bool(config.silence_render_ids)
    gate = frozenset(config.silence_render_ids) if census else frozenset(silence)
    family = gate | frozenset(config.quiet_render_ids)

    in_family = [t in family for t in tokens]
    if census and all(in_family):
        return True
    if census and hit_ceiling and 2 * sum(in_family) > len(tokens):
        return True

    gate_count = 0
    for flag, token in zip(in_family, tokens, strict=True):
        if flag:
            if token in gate:
                gate_count += 1
        else:
            # The run ended before the row did, so it is not the tail.
            if gate_count >= config.stall_run_tokens:
                return True
            gate_count = 0
    return False


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

    See ``docs/design/postprocess-detectors.md``.
    """
    if text_token_count < config.desperation_min_text_tokens:
        return None
    if len(tokens) < text_token_count * config.desperation_speech_per_text_token:
        return None

    earliest = _earliest_cut(min_tokens)
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

    # No seam, the babble is dense; fall back to the model's own best stop, if
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

    See ``docs/design/postprocess-detectors.md``.
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

    See ``docs/design/postprocess-detectors.md``.
    """
    if not is_terminal:
        return None
    if not (_earliest_cut(min_tokens) < eos_peak_at < token_count):
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

    See ``docs/design/postprocess-detectors.md``.
    """
    if token_count >= config.dropout_min_tokens:
        return False
    # The floor is deliberately not consulted, though `inspect` holds it and
    # hands it to the three tail rules. It answers a different question: how
    # short a row may be at all, against how short it is *for this text*. The
    # ratio is the conservative one the chunker budgets with, so a read that
    # produced less than one speech token per text token has not said the text
    # under any pronunciation.
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

    See ``docs/design/postprocess-detectors.md``.
    """
    if config.mode == "off" or not tokens:
        return Inspection(keep=len(tokens))

    # Early truncation, before anything else is considered.
    if _is_dropout(len(tokens), text_token_count, config):
        return Inspection(keep=len(tokens), reason="dropout", suspect=True)

    filler_cut = (
        # Terminal chunks only, like its three siblings.
        is_terminal
        and not ended
        and eos_peak_prob > config.filler_min_eos_probability
        and eos_peak_at > _earliest_cut(min_tokens)
        and eos_peak_at < len(tokens)
        and is_trailing_filler(tokens, eos_peak_at, silence=silence, config=config)
    )

    cut: int | None = None
    reason: Reason = "clean"
    starved = False
    # First, because it is the only rule that knows *exactly* where the failure
    # began. Every other anchor here is inferred, a stop peak the model was
    # unsure about, a silence run that might be a pause, a ratio that says
    # something is wrong without saying where. An exact repeated cycle is
    # evidence of a different quality, so it outranks all of them.
    looped = _loop_candidate(tokens, silence=silence, config=config)
    if looped is not None and looped[1] and config.repetition_resume == "condemn":
        # The decoder came back after the repeating region, so it was never locked, and
        # the cut would delete whatever it came back to say, the en0023 defect exactly.
        return Inspection(keep=len(tokens), reason="repetition", suspect=True)
    if looped is not None:
        cut, reason = looped[0], "repetition"
    elif is_stalled(tokens, hit_ceiling=hit_ceiling, silence=silence, config=config):
        # Condemned, never cut, before any tail rescue can run: a mid-row hole
        # is not removable by a tail cut, and a rescue that fired here would
        # trim the tail and ship the hole under its own reason. Routed like
        # `dropout`, reported whole, suspect, into the retry ladder.
        return Inspection(keep=len(tokens), reason="stall", suspect=True)
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
                # The starved rescue.
                starved = hit_ceiling and (
                    desperate < text_token_count * config.desperation_min_keep_per_text_token
                )

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
    suspect = starved or (
        cut is None
        and text_token_count >= config.desperation_min_text_tokens
        and len(tokens) >= text_token_count * config.desperation_speech_per_text_token
    )
    return Inspection(keep=keep, reason=reason, suspect=suspect)
