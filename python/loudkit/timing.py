"""Where each chunk — and, approximately, each word — lands in the waveform.

A reading app highlights the sentence it is speaking. That needs two different
kinds of answer, and this module is careful to keep them apart, because
conflating them is how a feature like this becomes a lie:

**Chunk times are exact.** The engine renders each chunk to its own waveform and
concatenates them, so it knows every chunk's sample offset and sample length
without estimating anything. :class:`ChunkTiming` reports those, converted to
seconds. Chunk *k*'s ``end`` is bit-identical to chunk *k+1*'s ``start``: both
are the same integer sample offset divided by the same sample rate, so a
highlight driven by them can neither gap nor overlap.

**Word times are estimated.** The model emits speech tokens, not an alignment;
nothing in this pipeline knows where a word begins. :class:`WordTiming`
distributes a chunk's real duration across its words in proportion to how long
each word is in characters, and that is all it is. It is right often enough to
be useful for a highlight at sentence scale and wrong in the ways you would
expect: a long word said fast, a short word held, a pause before a clause. The
error grows with the length of the chunk, because a single bad guess early
shifts everything after it — one sentence is usually fine, a long paragraph read
as one chunk is not. If you need real alignment, you need a forced aligner; this
is not one, and pretending otherwise would be worse than the estimate.

Both are computed *after* any time-stretch, on the waveform the caller actually
receives, so :class:`~loudkit.engine.Result.speed` needs no correction applied
to them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

__all__ = ["ChunkSpan", "ChunkTiming", "WordTiming", "timeline"]


class ChunkSpan(NamedTuple):
    """What one rendered chunk contributes to a timeline.

    The three facts the engine has at concatenation time and nothing else: the
    text it was asked to speak (post-funnel, which is what was tokenised), how
    many samples it rendered to, and how many speech tokens it took. Kept as an
    input type rather than assembling :class:`ChunkTiming` per chunk, because
    the offsets are only knowable once the order is known.
    """

    text: str
    samples: int
    tokens: int


@dataclass(frozen=True, slots=True)
class WordTiming:
    """One word's estimated span, in seconds from the start of the synthesis.

    **Estimated, by proportional allocation.** The chunk's real duration is
    divided among its words in proportion to their length in characters. There
    is no alignment model here and no per-word measurement — see the module
    docstring for what that costs you.
    """

    text: str
    """The word as it appears in the chunk, punctuation included.

    Punctuation stays attached because the split is on whitespace: a caller
    highlighting ``"end."`` wants the full stop lit with the word, and a caller
    matching back against their own text needs the substring to be a substring.
    """

    start: float
    end: float


@dataclass(frozen=True, slots=True)
class ChunkTiming:
    """One chunk's exact span, and its words' estimated ones.

    The two tiers in one object on purpose: a caller that trusts only the exact
    tier reads ``start``/``end`` and ignores ``words``, and the field names make
    it impossible to reach the estimate by accident.
    """

    text: str
    """The chunk's text after the speech funnel — what was tokenised, which is
    not always what the caller passed in (Polish respells embedded English, and
    numbers are read as words)."""

    start: float
    """Seconds from the start of this :class:`~loudkit.engine.Result`'s audio.

    Zero for the first chunk, and for every chunk of a streamed result: a
    streamed chunk is its own ``Result`` and does not know what preceded it, so
    the caller stitching the stream adds the offsets.
    """

    end: float
    tokens: int
    """Speech tokens this chunk generated. Duration over tokens is the pacing
    the postprocess detectors measure against, which is the other reason to
    carry it."""

    words: tuple[WordTiming, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start

    def shifted(self, by: float) -> ChunkTiming:
        """This timing moved later by ``by`` seconds, words included."""
        return ChunkTiming(
            text=self.text,
            start=self.start + by,
            end=self.end + by,
            tokens=self.tokens,
            words=tuple(
                WordTiming(text=w.text, start=w.start + by, end=w.end + by) for w in self.words
            ),
        )


def timeline(spans: Sequence[ChunkSpan], *, sample_rate: int) -> tuple[ChunkTiming, ...]:
    """Lay rendered chunks end to end and time them.

    Offsets accumulate in **samples**, not seconds, and are divided by the rate
    once at the end. Accumulating seconds instead would make chunk *k*'s ``end``
    and chunk *k+1*'s ``start`` two different sums of the same floats, differing
    in the last bit — a gap or an overlap of a few nanoseconds, invisible in a
    test that compares with a tolerance and visible as a flicker in a highlight
    that switches on ``time >= start``.
    """
    out: list[ChunkTiming] = []
    at = 0
    for span in spans:
        start = at / sample_rate
        at += span.samples
        end = at / sample_rate
        out.append(
            ChunkTiming(
                text=span.text,
                start=start,
                end=end,
                tokens=span.tokens,
                words=estimate_words(span.text, start=start, end=end),
            )
        )
    return tuple(out)


def estimate_words(text: str, *, start: float, end: float) -> tuple[WordTiming, ...]:
    """Split ``text`` on whitespace and share ``[start, end]`` out by length.

    The allocation is by **character count**, not by token count or by any
    acoustic measure: a word's characters are the only thing known here, and
    they correlate with duration well enough at sentence scale to drive a
    highlight. Whitespace itself is not charged for — the gap between two words
    belongs to whichever side of the boundary the caller's player is on, and
    splitting it would only invent a third kind of span.

    Boundaries are computed from a running character total rather than by adding
    per-word durations, so the spans cannot drift: the first ``start`` is exactly
    ``start``, the last ``end`` is exactly ``end``, and every interior boundary
    is shared by the two words that meet at it.
    """
    words = text.split()
    lengths = [len(w) for w in words]
    total = sum(lengths)
    if total == 0:
        return ()
    span = end - start
    out: list[WordTiming] = []
    seen = 0
    for word, length in zip(words, lengths, strict=True):
        at = start + span * (seen / total)
        seen += length
        out.append(WordTiming(text=word, start=at, end=start + span * (seen / total)))
    return tuple(out)
