"""Where each chunk, and, approximately, each word, lands in the waveform.

See ``docs/design/engine-pipeline.md``.
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
    is no alignment model here and no per-word measurement, see the module
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
    """The chunk's text after the speech funnel, what was tokenised, which is
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
    """Speech tokens this chunk generated.

    Not a pace on its own. A chunk's audio is this count times the frame
    rate, so duration over tokens is the same number for every chunk and a
    tolerance around it can never be exceeded. The drift measure
    :func:`loudkit.postprocess.pacing_outliers` compares is speech tokens
    over *text* tokens, which needs the chunk's text tokenised as well.
    Nothing in the engine computes it, so a caller that wants the measure
    builds the ratios from this number and its own.
    """

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

    See ``docs/design/engine-pipeline.md``.
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

    See ``docs/design/engine-pipeline.md``.
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
