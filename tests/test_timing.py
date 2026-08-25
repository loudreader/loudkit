"""Timestamps: exact at the chunk, estimated at the word.

The whole value of this feature is that a reading app can trust the first tier
and be told, loudly, not to trust the second in the same way. So the tests are
split the same way: the chunk assertions are equalities, the word assertions are
invariants (monotonic, inside the chunk, every word present) and nothing here
claims a word lands where a listener would say it does.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from loudkit.config import AlgorithmConfig, ChunkConfig, SamplingConfig
from loudkit.timing import ChunkSpan, estimate_words, timeline

from .test_engine import _engine, _voice

SAMPLE_RATE = 24_000


class TestTimeline:
    def test_chunks_are_adjacent_to_the_last_bit(self) -> None:
        """A highlight that switches on ``time >= start`` flickers on a gap and
        double-lights on an overlap, and both are invisible to a comparison
        with a tolerance. Offsets accumulate as integer samples for exactly
        this reason."""
        spans = [ChunkSpan("a b", 7_001, 3), ChunkSpan("c d e", 13_337, 5)]
        got = timeline(spans, sample_rate=SAMPLE_RATE)
        assert got[0].start == 0.0
        assert got[1].start == got[0].end
        assert got[-1].end == (7_001 + 13_337) / SAMPLE_RATE

    def test_the_spans_cover_the_whole_render_with_nothing_left_over(self) -> None:
        spans = [ChunkSpan("one", 100, 1), ChunkSpan("two", 200, 2), ChunkSpan("three", 300, 3)]
        got = timeline(spans, sample_rate=SAMPLE_RATE)
        assert sum(c.duration for c in got) == pytest.approx(600 / SAMPLE_RATE)
        assert [c.tokens for c in got] == [1, 2, 3]

    def test_an_empty_render_is_an_empty_timeline(self) -> None:
        assert timeline([], sample_rate=SAMPLE_RATE) == ()


class TestWordEstimates:
    def test_words_tile_the_chunk_without_gaps(self) -> None:
        words = estimate_words("alpha beta gamma", start=1.0, end=4.0)
        assert [w.text for w in words] == ["alpha", "beta", "gamma"]
        assert words[0].start == 1.0
        assert words[-1].end == 4.0
        for left, right in zip(words, words[1:], strict=False):
            assert left.end == right.start

    def test_times_are_monotonic_and_inside_the_chunk(self) -> None:
        words = estimate_words("a bb ccc dddd e", start=2.5, end=3.25)
        for w in words:
            assert 2.5 <= w.start <= w.end <= 3.25
        assert [w.start for w in words] == sorted(w.start for w in words)

    def test_a_longer_word_is_given_longer(self) -> None:
        """The whole content of the estimate: characters stand in for seconds.
        Nothing else here knows how long a word takes."""
        short, long = estimate_words("hi internationalisation", start=0.0, end=1.0)
        assert long.end - long.start > short.end - short.start

    def test_punctuation_stays_with_its_word(self) -> None:
        """A caller highlighting ``"end."`` wants the full stop lit with the
        word, and a caller matching back against their own text needs the
        substring to be a substring."""
        words = estimate_words("Hello, world!", start=0.0, end=1.0)
        assert [w.text for w in words] == ["Hello,", "world!"]

    def test_no_text_is_no_words_rather_than_a_division_by_zero(self) -> None:
        assert estimate_words("   ", start=0.0, end=1.0) == ()
        assert estimate_words("", start=0.0, end=1.0) == ()

    def test_length_is_counted_in_characters_not_bytes(self) -> None:
        """The four ports count code points too. A byte count would give Polish
        and Japanese text different word weights in Go than in Python, for text
        that reads identically."""
        ascii_word, accented = estimate_words("aaaa żółć", start=0.0, end=1.0)
        assert ascii_word.end - ascii_word.start == pytest.approx(accented.end - accented.start)


class TestTheEngineFillsThemIn:
    """The arithmetic above is only worth anything if the engine reports the
    render it actually produced. These run on the weight-free fakes."""

    def _long_engine(self) -> object:
        algo = AlgorithmConfig().with_(
            chunking=ChunkConfig(max_tokens=20, prefix_tokens=0),
            sampling=SamplingConfig(max_new_tokens=64),
        )
        return _engine(algo)

    def test_a_single_window_gets_one_span_covering_everything(self) -> None:
        result = _engine().synthesize("Hello there world.", _voice(), seed=1)
        assert len(result.chunks) == 1
        assert result.chunks[0].start == 0.0
        assert result.chunks[0].end == result.duration
        assert result.chunks[0].tokens == len(result.tokens)

    def test_a_two_chunk_render_reports_two_spans_at_the_right_offsets(self) -> None:
        engine = self._long_engine()
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        streamed = list(engine.stream(text, _voice(), seed=1))  # type: ignore[attr-defined]
        assert len(streamed) > 1, "the text has to actually split for this to mean anything"
        joined = engine.synthesize_long(text, _voice(), seed=1)  # type: ignore[attr-defined]

        assert len(joined.chunks) == len(streamed)
        at = 0
        for span, part in zip(joined.chunks, streamed, strict=True):
            assert span.start == at / joined.sample_rate
            at += len(part.audio)
            assert span.end == at / joined.sample_rate
        assert joined.chunks[-1].end == joined.duration

    def test_a_streamed_chunk_starts_at_zero_and_the_caller_stitches(self) -> None:
        """A streamed chunk is its own Result and cannot know what preceded it
        — reporting anything but zero would be a guess about the caller's
        playback."""
        engine = self._long_engine()
        for part in engine.stream(  # type: ignore[attr-defined]
            "One. Two. Three. Four. Five. Six.", _voice(), seed=1
        ):
            assert len(part.chunks) == 1
            assert part.chunks[0].start == 0.0
            assert part.chunks[0].end == part.duration

    def test_the_text_is_the_post_funnel_text(self) -> None:
        """What was tokenised, not what the caller typed: the funnel reads
        numbers as words, and a highlight matched against the input would drift
        the moment a digit appeared."""
        result = _engine().synthesize("I have 3 apples.", _voice(), seed=1)
        assert "three" in result.chunks[0].text

    def test_rendering_bare_tokens_still_spans_the_whole_result(self) -> None:
        """No text reached that path, so there is nothing to estimate — but a
        caller stitching results should not have to special-case it."""
        engine = _engine()
        result = engine.synthesize_tokens([1, 2, 3], _voice(), seed=1)
        assert len(result.chunks) == 1
        assert result.chunks[0].end == result.duration
        assert result.chunks[0].words == ()


def test_estimate_words_survives_a_chunk_of_one_word() -> None:
    (only,) = estimate_words("word", start=0.5, end=0.75)
    assert (only.start, only.end) == (0.5, 0.75)


def test_shifting_moves_the_words_with_the_chunk() -> None:
    (span,) = timeline([ChunkSpan("a bb", 240, 2)], sample_rate=SAMPLE_RATE)
    moved = span.shifted(1.0)
    assert moved.start == span.start + 1.0
    assert [w.start for w in moved.words] == [w.start + 1.0 for w in span.words]
    assert replace(span, text=span.text) == span
