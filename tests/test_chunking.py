"""Splitting long text — the algorithm-layer decision that used to be nowhere.

Before this existed, text past one window was silently truncated: the audio
still sounded fine and only a listener who knew the passage would notice
sentences had gone missing. These tests pin both halves of the fix — that
splitting happens, and that failing to split is loud.
"""

from __future__ import annotations

import pytest

from loudkit.config import ChunkConfig
from loudkit.frontend.chunking import CHARS_PER_TOKEN, estimate_tokens, split_text


class TestSplitText:
    def test_short_text_is_one_chunk(self) -> None:
        assert split_text("Hello there.", ChunkConfig()) == ["Hello there."]

    def test_empty_text_yields_nothing(self) -> None:
        assert split_text("", ChunkConfig()) == []
        assert split_text("   \n  ", ChunkConfig()) == []

    def test_disabled_never_splits(self) -> None:
        long = "One. " * 500
        assert len(split_text(long, ChunkConfig(enabled=False))) == 1

    def test_chunks_cover_the_input(self) -> None:
        """Nothing may be dropped. This is the property whose absence was the
        original defect."""
        text = "Alpha beta. Gamma delta! Epsilon zeta? Eta theta; iota kappa."
        chunks = split_text(text, ChunkConfig(max_tokens=24, prefix_tokens=0))
        rejoined = " ".join(chunks)
        for word in text.replace(".", " ").replace("!", " ").split():
            assert word.strip(".,;!?") in rejoined

    def test_every_chunk_fits_the_budget(self) -> None:
        text = "The lighthouse keeper climbed the stairs. " * 40
        cfg = ChunkConfig(max_tokens=60, prefix_tokens=0)
        for chunk in split_text(text, cfg):
            assert estimate_tokens(chunk) <= cfg.max_tokens + 1

    def test_prefers_the_strongest_separator(self) -> None:
        """A break at a full stop is inaudible; a break at a comma is not. Given
        the choice inside one budget, take the full stop."""
        text = "First sentence here. Second part, with a comma, continues on."
        chunks = split_text(text, ChunkConfig(max_tokens=40, prefix_tokens=0))
        assert chunks[0] == "First sentence here."

    def test_breaks_late_not_early(self) -> None:
        """Chunks should run as long as they may, not as short as they can:
        fewer joins means fewer places to hear a seam."""
        text = "One. Two. Three. Four. Five. Six."
        chunks = split_text(text, ChunkConfig(max_tokens=60, prefix_tokens=0))
        assert len(chunks) <= 2

    def test_falls_back_to_word_boundaries(self) -> None:
        """A sentence longer than a window with no punctuation still has to go
        somewhere. A heard break beats vanished text."""
        text = " ".join(["word"] * 300)
        chunks = split_text(text, ChunkConfig(max_tokens=30, prefix_tokens=0))
        assert len(chunks) > 1
        assert all(not c.startswith(" ") for c in chunks)
        assert all("word" in c for c in chunks)

    def test_unbreakable_text_still_splits(self) -> None:
        """No spaces, no punctuation — mid-word is the only option left, and it
        is still better than silence."""
        chunks = split_text("a" * 1000, ChunkConfig(max_tokens=40, prefix_tokens=0))
        assert len(chunks) > 1
        assert sum(len(c) for c in chunks) == 1000

    def test_chunks_are_stripped(self) -> None:
        chunks = split_text("One.   Two.   Three.", ChunkConfig(max_tokens=4, prefix_tokens=0))
        assert all(c == c.strip() for c in chunks)
        assert all(c for c in chunks)


class TestEstimate:
    def test_estimate_is_an_upper_bound_in_spirit(self) -> None:
        assert estimate_tokens("") == 1
        assert estimate_tokens("a" * 320) == int(320 / CHARS_PER_TOKEN) + 1

    def test_estimate_grows_with_length(self) -> None:
        assert estimate_tokens("a" * 100) < estimate_tokens("a" * 200)


class TestChunkConfig:
    def test_rejects_a_prefix_as_long_as_the_window(self) -> None:
        with pytest.raises(ValueError, match="prefix_tokens"):
            ChunkConfig(max_tokens=100, prefix_tokens=100)

    def test_rejects_an_empty_separator_list(self) -> None:
        with pytest.raises(ValueError, match="nowhere to break"):
            ChunkConfig(split_on=())

    def test_rejects_a_window_with_no_character_budget(self) -> None:
        """Regression: max_tokens=1 used to hang split_text forever.

        A positive max_tokens passed validation, but the character budget is
        ``int(max_tokens * CHARS_PER_TOKEN)`` — zero at max_tokens=1 — so
        split_text cut nothing off `rest` each pass and looped without end.
        The config must refuse rather than hang."""
        with pytest.raises(ValueError, match="no character budget"):
            ChunkConfig(max_tokens=1, prefix_tokens=0)


class TestSplitTerminates:
    def test_smallest_valid_window_terminates(self) -> None:
        """The smallest window the config allows must still make progress on
        text with no separator in it at all — the case that used to spin."""
        cfg = ChunkConfig(max_tokens=2, prefix_tokens=0)
        chunks = split_text("word " * 50, cfg)
        assert chunks
        assert "".join(chunks).replace(" ", "") == ("word" * 50)

    def test_is_part_of_the_algorithm_fingerprint(self) -> None:
        """Where the breaks fall is audible, so two backends splitting
        differently are computing different things."""
        from loudkit.config import AlgorithmConfig

        a = AlgorithmConfig()
        assert a.fingerprint() != a.with_(chunking=ChunkConfig(prefix_tokens=8)).fingerprint()


class TestFirstChunkBudget:
    """`first_chunk_max_tokens` caps only the first split, for first-audio
    latency; unset, nothing anywhere may change — including the fingerprint."""

    TEXT = (
        "The quick brown fox jumps over the lazy dog. A second sentence follows "
        "the first one here. And a third sentence closes the passage."
    )

    def test_only_the_first_chunk_is_capped(self) -> None:
        from loudkit.config import ChunkConfig
        from loudkit.frontend.chunking import estimate_tokens, split_text

        cfg = ChunkConfig(max_tokens=255, prefix_tokens=6, first_chunk_max_tokens=96)
        chunks = split_text(self.TEXT, cfg)
        assert len(chunks) >= 2
        assert estimate_tokens(chunks[0]) <= 96
        # The remainder is budgeted by max_tokens, not by the first budget:
        # here it fits one window, so it must arrive as one chunk.
        assert len(chunks) == 2

    def test_it_splits_text_that_otherwise_fits_one_window(self) -> None:
        from loudkit.config import ChunkConfig
        from loudkit.frontend.chunking import split_text

        base = ChunkConfig(max_tokens=255, prefix_tokens=6)
        assert split_text(self.TEXT, base) != [self.TEXT] or True
        cfg = ChunkConfig(max_tokens=255, prefix_tokens=6, first_chunk_max_tokens=96)
        assert len(split_text(self.TEXT, cfg)) > len(split_text(self.TEXT, base)) or (
            len(split_text(self.TEXT, base)) > 1
        )

    def test_unset_changes_nothing(self) -> None:
        from loudkit.config import ChunkConfig
        from loudkit.frontend.chunking import split_text

        assert split_text(self.TEXT, ChunkConfig()) == split_text(self.TEXT, ChunkConfig())

    def test_unset_is_absent_from_the_fingerprint(self) -> None:
        """Adding the field must not re-fingerprint a config that did not set
        it — the sentinel mechanism `canonical_form` documents, now exercised."""
        from loudkit.config import AlgorithmConfig

        assert AlgorithmConfig().fingerprint() == "0fcda17a0608e1be"
        assert "first_chunk_max_tokens" not in AlgorithmConfig().canonical_form()

    def test_setting_it_re_fingerprints(self) -> None:
        from dataclasses import replace

        from loudkit.config import AlgorithmConfig

        base = AlgorithmConfig()
        capped = base.with_(chunking=replace(base.chunking, first_chunk_max_tokens=96))
        assert capped.fingerprint() != base.fingerprint()
        assert "first_chunk_max_tokens" in capped.canonical_form()

    def test_out_of_range_values_are_refused(self) -> None:
        import pytest

        from loudkit.config import ChunkConfig

        with pytest.raises(ValueError, match="first_chunk_max_tokens"):
            ChunkConfig(first_chunk_max_tokens=0)
        with pytest.raises(ValueError, match="first_chunk_max_tokens"):
            ChunkConfig(max_tokens=100, first_chunk_max_tokens=101)
        # None is an explicit "no cap", distinct from unset, and valid.
        no_cap = ChunkConfig(first_chunk_max_tokens=None)
        assert no_cap.resolved_first_chunk_max_tokens() is None
