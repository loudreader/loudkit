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

    # 181 estimated tokens: comfortably inside a 255-token window, and over the
    # 96-token first-chunk budget. `TEXT` is not usable here — it estimates 263
    # against 255, so the base config already returns two chunks and the name's
    # premise is false of it.
    FITS_ONE_WINDOW = (
        "The quick brown fox jumps over the lazy dog. "
        "A second sentence follows the first one here."
    )

    def test_it_splits_text_that_otherwise_fits_one_window(self) -> None:
        """The feature, stated as an assertion that can fail.

        This was two lines that could not: `assert ... or True` is a tautology,
        and the second had `len(split_text(TEXT, base)) > 1` as an escape
        hatch, which `TEXT` satisfies unconditionally. Verified by mutation —
        disabling `first_chunk_max_tokens` entirely in `split_text` left this
        test green, so the behaviour it is named for was pinned by nothing.
        """
        from loudkit.config import ChunkConfig
        from loudkit.frontend.chunking import split_text

        base = ChunkConfig(max_tokens=255, prefix_tokens=6)
        assert split_text(self.FITS_ONE_WINDOW, base) == [self.FITS_ONE_WINDOW], (
            "the case needs text that fits one window under the plain config"
        )
        capped = ChunkConfig(max_tokens=255, prefix_tokens=6, first_chunk_max_tokens=96)
        assert len(split_text(self.FITS_ONE_WINDOW, capped)) == 2

    def test_unset_changes_nothing(self) -> None:
        from loudkit.config import ChunkConfig
        from loudkit.frontend.chunking import split_text

        assert split_text(self.TEXT, ChunkConfig()) == split_text(
            self.TEXT, ChunkConfig(first_chunk_max_tokens=None)
        )

    def test_unset_is_absent_from_the_fingerprint(self) -> None:
        """Adding the field must not re-fingerprint a config that did not set
        it — the sentinel mechanism `canonical_form` documents, now exercised."""
        from dataclasses import replace

        from loudkit.config import _UNSET, AlgorithmConfig

        assert "first_chunk_max_tokens" not in AlgorithmConfig().canonical_form()
        assert (
            AlgorithmConfig().fingerprint()
            == AlgorithmConfig(
                chunking=replace(AlgorithmConfig().chunking, first_chunk_max_tokens=_UNSET)
            ).fingerprint()
        )

    def test_setting_it_re_fingerprints(self) -> None:
        from dataclasses import replace

        from loudkit.config import AlgorithmConfig

        base = AlgorithmConfig()
        capped = base.with_(chunking=replace(base.chunking, first_chunk_max_tokens=96))
        assert capped.fingerprint() != base.fingerprint()
        assert "first_chunk_max_tokens" in capped.canonical_form()

    def test_out_of_range_values_are_refused(self) -> None:
        from loudkit.config import ChunkConfig

        with pytest.raises(ValueError, match="first_chunk_max_tokens"):
            ChunkConfig(first_chunk_max_tokens=0)
        with pytest.raises(ValueError, match="first_chunk_max_tokens"):
            ChunkConfig(max_tokens=100, first_chunk_max_tokens=101)
        # None is an explicit "no cap", distinct from unset, and valid.
        no_cap = ChunkConfig(first_chunk_max_tokens=None)
        assert no_cap.resolved_first_chunk_max_tokens() is None


class TestMidSentencePeriods:
    """A period that does not end a sentence must not end a chunk.

    `"But Mr. Smith went home"` used to break after the title, and the chunk it
    produced was seven characters: its own utterance, its own derived seed, and
    a token ceiling proportional to seven characters, so it rendered with no
    room for a closing pause. It is the only chunk in a 9920-row rendered
    census that hit that ceiling.

    Surveyed over 1200 passages in ten languages: 59 of 3773 cuts landed on a
    period inside a sentence. Under this law, one.
    """

    # What the funnel hands the splitter for a mid-sentence ellipsis, long
    # enough that the fold is the only `. ` inside the first window.
    ELLIPSIS = (
        "Grzeja sie i swieca. ciepłem ktore pamietaja z lata i z kazdej innej "
        "pory roku na swiecie, a potem gasna powoli i nikt juz nie pamieta o "
        "nich zupelnie nic wiecej."
    )

    # Long enough that the only `. ` inside the first window is the title's.
    TITLE = (
        "But Mr. Smith went home to the house on the hill where he had lived "
        "for forty years without ever once complaining about any of it at all."
    )

    def test_the_defect_itself(self) -> None:
        chunks = split_text(self.TITLE, ChunkConfig())
        assert chunks[0] != "But Mr."
        assert not any(c.endswith("Mr.") for c in chunks)

    def test_the_old_law_is_still_namable(self) -> None:
        """`"break"` reproduces the seven-character chunk exactly. A pack that
        was measured before this field existed can say so."""
        chunks = split_text(self.TITLE, ChunkConfig(mid_sentence_period="break"))
        assert chunks[0] == "But Mr."

    def test_a_sentence_does_not_resume_in_lower_case(self) -> None:
        """The half of the law that no list could do.

        `speech_text` maps an ellipsis to `...` and then folds a run of
        `[.,;:]` to a single mark, so a mid-sentence ellipsis arrives here as a
        period. It is the dominant cause in Polish, which has no abbreviation
        cuts at all.
        """
        from loudkit.frontend.speechtext import speech_text

        prepared = speech_text("Grzeja sie i swieca... ciepłem", "pl")
        assert ". " in prepared, prepared
        assert not split_text(self.ELLIPSIS, ChunkConfig())[0].endswith("swieca.")
        assert split_text(self.ELLIPSIS, ChunkConfig(mid_sentence_period="break"))[0].endswith(
            "swieca."
        )

    def test_only_the_period_is_in_doubt(self) -> None:
        """`!` and `?` end sentences and `;` and `,` do not end them at all.

        Gating on the period is what keeps the lower-case test from vetoing
        every comma in the language: a comma is followed by a lower-case word
        almost every time it is written.
        """
        text = (
            "Alpha beta gamma delta, epsilon zeta eta theta, iota kappa lambda "
            "mu, nu xi omicron pi rho, sigma tau upsilon phi chi psi omega."
        )
        cfg = ChunkConfig()
        assert split_text(text, cfg) == split_text(
            text, ChunkConfig(mid_sentence_period="break")
        )
        assert split_text(text, cfg)[0].endswith(",")

    def test_it_walks_back_through_the_same_separator(self) -> None:
        """Two held candidates in a row, and an earlier one that is real: the
        search must keep walking rather than give up after the first."""
        text = (
            "Alpha ends here. Mrs. Watson and Mr. Holmes then agreed on the "
            "one point that had ever really mattered to either of them at all."
        )
        chunks = split_text(text, ChunkConfig(max_tokens=200, prefix_tokens=0))
        assert chunks[0] == "Alpha ends here."

    def test_it_falls_through_to_a_weaker_separator(self) -> None:
        """When every sentence end in the window is held, the split takes the
        latest comma. A comma break is heard; a chunk of `"Mr."` is heard
        worse."""
        text = (
            "Mr. Norrell, who had been waiting in the hall for the better part "
            "of an hour, said nothing at all to either of them about what he "
            "had seen there."
        )
        assert split_text(text, ChunkConfig())[0].endswith("hour,")

    def test_it_falls_through_to_a_word_boundary(self) -> None:
        """And when there is no weaker separator either, to a word boundary.

        Deliberate, and measured. Taking the held period after all was tried:
        over 1200 passages it saved six word breaks and cost seven period
        breaks, five of which left a title dangling at the end of a chunk. A
        word break is joined with a carried prefix and keeps its contour; a
        false full stop is read as one.
        """
        chunks = split_text(self.TITLE, ChunkConfig())
        assert chunks[0].endswith("of")  # a word boundary, not "Mr."

    def test_the_guard_on_the_character_in_front(self) -> None:
        """`"NASA"` ends in `"A"`, and `"A"` is a listed initial.

        Without the guard every word whose tail spells a listed entry would be
        held, and a list of twenty-two entries of one and two letters makes
        that ordinary rather than exotic.
        """
        text = (
            "The rocket that carried them up there was built by NASA. And the "
            "rest of the afternoon went by without anybody saying very much "
            "about it to anybody else at all."
        )
        assert split_text(text, ChunkConfig())[0].endswith("by NASA.")

    def test_a_title_at_the_very_start_is_held(self) -> None:
        """No character in front of the match at all: the guard must read that
        as a boundary, not index off the front of the string."""
        text = (
            "Mr. Smith went home to the house on the hill where he had lived "
            "for forty years without ever once complaining about any of it."
        )
        assert split_text(text, ChunkConfig())[0] != "Mr."

    def test_a_title_after_an_opening_quote_is_held(self) -> None:
        """The guard admits punctuation in front of the match. Requiring a
        space would have missed every line of dialogue, which is where titles
        actually live."""
        text = (
            "\u201cMr. Smith went home to the house on the hill where he had "
            "lived for forty years without complaining about any of it.\u201d"
        )
        assert not split_text(text, ChunkConfig())[0].endswith("Mr.")

    def test_the_next_character_may_end_the_window(self) -> None:
        """The lower-case test reads one character past the search window.

        The latest candidate can sit exactly at the end of the budget, and
        reading only the window would make the law depend on where the budget
        happens to fall — which the per-voice budget is about to make vary.
        """
        cfg = ChunkConfig(max_tokens=40, prefix_tokens=0)  # a 20-character window
        # The separator starts at index 19, so its space is the window's last
        # character and the "c" that decides the verdict is one past it.
        text = "aa, " + "b" * 15 + ". continues past the window here"
        assert split_text(text, cfg)[0] == "aa,"
        assert (
            split_text(
                text, ChunkConfig(max_tokens=40, prefix_tokens=0, mid_sentence_period="break")
            )[0]
            == "aa, " + "b" * 15 + "."
        )

    def test_an_empty_list_still_holds_on_lower_case(self) -> None:
        """The two halves are independent, and each is measurably worth its
        place: over 1200 passages the list alone leaves 22 harmful cuts and the
        lower-case test alone leaves 42, where together they leave one."""
        assert not split_text(self.ELLIPSIS, ChunkConfig(abbreviations=()))[0].endswith(
            "swieca."
        )
        assert split_text(self.TITLE, ChunkConfig(abbreviations=()))[0] == "But Mr."

    def test_the_list_is_data_and_the_law_is_code(self) -> None:
        """Replacing the tuple is the whole of adding a language."""
        text = (
            "Zobacz np. Tutaj wszystko jest opisane dokladnie tak jak powinno "
            "byc opisane w tym miejscu i nigdzie indziej na calym swiecie."
        )
        base = ChunkConfig(max_tokens=180, prefix_tokens=0)
        assert split_text(text, base)[0].endswith("np.")
        from dataclasses import replace

        assert not split_text(text, replace(base, abbreviations=("np",)))[0].endswith("np.")


class TestMidSentencePeriodConfig:
    def test_rejects_an_unknown_law(self) -> None:
        with pytest.raises(ValueError, match="unknown mid_sentence_period"):
            ChunkConfig(mid_sentence_period="hold-ish")

    def test_rejects_an_empty_entry(self) -> None:
        """An empty string is a suffix of everything: every period would be
        held and every split would fall to a word boundary, silently."""
        with pytest.raises(ValueError, match="empty string"):
            ChunkConfig(abbreviations=("Mr", ""))

    def test_both_fields_are_in_the_fingerprint(self) -> None:
        from loudkit.config import AlgorithmConfig

        base = AlgorithmConfig()
        assert (
            base.with_(chunking=ChunkConfig(mid_sentence_period="break")).fingerprint()
            != base.fingerprint()
        )
        assert (
            base.with_(chunking=ChunkConfig(abbreviations=("Mr",))).fingerprint()
            != base.fingerprint()
        )

    def test_the_shipping_list_is_sorted(self) -> None:
        """Order does not change the verdict but does change the hash, so the
        canonical spelling is sorted and a manifest diff stays readable."""
        entries = list(ChunkConfig().abbreviations)
        assert entries == sorted(entries)
        assert len(entries) == len(set(entries))

    def test_they_sit_where_the_ports_write_them_by_hand(self) -> None:
        """Five canonical forms are hand-written and their keys are sorted, so
        a new field's position in that order is part of the contract."""
        import json

        from loudkit.config import AlgorithmConfig

        chunking = json.loads(AlgorithmConfig().canonical_form())["algorithm"]["chunking"]
        assert list(chunking) == [
            "abbreviations",
            "cap_resplit",
            "enabled",
            "max_tokens",
            "mid_sentence_period",
            "prefix_tokens",
            "split_on",
        ]


class TestSplitInHalf:
    """The repair for a chunk the window could not hold."""

    def test_it_halves_at_a_word_boundary(self) -> None:
        from loudkit.frontend.chunking import split_in_half

        halves = split_in_half("one two three four five six")
        assert halves is not None
        assert " ".join(halves) == "one two three four five six"
        assert all(h == h.strip() for h in halves)

    def test_it_does_not_seek_punctuation(self) -> None:
        """The counter-intuitive half, and the measured one.

        Of 27 re-splits taken at the separator *nearest the middle*, every
        seam over a second fell on a comma; taking the nearest word break
        instead halved that count. A comma is an instruction to pause and this
        model has no pause-duration prior, so it takes the instruction and
        overshoots.

        Not sought is not avoided, and the test says so rather than claiming
        more than the rule delivers: where the nearest word break happens to
        follow a comma, that is the break taken.
        """
        from loudkit.frontend.chunking import split_in_half

        # A weaker separator sits nearer the middle than the comma does, and
        # the comma has no pull of its own.
        halves = split_in_half("aa bb, cc dddddddddddd ee")
        assert halves == ("aa bb, cc", "dddddddddddd ee")

    def test_it_counts_scalars_not_grapheme_clusters(self) -> None:
        """CRLF is one grapheme cluster and two Unicode scalars.

        Swift's ``Array(text)`` yields clusters, so it halved this string one
        word later than the other four ports until it was switched to
        ``unicodeScalars``. The funnel passes CRLF through verbatim, so any
        source with Windows line endings reaches this, and every port's other
        cases are ASCII-only and cannot see it.
        """
        from loudkit.frontend.chunking import split_in_half

        assert split_in_half("xx\r\nxx xx xxxxx") == ("xx\r\nxx", "xx xxxxx")

    def test_it_cuts_on_boundaries_the_funnel_keeps(self) -> None:
        """NBSP and tab survive `speech_text` and are word boundaries.

        Matching only U+0020 made a capped chunk whose separators were all
        non-breaking come back unsplittable, so it shipped its truncation --
        the exact failure `cap_resplit` exists to prevent. NBSP is ordinary in
        real prose: "10 000", French punctuation, typeset copy.
        """
        from loudkit.frontend.chunking import split_in_half
        from loudkit.frontend.speechtext import speech_text

        for sep in ("\u00a0", "\t", "\u202f"):
            assert speech_text(f"alpha{sep}beta", "en") == f"alpha{sep}beta", sep
            halves = split_in_half(f"alpha{sep}beta{sep}gamma")
            assert halves == (f"alpha{sep}beta", "gamma"), sep

    def test_a_single_unbroken_run_is_refused(self) -> None:
        """Splitting it would have to cut a word, which is worse than the
        truncation it would be repairing."""
        from loudkit.frontend.chunking import split_in_half

        assert split_in_half("omringden.") is None
        assert split_in_half("") is None
        # Trimming can empty a half the scan thought was interior. Unreachable
        # through the engine, which strips first, but this is exported.
        assert split_in_half("x \t") is None

    def test_neither_half_is_empty(self) -> None:
        from loudkit.frontend.chunking import split_in_half

        for text in ("a bb", "aaaaaaaa b", "a bbbbbbbb"):
            halves = split_in_half(text)
            assert halves is not None, text
            assert halves[0], text
            assert halves[1], text


class TestWordBoundaryFallback:
    """`split_text`'s last resort, when a window holds no punctuation.

    The boundary table was introduced for `split_in_half` and this fallback,
    ten lines away in the same file, was left matching U+0020 alone. Text whose
    every space is non-breaking -- HTML where `&nbsp;` won, and the funnel
    keeps NBSP -- therefore found no boundary and was cut mid-word, which is
    the failure `cap_resplit` exists to repair, arriving one stage earlier.
    """

    def test_it_breaks_on_a_non_breaking_space(self) -> None:
        from dataclasses import replace

        from loudkit.config import AlgorithmConfig
        from loudkit.frontend.chunking import split_text

        cfg = replace(AlgorithmConfig().chunking, max_tokens=255)
        words = [f"ord{i:02d}" for i in range(30)]
        chunks = split_text("\u00a0".join(words), cfg)
        assert len(chunks) > 1, "the case needs to cross a window"
        for chunk in chunks:
            assert chunk.split("\u00a0")[-1] in words, f"cut mid-word: {chunk[-12:]!r}"


class TestCapResplitConfig:
    def test_the_default_is_the_law(self) -> None:
        from loudkit.config import ChunkConfig

        assert ChunkConfig().cap_resplit == "word"

    def test_unknown_values_are_refused_by_name(self) -> None:
        from loudkit.config import ChunkConfig

        with pytest.raises(ValueError, match="cap_resplit"):
            ChunkConfig(cap_resplit="halve")

    def test_off_names_the_pre_amendment_law(self) -> None:
        """A checkpoint measured before this field has to be able to say so."""
        from dataclasses import replace

        from loudkit.config import AlgorithmConfig

        base = AlgorithmConfig()
        off = base.with_(chunking=replace(base.chunking, cap_resplit="off"))
        assert off.fingerprint() != base.fingerprint()
        assert '"cap_resplit":"off"' in off.canonical_form()
