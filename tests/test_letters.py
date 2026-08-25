"""Acronyms, spelled in the language being read — and shouting left alone."""

from __future__ import annotations

import pytest

from loudkit.frontend.letters import spell_acronym, spell_acronyms, spells_acronyms
from loudkit.frontend.numbers import supported_languages


class TestEveryLanguageSpells:
    def test_all_twelve_have_a_letter_table(self) -> None:
        # Polish had one and the other eleven did not, so `FBI` reached the
        # model as raw graphemes for a grapheme engine to read as a word.
        for language in supported_languages():
            assert spells_acronyms(language), language

    @pytest.mark.parametrize("language", sorted(supported_languages()))
    def test_a_lone_acronym_is_spelled_without_leaving_capitals(self, language: str) -> None:
        said = spell_acronyms("the FBI said", language)
        assert "FBI" not in said, f"{language}: left the capitals in {said!r}"


class TestTheNameIsTheLanguagesOwn:
    @pytest.mark.parametrize(
        ("language", "expect"),
        [
            ("en", "see-eye-ay"),
            ("pl", "ce-i-a"),
            ("de", "ze-i-a"),
            ("es", "ce-i-a"),
            ("fr", "cé-i-a"),
            ("it", "ci-i-a"),
            ("fi", "see-ii-aa"),
            ("sv", "se-i-a"),
        ],
    )
    def test_cia(self, language: str, expect: str) -> None:
        # The whole point: one string, twelve readings. A letter name is written
        # in the target language's own orthography, because the engine is
        # grapheme-based and reads it with that language's letter-to-sound.
        assert spell_acronym("CIA", language) == expect


class TestShoutingIsNotSpelled:
    """`THIS IS IMPORTANT` must not become tee-aitch-eye-ess eye-ess …"""

    @pytest.mark.parametrize("language", sorted(supported_languages()))
    @pytest.mark.parametrize(
        "text", ["THIS IS IMPORTANT", "DO NOT TOUCH THIS", "TO JEST WAŻNE"]
    )
    def test_a_run_of_capitals_is_emphasis(self, language: str, text: str) -> None:
        assert spell_acronyms(f"He said {text} loudly", language).count(text) == 1

    @pytest.mark.parametrize("language", sorted(supported_languages()))
    def test_a_wholly_capitalised_text_is_passed_through(self, language: str) -> None:
        shout = "PLEASE READ THIS NOW"
        assert spell_acronyms(shout, language) == shout

    @pytest.mark.parametrize("language", sorted(supported_languages()))
    def test_an_island_among_ordinary_text_still_spells(self, language: str) -> None:
        # The context rule must not have bought quiet by refusing everything.
        assert spell_acronyms("the FBI said", language) != "the FBI said"


class TestWordsAreNotSpelled:
    def test_nasa_is_a_word_everywhere(self) -> None:
        for language in supported_languages():
            assert spell_acronym("NASA", language) == "nasa", language

    def test_the_word_list_is_national(self) -> None:
        # LOT is an airline in Poland and a common noun in English; only one of
        # them should be spelled out.
        assert spell_acronym("LOT", "pl") == "lot"
        assert spell_acronym("LOT", "en") == "el-oh-tee"

    def test_something_too_long_is_left_alone(self) -> None:
        # A listener can read SIGGRAPH; they cannot un-hear it spelled.
        assert spell_acronym("SIGGRAPH", "en") is None

    def test_a_letter_with_no_name_refuses_the_whole_word(self) -> None:
        # Half-spelling is worse than not spelling: Italian has no `j` name in
        # its own alphabet table beyond the borrowed one, so anything it cannot
        # name entirely is left as written.
        assert spell_acronym("Ab", "en") is None  # not all caps
        assert spell_acronym("A", "en") is None  # too short


class TestThroughTheFunnel:
    @pytest.mark.parametrize(("language", "expect"), [("pl", "ce-i-a"), ("en", "see-eye-ay")])
    def test_the_funnel_spells_in_the_render_language(self, language: str, expect: str) -> None:
        from loudkit.frontend.polish import speech_text

        assert expect in speech_text("The CIA said so.", language)

    def test_the_polish_respeller_no_longer_spells_shouting(self) -> None:
        # It saw one word at a time and so could not tell an initialism from a
        # shout: it spelled "TO JEST WAŻNE" as te-ha-i-es i-es …
        from loudkit.frontend.polish import speech_text

        assert "TO JEST WAŻNE" in speech_text("TO JEST WAŻNE.", "pl")
