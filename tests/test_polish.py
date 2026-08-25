"""The Polish speech funnel — a bit-parity port of the Swift engine's
SpeechText + LexicalRespelling.

The shipped Swift engine reads Polish text through ``SpeechText.prepared``
before tokenising; this is the Python half of that contract, so the two
engines read the same Polish text identically. The expected values below are
the ones the Swift code's own comments document and the ear test approved —
a drift here means the Python reader and the Swift reader disagree on Polish.
"""

from __future__ import annotations

import pytest

from loudkit.frontend.polish import (
    _drop_footnote_markers,
    _punctuation_for_speech,
    _speak_symbols,
    _strip_invisibles,
    lexical_respelling,
    speech_text,
)


class TestFunnel:
    def test_invisibles_are_stripped(self) -> None:
        assert _strip_invisibles("he\u200bllo") == "hello"
        assert _strip_invisibles("\ufefflead") == "lead"
        assert _strip_invisibles("plain") == "plain"

    def test_symbols_become_words(self) -> None:
        assert _speak_symbols("15%", "pl") == "15 procent "
        # `5 €` is a currency *suffix* now, so it leaves this pass spaced the
        # way `$5` does rather than with the generic symbol loop's padding —
        # which the whitespace pass collapsed to the same thing either way. The
        # reason it moved is `0.49¢`: an amount followed by a currency mark is a
        # price, and reaching the clock reader with its dot intact made German
        # say "null Uhr neunundvierzig Cent".
        assert _speak_symbols("5 €", "pl") == "5 euro"
        assert _speak_symbols("a → b", "pl") == "a ,  b"

    def test_currency_prefix_becomes_suffix(self) -> None:
        # "$5" reads "5 dollars", not "dollars 5"
        assert _speak_symbols("$5", "en") == "5 dollars"
        # seven of nine used to hear English; now they hear their own
        assert _speak_symbols("$5", "de") == "5 Dollar"
        assert _speak_symbols("€10", "es") == "10 euros"

    def test_footnote_markers_are_dropped(self) -> None:
        assert _drop_footnote_markers("text[12]") == "text"
        assert _drop_footnote_markers("a[3, 4]b[1-5]") == "ab"
        # a real bracketed phrase survives (bounded at 20 chars)
        assert "[a real phrase]" in _drop_footnote_markers("[a real phrase]")

    def test_non_prosodic_punctuation_becomes_space(self) -> None:
        out = _punctuation_for_speech("hello#world")
        assert "#" not in out
        assert "hello" in out
        assert "world" in out

    def test_prosodic_punctuation_survives(self) -> None:
        assert _punctuation_for_speech("Hello, world!") == "Hello, world!"

    def test_number_separators_survive_between_digits(self) -> None:
        assert _punctuation_for_speech("2.5") == "2.5"
        assert _punctuation_for_speech("3/4") == "3/4"
        # a lone period is prosodic and survives
        assert _punctuation_for_speech("hello. world") == "hello. world"

    def test_in_word_hyphen_survives(self) -> None:
        assert _punctuation_for_speech("well-known") == "well-known"


class TestRespelling:
    def test_curated_lexicon(self) -> None:
        for word, want in [
            ("download", "dałnloud"),
            ("deadline", "dedlajn"),
            ("feedback", "fidbek"),
            ("weekend", "łikend"),
            ("workflow", "łorkfloł"),
            ("github", "githab"),
            ("release", "rilis"),
        ]:
            assert lexical_respelling(word, "pl") == want, word

    def test_case_is_preserved(self) -> None:
        assert lexical_respelling("GitHub", "pl") == "Githab"
        assert lexical_respelling("Download", "pl") == "Dałnloud"

    def test_phrases_respell_as_a_unit(self) -> None:
        assert lexical_respelling("release notes", "pl") == "rilis nołc"
        assert lexical_respelling("pull request", "pl") == "pul rekłest"
        assert lexical_respelling("code review", "pl") == "koud riwju"

    def test_only_polish_is_respelled(self) -> None:
        assert lexical_respelling("download", "en") == "download"

    def test_numbers_become_cardinals(self) -> None:
        assert lexical_respelling("0", "pl") == "zero"
        assert lexical_respelling("1", "pl") == "jeden"
        assert lexical_respelling("15", "pl") == "piętnaście"
        assert lexical_respelling("101", "pl") == "sto jeden"
        assert lexical_respelling("1234", "pl") == "tysiąc dwieście trzydzieści cztery"

    def test_decimals_read_whole_comma_fraction(self) -> None:
        assert lexical_respelling("2.5", "pl") == "dwa przecinek pięć"

    def test_acronyms_are_spelled_earlier_in_the_funnel_now(self) -> None:
        """The respeller no longer owns this decision.

        It saw one word at a time, so it could not tell an initialism from a
        shout and spelled "TO JEST WAŻNE" letter by letter.
        `loudkit.letters.spell_acronyms` decides for all twelve languages while
        the surrounding capitals are still visible; the respeller now sees the
        already-spelled lowercase form and leaves it alone.
        """
        from loudkit.frontend.letters import spell_acronyms
        from loudkit.frontend.polish import speech_text

        assert spell_acronyms("GPT", "pl") == "gie-pe-te"
        assert spell_acronyms("USB", "pl") == "u-es-be"
        # word-acronyms keep their word form
        assert spell_acronyms("NASA", "pl") == "nasa"
        assert spell_acronyms("PIN", "pl") == "pin"
        # and the whole funnel still produces the Polish letter names
        assert "gie-pe-te" in speech_text("Model GPT jest dobry.", "pl")

    def test_english_word_alone_stays_polish(self) -> None:
        # gated out of the lexicon by the frequency gate
        assert lexical_respelling("brown", "pl") == "brown"

    def test_english_span_transliterates(self) -> None:
        # inside a 4+ word span the gate is ignored
        assert lexical_respelling("the quick brown fox", "pl") == "da kłyk brałn faks"

    def test_inflection_via_stem(self) -> None:
        assert lexical_respelling("update", "pl") == "apdejt"
        assert lexical_respelling("updates", "pl") == "apdejc"
        # apostrophe form: the ending survives the respelling, vowel-folded
        # ("dedlajn" ends in a consonant, so "u" just appends)
        assert lexical_respelling("deadline'u", "pl") == "dedlajnu"

    def test_polish_words_are_untouched(self) -> None:
        assert lexical_respelling("temperatura", "pl") == "temperatura"
        assert lexical_respelling("piątku", "pl") == "piątku"


class TestEndToEnd:
    def test_polish_sentence_with_anglicism(self) -> None:
        assert speech_text("Pobierz download i zrób code review.", "pl") == (
            "Pobierz dałnloud i zrób koud riwju."
        )

    def test_polish_sentence_with_percent(self) -> None:
        assert speech_text("Rabat 15% na weekend!", "pl") == (
            "Rabat piętnaście procent na łikend!"
        )

    def test_english_sentence_passes_through_funnel(self) -> None:
        # the funnel is a no-op on clean prose except collapsing runs of spaces
        out = speech_text("The quick brown fox jumps over the lazy dog.", "en")
        assert "The quick brown fox jumps over the lazy dog." in out

    def test_polish_needs_the_lexicon(self) -> None:
        # loading the 6.5 MB lexicon lazily is what non-Polish runs avoid
        pytest.importorskip("importlib.resources")
        from loudkit.frontend.polish import PolishLexicon

        assert "download" in PolishLexicon.generated()


class TestSharedFunnelFixture:
    """The funnel against the fixture the Swift target checks too.

    Hand-written cases in five languages are five tests of five different
    things: this file had twenty of them and the Swift funnel — the
    implementation the others are described as bit-parity ports *of* — had
    none. One fixture is one test of one thing, and it found three real
    divergences the first time it ran, including Swift deleting Arabic-Indic
    digits from a Polish passage outright.
    """

    @staticmethod
    def _fixture() -> dict:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parent / "data" / "conformance" / "speechtext.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_every_case_matches(self) -> None:
        cases = self._fixture()["cases"]
        assert cases, "the fixture is empty; nothing was compared"
        for case in cases:
            got = speech_text(case["text"], case["language"])
            assert got == case["expected"], (
                f"{case['text']!r} / {case['language']!r}: "
                f"expected {case['expected']!r}, got {got!r}"
            )

    def test_disputed_cases_still_differ_as_recorded(self) -> None:
        """The known Swift/Python disagreements, pinned so they cannot drift.

        These are judgements about how a Polish reader says a loanword, not
        bugs with an obvious side — so they are recorded rather than resolved.
        Pinning the Python half means a silent change to either implementation
        shows up here instead of quietly making the record wrong.
        """
        for case in self._fixture()["disputed"]:
            assert speech_text(case["text"], case["language"]) == case["python"], (
                f"the recorded Python output for {case['text']!r} is stale"
            )

    def test_unicode_digits_are_never_dropped(self) -> None:
        """Losing text is not an available outcome.

        The Swift funnel returned an empty string for a Polish passage of
        Arabic-Indic numerals: `Int("١٢٣")` is nil there, and the digit-by-digit
        fallback mapped through an ASCII-keyed table with `compactMap`, which
        drops what it cannot map. Pinned on this side too, because a port
        rewriting the number path is exactly when it would come back.

        The English half used to assert the opposite — that the digits arrived
        at the model as written — and that was the divergence, not the pin. The
        number pass is ASCII by design, so eleven languages passed these through
        untouched while Polish read them, because the respeller's own digit test
        is `str.isdigit()` and that is true of every Unicode decimal digit. One
        funnel, one fingerprint, two answers. They are folded to ASCII beside
        NFC now, so all twelve say the same number.
        """
        assert speech_text("١٢٣", "pl") == speech_text("123", "pl")
        assert speech_text("١٢٣", "en") == speech_text("123", "en")
        # The separator that travels with them, and the reason the fold has to
        # know the language: U+066B is a decimal point, and in the eleven
        # comma-decimal languages folding it to a dot would make "٣٫١٤" the
        # written form of a clock time.
        assert speech_text("٣٫١٤", "en") == speech_text("3.14", "en")
        assert speech_text("٣٫١٤", "pl") == speech_text("3,14", "pl")
        assert speech_text("٣٫١٤", "de") == speech_text("3,14", "de")
