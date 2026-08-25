"""The number verbalizer, against hand-written expectations.

The fixture is the contract five implementations share; these tests add the
things a fixture cannot express — that the grammar file is well-formed, that
failure is loud, and that composition holds across ranges nobody enumerated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from loudkit.frontend.numbers import (
    NumberGrammarError,
    cardinal,
    expand,
    expand_times,
    supported_languages,
)

FIXTURE = Path(__file__).parent / "data" / "conformance" / "numbers.json"


@pytest.fixture(scope="module")
def fx() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class TestAgainstTheFixture:
    def test_every_cardinal(self, fx: dict[str, Any]) -> None:
        assert fx["cardinals"], "the fixture has no cardinals; nothing was compared"
        for language, cases in fx["cardinals"].items():
            for case in cases:
                got = cardinal(case["value"], language)
                assert got == case["expect"], f"{language} {case['value']}"

    def test_every_agreement(self, fx: dict[str, Any]) -> None:
        assert fx["gendered"], "the fixture has no gendered cases"
        for case in fx["gendered"]:
            got = cardinal(case["value"], case["language"], gender=case["gender"])
            assert got == case["expect"], f"{case['language']} {case['value']} {case['gender']}"

    def test_the_fixture_covers_every_shipped_language(self, fx: dict[str, Any]) -> None:
        # A language with a grammar and no cases is a language nobody checked.
        assert set(fx["cardinals"]) == set(supported_languages())


class TestTheGrammarsAreWellFormed:
    """Structural invariants, so a malformed entry fails here rather than as a
    wrong word in a range the fixture happens not to cover."""

    @pytest.mark.parametrize("language", supported_languages())
    def test_every_value_under_a_thousand_says_something(self, language: str) -> None:
        for value in range(1000):
            said = cardinal(value, language)
            assert said, f"{language} {value} produced nothing"
            assert not any(ch.isdigit() for ch in said), (
                f"{language} {value} left digits in {said!r} — a number that reaches the "
                "model as digits is the failure this module exists to remove"
            )

    @pytest.mark.parametrize("language", supported_languages())
    def test_no_word_is_produced_with_stray_whitespace(self, language: str) -> None:
        for value in (0, 21, 101, 999, 1000, 1001, 2026, 100000):
            said = cardinal(value, language)
            assert said == said.strip(), f"{language} {value}: {said!r}"
            assert "  " not in said, f"{language} {value}: doubled space in {said!r}"

    @pytest.mark.parametrize("language", supported_languages())
    def test_distinct_values_get_distinct_words(self, language: str) -> None:
        # Not a deep property — but a collision under a hundred means a table
        # entry is duplicated, which is the likeliest way to mistype a grammar.
        seen: dict[str, int] = {}
        for value in range(100):
            said = cardinal(value, language)
            assert said not in seen, f"{language}: {value} and {seen[said]} both say {said!r}"
            seen[said] = value


class TestFailureIsLoud:
    def test_an_unknown_language_raises(self) -> None:
        with pytest.raises(NumberGrammarError, match="no number grammar"):
            cardinal(1, "xx")

    def test_a_value_past_the_largest_scale_raises(self) -> None:
        # Silently reading digits back would be indistinguishable from success.
        with pytest.raises(NumberGrammarError):
            cardinal(10**15, "pl")

    def test_negatives_are_read_not_dropped(self) -> None:
        assert cardinal(-5, "en") == "minus five"


class TestAgreement:
    def test_an_unknown_gender_falls_back_to_the_citation_form(self) -> None:
        # A caller who guesses a gender name should get a usable number rather
        # than an exception: the failure is a wrong inflection, not a crash, and
        # a crash in the frontend costs the whole utterance.
        assert cardinal(2, "pl", gender="nonexistent") == cardinal(2, "pl")

    def test_a_language_without_agreement_ignores_the_argument(self) -> None:
        assert cardinal(1, "en", gender="f") == "one"

    def test_polish_virile_is_a_distinct_series(self) -> None:
        # dwaj/trzej/czterej are the forms for groups of men, and they are the
        # single most visible thing a morphology-blind normaliser gets wrong.
        assert cardinal(2, "pl", gender="virile") == "dwaj"
        assert cardinal(3, "pl", gender="virile") == "trzej"
        assert cardinal(2, "pl", gender="f") == "dwie"
        assert cardinal(2, "pl") == "dwa"


class TestSlavicScaleInflection:
    """Polish inflects the scale noun by its multiplier, and the rule reads the
    last two digits rather than the whole number."""

    @pytest.mark.parametrize(
        ("value", "expect"),
        [
            (1000, "tysiąc"),
            (2000, "dwa tysiące"),
            (5000, "pięć tysięcy"),
            (12000, "dwanaście tysięcy"),
            (22000, "dwadzieścia dwa tysiące"),
            (112000, "sto dwanaście tysięcy"),
            (122000, "sto dwadzieścia dwa tysiące"),
        ],
    )
    def test_thousands(self, value: int, expect: str) -> None:
        assert cardinal(value, "pl") == expect


class TestExpandInRunningText:
    """The seam between this module and the funnel.

    `expand` differs from `cardinal` in one deliberate way: it never raises and
    never leaves digits behind. A library call has a caller who can decide what
    to do about an impossible number; a text funnel has a user whose sentence
    still has to be spoken.
    """

    def test_a_plain_integer(self) -> None:
        assert expand("I have 21 apples.", "en") == "I have twenty-one apples."

    def test_a_decimal_is_read_digit_by_digit_after_the_mark(self) -> None:
        # "three point five", never "three point five tenths": a decimal is a
        # sequence of digits, and leading zeros there carry meaning a cardinal
        # would eat.
        assert expand("3.5", "en") == "three point five"
        assert expand("0.49", "en") == "zero point four nine"

    def test_the_decimal_mark_is_the_languages_own(self) -> None:
        # Eight of the nine write the decimal with a comma. Reading an English
        # grouping comma aloud would be absurd, so the other mark is dropped —
        # which is what a reader does with it.
        assert expand("3,5", "pl") == "trzy przecinek pięć"
        assert expand("1,200", "en") == "one thousand two hundred"

    def test_a_number_past_every_scale_is_read_digit_by_digit(self) -> None:
        # Not an error: a number this size in running text is an identifier.
        said = expand("Order 98765432109876543210 shipped.", "en")
        assert not any(ch.isdigit() for ch in said)
        assert said.startswith("Order nine eight seven")

    def test_an_unknown_language_leaves_the_text_alone(self) -> None:
        # Better a digit the model mumbles than a crash that costs the sentence.
        assert expand("21 apples", "xx") == "21 apples"

    def test_nothing_is_invented_where_there_are_no_digits(self) -> None:
        for language in supported_languages():
            assert expand("no numbers here", language) == "no numbers here"


class TestDigitsThatAreNotQuantities:
    """A digit run with two or more separators is a version, an address or a
    date — never a number.

    All three used to be read as numbers, and the way they failed was worse than
    the failure. In English, where the decimal mark is the dot, ``str.partition``
    split ``1.2.3`` once and handed ``"2.3"`` to a digit-by-digit reader, so
    every character of it reached ``int()`` and raised ``ValueError`` — a crash
    in ``Engine.synthesize``, on a sentence anyone might write. Where the decimal
    mark is the comma the dots were treated as thousands grouping and the
    segments were concatenated, so ``192.168.0.1`` was spoken, confidently, as
    *nineteen million two hundred sixteen thousand eight hundred one*.

    These are regression tests: each string below is one that shipped wrong.
    """

    NOT_QUANTITIES = ["1.2.3", "1.2.3.4", "192.168.0.1", "12.03.2026", "10.0.0.255"]

    @pytest.mark.parametrize("language", sorted(supported_languages()))
    @pytest.mark.parametrize("literal", NOT_QUANTITIES)
    def test_left_exactly_as_written(self, language: str, literal: str) -> None:
        assert expand(literal, language) == literal

    @pytest.mark.parametrize("language", sorted(supported_languages()))
    @pytest.mark.parametrize("literal", NOT_QUANTITIES)
    def test_never_raises(self, language: str, literal: str) -> None:
        """The crash was the reason this class exists, so it is asserted apart
        from the value: a future rewrite that returns something wrong is a bug,
        one that raises is an outage."""
        expand(f"see {literal} for details", language)

    @pytest.mark.parametrize("language", sorted(supported_languages()))
    def test_real_numbers_still_read(self, language: str) -> None:
        """The guard must not have bought correctness by refusing everything."""
        assert expand("7", language) != "7"
        # One separator is a decimal or a grouping, depending on the language's
        # own marks — either way it is a quantity and must be spoken.
        assert expand("2,5", language) != "2,5"
        assert expand("2.5", language) != "2.5"

    @pytest.mark.parametrize("language", sorted(set(supported_languages()) - {"en"}))
    def test_grouped_thousands_are_still_a_number(self, language: str) -> None:
        """Two separators that *group* are a number: 1.234.567 is a million and
        a bit in all eleven languages whose decimal mark is the comma. The rule
        is 'three digits after the first separator', not 'at most one separator'
        — a guard that refused every multi-separator run would silently stop
        reading grouped thousands, which is a regression dressed as a fix."""
        assert expand("1.234.567", language) != "1.234.567"

    def test_grouped_thousands_with_commas(self) -> None:
        """English is the one language here whose decimal mark is the dot, so
        it is the only one that groups with commas."""
        assert expand("1,234,567", "en") != "1,234,567"


class TestATimeIsNotPartOfADate:
    """``12.03.2026`` is a date, and ``expand_times`` used to eat its first half.

    The pattern matched ``12.03`` inside it — a word boundary sits between a
    digit and a dot — so the ordinary written date of five of the twelve
    languages was spoken as a clock time with the year trailing behind it:
    *zwölf Uhr drei, zweitausendsechsundzwanzig*.
    """

    @pytest.mark.parametrize("language", sorted(supported_languages()))
    def test_a_dotted_date_is_not_a_time(self, language: str) -> None:
        assert expand_times("12.03.2026", language) == "12.03.2026"
        assert expand_times("am 05.11.2025 kam", language) == "am 05.11.2025 kam"

    @pytest.mark.parametrize("language", sorted(supported_languages()))
    def test_a_real_time_still_reads(self, language: str) -> None:
        assert expand_times("14:30", language) != "14:30"
        # A sentence-final time keeps working: the dot after it is followed by
        # nothing, not by a digit.
        assert expand_times("at 14:30.", language) != "at 14:30."

    @pytest.mark.parametrize("language", sorted(supported_languages()))
    def test_a_dotted_time_reads_only_where_the_dot_is_not_a_decimal_point(
        self, language: str
    ) -> None:
        """`14.30` is half past two in eleven of these languages and a number in
        the twelfth, and the grammar file is what says which.

        A language that writes clock times with a dot does not use the dot as
        its decimal separator — German writes `14.30 Uhr` and `2,50 €`, English
        writes `2:30` and `$2.50`. This assertion used to read `!= "14.30"` for
        all twelve, which made every English decimal with two fraction digits a
        clock time: `$0.49` was *zero forty-nine dollars* and `3.14` was *three
        fourteen*. Only two-digit fractions were affected, so `0.5` and `25.99`
        stayed right and it went unnoticed — and the shared fixture pinned
        `decimal 0.49` with the wrong reading, so all five implementations
        reproduced it exactly and reported parity.
        """
        from loudkit.frontend.numbers import _grammars

        dotted_is_a_time = _grammars()[language].decimal_separator != "."
        assert (expand_times("14.30", language) != "14.30") is dotted_is_a_time

    def test_an_english_decimal_is_a_decimal(self) -> None:
        from loudkit.frontend.polish import speech_text

        assert speech_text("Pi equals 3.14.", "en") == "Pi equals three point one four."
        assert speech_text("It costs $0.49.", "en") == "It costs zero point four nine dollars."


class TestAWrittenInfixIsNotSaidTwice:
    """German writes the time with the word the spoken form also carries.

    ``um 14.30 Uhr`` was read *vierzehn Uhr dreißig Uhr* — the reading appends
    the grammar's infix between hour and minutes, where it belongs, and the
    written ``Uhr`` behind the digits stood there besides. The written word is
    that same spoken token, so a time followed immediately by it consumes it.
    Every German clock sentence in conventional written form hit this; no test
    looked, because every existing "Uhr" assertion targeted the decimal/time
    disambiguation and Polish, the fixture's other language, has no infix.
    """

    def test_the_written_word_is_consumed(self) -> None:
        assert expand_times("um 14:30 Uhr", "de") == "um vierzehn Uhr dreißig"
        # Tab before the word consumes exactly like a space.
        assert expand_times("um 14:30\tUhr", "de") == "um vierzehn Uhr dreißig"
        assert expand_times("um 24:00 Uhr an.", "de") == "um vierundzwanzig Uhr an."

    def test_the_dotted_form_too(self) -> None:
        assert expand_times("Termin um 14.30 Uhr.", "de") == "Termin um vierzehn Uhr dreißig."

    def test_without_the_word_nothing_changes(self) -> None:
        assert expand_times("um 14:30", "de") == "um vierzehn Uhr dreißig"

    def test_a_standalone_uhr_is_not_touched(self) -> None:
        """The noun on its own — *die Uhr tickt* — is not part of any time."""
        assert (
            expand_times("Es ist 14:30 Uhr und die Uhr tickt.", "de")
            == "Es ist vierzehn Uhr dreißig und die Uhr tickt."
        )

    def test_infix_inside_a_longer_word_stays_whole(self) -> None:
        """*Uhrzeit* is one word; consuming its head would mangle the rest."""
        assert (
            expand_times("Die Uhrzeit ist 14:30.", "de")
            == "Die Uhrzeit ist vierzehn Uhr dreißig."
        )

    def test_other_languages_have_no_infix_to_double(self) -> None:
        """Eleven of the twelve grammars carry an empty infix, so there is
        nothing to consume and nothing to double; the next word survives."""
        assert expand_times("at 14:30 sharp", "en") == "at fourteen thirty sharp"


class TestMeaningThatUsedToBeLost:
    """Four readings that changed what the text said, not just how it sounded."""

    @pytest.mark.parametrize("language", sorted(supported_languages()))
    def test_a_leading_minus_is_spoken(self, language: str) -> None:
        # "-5 degrees" was read as *five degrees*: the punctuation pass turned a
        # hyphen outside a word into a space, and by then nothing knew a number
        # had lost its sign. A temperature became its own opposite.
        said = expand("-5", language)
        assert said != expand("5", language), f"{language}: the sign vanished"

    def test_a_hyphen_between_numbers_is_still_a_range(self) -> None:
        # The minus must not eat a range: "pages 3-5" is two numbers.
        assert expand("pages 3-5", "en") == "pages three-five"

    @pytest.mark.parametrize("language", sorted(supported_languages()))
    def test_spaces_group_thousands(self, language: str) -> None:
        # "200 000" was *two hundred zero zero zero*.
        assert expand("200 000", language) == expand("200000", language)

    def test_a_four_digit_group_does_not_join_its_neighbour(self) -> None:
        # "in 2024 200 people" is two numbers, and joining them would invent
        # 2024200. Only a first group of one to three digits can start a
        # space-grouped number.
        said = expand("2024 200", "en")
        assert said == "two thousand and twenty-four two hundred"

    @pytest.mark.parametrize(
        ("text", "expect"),
        [
            # A one- or two-digit tail is not a thousands group, so the space in
            # front of it is an ordinary space and the number behind it is a
            # token of its own. `text[i:i+3].isdigit()` is True of a one-character
            # slice, and that is how the reference came to refuse these: the `2`
            # counted as a group, the walk crossed the space into `R2`, and a
            # number nothing was glued to went unsaid. Go, Rust and JS all check
            # the width; this was the reference wrong and three ports right.
            ("R2 2", "R2 two"),
            ("R2 12", "R2 twelve"),
            ("R2 5", "R2 five"),
            # Still refused: three digits and no fourth *is* a group, so the
            # space is inside the number and the token reaches the letter.
            ("x200 000", "x200 000"),
            ("200 000x", "200 000x"),
            ("a1 000 000", "a1 000 000"),
        ],
    )
    def test_a_short_tail_is_not_a_thousands_group(self, text: str, expect: str) -> None:
        assert expand(text, "en") == expect

    @pytest.mark.parametrize(
        ("text", "language", "expect"),
        [
            # A ragged group — three digits and a fourth — is why the pattern
            # refused to bind the run, so the forward walk has to reach it
            # rather than stop at it. `1 0023R` matched the `1` alone and read
            # "en 0023R": half a run spoken with the rest welded to a letter,
            # which is the class the right-hand guard exists to stop. Go and
            # Rust never had it, because their engines bind `1 002` greedily and
            # find the `R` behind it.
            ("1 0023R", "da", "1 0023R"),
            ("1 234 5672.5E+1", "da", "1 234 5672.5E+1"),
        ],
    )
    def test_a_ragged_run_reaches_the_letter_glued_to_it(
        self, text: str, language: str, expect: str
    ) -> None:
        assert expand(text, language) == expect

    @pytest.mark.parametrize(
        ("text", "language", "expect"),
        [
            # ...and the backward walk may *not* be that loose, which is the one
            # asymmetry in the two walks. `1000` is four digits, so it is no
            # thousands group and the space behind it groups nothing; a walk
            # that crossed anyway found the `e` of an unrelated exponent and
            # left a whole number unsaid. Measured, not argued: the loose
            # question in both directions changes 60 readings in 4800 generated
            # sentences and 56 of them are losses.
            ("e3 1000", "sv", "e3 ettusen"),
            ("1e+3 1000", "sv", "1e+3 ettusen"),
            # Three digits and not fewer, so the walk stops where the run stops.
            ("1000 5.1e+3", "en", "one thousand 5.1e+3"),
            ("R2 5 iOS", "de", "R2 fünf iOS"),
        ],
    )
    def test_the_backward_walk_stays_inside_its_own_number(
        self, text: str, language: str, expect: str
    ) -> None:
        assert expand(text, language) == expect

    def test_the_walks_ask_about_ascii_digits(self) -> None:
        # `str.isdigit` is true of `²` and of every Unicode decimal digit, none
        # of which `_DIGIT_RUN` can match, so the walk answered questions about
        # characters that are not part of any number it reads: `R²` counted as a
        # digit behind the space and swallowed the token after it. Arabic-Indic
        # digits are folded to ASCII before the funnel reaches here.
        assert expand("R² 200", "en") == "R² two hundred"

    @pytest.mark.parametrize(
        ("text", "expect"),
        [
            ("1st", "first"),
            ("2nd", "second"),
            ("5th place", "fifth place"),
            ("the 22nd", "the twenty-second"),
            ("40th", "fortieth"),
        ],
    )
    def test_english_ordinals_are_words(self, text: str, expect: str) -> None:
        # The number pass expanded the digits and left the suffix: *onest*,
        # *fiveth place*, *twenty-twond*.
        from loudkit.frontend.dates import expand_ordinals

        assert expand_ordinals(text, "en") == expect

    def test_an_ordinal_past_the_tables_is_left_whole(self) -> None:
        # Half-saying it would be worse than not saying it.
        from loudkit.frontend.dates import expand_ordinals

        assert expand_ordinals("1000th", "en") == "1000th"
