"""Dates, against what the national language authorities actually say.

Every expectation here traces to a source named in `docs/dates.md`; the ones
that would be wrong in a plausible-looking way are called out in the test that
pins them, because those are the ones a future rewrite will get wrong again.
"""

from __future__ import annotations

import pytest

from loudkit.frontend.dates import (
    expand_dates,
    month_name,
    ordinal_day,
    say_year,
    supported_languages,
)


class TestTheRosterIsComplete:
    def test_every_number_language_can_say_a_date(self) -> None:
        from loudkit.frontend.numbers import supported_languages as numbers

        assert set(supported_languages()) == set(numbers())

    @pytest.mark.parametrize("language", supported_languages())
    def test_every_day_and_month_says_something(self, language: str) -> None:
        for day in range(1, 32):
            said = ordinal_day(day, language)
            assert said, f"{language} day {day} produced nothing"
            assert not any(ch.isdigit() for ch in said), f"{language} day {day}: {said!r}"
        for month in range(1, 13):
            said = month_name(month, language)
            assert said, f"{language} month {month} produced nothing"
            assert not any(ch.isdigit() for ch in said), f"{language} month {month}: {said!r}"


class TestTheIrregularsAGeneratorGetsWrong:
    """The day words are written out rather than derived, and this is why."""

    @pytest.mark.parametrize(
        ("language", "day", "expect"),
        [
            # A naive "+th" produces *fiveth, *eightth, *nineth, *twelveth.
            ("en", 5, "fifth"),
            ("en", 8, "eighth"),
            ("en", 9, "ninth"),
            ("en", 12, "twelfth"),
            ("en", 20, "twentieth"),
            # Duden: siebte loses the -en-, achte has one t.
            ("de", 7, "siebte"),
            ("de", 8, "achte"),
            # ...but 27 keeps the full sieben.
            ("de", 27, "siebenundzwanzigste"),
            # Retskrivningsordbogen: `elvte` was added and then withdrawn, and
            # the crowd-sourced ordinal lists still carry it. 30 is `tredivte`,
            # not *trediveste*.
            ("da", 11, "ellevte"),
            ("da", 30, "tredivte"),
            # Danish never reformed the unit-before-ten order; Norwegian did, in
            # 1951. Emitting `enogtyvende` for Norwegian would be Danish.
            ("da", 21, "enogtyvende"),
            ("no", 21, "tjueførste"),
            # SAOL: `artonde`, and `adertonde` is not in it.
            ("sv", 18, "artonde"),
            ("sv", 11, "elfte"),
            # Finnish repeats the ordinal suffix in every part of a compound.
            ("fi", 21, "kahdeskymmenesensimmäinen"),
            ("fi", 12, "kahdestoista"),
            # Polish dates are genitive: the nominative is what a naive port
            # produces and what a listener notices first.
            ("pl", 12, "dwunastego"),
            ("pl", 21, "dwudziestego pierwszego"),
        ],
    )
    def test_day_word(self, language: str, day: int, expect: str) -> None:
        assert ordinal_day(day, language) == expect

    def test_german_selects_its_ending_by_the_frame(self) -> None:
        # "der zwölfte März" but "am zwölften März" — a weak adjective, and the
        # only inflection-by-frame among the twelve.
        assert ordinal_day(12, "de") == "zwölfte"
        assert ordinal_day(12, "de", oblique=True) == "zwölften"
        assert expand_dates("der 12. März 2026", "de").startswith("der zwölfte")
        assert expand_dates("am 12. März 2026", "de").startswith("am zwölften")


class TestTheMonthIsNotAlwaysNominative:
    def test_polish_months_are_genitive(self) -> None:
        # "12 marca", never "12 marzec" — the nominative in a date is the
        # mistake that gives a Polish listener the strongest jolt.
        assert month_name(3, "pl") == "marca"
        assert month_name(1, "pl") == "stycznia"

    def test_finnish_months_are_partitive(self) -> None:
        # Kotus: the day is the head and the month a partitive complement —
        # "the twelfth [day] *of* March".
        assert month_name(3, "fi") == "maaliskuuta"

    def test_everyone_else_is_nominative(self) -> None:
        assert month_name(3, "de") == "März"
        assert month_name(3, "es") == "marzo"
        assert month_name(3, "sv") == "mars"


class TestTheDayFormPerLanguage:
    @pytest.mark.parametrize(
        ("language", "text", "expect_in"),
        [
            # Cardinal languages: the day is just a number.
            ("nl", "1 mei", "één"),  # the stressed numeral, not the article
            ("nl", "12 maart", "twaalf"),
            # DPD: "uno" in Spain, "primero" in America. Spain is the default.
            ("es", "1 de mayo", "uno"),
            # Ciberdúvidas 31237: a bare numeral reads as a cardinal in pt-PT.
            ("pt", "1 de maio", "um"),
            # Cardinal except the first: French and Italian only.
            ("fr", "1 mai", "premier"),
            ("fr", "12 mars", "douze"),
            ("it", "1 marzo", "primo"),
            ("it", "12 marzo", "dodici"),
        ],
    )
    def test_day(self, language: str, text: str, expect_in: str) -> None:
        assert expect_in in expand_dates(text, language)


class TestYears:
    @pytest.mark.parametrize(
        ("language", "year", "expect"),
        [
            ("en", 1992, "nineteen ninety-two"),
            ("en", 1905, "nineteen oh five"),  # never "nineteen five"
            ("en", 1900, "nineteen hundred"),
            ("en", 2026, "twenty twenty-six"),
            # GfdS explicitly rejects `zwanzighundert…`: German did not follow
            # the English "twenty-sixteen" shift.
            ("de", 1992, "neunzehnhundertzweiundneunzig"),
            # Isof has recommended the tjugohundra- series for decades.
            ("sv", 1992, "nittonhundranittiotvå"),
            # PWN's worked example. Only the tens and units decline.
            ("pl", 1992, "tysiąc dziewięćset dziewięćdziesiątego drugiego"),
            ("pl", 2026, "dwa tysiące dwudziestego szóstego"),
            # PWN rejects *dwutysięczny pierwszy* for 2001.
            ("pl", 2000, "dwutysięcznego"),
            ("pl", 2001, "dwa tysiące pierwszego"),
        ],
    )
    def test_year(self, language: str, year: int, expect: str) -> None:
        assert say_year(year, language) == expect

    def test_spanish_never_splits_a_year(self) -> None:
        # RAE, verbatim: a year is read as its cardinal "y no por bloques de dos
        # cifras, como sucede en inglés" — 2021 is dos mil veintiuno, never
        # veinte veintiuno.
        assert say_year(2021, "es") == "dos mil veintiuno"
        assert "veinte veinti" not in say_year(2021, "es")


class TestThePrepositionsThatAreSpoken:
    def test_spanish_speaks_both_de(self) -> None:
        # DPD: "de" between day and month and between month and year, and bare
        # "de" is preferred over "del".
        said = expand_dates("12 de marzo de 2026", "es")
        assert said == "doce de marzo de dos mil veintiséis"

    def test_portuguese_keeps_its_conjunction_in_the_year(self) -> None:
        # The pt/es split: Portuguese inserts "e" where Spanish has nothing.
        assert "dois mil e vinte e seis" in expand_dates("12 de março de 2026", "pt")

    def test_english_follows_the_written_order(self) -> None:
        # Both dialects say it this way when the day comes first, so no locale
        # flag is needed to choose between them.
        assert expand_dates("12 March 2026", "en").startswith("the twelfth of March")
        assert expand_dates("March 12, 2026", "en").startswith("March twelfth")


class TestWhatIsNotADate:
    """Reading a version number as a date is worse than leaving it alone."""

    @pytest.mark.parametrize("language", supported_languages())
    @pytest.mark.parametrize(
        "literal", ["1.2.3", "192.168.0.1", "13.13.2026", "32.01.2026", "0.5.2026"]
    )
    def test_left_alone(self, language: str, literal: str) -> None:
        assert literal in expand_dates(f"see {literal} now", language)

    def test_english_refuses_an_ambiguous_all_numeric_date(self) -> None:
        # 3/12 is March twelfth to half the English-speaking world and the third
        # of December to the other half. A listener recovers from hearing the
        # digits; they cannot recover from a confident wrong month.
        assert expand_dates("3/12/2026", "en") == "3/12/2026"
        # With a field over twelve the order is forced whatever the dialect.
        assert expand_dates("25/12/2026", "en") != "25/12/2026"

    def test_swedish_has_no_dotted_dates(self) -> None:
        # Swedish marks an ordinal with a colon (1:a), never a trailing period,
        # so `12.` in Swedish is a list number or a sentence end. Porting the
        # Danish/Norwegian rule across would make the voice say "tolfte" at the
        # end of sentences.
        assert expand_dates("12.3.2026", "sv") == "12.3.2026"
        assert "tolfte mars" in expand_dates("den 12 mars 2026", "sv")

    @pytest.mark.parametrize("language", supported_languages())
    def test_nothing_is_invented_where_there_is_no_date(self, language: str) -> None:
        assert expand_dates("no dates here at all", language) == "no dates here at all"


class TestTheFunnelRunsDatesFirst:
    """The ordering is the reason this pass exists."""

    def test_a_dotted_date_is_not_eaten_by_the_clock(self) -> None:
        from loudkit.frontend.polish import speech_text

        said = speech_text("Spotkanie 12.03.2026 o 14:30.", "pl")
        assert "dwunastego marca" in said
        # ...and the real time still reads.
        assert "czternaście trzydzieści" in said

    def test_every_implementation_reads_this_date(self) -> None:
        """The pass is no longer Python's alone, so there is no ported variant.

        This used to assert the opposite — that the default funnel left the date
        to the number pass — because Swift, Go, Rust and JS had no date reader
        and the fingerprint would otherwise have claimed a parity that did not
        exist. All four have one now, verified against 144 generated pairs
        apiece, so the gate came out and this asserts what all five do.
        """
        from loudkit.frontend.polish import speech_text

        for lang, expected in (("pl", "dwunastego marca"), ("de", "zwölfte März")):
            said = speech_text("Spotkanie 12.03.2026.", lang)
            assert expected in said, f"{lang}: {said}"

    def test_a_version_number_survives_the_whole_funnel(self) -> None:
        from loudkit.frontend.polish import speech_text

        assert "1.2.3" in speech_text("Wersja 1.2.3 i adres 192.168.0.1.", "pl")


class TestADecimalIsNotADate:
    """`3.5.` at the end of a sentence is a number, and used to be a date.

    The yearless `12.3.` that German, Danish, Finnish and Norwegian write has a
    closing period indistinguishable from a sentence's, so `Die Zahl ist 3.5.`
    came out as *dritte Mai* — in ten of the twelve languages. There is no
    evidence in the string to separate the two readings, so this module does
    what it does everywhere else when evidence runs out: nothing.
    """

    @pytest.mark.parametrize("language", supported_languages())
    def test_a_sentence_final_decimal_stays_a_number(self, language: str) -> None:
        from loudkit.frontend.polish import speech_text

        said = speech_text("Die Zahl ist 3.5.", language)
        for month in ("Mai", "maja", "mayo", "maj", "toukokuuta", "May"):
            assert month not in said, f"{language}: read a decimal as a date — {said!r}"

    @pytest.mark.parametrize("language", supported_languages())
    def test_a_yearless_numeric_date_is_left_alone(self, language: str) -> None:
        assert expand_dates("12.3.", language) == "12.3."

    @pytest.mark.parametrize("language", sorted(set(supported_languages()) - {"en", "sv"}))
    def test_a_dated_year_still_reads(self, language: str) -> None:
        # The guard must not have bought safety by refusing every date. English
        # and Swedish are excluded deliberately, not by oversight: English field
        # order is unresolvable and Swedish marks an ordinal with a colon rather
        # than a period, so neither has a dotted date to read.
        assert expand_dates("12.03.2026", language) != "12.03.2026"

    def test_a_month_name_is_evidence_enough_without_a_year(self) -> None:
        # A yearless date still reads when it is written with a month name,
        # because the name is the evidence the digits lacked.
        assert "zwölfte" in expand_dates("der 12. März", "de")


class TestObliqueReadsTheCurrentPass:
    """`is_oblique` examined the original text at offsets from the substituted
    one: after an ISO date expanded to a long spoken form, every later match's
    offset pointed into the wrong string, and German `am`/`den`/`vom` detection
    read the wrong preceding word.
    """

    def test_two_date_forms_in_one_passage(self) -> None:
        from loudkit.frontend.dates import expand_dates

        text = "Am 2026-03-04 und am 5.03.2026 passiert es."
        out = expand_dates(text, "de")
        # The second date follows "am" too; both must take the oblique -en.
        assert "vierten März" in out or "vierter März" in out
        assert "fünften März" in out
