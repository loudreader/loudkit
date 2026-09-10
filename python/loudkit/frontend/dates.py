"""Dates, said out loud, the pass that has to run before times and numbers.

See ``docs/design/text-funnel.md``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import cache
from typing import Any, NamedTuple

from .numbers import NumberGrammarError, cardinal
from .textconfig import grammar_languages

__all__ = [
    "expand_dates",
    "expand_ordinals",
    "month_name",
    "ordinal",
    "ordinal_day",
    "say_year",
    "supported_languages",
]


def _ordinals(block: dict[str, Any]) -> dict[str, Any]:
    """The ordinal tables, or empty ones for a language that writes no suffix."""
    return {
        "ordinal_suffixes": tuple(block.get("suffixes", ())),
        "ordinal_units": {int(k): v for k, v in block.get("units", {}).items() if v},
        "ordinal_teens": {int(k): v for k, v in block.get("teens", {}).items() if v},
        "ordinal_tens": {int(k): v for k, v in block.get("tens", {}).items() if v},
        "ordinal_joiner": block.get("tens_joiner", "-"),
    }


_MAX_YEAR = 2999
"""Above this a four-digit run is an identifier, not a year. Chosen rather than
computed: a fixed bound is checkable, and nothing this library reads is dated
past it."""

_MIN_YEAR = 1000
"""A three-digit year exists but a three-digit *anything* is far more often a
quantity, and there is no signal in the string to tell them apart."""

_YEAR = "[12][0-9]{3}"
"""What a year looks like, for the five patterns that match one.

The same range as :data:`_MIN_YEAR` to :data:`_MAX_YEAR`, written as a class
because that is what a regex can use. Restating it at each pattern is how the
bound and the patterns come to disagree; the two are pinned equal by a test."""

_DAYS_IN_MONTH = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
"""February is 29 rather than 28 on purpose: this is a plausibility bound, not a
calendar. Refusing 29 February in a non-leap year would mean rejecting a date a
human wrote deliberately, and the cost of accepting it is nothing."""


class _Dates(NamedTuple):
    """One language's date rules, read from ``numbers.json``."""

    day_form: str
    day_words: dict[int, str]
    day_words_oblique: dict[int, str]
    oblique_triggers: tuple[str, ...]
    day_one_word: str
    months: tuple[str, ...]
    day_month_infix: str
    month_year_infix: str
    day_first_prefix: str
    day_first_infix: str
    year_rule: str
    year_units: dict[int, str]
    year_teens: dict[int, str]
    year_tens: dict[int, str]
    year_two_thousand: str
    dotted_is_ambiguous: bool
    no_dotted_dates: bool
    ordinal_suffixes: tuple[str, ...]
    ordinal_units: dict[int, str]
    ordinal_teens: dict[int, str]
    ordinal_tens: dict[int, str]
    ordinal_joiner: str


@cache
def _rules() -> dict[str, _Dates]:
    """The `dates` and `ordinals` blocks of the shared grammar file.

    Built here rather than folded into the number grammars because that loader
    makes a dataclass with a fixed field list, and adding dates to it would put
    a text concern inside the numeral interpreter every port mirrors field for
    field. It is the same parse either way: `grammar_languages` reads the file
    once for all three passes.
    """
    out: dict[str, _Dates] = {}
    for lang, entry in grammar_languages().items():
        block = entry.get("dates")
        if not block:
            continue
        out[lang] = _Dates(
            day_form=block.get("day_form", "cardinal"),
            day_words={int(k): v for k, v in block.get("day_words", {}).items()},
            day_words_oblique={
                int(k): v for k, v in block.get("day_words_oblique", {}).items()
            },
            oblique_triggers=tuple(block.get("oblique_triggers", ())),
            day_one_word=block.get("day_one_word", ""),
            months=tuple(block.get("months", ())),
            day_month_infix=block.get("day_month_infix", ""),
            month_year_infix=block.get("month_year_infix", ""),
            day_first_prefix=block.get("day_first_prefix", ""),
            day_first_infix=block.get("day_first_infix", ""),
            year_rule=block.get("year_rule", "cardinal"),
            year_units={int(k): v for k, v in block.get("year_units", {}).items()},
            year_teens={int(k): v for k, v in block.get("year_teens", {}).items()},
            year_tens={int(k): v for k, v in block.get("year_tens", {}).items()},
            year_two_thousand=block.get("year_two_thousand", ""),
            dotted_is_ambiguous=bool(block.get("dotted_is_ambiguous", False)),
            no_dotted_dates=bool(block.get("no_dotted_dates", False)),
            **_ordinals(entry.get("ordinals", {})),
        )
    return out


def supported_languages() -> tuple[str, ...]:
    """The language ids that can speak a date, sorted."""
    return tuple(sorted(_rules()))


def month_name(month: int, language: str) -> str:
    """The month's name in the case a date puts it in.

    Polish returns the genitive (``marca``) and Finnish the partitive
    (``maaliskuuta``): those are the only forms that appear in a date, and the
    nominative in a date is a mistake a listener notices immediately.
    """
    rules = _rules().get(language)
    if not 1 <= month <= 12:
        raise NumberGrammarError(f"no month {month!r}: months are numbered 1-12 ({language!r})")
    if rules is None:
        raise NumberGrammarError(f"no date grammar for {language!r}")
    return rules.months[month - 1]


def ordinal_day(day: int, language: str, *, oblique: bool = False) -> str:
    """The day-of-month word, in whatever form this language's dates take.

    See ``docs/design/text-funnel.md``.
    """
    rules = _rules().get(language)
    if rules is None:
        raise NumberGrammarError(f"no date grammar for {language!r}")
    if not 1 <= day <= 31:
        raise NumberGrammarError(f"{day} is not a day of a month")
    if oblique and rules.day_words_oblique:
        return rules.day_words_oblique[day]
    if rules.day_words:
        return rules.day_words[day]
    # Cardinal languages: the whole point is that the day is just a number,
    # except where the first of the month is lexicalised.
    if day == 1 and rules.day_one_word:
        return rules.day_one_word
    return cardinal(day, language)


def say_year(year: int, language: str) -> str:
    """A year, read the way this language reads years.

    English and Norwegian split it; German, Dutch and Swedish group it in
    hundreds; Danish takes the long form, hundreds joined with *og*; Polish
    declines the last two digits. The other five say one plain cardinal.
    Spanish is the explicit case:
    the RAE writes that a year is read as its cardinal *"y no por bloques de dos
    cifras, como sucede en inglés"*, so ``2021`` is *dos mil veintiuno* and never
    *veinte veintiuno*.
    """
    rules = _rules().get(language)
    if rules is None:
        raise NumberGrammarError(f"no date grammar for {language!r}")
    if rules.year_rule == "pl_ordinal_genitive":
        # Dispatched here rather than from the table: it is the one reader that
        # needs the rules object, and the table holds plain int readers.
        return _year_pl(year, rules)
    reader = _YEAR_READERS.get(rules.year_rule)
    if reader is None:
        return cardinal(year, language)
    return reader(year)


def _year_pl(year: int, rules: _Dates) -> str:
    """Only the tens and units of a Polish year decline.

    PWN's worked example is *tysiąc dziewięćset dziewięćdziesiątego drugiego*:
    the thousands and hundreds keep their cardinal form and the ordinal genitive
    lands on the last two digits. Where those are zero the declension moves left,
    which is why 2000 has its own word, and why PWN rejects *dwutysięczny
    pierwszy* for 2001, which takes the ordinary shape instead.
    """
    if year == 2000 and rules.year_two_thousand:
        return rules.year_two_thousand
    head, rest = divmod(year, 100)
    lead = cardinal(head * 100, "pl") if head else ""
    if rest == 0:
        # Nothing left to decline; a bare cardinal is as close as this gets
        # without an ordinal for the hundreds, which no date in range needs.
        return lead
    if rest in rules.year_teens:
        tail = rules.year_teens[rest]
    else:
        tens, units = divmod(rest, 10)
        words = [rules.year_tens.get(tens * 10, ""), rules.year_units.get(units, "")]
        tail = " ".join(w for w in words if w)
    return f"{lead} {tail}".strip()


def _year_en(year: int) -> str:
    if year in {1000, 2000} or 2001 <= year <= 2009:
        return cardinal(year, "en")
    if 1000 < year < 2000 or year >= 2100:
        century, rest = divmod(year, 100)
        if rest == 0:
            return f"{cardinal(century, 'en')} hundred"
        if rest < 10:
            # "nineteen oh five", never "nineteen five", which no speaker says.
            return f"{cardinal(century, 'en')} oh {cardinal(rest, 'en')}"
        return f"{cardinal(century, 'en')} {cardinal(rest, 'en')}"
    if 2010 <= year <= 2099:
        return f"twenty {cardinal(year % 100, 'en')}"
    return cardinal(year, "en")


def _year_de(year: int) -> str:
    # 1100–1999 group in hundreds; from 2000 the thousands form is used, and
    # the GfdS explicitly rejects `zwanzighundert…`, German did not follow the
    # English "twenty-sixteen" shift.
    if 1100 <= year <= 1999:
        century, rest = divmod(year, 100)
        head = f"{cardinal(century, 'de')}hundert"
        return head if rest == 0 else f"{head}{cardinal(rest, 'de')}"
    return cardinal(year, "de")


def _year_nl(year: int) -> str:
    # Taalunie's citation form is the full `negentienhonderd…`; the shortened
    # `negentien tweeënnegentig` is a conversational reduction.
    if 1100 <= year <= 1999:
        century, rest = divmod(year, 100)
        head = f"{cardinal(century, 'nl')}honderd"
        return head if rest == 0 else f"{head}{cardinal(rest, 'nl')}"
    return cardinal(year, "nl")


def _year_sv(year: int) -> str:
    # Isof/Språkrådet has recommended the `tjugohundra…` series for decades;
    # `tvåtusen…` is common in usage and is accepted on input, not emitted.
    if 1100 <= year <= 2099:
        century, rest = divmod(year, 100)
        head = f"{cardinal(century, 'sv')}hundra"
        return head if rest == 0 else f"{head}{cardinal(rest, 'sv')}"
    return cardinal(year, "sv")


def _year_no(year: int) -> str:
    # Norwegian splits 1100–1999 and drops `hundre`: 1972 is `nittensyttito`.
    # Språkrådet's main recommendation from 2000 on is the `totusenog…` form.
    if 1100 <= year <= 1999:
        century, rest = divmod(year, 100)
        if rest == 0:
            return f"{cardinal(century, 'no')}hundre"
        return f"{cardinal(century, 'no')}{cardinal(rest, 'no')}"
    return cardinal(year, "no")


def _year_da(year: int) -> str:
    # Dansk Sprognævn: the long form is the one that works for every year, and
    # the short "telephone-number" form is explicitly poor for a century's first
    # decade. The long form is what this emits.
    if 1100 <= year <= 1999:
        century, rest = divmod(year, 100)
        head = f"{cardinal(century, 'da')} hundrede"
        return head if rest == 0 else f"{head} og {cardinal(rest, 'da')}"
    return cardinal(year, "da")


_YEAR_READERS: dict[str, Callable[[int], str]] = {
    "en_split": _year_en,
    "de_hundreds": _year_de,
    "nl_hundreds": _year_nl,
    "sv_hundreds": _year_sv,
    "no_split": _year_no,
    "da_long": _year_da,
}


def _valid(day: int, month: int, year: int | None) -> bool:
    if not 1 <= month <= 12:
        return False
    if not 1 <= day <= _DAYS_IN_MONTH[month - 1]:
        return False
    return year is None or _MIN_YEAR <= year <= _MAX_YEAR


def _spoken(day: int, month: int, year: int | None, language: str, *, oblique: bool) -> str:
    rules = _rules()[language]
    parts = [ordinal_day(day, language, oblique=oblique)]
    if rules.day_month_infix:
        parts.append(rules.day_month_infix)
    parts.append(month_name(month, language))
    if year is not None:
        if rules.month_year_infix:
            parts.append(rules.month_year_infix)
        parts.append(say_year(year, language))
    return " ".join(parts)


# A written date is bounded by a word boundary, and not by digits alone: a run
# that continues into a letter is an identifier, and `25/03/2026x` is no more a
# date than `x25/03/2026` is. `\w` and not `[A-Za-z0-9]`, so the boundary is the
# one `\b` already asserts at the end of the dotted form, in every script.
_BOUND_BEFORE = r"(?<![\w.,:/-])"

# `12.03.2026`, `12.3.2026`, and the yearless `12.3.` that German, Danish, Finnish and
# Norwegian write with a closing period.
_DOTTED = re.compile(rf"{_BOUND_BEFORE}([0-3]?[0-9])\.([01]?[0-9])\.({_YEAR})\b")
# `12/03/2026`. Day-first in every language here; English is handled separately
# because it is the one language where the field order is genuinely ambiguous.
_SLASHED = re.compile(rf"{_BOUND_BEFORE}([0-3]?[0-9])/([01]?[0-9])/({_YEAR})(?![\w/])")
# ISO. Unambiguous by definition, and the Swedish norm. The `T` before a clock
# time is taken with the date: it is a field separator and not a letter, and left
# behind it glues to the last word of the date.
_ISO = re.compile(
    rf"{_BOUND_BEFORE}({_YEAR})-([01][0-9])-([0-3][0-9])"
    r"(?:(T)(?=[0-2][0-9]:[0-5][0-9])|(?![\w-]))"
)


def _word_before(m: re.Match[str], word: str) -> bool:
    """Whether ``word`` is what stands immediately before the match.

    Reads the string the pass is substituting over, so the neighbour is the
    previous pass's output rather than the original text, and compares folded
    case because a sentence may open with the article.
    """
    before = m.string[: m.start()].rstrip()
    if not before:
        return False
    return before.rsplit(None, 1)[-1].lower().strip(",;:") == word.lower()


def _is_oblique(oblique_words: set[str], m: re.Match[str]) -> bool:
    """German only: `am`/`den`/`vom` before the day select the -en ending.

    Reads the word before the match out of ``m.string``, which is the string
    the pass is substituting over, so the context is the previous pass's
    output and not the original text.
    """
    if not oblique_words:
        return False
    before = m.string[: m.start()].rstrip()
    tail = before.rsplit(None, 1)[-1].lower().strip(",;:") if before else ""
    return tail in oblique_words


def expand_dates(text: str, language: str) -> str:
    """Every written date in ``text``, said the way ``language`` says it.

    Never raises and never invents: a run that fails the bounds check, or whose
    field order cannot be resolved, is returned exactly as it was written.
    """
    rules = _rules().get(language)
    if rules is None:
        return text

    oblique_words = {w.lower() for w in rules.oblique_triggers}

    def iso(m: re.Match[str]) -> str:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not _valid(day, month, year):
            return m.group(0)
        said = _spoken(day, month, year, language, oblique=_is_oblique(oblique_words, m))
        # The separator becomes the space that keeps the date and the time two
        # spoken units. A word for it would be a per-language fact this grammar
        # does not carry, and a plainer reading is not a wrong one.
        return f"{said} " if m.group(4) else said

    def dotted(m: re.Match[str]) -> str:
        if rules.no_dotted_dates or rules.dotted_is_ambiguous:
            # Swedish marks an ordinal with a colon (`1:a`), never a trailing
            # period, so `12.` there is a list number or a sentence end. English
            # writes dotted dates almost never, and when it does the field order
            # is as unresolvable as in the slashed form.
            return m.group(0)
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not _valid(day, month, year):
            return m.group(0)
        return _spoken(day, month, year, language, oblique=_is_oblique(oblique_words, m))

    def slashed(m: re.Match[str]) -> str:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if language == "en" and day <= 12:
            # `3/12/2026` is March twelfth to half the English-speaking world and
            # the third of December to the other half, and nothing in the string
            # says which. A listener recovers from hearing the digits; they
            # cannot recover from a confident wrong month. Past this branch the
            # day is over 12, so the order is forced whatever the dialect.
            return m.group(0)
        if not _valid(day, month, year):
            return m.group(0)
        return _spoken(day, month, year, language, oblique=_is_oblique(oblique_words, m))

    # Each pass reads the string the previous pass produced, which is the
    # string its own matches carry as `m.string`, so oblique detection reads
    # the right preceding word at the right offset.
    out = text
    for pattern, handler in ((_ISO, iso), (_DOTTED, dotted), (_SLASHED, slashed)):
        out = pattern.sub(handler, out)
    return _textual(out, language)


def _textual(text: str, language: str) -> str:
    """`12 marca 2026`, `12. März 2026`, `March 12, 2026`, a written month name
    beside a bare day.

    The month name is the disambiguator: with one present there is no field
    order to guess, so this runs for every language including English.
    """
    rules = _rules()[language]
    names = "|".join(re.escape(n) for n in rules.months)
    oblique_words = {w.lower() for w in rules.oblique_triggers}

    # Spanish and Portuguese speak a preposition between every part, so the
    # written form carries it too: "12 de marzo de 2026". Optional in the
    # pattern rather than in a second pattern, because a language either has the
    # infix everywhere or nowhere.
    infix = rf"(?:\s+{re.escape(rules.day_month_infix)})?" if rules.day_month_infix else ""
    yinfix = rf"(?:\s+{re.escape(rules.month_year_infix)})?" if rules.month_year_infix else ""
    day_first = re.compile(
        rf"(?<![\w])([0-3]?[0-9])\.?{infix}\s+({names})(?:{yinfix}\s+({_YEAR}))?(?!\w)",
        re.IGNORECASE,
    )

    def say_day_first(m: re.Match[str]) -> str:
        day = int(m.group(1))
        month = _month_index(m.group(2), rules)
        year = int(m.group(3)) if m.group(3) else None
        if month is None or not _valid(day, month, year):
            return m.group(0)
        spoken = _spoken(day, month, year, language, oblique=_is_oblique(oblique_words, m))
        if rules.day_first_prefix or rules.day_first_infix:
            # English written day-first reads "the twelfth of March": both
            # dialects say it that way when the day comes first, so no locale
            # flag is needed to choose.
            head = ordinal_day(day, language)
            rest = [month_name(month, language)]
            if year is not None:
                rest.append(say_year(year, language))
            joined = " ".join(rest)
            prefix = ""
            if rules.day_first_prefix and not _word_before(m, rules.day_first_prefix):
                # The sentence may already carry the article: "the 3 April
                # minutes" is a noun phrase whose determiner is written, and a
                # second one is a stammer.
                prefix = f"{rules.day_first_prefix} "
            infix = f" {rules.day_first_infix} " if rules.day_first_infix else " "
            return f"{prefix}{head}{infix}{joined}"
        return spoken

    out = day_first.sub(say_day_first, text)

    month_first = re.compile(
        rf"(?<![\w])({names})\s+([0-3]?[0-9])(?:(?:st|nd|rd|th)\b)?,?(?:\s+({_YEAR}))?(?!\w)",
        re.IGNORECASE,
    )

    def say_month_first(m: re.Match[str]) -> str:
        month = _month_index(m.group(1), rules)
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else None
        if month is None or not _valid(day, month, year):
            return m.group(0)
        parts = [month_name(month, language), ordinal_day(day, language)]
        if year is not None:
            parts.append(say_year(year, language))
        return " ".join(parts)

    if rules.day_first_infix:
        # Month-first is an English shape. Reading it in a language that never
        # writes it would be inventing a construction the language does not use.
        out = month_first.sub(say_month_first, out)
    return out


def _month_index(name: str, rules: _Dates) -> int | None:
    lowered = name.lower()
    for i, candidate in enumerate(rules.months):
        if candidate.lower() == lowered:
            return i + 1
    return None


def ordinal(value: int, language: str) -> str | None:
    """``value`` as a written-out ordinal, or ``None`` if this language has no table for
    it.

    See ``docs/design/text-funnel.md``.
    """
    rules = _rules().get(language)
    if rules is None or not rules.ordinal_units:
        return None
    if value < 0:
        return None
    head, rest = divmod(value, 100)
    tail = _two_digit_ordinal(rest, rules)
    if tail is None:
        return None
    if head == 0:
        return tail
    lead = cardinal(head * 100, language)
    return f"{lead} {tail}" if rest else lead


def _two_digit_ordinal(value: int, rules: _Dates) -> str | None:
    if value in rules.ordinal_teens:
        return rules.ordinal_teens[value]
    tens, units = divmod(value, 10)
    if units == 0:
        return rules.ordinal_tens.get(tens * 10)
    if tens == 0:
        return rules.ordinal_units.get(units)
    ten_word = cardinal(tens * 10, "en")
    unit_word = rules.ordinal_units.get(units)
    if unit_word is None:
        return None
    return f"{ten_word}{rules.ordinal_joiner}{unit_word}"


def expand_ordinals(text: str, language: str) -> str:
    """``1st`` and ``22nd`` as words.

    See ``docs/design/text-funnel.md``.
    """
    rules = _rules().get(language)
    if rules is None or not rules.ordinal_suffixes:
        return text
    pattern = re.compile(rf"\b([0-9]+)({'|'.join(rules.ordinal_suffixes)})\b", re.IGNORECASE)

    def say(m: re.Match[str]) -> str:
        said = ordinal(int(m.group(1)), language)
        return said if said is not None else m.group(0)

    return pattern.sub(say, text)
