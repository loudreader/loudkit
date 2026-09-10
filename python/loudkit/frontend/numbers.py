"""Numbers, said out loud, in the twelve languages the kit speaks.

See ``docs/design/text-funnel.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import Any

# Re-exported, not defined here: `NumberGrammarError` is part of this module's
# grammar contract and `loudkit.frontend.numbers.NumberGrammarError` is the name callers
# know, but it inherits `LoudkitError` and this module imports that, so the
# class has to be declared on the other side of the edge.
from ..errors import NumberGrammarError
from .textconfig import grammar_languages

__all__ = [
    "Grammar",
    "unit_word",
    "NumberGrammarError",
    "cardinal",
    "expand",
    "expand_abbreviations",
    "expand_roman_numerals",
    "expand_times",
    "scale_nouns",
    "scale_suffix_word",
    "supported_languages",
]

# ASCII digits only, explicitly: `\d` means Unicode in Python and Rust but ASCII in Go
# and JS, so a class written as `\d` matches a different set of digits in each port and
# the five funnels stop agreeing on the same text.
_DIGIT_RUN = re.compile(
    r"(?<![\w])(-(?=[0-9]))?([0-9]{1,3}(?: [0-9]{3})+(?! ?[0-9])|[0-9]+)"
    r"((?:[.,][0-9]+)*)(?![\w])"
)


_FOREIGN_DIGITS = {
    # Arabic-Indic and Eastern Arabic-Indic, folded to ASCII with their own
    # decimal and thousands separators.
    **{chr(0x0660 + n): str(n) for n in range(10)},
    **{chr(0x06F0 + n): str(n) for n in range(10)},
    "\u066a": "%",  # ARABIC PERCENT SIGN
}
"""Digit systems this funnel reads, mapped to the ASCII the rest of it matches.

See ``docs/design/text-funnel.md``.
"""

_FOREIGN_DIGIT_RUN = re.compile("[" + "".join(_FOREIGN_DIGITS) + "\u066b\u066c]")


def decimal_separator(language: str) -> str:
    """The mark this language writes between a whole number and its fraction.

    Public because the speech funnel needs it outside the number pass: a
    currency amount is the one place a dot between digits is known not to be a
    clock time, and the funnel has to say so while the currency symbol is still
    in hand. Defaults to ``"."`` for a language with no grammar, matching every
    other fallback here.
    """
    grammar = _grammars().get(language)
    return grammar.decimal_separator if grammar else "."


def fold_foreign_digits(text: str, language: str) -> str:
    """Foreign digit systems and their separators, as this language spells them.

    See ``docs/design/text-funnel.md``.
    """
    grammar = _grammars().get(language)
    decimal = grammar.decimal_separator if grammar else "."
    grouping = "," if decimal == "." else "."
    table = {**_FOREIGN_DIGITS, "\u066b": decimal, "\u066c": grouping}
    return _FOREIGN_DIGIT_RUN.sub(lambda m: table[m.group(0)], text)


_UNICODE_MINUS = re.compile("[\u2212\u2010](?=[0-9])")
"""U+2212 MINUS SIGN and U+2010 HYPHEN, where a digit follows: folded to ASCII.

See ``docs/design/text-funnel.md``.
"""

_PHONE_RUN = re.compile(r"\+[0-9][0-9 ]*[0-9]")
"""An E.164 telephone number: a plus, then digits, possibly grouped by spaces.

See ``docs/design/text-funnel.md``.
"""

_GROUP_DIGITS = 3
"""Digits in a thousands group. Every group after the first is exactly this."""

_END_OF_DAY_HOUR = 24
"""ISO 8601's 24:00. Admitted as an hour, and only with a zero minute."""

_MIN_E164_DIGITS = 8
"""Below this a plus-signed run is a delta, not a telephone number.

Eight because a country code plus a national number reaches it and a plausible
signed quantity does not: the largest thing anyone writes as "+N NNN NNN" is
seven digits, a million.
"""


@dataclass(frozen=True, slots=True)
class Scale:
    """One scale noun (thousand, million …) and how it behaves.

    The CLDR differential showed the behaviours are per *scale*, not per
    language: German writes ``eintausend`` solid but ``eine Million`` as two
    words with a feminine one; Danish says ``tusind og et`` but ``en million
    et``. A single language-level flag was measurably wrong in four languages.
    """

    value: int

    forms: tuple[str, ...]
    """One word: uninflected. Two: singular / plural (*Million / Millionen*).
    Three: the Slavic singular / few / many (*tysiąc / tysiące / tysięcy*)."""

    multiplier_agrees: bool
    """Whether the counted noun's gender reaches this scale's multiplier.
    Portuguese *duas mil*, *mil* is transparent, against Polish *dwa
    tysiące*, where the multiplier agrees with *tysiąc* itself."""

    one_word: str
    """What is said for a multiplier of exactly one. Empty means the bare scale
    word (*mille*, *tusind*); ``"~"`` means compose it like any other
    multiplier (*one thousand*); anything else is the literal word, German
    *eine* (Million is feminine), Italian *un*."""

    separate: bool
    """Whether the scale word takes spaces around it even in a language that
    writes numbers solid. *neunhundert…neunzig **Millionen** …* against
    *…tausend* glued."""

    link: str
    """What joins this scale's group to what follows, when no small-tail joiner
    fires. Finnish glues the multiplier to *tuhatta* but separates the groups
    with a space, *kaksituhatta kaksikymmentäkuusi*, and Swedish does the
    same at the thousand boundary while writing everything else solid. Empty
    means the language's ``word_join``."""

    multiplier_gender: str
    """A gender the multiplier is composed in, overriding the caller's.
    Swedish thousands take the common form, *tjugoentusen*, not
    *tjugoetttusen*, which would also break the rule against three identical
    consonants. Empty means the scale noun's own default (no gender)."""

    small_joiner: str
    """What joins this scale's group to a remainder under a hundred. English
    *one thousand **and** one*, Danish *tusind **og** et*. Includes its own
    behaviour nowhere: it is inserted as a word."""


@dataclass(frozen=True, slots=True)
class Grammar:
    """How one language builds a number word. The data half of this module.

    Every field is a property of the language, settled against its own reference
    grammar and checked by the conformance fixture. None of them is a tuning
    knob.
    """

    ones: tuple[str, ...]
    """Words for 0–9, in order."""

    teens: tuple[str, ...]
    """Words for 10–19, in order. Irregular in every language here."""

    tens: tuple[str, ...]
    """Words for 20, 30, … 90, eight entries, index 0 is twenty."""

    hundred: str
    """The word for a hundred when it stands after a multiplier."""

    hundreds: tuple[str, ...]
    """Words for 100, 200, … 900 when the language does not build them
    compositionally (Spanish *doscientos*, Portuguese *duzentos*). Empty when it
    does, in which case ``hundred`` is used with a multiplier."""

    scales: tuple[Scale, ...]
    """Scale nouns from largest to smallest, above a hundred, see :class:`Scale`."""

    units_before_tens: bool
    """*einundzwanzig*: the unit is spoken first, joined by ``unit_tens_joiner``."""

    unit_tens_joiner: str
    """What sits between unit and ten, **including its own spacing**.

    English ``"-"`` gives *twenty-one*, Spanish ``" y "`` gives *treinta y
    uno*, German ``"und"`` gives *einundzwanzig*. Carrying the spaces in the
    string rather than deriving them from a flag is what lets one line of code
    serve a hyphenating language, a spacing one and a compounding one.
    """

    tens_joiner_exceptions: tuple[tuple[int, str], ...]
    """Values whose joiner differs from the rule. French joins 21…71 with *et*
    and everything else with a hyphen; listing the exceptions is shorter and
    more checkable than a rule that predicts them."""

    hundred_joiner: str
    """What sits between the hundreds and the remainder (English *and*,
    Portuguese *e*), or empty when they simply abut."""

    scale_joiner: str
    """What sits between a scale group and what follows it. Portuguese needs
    *e* before a remainder under a hundred (*mil e oitocentos*) and nothing
    otherwise; the rule is in ``scale_joiner_below``."""

    scale_joiner_below: int
    """Insert ``scale_joiner`` only when the remainder is under this. Zero
    disables it, and a large value makes it unconditional."""

    one_before_hundred: bool
    """Whether *one* is spoken before *hundred* (English yes, Italian and Dutch
    no: *cento*, *honderd*)."""

    one_before_scale: bool
    """The same question for thousands and above. Italian *mille*, not
    *unomille*."""

    word_join: str
    """What separates the parts in writing, a space everywhere except German,
    Dutch and Danish, which write the whole number as one word. This is
    orthography, not phonology, but the model reads graphemes, so it matters."""

    combining_ones: tuple[tuple[int, str], ...]
    """Forms a unit takes when it is *part of* a larger number rather than
    standing alone.

    German is the clear case: *eins* answers "how many", but every compound uses
    *ein*, **ein**undzwanzig, **ein**hundert, **ein**tausend. This is position,
    not gender, and conflating the two would make the caller pass a gender to
    get a form that has nothing to do with gender.
    """

    scale_joiner_on_round_hundreds: bool
    """Portuguese inserts *e* after a scale when the remainder is a whole number
    of hundreds, *mil e oitocentos* (1800) but *mil oitocentos e noventa e
    dois* (1892). The rule is from Cunha & Cintra and is not derivable from the
    magnitude alone, which is why it is a field rather than a threshold."""

    exceptions: tuple[tuple[int, str], ...]
    """Values whose form is simply listed: Spanish *veintiuno*, Italian
    *ventotto*, French *quatre-vingts*. Checked before anything is composed."""

    minus_word: str
    """The language's own word for a negative, taken from CLDR rather than left
    English: *menos*, *moins*, *meno*, *min*. It always joins with a space, even
    in languages that write numbers solid (German *minus eins*, not
    *minuseins*)."""

    gender_scopes: tuple[tuple[int, str], ...]
    """Where each value's gender agreement applies. Absent means everywhere.

    See ``docs/design/text-funnel.md``.
    """

    hundreds_gendered: tuple[tuple[str, tuple[str, ...]], ...]
    """Gendered variants of the explicit hundreds table. Spanish and Portuguese
    inflect the whole series, *doscientas*, *duzentas*, not just the unit."""

    hundred_plural_final: str
    """French *deux cents* but *deux cent un*: the multiplied hundred takes a
    plural mark only when nothing follows it. Empty for everyone else."""

    scale_large_joiner: str
    """What joins a scale group to a remainder of a hundred or more. English
    reads long numbers with a breath, *nine hundred thousand, nine hundred* -
    and the comma is that breath in graphemes. Empty means ``word_join``."""

    decimal_separator: str
    """Which mark separates the whole part from the fraction. Eleven of the
    twelve use a comma; English is the exception, and a ``3.5`` read in a comma
    locale is *thirty-five*."""

    decimal_word: str
    """What that mark is called out loud, *point*, *przecinek*, *Komma*."""

    time_infix: str
    """The spoken word between clock hour and minute, where the language has
    one, German *vierzehn Uhr dreißig*. Empty elsewhere: Kotus reads 14.30 as
    *neljätoista kolmekymmentä* and Isof's 16.31 is *sexton trettioett*."""

    abbreviations: tuple[tuple[str, str], ...]
    """Expandable abbreviations, from each language authority's own list, and
    only the unambiguous ones. Swedish *s.k.* collapses three inflections,
    Finnish *v.* has four readings, *mm.* collides with millimetres: the traps
    stay out on purpose, because a wrong expansion is worse than a spelled
    abbreviation."""

    unit_words: tuple[tuple[str, str], ...]
    """Per-language wording for currency and measure symbols: ``$`` is
    *dólares* to a Spanish render and *Dollar* to a German one. Per language and
    not per pair of languages: one (en, pl) table with an ``en`` fallback reads
    English to the other ten."""

    genders: tuple[tuple[str, tuple[tuple[int, str], ...]], ...]
    """Per-gender overrides for the values that agree. Polish *dwa / dwie*,
    Spanish *uno / una*, German *ein / eine*. Keyed by a gender name the caller
    supplies; absent means the language does not inflect that value."""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Grammar:
        """Build from the JSON, failing loudly on a malformed entry.

        Parsed strictly rather than with ``.get`` defaults: a grammar missing a
        field is a grammar that silently produces a wrong word for some range
        outside the fixture's coverage, and a wrong word said out loud is
        harder to find than a refusal at load time.
        """

        def need(key: str, kind: type) -> Any:
            if key not in raw:
                raise NumberGrammarError(f"number grammar is missing {key!r}")
            value = raw[key]
            if not isinstance(value, kind):
                raise NumberGrammarError(
                    f"number grammar field {key!r} must be {kind.__name__}, got {value!r}"
                )
            return value

        def scale_need(entry: Any, key: str, index: int) -> Any:
            # `entry[key]` raised KeyError here, which is not the refusal this
            # loader promises and does not say which scale is short.
            if not isinstance(entry, dict) or key not in entry:
                raise NumberGrammarError(
                    f"number grammar scale {index} is missing {key!r}: {entry!r}"
                )
            return entry[key]

        scales = tuple(
            Scale(
                value=int(scale_need(entry, "value", i)),
                forms=tuple(str(f) for f in scale_need(entry, "forms", i)),
                multiplier_agrees=bool(entry.get("multiplier_agrees", False)),
                one_word=str(entry.get("one", "~")),
                separate=bool(entry.get("separate", False)),
                link=str(entry.get("link", "")),
                multiplier_gender=str(entry.get("multiplier_gender", "")),
                small_joiner=str(entry.get("small_joiner", "")),
            )
            for i, entry in enumerate(need("scales", list))
        )
        genders = tuple(
            (
                str(name),
                tuple((int(k), str(v)) for k, v in sorted(forms.items(), key=_as_int)),
            )
            for name, forms in sorted(raw.get("genders", {}).items())
        )
        # The digit tables are indexed, never searched, so a short one is an
        # IndexError from inside a sentence rather than a refusal at load. The
        # counts are what the ranges need: 0-9, 10-19, 20-90 by ten, and either
        # no hundreds column or 100-900 by hundred.
        for key, sizes in (("ones", (10,)), ("teens", (10,)), ("tens", (8,))):
            if len(need(key, list)) not in sizes:
                raise NumberGrammarError(
                    f"number grammar {key!r} must have {sizes[0]} entries, "
                    f"got {len(need(key, list))}"
                )
        if len(raw.get("hundreds", [])) not in (0, 9):
            raise NumberGrammarError(
                f"number grammar 'hundreds' must have 0 or 9 entries, "
                f"got {len(raw.get('hundreds', []))}"
            )
        return cls(
            ones=tuple(str(w) for w in need("ones", list)),
            teens=tuple(str(w) for w in need("teens", list)),
            tens=tuple(str(w) for w in need("tens", list)),
            hundred=str(need("hundred", str)),
            hundreds=tuple(str(w) for w in raw.get("hundreds", [])),
            scales=scales,
            units_before_tens=bool(need("units_before_tens", bool)),
            unit_tens_joiner=str(raw.get("unit_tens_joiner", "")),
            tens_joiner_exceptions=tuple(
                (int(k), str(v))
                for k, v in sorted(raw.get("tens_joiner_exceptions", {}).items(), key=_as_int)
            ),
            hundred_joiner=str(raw.get("hundred_joiner", "")),
            scale_joiner=str(raw.get("scale_joiner", "")),
            scale_joiner_below=int(raw.get("scale_joiner_below", 0)),
            combining_ones=tuple(
                (int(k), str(v))
                for k, v in sorted(raw.get("combining_ones", {}).items(), key=_as_int)
            ),
            scale_joiner_on_round_hundreds=bool(
                raw.get("scale_joiner_on_round_hundreds", False)
            ),
            one_before_hundred=bool(need("one_before_hundred", bool)),
            one_before_scale=bool(need("one_before_scale", bool)),
            word_join=str(need("word_join", str)),
            exceptions=tuple(
                (int(k), str(v))
                for k, v in sorted(raw.get("exceptions", {}).items(), key=_as_int)
            ),
            minus_word=str(need("minus_word", str)),
            gender_scopes=tuple(
                (int(k), str(v))
                for k, v in sorted(raw.get("gender_scopes", {}).items(), key=_as_int)
            ),
            hundreds_gendered=tuple(
                (str(name), tuple(str(w) for w in forms))
                for name, forms in sorted(raw.get("hundreds_gendered", {}).items())
            ),
            hundred_plural_final=str(raw.get("hundred_plural_final", "")),
            scale_large_joiner=str(raw.get("scale_large_joiner", "")),
            decimal_separator=str(need("decimal_separator", str)),
            decimal_word=str(need("decimal_word", str)),
            time_infix=str(raw.get("time_infix", "")),
            abbreviations=tuple(
                sorted(
                    ((str(k), str(v)) for k, v in raw.get("abbreviations", {}).items()),
                    key=lambda kv: -len(kv[0]),
                )
            ),
            unit_words=tuple(
                sorted((str(k), str(v)) for k, v in raw.get("unit_words", {}).items())
            ),
            genders=genders,
        )

    def listed(self, value: int) -> str | None:
        """An outright-listed form for ``value``, or ``None``."""
        for at, word in self.exceptions:
            if at == value:
                return word
        return None

    def combining(self, value: int) -> str | None:
        """The form ``value`` takes inside a larger number, or ``None``."""
        for at, word in self.combining_ones:
            if at == value:
                return word
        return None

    def gendered(
        self, value: int, gender: str | None, *, position: str = "standalone"
    ) -> str | None:
        """The form ``value`` takes in ``gender``, or ``None`` if it does not inflect,
        which is the common case, and why this returns an option rather than a default.

        See ``docs/design/text-funnel.md``.
        """
        if gender is None:
            return None
        scope = "always"
        for at, name in self.gender_scopes:
            if at == value:
                scope = name
                break
        if scope == "standalone" and position != "standalone":
            return None
        if scope == "outside_tens" and position == "tens_pair":
            return None
        for name, forms in self.genders:
            if name != gender:
                continue
            for at, word in forms:
                if at == value:
                    return word
        return None


def _as_int(item: tuple[str, Any]) -> int:
    """Sort key for a JSON object whose keys are numbers written as strings.

    Sorted numerically rather than lexically so the tuple order is the order a
    reader expects, and so two ports building the same grammar from the same
    file agree on it.
    """
    return int(item[0])


@cache
def _grammars() -> dict[str, Grammar]:
    """The shipped grammar table, parsed once.

    Read from a data file rather than written as literals so that five
    implementations share one source of truth. A rule that lives in code is
    twelve languages times five ports of opportunity to drift.
    """
    return {lang: Grammar.from_dict(entry) for lang, entry in grammar_languages().items()}


def supported_languages() -> tuple[str, ...]:
    """Language ids this module can verbalize, sorted."""
    return tuple(sorted(_grammars()))


def cardinal(value: int, language: str, *, gender: str | None = None) -> str:
    """``value`` as words.

    See ``docs/design/text-funnel.md``.
    """
    grammars = _grammars()
    if language not in grammars:
        raise NumberGrammarError(
            f"no number grammar for {language!r}; have {', '.join(supported_languages())}"
        )
    grammar = grammars[language]
    # Past the largest scale the composition still *runs*, it stacks scales and
    # says "a million milliards", but that is not what any of these languages
    # calls the number, and a wrong word is worse than a refusal. A value this
    # large in running text is almost always an identifier rather than a
    # quantity, and reading it as one is a different decision the caller owns.
    ceiling = grammar.scales[0].value * 1000 if grammar.scales else 1000
    if abs(value) >= ceiling:
        raise NumberGrammarError(
            f"{value} is past the largest scale {language!r} has a word for "
            f"({ceiling:,}); read it digit by digit or split it"
        )
    if value < 0:
        # Always a spaced word, even in solid-writing languages: *minus eins*.
        return f"{grammar.minus_word} {cardinal(-value, language, gender=gender)}"
    # Standalone agreement applies to the whole number only: Polish *jedna*
    # alone, but *sto jeden*, the trailing 1 of a larger number is compound
    # context even though it ends the number.
    standalone = grammar.gendered(value, gender, position="standalone")
    if standalone is not None:
        return standalone
    return _compose(value, grammar, gender)


def _compose(value: int, g: Grammar, gender: str | None, as_multiplier: bool = False) -> str:
    """The whole number, largest scale first."""
    listed = g.listed(value)
    if listed is not None:
        return listed
    if value < 100:
        return _below_hundred(value, g, gender, as_multiplier=as_multiplier)

    for scale in g.scales:
        if value >= scale.value:
            return _scale_group(value, scale, g, gender)
    return _hundreds_group(value, g, gender)


def _scale_group(value: int, scale: Scale, g: Grammar, gender: str | None) -> str:
    """One scale and everything under it: ``2_400`` as "two thousand four hundred"."""
    count, rest = divmod(value, scale.value)
    join = " " if scale.separate else g.word_join
    link_default = scale.link or (" " if scale.separate else g.word_join)

    if count == 1 and scale.one_word != "~":
        head = (
            _scale_word(1, scale.forms)
            if not scale.one_word
            else f"{scale.one_word}{join}{_scale_word(1, scale.forms)}"
        )
    else:
        # Whether the counted noun's gender reaches the multiplier is a fact
        # about the scale noun: Portuguese *duas mil* (mil is transparent),
        # Polish *dwa tysiące* (tysiąc agrees with itself).
        if scale.multiplier_gender:
            multiplier_gender: str | None = scale.multiplier_gender
        elif scale.multiplier_agrees:
            multiplier_gender = gender
        else:
            multiplier_gender = None
        multiplier = _compose(count, g, multiplier_gender, as_multiplier=True)
        head = f"{multiplier}{join}{_scale_word(count, scale.forms)}"

    if not rest:
        return head

    # Which joiner reaches the remainder is a fact about the remainder's size:
    # a small tail gets the language's spoken link (*and one*, *og et*, *e um*),
    # a large one gets the long-number breath (English's comma) or nothing.
    round_hundreds = g.scale_joiner_on_round_hundreds and rest >= 100 and rest % 100 == 0
    if scale.small_joiner and (rest < 100 or round_hundreds):
        # Spaced either way: `join` is a space or empty, because every
        # grammar's `word_join` is one of those two, and both spellings of the
        # link came out the same.
        link = f" {scale.small_joiner} "
    elif rest >= 100 and count >= 100 and g.scale_large_joiner:
        link = g.scale_large_joiner
    else:
        link = link_default
    return f"{head}{link}{_compose(rest, g, gender)}"


def _scale_word(count: int, forms: tuple[str, ...]) -> str:
    """The scale noun in the form ``count`` of them takes.

    One form means the language does not inflect it. Three means the Slavic
    pattern: singular for exactly one, "few" for 2–4 outside the teens, "many"
    otherwise. Polish *pięć tysięcy* but *dwadzieścia dwa tysiące*, the rule
    reads the last two digits, not the whole number, which is why 12 and 112
    both take the "many" form while 22 does not.
    """
    if len(forms) == 1 or count == 1:
        return forms[0]
    if len(forms) == 2:  # singular / plural: Million / Millionen
        return forms[1]
    last_two, last = count % 100, count % 10
    if 2 <= last <= 4 and not 12 <= last_two <= 14:
        return forms[1]
    return forms[2]


def _hundreds_group(value: int, g: Grammar, gender: str | None) -> str:
    """100–999."""
    count, rest = divmod(value, 100)
    parts: list[str] = []
    hundreds = g.hundreds
    if gender is not None:
        for name, forms in g.hundreds_gendered:
            if name == gender:
                hundreds = forms
                break
    if hundreds:
        parts.append(hundreds[count - 1])
    elif count == 1 and not g.one_before_hundred:
        parts.append(g.hundred)
    else:
        parts.append(_compose(count, g, None, as_multiplier=True))
        # French *deux cents* / *deux cent un*: the plural mark appears only
        # when the multiplied hundred ends the number.
        if count > 1 and not rest and g.hundred_plural_final:
            parts.append(g.hundred_plural_final)
        else:
            parts.append(g.hundred)
    if rest:
        if g.hundred_joiner:
            parts.append(g.hundred_joiner)
        # The remainder ends the number, so it is not a multiplier.
        parts.append(_below_hundred(rest, g, gender))
    return g.word_join.join(p for p in parts if p)


def _unit_word(value: int, g: Grammar, gender: str | None, as_multiplier: bool) -> str:
    """A single digit, in the most specific form that applies.

    Agreement first (it is a property of the sentence and outranks everything),
    then the combining form if this digit multiplies something, then the
    citation form.
    """
    agreed = g.gendered(value, gender, position="tens_pair" if as_multiplier else "tail")
    if agreed is not None:
        return agreed
    if as_multiplier:
        combining = g.combining(value)
        if combining is not None:
            return combining
    return g.ones[value]


def _below_hundred(
    value: int, g: Grammar, gender: str | None, as_multiplier: bool = False
) -> str:
    """0–99, where all the interesting variation lives.

    ``as_multiplier`` says this group multiplies a hundred or a scale rather
    than ending the number, which is what selects German's combining *ein* over
    its standalone *eins*: *ein*hundert**eins**, both forms in one word.
    """
    fixed = g.gendered(value, gender, position="tail") or g.listed(value)
    if fixed is not None:
        return fixed
    if value < 10:
        return _unit_word(value, g, gender, as_multiplier)
    if value < 20:
        return g.teens[value - 10]

    ten, unit = divmod(value, 10)
    ten_word = g.gendered(ten * 10, gender, position="tail") or g.tens[ten - 2]
    if unit == 0:
        return ten_word

    # A unit inside a tens pair is always in composition, whatever the group
    # itself is doing: German's *ein*undzwanzig holds even when the pair ends
    # the number.
    unit_word = _unit_word(unit, g, gender, as_multiplier=True)
    joiner = g.unit_tens_joiner
    for at, override in g.tens_joiner_exceptions:
        if at == value:
            joiner = override
            break

    # The joiner carries its own spacing, so both orders are one concatenation.
    if g.units_before_tens:
        return f"{unit_word}{joiner}{ten_word}"
    return f"{ten_word}{joiner}{unit_word}"


def expand(text: str, language: str, *, gender: str | None = None) -> str:
    """Every run of digits in ``text``, said as words.

    See ``docs/design/text-funnel.md``.
    """
    grammar = _grammars().get(language)
    if grammar is None:
        return text

    def say(match: re.Match[str]) -> str:
        # Normalised once, here, so everything downstream sees one shape: a sign kept
        # apart from the digits, and thousands spaces gone.
        if (
            _glued_to_a_word(match.string, match.start())
            or _glued_forward(match.string, match.end())
            or _truncated_by_a_fraction(match.string, match.end())
        ):
            return match.group(0)
        sign, whole, fraction = match.group(1) or "", match.group(2), match.group(3)
        literal = whole.replace(" ", "") + fraction
        if not _is_number(literal, grammar):
            return match.group(0)
        said = _say_number(literal, grammar, language, gender)
        return f"{grammar.minus_word} {said}" if sign else said

    def say_phone(match: re.Match[str]) -> str:
        # The same two guards the digit run answers to, for the same reason: a
        # run inside a word is part of an identifier, and `+12345678abc` is not
        # a telephone number in any country.
        if _glued_to_a_word(match.string, match.start()) or _glued_forward(
            match.string, match.end()
        ):
            return match.group(0)
        digits = "".join(ch for ch in match.group(0) if _is_ascii_digit(ch))
        if len(digits) < _MIN_E164_DIGITS:
            # Left exactly as written, plus and all, for `_DIGIT_RUN` to read as
            # the quantity it is.
            return match.group(0)
        return " ".join(_digit_by_digit(digits, language, gender))

    # The sign is folded before anything looks for one: `_DIGIT_RUN` matches
    # ASCII `-`, and a typographic minus reached the punctuation pass instead
    # and became a space.
    return _DIGIT_RUN.sub(say, _PHONE_RUN.sub(say_phone, _UNICODE_MINUS.sub("-", text)))


def _is_ascii_digit(c: str) -> bool:
    """One ASCII digit, the same class :data:`_DIGIT_RUN` matches.

    ``str.isdigit`` is true of ``²`` and of every Unicode decimal digit, so the
    walks answered questions about characters the pattern cannot match. Go,
    Rust and JS all test ASCII here; this is the reference joining them.
    """
    return "0" <= c <= "9"


def _starts_a_group(text: str, i: int) -> bool:
    """Whether three digits start at ``i``.

    See ``docs/design/text-funnel.md``.
    """
    group = text[i : i + _GROUP_DIGITS]
    return len(group) == _GROUP_DIGITS and all(_is_ascii_digit(c) for c in group)


def _continues_a_group(text: str, i: int) -> bool:
    """...and no fourth digit behind them.

    See ``docs/design/text-funnel.md``.
    """
    after = i + _GROUP_DIGITS
    return _starts_a_group(text, i) and (after >= len(text) or not _is_ascii_digit(text[after]))


def _glued_to_a_word(text: str, start: int) -> bool:
    """Whether the digit run at ``start`` sits inside a token containing a letter.

    See ``docs/design/text-funnel.md``.
    """
    i = start
    # `,` is in the walk for the same reason `.` is: it is a numeric separator, and the
    # fraction group already treats the two as one class.
    while i > 0:
        previous = text[i - 1]
        if previous.isalnum() or previous in "_.,-+":
            i -= 1
        elif (
            previous == " "
            and i >= 2  # a digit, then the space
            and _is_ascii_digit(text[i - 2])
            and _continues_a_group(text, i)
        ):
            # A thousands space, judged by the group the walk is stepping *out of*
            # rather than the one behind it.
            i -= 1
        else:
            return False
        if text[i].isalpha():
            return True
    return False


def _glued_forward(text: str, end: int) -> bool:
    """Whether the token continues past the match into a letter.

    See ``docs/design/text-funnel.md``.
    """
    i = end
    while i < len(text):
        c = text[i]
        if c.isalnum() or c in "_.,-+":
            if c.isalpha():
                return True
            i += 1
        elif c == " " and _is_ascii_digit(text[i - 1]) and _starts_a_group(text, i + 1):
            # Three digits and not fewer, so the walk stops where the run stops:
            # `1000 5.1e+3` keeps its `1000` rather than crossing into an
            # exponent two tokens away, four of the fuzzer's eight Go
            # divergences were that, with Go right and this side wrong, and the
            # `5` of `R2 5 iOS` is its own number.
            i += 1
        else:
            return False
    return False


def _truncated_by_a_fraction(text: str, end: int) -> bool:
    """Whether a decimal point with digits behind it follows the match.

    See ``docs/design/text-funnel.md``.
    """
    return end + 1 < len(text) and text[end] in ".," and _is_ascii_digit(text[end + 1])


def _is_number(literal: str, g: Grammar) -> bool:
    """Whether a digit run is a *quantity*, or just digits with dots in them.

    See ``docs/design/text-funnel.md``.
    """
    grouping = "," if g.decimal_separator == "." else "."
    whole, _, fraction = literal.partition(g.decimal_separator)
    # A second decimal mark in what should be the fraction is not a quantity.
    # `partition` splits once, so "1.2.3" leaves "2.3" here, and `int` must
    # never see it.
    if grouping in fraction or g.decimal_separator in fraction:
        return False
    segments = whole.split(grouping)
    if len(segments) == 1:
        return True
    if all(len(seg) == 3 for seg in segments[1:]) and 1 <= len(segments[0]) <= 3:
        return True
    # Two segments and no fraction is the "2.5 GB" shape: the mark that is not
    # this language's decimal separator, used as one anyway. Three or more is
    # not a number in any convention.
    return len(segments) == 2 and not fraction


def _say_number(literal: str, g: Grammar, language: str, gender: str | None) -> str:
    """One digit run, with its separators resolved.

    The mark that is not the language's decimal separator is only a grouping
    mark when it groups: every following segment exactly three digits. A Polish
    "1.000" is a thousand; a Polish "2.5" is not twenty-five, the dot there is
    a de-facto decimal, and 2.5 GB read as 25 GB is a changed meaning, which is
    the one error class this module must never commit.
    """
    grouping = "," if g.decimal_separator == "." else "."
    whole, _, fraction = literal.partition(g.decimal_separator)
    segments = whole.split(grouping)
    if len(segments) > 1:
        if all(len(seg) == 3 for seg in segments[1:]):
            whole = "".join(segments)
        elif not fraction and len(segments) == 2:
            whole, fraction = segments[0], segments[1]
        else:
            whole = "".join(segments)
    fraction = fraction.replace(grouping, "")

    parts = [_say_integer(whole, language, gender)]
    if fraction:
        parts.append(g.decimal_word)
        # The fractional part is read digit by digit, "point four nine", not
        # "point forty-nine", because that is how a decimal is said, and
        # because leading zeros carry meaning there that a cardinal would eat.
        parts.extend(_digit_by_digit(fraction, language, gender))
    return " ".join(p for p in parts if p)


def _say_integer(digits: str, language: str, gender: str | None) -> str:
    # Leading zeros mean a code, not a quantity: 0042 is zero zero four two,
    # never forty-two, int() would silently eat the zeros that carry meaning.
    if len(digits) > 1 and digits.startswith("0"):
        return " ".join(_digit_by_digit(digits, language, gender))
    try:
        return cardinal(int(digits), language, gender=gender)
    except NumberGrammarError:
        return " ".join(_digit_by_digit(digits, language, gender))


def _digit_by_digit(digits: str, language: str, gender: str | None) -> list[str]:
    return [cardinal(int(ch), language, gender=gender) for ch in digits]


_ROMAN_RUN = re.compile(r"(?<![0-9A-Za-z])([IVX]{2,})(?![0-9A-Za-z])")
"""A Roman numeral written with I, V and X, and nothing else.

L, C, D and M are left out, and that is the whole rule rather than an
optimisation of it. Every two-letter initialism that is also a valid Roman
numeral needs one of them -- CD, CV, DC, MC, MD, XL, CM -- and so does the only
common English word that is one, MIX. What remains is 2 to 39, which is where
chapter, act, volume, war and regnal numbers live.
"""

_ROMAN_CANONICAL = re.compile(r"(X{0,3})(IX|IV|V?I{0,3})")
"""The only spellings 2 to 39 has. Matched whole, so `IIX` and `VV` are refused.

They are letters that happen to be in the alphabet rather than numbers, and a
token this refuses is a token the acronym pass still sees.
"""

_TEN = 10


def _roman_value(numeral: str) -> int | None:
    """``numeral`` as a number, or ``None`` when it is not one."""
    matched = _ROMAN_CANONICAL.fullmatch(numeral)
    if matched is None:
        return None
    units = {"IX": 9, "IV": 4}.get(matched.group(2))
    if units is None:
        tail = matched.group(2)
        units = (5 if tail.startswith("V") else 0) + tail.count("I")
    return len(matched.group(1)) * _TEN + units


def expand_roman_numerals(text: str, language: str) -> str:
    """``Chapter IV`` and ``World War II``, said as numbers.

    See ``docs/design/text-funnel.md``.
    """
    if language not in _grammars():
        return text

    def say(match: re.Match[str]) -> str:
        value = _roman_value(match.group(1))
        if value is None:
            return match.group(0)
        return cardinal(value, language)

    return _ROMAN_RUN.sub(say, text)


_SCALE_SUFFIXES: tuple[tuple[str, int], ...] = (
    ("bn", 1_000_000_000),
    ("tn", 1_000_000_000_000),
    ("k", 1_000),
    ("m", 1_000_000),
    ("b", 1_000_000_000),
    ("t", 1_000_000_000_000),
)
"""The letters a price abbreviates its magnitude with, longest first.

Only beside a currency mark, which is what makes them unambiguous: a bare `5m`
is five metres as readily as five million, and `20k` is a race distance. `$5m`
is a sum of money in every convention that writes it.
"""

SCALE_SUFFIX_PATTERN = "|".join(
    f"[{s[0]}{s[0].upper()}]" + "".join(f"[{c}{c.upper()}]" for c in s[1:])
    for s, _ in _SCALE_SUFFIXES
)
"""The suffixes above as one alternation, either case, for the funnel's pattern."""


def scale_suffix_word(suffix: str, count: int, language: str) -> str | None:
    """The scale noun ``suffix`` abbreviates, in the form ``count`` of them takes.

    ``None`` where the language has no noun for that magnitude, which leaves the
    letter written rather than guessing at a word for it.
    """
    grammar = _grammars().get(language)
    if grammar is None:
        return None
    for spelling, value in _SCALE_SUFFIXES:
        if spelling != suffix.lower():
            continue
        for scale in grammar.scales:
            if scale.value == value:
                return _scale_word(count, scale.forms)
    return None


def scale_nouns(language: str) -> tuple[str, ...]:
    """Every form of every scale noun this language has, longest first.

    Longest first because they go into an alternation, where a shorter form that
    prefixes a longer one would match first and leave the rest of the word behind.
    """
    grammar = _grammars().get(language)
    if grammar is None:
        return ()
    forms = {form for scale in grammar.scales for form in scale.forms if form}
    return tuple(sorted(forms, key=lambda word: (-len(word), word)))


def unit_word(symbol: str, language: str) -> str | None:
    """The word ``symbol`` takes in ``language``, or ``None`` if unknown.

    The seam the funnel's symbol pass uses: it keeps owning *where* the word
    goes (a currency mark is written before its amount and spoken after it) and
    asks here only *which* word, because the which is a per-language fact that
    lives with the other per-language facts.
    """
    grammar = _grammars().get(language)
    if grammar is None:
        return None
    for at, word in grammar.unit_words:
        if at == symbol:
            return word
    return None


_TIME_RUN = re.compile(
    r"(?<![0-9.,:])([01]?[0-9]|2[0-4]):([0-5][0-9])(?::([0-5][0-9]))?(?![.,:]?[0-9])"
)
"""``14:30`` and ``14:30:45``, and nothing that merely contains them.

See ``docs/design/text-funnel.md``.
"""

_DOTTED_TIME_RUN = re.compile(r"(?<![0-9.,:])([01]?[0-9]|2[0-4])\.([0-5][0-9])(?![.,:]?[0-9])")
"""``14.30``, which is a clock time in some of these languages and a decimal in others, so
this pattern is only applied where the language says it is a time.

See ``docs/design/text-funnel.md``.
"""


def _ascii_letter_at(text: str, at: int) -> bool:
    """Whether an ASCII letter stands at ``at``.

    The class the written infix is already guarded against, so the two rules
    that decide where a spoken time ends answer to one alphabet in all five
    implementations rather than to five spellings of ``\\w``.

    See ``docs/design/text-funnel.md``.
    """
    if at >= len(text):
        return False
    return "A" <= text[at] <= "Z" or "a" <= text[at] <= "z"


@cache
def _time_patterns(time_infix: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """The two clock-time patterns, extended to consume a written infix word.

    The whitespace between the time and the infix may be absent: a word is the
    same word whether or not a space was typed in front of it, and an infix the
    match does not take is an infix the reading says twice.

    See ``docs/design/text-funnel.md``.
    """
    suffix = f"(?:[ \\t]*{re.escape(time_infix)}(?![0-9A-Za-z]))?"
    return (
        re.compile(_TIME_RUN.pattern + suffix),
        re.compile(_DOTTED_TIME_RUN.pattern + suffix),
    )


def expand_times(text: str, language: str) -> str:
    """Clock times as words: ``14:30`` and ``14.30`` become their reading.

    The reading is words, and a word is separated from what follows it. A
    meridiem written against the digits is the ordinary case: ``3:45pm`` reads
    "three forty-five pm", the same as ``3:45 pm``, because a space in the
    source is not what makes them two words.

    See ``docs/design/text-funnel.md``.
    """
    grammar = _grammars().get(language)
    if grammar is None:
        return text
    time_run, dotted_time_run = (
        _time_patterns(grammar.time_infix)
        if grammar.time_infix
        else (_TIME_RUN, _DOTTED_TIME_RUN)
    )

    def say(match: re.Match[str], *, has_seconds: bool = True) -> str:
        hour, minute = int(match.group(1)), int(match.group(2))
        # A zero seconds field says nothing the hour and minute have not already
        # said, so `10:30:00` reads exactly as `10:30` does.
        seconds = int(match.group(3)) if has_seconds and match.group(3) else 0
        # 24 is admitted only with a zero minute: ISO 8601 writes end-of-day as
        # 24:00, and without it the hour and the minutes were read as two unrelated
        # numbers with the colon left standing between them: "twenty-four:zero zero"
        # reaching the model as written. 24:30 is not a time in any convention and stays
        # as written.
        if hour == _END_OF_DAY_HOUR and (minute or seconds):
            return match.group(0)
        parts = [cardinal(hour, language)]
        if grammar.time_infix:
            parts.append(grammar.time_infix)
        # 14:05 keeps its zero spoken where the minute is under ten in
        # languages without an infix, "fourteen oh five" territory, but
        # the plain cardinal is never *wrong*, only plainer, so it ships.
        # A zero minute is dropped from `14:00` and kept in `14:00:45`, where
        # dropping it would move the seconds into the minutes' place.
        if minute or seconds:
            parts.append(cardinal(minute, language))
        if seconds:
            parts.append(cardinal(seconds, language))
        spoken = " ".join(parts)
        return spoken + " " if _ascii_letter_at(match.string, match.end()) else spoken

    out = time_run.sub(say, text)
    # A dot means a time only where it does not already mean a decimal point,
    # and a dotted time carries no seconds: `10.30.45` is a version string as
    # readily as a timestamp, where `10:30:45` is a timestamp in every convention.
    if grammar.decimal_separator != ".":
        out = dotted_time_run.sub(lambda m: say(m, has_seconds=False), out)
    return out


def expand_abbreviations(text: str, language: str) -> str:
    """The authority-listed abbreviations, written out.

    Longest first, so ``fr.o.m.`` cannot be half-eaten by a shorter entry, and
    matched only at word boundaries, an abbreviation inside a word is part of
    the word. Runs before the punctuation pass: the periods inside ``z.B.``
    would otherwise become prosodic stops and break the sentence mid-phrase.
    """
    grammar = _grammars().get(language)
    if grammar is None or not grammar.abbreviations:
        return text
    out = text
    for written, spoken in grammar.abbreviations:
        pattern = r"(?<![\w.])" + re.escape(written) + r"(?![\w.])"

        # `spoken` is data, so it goes through a callable: a backslash or
        # group reference inside it would otherwise be read as a replacement
        # template and corrupt the output.
        def _substitute(_match: re.Match[str], _spoken: str = spoken) -> str:
            return _spoken

        out = re.sub(pattern, _substitute, out)
    return out
