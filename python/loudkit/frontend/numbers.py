"""Numbers, said out loud, in the twelve languages the kit speaks.

Why this exists at all
----------------------

Digits do not reach the model. The frontend tokenises **graphemes**, and a
checkpoint trained on normalised transcripts has never seen ``45`` — so a digit
is at best a dead embedding and at worst dropped. The failure is not a
mispronunciation, it is silence or garbage where a number should be, and no
amount of acoustic quality repairs it.

Why it is written here rather than taken from a library
-------------------------------------------------------

Two reasons, and the second is the one that matters.

The licence: the obvious dependency is LGPL-2.1, which is not a licence this kit
can carry into every embedding it is meant for.

The grammar: that library has **no case or gender machinery for Polish at all**
— one nominative form per numeral — while shipping full six-case declension for
Russian in the same release. Polish is the language where getting this right
matters most (a published measurement puts a morphology-blind normaliser at
~30% against ~91% for a morphology-aware one), so the dependency would have to
be replaced for the hardest language anyway. There is nothing to inherit.

How it is organised, and why
----------------------------

**The grammar is data; only the interpreter is code.** Nine languages × five
implementations is forty-five chances for a rule to drift. One JSON file read by
five small interpreters is one chance, and the fixture catches it.

The data format follows the shape these systems actually have: a regular
generative core, plus a listed set of irregular forms. Every European number
grammar in this set is "units, teens, tens, scales, and a composition rule",
with the interesting variation in *how* the pieces join:

- **order** — German, Dutch and Danish say the unit first (*einundzwanzig*,
  literally "one-and-twenty"); the rest say the ten first.
- **joiners** — Spanish *treinta y uno*, Portuguese *vinte e um*, French
  *soixante et onze* but *quatre-vingt-un*, German's bare *und*.
- **elision** — Italian *ventuno* and *ventotto* drop the ten's final vowel
  before a vowel-initial unit.
- **agreement** — Polish and Spanish inflect the numeral for the gender of what
  is counted; Polish additionally for case.

Rather than model each of those as a rule engine, irregular *values* are listed
outright. That is not a shortcut: it is how the grammars are described in their
own reference works, it keeps the interpreter small enough to port honestly, and
a listed form can be checked by eye against a dictionary.

What it deliberately does not do
--------------------------------

Nothing here reads context. A numeral's case in Polish, or its gender in
Spanish, is a property of the *sentence*, not of the number — see
:func:`cardinal` and the ``grammar`` argument for the seam where a caller with
that knowledge supplies it. Choosing it automatically needs a morphological
tagger, and that is a different component with a different failure mode.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

# Re-exported, not defined here: `NumberGrammarError` is part of this module's
# grammar contract and `loudkit.numbers.NumberGrammarError` is the name callers
# know, but it now inherits `LoudkitError` — and this module imports that, so
# the class has to be declared on the other side of the edge.
from ..errors import NumberGrammarError

__all__ = [
    "Grammar",
    "unit_word",
    "NumberGrammarError",
    "cardinal",
    "expand",
    "expand_abbreviations",
    "expand_times",
    "supported_languages",
]

# ASCII digits only, explicitly: `\d` means Unicode in Python and Rust but
# ASCII in Go and JS, and a character class that differs per implementation is
# a cross-port divergence waiting to fire. Eastern-Arabic digits keep their
# existing, already five-way-conformant path through the Polish respeller.
# A leading minus is part of the number, and a space can group thousands.
#
# Both were losses of meaning rather than of polish. "-5 degrees" was read as
# *five degrees* — the sign silently gone, the temperature the opposite of what
# was written — because the punctuation pass turned a hyphen outside a word into
# a space, and by then nothing knew a number had lost anything. "200 000" came
# out as *two hundred zero zero zero*.
#
# The minus is only a minus where it cannot be a hyphen or a range: at a
# boundary, with a digit right behind it. The space only groups when every group
# after the first is exactly three digits and the first is one to three, so
# "in 2024 200 people" stays two numbers — 2024 is four digits and disqualifies
# the join.
#
# `(?![\w])` on the far end is the mirror of the lookbehind, and it was missing.
# A run glued to a word on the *left* was left alone; a run glued to one on the
# right was expanded up to the letter and then abandoned, which is worse than
# either whole answer: "5x3" came out *fivex3* and "1e6" came out *onee6* — a
# word welded to a digit, which is not a reading of anything. Now a digit run is
# part of a word if either end touches one, and the token is left as written for
# the model to read, exactly as `iOS18` always was.
#
# `(?! ?[0-9])` is what makes that last clause true rather than nearly true. A
# grouped run must reach a boundary; a partial one is not a grouped number.
# Without it the regex simply took the longest prefix that fit and abandoned the
# rest, which is how "+1 202 555 0199" — a phone number, and one that no
# grouping rule should ever have matched — was read as "one billion two hundred
# and two million, five hundred and fifty-five thousand and nineteen" with a
# bare "9" left dangling behind it. Eight of the twelve languages said it that
# way. Refusing the whole join drops each group back to being its own number,
# which is what a run of unequal groups is.
#
# That fixes the ragged case and not the tidy one: "+48 123 456 789" *is* a
# valid 1-3-plus-threes grouping, so no care about boundaries saves it. See
# `_PHONE_RUN`, which takes it before this pattern ever sees it.
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

Folded because the alternative was twelve different answers to the same string.
`_DIGIT_RUN` is ASCII by design, so ``١٢٣`` reached the model as written in
eleven languages — and in Polish it did not, because the respeller's own digit
test is `str.isdigit()`, which is true of every Unicode decimal digit. The same
input read as *sto dwadzieścia trzy* in one language and as three raw code
points in the rest, from one funnel reporting one fingerprint.

The separators matter more than the digits. U+066B is a decimal point, and it
is not in the `[.,]` this module looks for, so ``٣٫١٤`` lost its separator
entirely and was read as two numbers: *trzy czternaście*. That is the same
change of meaning as reading a decimal as a clock time, arriving through a
character set instead of a pattern.
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

    Applied by the funnel next to NFC rather than in the number pass, and for
    the same reason NFC is: it is a normalisation, and every pass after it
    should see one spelling. In the number pass it was also too late for the
    percent sign, since the table that turns U+066A into a word runs earlier.

    Language-dependent for the separators, and this is not a detail. U+066B is a
    *decimal* separator, so folding it to a dot everywhere turned ``٣٫١٤`` into
    ``3.14`` — which in the eleven languages that write decimals with a comma is
    the written form of a clock time, and read out as *drei Uhr vierzehn*. The
    same wrong reading this module had just been fixed for, arriving through a
    character set instead of a pattern. It folds to whatever mark the language
    actually uses.
    """
    grammar = _grammars().get(language)
    decimal = grammar.decimal_separator if grammar else "."
    grouping = "," if decimal == "." else "."
    table = {**_FOREIGN_DIGITS, "\u066b": decimal, "\u066c": grouping}
    return _FOREIGN_DIGIT_RUN.sub(lambda m: table[m.group(0)], text)


_UNICODE_MINUS = re.compile("[\u2212\u2010](?=[0-9])")
"""U+2212 MINUS SIGN and U+2010 HYPHEN, where a digit follows: folded to ASCII.

Everything downstream reads the sign as `-`, so a typographically correct minus
was not a sign at all -- it fell through to the punctuation pass, which turns a
symbol between a space and a digit into a space, and "−5" was read as *five*.
The same loss of meaning the ASCII case was fixed for, and the temperature is
the opposite of what was written either way.

Only these two, and only before a digit. U+2013 EN DASH is how a *range* is
written ("1979–1983"), and U+2014 EM DASH is punctuation; folding either into a
minus would invent a sign where the text has none. The digit lookahead is what
keeps this from touching a hyphenated word.
"""

_PHONE_RUN = re.compile(r"\+[0-9][0-9 ]*[0-9]")
"""An E.164 telephone number: a plus, then digits, possibly grouped by spaces.

Read digit by digit, and taken before `_DIGIT_RUN` because it is the one shape
that pattern cannot decline on its own. "+48 123 456 789" is a perfectly valid
one-to-three-then-threes grouping, so it was read as *forty-eight billion one
hundred and twenty-three million…* in eight of the twelve languages, and
"+1 202 555 0199" came out as a ten-digit cardinal with a bare "9" dangling
behind it, because the last group has four digits and only three of them fit.

The plus is the evidence. E.164 requires one and a grouped thousand never
carries one, so this is not a phone-number *detector* -- it does not guess at
national formats, area codes or separators -- it is the one written form that
says outright it is not a quantity.

`_MIN_E164_DIGITS` keeps it away from a signed number. "+5 degrees" and
"+250 points" are deltas, not numbers to spell out, and "+1 000 000 users" is a
million however it is punctuated. E.164 allows fifteen digits and a number
short enough to be a delta is not one.
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
    Portuguese *duas mil* — *mil* is transparent — against Polish *dwa
    tysiące*, where the multiplier agrees with *tysiąc* itself."""

    one_word: str
    """What is said for a multiplier of exactly one. Empty means the bare scale
    word (*mille*, *tusind*); ``"~"`` means compose it like any other
    multiplier (*one thousand*); anything else is the literal word — German
    *eine* (Million is feminine), Italian *un*."""

    separate: bool
    """Whether the scale word takes spaces around it even in a language that
    writes numbers solid. *neunhundert…neunzig **Millionen** …* against
    *…tausend* glued."""

    link: str
    """What joins this scale's group to what follows, when no small-tail joiner
    fires. Finnish glues the multiplier to *tuhatta* but separates the groups
    with a space — *kaksituhatta kaksikymmentäkuusi* — and Swedish does the
    same at the thousand boundary while writing everything else solid. Empty
    means the language's ``word_join``."""

    multiplier_gender: str
    """A gender the multiplier is composed in, overriding the caller's.
    Swedish thousands take the common form — *tjugoentusen*, not
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
    """Words for 20, 30, … 90 — eight entries, index 0 is twenty."""

    hundred: str
    """The word for a hundred when it stands after a multiplier."""

    hundreds: tuple[str, ...]
    """Words for 100, 200, … 900 when the language does not build them
    compositionally (Spanish *doscientos*, Portuguese *duzentos*). Empty when it
    does, in which case ``hundred`` is used with a multiplier."""

    scales: tuple[Scale, ...]
    """Scale nouns from largest to smallest, above a hundred — see :class:`Scale`."""

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
    """What separates the parts in writing — a space everywhere except German,
    Dutch and Danish, which write the whole number as one word. This is
    orthography, not phonology, but the model reads graphemes, so it matters."""

    combining_ones: tuple[tuple[int, str], ...]
    """Forms a unit takes when it is *part of* a larger number rather than
    standing alone.

    German is the clear case: *eins* answers "how many", but every compound uses
    *ein* — **ein**undzwanzig, **ein**hundert, **ein**tausend. This is position,
    not gender, and conflating the two would make the caller pass a gender to
    get a form that has nothing to do with gender.
    """

    scale_joiner_on_round_hundreds: bool
    """Portuguese inserts *e* after a scale when the remainder is a whole number
    of hundreds — *mil e oitocentos* (1800) but *mil oitocentos e noventa e
    dois* (1892). The rule is from Cunha & Cintra and is not derivable from the
    magnitude alone, which is why it is a field rather than a threshold."""

    exceptions: tuple[tuple[int, str], ...]
    """Values whose form is simply listed: Spanish *veintiuno*, Italian
    *ventotto*, French *quatre-vingts*. Checked before anything is composed."""

    minus_word: str
    """The language's own word for a negative. It was English everywhere until
    the CLDR differential caught it: *menos*, *moins*, *meno*, *min* — and it
    always joins with a space, even in languages that write numbers solid
    (German *minus eins*, not *minuseins*)."""

    gender_scopes: tuple[tuple[int, str], ...]
    """Where each value's gender agreement applies. Absent means everywhere.

    Three scopes exist in this language set, and they were found by the CLDR
    differential rather than guessed:

    - ``"standalone"`` — only when the value is the entire number. Polish:
      *jedna kobieta*, but *sto jeden kobiet* and *dwadzieścia jeden* — while
      Polish 2 agrees everywhere (*dwadzieścia dwie*).
    - ``"outside_tens"`` — everywhere except inside the solid tens compound.
      Danish: *hundrede og et* (agrees), but *enogtyve* (does not).
    - the default — everywhere. Spanish: *treinta y una*.
    """

    hundreds_gendered: tuple[tuple[str, tuple[str, ...]], ...]
    """Gendered variants of the explicit hundreds table. Spanish and Portuguese
    inflect the whole series — *doscientas*, *duzentas* — not just the unit."""

    hundred_plural_final: str
    """French *deux cents* but *deux cent un*: the multiplied hundred takes a
    plural mark only when nothing follows it. Empty for everyone else."""

    scale_large_joiner: str
    """What joins a scale group to a remainder of a hundred or more. English
    reads long numbers with a breath — *nine hundred thousand, nine hundred* —
    and the comma is that breath in graphemes. Empty means ``word_join``."""

    decimal_separator: str
    """Which mark separates the whole part from the fraction. Eight of the nine
    use a comma; English is the exception, and a ``3.5`` read in a comma locale
    is *thirty-five*."""

    decimal_word: str
    """What that mark is called out loud — *point*, *przecinek*, *Komma*."""

    time_infix: str
    """The spoken word between clock hour and minute, where the language has
    one — German *vierzehn Uhr dreißig*. Empty elsewhere: Kotus reads 14.30 as
    *neljätoista kolmekymmentä* and Isof's 16.31 is *sexton trettioett*."""

    abbreviations: tuple[tuple[str, str], ...]
    """Expandable abbreviations, from each language authority's own list — and
    only the unambiguous ones. Swedish *s.k.* collapses three inflections,
    Finnish *v.* has four readings, *mm.* collides with millimetres: the traps
    stay out on purpose, because a wrong expansion is worse than a spelled
    abbreviation."""

    unit_words: tuple[tuple[str, str], ...]
    """Per-language wording for currency and measure symbols: ``$`` is
    *dólares* to a Spanish render and *Dollar* to a German one. It lived in the
    funnel as an (en, pl) pair with ``pl if polish else en`` — which meant
    seven of nine languages heard English."""

    genders: tuple[tuple[str, tuple[tuple[int, str], ...]], ...]
    """Per-gender overrides for the values that agree. Polish *dwa / dwie*,
    Spanish *uno / una*, German *ein / eine*. Keyed by a gender name the caller
    supplies; absent means the language does not inflect that value."""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Grammar:
        """Build from the JSON, failing loudly on a malformed entry.

        Parsed strictly rather than with ``.get`` defaults: a grammar missing a
        field is a grammar that will silently produce a wrong word for some
        range outside the fixture's coverage.
        to write down.
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

        scales = tuple(
            Scale(
                value=int(entry["value"]),
                forms=tuple(str(f) for f in entry["forms"]),
                multiplier_agrees=bool(entry.get("multiplier_agrees", False)),
                one_word=str(entry.get("one", "~")),
                separate=bool(entry.get("separate", False)),
                link=str(entry.get("link", "")),
                multiplier_gender=str(entry.get("multiplier_gender", "")),
                small_joiner=str(entry.get("small_joiner", "")),
            )
            for entry in need("scales", list)
        )
        genders = tuple(
            (
                str(name),
                tuple((int(k), str(v)) for k, v in sorted(forms.items(), key=_as_int)),
            )
            for name, forms in sorted(raw.get("genders", {}).items())
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
        """The form ``value`` takes in ``gender``, or ``None`` if it does not
        inflect — which is the common case, and why this returns an option
        rather than a default.

        ``position`` names where in the number this value sits — "standalone"
        (it is the whole number), "tail" (it ends a larger number), or
        "tens_pair" (inside the solid units-and-tens compound) — and the
        grammar's per-value scope decides whether agreement reaches it there.
        See ``gender_scopes``.
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
    implementations share one source of truth. A rule that lives in code is nine
    languages times five ports of opportunity to drift.
    """
    raw = json.loads(
        resources.files("loudkit.models.data").joinpath("numbers.json").read_text("utf-8")
    )
    return {lang: Grammar.from_dict(entry) for lang, entry in raw["languages"].items()}


def supported_languages() -> tuple[str, ...]:
    """Language ids this module can verbalize, sorted."""
    return tuple(sorted(_grammars()))


def cardinal(value: int, language: str, *, gender: str | None = None) -> str:
    """``value`` as words.

    Args:
        value: the integer to say. Negatives are read with the language's own
            minus word; there is no locale in this set that omits it.
        language: one of :func:`supported_languages`.
        gender: the grammatical gender of the counted noun, for the languages
            that agree with it — Polish and Spanish inflect *one* and *two*,
            German inflects *one*. ``None`` gives the citation form, which is
            what a bare number in a list wants and what every other language
            here uses unconditionally.

    Raises:
        NumberGrammarError: if the language is unknown, or the value is larger
            than the grammar's largest scale. Raising rather than falling back
            to digits is deliberate — see :class:`NumberGrammarError`.

    Examples::

        cardinal(21, "en")                  # 'twenty-one'
        cardinal(21, "de")                  # 'einundzwanzig'
        cardinal(71, "fr")                  # 'soixante et onze'
        cardinal(2, "pl", gender="f")       # 'dwie'
    """
    grammars = _grammars()
    if language not in grammars:
        raise NumberGrammarError(
            f"no number grammar for {language!r}; have {', '.join(supported_languages())}"
        )
    grammar = grammars[language]
    # Past the largest scale the composition still *runs* — it stacks scales and
    # says "a million milliards" — but that is not what any of these languages
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
    # alone, but *sto jeden* — the trailing 1 of a larger number is compound
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
        link = f"{join}{scale.small_joiner}{join}" if join else f" {scale.small_joiner} "
        if join == " ":
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
    otherwise. Polish *pięć tysięcy* but *dwadzieścia dwa tysiące* — the rule
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

    The seam between this module and the speech funnel. It runs *after* the
    symbol pass, so a currency amount has already become "250 pounds" and only
    the ``250`` is left to say, and *before* the punctuation pass, which turns
    a decimal separator into a space and would leave two unrelated numbers.

    **It never raises**, and it never *half*-reads a number: a value past the
    largest scale the grammar has a word for is read digit by digit, which is
    what such a number almost always is — an identifier, a code, a serial. That
    is a deliberate difference from :func:`cardinal`, which refuses: a library
    call has a caller who can decide, and a text funnel has a user whose
    sentence must still be spoken.

    It does leave digits behind, deliberately: a run glued to a word is part
    of that word and stays written —
    `iOS18`, `r123`, `v1.2.3`, `5x3` — because *iOSeighteen* is not a reading of
    anything. So is a dotted run that no convention resolves: `1.2.3` is a
    version, `192.168.0.1` an address, and `18.08.2026` in English a date whose
    field order half the English-speaking world reads the other way round.
    Digits reaching the model are a reading the model can make; a confident
    wrong number is one that cannot be undone.

    A separator between digits is a decimal mark only when it is *the*
    language's decimal mark. English ``3.5`` is three point five; English
    ``3,500`` is a grouped thousand, and reading its comma aloud would be
    absurd — so the other mark is simply dropped, which is what a reader does
    with it.
    """
    grammar = _grammars().get(language)
    if grammar is None:
        return text

    def say(match: re.Match[str]) -> str:
        # Normalised once, here, so everything downstream sees one shape: a sign
        # kept apart from the digits, and thousands spaces gone. The alternative
        # was teaching every later function about two more spellings of a
        # number.
        # The lookbehind sees one character, and an identifier can put a dot
        # between the letter and the digits: in "v1.2.3" the scan starts at the
        # `2`, because what precedes it is a dot rather than a word character,
        # and the version came out "v1.two point three" — half read, half not.
        # Walking back over word characters and dots answers the question the
        # lookbehind was asking: is this run part of a token that has a letter
        # in it?
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
        digits = "".join(ch for ch in match.group(0) if ch.isdigit())
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

    The shape :data:`_DIGIT_RUN` binds as a group after the first, and so the
    shape a space in front of them may be grouping.

    The slice has to *have* three, which is where the reference was wrong and
    the three ports checking a width were right: ``"2"[0:3].isdigit()`` is True,
    so a lone digit passed for a group and `R2 2` had the backward walk cross
    the space, reach the `R` and refuse a number nothing was glued to.
    """
    group = text[i : i + _GROUP_DIGITS]
    return len(group) == _GROUP_DIGITS and all(_is_ascii_digit(c) for c in group)


def _continues_a_group(text: str, i: int) -> bool:
    """...and no fourth digit behind them.

    A group the pattern could have *bound*, rather than a ragged run that only
    looks like one — the first group may be one to three digits and says nothing
    about whether the space groups, so this is asked of the half whose width
    :data:`_DIGIT_RUN` fixes.

    Which of the two questions to ask differs by direction, and the asymmetry is
    the measurement rather than an oversight; see :func:`_glued_forward`.
    """
    after = i + _GROUP_DIGITS
    return _starts_a_group(text, i) and (after >= len(text) or not _is_ascii_digit(text[after]))


def _glued_to_a_word(text: str, start: int) -> bool:
    r"""Whether the digit run at ``start`` sits inside a token containing a letter.

    Walks back over word characters and dots — the two things an identifier puts
    between its letters and its digits — and answers yes on the first letter. A
    space, a comma or any other separator ends the walk and the answer is no.

    Only backwards: the forward direction is `(?![\w])` in the pattern, which
    the regex can express. Backwards it cannot, because the lookbehind sees one
    character and the letter may be several away.
    """
    i = start
    # `,` is in the walk for the same reason `.` is: it is a numeric
    # separator, and the fraction group already treats the two as one class.
    # Without it the walk stopped at the comma and `x3,14` read
    # "x3,vierzehn" — the `3` refused for being glued to the `x`, the `14`
    # matched on its own because nothing connected it to either.
    #
    # `-` and `+` are in the walk because an exponent puts one between the
    # letter and the digits: in `1e-3` the scan starts at the `3`, walks back
    # over `-` to `e`, and stops calling it a number. A bare `-5` is unaffected
    # — the walk reaches a space or the start of the string and finds no letter.
    #
    # A *grouping* space is crossed too, and that one was a genuine parity
    # break. `x200 000` binds as a single match in Go and Rust, whose engines
    # have no backtracking, so the lookbehind refuses the whole run and the
    # token is left written. Python, JS and Swift backtrack, match the
    # standalone `000`, and read "x200 zero zero zero" — half a token spoken,
    # which is the class the right-hand guard exists to stop. Five
    # implementations, two answers, and no fixture case with a group glued to a
    # letter to notice.
    while i > 0:
        previous = text[i - 1]
        if previous.isalnum() or previous in "_.,-+":
            i -= 1
        elif (
            previous == " "
            and i >= 2  # noqa: PLR2004 - a digit, then the space
            and _is_ascii_digit(text[i - 2])
            and _continues_a_group(text, i)
        ):
            # A thousands space, judged by the group the walk is stepping *out
            # of* rather than the one behind it.
            #
            # Two wrong versions preceded this. "A digit on each side" crossed
            # `R2 5`, which is not a grouped number, and refused the `5`.
            # "Exactly three digits behind the space" then broke `a1 000 000`,
            # where the first group is legitimately one digit.
            #
            # The discriminator is the group being continued: `_DIGIT_RUN` binds
            # groups of exactly three after the first, so `000` can be a
            # continuation and `5` cannot. Behind the space may be one to three
            # digits — that is the first group, and its width was never the
            # question.
            #
            # A digit behind the space as well, and that is not redundant: the
            # walk crosses repeatedly, so `Sold 200 000` went `000` -> `200` ->
            # and then over the space before `200` into "Sold", refusing a
            # number nothing was glued to. The space that ends a word is not a
            # thousands space no matter what follows it.
            #
            # `_continues_a_group` and not `_starts_a_group`, which is what the
            # forward walk asks: admitting a fourth digit here reaches the `e` of
            # `e3 1000` and welds two tokens into one, unsaying a four-digit
            # number that shares nothing with the exponent but a space.
            i -= 1
        else:
            return False
        if text[i].isalpha():
            return True
    return False


def _glued_forward(text: str, end: int) -> bool:
    r"""Whether the token continues past the match into a letter.

    The mirror of :func:`_glued_to_a_word`, and needed for the same reason it
    is: `200 000x` matched `200` alone, because the grouped alternative reached
    the `x` and the right-hand guard refused it — so the regex backtracked to
    the first group and read "two hundred 000x". Go and Rust, which do not
    backtrack, left the whole token written. Half a token spoken again, and the
    opposite half from `x200 000`.

    A grouping space is crossed, so `200 000x` is one token; the ordinary space
    in `2024 200 people` is not, because what follows *it* is a word rather than
    a digit group, and those two numbers stay two numbers.

    Forwards the group may be *ragged* — three digits and a fourth — where
    backwards it may not, and the asymmetry is the measurement rather than an
    oversight. This walk finishes the run the pattern refused to bind, and a
    ragged group is exactly why it refused: `1 0023R` matched the `1` alone and
    read "en 0023R", half a run spoken with the rest welded to a letter, which
    is the class the right-hand guard exists to stop. Backwards the group *is*
    the match, whose width the pattern already fixed, and the same looseness
    there swallows the `1000` of `e3 1000` — a four-digit number across an
    ordinary space, unrelated to the exponent behind it. Over 4800 fuzzer
    sentences the loose question in both directions changes 60 readings and 56
    of them are losses; asked forwards only, 20 change — four numbers that had
    gone unsaid, sixteen ragged runs no longer read half way.
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
            # exponent two tokens away — four of the fuzzer's eight Go
            # divergences were that, with Go right and this side wrong — and the
            # `5` of `R2 5 iOS` is its own number.
            i += 1
        else:
            return False
    return False


def _truncated_by_a_fraction(text: str, end: int) -> bool:
    r"""Whether a decimal point with digits behind it follows the match.

    The fraction group is `(?:[.,][0-9]+)*`, which can match zero times, and the
    regex will happily shrink it to zero so that the trailing `(?![\w])` lands
    on the dot instead of on a letter. `1.5e3` matched just the `1` that way and
    read "one.5e3"; `3.14abc` read "three.14abc". A word welded to digits, which
    is what the right-hand guard was added to stop, arriving through the one
    part of the pattern that is allowed to disappear.

    A number that really ends here has nothing of the sort behind it: `3.14.` at
    the end of a sentence is followed by a dot and then a space, and `1,000` by
    a space. Only a separator *with a digit after it* means the match stopped
    early.
    """
    return end + 1 < len(text) and text[end] in ".," and text[end + 1].isdigit()


def _is_number(literal: str, g: Grammar) -> bool:
    """Whether a digit run is a *quantity*, or just digits with dots in them.

    ``1.2.3``, ``192.168.0.1`` and ``12.03.2026`` are a version, an address and a
    date. None is a number. Partition splits on one separator at a time, so
    the leftovers reach ``int()`` here: treating the remainder as a quantity
    either raises ``ValueError`` on ordinary text or — with a comma decimal
    mark, where segments concatenate — speaks ``192.168.0.1`` as *nineteen
    million two hundred sixteen thousand eight hundred one*.

    A run is a quantity when it has at most one separator, or when its
    separators actually group — every segment after the first exactly three
    digits, and the first one to three. Anything else is left exactly as it was
    written, for a later pass (a date) or for the reader to deal with.
    """
    grouping = "," if g.decimal_separator == "." else "."
    whole, _, fraction = literal.partition(g.decimal_separator)
    # A second decimal mark in what should be the fraction: not a quantity, and
    # historically the crash. `partition` splits once, so "1.2.3" left "2.3"
    # here and every character of it was fed to `int`.
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
    "1.000" is a thousand; a Polish "2.5" is not twenty-five — the dot there is
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
        # The fractional part is read digit by digit — "point four nine", not
        # "point forty-nine" — because that is how a decimal is said, and
        # because leading zeros carry meaning there that a cardinal would eat.
        parts.extend(_digit_by_digit(fraction, language, gender))
    return " ".join(p for p in parts if p)


def _say_integer(digits: str, language: str, gender: str | None) -> str:
    # Leading zeros mean a code, not a quantity: 0042 is zero zero four two,
    # never forty-two — int() would silently eat the zeros that carry meaning.
    if len(digits) > 1 and digits.startswith("0"):
        return " ".join(_digit_by_digit(digits, language, gender))
    try:
        return cardinal(int(digits), language, gender=gender)
    except NumberGrammarError:
        return " ".join(_digit_by_digit(digits, language, gender))


def _digit_by_digit(digits: str, language: str, gender: str | None) -> list[str]:
    return [cardinal(int(ch), language, gender=gender) for ch in digits]


def unit_word(symbol: str, language: str) -> str | None:
    """The word ``symbol`` takes in ``language``, or ``None`` if unknown.

    The seam the funnel's symbol pass uses: it keeps owning *where* the word
    goes (a currency mark is written before its amount and spoken after it) and
    asks here only *which* word — because the which is a per-language fact that
    lives with the other per-language facts.
    """
    grammar = _grammars().get(language)
    if grammar is None:
        return None
    for at, word in grammar.unit_words:
        if at == symbol:
            return word
    return None


_TIME_RUN = re.compile(r"(?<![\d.,:])([01]?[0-9]|2[0-4]):([0-5][0-9])(?![.,:]?\d)")
"""``14:30``, and nothing that merely contains it.

A colon between an hour and two minute digits is a clock time in every language
here, which is why this is the pattern that needs no help deciding.

The lookarounds are the whole point. ``\\b`` on its own let this match *inside* a
longer run and read the front of it as a time, with the rest left behind as a
separate number. A time is a time only when nothing else is attached to either
end, so a digit, a dot, a comma or a colon on either side disqualifies it.
``14:30.`` at the end of a sentence still matches, because what follows the dot
there is not a digit.
"""

_DOTTED_TIME_RUN = re.compile(r"(?<![\d.,:])([01]?[0-9]|2[0-4])\.([0-5][0-9])(?![.,:]?\d)")
"""``14.30``, which is a clock time in some of these languages and a decimal in
others — so this pattern is only applied where the language says it is a time.

The dot is the whole difficulty. ``14.30`` is how German, Danish, Finnish,
Norwegian and Swedish write half past two in the afternoon; ``3.14`` is how
English writes pi. The shapes are identical and no lookaround separates them.

What separates them is already in the grammar file: **a language that writes
clock times with a dot does not use the dot as its decimal separator.** German
writes ``14.30 Uhr`` and ``2,50 €``; English writes ``2:30`` and ``$2.50``. So
this pattern applies exactly where ``decimal_separator`` is not ``.``, which
today means everywhere except English.

Applying the dotted pattern in a dot-decimal language reads every two-digit
decimal as a clock: ``$0.49`` as *zero forty-nine dollars*, ``3.14`` as *three
fourteen*. The pattern is therefore gated on the grammar's decimal separator,
which is what keeps those decimals intact.
"""


@cache
def _time_patterns(time_infix: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """The two clock-time patterns, extended to consume a written infix word.

    German writes the time *with* the word the spoken form also carries:
    ``um 14.30 Uhr``. The reading puts the infix where it belongs — between
    hour and minutes, *vierzehn Uhr dreißig* — so the written ``Uhr`` is that
    same spoken token, not an additional one, and leaving it standing said it
    twice: *vierzehn Uhr dreißig Uhr*. When the source carries the infix
    immediately after the time, the match swallows it and the normal reading
    supplies the one copy.

    Every piece is spelled out because five implementations must match
    identically: the whitespace run is ASCII space and tab (regex engines
    disagree on what ``\\s`` covers), the guard refuses an ASCII letter or
    digit so *Uhrzeit* keeps its word whole, and case matters — the grammar
    data says ``Uhr`` and this rule does not reach past that.
    """
    suffix = f"(?:[ \\t]+{re.escape(time_infix)}(?![0-9A-Za-z]))?"
    return (
        re.compile(_TIME_RUN.pattern + suffix),
        re.compile(_DOTTED_TIME_RUN.pattern + suffix),
    )


def expand_times(text: str, language: str) -> str:
    """Clock times as words: ``14:30`` and ``14.30`` become their reading.

    Runs before :func:`expand`, which would otherwise read the separator's two
    sides as unrelated numbers and leave the colon behind. An hour of 0–23 and
    exactly two minute digits, so ``3.5`` (a decimal) and ``1.000`` (a grouped
    thousand) never match — and, for the dotted form, only in the languages
    whose decimal separator is not the dot. See ``_DOTTED_TIME_RUN``: the same
    ``H.mm`` is a time in German and a price in English, and the grammar file
    already knows which is which.

    The reading is hour and minute as cardinals with the language's own infix:
    *vierzehn Uhr dreißig*, *neljätoista kolmekymmentä*, *fourteen thirty*. A
    minute of zero says the hour alone. Deliberately not the colloquial clock
    (*half three* is 2:30 in six of these languages and 3:30 in none of them —
    emitting it wrong by an hour is the highest-severity mistake a time reader
    can make, so the plain reading wins until the colloquial one is measured).

    A written infix directly after the time — German ``um 14.30 Uhr`` — is
    consumed rather than duplicated; see :func:`_time_patterns`.
    """
    grammar = _grammars().get(language)
    if grammar is None:
        return text
    time_run, dotted_time_run = (
        _time_patterns(grammar.time_infix)
        if grammar.time_infix
        else (_TIME_RUN, _DOTTED_TIME_RUN)
    )

    def say(match: re.Match[str]) -> str:
        hour, minute = int(match.group(1)), int(match.group(2))
        # 24 is admitted only with a zero minute: ISO 8601 writes end-of-day as
        # 24:00, and without it the hour and the minutes were read as two unrelated
        # numbers with the colon left standing between them -- "twenty-four:zero zero"
        # reaching the model as written. 24:30 is not a time in any convention and stays
        # as written.
        if hour == _END_OF_DAY_HOUR and minute:
            return match.group(0)
        parts = [cardinal(hour, language)]
        if grammar.time_infix:
            parts.append(grammar.time_infix)
        if minute:
            # 14:05 keeps its zero spoken where the minute is under ten in
            # languages without an infix — "fourteen oh five" territory — but
            # the plain cardinal is never *wrong*, only plainer, so it ships.
            parts.append(cardinal(minute, language))
        return " ".join(parts)

    out = time_run.sub(say, text)
    # A dot means a time only where it does not already mean a decimal point.
    if grammar.decimal_separator != ".":
        out = dotted_time_run.sub(say, out)
    return out


def expand_abbreviations(text: str, language: str) -> str:
    """The authority-listed abbreviations, written out.

    Longest first, so ``fr.o.m.`` cannot be half-eaten by a shorter entry, and
    matched only at word boundaries — an abbreviation inside a word is part of
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
