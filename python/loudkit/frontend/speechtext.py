"""The text funnel: :func:`speech_text` for all twelve languages, and
:func:`lexical_respelling` for Polish.

The reference. The Rust, Go, TypeScript and Swift copies mirror this file
and are held to it by ``tests/data/conformance``.

See ``docs/design/text-funnel.md``.
"""

from __future__ import annotations

import bisect
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache, lru_cache
from string import ascii_lowercase
from typing import Any

from .dates import expand_dates, expand_ordinals
from .letters import letter_name, spell_acronym, spell_acronyms
from .numbers import (
    SCALE_SUFFIX_PATTERN,
    cardinal,
    decimal_separator,
    expand_abbreviations,
    expand_roman_numerals,
    expand_times,
    fold_foreign_digits,
    scale_nouns,
    scale_suffix_word,
    supported_languages,
    unit_word,
)
from .numbers import expand as expand_numbers
from .textconfig import NUMERALS_PATH, PL_RULES_PATH, RESPELL_PATH

__all__ = ["speech_text", "lexical_respelling", "PolishLexicon"]

# ------------------------------------------------------------------ funnel

# Zero-width and formatting characters, and the soft hyphen, invisible in
# every editor, and a grapheme engine sees them as letters (the model reads a
# word that does not exist in any text it was trained on).
_INVISIBLES = frozenset("\u200b\u200c\u200d\u2060\ufeff\u00ad\u180e\u200e\u200f")

# Symbols the model cannot voice, in the order the pass replaces them. The
# first family (→ ✓ ✗ ≈ ≥) is literally outside the vocabulary, the tokenizer
# emits [UNK] and the model receives nothing, while ¢ ° % $ do tokenize and are
# read at the ear's discretion. Both get words.
#
# Which word is a per-language fact and lives with the other per-language facts,
# in `unit_words` in the grammar: a symbol is spoken in the language being read.
# `_SYMBOL_MARKS` below carries the rest. A symbol no grammar covers is left
# written, which is what this module does everywhere the evidence runs out, and
# what the four ports do: spelling an uncovered symbol out of the English row
# is the one thing this table exists to prevent.
_SPOKEN_SYMBOLS: tuple[str, ...] = (
    "%",
    "°",
    "¢",
    "€",
    "£",
    "¥",
    "₹",
    "×",
    "÷",
    "≈",
    "≥",
    "≤",
    "≠",
    "±",
    "→",
    "←",
    "⇒",
    "✓",
    "✔",
    "✗",
    "✘",
    "•",
    "·",
    "▪",
    "◦",
    "…",
    "&",
    "@",
)

# An arrow, a bullet and an ellipsis are the same pause in every language, so
# they are a rule here rather than a row in twelve grammars.
_SYMBOL_MARKS: Mapping[str, str] = {
    "→": ",",
    "←": ",",
    "⇒": ",",
    "•": ",",
    "·": ",",
    "▪": ",",
    "◦": ",",
    "…": "...",
}

# The ASCII spellings of the comparison operators, longest first, each named by
# the mathematical symbol whose word it shares. Only these six: `-`, `/`, `.`
# and `+` are ranges, paths, decimals and hyphens far more often than operators,
# and a word put on one of them changes prose that reads correctly today.
_ASCII_OPERATORS: tuple[tuple[str, str], ...] = (
    ("<=", "≤"),
    (">=", "≥"),
    ("!=", "≠"),
    ("==", "="),
    ("<", "<"),
    (">", ">"),
)

_MARKUP_TAG = re.compile(r"</?[A-Za-z!][^<>]*>")
"""A markup tag, comment or declaration, which is not text anyone reads aloud.

The name inside the angle brackets otherwise reaches the model as a word, and
`<!-- ... -->` additionally leaves its `!` behind as a sentence-final
exclamation. A tag is replaced by a space rather than by nothing, because two
block tags meeting back to back are two paragraphs and not one glued word.
"""

# `$` and `£` before a number read as a prefix in writing and a SUFFIX in
# speech: "$5" is "five dollars", not "dollars five". The wording comes from
# `loudkit.frontend.numbers.unit_word`; this set only says which symbols are written
# prefix.
_CURRENCY_PREFIXES = ("$", "£", "€", "¥", "₹")

_CURRENCY_SYMBOLS = (*_CURRENCY_PREFIXES, "¢")
"""Marks that make the number beside them a price, whichever side they sit.

`¢` is here and not in `_CURRENCY_PREFIXES` because `¢49` is not a written order;
it is a suffix in every convention, which is precisely why the prefix pass
never saw it and `0.49¢` reached the clock reader intact."""

# Punctuation that carries prosody stays; the rest becomes a space. These are
# language models trained on punctuated text, so the final period is the
# strongest stop cue, the comma the continuation cue, the question mark the
# only route to interrogative intonation.
_PROSODIC = frozenset(
    ".,!?;:\u2014\u2013\u2026\"\u201c\u201d\u201e«»()'\u2019"
    # The inverted marks.
    "\u00bf\u00a1"
)


def _strip_invisibles(text: str) -> str:
    if not any(sc in _INVISIBLES for sc in text):
        return text
    return "".join(ch for ch in text if ch not in _INVISIBLES)


def _priced(amount: str, language: str) -> str:
    """A currency amount, with its decimal mark spelled the way this language does.

    See ``docs/design/text-funnel.md``.
    """
    separator = decimal_separator(language)
    if separator == ".":
        return amount
    if re.fullmatch(rf"{_DIGIT}+\.{_DIGIT}+", amount):
        return amount.replace(".", separator)
    return amount


def _drop_markup_tags(text: str) -> str:
    if "<" not in text:
        return text
    return _MARKUP_TAG.sub(" ", text)


def _speak_operators(text: str, language: str) -> str:
    """The ASCII comparison operators, as words in this language.

    See ``docs/design/text-funnel.md``.
    """

    def say(m: re.Match[str]) -> str:
        written = m.group(1)
        for spelling, symbol in _ASCII_OPERATORS:
            if spelling == written:
                word = unit_word(symbol, language)
                # An operator no grammar covers stays written, like every other
                # symbol this module has no word for.
                return word if word is not None else written
        return written

    return _OPERATOR_RUN.sub(say, text)


@cache
def _scale_pattern(language: str) -> str:
    """The optional magnitude that may follow a currency amount, as a regex.

    Two alternatives and two groups: the abbreviating letter glued to the
    digits, and the scale noun written beside them. Both cases of each noun are
    spelled out rather than asked of a case-insensitive flag, because
    JavaScript has no inline flag group and a pattern that needs one is a
    pattern the five implementations cannot share.
    """
    nouns = scale_nouns(language)
    if not nouns:
        return ""
    written = "|".join(
        re.escape(form)
        for noun in nouns
        for form in dict.fromkeys((noun, noun[:1].upper() + noun[1:]))
    )
    return rf"(?:({SCALE_SUFFIX_PATTERN})|{_SPACE}({written}))(?![A-Za-z])"


def _scale_after(amount: str, suffix: str, spelled: str, language: str) -> str:
    """The magnitude word standing between a price and its currency, or ``""``.

    A written scale reaches speech in two shapes and both belong before the
    currency word: the letter glued to the digits (`$2.5M`) and the noun beside
    them (`$5 million`). The noun is already this language's own word and is
    kept as written; the letter is a number, so the grammar's scale noun is
    asked for the form this count takes.
    """
    if spelled:
        return spelled
    if not suffix:
        return ""
    whole = re.sub(r"[^0-9]", "", amount.partition(".")[0].partition(",")[0])
    return scale_suffix_word(suffix, int(whole) if whole else 0, language) or ""


def _speak_symbols(text: str, lang: str | None) -> str:
    """Symbols become words in the render's own language.

    See ``docs/design/text-funnel.md``.
    """
    out = text
    language = lang if lang and lang in supported_languages() else "en"
    out = _speak_operators(out, language)

    scale = _scale_pattern(language)

    # Prefix currencies first, while the digits still follow the symbol. The
    # number, and NOT the sentence punctuation behind it: a greedy [\d.,]*
    # would swallow the comma in "£250,".
    for symbol in _CURRENCY_PREFIXES:
        word = unit_word(symbol, language)
        if word is None:
            continue
        # No letter guard in the pattern: `_letter_before` decides, because a
        # `\w` lookbehind admits Nl and No. See docs/design/text-funnel.md.
        pattern = re.escape(symbol) + rf"{_SPACE}?({_DIGIT}+(?:[.,]{_DIGIT}+)*)(?:{scale})?"

        # `word=word` binds this iteration's value rather than closing over the
        # loop variable. `re.sub` runs eagerly so late binding would not bite
        # today, which is exactly the kind of "correct by accident" the next
        # edit breaks.
        def _say_amount(m: re.Match[str], word: str = word) -> str:
            if _letter_before(m.string, m.start()):
                # Not a price: a mark glued to the end of a word. Left written,
                # which is what this module does everywhere the evidence runs
                # out.
                return m.group(0)
            amount = _priced(m.group(1), language)
            suffix, spelled = m.group(2) or "", m.group(3) or ""
            said = _scale_after(m.group(1), suffix, spelled, language)
            if suffix and not said:
                # A scale this language has no noun for. The suffix stays
                # written, which is what it did before the amount was moved.
                return f"{amount} {word}{suffix}"
            return f"{amount} {said} {word}" if said else f"{amount} {word}"

        out = re.sub(pattern, _say_amount, out)
    # ...and the same amount with the symbol *behind* it.
    for symbol in _CURRENCY_SYMBOLS:
        word = unit_word(symbol, language)
        if word is None or symbol not in out:
            continue
        # No letter guard on this side: only whitespace may sit between the
        # amount and the mark, so `3,14 R$` never matches in the first place.
        pattern = rf"({_DIGIT}+(?:[.,]{_DIGIT}+)*){_SPACE}?" + re.escape(symbol)

        def _say_suffix(m: re.Match[str], word: str = word) -> str:
            return f"{_priced(m.group(1), language)} {word}"

        out = re.sub(pattern, _say_suffix, out)
    for symbol in _SPOKEN_SYMBOLS:
        if symbol not in out:
            continue
        replacement = unit_word(symbol, language) or _SYMBOL_MARKS.get(symbol)
        if replacement is None:
            continue
        # A word replacement needs spaces around it; a punctuation one must
        # not gain a space BEFORE it or the comma floats.
        spaced = (
            (replacement + " ")
            if len(replacement) == 1 and replacement in ",."
            else (" " + replacement + " ")
        )
        out = out.replace(symbol, spaced)
    return out


def _letter_before(text: str, at: int) -> bool:
    """Whether the character before ``at`` is a letter, as Unicode category L means it.

    See ``docs/design/text-funnel.md``.
    """
    return at > 0 and unicodedata.category(text[at - 1]).startswith("L")


def _is_word_char(ch: str) -> bool:
    """``\\p{L}`` or ``\\p{Nd}``: the word class all five funnels test.

    Not ``str.isalnum()``, which is also true for Nl and No -- Roman numerals,
    circled digits, superscripts, vulgar fractions. Those are exactly the
    characters :func:`fold_numerals` is about to replace, so treating one as a
    word character asks whether a numeral needs separating from a numeral, and
    Python answered yes where the other four answered no.
    """
    category = unicodedata.category(ch)
    return category.startswith("L") or category == "Nd"


def fold_numerals(text: str) -> str:
    """Every number character becomes something a reader can say aloud.

    See ``docs/design/text-funnel.md``.
    """
    if not any(_folded_numeral(ch) for ch in text):
        return text
    out: list[str] = []
    for i, ch in enumerate(text):
        folded = _folded_numeral(ch)
        if folded is None:
            out.append(ch)
            continue
        spelled, is_digit = folded
        # A slash inside a spelled numeral is a fraction bar, asserted by the
        # character itself: `½` is a half wherever it stands, where a typed `1/2`
        # is a fraction, a date or the `24/7` of ordinary prose. The division
        # sign is the mark the symbol table already has a word for in every
        # language, so the reading comes from the grammar and not from here.
        spelled = spelled.replace("/", "÷")
        if is_digit:
            # A digit replacing a digit, so it joins the run it was already in:
            # `5०3` is five hundred and three, one number, and spacing it would
            # read it as three.
            out.append(spelled)
            continue
        if out and out[-1][-1:] and _is_word_char(out[-1][-1]):
            out.append(" ")
        out.append(spelled)
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if nxt and _is_word_char(nxt):
            out.append(" ")
    return "".join(out)


def _folded_numeral(ch: str) -> tuple[str, bool] | None:
    """``(what it reads as, whether it is a decimal digit)``, or ``None``.

    See ``docs/design/text-funnel.md``.
    """
    code = ord(ch)
    if 0x30 <= code <= 0x39:
        return None
    table = _numerals()
    spelled = table.spelled.get(code)
    if spelled is not None:
        return spelled, False
    zeros = table.decimal_zeros
    index = bisect.bisect_right(zeros, code) - 1
    if index >= 0:
        offset = code - zeros[index]
        if 0 <= offset <= 9:
            return str(offset), True
    return None


@dataclass(frozen=True, slots=True)
class _Numerals:
    """The generated fold table: block zeros, and every spelled numeral."""

    decimal_zeros: tuple[int, ...]
    spelled: Mapping[int, str]


@lru_cache(maxsize=1)
def _numerals() -> _Numerals:
    """`numerals.json`, parsed once."""
    raw = json.loads(NUMERALS_PATH.read_text(encoding="utf-8"))
    return _Numerals(
        decimal_zeros=tuple(raw["decimal_zeros"]),
        spelled={int(k): v for k, v in raw["spelled"].items()},
    )


WHITE_SPACE = (
    "\u0009\u000a\u000b\u000c\u000d\u0020\u0085\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)
"""Unicode White_Space, written out, for every regex in this funnel.

See ``docs/design/text-funnel.md``.
"""

_SPACE = f"[{re.escape(WHITE_SPACE)}]"
"""``\\s`` as this funnel means it: the characters above and nothing else.

Python's ``\\s`` additionally matches U+001C to U+001F, which Unicode does not
call whitespace and which no port matches. Written as a class so the five
implementations read the same character set.
"""

_DIGIT = "[0-9]"
"""``\\d`` as this funnel means it.

Python and Rust read ``\\d`` as every Unicode decimal digit; Go's RE2 reads it
as ASCII. Only ASCII digits reach the number grammars, so the class is spelled
out and the five agree by construction rather than by coincidence.
"""

_OPERATOR_RUN = re.compile(rf"(?<={_SPACE})(<=|>=|!=|==|<|>)(?={_SPACE})")
"""An ASCII operator with whitespace on both sides.

The spacing is the evidence that the mark is an operator and not markup or an
emoticon: `<p>`, `</div>`, `<3` and `a<b` all keep the mark written, and the
funnel leaves written what it cannot read.
"""


def _drop_footnote_markers(text: str) -> str:
    if "[" not in text:
        return text
    # `[0-9]` and the explicit space class, not `\d` and `\s`: RE2 reads both
    # as ASCII and Python reads both as Unicode, so a marker separated by a
    # non-breaking or thin space, ordinary French and German typography,
    # survived in Go and was then *read aloud* by the number pass.
    return re.sub(rf"\[[0-9{re.escape(WHITE_SPACE)},;\-–—]{{1,20}}\]", "", text)


_NOT_SPACE_IN_THE_PORTS = "\u001c\u001d\u001e\u001f"
"""The four characters `str.isspace()` calls whitespace and Unicode does not.

See ``docs/design/text-funnel.md``.
"""


def _is_space(sc: str) -> bool:
    """Unicode White_Space, which is what the other four implementations use."""
    return sc.isspace() and sc not in _NOT_SPACE_IN_THE_PORTS


def _punctuation_for_speech(text: str) -> str:
    out: list[str] = []
    scalars = list(text)
    for i, sc in enumerate(scalars):
        if sc.isalpha() or sc.isdecimal() or _is_space(sc) or sc in _PROSODIC:
            out.append(sc)
            continue
        prev = scalars[i - 1] if i > 0 else None
        nxt = scalars[i + 1] if i + 1 < len(scalars) else None
        # Between digits, "." and "," are numeric separators and "-" and "/"
        # are ranges and fractions, meaning, not decoration.
        between_digits = (
            prev is not None and prev.isdecimal() and nxt is not None and nxt.isdecimal()
        )
        if between_digits and sc in "-/:.":
            out.append(sc)
            continue
        # A hyphen inside a token is part of the token ("well-known", "1e-3").
        if (
            sc in "-+"
            and prev is not None
            and prev.isalnum()
            and nxt is not None
            and nxt.isalnum()
        ):
            out.append(sc)
            continue
        out.append(" ")
    return "".join(out)


def speech_text(text: str, language_id: str | None) -> str:
    """The one place text becomes something the engine is handed.

    See ``docs/design/text-funnel.md``.
    """
    lang = language_id.lower() if language_id else language_id
    # Every pass that needs a grammar reads English when the caller named no
    # language. `_speak_symbols` and `lexical_respelling` take the raw tag
    # instead: both have their own answer for "no language given".
    tagged = lang or "en"
    # NFC first, before anything inspects a character.
    out = unicodedata.normalize("NFC", text)
    # Beside NFC because it is the same kind of pass: one spelling for every
    # pass that follows. Before `_speak_symbols`, so the folded percent sign
    # reaches the table that turns it into a word.
    out = fold_foreign_digits(out, tagged)
    out = _strip_invisibles(out)
    # Before the symbol pass, which would otherwise read a tag's angle brackets
    # as comparison operators and its attributes as text.
    out = _drop_markup_tags(out)
    # Before the symbol table, not after.
    out = fold_numerals(out)
    out = _speak_symbols(out, lang)
    out = _drop_footnote_markers(out)
    # Before the acronym pass, which spells a Roman numeral letter by letter,
    # and after the numeral fold, which is what turns `Ⅳ` into the `IV` this
    # pass reads.
    out = expand_roman_numerals(out, tagged)
    # Numbers after footnotes (a dropped [12] must not become words first) and before
    # punctuation, which would turn a decimal separator into a space and leave "3.5" as
    # two numbers.
    out = spell_acronyms(out, tagged)
    out = expand_dates(out, tagged)
    # Ordinals before numbers, for the same reason dates go before both: the
    # number pass expands the digits and leaves the suffix stuck to them, so
    # "1st" arrived as *onest*.
    out = expand_ordinals(out, tagged)
    out = expand_abbreviations(out, tagged)
    out = expand_times(out, tagged)
    out = expand_numbers(out, tagged)
    out = _punctuation_for_speech(out)
    out = lexical_respelling(out, lang)
    out = re.sub(r"[ \t]{2,}", " ", out)
    # A symbol that became a comma must not keep the space in front of it
    # ("0.49 → 0.24" would otherwise read "zero point four nine ,").
    out = re.sub(rf"[{re.escape(WHITE_SPACE)}]+([.,;:!?])", r"\1", out)
    # A run of clause marks is one clause mark.
    out = re.sub(rf"([.,;:])(?:[{re.escape(WHITE_SPACE)}]*[.,;:])+", r"\1", out)
    return out.strip()


# -------------------------------------------------------- lexical respelling


@dataclass(frozen=True, slots=True)
class _PolishRespell:
    """``pl_en_respell.json``, parsed and shape-checked once."""

    payload: dict[str, object]
    generated: dict[str, str]
    respell_all: dict[str, str]
    words: frozenset[str]
    polish: frozenset[str]


def _respell_object(payload: Mapping[str, object], key: str) -> dict[str, str]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"pl_en_respell.json: {key!r} must be a JSON object")
    return value


def _respell_array(payload: Mapping[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"pl_en_respell.json: {key!r} must be a JSON array")
    return value


@lru_cache(maxsize=1)
def _polish_respell() -> _PolishRespell:
    """``pl_en_respell.json``, read once, lazily.

    Each of the four values is checked for its container shape, which is what
    a truncated or half-written file gets wrong, and the message names the key.
    Their 400,000 elements are not walked one by one: that costs 65 ms against
    a 99 ms parse, on every process that says a Polish word.

    Both sets are built here rather than on first use. They are consulted per
    word of every Polish passage, so rebuilding either per call is quadratic in
    words times lexicon size, and any passage that reaches one reaches all four.
    """
    loaded = json.loads(RESPELL_PATH.read_text(encoding="utf-8"))
    # Raised, not asserted: the file is data, and `python -O` would strip the
    # check and fail later inside a lookup.
    if not isinstance(loaded, dict):
        raise ValueError("pl_en_respell.json must be a JSON object")
    return _PolishRespell(
        payload=loaded,
        generated=_respell_object(loaded, "respell"),
        respell_all=_respell_object(loaded, "respellAll"),
        words=frozenset(_respell_array(loaded, "words")),
        polish=frozenset(_respell_array(loaded, "polish")),
    )


class PolishLexicon:
    """The generated long tail: CMUdict → Polish orthography, ~110k words.

    ``tools/gen_pl_respell.py`` produces the JSON with the common-Polish gate
    baked in; the curated lexicon in this module always wins (its forms were
    approved by ear). A named door onto :func:`_polish_respell`, which holds
    the one copy and the loading rules.
    """

    @classmethod
    def payload(cls) -> dict[str, object]:
        return _polish_respell().payload

    @classmethod
    def generated(cls) -> dict[str, str]:
        return _polish_respell().generated

    @classmethod
    def words(cls) -> frozenset[str]:
        """The English word list."""
        return _polish_respell().words

    @classmethod
    def respell_all(cls) -> dict[str, str]:
        return _polish_respell().respell_all

    @classmethod
    def polish(cls) -> frozenset[str]:
        """Words that are Polish and must not be respelled."""
        return _polish_respell().polish


@dataclass(frozen=True, slots=True)
class _PolishRules:
    """``pl_respell_rules.json``, parsed and shape-checked once."""

    phrases: tuple[tuple[str, str], ...]
    """Multi-word anglicisms respelled as a unit, BEFORE the word pass. Ordered,
    and applied in the file's order: "release notes" word by word would read
    "notes" as the Polish homograph (the notebook), which must stay Polish
    inside the phrase."""
    lexicon: Mapping[str, str]
    """The curated lexicon: common anglicisms to their Polish phonetic
    respelling. Every entry is there because the grapheme reading audibly fails
    and the respelling is the accepted spoken form, so words Poles already read
    correctly by Polish rules (laptop, internet, blog, film) are deliberately
    absent. It always wins over the generated long tail, whose forms nobody
    approved by ear."""
    keep_polish: frozenset[str]
    """English words that are ALSO everyday Polish words. The word pass leaves
    them alone and only a phrase may respell them. Two families: Polish
    homographs, and loanwords Poles read ORTHOGRAPHICALLY ("bug" is [bug] in
    Polish mouths, never [bag])."""
    function_words: frozenset[str]
    """Polish function words that happen to spell English words ("i" = I, "to" =
    to, "on" = on). Never members of an English span, or a span eats the Polish
    conjunction after it."""
    endings: frozenset[str]
    """The Polish case and derivation endings these loanwords actually take."""


_PHRASE_PAIR = 2
"""A phrase entry is written form and spoken form, and nothing else."""


def _rules_array(payload: Mapping[str, object], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"pl_respell_rules.json: {key!r} must be a JSON array")
    return value


@lru_cache(maxsize=1)
def _polish_rules() -> _PolishRules:
    """``pl_respell_rules.json``, read once, lazily.

    The hand-written half of Polish respelling: the phrases, the curated
    lexicon, and the three word lists the inflection walk consults. Data rather
    than literals for the same reason the number grammars are, so that five
    implementations read one file and a word approved by ear is approved once.

    Not in :func:`~loudkit.frontend.textconfig.grammar_digest`, which is a gap:
    an edit here changes spoken words under the same sixteen hex digits unless
    ``FUNNEL_PORTED`` is bumped by hand. Adding the file to the digest moves the
    digest, so it waits for a release that is allowed to move it.
    """
    loaded = json.loads(PL_RULES_PATH.read_text(encoding="utf-8"))
    # Raised, not asserted: the file is data, and `python -O` would strip the
    # check and fail later inside a lookup.
    if not isinstance(loaded, dict):
        raise ValueError("pl_respell_rules.json must be a JSON object")
    lexicon = loaded.get("lexicon")
    if not isinstance(lexicon, dict):
        raise ValueError("pl_respell_rules.json: 'lexicon' must be a JSON object")
    phrases = _rules_array(loaded, "phrases")
    if not all(isinstance(pair, list) and len(pair) == _PHRASE_PAIR for pair in phrases):
        raise ValueError("pl_respell_rules.json: each phrase must be [written, spoken]")
    return _PolishRules(
        phrases=tuple((str(pair[0]), str(pair[1])) for pair in phrases),
        lexicon=lexicon,
        keep_polish=frozenset(_rules_array(loaded, "keep_polish")),
        function_words=frozenset(_rules_array(loaded, "function_words")),
        endings=frozenset(_rules_array(loaded, "endings")),
    )


_MAX_ACRONYM = 5
"""How long an all-caps token may be before this pass stops calling it an acronym.

`letters.spell_acronym` reads a listed word acronym of any length as a word,
which is the right answer where it is asked, at the top of the funnel with the
neighbouring capitals still visible. Down here the only question is whether the
English-run detector should skip the token, and a long all-caps run is a
heading or a shout far more often than an initialism."""


def _spelled_acronym(word: str) -> str | None:
    """The acronym reading the English-run gate below asks about.

    `letters` owns the spelling for all twelve languages; this adds the length
    cap that makes the question a narrower one.
    """
    if len(word) > _MAX_ACRONYM:
        return None
    return spell_acronym(word, "pl")


@lru_cache(maxsize=1)
def _digit_words() -> Mapping[str, str]:
    """The ten ASCII digits as Polish words, for reading a token one character
    at a time.

    ASCII only, and a table rather than ten `cardinal` calls per token, because
    the callers below ask "is this character a digit I can say" of every
    character of every word. `str.isdecimal` is true of `٥` and `൬` as well, and a
    token holding one is not a token this pass says character by character.
    """
    return {str(digit): cardinal(digit, "pl") for digit in range(10)}


_MAX_SPOKEN_DIGITS = 6
"""How many digits a bare token may carry and still be read as one number.

Past six the run is an identifier, an order number or a phone number far more
often than a quantity, and *nine hundred eighty seven thousand six hundred
fifty four* is a worse reading of `9876543210` than the digits themselves."""


def _number_words(token: str) -> str | None:
    """A run of digits as Polish cardinal words, or ``None`` to leave it alone.

    The grammar is `numbers.cardinal`'s; what this adds is the two refusals the
    respelling pass makes on top of it, a leading zero and a run too long to be
    a quantity, both of which read better digit by digit.
    """
    if len(token) > _MAX_SPOKEN_DIGITS or (token.startswith("0") and token != "0"):
        return None
    # `isdecimal`, not `isdigit`: the latter is true of `²` and `③`, which
    # `int()` then refuses, a token this function accepted and could not
    # convert. Nd is exactly what `int()` reads.
    if not token.isdecimal():
        return None
    return cardinal(int(token), "pl")


@lru_cache(maxsize=1)
def _code_letter_names() -> Mapping[str, str]:
    """The letter names a mixed letter-digit token is spelled with.

    The names are the grammar file's, so `GPT` is *gie-pe-te* in one place. The
    ASCII narrowing is this pass's own: `numbers.json` also names the nine
    Polish letters, and reading them here would turn `Ż1`, left written today,
    into *żet jeden*. That is a change in what Polish says, so it belongs to a
    `FUNNEL_PORTED` bump and not to a table move.
    """
    names = {ch: letter_name(ch, "pl") for ch in ascii_lowercase}
    return {ch: name for ch, name in names.items() if name is not None}


_MAX_SPELLED_CODE = 8
"""How long a mixed letter-digit token may be before it is left written.

Spelling `R2` character by character is how a Polish reader says it; doing the
same to an eleven-character identifier is a wall of letter names no listener follows.
Past this length the token is left as written, which is what the number pass
does with the same input for the same reason.
"""


def _spelled_code_token(word: str) -> str | None:
    """`R2` as *er dwa*, or ``None`` when the token cannot be spelled whole.

    See ``docs/design/text-funnel.md``.
    """
    has_letter = any(ch.isalpha() for ch in word)
    has_digit = any(ch.isdecimal() for ch in word)
    if not (has_letter and has_digit):
        return None
    if len(word) > _MAX_SPELLED_CODE:
        return None
    digits = _digit_words()
    letters = _code_letter_names()
    parts: list[str] = []
    for ch in word:
        if ch in digits:
            parts.append(digits[ch])
            continue
        name = letters.get(ch.lower())
        if name is None:
            # `ü`, `ż`, `é`, a letter this table has no name for. Refusing the
            # whole token is the only answer that does not delete it silently.
            return None
        parts.append(name)
    if not parts:
        return None
    return " ".join(parts)


def _lookup(word: str) -> str | None:
    rules = _polish_rules()
    if word in rules.lexicon:
        return rules.lexicon[word]
    if word in rules.keep_polish:
        return None
    return PolishLexicon.generated().get(word)


def _match_case(original: str, respelled: str) -> str:
    if not original or not original[0].isupper():
        return respelled
    return respelled[:1].upper() + respelled[1:]


def _respelled(word: str) -> str:  # noqa: PLR0911, one chain, kept diffable against the ports
    # No acronym branch here. `loudkit.frontend.letters.spell_acronyms` owns that
    # decision for all twelve languages and takes it earlier in the funnel, where
    # the surrounding capitals are still visible. This pass sees one word at a
    # time and so cannot tell an initialism from a shout: it would spell
    # "THIS IS FINE" as te-ha-i-es i-es ef-i-en-e.
    code = _spelled_code_token(word)
    if code:
        return code
    lower = word.lower()
    hit = _lookup(lower)
    if hit:
        return _match_case(word, hit)
    # Digits-only tokens: cardinal words when sane, digit-by-digit when weird
    # (leading zeros, longer than six digits).
    if not any(ch.isalpha() for ch in word):
        said = _number_words(word)
        if said:
            return said
        # Digit by digit, but never to *nothing*.
        digits = _digit_words()
        return " ".join(digits.get(ch, ch) for ch in word)
    # Nothing under three letters declines from a dictionary stem, and Polish
    # is full of one-letter words ("i", "w", "z").
    if len(lower) <= 3:
        return word
    # A word the Polish frequency list knows is POLISH: hands off.
    if lower in PolishLexicon.polish():
        return word
    # Inflected: longest dictionary stem + a known Polish ending, with or
    # without the apostrophe ("deadline'u", "maila", "updatem").
    for cut in range(len(lower) - 1, 1, -1):
        stem = lower[:cut]
        suffix = lower[cut:]
        if suffix.startswith(("'", "’")):
            suffix = suffix[1:]
        hit = _lookup(stem) or _lookup(stem + "e")  # silent-e: "update"→"updatem"
        if hit is None or suffix not in _polish_rules().endings:
            continue
        base = hit
        # The respelling's trailing vowel folds into a vowel-initial ending
        # ("dedlajn" + "u", but "miting" + "u", only vowels collide).
        if base and base[-1] in "aeiouy" and suffix and suffix[0] in "aeiouy":
            base = base[:-1]
        return _match_case(word, base + suffix)
    return word


def lexical_respelling(text: str, language_id: str | None) -> str:
    """Respell ``text`` for the given language. Only Polish has a lexicon
    today; every other language returns the text untouched.

    Case-insensitive on the language id: the frontend lowercases its tag, so a
    caller passing ``"PL"`` would otherwise get Polish tokenisation without
    Polish respelling.
    """
    if language_id is None or language_id.lower() != "pl":
        return text
    out = _respell_symbols(text)
    out = _respell_phrases(out)
    return _respell_words(out)


def _respell_symbols(text: str) -> str:
    """Math and unit symbols the model cannot say, as Polish words, with
    context guards, because "-" is also a hyphen and "/" is also a path.

    Reachable only through a direct :func:`lexical_respelling` call. Inside
    :func:`speech_text` every rule is already spent: ``_speak_symbols`` has
    taken ``%`` and ``°``, ``expand_numbers`` has turned the digits the
    ``(?<=\\d)`` guards need into words, and ``_punctuation_for_speech`` has
    made a space of ``= + < > - / * ^``. Measured over 7,746 funnel inputs it
    changed nothing; called directly it rewrites all of them. Moving the pass
    earlier changes what Polish says and bumps ``FUNNEL_PORTED``, so it is a
    0.1.2 decision, not a 0.1.1 one.
    """
    out = text
    rules = [
        (rf"(?<={_DIGIT}){_SPACE}?%", " procent"),
        (rf"(?<={_DIGIT}){_SPACE}?°C", " stopni Celsjusza"),
        (rf"(?<={_DIGIT}){_SPACE}?°", " stopni"),
        (rf"(?<={_DIGIT}){_SPACE}*/{_SPACE}*(?={_DIGIT})", " przez "),
        (rf"(?<={_DIGIT}){_SPACE}*\*{_SPACE}*(?={_DIGIT})", " razy "),
        (rf"(?<={_DIGIT}){_SPACE}*\^{_SPACE}*(?={_DIGIT})", " do potęgi "),
        (rf"{_SPACE}={_SPACE}", " równa się "),
        (rf"{_SPACE}\+{_SPACE}", " plus "),
        (rf"{_SPACE}<{_SPACE}", " mniejsze niż "),
        (rf"{_SPACE}>{_SPACE}", " większe niż "),
        (rf"{_SPACE}-{_SPACE}", " minus "),
    ]
    for pattern, replacement in rules:
        out = re.sub(pattern, replacement, out)
    return out


def _respell_phrases(text: str) -> str:
    out = text
    for phrase, spoken in _polish_rules().phrases:
        out = re.sub(phrase, spoken, out, flags=re.IGNORECASE)
    return out


def _respell_words(text: str) -> str:  # noqa: PLR0912, PLR0915, one decision chain,
    # kept diffable against the four ports. Split it here and not there and a
    # divergence stops showing up as a diff, which is how it would be found.
    # words[i] with seps[i+1] after it; seps[0] is anything before the first
    # word. Clean alternation.
    words: list[str] = []
    seps: list[str] = [""]
    in_word = False
    for ch in text:
        # The apostrophe stays inside the word: "deadline'u" is one token to a Polish
        # reader and its ending must survive the respelling.
        if ch.isalpha() or ch.isdecimal() or ch in "'’":
            if not in_word:
                words.append("")
                in_word = True
            words[-1] += ch
        else:
            if in_word:
                seps.append("")
                in_word = False
            seps[-1] += ch
    if in_word:
        seps.append("")

    def is_digits(w: str) -> bool:
        return bool(w) and all(ch.isdecimal() for ch in w)

    # A RUN of English words is a quotation, not code-switching: four or more
    # in a row read better with the real English lexicon than as four Polish
    # transliterations. Short bursts stay with the lexicon.
    rules = _polish_rules()
    is_english = []
    for word in words:
        lower = word.lower()
        is_english.append(
            _spelled_acronym(word) is None
            and lower not in rules.keep_polish
            and lower not in rules.function_words
            and (_lookup(lower) is not None or lower in PolishLexicon.words())
        )

    out = seps[0]
    i = 0
    while i < len(words):
        # A run of digit groups chained by "." or ",", the collector split "2.5" into
        # two tokens around the point, and "192.168.0.1" into four.
        if _spelled_code_token(words[i]) is not None:
            end = i
            while (
                end + 1 < len(words)
                and seps[end + 1] in (".", ",")
                and (is_digits(words[end + 1]) or _spelled_code_token(words[end + 1]))
            ):
                end += 1
            if end > i:
                for k in range(i, end + 1):
                    out += words[k] + seps[k + 1]
                i = end + 1
                continue

        if is_digits(words[i]):
            end = i
            while (
                end + 1 < len(words)
                and is_digits(words[end + 1])
                and seps[end + 1] in (".", ",")
            ):
                end += 1
            groups = end - i + 1
            if groups >= 3:
                for k in range(i, end + 1):
                    out += words[k] + seps[k + 1]
                i = end + 1
                continue
            if groups == 2:
                digits = _digit_words()
                whole = _number_words(words[i]) or words[i]
                frac = " ".join(digits[ch] for ch in words[i + 1] if ch in digits)
                out += whole + " przecinek " + frac + seps[i + 2]
                i += 2
                continue
        if is_english[i]:
            j = i
            while j < len(words) and is_english[j]:
                j += 1
            if j - i >= 4:
                # Inside a detected English span every word transliterates,
                # gate ignored, "brown" alone stays Polish, "brown" inside
                # "the quick brown fox" becomes "brałn".
                for k in range(i, j):
                    lower = words[k].lower()
                    hit = rules.lexicon.get(lower) or PolishLexicon.respell_all().get(lower)
                    out += _match_case(words[k], hit or words[k]) + seps[k + 1]
                i = j
                continue
        out += _respelled(words[i]) + seps[i + 1]
        i += 1
    return out
