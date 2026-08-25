"""The Polish text funnel — a bit-parity port of the Swift engine's SpeechText
and LexicalRespelling.

The shipped engine reads Polish text through a two-stage funnel before
tokenising: :func:`speech_text` scrubs the raw text (invisible characters,
symbols, footnote markers, punctuation), then :func:`lexical_respelling`
rewrites English words embedded in Polish the way a Polish reader says them
("download" → "dałnloud", "deadline'u" → "dedlajnu"), numbers to Polish
cardinals, and acronyms/code tokens to spelled-out Polish letter names.

The engine is grapheme-based with ONE language tag per utterance, so a Polish
render reads "download" with Polish letter-to-sound rules and mangles it.
Respelling won the ear test over inline ``[en]`` tag switching: "dałnloud" is
not a hack, it is how the word actually sounds in a Polish sentence, accent
included. This is the same promise as everywhere else in this library — the
Swift and Python engines must read the same text identically, and this module
is the Python half of that contract.

Dictionary-first and ONLY dictionary: a rule-based English G2P bolted on here
would misfire on real Polish words, and the cost of a false positive (mangling
native text) is far higher than the cost of a miss. Inflections ride as
suffixes: Poles decline these words ("maila", "deadline'u"), so matching is
stem + known Polish ending, with apostrophe forms handled.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from importlib import resources
from typing import cast

from .dates import expand_dates, expand_ordinals
from .letters import spell_acronyms
from .numbers import (
    decimal_separator,
    expand_abbreviations,
    expand_times,
    fold_foreign_digits,
    unit_word,
)
from .numbers import expand as expand_numbers

__all__ = ["speech_text", "lexical_respelling", "PolishLexicon"]

# ------------------------------------------------------------------ funnel

# Zero-width and formatting characters, and the soft hyphen — invisible in
# every editor, and a grapheme engine sees them as letters (the model reads a
# word that does not exist in any text it was trained on).
_INVISIBLES = frozenset("\u200b\u200c\u200d\u2060\ufeff\u00ad\u180e\u200e\u200f")

# Symbols the model cannot voice, as words: (en, pl). The first family
# (→ ✓ ✗ ≈ ≥) is literally outside the vocabulary — the tokenizer emits [UNK]
# and the model receives nothing — while ¢ ° % $ do tokenize and are read at
# the ear's discretion. Both get words.
_SYMBOL_WORDS: Mapping[str, tuple[str, str]] = {
    "%": ("percent", "procent"),
    "°": ("degrees", "stopni"),
    "¢": ("cents", "centów"),
    "€": ("euro", "euro"),
    "£": ("pounds", "funtów"),
    "¥": ("yen", "jenów"),
    "₹": ("rupees", "rupii"),
    "×": ("times", "razy"),
    "÷": ("divided by", "podzielone przez"),
    "≈": ("about", "około"),
    "≥": ("at least", "co najmniej"),
    "≤": ("at most", "najwyżej"),
    "≠": ("not equal to", "różne od"),
    "±": ("plus minus", "plus minus"),
    "→": (",", ","),
    "←": (",", ","),
    "⇒": (",", ","),
    "✓": ("yes", "tak"),
    "✔": ("yes", "tak"),
    "✗": ("no", "nie"),
    "✘": ("no", "nie"),
    "•": (",", ","),
    "·": (",", ","),
    "▪": (",", ","),
    "◦": (",", ","),
    "…": ("...", "..."),
    "&": ("and", "i"),
    "@": ("at", "małpa"),
}

# `$` and `£` before a number read as a prefix in writing and a SUFFIX in
# speech: "$5" is "five dollars", not "dollars five". The wording comes from
# `loudkit.numbers.unit_word`; this set only says which symbols are written
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
_PROSODIC = frozenset(".,!?;:\u2014\u2013\u2026\"\u201c\u201d\u201e«»()'\u2019")


def _strip_invisibles(text: str) -> str:
    if not any(sc in _INVISIBLES for sc in text):
        return text
    return "".join(ch for ch in text if ch not in _INVISIBLES)


def _priced(amount: str, language: str) -> str:
    """A currency amount, with its decimal mark spelled the way this language does.

    The one place a dot between digits is *known* not to be a clock time, and
    the last place that knows it. `$0.49` reads as a price in every language on
    earth, but by the time the funnel reaches `expand_times` the symbol has
    already become a trailing word and the dot is indistinguishable from the one
    in `14.30` — which in the eleven comma-decimal languages *is* how a time is
    written. So German answered "null Uhr neunundvierzig Dollar": zero o'clock
    forty-nine dollars.

    Rewriting the separator here, where the currency symbol is still in hand, is
    what removes the ambiguity rather than adding a rule that tries to guess it
    back later. Only a lone dot with a plain fraction is touched — `$1,234.56`
    carries a grouping mark this cannot safely reinterpret, and leaving it is the
    same refusal the rest of this module makes when evidence runs out.
    """
    separator = decimal_separator(language)
    if separator == ".":
        return amount
    if re.fullmatch(r"\d+\.\d+", amount):
        return amount.replace(".", separator)
    return amount


def _speak_symbols(text: str, lang: str | None) -> str:
    """Symbols become words in the render's own language.

    This pass owns *where* a word goes — a currency mark is written before its
    amount and spoken after it — and asks ``loudkit.numbers`` *which* word,
    because which word is a per-language fact. A two-language table would
    render "$5" in German as "5 dollars": the wording must cover every
    shipped language.

    A language without a wording table falls back to English rather than to
    silence — the symbol is at least said, if with an accent.
    """
    out = text
    language = lang if lang and unit_word("%", lang) else "en"

    # Prefix currencies first, while the digits still follow the symbol. The
    # number, and NOT the sentence punctuation behind it: a greedy [\d.,]*
    # would swallow the comma in "£250,".
    for symbol in _CURRENCY_PREFIXES:
        word = unit_word(symbol, language)
        if word is None:
            continue
        # `(?<![^\W\d_])` — not preceded by a letter. `R$` is the Brazilian
        # real, `HK$` the Hong Kong dollar, `NT$` the Taiwan dollar, and this
        # table has a wording for none of them. Matching the `$` alone read
        # `R$3,14` as "R3,14 Dollar": the wrong currency, said confidently, with
        # the orphaned `R` still in front of it. A multi-character mark this
        # module cannot name is left written, which is the same answer it gives
        # everywhere else when the evidence runs out.
        pattern = r"(?<![^\W\d_])" + re.escape(symbol) + r"\s?(\d+(?:[.,]\d+)*)"

        # `word=word` binds the loop variable rather than closing over it.
        # `re.sub` runs eagerly so late binding would not bite today, which is
        # exactly the kind of "correct by accident" the next edit breaks.
        def _say_amount(m: re.Match[str], word: str = word) -> str:
            return f"{_priced(m.group(1), language)} {word}"

        out = re.sub(pattern, _say_amount, out)
    # ...and the same amount with the symbol *behind* it. `2.50 €` and `0.49¢`
    # are prices by exactly the evidence `€2.50` is, and were reaching the time
    # pass with the dot intact: German answered "zwei Uhr fünfzig Euro". The
    # prefix pass above could not see them and the generic symbol loop below
    # replaces the mark without ever looking at the number.
    #
    # Currency written as a *word* — `5.50 zł`, `12.30 kr` — is not covered.
    # Those are ordinary words to every pass here, and telling them from a unit
    # or a name needs a per-language lexicon rather than a symbol table.
    for symbol in _CURRENCY_SYMBOLS:
        word = unit_word(symbol, language)
        if word is None or symbol not in out:
            continue
        # No letter guard on this side: only whitespace may sit between the
        # amount and the mark, so `3,14 R$` never matches in the first place.
        pattern = r"(\d+(?:[.,]\d+)*)\s?" + re.escape(symbol)

        def _say_suffix(m: re.Match[str], word: str = word) -> str:
            return f"{_priced(m.group(1), language)} {word}"

        out = re.sub(pattern, _say_suffix, out)
    for symbol, fallback in _SYMBOL_WORDS.items():
        if symbol not in out:
            continue
        replacement = unit_word(symbol, language) or (
            fallback[1] if language == "pl" else fallback[0]
        )
        # A word replacement needs spaces around it; a punctuation one must
        # not gain a space BEFORE it or the comma floats.
        spaced = (
            (replacement + " ")
            if len(replacement) == 1 and replacement in ",."
            else (" " + replacement + " ")
        )
        out = out.replace(symbol, spaced)
    return out


def _drop_footnote_markers(text: str) -> str:
    if "[" not in text:
        return text
    return re.sub(r"\[[\d\s,;\-–—]{1,20}\]", "", text)


def _punctuation_for_speech(text: str) -> str:
    out: list[str] = []
    scalars = list(text)
    for i, sc in enumerate(scalars):
        if sc.isalpha() or sc.isdecimal() or sc.isspace() or sc in _PROSODIC:
            out.append(sc)
            continue
        prev = scalars[i - 1] if i > 0 else None
        nxt = scalars[i + 1] if i + 1 < len(scalars) else None
        # Between digits, "." and "," are numeric separators and "-" and "/"
        # are ranges and fractions — meaning, not decoration.
        between_digits = (
            prev is not None and prev.isdecimal() and nxt is not None and nxt.isdecimal()
        )
        if between_digits and sc in "-/:.":
            out.append(sc)
            continue
        # A hyphen inside a token is part of the token ("well-known", "1e-3").
        #
        # Letters on both sides alone leaves the exponent in
        # `1e-3` to become a space: the number pass had already declined to read
        # a token with a letter in it, and then punctuation took it apart anyway,
        # so the model was handed "1e 3". Either end being alphanumeric is the
        # rule that keeps a token whole, and a hyphen between spaces is
        # untouched by both readings.
        # `+` alongside `-`, for the same reason and the same shape. The number pass
        # declines `1e+3` as a token with a letter in it, and then punctuation took the
        # token apart anyway — "1e 3" — because only the hyphen was kept between
        # alphanumerics. A `+` between spaces is still a space in both readings, so
        # nothing that was a separator becomes a character.
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

    Every pass here runs in all five implementations. Dates, ordinals, acronyms
    and NFC were Python-only for a while, gated behind an explicit ``recipe`` so
    the fingerprint could not claim a parity that did not exist; the four ports
    have them now, so the gate is gone and so is the divergence it described.

    Order matters and is deliberate: invisible characters first (they are not
    whitespace by Unicode's rules), symbols that carry meaning become words
    while the digits around them are still intact, footnote markers before
    punctuation rules, punctuation last (prosodic marks stay exactly where
    they are — everything else becomes a space).

    The language id is compared case-insensitively. ``GraphemeTextFrontend``
    lowercases its own tag, so ``"PL"`` produced Polish *tokens* while silently
    skipping the Polish respelling here — the same utterance read half one way
    and half the other, with nothing to indicate it.
    """
    lang = language_id.lower() if language_id else language_id
    # NFC first, before anything inspects a character.
    #
    # Unicode lets the same character arrive two ways: Polish ą as U+0105, or as
    # a + U+0328; Danish å as U+00E5 or a + U+030A. The tokenizer's vocabulary
    # holds one of them. Without this the decomposed spelling reaches it as a
    # base letter followed by an unknown combining mark, and every rule below —
    # every regex, every lexicon lookup, every character class — is matching
    # against a string the author never pictured.
    #
    # It also has to come before `_strip_invisibles`, which removes format
    # characters: normalisation can compose a sequence into a single character,
    # and doing it afterwards would leave that composition unexamined.
    out = unicodedata.normalize("NFC", text)
    # Beside NFC because it is the same kind of pass: one spelling for every
    # pass that follows. Before `_speak_symbols`, so the folded percent sign
    # reaches the table that turns it into a word.
    out = fold_foreign_digits(out, lang or "en")
    out = _strip_invisibles(out)
    out = _speak_symbols(out, lang)
    out = _drop_footnote_markers(out)
    # Numbers after footnotes (a dropped [12] must not become words first) and
    # before punctuation, which would turn a decimal separator into a space and
    # leave "3.5" as two unrelated numbers. The symbol pass has already moved a
    # currency mark behind its amount, so "£250" arrives here as "250 pounds"
    # with only the digits left to say. Wired in the same commit as the four
    # ports' interpreters — a Python-only expansion would make five funnels
    # produce different text while the fingerprint declared them identical.
    # Dates first, and this ordering is the whole reason the pass exists:
    # `12.03.2026` is the ordinary written date of five of these languages, and
    # both passes below want a piece of it. The clock pattern matches `12.03`
    # and the digit run matches the lot — so a date recognised any later has
    # already been eaten and read as a time with a stray year, or as one
    # eight-digit number.
    # Acronyms before dates and numbers, while the capitals are still capitals:
    # every later pass lowercases or rewrites, and a spelled acronym has to be
    # decided while the only evidence — that the word stands alone in caps —
    # still exists. Polish had this and the other eleven languages did not, so
    # `FBI` reached the model as raw graphemes for a grapheme engine to read as
    # a word.
    # Not gated: Swift and Go spell acronyms too — inside their Polish
    # respelling modules rather than as a funnel pass of their own — and the
    # shared fixture pins the Polish result. Gating this here broke a case all
    # five implementations already agreed on. What is still Python-only is the
    # reach: this pass spells acronyms in every language, the ports' respeller
    # only in Polish, so `CIA` in English text is a real remaining divergence
    # and is recorded as one in the fixture's `divergent` block.
    out = spell_acronyms(out, lang or "en")
    out = expand_dates(out, lang or "en")
    # Ordinals before numbers, for the same reason dates go before both: the
    # number pass expands the digits and leaves the suffix stuck to them, so
    # "1st" arrived as *onest*.
    out = expand_ordinals(out, lang or "en")
    out = expand_abbreviations(out, lang or "en")
    out = expand_times(out, lang or "en")
    out = expand_numbers(out, lang or "en")
    out = _punctuation_for_speech(out)
    out = lexical_respelling(out, lang)
    out = re.sub(r"[ \t]{2,}", " ", out)
    # A symbol that became a comma must not keep the space in front of it
    # ("0.49 → 0.24" would otherwise read "zero point four nine ,").
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    # A run of clause marks is one clause mark.
    #
    # Written as a run rather than a pair on purpose: `re.sub` does not overlap
    # its matches, so a pair rule turns "..." into ".." on one pass and "." on
    # the next — a funnel that is not idempotent, and therefore one whose output
    # depends on the pass count, which is not a property of the text. An ellipsis is the
    # common way in; it reaches here as three periods from the symbol map.
    out = re.sub(r"([.,;:])(?:[\s]*[.,;:])+", r"\1", out)
    return out.strip()


# -------------------------------------------------------- lexical respelling


class PolishLexicon:
    """The generated long tail: CMUdict → Polish orthography, ~110k words.

    ``tools/gen_pl_respell.py`` produces the JSON with the common-Polish gate
    baked in; the curated lexicon in this module always wins (its forms were
    approved by ear). Loaded lazily so a non-Polish run never pays for it.
    """

    _payload: dict[str, object] | None = None
    _words: frozenset[str] | None = None
    _polish: frozenset[str] | None = None

    @classmethod
    def payload(cls) -> dict[str, object]:
        if cls._payload is None:
            import json

            ref = resources.files("loudkit.models.data").joinpath("pl_en_respell.json")
            with resources.as_file(ref) as path:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            # Raised, not asserted: the file is data, and `python -O` would
            # strip the check and fail later inside a cast.
            if not isinstance(loaded, dict):
                raise ValueError("pl_en_respell.json must be a JSON object")
            cls._payload = loaded
        return cls._payload

    @classmethod
    def generated(cls) -> dict[str, str]:
        return cast(dict[str, str], cls.payload()["respell"])

    @classmethod
    def words(cls) -> frozenset[str]:
        """The English word list, built once.

        This is consulted **per word** of every Polish passage, so the set is
        built once. Rebuilding it per call is quadratic in the wrong variable —
        words × lexicon size — and runs entirely under the server's
        single-flight lock. The lists are immutable data loaded once; the sets
        over them are too.
        """
        if cls._words is None:
            cls._words = frozenset(cast(list[str], cls.payload()["words"]))
        return cls._words

    @classmethod
    def respell_all(cls) -> dict[str, str]:
        return cast(dict[str, str], cls.payload()["respellAll"])

    @classmethod
    def polish(cls) -> frozenset[str]:
        """Words that are Polish and must not be respelled. Built once — see
        :meth:`words`."""
        if cls._polish is None:
            cls._polish = frozenset(cast(list[str], cls.payload()["polish"]))
        return cls._polish


# Multi-word anglicisms respelled as a unit, BEFORE the word pass — "release
# notes" word-by-word would read "notes" as the Polish homograph (the
# notebook) and must stay Polish inside the phrase.
_PHRASES: list[tuple[str, str]] = [
    ("release notes", "rilis nołc"),
    ("pull request", "pul rekłest"),
    ("code review", "koud riwju"),
    ("open source", "oupen sors"),
    ("happy hour", "hepi ałer"),
]

# English words that are ALSO everyday Polish words — the word pass leaves
# them alone, and only a phrase above may respell them. Two families: Polish
# homographs, and loanwords Poles read ORTHOGRAPHICALLY ("bug" is [bug] in
# Polish mouths, never [bag]).
_KEEP_POLISH = frozenset(
    [
        "notes",
        "pilot",
        "problem",
        "prom",
        "kit",
        "bug",
        "buga",
        "bugi",
        "bugach",
        "bugów",
        "log",
        "logi",
        "logach",
        "spam",
        "port",
        "host",
        "linux",
        "unix",
        "python",
        "ruby",
    ]
)

# GPT → "gie-pe-te": an all-caps token is read letter by letter with POLISH
# letter names. A short allowlist covers acronyms said as WORDS (NASA, RAM).
_LETTER_NAMES: Mapping[str, str] = {
    "a": "a",
    "b": "be",
    "c": "ce",
    "d": "de",
    "e": "e",
    "f": "ef",
    "g": "gie",
    "h": "ha",
    "i": "i",
    "j": "jot",
    "k": "ka",
    "l": "el",
    "m": "em",
    "n": "en",
    "o": "o",
    "p": "pe",
    "q": "ku",
    "r": "er",
    "s": "es",
    "t": "te",
    "u": "u",
    "v": "fał",
    "w": "wu",
    "x": "iks",
    "y": "igrek",
    "z": "zet",
}

_WORD_ACRONYMS = frozenset(
    ["nasa", "ram", "rom", "pin", "vat", "sim", "lot", "pesel", "nato", "zus", "nfz", "pit"]
)

# Polish function words that happen to spell English words ("i" = I, "to" =
# to, "on" = on) — never span members, or a span eats the Polish conjunction
# after it.
_POLISH_FUNCTION_WORDS = frozenset(
    [
        "i",
        "a",
        "o",
        "u",
        "w",
        "z",
        "no",
        "to",
        "ta",
        "ten",
        "on",
        "ona",
        "my",
        "ja",
        "do",
        "po",
        "za",
        "na",
        "od",
        "ale",
        "czy",
        "tak",
        "nie",
        "co",
        "jak",
        "go",
        "mu",
        "je",
        "ma",
        "by",
        "się",
        "był",
        "mam",
        "dam",
    ]
)

# Polish case/derivation endings these loanwords actually take.
_POLISH_ENDINGS = frozenset(
    [
        "a",
        "u",
        "e",
        "y",
        "i",
        "em",
        "ie",
        "ę",
        "ą",
        "om",
        "ach",
        "ami",
        "ów",
        "owi",
        "cie",
        "sie",
        "owa",
        "owe",
        "ować",
        "uje",
        "ujesz",
        "ujemy",
        "ujecie",
        "ują",
    ]
    + ["owy", "owych", "owego", "owym"]
)

_UNITS = ["", "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem", "osiem", "dziewięć"]
_TEENS = [
    "dziesięć",
    "jedenaście",
    "dwanaście",
    "trzynaście",
    "czternaście",
    "piętnaście",
    "szesnaście",
    "siedemnaście",
    "osiemnaście",
    "dziewiętnaście",
]
_TENS = [
    "",
    "",
    "dwadzieścia",
    "trzydzieści",
    "czterdzieści",
    "pięćdziesiąt",
    "sześćdziesiąt",
    "siedemdziesiąt",
    "osiemdziesiąt",
    "dziewięćdziesiąt",
]
_HUNDREDS = [
    "",
    "sto",
    "dwieście",
    "trzysta",
    "czterysta",
    "pięćset",
    "sześćset",
    "siedemset",
    "osiemset",
    "dziewięćset",
]

_DIGIT_WORDS: Mapping[str, str] = {
    "0": "zero",
    "1": "jeden",
    "2": "dwa",
    "3": "trzy",
    "4": "cztery",
    "5": "pięć",
    "6": "sześć",
    "7": "siedem",
    "8": "osiem",
    "9": "dziewięć",
}

# The curated lexicon: common anglicisms → Polish phonetic respelling.
# Every entry was chosen because the grapheme reading audibly fails and the
# respelling is the accepted spoken form. Words Poles already read correctly
# by Polish rules (laptop, internet, blog, film) are deliberately absent.
_LEXICON: Mapping[str, str] = {
    # computing, the reason this file exists
    "youtube": "jutjub",
    "github": "githab",
    "seek": "sik",
    "utf": "u te ef",
    "pbcopy": "pi bi kopi",
    "pbpaste": "pi bi pejst",
    "json": "dżejson",
    "jsonl": "dżejson el",
    "ffmpeg": "ef ef em peg",
    "npm": "en pe em",
    "sudo": "sudo",
    "ssh": "es es ha",
    "html": "ha te em el",
    "css": "ce es es",
    "sql": "es ku el",
    "chatgpt": "czat dżi pi ti",
    "download": "dałnloud",
    "downloads": "dałnloudy",
    "upload": "aploud",
    "update": "apdejt",
    "upgrade": "apgrejd",
    "backup": "bekap",
    "online": "onlajn",
    "offline": "oflajn",
    "email": "imejl",
    "mail": "mejl",
    "gmail": "dżimejl",
    "browser": "brałzer",
    "cache": "kesz",
    "chat": "czat",
    "cloud": "klałd",
    "code": "koud",
    "commit": "komit",
    "cookie": "kuki",
    "cookies": "kukis",
    "deadline": "dedlajn",
    "debug": "dibag",
    "default": "difolt",
    "desktop": "desktop",
    "developer": "dewełoper",
    "device": "diwajs",
    "display": "displej",
    "drive": "drajw",
    "driver": "drajwer",
    "feature": "ficzer",
    "feedback": "fidbek",
    "firmware": "firmłer",
    "framework": "frejmłork",
    "freelancer": "frilanser",
    "hardware": "hardłer",
    "software": "softłer",
    "homepage": "houmpejdż",
    "interface": "interfejs",
    "iphone": "ajfon",
    "ipad": "ajpad",
    "mac": "mak",
    "macbook": "makbuk",
    "level": "lewel",
    "manager": "menedżer",
    "meeting": "miting",
    "notebook": "noutbuk",
    "notification": "notyfikacja",
    "open": "oupen",
    "phone": "foun",
    "release": "rilis",
    "review": "riwju",
    "screen": "skrin",
    "screenshot": "skrinszot",
    "server": "serwer",
    "share": "szer",
    "smartphone": "smartfon",
    "stream": "strim",
    "streaming": "striming",
    "streamer": "strimer",
    "team": "tim",
    "timeline": "tajmlajn",
    "touchpad": "taczpad",
    "voucher": "wałczer",
    "wallpaper": "łolpejper",
    "website": "łebsajt",
    "wifi": "łajfaj",
    "workflow": "łorkfloł",
    "workshop": "łorkszop",
    # everyday code-switching
    "business": "biznes",
    "brunch": "brancz",
    "budget": "badżet",
    "case": "kejs",
    "challenge": "czalendż",
    "coach": "koucz",
    "cool": "kul",
    "crush": "krasz",
    "design": "dizajn",
    "designer": "dizajner",
    "fake": "fejk",
    "fashion": "faszyn",
    "game": "gejm",
    "gamer": "gejmer",
    "influencer": "influenser",
    "joke": "dżouk",
    "juice": "dżus",
    "lifestyle": "lajfstajl",
    "like": "lajk",
    "lunch": "lancz",
    "mainstream": "mejnstrim",
    "make": "mejk",
    "makeup": "mejkap",
    "nice": "najs",
    "outfit": "ałtfit",
    "please": "pliz",
    "podcast": "podkast",
    "sale": "sejl",
    "shake": "szejk",
    "shopping": "szoping",
    "show": "szoł",
    "size": "sajz",
    "sorry": "sory",
    "ticket": "tiket",
    "trade": "trejd",
    "vibe": "wajb",
    "weekend": "łikend",
    "wow": "łał",
}


def _spelled_acronym(word: str) -> str | None:
    if not 2 <= len(word) <= 5 or not word.isupper() or not word.isalpha():
        return None
    lower = word.lower()
    if lower in _WORD_ACRONYMS:
        return lower
    names = [_LETTER_NAMES.get(c) for c in lower]
    if any(n is None for n in names):
        return None
    return "-".join(n for n in names if n is not None)


def _number_words(token: str) -> str | None:
    if len(token) > 6 or (token.startswith("0") and token != "0"):
        return None
    if not token.isdigit():
        return None
    value = int(token)
    if value == 0:
        return "zero"

    def under1000(n: int) -> list[str]:
        parts: list[str] = []
        if n >= 100:
            parts.append(_HUNDREDS[n // 100])
        rest = n % 100
        if 10 <= rest <= 19:
            parts.append(_TEENS[rest - 10])
        else:
            if rest >= 20:
                parts.append(_TENS[rest // 10])
            if rest % 10 > 0:
                parts.append(_UNITS[rest % 10])
        return parts

    parts: list[str] = []
    thousands = value // 1000
    if thousands > 0:
        if thousands == 1:
            parts.append("tysiąc")
        else:
            parts += under1000(thousands)
            last_two = thousands % 100
            last = thousands % 10
            if 12 <= last_two <= 14:
                parts.append("tysięcy")
            elif 2 <= last <= 4:
                parts.append("tysiące")
            else:
                parts.append("tysięcy")
    parts += under1000(value % 1000)
    return " ".join(parts)


_MAX_SPELLED_CODE = 8
"""How long a mixed letter-digit token may be before it is left written.

Spelling `R2` character by character is how a Polish reader says it; doing the
same to an eleven-character identifier is a wall of letter names no listener follows.
Past this length the token is left as written, which is what the number pass
does with the same input for the same reason.
"""


def _spelled_code_token(word: str) -> str | None:
    """`R2` as *er dwa*, or ``None`` when the token cannot be spelled whole.

    All or nothing: a character with no letter name refuses the token rather
    than skipping silently (a dropped `ü` changes *Müller123* from a name into
    a different name), and the token must fit whole — truncation would drop
    digits from `żelazny2024` without a trace.

    Both are the failure this funnel spent the week closing, one pass further
    on: a token half-read is worse than a token left written, because the
    listener cannot tell that anything was dropped. If every character has a
    name and the token is short enough, it is spelled; otherwise the model gets
    it as written and reads it however it reads it.
    """
    has_letter = any(ch.isalpha() for ch in word)
    has_digit = any(ch.isdigit() for ch in word)
    if not (has_letter and has_digit):
        return None
    if len(word) > _MAX_SPELLED_CODE:
        return None
    parts: list[str] = []
    for ch in word:
        if ch in _DIGIT_WORDS:
            parts.append(_DIGIT_WORDS[ch])
            continue
        name = _LETTER_NAMES.get(ch.lower())
        if name is None:
            # `ü`, `ż`, `é` — a letter this table has no name for. Refusing the
            # whole token is the only answer that does not delete it silently.
            return None
        parts.append(name)
    if not parts:
        return None
    return " ".join(parts)


def _lookup(word: str) -> str | None:
    if word in _LEXICON:
        return _LEXICON[word]
    if word in _KEEP_POLISH:
        return None
    return PolishLexicon.generated().get(word)


def _match_case(original: str, respelled: str) -> str:
    if not original or not original[0].isupper():
        return respelled
    return respelled[:1].upper() + respelled[1:]


def _respelled(word: str) -> str:  # noqa: PLR0911 — the Swift port's decision chain, kept 1:1
    # No acronym branch here any more. `loudkit.letters.spell_acronyms` owns
    # that decision for all twelve languages and takes it earlier in the funnel,
    # where the surrounding capitals are still visible — this pass sees one word
    # at a time and so could not tell an initialism from a shout. It spelled
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
        cardinal = _number_words(word)
        if cardinal:
            return cardinal
        return " ".join(_DIGIT_WORDS[ch] for ch in word if ch in _DIGIT_WORDS)
    # Nothing under three letters declines from a dictionary stem — and Polish
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
        if hit is None or suffix not in _POLISH_ENDINGS:
            continue
        base = hit
        # The respelling's trailing vowel folds into a vowel-initial ending
        # ("dedlajn" + "u", but "miting" + "u" — only vowels collide).
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
    """Math and unit symbols the model cannot say, as Polish words — with
    context guards, because "-" is also a hyphen and "/" is also a path."""
    out = text
    rules = [
        (r"(?<=\d)\s?%", " procent"),
        (r"(?<=\d)\s?°C", " stopni Celsjusza"),
        (r"(?<=\d)\s?°", " stopni"),
        (r"(?<=\d)\s*/\s*(?=\d)", " przez "),
        (r"(?<=\d)\s*\*\s*(?=\d)", " razy "),
        (r"(?<=\d)\s*\^\s*(?=\d)", " do potęgi "),
        (r"\s=\s", " równa się "),
        (r"\s\+\s", " plus "),
        (r"\s<\s", " mniejsze niż "),
        (r"\s>\s", " większe niż "),
        (r"\s-\s", " minus "),
    ]
    for pattern, replacement in rules:
        out = re.sub(pattern, replacement, out)
    return out


def _respell_phrases(text: str) -> str:
    out = text
    for phrase, spoken in _PHRASES:
        out = re.sub(phrase, spoken, out, flags=re.IGNORECASE)
    return out


def _respell_words(text: str) -> str:  # noqa: PLR0912, PLR0915 — a 1:1 port of the
    # Swift decision chain. Splitting it would make the two versions diff-unreadable,
    # which is the property that catches divergences between the two paths.
    # words[i] with seps[i+1] after it; seps[0] is anything before the first
    # word. Clean alternation.
    words: list[str] = []
    seps: list[str] = [""]
    in_word = False
    for ch in text:
        # The apostrophe stays inside the word: "deadline'u" is one token to a
        # Polish reader and its ending must survive the respelling.
        if ch.isalpha() or ch.isdigit() or ch in "'’":
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
        return bool(w) and all(ch.isdigit() for ch in w)

    # A RUN of English words is a quotation, not code-switching: four or more
    # in a row read better with the real English lexicon than as four Polish
    # transliterations. Short bursts stay with the lexicon.
    is_english = []
    for word in words:
        lower = word.lower()
        is_english.append(
            _spelled_acronym(word) is None
            and lower not in _KEEP_POLISH
            and lower not in _POLISH_FUNCTION_WORDS
            and (_lookup(lower) is not None or lower in PolishLexicon.words())
        )

    out = seps[0]
    i = 0
    while i < len(words):
        # A run of digit groups chained by "." or "," — the collector split
        # "2.5" into two tokens around the point, and "192.168.0.1" into four.
        #
        # The whole run is measured before any of it is read, because the
        # decision belongs to the run and not to its first pair. Two groups is
        # a decimal: "dwa przecinek pięć". Three or more is a version, an
        # address or a date, and is left exactly as written — reading the first
        # pair gave "jeden przecinek dwa" with a stray ".trzy" behind it, and
        # skipping only that pair simply moved the same mistake one group along.
        # `numbers.expand` refuses these by the same rule; this pass is a second,
        # independent reader of digits and has to agree with it.
        # The same run, when it starts with a token that has a letter in it.
        # `v1.2.3` collected as ["v1", "2", "3"], and the branch below only
        # starts on a pure-digit group — so the chain was never measured, `v1`
        # went to the code speller and the rest read as numbers: "fał
        # jeden.dwa przecinek trzy". A version is a version whether or not its
        # first group happens to be all digits, and `numbers.expand` already
        # declines the whole token for exactly that reason.
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
                whole = _number_words(words[i]) or words[i]
                frac = " ".join(_DIGIT_WORDS[ch] for ch in words[i + 1] if ch in _DIGIT_WORDS)
                out += whole + " przecinek " + frac + seps[i + 2]
                i += 2
                continue
        if is_english[i]:
            j = i
            while j < len(words) and is_english[j]:
                j += 1
            if j - i >= 4:
                # Inside a detected English span every word transliterates,
                # gate ignored — "brown" alone stays Polish, "brown" inside
                # "the quick brown fox" becomes "brałn".
                for k in range(i, j):
                    lower = words[k].lower()
                    hit = _LEXICON.get(lower) or PolishLexicon.respell_all().get(lower)
                    out += _match_case(words[k], hit or words[k]) + seps[k + 1]
                i = j
                continue
        out += _respelled(words[i]) + seps[i + 1]
        i += 1
    return out
