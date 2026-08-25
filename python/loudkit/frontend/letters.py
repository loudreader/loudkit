"""Acronyms, spelled in the language being read.

``CIA`` is *see-eye-ay* in an English render and *ce-i-a* in a Polish one, and
those are not two spellings of one thing — they are what the two languages
actually say. The engine is grapheme-based with a single language tag per
utterance, so the letter name has to be written in the target language's own
orthography: English ``see`` reads as /siː/ under English letter-to-sound rules,
Polish ``ce`` reads as /t͡sɛ/ under Polish ones, and putting either into the
other's render produces a word in no one's language.

The table is per language: letter names are orthography-specific (Polish
``FBI`` is *ef-be-i*), so a render without its own table reaches the model with
raw graphemes — which a grapheme engine reads as a word-shaped thing rather
than as letters. That is the same
failure the currency wording had before it moved into the shared grammar file,
and it is fixed the same way: the tables are data, one per language, read by
every implementation.

**What is not spelled.** An acronym that is a word in its language stays a word:
``NASA`` and ``NATO`` everywhere, ``SIDA`` and ``OVNI`` in the Romance three,
``PESEL`` and ``ZUS`` in Polish, ``TUTKA`` in Finnish. Those lists are per
language because the fact is: ``LOT`` is an airline in Poland and a common noun
in English, and only one of them should be spelled out.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

__all__ = [
    "letter_name",
    "spell_acronym",
    "spell_acronyms",
    "spells_acronyms",
    "word_acronyms",
]

_MIN_LETTERS = 2
_MAX_LETTERS = 5
"""Above five letters an all-caps run is far more often a shout, a product name
or a heading than an initialism, and spelling one out is a worse error than
leaving it — the listener can read ``SIGGRAPH``; they cannot un-hear
*ess-eye-gee-gee-ar-ay-pee-aitch*."""


@lru_cache(maxsize=1)
def _tables() -> dict[str, tuple[dict[str, str], frozenset[str]]]:
    path = Path(__file__).parent.parent / "models" / "data" / "numbers.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[dict[str, str], frozenset[str]]] = {}
    for lang, entry in doc.get("languages", {}).items():
        names = entry.get("letter_names")
        if not names:
            continue
        out[lang] = (dict(names), frozenset(entry.get("word_acronyms", ())))
    return out


def spells_acronyms(language: str) -> bool:
    """Whether this language has a letter table at all."""
    return language in _tables()


def word_acronyms(language: str) -> frozenset[str]:
    """The acronyms this language reads as words rather than spelling."""
    entry = _tables().get(language)
    return entry[1] if entry else frozenset()


def letter_name(letter: str, language: str) -> str | None:
    """What ``language`` calls one letter, or ``None`` if it has no name for it.

    ``None`` rather than a guess: a letter with no entry means the acronym is
    left alone entirely, because half-spelling one (*ef-be-**q***) is worse than
    not spelling it.
    """
    entry = _tables().get(language)
    if entry is None:
        return None
    return entry[0].get(letter.lower())


def spell_acronym(word: str, language: str) -> str | None:
    """``word`` as spelled-out letters, or ``None`` if it should be left alone.

    Returns ``None`` — meaning "not an acronym, or not one I can spell" — for a
    word that is not all-caps, is too short or too long, is a word in this
    language, or contains a letter this language has no name for.
    """
    if len(word) < _MIN_LETTERS or not word.isupper() or not word.isalpha():
        return None
    lowered = word.lower()
    entry = _tables().get(language)
    if entry is None:
        return None
    names_by_letter, words = entry
    if lowered in words:
        # A word, not an initialism: read as itself, lowercased so no later
        # pass mistakes it for an acronym again.
        #
        # Checked *before* the length cap: the cap is about how long a thing
        # may be before spelling it becomes worse than leaving it, and it has
        # nothing to say about a whole word. Cap first and every entry over
        # five letters is dead — UNESCO, UNICEF and INTERPOL never reach this
        # branch, so table entries beyond that length could do nothing.
        return lowered
    if len(word) > _MAX_LETTERS:
        return None
    names = [names_by_letter.get(ch) for ch in lowered]
    if any(name is None for name in names):
        return None
    # Hyphens rather than spaces: they keep the letters one prosodic unit, so
    # the model reads a run of names instead of a list of tiny words.
    return "-".join(name for name in names if name is not None)


def spell_acronyms(text: str, language: str) -> str:
    """Every lone acronym in ``text``, spelled the way ``language`` spells it.

    **Shouting is left alone**, and the rule for telling it from an initialism
    is context rather than anything inside the word. An initialism appears as a
    single capitalised island in ordinary text — "the CIA said" — while emphasis
    comes in runs. That distinction is not available from the word itself: `IT`
    is a word, an initialism and a shout depending only on what sits beside it,
    and no table can separate those. So a capitalised word spells out only when
    neither neighbour is also capitalised, and a text that is *entirely*
    capitals is passed through whole, because someone pasted a headline and
    spelling all of it would be the loudest possible wrong answer.
    """
    if not spells_acronyms(language) or not any(ch.isupper() for ch in text):
        return text

    tokens = re.split(r"(\W+)", text)
    words = [t for t in tokens if t and t.isalpha() and len(t) > 1]
    if len(words) > 1 and all(t.isupper() for t in words):
        # The whole text is capitals: someone pasted a shout, or a headline.
        # Spelling all of it would be the loudest possible wrong answer.
        #
        # More than one word, though. A text that is a single capitalised token
        # — `synthesize("GPT")` — is an acronym on its own, not a shout: there
        # is no run to read emphasis from, and refusing it would mean the one
        # call shaped exactly like "say this acronym" was the one that did not.
        return text

    def is_caps(index: int) -> bool:
        token = tokens[index]
        return bool(token) and token.isalpha() and token.isupper() and len(token) > 1

    out = list(tokens)
    for i, token in enumerate(tokens):
        if not is_caps(i):
            continue
        # Neighbours, skipping the separator token between words.
        before = is_caps(i - 2) if i >= 2 else False
        after = is_caps(i + 2) if i + 2 < len(tokens) else False
        if before or after:
            continue  # part of a run: emphasis, not an initialism
        said = spell_acronym(token, language)
        if said is not None:
            out[i] = said
    return "".join(out)
