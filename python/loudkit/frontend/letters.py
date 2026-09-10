"""Acronyms, spelled in the language being read.

See ``docs/design/text-funnel.md``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import NamedTuple

from .textconfig import grammar_languages

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
leaving it, the listener can read ``SIGGRAPH``; they cannot un-hear
*ess-eye-gee-gee-ar-ay-pee-aitch*."""


class _LetterTable(NamedTuple):
    """One language's spelling tables, read from ``numbers.json``."""

    names: dict[str, str]
    """Letter to its spoken name."""
    word_acronyms: frozenset[str]
    """All-caps words this language says as words, not letter by letter."""


@lru_cache(maxsize=1)
def _tables() -> dict[str, _LetterTable]:
    out: dict[str, _LetterTable] = {}
    for lang, entry in grammar_languages().items():
        names = entry.get("letter_names")
        if not names:
            continue
        out[lang] = _LetterTable(dict(names), frozenset(entry.get("word_acronyms", ())))
    return out


def spells_acronyms(language: str) -> bool:
    """Whether this language has a letter table at all."""
    return language in _tables()


def word_acronyms(language: str) -> frozenset[str]:
    """The acronyms this language reads as words rather than spelling."""
    entry = _tables().get(language)
    return entry.word_acronyms if entry else frozenset()


def letter_name(letter: str, language: str) -> str | None:
    """What ``language`` calls one letter, or ``None`` if it has no name for it.

    ``None`` rather than a guess: a letter with no entry means the acronym is
    left alone entirely, because half-spelling one (*ef-be-**q***) is worse than
    not spelling it.
    """
    entry = _tables().get(language)
    if entry is None:
        return None
    return entry.names.get(letter.lower())


def spell_acronym(word: str, language: str) -> str | None:
    """``word`` as spelled-out letters, or ``None`` if it should be left alone.

    Returns ``None``, meaning "not an acronym, or not one I can spell", for a
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
        # A word, not an initialism: read as itself, lowercased so no later pass
        # mistakes it for an acronym again.
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

    See ``docs/design/text-funnel.md``.
    """
    if not spells_acronyms(language) or not any(ch.isupper() for ch in text):
        return text

    tokens = re.split(r"(\W+)", text)
    words = [t for t in tokens if t and t.isalpha() and len(t) > 1]
    if len(words) > 1 and all(t.isupper() for t in words):
        # The whole text is capitals: someone pasted a shout, or a headline.
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
