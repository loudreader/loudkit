"""Splitting text that is longer than one window.

See ``docs/design/text-funnel.md``.
"""

from __future__ import annotations

import re

from ..config import ChunkConfig

__all__ = ["split_text", "split_in_half", "estimate_tokens", "CHARS_PER_TOKEN"]

WORD_BOUNDARIES = ("\u0020", "\u0009", "\u000a", "\u000d", "\u00a0", "\u2007", "\u202f")
"""Characters `split_in_half` may cut on, written out rather than tested for.

See ``docs/design/text-funnel.md``.
"""

CHARS_PER_TOKEN = 0.5
"""Characters of text per speech token, measured on this model.

See ``docs/design/text-funnel.md``.
"""


def estimate_tokens(text: str) -> int:
    """Conservative upper estimate of the speech tokens ``text`` will produce."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _is_ascii_word(char: str) -> bool:
    """Whether ``char`` is an ASCII letter or digit.

    Four comparisons, and no Unicode class: the four ports have to answer this
    identically, and `str.isalnum`, `unicode.IsLetter`, `char::is_alphanumeric`
    and `CharacterSet.alphanumerics` are four different answers about what a
    letter is. ASCII is the part they cannot disagree on.
    """
    return "0" <= char <= "9" or "A" <= char <= "Z" or "a" <= char <= "z"


def _holds(look: str, at: int, sep: str, config: ChunkConfig) -> bool:
    """Whether the candidate at ``at`` is a period inside a sentence.

    See ``docs/design/text-funnel.md``.
    """
    if config.mid_sentence_period != "hold" or not sep or sep[0] != ".":
        return False

    # A sentence does not resume in lower case.
    after = at + len(sep)
    if after < len(look) and "a" <= look[after] <= "z":
        return True

    # Or the token in front of the period is a listed abbreviation. Entries
    # carry no period of their own, the period belongs to the separator.
    return _listed_abbreviation(look, 0, at, config)


def _listed_abbreviation(text: str, start: int, dot: int, config: ChunkConfig) -> bool:
    for abbreviation in config.abbreviations:
        if text.endswith(abbreviation, start, dot):
            before = dot - len(abbreviation)
            if before == start or not _is_ascii_word(text[before - 1]):
                return True
    return False


# Only the standard separator policy interprets punctuation in its prose context.
# An explicitly ordered custom list keeps its literal matching semantics.
_PROSE_SEPARATORS = (". ", "! ", "? ", "; ", ", ")
_PUNCTUATION = re.compile(r"""[.!?;:,–\u2014]["'”’»›“‘«‹)\]}]*""")
_MAX_SENTENCE_OVERRUN = 24
_CLAUSE_END = re.compile(r"(?:although|because|unless|whereas|while)\Z")
_WHILE_SUBJECT = re.compile(r"(?:the|they|we|she|he|it|you)(?=[ \t\n\r\u00a0\u2007\u202f]|$)")


def _next_word(text: str, end: int) -> int:
    while end < len(text) and text[end].isspace():
        end += 1
    return end


def _held_period(text: str, start: int, dot: int, following: int, config: ChunkConfig) -> bool:
    if config.mid_sentence_period != "hold" or following == len(text):
        return False
    if "a" <= text[following] <= "z":
        return True
    if (
        dot > start
        and (dot == start + 1 or not _is_ascii_word(text[dot - 2]))
        and "A" <= text[dot - 1] <= "Z"
        and text[dot - 1] != "I"
        and "A" <= text[following] <= "Z"
    ):
        return True
    return _listed_abbreviation(text, start, dot, config)


def _prose_cut(
    text: str, start: int, budget: int, overrun: int, config: ChunkConfig, language: str
) -> int:
    """Return the next absolute cut. The token window is the final authority on fit."""
    limit = start + budget
    sentence = extension = clause = comma = 0
    pauses = 0
    for mark in _PUNCTUATION.finditer(text, start, min(len(text), limit + overrun)):
        end = mark.end()
        if end < len(text) and not text[end].isspace():
            continue
        i = mark.start()
        char = text[i]
        if char in ".!?":
            if char == "." and _held_period(text, start, i, _next_word(text, end), config):
                continue
            if end <= limit:
                sentence = end
            elif not extension:
                extension = end
        elif end <= limit and (budget < 64 or (end - start) * 4 >= budget):
            if budget >= 64 and len(text) - end <= 24 and len(text[end:].split()) == 1:
                continue
            if char == ",":
                comma = end
                pauses += 1
            elif char in ";:" or (
                i > start
                and text[i - 1].isspace()
                and i + 1 < len(text)
                and text[i + 1].isspace()
            ):
                clause = end
                pauses += 1
    return (
        sentence
        or (extension if pauses < 2 else 0)
        or clause
        or comma
        or _word_cut(text, start, budget, config, language)
    )


def _word_cut(text: str, start: int, budget: int, config: ChunkConfig, language: str) -> int:
    # A right-edge cut on a nearly full final window can leave just one word
    # to be rendered as an entire utterance. Balance this pair inside the budget.
    remaining = len(text) - start
    balance = budget >= 64 and (remaining - budget) * 4 < budget
    limit = start + budget
    best = latest = 0
    for i in range(start + 1, limit + 1):
        if text[i] not in WORD_BOUNDARIES or text[i - 1] in WORD_BOUNDARIES:
            continue
        latest = i
        if text[i - 1] == "." and _held_period(text, start, i - 1, _next_word(text, i), config):
            continue
        distance = abs(2 * (i - start) - remaining)
        if not balance or not best or distance < abs(2 * (best - start) - remaining):
            best = i
    cut = best or latest or limit
    if language == "en" and budget >= 64 and best:
        return _english_clause_cut(text, start, cut, budget)
    return cut


def _english_clause_cut(text: str, start: int, cut: int, budget: int) -> int:
    # "through the valley while | the children followed" loses the clause
    # before its subject. Move only this last word, keeping both windows viable.
    final_word = _CLAUSE_END.search(text, max(start, cut - len("although")), cut)
    if (
        final_word
        and final_word.start() > start
        and text[final_word.start() - 1] in WORD_BOUNDARIES
    ):
        before = final_word.start()
        subject = _next_word(text, cut)
        if (
            (before - start - 1) * 4 >= budget
            and (len(text) - subject > budget or len(text) - before <= budget)
            and (final_word[0] != "while" or _WHILE_SUBJECT.match(text, subject))
        ):
            return before

    return cut


def _literal_cut(text: str, start: int, budget: int, config: ChunkConfig) -> int:
    """Honor custom separator order and take the latest literal match at each priority."""
    head = text[start : start + budget + 1]
    look = text[start : start + budget + 2]  # The period guard reads one character further.
    for sep in config.split_on:
        found = head.rfind(sep)
        while found > 0 and _holds(look, found, sep, config):
            found = head.rfind(sep, 0, found)
        if found > 0:
            return start + found + len(sep)
    gap = max(head.rfind(b) for b in WORD_BOUNDARIES)
    return start + (gap if gap > 0 else max(1, budget))


def split_text(text: str, config: ChunkConfig, language: str = "en") -> list[str]:
    """Split ``text`` into pieces that each fit one window.

    See ``docs/design/text-funnel.md``.
    """
    language = language.lower()
    text = text.strip()
    if not text:
        return []
    first_budget_tokens = config.resolved_first_chunk_max_tokens() if config.enabled else None
    if not config.enabled or (
        first_budget_tokens is None and estimate_tokens(text) <= config.max_tokens
    ):
        return [text]

    budget = int(config.max_tokens * CHARS_PER_TOKEN)
    # The first chunk may carry its own, smaller budget: time to first audio
    # is the first chunk's generation plus its render, and both scale with its
    # length. Applied to the first split only; every later chunk runs long.
    first_budget = (
        int(first_budget_tokens * CHARS_PER_TOKEN)
        if first_budget_tokens is not None
        else budget
    )
    chunks: list[str] = []
    # Keep indices into the original text so splitting a chapter requires one forward pass.
    at = 0
    end = len(text)

    while at < end:
        this_budget = first_budget if not chunks else budget
        if end - at <= this_budget:
            chunks.append(text[at:].strip())
            break

        if config.split_on == _PROSE_SEPARATORS:
            overrun = (
                _MAX_SENTENCE_OVERRUN
                if config.max_tokens == 255
                and (chunks or first_budget_tokens in (None, config.max_tokens))
                and config.cap_resplit == "word"
                else 0
            )
            cut = _prose_cut(text, at, this_budget, overrun, config, language)
        else:
            cut = _literal_cut(text, at, this_budget, config)
        chunks.append(text[at:cut].strip())
        at = _next_word(text, cut)

    return [c for c in chunks if c]


def split_in_half(text: str) -> tuple[str, str] | None:
    """Halve a chunk the window could not hold, at a word boundary.

    See ``docs/design/text-funnel.md``.
    """
    middle = len(text) / 2
    best: tuple[float, int] | None = None
    for index, character in enumerate(text):
        # Interior only: a boundary at either end yields an empty half, and
        # `split_text` would have to be asked for it all over again.
        if character in WORD_BOUNDARIES and 0 < index < len(text) - 1:
            distance = abs(index - middle)
            if best is None or distance < best[0]:
                best = (distance, index + 1)
    if best is None:
        return None
    at = best[1]
    first, second = text[:at].strip(), text[at:].strip()
    # Stripping can empty a half that the scan thought was interior, on input
    # whose boundary run is all whitespace. `split_text` strips first, so the
    # engine cannot reach it, but this is exported, and a caller handed an
    # empty half would render silence and call it speech.
    if not first or not second:
        return None
    return first, second
