"""Splitting text that is longer than one window.

See ``docs/design/text-funnel.md``.
"""

from __future__ import annotations

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
    boundary = look[:at]
    for abbreviation in config.abbreviations:
        if not boundary.endswith(abbreviation):
            continue
        before = len(boundary) - len(abbreviation)
        if before > 0 and _is_ascii_word(boundary[before - 1]):
            continue  # the tail of a longer word, not a word of its own
        return True
    return False


def split_text(text: str, config: ChunkConfig) -> list[str]:
    """Split ``text`` into pieces that each fit one window.

    See ``docs/design/text-funnel.md``.
    """
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
    # An index into `text`, not a shrinking copy of it. Cutting the remainder
    # away each round copied the whole tail once per chunk, so the work grew
    # with the square of the text: 0.2 s for 1 MB, 12 s for 8 MB, all of it
    # before the first token is generated. The two slices below are one window
    # each, so the walk is linear and 8 MB now costs 0.6 s. The chunks are the
    # same chunks; `tests/test_chunking.py` holds them against the fixture.
    at = 0
    end = len(text)

    while at < end:
        this_budget = first_budget if not chunks else budget
        if end - at <= this_budget:
            chunks.append(text[at:].strip())
            break

        head = text[at : at + this_budget + 1]
        # One character past the window, and used only by `_holds`: the latest
        # candidate can end the window exactly, and the test reads the
        # character after it. The search itself stays inside the budget.
        look = text[at : at + this_budget + 2]
        cut = -1
        # Strongest separator first, and within a separator the LATEST break, so
        # chunks run as long as they may rather than as short as they can.
        for sep in config.split_on:
            found = head.rfind(sep)
            # A period inside a sentence is not a boundary, so the search keeps
            # walking back through this separator's own occurrences before it
            # gives up and tries a weaker one. Searching `head[:found]` skips an
            # occurrence overlapping the held one, which no separator here can
            # have.
            while found > 0 and _holds(look, found, sep, config):
                found = head.rfind(sep, 0, found)
            if found > 0:
                cut = found + len(sep)
                break
        if cut <= 0:
            # No punctuation in a whole window's worth of text.
            cut = max(head.rfind(b) for b in WORD_BOUNDARIES)
        if cut <= 0:
            cut = this_budget  # one unbroken run longer than the budget: mid-word
        # Never zero: a cut of 0 leaves `rest` unchanged and the loop spins
        # forever. ChunkConfig refuses a max_tokens that small, so this is the
        # second line of defence, for a config built some other way.
        cut = max(cut, 1)

        chunks.append(text[at : at + cut].strip())
        at += cut
        while at < end and text[at].isspace():
            at += 1

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
