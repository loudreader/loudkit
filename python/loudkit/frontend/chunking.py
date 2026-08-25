"""Splitting text that is longer than one window.

A window carries about 255 speech tokens, roughly ten seconds. Anything longer
has to be split, generated in pieces and joined — and *where* the splits fall is
audible, so it is an algorithm-layer decision rather than a caller's convenience.

The rule is simple: break at the strongest punctuation available, as late as
possible. A break at a full stop is inaudible; a break mid-clause is not. When a
single sentence is too long for a window on its own, it gets broken at the best
available comma, and if it has none, at a word boundary — a bad break, but a
break the caller can hear and complain about, which is better than text that
silently disappears.

Splitting is done on characters rather than tokens because it has to happen
before the tokenizer runs, so the token budget is estimated. The estimate is
deliberately conservative: producing a chunk that overflows the window costs a
hard failure, while producing one slightly too short costs nothing but a join.
"""

from __future__ import annotations

from ..config import ChunkConfig

__all__ = ["split_text", "estimate_tokens", "CHARS_PER_TOKEN"]

CHARS_PER_TOKEN = 0.5
"""Characters of text per speech token, measured on this model.

Measured on the reference voice across English, Polish (after the respelling
funnel) and German: 0.53-0.64 characters of prepared text per speech token,
consistent with ~25 speech tokens/s at ~14-16 characters/s of narration. The
old constant (3.2) was the inverse unit mistake — it let a chunk carry ~816
characters, which the generator turned into ~1300 tokens against a 255-token
window, silently dropping the tail of every over-long chunk.

The constant is the **low end with margin** (0.5 < the 0.53 measured minimum),
not the middle of the range. At 255 tokens a budget of ~127 characters maps to
at most ~254 tokens even for the densest measured language, so a chunk that
fits is guaranteed not to overflow the window. Being conservative here costs
nothing but slightly more, slightly shorter chunks; being wrong costs a
`ValueError` in the middle of a stream.
"""


def estimate_tokens(text: str) -> int:
    """Conservative upper estimate of the speech tokens ``text`` will produce."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def split_text(text: str, config: ChunkConfig) -> list[str]:
    """Split ``text`` into pieces that each fit one window.

    Args:
        text: the passage to read.
        config: the chunking policy, from :class:`~loudkit.config.AlgorithmConfig`.

    Returns:
        Chunks in order, each stripped of surrounding whitespace, together
        covering the input. Never empty for non-empty input.

    Example:
        >>> from loudkit.config import ChunkConfig
        >>> split_text("One. Two. Three.", ChunkConfig(max_tokens=12, prefix_tokens=0))
        ['One.', 'Two.', 'Three.']
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
    rest = text

    while rest:
        this_budget = first_budget if not chunks else budget
        if len(rest) <= this_budget:
            chunks.append(rest.strip())
            break

        head = rest[: this_budget + 1]
        cut = -1
        # Strongest separator first, and within a separator the LATEST break, so
        # chunks run as long as they may rather than as short as they can.
        for sep in config.split_on:
            found = head.rfind(sep)
            if found > 0:
                cut = found + len(sep)
                break
        if cut <= 0:
            # No punctuation in a whole window's worth of text. Break at the last
            # word boundary; it will be heard, and that is the point.
            cut = head.rfind(" ")
        if cut <= 0:
            cut = this_budget  # one unbroken run longer than the budget: mid-word
        # Never zero: a cut of 0 leaves `rest` unchanged and the loop spins
        # forever. ChunkConfig refuses a max_tokens that small, so this is the
        # second line of defence, for a config built some other way.
        cut = max(cut, 1)

        chunks.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()

    return [c for c in chunks if c]
