"""What the text funnel is, expressed as part of the algorithm.

See ``docs/design/text-funnel.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = ["TextConfig", "grammar_digest", "grammar_languages", "FUNNEL_PORTED"]

FUNNEL_PORTED = "funnel-6"
"""Which text funnel this build runs, hashed into the algorithm fingerprint.

See ``docs/design/text-funnel.md``.
"""

GRAMMAR_PATH = Path(__file__).parent.parent / "models" / "data" / "numbers.json"
"""Language rules, cardinals, months, ordinals, letter names, unit words."""

RESPELL_PATH = Path(__file__).parent.parent / "models" / "data" / "pl_en_respell.json"
"""The Polish English-respelling lexicon: 110k entries, 6.5 MB.

Hashed alongside the grammar because it is a funnel input exactly as the
grammar is: it changes the spoken tokens, and an unhashed input would let a
build say different words under the same sixteen hex digits.
"""

NUMERALS_PATH = Path(__file__).parent.parent / "models" / "data" / "numerals.json"
"""What every number character folds to, see `tools/make_numerals.py`.

Hashed with the grammar and the lexicon because it is a funnel input exactly as
they are: it decides the words a numeral becomes, and an unhashed input would
let two builds say different things under the same sixteen hex digits. It also
pins the *Unicode version* the fold uses, which was the last thing about this
layer that each runtime answered for itself.
"""

PL_RULES_PATH = Path(__file__).parent.parent / "models" / "data" / "pl_respell_rules.json"
"""The hand-written half of Polish respelling: phrases, curated lexicon, word lists.

Below the digest and not in it, which is a gap rather than a decision. It is a
funnel input exactly as the three above are, so an edit here changes spoken
words under the same sixteen hex digits unless ``FUNNEL_PORTED`` is bumped by
hand. Adding it to the digest moves the digest and re-pins five ports, so it
waits for a release that is allowed to move it. Until then it is as unhashed as
the module literals it replaced.
"""

_DIGEST_HEX = 16
"""Half a SHA-256, like the fingerprint itself: long enough that a collision is
not a thing that happens, short enough to read in a log line."""


@lru_cache(maxsize=1)
def grammar_digest() -> str:
    """The digest of the funnel's data, grammar, lexicon and numerals, in order.

    See ``docs/design/text-funnel.md``.
    """
    return hashlib.sha256(
        GRAMMAR_PATH.read_bytes() + RESPELL_PATH.read_bytes() + NUMERALS_PATH.read_bytes()
    ).hexdigest()[:_DIGEST_HEX]


@lru_cache(maxsize=1)
def grammar_languages() -> dict[str, Any]:
    """The per-language blocks of the grammar file, located once and parsed once.

    Three passes read this file for different blocks: number grammars, date and
    ordinal rules, letter names. Each spelling its own path to it is how the
    parsed table and the hashed bytes drift apart, and this module is where the
    path already lives because it is also what hashes it.
    """
    doc: dict[str, Any] = json.loads(GRAMMAR_PATH.read_text(encoding="utf-8"))
    languages: dict[str, Any] = doc["languages"]
    return languages


@dataclass(frozen=True, slots=True)
class TextConfig:
    """The funnel's identity: its code version and its data's digest."""

    recipe: str = FUNNEL_PORTED
    """The funnel's own code version. Bump when the passes change what they emit for text
    they already handled, a new language or a new table moves ``grammar`` on its own and
    needs no bump here.

    See ``docs/design/text-funnel.md``.
    """

    grammar: str = field(default_factory=grammar_digest)
    """Digest of the three funnel data files, see :func:`grammar_digest`.
    Computed, never written by hand: the point is that it cannot be forgotten."""
