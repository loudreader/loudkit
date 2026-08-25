"""What the text funnel is, expressed as part of the algorithm.

The funnel decides what string the model is handed, so it decides what the
model says. It is therefore part of the identity contract: two builds reporting
the same sixteen hex digits must not speak differently.

Two fields, because the funnel changes in two ways and only one of them can be
detected automatically:

* ``recipe`` names the funnel's *code* — the pass order, the classification
  rules, the realisation logic. A human bumps it, the same way
  ``recipe_version`` is bumped for the sampling law.
* ``grammar`` is the digest of the shared data file every implementation reads.
  It moves when the data moves and requires no manual bump.

The digest is what makes a data edit visible without anyone remembering to
declare it — and because each of the five implementations hashes *its own copy*
of that file, a port whose copy has drifted computes a different fingerprint
and the engine refuses to start.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

__all__ = ["TextConfig", "grammar_digest", "FUNNEL_PORTED"]

FUNNEL_PORTED = "funnel-2"
"""The funnel every implementation runs.

NFC, invisibles, symbols, footnote markers, acronyms, dates, ordinals,
abbreviations, clock times, numbers, punctuation, respelling — in that order, in
all five.

`funnel-1` was the funnel before NFC, acronym spelling, dates and ordinals
existed. Those four landed in Python first and in the four ports afterwards;
`funnel-2` is the funnel with all of them, in all five. The bump is what keeps
"same sixteen hex digits" from covering two builds that read `12.03.2026`
differently — the passes changed what the funnel emits for text it already
handled, which is exactly what this field is bumped for.
"""

GRAMMAR_PATH = Path(__file__).parent.parent / "models" / "data" / "numbers.json"
"""Language rules — cardinals, months, ordinals, letter names, unit words."""

RESPELL_PATH = Path(__file__).parent.parent / "models" / "data" / "pl_en_respell.json"
"""The Polish English-respelling lexicon: 110k entries, 6.5 MB.

Hashed alongside the grammar because it is a funnel input exactly as the
grammar is: it changes the spoken tokens, and an unhashed input would let a
build say different words under the same sixteen hex digits.
"""

_DIGEST_HEX = 16
"""Half a SHA-256, like the fingerprint itself: long enough that a collision is
not a thing that happens, short enough to read in a log line."""


@lru_cache(maxsize=1)
def grammar_digest() -> str:
    """The digest of the funnel's data — grammar and lexicon, in that order.

    Hashed as raw bytes rather than as parsed JSON: two files that differ only
    in whitespace produce the same speech, but they are also not the same file,
    and "the ports must ship byte-identical data" is a cheaper contract to keep
    than "the ports must ship semantically equivalent data".
    """
    return hashlib.sha256(GRAMMAR_PATH.read_bytes() + RESPELL_PATH.read_bytes()).hexdigest()[
        :_DIGEST_HEX
    ]


@dataclass(frozen=True, slots=True)
class TextConfig:
    """The funnel's identity: its code version and its data's digest."""

    recipe: str = FUNNEL_PORTED
    """The funnel's own code version. Bump when the passes change what they emit
    for text they already handled — a new language or a new table moves
    ``grammar`` on its own and needs no bump here.

    Every implementation runs the same passes under one recipe value; that is
    the state in which this field's promise holds. A build whose passes differ
    while reporting the same recipe would cover divergent readings of
    ``12.03.2026``, ``1st``, ``CIA`` or a decomposed ``ą`` under one
    fingerprint — the failure :data:`FUNNEL_PORTED` exists to prevent.
    """

    grammar: str = field(default_factory=grammar_digest)
    """Digest of ``models/data/numbers.json``. Computed, never written by hand:
    the point is that it cannot be forgotten."""
