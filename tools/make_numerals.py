"""Write `numerals.json`: what every number character folds to.

The funnel turns a number character with no ASCII spelling into something a
reader can say. Doing that from each runtime's own Unicode tables was wrong
twice over, and both ways were measured:

* **Walking to a block start breaks where blocks touch.** Every `Nd` block is
  ten contiguous code points, but the blocks themselves are contiguous too,
  `MATHEMATICAL SANS-SERIF DIGIT NINE` sits immediately before
  `MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO`, so a walk that stops at "the
  previous character is not a digit" walks straight out of its own block and
  reads a zero as a nine.
* **NFKC does not reach every numeral.** `ETHIOPIC NUMBER TEN`, the Aegean
  numbers, the Kaktovik digits and the Meroitic numerals have no compatibility
  decomposition, so they came through unchanged and were then deleted by the
  punctuation pass, silently, which is the failure the fold exists to prevent.

So the answer is a table, generated once from Python's own `unicodedata` and
shipped as data the five implementations read. That also removes the last
runtime dependence on each port's Unicode version: two builds hashing the same
`numerals.json` fold the same way whatever their ICU says.

Usage:

    .venv/bin/python tools/make_numerals.py
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from fractions import Fraction
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "python" / "loudkit" / "models" / "data"
OUT = DATA / "numerals.json"

PROVENANCE = DATA / "numerals.provenance.json"
"""Where the prose and the UCD version go, because they are not audible.

`numerals.json` is hashed **as raw bytes** into the grammar digest and through
it into `algorithm_fingerprint`, and the fingerprint's whole promise is that it
moves when the audio moves. A description sitting inside the hashed file breaks
that promise in the most embarrassing direction: correcting a sentence about
the data re-fingerprinted the engine, five ports re-pinned their digests, and
not one sample rendered differently.

So the hashed file carries the two members the fold reads and nothing else, and
everything a reader wants to know about it lives here, beside it, unhashed and
unshipped to the ports.
"""

MAX_CODE_POINT = 0x110000
FRACTION_SLASH = "⁄"

EXPECTED_UNICODE = "16.0.0"
"""The UCD this table is cut from, pinned to the interpreter that carries it.

`unicodedata` is whatever the running interpreter bundles -- 15.0.0 on Python
3.12, 16.0.0 on 3.14 -- and this project supports 3.10 through 3.14 in one CI
matrix. Regenerating on whichever interpreter happened to be first on the path
would silently produce a different table, which is a different
`algorithm_fingerprint` and a different reading of the same text.

So the version is stated, and a mismatch is refused rather than written. Moving
the pin is a deliberate act: it adds numerals (15.0 to 16.0 added eighty
digits across eight blocks), it changes the digest, and it belongs in a commit
that says so.

The *runtime's* Unicode version does not matter, and that is the point of
shipping a table at all: `fold_numerals` asks this file both whether a
character is a numeral and what it reads as, so a 3.10 process folds the 16.0
digits exactly as a 3.14 one does.
"""

MAX_DENOMINATOR = 1000
"""How far `_spoken` looks for the exact fraction behind a float.

`unicodedata.numeric` returns a `float`, so `GREEK TWO THIRDS SIGN` arrives as
0.6666666666666666 and used to be written out as `0.666667` -- read aloud, in
full, as *zero point six six six six six seven*, which is not the value of the
symbol and not what any reader would say. Every fraction in the UCD has a
denominator well under this bound -- the largest is 320, in the Tamil fractions
(`TAMIL FRACTION ONE THREE-HUNDRED-AND-TWENTIETH`), with 160 and 80 in the Tamil
and Malayalam blocks behind it -- so the rational behind the float is
recoverable exactly; the result is checked against the float before it is
written, and a value that does not come back exactly is refused rather than
rounded.
"""


def _decimal_block_zeros() -> list[int]:
    """The code point of ``0`` in every decimal-digit block.

    A digit's value is ``code point - the zero of its block``, and the zero is
    the only thing a port needs to know. Emitted sorted so a reader can binary
    search, and so the file is stable across regenerations.
    """
    zeros: list[int] = []
    for code in range(MAX_CODE_POINT):
        char = chr(code)
        if unicodedata.category(char) != "Nd":
            continue
        if unicodedata.decimal(char, None) == 0:
            zeros.append(code)
    return zeros


def _spoken(char: str) -> str | None:
    """What a ``No``/``Nl`` character becomes, or ``None`` to leave it alone.

    NFKC first, because it is the reading a person would write: `²` is `2`,
    `½` is `1/2`, `Ⅳ` is `IV`. Where there is no decomposition the numeric
    value is the only thing left that is true, and it is better than the
    silence this pass replaced, `ETHIOPIC NUMBER TEN` reads as ten rather
    than as nothing.
    """
    folded = unicodedata.normalize("NFKC", char)
    if folded != char:
        return folded.replace(FRACTION_SLASH, "/")
    value = unicodedata.numeric(char, None)
    if value is None:
        return None
    if value == int(value) and value >= 0:
        return str(int(value))
    # A fraction with no decomposition, written as the fraction it is. The
    # float is a rounded view of an exact rational -- `2/3` arrives as
    # 0.6666666666666666 -- and printing the float said *zero point six six six
    # six six seven* for a symbol that means two thirds.
    exact = Fraction(value).limit_denominator(MAX_DENOMINATOR)
    if float(exact) != value:
        raise ValueError(
            f"U+{ord(char):04X} has numeric value {value!r}, which is not a "
            f"fraction with a denominator under {MAX_DENOMINATOR}. Widen "
            f"MAX_DENOMINATOR once you have checked what the character is."
        )
    if exact.denominator == 1:
        return str(exact.numerator)
    return f"{exact.numerator}/{exact.denominator}"


def main() -> None:
    if unicodedata.unidata_version != EXPECTED_UNICODE:
        raise SystemExit(
            f"this interpreter carries Unicode {unicodedata.unidata_version} and "
            f"numerals.json is cut from {EXPECTED_UNICODE}. Run this under the "
            f"interpreter that carries the pinned version (python3.14 for "
            f"16.0.0), or move EXPECTED_UNICODE deliberately -- it changes the "
            f"table, the grammar digest and the algorithm fingerprint."
        )
    zeros = _decimal_block_zeros()
    spelled: dict[str, str] = {}
    unspoken: list[int] = []
    for code in range(MAX_CODE_POINT):
        char = chr(code)
        if unicodedata.category(char) not in ("No", "Nl"):
            continue
        text = _spoken(char)
        if text is None:
            unspoken.append(code)
            continue
        spelled[str(code)] = text

    about = (
        "What every number character folds to, generated by "
        "tools/make_numerals.py from Python's unicodedata. `decimal_zeros` is "
        "the code point of 0 in each decimal-digit block: a digit's value is "
        "its distance from the largest zero at or below it, within ten. "
        "`spelled` is every No/Nl character and the text it reads as -- ASCII "
        "for all but 27 of them, which decompose to CJK ideographs (some in "
        "parenthesised form) and leave the fold as the characters they mean. "
        "Data rather than five runtimes' own tables, because walking to a block "
        "start is wrong where blocks touch and NFKC does not reach every "
        "numeral -- and because two builds that hash the same file must fold "
        "the same way whatever their ICU says. The five funnels also decide "
        "WHETHER a character is a numeral from this file, so the reading does "
        "not move with the runtime's own Unicode version either. This text is "
        "NOT in numerals.json: that file is hashed into the fingerprint, and "
        "prose has no business moving a fingerprint."
    )
    # Data only: every member here is read by the fold, and nothing here is
    # read by a human. See `PROVENANCE`.
    #
    # `newline="\n"` on both, not the platform's. `numerals.json` is hashed
    # into the grammar digest and `numerals.provenance.json` is compared byte
    # for byte against the generator, so a regeneration under a text mode that
    # translates would move the fingerprint on Windows and nowhere else.
    OUT.write_text(
        json.dumps({"decimal_zeros": zeros, "spelled": spelled}, ensure_ascii=False, indent=1)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    PROVENANCE.write_text(
        json.dumps(
            {
                "about": about,
                "generated_by": "tools/make_numerals.py",
                "unicode_version": unicodedata.unidata_version,
                "hashed": False,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"wrote {OUT} ({len(zeros)} decimal blocks, {len(spelled)} spelled, "
        f"{len(unspoken)} with no value and left alone)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    # In the entry point, not in `main`, which the suite calls as a function --
    # and above `main`, which opens with the interpreter gate. "Which
    # interpreter do I need" is the question `--help` is asked, and a gate
    # above the parse answers it with the refusal instead of the answer.
    # Both paths written are fixed, so any argument is a misreading.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    main()
