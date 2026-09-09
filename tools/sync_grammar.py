#!/usr/bin/env python3
"""Copy the shared funnel data files to the ports and report the digest.

The digest is in the fingerprint, so the five implementations must ship the same
bytes or they disagree about what algorithm they are. Doing that by hand is how
the copies fell two features behind on 2026-08-17 with nothing detecting it.

Run after any edit to a file in ``SYNCED``, and paste the printed digest into
the ports' pinned tests. Not every synced file is in the digest:
``pl_respell_rules.json`` is a funnel input that sits below it, so copying it
moves nothing to re-pin. See ``loudkit.frontend.textconfig.PL_RULES_PATH``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from loudkit.frontend.textconfig import grammar_digest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "python" / "loudkit" / "models" / "data"
SOURCE = DATA / "numbers.json"
PORT_DIRS = [
    ROOT / "go" / "speechtext",
    ROOT / "rust" / "src",
    ROOT / "swift" / "LoudKitText" / "Resources",
    # js/data is generated and gitignored; its own prebuild step copies it.
    ROOT / "js" / "data",
]
# Every funnel data file, and every port that needs a copy of it. `numerals.json`
# joined the set when the numeral fold stopped asking each runtime's own Unicode
# tables: walking to a decimal block's start is wrong where blocks touch, and
# NFKC does not reach every numeral, so what a number character becomes is data
# now: data that five implementations must hold byte-identical.
#
# A file is copied to the ports that *read* it, not to all four, because a copy
# nothing reads is a file that can rot without any gate noticing. Go, Rust and
# JS still carry the Polish respelling tables as literals in their own sources,
# so `pl_respell_rules.json` reaches Swift alone until they read it too; add
# each of them here as it does.
SYNCED = {
    "numbers.json": PORT_DIRS,
    "numerals.json": PORT_DIRS,
    "pl_respell_rules.json": [ROOT / "swift" / "LoudKitText" / "Resources"],
}
COPIES = [directory / name for name, dirs in SYNCED.items() for directory in dirs]


def main() -> int:
    if not SOURCE.is_file():
        print(f"missing {SOURCE}", file=sys.stderr)
        return 1
    for name in SYNCED:
        if not (DATA / name).is_file():
            print(f"missing {DATA / name}", file=sys.stderr)
            return 1
    for target in COPIES:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATA / target.name, target)
        print(f"copied -> {target.relative_to(ROOT)}")
    # The runtime's own recipe, not a restatement of it. Spelling the three
    # files and their order out here once printed a value no implementation
    # computes, and the ports were told to pin it.
    print(f"\ngrammar digest: {grammar_digest()}")
    print("pin it in: go/config/grammar_digest_test.go,")
    print("           tests/LoudKitTextTests/GrammarDigestTests.swift")
    # The digest is inside the fingerprint, so moving it moves that too, and the
    # fingerprint is pinned in more places than this one is. The fixture is
    # where the value comes from; the tool that writes it is the tool that
    # decides it.
    print("\nthe fingerprint moves with it: regenerate tests/data/conformance")
    print("with tools/make_conformance.py, then re-pin from the fixture.")
    return 0


if __name__ == "__main__":
    # In the entry point, not in `main`, which the suite calls as a function.
    # Sources and destinations are all in `SYNCED`, so any argument is a
    # misreading; parsing is what makes `--help` print help rather than copy
    # data files into four port trees.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    raise SystemExit(main())
