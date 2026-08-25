#!/usr/bin/env python3
"""Copy the shared grammar file to the four ports and report the digest.

The digest is in the fingerprint, so the five implementations must ship the same
bytes or they disagree about what algorithm they are. Doing that by hand is how
the copies fell two features behind on 2026-08-17 with nothing detecting it.

Run after any edit to ``python/loudkit/models/data/numbers.json``, and paste the
printed digest into the ports' pinned tests.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "python" / "loudkit" / "models" / "data" / "numbers.json"
COPIES = [
    ROOT / "go" / "speechtext" / "numbers.json",
    ROOT / "rust" / "src" / "numbers.json",
    ROOT / "swift" / "LoudKitText" / "Resources" / "numbers.json",
    # js/data is generated and gitignored; its own prebuild step copies it.
    ROOT / "js" / "data" / "numbers.json",
]


def main() -> int:
    if not SOURCE.is_file():
        print(f"missing {SOURCE}", file=sys.stderr)
        return 1
    payload = SOURCE.read_bytes()
    missing = [t for t in COPIES if not t.parent.is_dir()]
    if missing:
        # Skipping a port whose directory is absent meant this exited 0 having
        # synced four of five. `js/data/` is gitignored and does
        # not exist on a fresh clone, so the JS copy was routinely left at an
        # older digest by a command that reported success — and the digest is
        # the thing five implementations use to agree they read the same file.
        for t in missing:
            print(f"missing directory for {t.relative_to(ROOT)}", file=sys.stderr)
        print(
            "create them (mkdir -p) and re-run: a partial sync is how the ports "
            "drift while this says it worked",
            file=sys.stderr,
        )
        return 1
    for target in COPIES:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE, target)
        print(f"copied -> {target.relative_to(ROOT)}")
    # Grammar and lexicon, in that order — the same two files the fingerprint
    # hashes. Printing the grammar's digest alone told the ports to pin a value
    # no implementation computes.
    respell = SOURCE.parent / "pl_en_respell.json"
    digest = hashlib.sha256(payload + respell.read_bytes()).hexdigest()[:16]
    print(f"\ngrammar digest: {digest}")
    print("pin it in: go/config/zz_fp_test.go,")
    print("           tests/LoudKitTextTests/GrammarDigestTests.swift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
