"""The generated respelling lexicon must not drift across its copies.

The 110k-word Polish respelling lexicon is generated once by
``tools/gen_pl_respell.py``. Four copies are committed: the Python package (the
canonical copy), Swift, Go and Rust. JavaScript deliberately generates its
``js/data`` copy during ``prebuild`` because that directory is package output;
the npm tarball check verifies that fifth copy. A regeneration that changes a
committed file must update all four or be caught here — a drift between the
Python engine and a binding would make Polish text read differently per
implementation, which is exactly the defect this library exists to prevent.

The check is byte-level: the canonical source is authoritative, and every
committed copy must equal it exactly.

Two of the five used to be missing from ``_COPIES`` while the docstring
claimed all of them were covered, and the second test below silently *skipped*
after the Swift resources moved to their own target — 6.3 MB of lexicon per
port, pinned by a test that had stopped running. Every path is asserted to
exist, so a move breaks the build instead of quietly widening the hole.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CANONICAL = REPO / "python" / "loudkit" / "models" / "data" / "pl_en_respell.json"

_COPIES = {
    "go": REPO / "go" / "speechtext" / "pl_en_respell.json",
    "rust": REPO / "rust" / "src" / "pl_en_respell.json",
    "swift": REPO / "swift" / "LoudKitText" / "Resources" / "pl_en_respell.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_binding_copies_match_the_canonical_lexicon() -> None:
    assert CANONICAL.exists(), "canonical pl_en_respell.json missing"
    want = _sha256(CANONICAL)
    for label, path in _COPIES.items():
        assert path.exists(), f"{label} copy of pl_en_respell.json missing"
        assert _sha256(path) == want, (
            f"{label} copy of pl_en_respell.json differs from the canonical "
            "file — regenerate all copies from tools/gen_pl_respell.py, or "
            "the bindings will read Polish differently from Python"
        )


def test_regenerating_the_lexicon_is_reproducible() -> None:
    """The committed lexicon is what ``tools/gen_pl_respell.py`` produces.

    The generator rewrites every copy in place, so the committed bytes have to
    be read *before* it runs: comparing two files it has just written proves
    only that it wrote the same thing twice, and a drift between the generator
    and the tree would then surface as a dirty working directory rather than as
    a red test. Same shape the gRPC stub check uses — snapshot, regenerate,
    compare, restore.

    Restoring is not tidiness. A test may not leave the working tree changed,
    and a failure here is exactly the case that would.
    """
    import subprocess
    import sys

    if not (REPO / "tools" / "cmudict.dict").exists():
        pytest.skip("cmudict.dict not present (tools/ is incomplete here)")

    # Asserted, not skipped: these paths moved once with the LoudKitText target
    # and the test went quiet rather than red — the failure mode the module
    # docstring is about.
    written = {"python": CANONICAL, **_COPIES}
    for label, path in written.items():
        assert path.exists(), f"the {label} lexicon copy is not at {path}"

    # `js/data/` is package output and gitignored, so the generator's fifth
    # copy may not be there. Whatever the generator creates that was not there
    # before is removed again below.
    js_copy = REPO / "js" / "data" / "pl_en_respell.json"
    committed = {path: path.read_bytes() for path in written.values()}
    js_existed = js_copy.exists()

    try:
        subprocess.run(
            [sys.executable, str(REPO / "tools" / "gen_pl_respell.py")],
            check=True,
        )
        for label, path in written.items():
            assert _sha256(path) == hashlib.sha256(committed[path]).hexdigest(), (
                f"gen_pl_respell.py does not reproduce the committed {label} copy of "
                "pl_en_respell.json — the ports would read Polish differently from "
                "each other; regenerate and commit all five copies"
            )
    finally:
        for path, data in committed.items():
            path.write_bytes(data)
        if not js_existed:
            js_copy.unlink(missing_ok=True)
