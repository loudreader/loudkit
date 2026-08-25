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
    """The generator must be deterministic and match the committed lexicon.

    The generator writes to ``swift/LoudKitText/Resources/pl_en_respell.json``
    (the Swift engine's copy). Regenerating it must reproduce the canonical
    Python copy byte-for-byte — which is also the proof that the Swift engine
    and the Python package read the same lexicon. A non-deterministic or
    drifting generator would break the four-way byte-identity that
    ``test_binding_copies_match_the_canonical_lexicon`` pins.
    """
    pytest.importorskip("subprocess")
    import subprocess
    import sys

    if not (REPO / "tools" / "cmudict.dict").exists():
        pytest.skip("cmudict.dict not present (tools/ is incomplete here)")

    # Asserted, not skipped: this path moved with the LoudKitText target and
    # the test went quiet rather than red — which is the whole failure mode the
    # module docstring is about.
    swift_copy = _COPIES["swift"]
    assert swift_copy.exists(), f"the Swift lexicon copy is not at {swift_copy}"

    subprocess.run(
        [sys.executable, str(REPO / "tools" / "gen_pl_respell.py")],
        check=True,
    )
    assert _sha256(swift_copy) == _sha256(CANONICAL), (
        "gen_pl_respell.py output (Swift copy) differs from the canonical "
        "Python copy — the Swift and Python engines would read Polish "
        "differently; commit the regenerated file to all five copies"
    )
