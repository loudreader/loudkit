"""The freeze declaration in COMPATIBILITY.md, checked against the code.

A frozen format is a promise to a reader who is not in this repository, and
the promise lives in prose. Prose does not fail a build. So the numbers in
that table are read back here: bump ``VOICE_FORMAT_VERSION`` or add an error
code without saying so on the page, and this fails rather than the release
quietly breaking somebody's parser.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from loudkit.errors import LoudkitError
from loudkit.voice import VOICE_FORMAT_VERSION

PAGE = Path(__file__).resolve().parents[1] / "docs/reference/COMPATIBILITY.md"


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _catalog() -> set[str]:
    """Every code the error classes actually carry."""
    seen: set[str] = set()
    stack = [LoudkitError]
    while stack:
        cls = stack.pop()
        code = getattr(cls, "code", None)
        if isinstance(code, str) and code:
            seen.add(code)
        stack.extend(cls.__subclasses__())
    return seen


def test_the_voice_format_version_on_the_page_is_the_one_in_the_code(page: str) -> None:
    row = re.search(r"\|\s*voice profile\s*\|[^|]*\|\s*`(\d+)`\s*\|", page)
    assert row is not None, "the frozen-formats table has no voice profile row"
    assert int(row.group(1)) == VOICE_FORMAT_VERSION


def test_the_error_catalog_on_the_page_is_the_whole_catalog(page: str) -> None:
    section = page.split("The error-code catalog is frozen with them.", 1)
    assert len(section) == 2, "the freeze declaration no longer names the catalog"
    # The codes are written as `code` spans in the paragraph that follows.
    documented = set(re.findall(r"`([a-z_]+)`", section[1].split("\n\n", 2)[0]))
    actual = _catalog()
    assert documented == actual, (
        f"documented but not raised: {sorted(documented - actual)}; "
        f"raised but not documented: {sorted(actual - documented)}"
    )


def test_the_checkpoint_manifest_row_names_the_format_the_loader_accepts(page: str) -> None:
    from loudkit import checkpoint

    row = re.search(
        r"\|\s*checkpoint manifest\s*\|[^|]*\|\s*`([a-z-]+)`\s*/\s*`(\d+)`\s*\|", page
    )
    assert row is not None, "the frozen-formats table has no checkpoint manifest row"
    name, version = row.group(1), int(row.group(2))
    source = Path(checkpoint.__file__).read_text(encoding="utf-8")
    assert f'"{name}"' in source, f"{name!r} is not the format string the loader checks"
    assert str(version) in source
