"""One version everywhere, and a CHANGELOG date the moment a release is real.

Two gates, in two states.

**Always on.** The four published manifests must agree on one version:
``pyproject.toml``, ``rust/Cargo.toml``, ``js/package.json`` and
``js/package-lock.json`` (which carries the version twice). Each ecosystem
spells the same release differently — PEP 440 ``0.1.0.dev0``, npm
``0.1.0-dev0``, Cargo ``0.1.0-dev.0`` — so the comparison is on the
normalised pair, not on the string. The lockfile is here because it is what
drifted: ``package.json`` moved to ``0.1.0`` and the lock stayed on
``0.1.0-dev0``, and ``release.yml`` compares the tag against
``pyproject.toml`` alone, so nothing looked at it. ``npm publish`` reads
``package.json``, but ``npm ci`` reads the lock, and a lock that names another
version is the CI build of a release nobody released.

**Dormant until a release is prepared.** The second gate asserts that a stable
version (no ``dev``, no pre-release suffix) ships with a real CHANGELOG date
and no pre-release banners. A release is "prepared" when either is true:

* ``CHANGELOG.md``'s top entry carries a real date instead of ``XXXX-XX-XX``,
  which is RELEASING.md §2; or
* the run is a tag build (``GITHUB_REF_TYPE=tag``), or ``LOUDKIT_RELEASE_TAG``
  is set by hand.

The release commit fills the date, which turns this gate on permanently. From
then on it fails until the version is stable and every marker in RELEASING.md
§9 is gone. A tag build turns it on independently, so a tag cut from an
unprepared tree fails here rather than on the registry.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PLACEHOLDER_DATE = "XXXX-XX-XX"

# RELEASING.md §9 lists the wording that becomes false at publish. Each entry
# is (file, marker, what it is), and each marker is a short substring rather
# than a sentence, so a reflow does not fail a release.
PRERELEASE_BANNERS = (
    ("README.md", "packages are not yet", "the README pre-release note"),
    (
        "notebooks/loudkit_quickstart.ipynb",
        "Pre-release.",
        "the Colab pre-release note",
    ),
    (
        "docs/reference/troubleshooting.md",
        "lands on PyPI with the 0.1.0 release",
        "the not-on-PyPI-yet paragraph",
    ),
    (
        "site/scripts/sync-docs.mjs",
        "banner:",
        "the site-wide banner written into every generated page",
    ),
    ("site/src/handwritten/index.mdx", "lk-banner", "the landing-page banner block"),
    ("site/src/handwritten/demo.mdx", "banner:", "the demo page banner front matter"),
    # Not banners, and that is why they were missed: an install line that names
    # a branch or a git URL is a pre-release instruction wearing ordinary
    # syntax. A reader who copies one after 0.1.0 ships builds from `main`
    # instead of the tag, which is the moving target the release exists to
    # replace.
    (
        "site/src/handwritten/index.mdx",
        'git = "https://github.com/loudreader/loudkit"',
        'the landing page\'s Rust git dependency, which becomes loudkit = "0.1"',
    ),
    (
        "site/src/handwritten/index.mdx",
        'branch: "main"',
        'the landing page\'s Swift branch pin, which becomes from: "0.1.0"',
    ),
    (
        "docs/guides/10-swift.md",
        'branch: "main"',
        'guide 10\'s Swift branch pin, which becomes from: "0.1.0"',
    ),
)


def _tomllib():  # type: ignore[no-untyped-def]
    """``tomllib``, or ``tomli`` on 3.10, which the CI matrix still runs."""
    try:
        import tomllib

        return tomllib
    except ModuleNotFoundError:  # pragma: no cover - only on 3.10
        import tomli

        return tomli


def _normalised(raw: str) -> tuple[str, str]:
    """``(release, prerelease)``, each ecosystem's punctuation removed."""
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(.*)", raw.strip())
    assert match, f"version {raw!r} does not start with a major.minor.patch triple"
    return match.group(1), re.sub(r"[^0-9a-z]", "", match.group(2).lower())


def _manifest_versions() -> dict[str, str]:
    """The raw version string each published manifest carries."""
    tomllib = _tomllib()
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    cargo = tomllib.loads((REPO / "rust" / "Cargo.toml").read_text(encoding="utf-8"))
    package = json.loads((REPO / "js" / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((REPO / "js" / "package-lock.json").read_text(encoding="utf-8"))
    return {
        "pyproject.toml": pyproject["project"]["version"],
        "rust/Cargo.toml": cargo["package"]["version"],
        "js/package.json": package["version"],
        "js/package-lock.json": lock["version"],
        'js/package-lock.json packages[""]': lock["packages"][""]["version"],
    }


def _changelog_head() -> tuple[str, str]:
    """``(version, date)`` from the top ``## [x.y.z] — date`` heading."""
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^## \[([^\]]+)\]\s*[—–-]\s*(\S+)", text, re.MULTILINE)
    assert match, "CHANGELOG.md has no `## [version] — date` heading to read"
    return match.group(1), match.group(2)


def _release_tag() -> str | None:
    """The tag being built, when this run is a tag build."""
    ref = os.environ.get("GITHUB_REF", "")
    if os.environ.get("GITHUB_REF_TYPE") == "tag" and ref.startswith("refs/tags/"):
        return ref.rsplit("/", 1)[-1]
    return os.environ.get("LOUDKIT_RELEASE_TAG") or None


def _release_is_prepared() -> tuple[bool, str]:
    """Whether a real release is being cut, and what says so."""
    _version, date = _changelog_head()
    if date != PLACEHOLDER_DATE:
        return True, f"CHANGELOG.md carries a real date ({date})"
    tag = _release_tag()
    if tag:
        return True, f"this is a tag build ({tag})"
    return False, ""


def test_the_four_published_manifests_agree_on_one_version() -> None:
    """pyproject, Cargo, package.json and package-lock all say the same thing.

    ``release.yml`` checks the tag against ``pyproject.toml`` and nothing else,
    and each registry publish is irreversible, so the only place this can be
    caught cheaply is here.
    """
    versions = _manifest_versions()
    expected = _normalised(versions["pyproject.toml"])
    for label, raw in versions.items():
        assert _normalised(raw) == expected, (
            f"{label} says {raw!r} and pyproject.toml says "
            f"{versions['pyproject.toml']!r} — the release is half-flipped"
        )


def test_the_changelog_names_the_version_the_manifests_carry() -> None:
    """The top CHANGELOG entry is about the release being prepared, whatever
    its date. A 0.1.0 tree whose changelog heads at 0.0.9 ships notes for
    another release."""
    versions = _manifest_versions()
    changelog_version, _date = _changelog_head()
    assert _normalised(changelog_version)[0] == _normalised(versions["pyproject.toml"])[0], (
        f"CHANGELOG.md heads at {changelog_version!r} and the manifests carry "
        f"{versions['pyproject.toml']!r}"
    )


def test_a_prepared_release_is_stable_dated_and_unbannered() -> None:
    """Dormant while the CHANGELOG date is the placeholder and no tag is built.

    Once the date goes in, or on a tag build, it requires a stable version, a
    real date, and every pre-release marker in RELEASING.md §9 removed.
    """
    prepared, because = _release_is_prepared()
    if not prepared:
        return

    versions = _manifest_versions()
    release, prerelease = _normalised(versions["pyproject.toml"])
    assert not prerelease, (
        f"{because}, but the version is {versions['pyproject.toml']!r} — a "
        f"release is cut from a stable version, not a pre-release"
    )

    changelog_version, date = _changelog_head()
    assert date != PLACEHOLDER_DATE, (
        f"{because}, and CHANGELOG.md still carries {PLACEHOLDER_DATE}. "
        f"Put the release date on the [{changelog_version}] heading (RELEASING.md §2)."
    )
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date), (
        f"CHANGELOG.md's [{changelog_version}] date is {date!r}; write it as YYYY-MM-DD"
    )
    assert "unreleased" not in _changelog_head_line().lower(), (
        f"CHANGELOG.md's [{changelog_version}] heading still says unreleased"
    )

    left = [
        f"{path} still carries {what} ({marker!r})"
        for path, marker, what in PRERELEASE_BANNERS
        if (REPO / path).exists() and marker in (REPO / path).read_text(encoding="utf-8")
    ]
    assert not left, (
        f"{because}, so the pre-release wording is false. RELEASING.md §9 lists "
        f"every one of these:\n  " + "\n  ".join(left)
    )
    assert _normalised(changelog_version)[0] == release


def _changelog_head_line() -> str:
    """The whole top ``## [...]`` line, banner wording included."""
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^## \[.*$", text, re.MULTILINE)
    assert match, "CHANGELOG.md has no `## [...]` heading"
    return match.group(0)


_ROW = re.compile(r"^\|(.+?)\|(.+?)\|")
_BACKTICKED = re.compile(r"`(.+)`", re.S)


def _section_nine_rows() -> set[tuple[str, str]]:
    """§9's table as ``(path, marker)``, the shape ``PRERELEASE_BANNERS`` has."""
    text = (REPO / "RELEASING.md").read_text()
    start = text.index("## 9.")
    nxt = text.find("\n## ", start + 1)
    section = text[start : nxt if nxt != -1 else len(text)]
    rows: set[tuple[str, str]] = set()
    for line in section.splitlines():
        row = _ROW.match(line.strip())
        if row is None:
            continue
        cells = [_BACKTICKED.search(c.strip()) for c in row.groups()]
        if all(cells):
            rows.add((cells[0].group(1), cells[1].group(1)))  # type: ignore[union-attr]
    return rows


def test_section_nine_is_the_gate_written_out() -> None:
    """RELEASING.md §9 and ``PRERELEASE_BANNERS`` are one list in two places.

    §9 is what a human reads before cutting the release; the tuple is what
    refuses the tag if they miss one. They have drifted twice -- once by
    quoting strings that had already been replaced, and once by a rewrite that
    dropped two rows and said in prose that those files were clean while the
    gate was still watching them, and they still were not.

    Compared as sets of ``(path, marker)``, and in both directions. A first
    attempt at this asserted only that each marker appeared *somewhere* in the
    section, which is three holes wide: it passed with a marker filed under the
    wrong path, with one of the two `branch: "main"` rows deleted -- the marker
    still appeared, under the other file -- and with an invented row for a
    marker nothing enforces, which is a promise to a reader that no test keeps.
    """
    documented = _section_nine_rows()
    enforced = {(path, marker) for path, marker, _ in PRERELEASE_BANNERS}
    missing = sorted(enforced - documented)
    invented = sorted(documented - enforced)
    assert not missing, (
        "the tag gate enforces these and RELEASING.md §9 does not list them:\n"
        + "\n".join(f"  {p} :: {m!r}" for p, m in missing)
    )
    assert not invented, "RELEASING.md §9 lists these and nothing enforces them:\n" + "\n".join(
        f"  {p} :: {m!r}" for p, m in invented
    )
