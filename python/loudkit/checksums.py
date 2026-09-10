"""The release's ``SHA256SUMS``, checked against the bytes that arrived.

This module parses the manifest, decides which names may address bytes inside
a release, walks a fetched directory and hashes what the manifest lists. Three
rules, shared with the four ports: a missing manifest is fatal under
``loudreader/`` and skipped elsewhere; a listed file that was not fetched is
skipped; a fetched file the manifest does not list is refused when it is
weights, refused under ``loudreader/`` whatever it is, and reported otherwise.
The streamed hash is :func:`loudkit.checkpoint.file_sha256`, which the
checkpoint reader shares. What a release is, and whether an official one is a
release, lives in :mod:`loudkit.release`; fetching lives in :mod:`loudkit.hub`.
"""

from __future__ import annotations

import os
import posixpath
import re
from pathlib import Path
from typing import Any

from .checkpoint import file_sha256
from .release import (
    _OFFICIAL_ORG,
    VOICE_SUFFIX,
    _check_release_record,
    _is_official,
    _require_releasable,
)

_SUMS_LINE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")

_DRIVE_LETTER = re.compile(r"^[A-Za-z]:/")
"""``C:/`` at the very start: a Windows absolute path wearing POSIX separators.

Deliberately narrow: a colon is a legal character in a POSIX filename, so
``voices/a:b.safetensors`` stays a name. Only drive-plus-separator is a claim
about a filesystem root.
"""

_HUB_BOOKKEEPING = frozenset({"SHA256SUMS", ".gitattributes", ".loudkit-release.json"})
"""Files a snapshot holds that its own manifest cannot list: the manifest
itself, the LFS rules the hub client materialises into every snapshot, and the
receipt :mod:`loudkit.hub` writes after verifying. Named exactly, never by
shape: ``.hidden.safetensors`` is weights."""

_HUB_CACHE_DIR = ".cache"
"""The hub client's own scratch at the root of a ``local_dir`` download."""


def _missing_manifest(where: str) -> ValueError:
    return ValueError(
        f"{where}: no SHA256SUMS. Every {_OFFICIAL_ORG} release ships one, so a "
        "download without it cannot be checked against anything and will not be "
        "used. Retry the download, and pass a revision you trust."
    )


def _files_under(root: Path) -> list[str]:
    """Every file in the snapshot, as a POSIX path relative to ``root``, minus
    the hub's own furniture. Symlinks are followed: the hub cache is made of
    them."""
    out: list[str] = []
    for parent, dirs, files in os.walk(root):
        if Path(parent) == root:
            dirs[:] = [d for d in dirs if d != _HUB_CACHE_DIR]
        for name in files:
            path = Path(parent, name)
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if rel not in _HUB_BOOKKEEPING:
                out.append(rel)
    return sorted(out)


def _rejected_name(name: str) -> str | None:
    """Why ``name`` cannot address bytes inside a snapshot, or ``None``.

    A manifest name is joined onto the snapshot root and then read, so it has
    to be a normalised relative POSIX path: nested names stay legal, and what
    is refused is a name that can address bytes the snapshot does not contain.
    """
    if "\\" in name:
        return "is not a POSIX path (it contains a backslash)"
    if name.startswith("/") or _DRIVE_LETTER.match(name):
        return "is absolute, and a manifest name is relative to the snapshot root"
    parts = name.split("/")
    if ".." in parts:
        return "escapes the snapshot root with '..'"
    if "" in parts or "." in parts:
        return "is not normalised (an empty or '.' path component)"
    if posixpath.normpath(name) != name:
        return f"is not normalised (it names {posixpath.normpath(name)!r})"
    return None


def _parse_sha256sums(sums: Path) -> dict[str, str]:
    """Parse a ``SHA256SUMS`` file, refusing lines it cannot understand.

    Every non-empty line is a 64-hex digest, two spaces, and a normalised
    relative POSIX path, the format ``tools/build_release.py`` writes. A
    mangled line, a name that escapes the root and a name listed twice are
    each refused with the line number: a manifest that verifies nothing, or
    that disagrees with itself, must fail loudly.
    """
    out: dict[str, str] = {}
    for number, raw_line in enumerate(sums.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.rstrip("\r")
        if not line.strip():
            continue
        match = _SUMS_LINE.match(line)
        if match is None:
            raise ValueError(
                f"{sums}: malformed SHA256SUMS line {number}: {line!r}: the "
                "release manifest does not look like this project's output; "
                "refusing to verify against it"
            )
        name = match.group(2)
        rejected = _rejected_name(name)
        if rejected is not None:
            raise ValueError(
                f"{sums}: line {number}: {name!r} {rejected}; refusing to "
                "verify against a manifest that names files outside the "
                "release it describes"
            )
        if name in out:
            raise ValueError(
                f"{sums}: line {number}: duplicate entry for {name!r}: the "
                "manifest disagrees with itself about one file, so what it "
                "verifies depends on which entry is read. Rebuild the release."
            )
        out[name] = match.group(1)
    if not out:
        # Raised, not asserted: a manifest is external data and `python -O`
        # strips asserts, which turned an empty file into an empty dict that
        # verified everything.
        raise ValueError(f"{sums}: no checksum entries")
    return out


def _verify_sha256sums(root: Path, *, repo: str | None = None) -> None:
    """Check a freshly fetched snapshot against the release's own ``SHA256SUMS``.

    Run once, when the bytes arrive; a load of an already fetched snapshot does
    not come back here. An official snapshot must also be a release
    (:func:`loudkit.release._require_releasable`). The manifest authenticates
    the download, not the publisher: pin ``revision=`` for that.

    What is on disk is judged, not what this fetch asked for, which is what
    :func:`_judge_unlisted` needs: a port that hashes only the files it just
    pulled cannot see the weights the manifest never listed. The walk itself
    is free next to the hashing, at 1.2 ms against 1.5 s over a 3.65 GB set.
    """
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        if _is_official(repo):
            raise _missing_manifest(f"{repo} ({root})")
        return

    entries = _parse_sha256sums(sums)
    if _is_official(repo):
        _require_releasable(root, f"{repo} ({root})")
    present = _files_under(root)
    bad = [
        name
        for name in present
        if name in entries and file_sha256(root / name) != entries[name]
    ]
    if bad:
        raise ValueError(
            f"{root}: downloaded file(s) failed the release checksum: "
            + ", ".join(bad)
            + ": delete the cached snapshot and retry, or pin a revision you trust."
        )
    if not any(name in entries for name in present):
        raise ValueError(
            f"{root}: SHA256SUMS lists {len(entries)} file(s) but none of them "
            "was fetched: the manifest and the download disagree."
        )
    _judge_unlisted(root, repo, [name for name in present if name not in entries])


def _judge_unlisted(root: Path, repo: str | None, unlisted: list[str]) -> None:
    """The verdict on fetched files the manifest says nothing about.

    Weights are refused under any repo: those are the bytes loudkit opens.
    Under an official repo everything uncovered is refused, because the
    builder checksums every file a release ships. A third-party snapshot gets
    a warning naming the file, since no builder promised coverage there.
    """
    weights = [name for name in unlisted if name.endswith(VOICE_SUFFIX)]
    if weights:
        raise ValueError(
            f"{root}: SHA256SUMS does not list " + ", ".join(weights) + ": these are "
            "weights loudkit would open with nothing vouching for them. Delete the "
            "cached snapshot and retry, or pin a revision you trust."
        )
    if not unlisted:
        return
    if _is_official(repo):
        raise ValueError(
            f"{root}: SHA256SUMS does not list " + ", ".join(unlisted) + ". A "
            f"{_OFFICIAL_ORG} release checksums every file it ships, so "
            "these did not come from the release. Delete the cached "
            "snapshot and retry, or pin a revision you trust."
        )
    import warnings

    warnings.warn(
        f"{root}: not covered by SHA256SUMS and therefore not verified: " + ", ".join(unlisted),
        stacklevel=3,
    )


# ------------------------------------------------- the single-file door

_HUB_NOT_FOUND = frozenset(
    {
        "EntryNotFoundError",
        "RemoteEntryNotFoundError",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
    }
)
"""What the hub client raises for "it is not there". Matched by name rather
than imported: the client is an optional extra."""


def _is_not_found(exc: BaseException) -> bool:
    """Whether the hub said "it is not there". Walks the class chain, so a
    subclass the client adds later is recognised without an edit here."""
    return any(cls.__name__ in _HUB_NOT_FOUND for cls in type(exc).__mro__)


_HUB_ENTRY_NOT_FOUND = frozenset({"EntryNotFoundError", "RemoteEntryNotFoundError"})
"""The repository answered and the file is not in it. A missing repository or
revision, and an offline miss (``LocalEntryNotFoundError``), are different
conditions: ``hub`` names each of those itself."""


def _is_entry_not_found(exc: BaseException) -> bool:
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & _HUB_ENTRY_NOT_FOUND) and "LocalEntryNotFoundError" not in names


def _release_sums(hub: Any, repo: str, revision: str | None) -> dict[str, str] | None:
    """The release's ``SHA256SUMS``, parsed, or ``None`` for a third-party repo
    that ships none. An official repo without one is refused here; a repository
    that is not there at all is not "a release without a manifest", so that
    error passes through to the caller's diagnosis."""
    try:
        sums_path = hub.hf_hub_download(repo_id=repo, filename="SHA256SUMS", revision=revision)
    except Exception as exc:  # the client raises its own hierarchy
        if not _is_entry_not_found(exc):
            raise  # missing repo or revision, offline, timeout, proxy 500: never "no manifest"
        if _is_official(repo):
            raise _missing_manifest(repo) from exc
        return None
    return _parse_sha256sums(Path(sums_path))


def _verify_against_release_sums(
    hub: Any, repo: str, revision: str | None, name: str, path: Path
) -> None:
    """Hash one freshly fetched file against the release's ``SHA256SUMS``, by
    the same rules as :func:`_verify_sha256sums`."""
    entries = _release_sums(hub, repo, revision)
    if entries is None:
        return
    if _is_official(repo):
        _require_verified_release(hub, repo, revision, entries)
    expected = entries.get(name)
    if expected is None:
        raise ValueError(
            f"{repo}: SHA256SUMS does not list {name}: the file was downloaded "
            "and nothing vouches for it. Pin a revision you trust, or pass a "
            "local path."
        )
    if file_sha256(path) != expected:
        raise ValueError(
            f"{path}: downloaded file failed the release checksum: delete the "
            "cached file and retry, or pin a revision you trust."
        )


def _require_verified_release(
    hub: Any, repo: str, revision: str | None, entries: dict[str, str]
) -> None:
    """Fetch an official repo's ``release.json`` and hold it to its claim. The
    record is checked against the manifest before it is believed."""
    try:
        fetched: str = hub.hf_hub_download(
            repo_id=repo, filename="release.json", revision=revision
        )
    except Exception as exc:  # the client raises its own hierarchy
        if not _is_not_found(exc):
            raise
        raise ValueError(
            f"{repo}: no release.json. Every {_OFFICIAL_ORG} release records "
            "its profile and its verified flag there, so nothing fetched from "
            "this repo can prove it comes from a release. Pin a revision you "
            "trust."
        ) from exc
    record = Path(fetched)
    expected = entries.get("release.json")
    if expected is None:
        raise ValueError(
            f"{repo}: SHA256SUMS does not list release.json, so the record "
            "that would vouch for this release is itself vouched for by "
            "nothing. Pin a revision you trust."
        )
    if file_sha256(record) != expected:
        raise ValueError(
            f"{record}: release.json failed the release checksum: delete the "
            "cached file and retry, or pin a revision you trust."
        )
    _check_release_record(record, repo)
