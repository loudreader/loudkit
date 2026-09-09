"""Judge a bundle from disk alone: roster, the two halves, allowlists, checksums.

``check_bundle`` is the one function both the build and ``--verify-only``
run, so the pre-upload check and the post-build check cannot drift. Nothing
here trusts a digest remembered in memory: every listed file is re-hashed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import (
    BRANDING,
    CHECKPOINT_NAME,
    DOCUMENTS,
    ENROLL_COREML,
    ENROLL_ONNX,
    ENROLLMENT_CHECKPOINT_NAME,
    EXPORT_RECORD,
    KNOWN_PROFILES,
    RELEASE_PROFILES,
    ROSTER_PATH,
    ROSTER_PER_LANGUAGE,
    ROSTER_SIZE,
    SAMPLES,
    STRICT,
    SYNTHESIS_COREML,
    SYNTHESIS_ONNX,
    TURBO_CHECKPOINT_NAME,
    TURBO_PROFILE,
    UNCHECKSUMMED,
    VOICE_ENCODER_NAME,
    BuildRefusedError,
    payload_sha256,
    read_header,
    sha256,
    synthesis_files,
)
from .assemble import _checksum_entries

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from collections.abc import Sequence


# ----------------------------------------------------------------- the roster


def roster_names() -> tuple[str, ...]:
    """The canonical 28 voices, in the order the provenance file lists them.

    Read rather than hard-coded, and checked for shape: 28 voices, ten English
    voices and two for each of the other nine languages. Without the shape check
    a truncated provenance
    file would quietly shrink what ``full-0.1`` requires, which is the same
    class of defect as the profile accepting one arbitrary voice.

    Raises:
        BuildRefusedError: the file is absent, malformed, or not the canonical
            roster.
    """
    if not ROSTER_PATH.is_file():
        raise BuildRefusedError(
            [("voice roster", str(ROSTER_PATH), "the roster is the source of truth for the 28")]
        )
    try:
        entries = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BuildRefusedError(
            [("voice roster", str(ROSTER_PATH), f"unreadable: {exc}")]
        ) from exc

    problems: list[tuple[str, str, str]] = []
    if not isinstance(entries, list):
        raise BuildRefusedError(
            [("voice roster", str(ROSTER_PATH), "expected a list of voices")]
        )

    names: list[str] = []
    languages: dict[str, int] = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "name" not in entry or "language_id" not in entry:
            problems.append(
                (f"roster entry {i}", str(ROSTER_PATH), "needs 'name' and 'language_id'")
            )
            continue
        names.append(str(entry["name"]))
        languages[str(entry["language_id"])] = languages.get(str(entry["language_id"]), 0) + 1

    if len(set(names)) != len(names):
        duplicated = sorted({n for n in names if names.count(n) > 1})
        problems.append(
            ("voice roster", str(ROSTER_PATH), f"names repeat: {', '.join(duplicated)}")
        )
    if len(names) != ROSTER_SIZE:
        problems.append(
            ("voice roster", str(ROSTER_PATH), f"{len(names)} voices, expected {ROSTER_SIZE}")
        )
    uneven = sorted(
        lang
        for lang in languages.keys() | ROSTER_PER_LANGUAGE.keys()
        if languages.get(lang) != ROSTER_PER_LANGUAGE.get(lang)
    )
    if uneven:
        problems.append(
            (
                "voice roster",
                str(ROSTER_PATH),
                f"{ROSTER_PER_LANGUAGE} voices per language; these differ: {', '.join(uneven)}",
            )
        )
    if problems:
        raise BuildRefusedError(problems)
    return tuple(names)


def _shipped_name(src: Path) -> str:
    """The name a voice profile ships under.

    The reference voice ships as ``testvoice.safetensors``, the name the
    quickstart and the tutorials use, not the internal
    ``testvoice.voice.safetensors``. The ``.voice`` suffix is a test naming
    convention, and stripping it here is what makes the README's load path
    work for a stranger. A voice already named without the suffix is copied
    as-is, which is exactly why two sources can land on one bundle path.
    """
    return src.name.replace(".voice.safetensors", ".safetensors")


def _voice_sources(voice_dir: Path) -> dict[str, Path]:
    """Bundle name -> source file, refusing when two sources want one name.

    ``a.voice.safetensors`` and ``a.safetensors`` both normalise to
    ``voices/a.safetensors``. The second copy overwrote the first, both were
    hashed, and ``SHA256SUMS`` ended up with two lines for one path holding
    two different digests: the builder reported success and ``shasum -c``
    then failed on the published bundle.

    Raises:
        BuildRefusedError: naming both sources of every collision.
    """
    claims: dict[str, list[Path]] = {}
    if voice_dir.is_dir():
        for src in sorted(voice_dir.glob("*.safetensors")):
            claims.setdefault(_shipped_name(src), []).append(src)
    problems = [
        (
            f"voices/{name}",
            f"{len(sources)} sources: {', '.join(s.name for s in sources)}",
            "one source per shipped name; rename or remove one",
        )
        for name, sources in sorted(claims.items())
        if len(sources) > 1
    ]
    if problems:
        raise BuildRefusedError(problems)
    return {name: sources[0] for name, sources in claims.items()}


# --------------------------------------------------------------- the two halves


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
"""A sha256, lowercase hex."""


def _read_header(path: Path) -> tuple[dict[str, Any], list[str]]:
    """A safetensors file's embedded manifest and its tensor names.

    The container parse is :func:`loudkit.checkpoint.read_header`, the one the
    runtime hashes payloads through; what this adds is the embedded manifest,
    which is a release's question and not the runtime's. Nothing here loads a
    tensor, so checking a 1.27 GB pair costs two short reads, which is what
    lets the check sit in the preflight where a refusal is free rather than at
    the end where it has already copied 4.3 GB.

    Raises:
        ValueError: the file is not a safetensors file, or carries no manifest.
    """
    header, _payload_start = read_header(path)
    meta = header.get("__metadata__")
    if not isinstance(meta, dict) or "manifest" not in meta:
        raise ValueError("no embedded manifest")
    try:
        manifest = json.loads(meta["manifest"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"unreadable embedded manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("the embedded manifest is not an object")
    return manifest, sorted(k for k in header if k != "__metadata__")


def _split_problems(paths: dict[str, Path]) -> list[tuple[str, str, str]]:
    """Whether these two files really are the two halves of one checkpoint.

    ``tools/split_checkpoint.py`` proves the split when it writes the pair.
    This proves it again about the two files actually being shipped, which is
    a different claim: the halves of two different packing runs each pass
    their own split, and together they are a checkpoint that never existed.

    Four things, none of which needs the packed original:

    * each file claims the ``artifact_role`` its name is supposed to carry;
    * both carry the same ``split`` provenance: same source payload digest,
      same source tensor count, same canonical filenames by role;
    * **disjoint**: no tensor name appears in both files;
    * **complete**: the digest of the sorted union of their tensor names is
      the ``source_tensor_names_sha256`` both manifests carry, and the union
      is the size the source was.

    Completeness is the half that a file listing cannot show. Dropping the
    speaker encoder from the enrollment file leaves two files that are still
    disjoint, still correctly named, still correctly rolled, and cannot enroll
    a voice. The name digest is what catches it.

    Returns ``(what, where, how)`` triples, empty when the pair holds together.
    """
    problems: list[tuple[str, str, str]] = []
    read: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for name, path in paths.items():
        try:
            read[name] = _read_header(path)
        except (OSError, ValueError) as exc:
            problems.append((name, f"{path}: {exc}", "split it with tools/split_checkpoint.py"))
    if problems:
        return problems

    for name, (manifest, _names) in read.items():
        role = manifest.get("artifact_role")
        want = "enrollment" if name == ENROLLMENT_CHECKPOINT_NAME else "synthesis"
        if role != want:
            problems.append(
                (
                    f"{name} artifact_role",
                    f"{role!r}, expected {want!r}",
                    "the two halves are the same shape; a swap is only visible here",
                )
            )

    blocks: dict[str, dict[str, Any]] = {}
    for name, (manifest, _names) in read.items():
        block = manifest.get("split")
        if isinstance(block, dict):
            blocks[name] = block
        else:
            problems.append(
                (
                    f"{name} split provenance",
                    "no split block in the embedded manifest",
                    "split it with tools/split_checkpoint.py",
                )
            )
    if len(blocks) != len(paths):
        return problems

    problems += _provenance_problems(blocks)
    if problems:
        return problems

    return _coverage_problems(
        {name: set(read[name][1]) for name in paths}, next(iter(blocks.values()))
    )


def _provenance_problems(blocks: dict[str, dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Both halves came out of the same split of the same packed checkpoint.

    Two files can each be a valid half and still not be *these* halves: the
    synthesis file of one packing run and the enrollment file of another are
    disjoint, correctly rolled, and describe a checkpoint that never existed.
    Agreeing on the source payload digest and the source tensor-name digest is
    what rules that out.
    """
    problems: list[tuple[str, str, str]] = []
    for key in ("source_payload_sha256", "source_tensor_names_sha256", "source_tensor_count"):
        absent = sorted(name for name, block in blocks.items() if block.get(key) is None)
        if absent:
            # Comparing str(None) to str(None) agrees, so a key missing from
            # both halves would read as provenance that matched. A half that
            # cannot say what it came from proves nothing about its partner.
            problems.append(
                (
                    f"split.{key}",
                    f"absent from {', '.join(absent)}",
                    "re-split with tools/split_checkpoint.py; a half must record its source",
                )
            )
            continue
        values = {str(block.get(key)) for block in blocks.values()}
        if len(values) != 1:
            problems.append(
                (
                    f"split.{key}",
                    f"the two files disagree: {', '.join(sorted(values))}",
                    "these are halves of two different checkpoints; split one, once",
                )
            )
    filenames = {
        "enrollment": ENROLLMENT_CHECKPOINT_NAME,
        "synthesis": next(name for name in blocks if name != ENROLLMENT_CHECKPOINT_NAME),
    }
    return problems + [
        (f"{name} split.roles", repr(block.get("roles")), f"a release ships {filenames}")
        for name, block in blocks.items()
        if block.get("roles") != filenames
    ]


def _coverage_problems(
    names_by_file: dict[str, set[str]], reference: dict[str, Any]
) -> list[tuple[str, str, str]]:
    """Disjoint and complete, judged from tensor names and the split block.

    Disjointness is a set intersection. Completeness is the digest of the
    sorted union against the ``source_tensor_names_sha256`` the split recorded,
    which is the only check here that can see a tensor that is in neither file:
    a listing of what is present cannot show what is absent.
    """
    problems: list[tuple[str, str, str]] = []
    both = sorted(set.intersection(*names_by_file.values()))
    if both:
        problems.append(
            (
                "the split is not disjoint",
                f"{len(both)} tensor(s) in both files: {', '.join(both[:5])}"
                + (" ..." if len(both) > 5 else ""),
                "one tensor, one file; split it again with tools/split_checkpoint.py",
            )
        )
    union = sorted(set.union(*names_by_file.values()))
    expected_digest = str(reference.get("source_tensor_names_sha256"))
    expected_count = reference.get("source_tensor_count")
    digest = hashlib.sha256("\n".join(union).encode()).hexdigest()
    if len(union) != expected_count or digest != expected_digest:
        problems.append(
            (
                "the split is not complete",
                f"{len(union)} tensor(s) across the pair, {digest[:12]}…; "
                f"the source had {expected_count}, {expected_digest[:12]}…",
                "a tensor went missing between the split and here; split it again",
            )
        )
    return problems


# ------------------------------------------------------------- the allowlist


def _allowlist(
    *, roster: Sequence[str], ships_onnx: bool, ships_coreml: bool
) -> tuple[set[str], tuple[str, ...]]:
    """Exactly what a ``full-0.1`` bundle holds.

    Returns ``(paths, package_prefixes)``. The paths are exact; the prefixes
    are the CoreML packages, whose internal layout belongs to
    coremltools and is not this tool's to enumerate. Everything else is named
    one file at a time, so an unexpected file is an error rather than a
    passenger.
    """
    paths = {
        CHECKPOINT_NAME,
        ENROLLMENT_CHECKPOINT_NAME,
        VOICE_ENCODER_NAME,
        "manifest.json",
        "tokenizer.json",
        "SHA256SUMS",
        "release.json",
    }
    paths.update(f"voices/{name}.safetensors" for name in roster)
    paths.update(name for _source, name, _key in DOCUMENTS)
    paths.add(BRANDING[1])
    paths.update(name for _source, name in SAMPLES)
    if ships_onnx:
        paths.update(f"onnx/{name}" for name in SYNTHESIS_ONNX + ENROLL_ONNX)
        paths.add(f"onnx/{EXPORT_RECORD}")
    if ships_coreml:
        paths.add(f"coreml/{EXPORT_RECORD}")
    prefixes = (
        tuple(f"coreml/{name}/" for name in SYNTHESIS_COREML + ENROLL_COREML)
        if ships_coreml
        else ()
    )
    return paths, prefixes


def turbo_allowlist(roster: Sequence[str]) -> tuple[set[str], tuple[str, ...]]:
    """The fusion decoder bundle, including the shared enrollment assets."""
    paths, _ = _allowlist(roster=roster, ships_onnx=False, ships_coreml=False)
    paths.remove(CHECKPOINT_NAME)
    paths.add(TURBO_CHECKPOINT_NAME)
    paths.update(
        f"onnx/{name}" for name in synthesis_files("fusion_mtp2", ".onnx") + ENROLL_ONNX
    )
    paths.add(f"onnx/{EXPORT_RECORD}")
    paths.add(f"coreml/{EXPORT_RECORD}")
    prefixes = tuple(
        f"coreml/{name}/"
        for name in synthesis_files("fusion_mtp2", ".mlpackage") + ENROLL_COREML
    )
    return paths, prefixes


def _audit(root: Path, paths: set[str], prefixes: Sequence[str]) -> list[str]:
    """Differences between what is in ``root`` and what the profile names."""
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    unexpected = sorted(
        p for p in present - paths if not any(p.startswith(x) for x in prefixes)
    )
    absent = sorted(paths - present)
    empty = sorted(x for x in prefixes if not any(p.startswith(x) for p in present))
    return (
        [f"the profile does not name it: {p}" for p in unexpected]
        + [f"the profile names it and it is not there: {p}" for p in absent]
        + [f"the package is empty: {p}" for p in empty]
    )


def _uncovered(out: Path, checksummed: Sequence[dict[str, object]]) -> list[str]:
    """Files in the bundle that ``SHA256SUMS`` does not name, and the reverse."""
    covered = {str(e["path"]) for e in checksummed}
    present = {
        f.relative_to(out).as_posix() for f in out.rglob("*") if f.is_file()
    } - UNCHECKSUMMED
    return sorted(
        [f"shipped without a checksum: {p}" for p in present - covered]
        + [f"checksummed but not shipped: {p}" for p in covered - present]
    )


# ---------------------------------------------------------- the closing audit


_SUMS_LINE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")


def _sums_entries(sums: Path) -> tuple[dict[str, str], list[str]]:
    """``SHA256SUMS`` as ``{path: digest}``, plus every line it gets wrong.

    A name is joined onto the bundle root and then read, so a name that is
    absolute, traverses with ``..``, or is not a plain relative POSIX path is
    a problem rather than an entry, which is the same rule ``hub._parse_sha256sums``
    holds a downloaded manifest to, applied to the bundle this tool is asked
    to vouch for.
    """
    entries: dict[str, str] = {}
    problems: list[str] = []
    for number, raw in enumerate(sums.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        match = _SUMS_LINE.match(raw.rstrip("\r"))
        if match is None:
            problems.append(f"SHA256SUMS line {number} is malformed: {raw!r}")
            continue
        name = match.group(2)
        parts = name.split("/")
        if "\\" in name or name.startswith("/") or ".." in parts or "." in parts:
            problems.append(
                f"SHA256SUMS line {number} names a file outside the bundle: {name!r}"
            )
            continue
        if name in entries:
            problems.append(f"SHA256SUMS line {number} lists {name!r} twice")
            continue
        entries[name] = match.group(1)
    return entries, problems


def check_bundle(out: Path) -> list[str]:
    """Everything a finished bundle must satisfy, judged from disk alone.

    One function on purpose: the build calls it on the staging directory after
    the closing gate, and ``--verify-only`` calls it on an assembled bundle
    before an upload, so the pre-upload check and the post-build check cannot
    drift apart. Nothing here trusts a digest remembered in memory: every
    listed file is re-hashed from the bytes on disk, which is what makes the
    call after the gate a check *of* the gate: a ``verify()`` that mutated a
    file or dropped one into the tree is caught here, because the manifests
    are written from the digests taken before it ran.

    Returns problem strings; an empty list is a bundle that holds together.
    The checks, in order:

    * ``release.json`` and ``SHA256SUMS`` exist, parse, and name a profile;
    * a bundle claiming to be a release (``full-0.1``, ``turbo-0.1``) records
      ``verified: true``;
    * the bundle contains no symlink, because a bundle is bytes and a link is an
      address that can point outside it;
    * every ``SHA256SUMS`` line hashes to the bytes on disk (``sha256sum -c``,
      in effect), and every file on disk has a line, ``SHA256SUMS`` excepted;
    * ``release.json`` and ``SHA256SUMS`` name the same files with the same
      digests, plus the one line for ``release.json`` itself;
    * a ``full-0.1`` or ``turbo-0.1`` bundle matches its profile's allowlist
      exactly.
    """
    manifest_path = out / "release.json"
    sums_path = out / "SHA256SUMS"
    if not manifest_path.is_file():
        return [f"no release.json in {out}"]
    if not sums_path.is_file():
        return [f"no SHA256SUMS in {out}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"release.json is unreadable: {exc}"]
    if not isinstance(manifest, dict):
        return ["release.json is not an object"]

    problems: list[str] = []
    profile = manifest.get("profile")
    strict = profile == STRICT
    if profile not in KNOWN_PROFILES:
        problems.append(f"release.json names no known profile: {profile!r}")
    if profile in RELEASE_PROFILES and manifest.get("verified") is not True:
        problems.append(
            f"release.json says {profile} without verified: true; the profile "
            "is the claim that the gate ran and passed"
        )

    entries, bad_lines = _sums_entries(sums_path)
    problems += bad_lines
    problems += _disk_agreement(out, entries)
    problems += _manifest_agreement(manifest, entries)
    if profile == TURBO_PROFILE:
        problems += _turbo_allowlist_agreement(out)
    if strict:
        problems += _allowlist_agreement(out)
    if strict or profile == TURBO_PROFILE:
        # The pair, not just the files. Preflight checks this before a build
        # copies anything, but --verify-only judges a directory nobody watched
        # being assembled, which is exactly where a swapped, mismatched or
        # incomplete pair would arrive. Its help says it re-audits the bundle,
        # so it has to mean the whole bundle.
        problems += _payload_agreement(out)
        problems += [
            f"{subject}: {detail} ({fix})"
            for subject, detail, fix in _split_problems(
                {
                    (
                        TURBO_CHECKPOINT_NAME if profile == TURBO_PROFILE else CHECKPOINT_NAME
                    ): out
                    / (TURBO_CHECKPOINT_NAME if profile == TURBO_PROFILE else CHECKPOINT_NAME),
                    ENROLLMENT_CHECKPOINT_NAME: out / ENROLLMENT_CHECKPOINT_NAME,
                }
            )
        ]
    return sorted(set(problems))


def _disk_agreement(out: Path, entries: dict[str, str]) -> list[str]:
    """Every checksum line hashes to the bytes on disk, and the reverse."""
    problems = [
        f"a bundle holds bytes, not links: {p.relative_to(out).as_posix()} is a symlink"
        for p in sorted(out.rglob("*"))
        if p.is_symlink()
    ]
    for name, digest in sorted(entries.items()):
        target = out / name
        if not target.is_file():
            problems.append(f"checksummed but not on disk: {name}")
        elif sha256(target) != digest:
            problems.append(f"the bytes on disk are not the bytes checksummed: {name}")
    present = {
        p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
    } - UNCHECKSUMMED
    return problems + [
        f"on disk with no checksum line: {p}" for p in sorted(present - set(entries))
    ]


def _manifest_agreement(manifest: dict[str, Any], entries: dict[str, str]) -> list[str]:
    """``release.json`` and ``SHA256SUMS`` name the same files and digests,
    plus the one line for ``release.json`` itself."""
    problems: list[str] = []
    declared: set[tuple[str, str]] = set()
    for entry in _checksum_entries(manifest):
        path = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            problems.append(
                f"release.json carries an entry with no path or no digest: {entry!r}"
            )
            continue
        declared.add((path, digest))
    from_sums = {(n, d) for n, d in entries.items() if n != "release.json"}
    if "release.json" not in entries:
        problems.append("SHA256SUMS does not cover release.json")
    return problems + [
        f"release.json and SHA256SUMS disagree about {name}"
        for name, _digest in sorted(declared.symmetric_difference(from_sums))
    ]


def _payload_agreement(out: Path) -> list[str]:
    """Each half's tensors still hash to the digest its own manifest records.

    The other checks read headers: names, roles, provenance. None of them opens
    a tensor. So a bit flipped after the split would be copied into the bundle,
    receive a fresh and perfectly correct ``SHA256SUMS`` line describing the
    flipped bytes, and pass every gate here. The manifest's
    ``tensor_payload_sha256`` is the one witness that predates the copy.

    Costly on purpose, and only in this function: it reads every tensor of both
    halves, about 1.27 GB. It runs where the expensive checks already live,
    once per build and once per ``--verify-only``, rather than in the preflight
    that exists to refuse cheaply.
    """

    problems: list[str] = []
    for name in (CHECKPOINT_NAME, TURBO_CHECKPOINT_NAME, ENROLLMENT_CHECKPOINT_NAME):
        path = out / name
        if not path.is_file():
            continue  # its absence is another check's finding, not this one's
        try:
            manifest, _ = _read_header(path)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            # A file too broken to read a header from is this check's finding
            # as much as a mismatched digest is. Raising here would end the
            # audit on its first bad file and report nothing about the rest.
            problems.append(f"{name}: cannot read its header ({exc})")
            continue
        recorded = manifest.get("tensor_payload_sha256")
        if not isinstance(recorded, str) or not _SHA256_HEX.fullmatch(recorded):
            problems.append(
                f"{name}: records tensor_payload_sha256={recorded!r}, which is not a "
                "sha256, so nothing vouches for its tensors"
            )
            continue
        try:
            actual = payload_sha256(path)
        except Exception as exc:  # any reader failure is a finding
            problems.append(f"{name}: cannot read its tensors ({exc})")
            continue
        if actual != recorded:
            problems.append(
                f"{name}: tensors hash to {actual[:12]}… but its manifest records "
                f"{recorded[:12]}…; these are not the bytes the split produced"
            )
    return problems


def _allowlist_agreement(out: Path) -> list[str]:
    """A ``full-0.1`` bundle matches the profile's allowlist exactly."""
    try:
        paths, prefixes = _allowlist(roster=roster_names(), ships_onnx=True, ships_coreml=True)
    except BuildRefusedError as refused:
        return [f"{what}: {where} -> {how}" for what, where, how in refused.problems]
    return _audit(out, paths, prefixes)


def _turbo_allowlist_agreement(out: Path) -> list[str]:
    """A ``turbo-0.1`` bundle matches its allowlist exactly."""
    try:
        paths, prefixes = turbo_allowlist(roster_names())
    except BuildRefusedError as refused:
        return [f"{what}: {where} -> {how}" for what, where, how in refused.problems]
    return _audit(out, paths, prefixes)
