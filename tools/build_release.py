#!/usr/bin/env python3
"""Assemble the directory that gets published, and check it is usable.

The point is not to copy files — that part is trivial. The point is what the
tool refuses. A publication tool has one job that no test of the library can
do for it: make a bundle that is *exactly* the release, or make nothing at
all.

Three properties carry that:

**Transactional.** Everything is assembled into a staging directory beside the
target and renamed into place only after every check has passed. A failed run
leaves no ``release.json``, no ``SHA256SUMS`` and no directory that looks
publishable. The staging directory is a sibling of ``--out``, so the closing
rename is on one filesystem and is atomic.

**Exact.** ``full-0.1`` validates the assembled bundle against an explicit
allowlist: the bundle holds what the profile names, and nothing else. An
unexpected file is an error, not a passenger. This is the difference between
"it has what we need" and "it is what we said".

**Named.** ``release.json`` records the profile that built it and whether the
closing gate ran and passed, so a consumer or a CI job can tell a releasable
bundle from a development one without guessing. It is written before
``SHA256SUMS`` and is covered by it: the file that says a bundle is
trustworthy is itself checksummed.

Two profiles decide what "complete" means:

``full-0.1``
    What a 0.1.0 release ships, and the default. Both halves of the
    checkpoint (``loudr-1.safetensors`` and
    ``loudr-1-enrollment.safetensors``), its manifest, the tokenizer,
    ``ve.safetensors``, the full roster of twenty voices by name, nine ONNX
    graphs, six CoreML packages, the four documents, two listening samples,
    ``release.json``, and ``SHA256SUMS`` covering every file. The tool checks
    the sources **before** it copies anything and refuses with a list of what
    is absent and which tool exports it. That check includes the pair itself:
    each half must claim the ``artifact_role`` its name carries, and the two
    together must be
    disjoint (no tensor in both) and complete (every tensor of the packed
    original in one of them), proved from their headers against the split
    provenance both manifests carry. Whether a *download* takes the
    enrollment half is a question for the client; a release carries it either
    way. The closing check then loads what it shipped: the checkpoint, the
    voice encoder, every voice, the ONNX graphs and the CoreML packages.
    ``--skip-verify`` is refused under this profile: a bundle that claims the
    profile is a bundle that passed the gate.

``lenient``
    A partial bundle for development. Whatever is present ships, whatever is
    absent is noted. Not releasable. The allowlist and the roster do not
    apply; the collision and checksum-coverage checks still do, because a
    bundle whose checksums do not verify is broken under any profile.

Nothing here uploads. It builds a local directory; publishing is a separate,
deliberate act.

    python tools/build_release.py --checkpoint … --out dist/loudr-1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from collections.abc import Sequence
    from types import ModuleType

    import numpy as np

REPO = Path(__file__).resolve().parent.parent

PROFILES = ("full-0.1", "lenient")
STRICT = "full-0.1"

# The name the release ships the checkpoint under. Every guide, every port and
# `hub._only_checkpoint_in` name this file; a bundle carrying `model.safetensors`
# is a different artefact wearing the same directory.
CHECKPOINT_NAME = "loudr-1.safetensors"

# The other half. `tools/split_checkpoint.py` writes the pair: the synthesis
# file above carries t3, the flow and the vocoder, and this one carries the two
# enrollment towers (the S3 speech tokenizer and the speaker encoder). A full
# release ships both. Whether a *download* takes the second one is a question
# for the client (enrolling from audio needs it, loading a shipped voice does
# not), but a release that does not carry it cannot answer the question at all.
ENROLLMENT_CHECKPOINT_NAME = "loudr-1-enrollment.safetensors"

# The role each name claims in its own manifest. Checked rather than assumed:
# the two files are interchangeable by name alone, and a bundle whose halves
# were swapped copies, checksums and passes every other check in this tool.
ARTIFACT_ROLES = {CHECKPOINT_NAME: "synthesis", ENROLLMENT_CHECKPOINT_NAME: "enrollment"}
# The same table the other way round, which is the shape `split.roles` carries
# in both manifests: role -> the filename a release ships it under.
ROLE_FILENAMES = {role: name for name, role in ARTIFACT_ROLES.items()}

VOICE_ENCODER_NAME = "ve.safetensors"

# The roster is data, not code: `docs/voices/roster/provenance.json` is the
# source of truth for which twenty voices a release carries, and it also
# carries their licences and their provenance. Reading it here means the
# builder and the model card cannot disagree about what the release is.
ROSTER_PATH = REPO / "docs" / "voices" / "roster" / "provenance.json"
ROSTER_SIZE = 20
ROSTER_PER_LANGUAGE = 2

# The nine graphs and six packages a 0.1.0 release ships, split by the tool
# that writes them so the refusal can name it. The enrollment triple is the
# half that went missing: SUPPORTED.md declares voice enrollment in five
# languages, and a bundle without these three graphs cannot enroll in any of
# them, on any port.
SYNTHESIS_ONNX = (
    "t3_cond.onnx",
    "t3_prefill.onnx",
    "t3_step.onnx",
    "flow_encoder.onnx",
    "flow_estimator.onnx",
    "vocoder.onnx",
)
ENROLL_ONNX = ("s3_tokenizer.onnx", "camp.onnx", "voice_encoder.onnx")
SYNTHESIS_COREML = (
    "flow_encoder.mlpackage",
    "flow_estimator.mlpackage",
    "vocoder.mlpackage",
)
ENROLL_COREML = ("s3_tokenizer.mlpackage", "camp.mlpackage", "voice_encoder.mlpackage")

# (source in the repo, name in the release, key in release.json). Each is its
# own key rather than one list, so every document carries a checksum line the
# way the checkpoint does.
DOCUMENTS = (
    ("docs/MODEL_CARD.md", "README.md", "readme"),
    ("LICENSE", "LICENSE", "license"),
    ("NOTICE", "NOTICE", "notice"),
    ("RESPONSIBLE_USE.md", "RESPONSIBLE_USE.md", "responsible_use"),
)

# (source in the repo, name in the release, key in release.json). The model
# card resolves this from the model repository itself, so the image must ship
# and be checksummed with the release rather than depend on another site.
BRANDING = ("assets/logo-wordmark.png", "logo.png", "logo")

# The model card's first job is to let a visitor hear the model. Keep those
# players self-contained on Hugging Face instead of depending on GitHub Pages
# or raw.githubusercontent.com: the release carries the exact bytes it embeds,
# and both manifests vouch for them like every graph and weight.
SAMPLES = (
    ("docs/voices/roster/audio/joe.opus", "samples/joe.opus"),
    ("docs/voices/roster/audio/kathleen.opus", "samples/kathleen.opus"),
)

# `SHA256SUMS` cannot contain its own digest, so it is the one file with no
# checksum line. This is the invariant behind the count RELEASING.md asks
# for: a bundle of N files carries exactly N-1 checksum lines.
UNCHECKSUMMED = frozenset({"SHA256SUMS"})


class BuildRefusedError(Exception):
    """A refusal carrying ``(what, where, how)`` triples to print."""

    def __init__(self, problems: Sequence[tuple[str, str, str]]) -> None:
        super().__init__(f"{len(problems)} problem(s)")
        self.problems = list(problems)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy(src: Path, dst: Path, root: Path, label: str) -> dict[str, object]:
    if not src.exists():
        raise FileNotFoundError(f"{label} missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    digest = sha256(dst)
    size = dst.stat().st_size
    print(f"  {label:22s} {size / 1e6:8.1f} MB  {digest[:16]}")
    # Path relative to the release root (the staging directory, which is
    # renamed to `--out`), so release.json and SHA256SUMS both name files the
    # way a verifier at the release root sees them: `loudr-1.safetensors`,
    # `voices/joe.safetensors`, …
    #
    # `.as_posix()` rather than `str()`: the separator belongs to the release,
    # not to the machine that assembled it. On Windows `str()` writes
    # `voices\joe.safetensors` into both files, and `sha256sum -c` on the
    # downloader's Linux box then looks for a file whose name contains a
    # backslash and reports it missing.
    return {"path": dst.relative_to(root).as_posix(), "sha256": digest, "bytes": size}


def copy_tree(src: Path, dst: Path, root: Path, label: str) -> list[dict[str, object]]:
    """Copy a directory of exported graphs, checksumming every file in it.

    The ONNX graphs and the CoreML packages were assembled outside this tool
    until now, which is why the published `SHA256SUMS` covered the checkpoint
    and the voices and left 589 MB of graphs unverified. A release either
    vouches for what it ships or it does not; these ship, so they are hashed.

    CoreML packages are directories, so the walk is recursive and every leaf
    gets its own line. `hub._verify_sha256sums` skips a listed file that was
    not fetched, so naming them here costs a default `load()` nothing: the
    entries are checked by whoever downloads the graphs and ignored by
    everyone else.
    """
    if not src.is_dir():
        raise FileNotFoundError(f"{label} missing: {src}")
    entries: list[dict[str, object]] = []
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        entries.append(copy(f, dst / f.relative_to(src), root, f"{label}/{f.name}"))
    if not entries:
        raise FileNotFoundError(f"{label} is empty: {src}")
    return entries


def _checksum_entries(files: dict[str, Any]) -> list[dict[str, object]]:
    """Flatten release.json entries into the ordered list SHA256SUMS names.

    ``files`` also carries scalar metadata — ``profile`` and ``verified`` —
    which is neither a file entry nor a list of them and is skipped here by
    type. ``release.json`` itself is not in ``files``; its line is appended by
    :func:`_write_manifests` after the file exists.
    """
    entries: list[dict[str, object]] = []
    for entry in files.values():
        if isinstance(entry, list):
            entries.extend(entry)  # voices, and the exported graph trees
        elif isinstance(entry, dict):
            entries.append(entry)
    return entries


# ----------------------------------------------------------------- the roster


def roster_names() -> tuple[str, ...]:
    """The canonical twenty, in the order the provenance file lists them.

    Read rather than hard-coded, and checked for shape: twenty voices, two per
    language, ten languages. Without the shape check a truncated provenance
    file would quietly shrink what ``full-0.1`` requires, which is the same
    class of defect as the profile accepting one arbitrary voice.

    Raises:
        BuildRefusedError: the file is absent, malformed, or not the canonical
            roster.
    """
    if not ROSTER_PATH.is_file():
        raise BuildRefusedError(
            [("voice roster", str(ROSTER_PATH), "the roster is the source of truth for the 20")]
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
    uneven = sorted(lang for lang, n in languages.items() if n != ROSTER_PER_LANGUAGE)
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

    The reference voice ships as ``testvoice.safetensors`` — the name the
    quickstart and the tutorials use — not the internal
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

    The header alone: eight bytes of length, then JSON. Nothing here loads a
    tensor, so checking a 1.27 GB pair costs two short reads, which is what
    lets the check sit in the preflight where a refusal is free rather than at
    the end where it has already copied 4.3 GB.

    Raises:
        ValueError: the file is not a safetensors file, or carries no manifest.
    """
    with path.open("rb") as f:
        raw_length = f.read(8)
        if len(raw_length) != 8:
            raise ValueError("too short to be a safetensors file")
        (length,) = struct.unpack("<Q", raw_length)
        if not 0 < length <= 100 << 20:
            raise ValueError(f"implausible header length {length}")
        try:
            header = json.loads(f.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"unreadable safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("the safetensors header is not an object")
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
        want = ARTIFACT_ROLES[name]
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
            # both halves used to read as provenance that matched. A half that
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
    return problems + [
        (f"{name} split.roles", repr(block.get("roles")), f"a release ships {ROLE_FILENAMES}")
        for name, block in blocks.items()
        if block.get("roles") != ROLE_FILENAMES
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


def _pair_refusal(ckpt: Path) -> list[tuple[str, str, str]]:
    """The enrollment half is beside the checkpoint, and the two are one split.

    Split out of :func:`_preflight` so the whole question, is the other file
    there and is it the other half of *this* file, is one call there and one
    place here.
    """
    enrollment = ckpt.parent / ENROLLMENT_CHECKPOINT_NAME
    if not enrollment.is_file():
        return [
            (
                f"enrollment checkpoint ({ENROLLMENT_CHECKPOINT_NAME})",
                str(enrollment),
                "split the packed checkpoint with tools/split_checkpoint.py",
            )
        ]
    if not ckpt.is_file():
        return []  # the missing synthesis half is already on the list
    return _split_problems({CHECKPOINT_NAME: ckpt, ENROLLMENT_CHECKPOINT_NAME: enrollment})


# -------------------------------------------------------------------- refusal


def _preflight(
    *,
    ckpt: Path,
    voice_encoder: Path | None,
    voices: dict[str, Path],
    roster: Sequence[str],
    onnx_dir: Path,
    coreml_dir: Path,
) -> list[tuple[str, str, str]]:
    """What ``full-0.1`` requires and this artefact set does not have.

    Runs before a single byte is copied. The old tool made the voice encoder,
    the voices, the graphs and the packages optional flags and checked only
    the torch path at the end, so a bundle missing the three enrollment graphs
    assembled, verified and published while SUPPORTED.md promised enrollment
    in five languages.

    Returns ``(what, where, how)`` triples, empty when the set is complete.
    """
    missing: list[tuple[str, str, str]] = []

    def need_file(label: str, path: Path, how: str) -> None:
        if not path.is_file():
            missing.append((label, str(path), how))

    def need_dir(label: str, path: Path, how: str) -> None:
        if not path.is_dir():
            missing.append((label, str(path), how))

    # The name is part of the artefact. A checkpoint under any other name
    # would ship, checksum and load on the build machine, and break every
    # guide and every port that names `loudr-1.safetensors`.
    if ckpt.name != CHECKPOINT_NAME:
        missing.append(
            (
                "checkpoint name",
                f"{ckpt.name} (at {ckpt})",
                f"a release ships {CHECKPOINT_NAME}; rename it or pack it again",
            )
        )
    need_file("checkpoint", ckpt, "pack it with tools/pack_assets.py")

    # The other half, and then whether the two are one checkpoint. A release
    # ships both files: which of them a given client downloads is a question
    # for the client, and a bundle carrying only the synthesis half cannot
    # enroll a voice on any port, in any of the five languages SUPPORTED.md
    # declares.
    missing += _pair_refusal(ckpt)

    need_file(
        "manifest.json", ckpt.parent / "manifest.json", "it belongs beside the checkpoint"
    )
    need_file(
        "tokenizer.json", ckpt.parent / "tokenizer.json", "it belongs beside the checkpoint"
    )

    if voice_encoder is None:
        missing.append(
            (
                VOICE_ENCODER_NAME,
                "(no --voice-encoder given)",
                f"pass --voice-encoder {VOICE_ENCODER_NAME}",
            )
        )
    else:
        need_file(
            VOICE_ENCODER_NAME, voice_encoder, f"pass --voice-encoder {VOICE_ENCODER_NAME}"
        )

    # The roster, by name. "At least one voice" accepted a bundle carrying a
    # single arbitrary profile, which loads, speaks and is not the release.
    for name in roster:
        if f"{name}.safetensors" not in voices:
            missing.append(
                (
                    f"voices/{name}.safetensors",
                    "(absent from the voice directory)",
                    "the roster is docs/voices/roster/provenance.json",
                )
            )
    wanted = {f"{name}.safetensors" for name in roster}
    for stranger in sorted(set(voices) - wanted):
        missing.append(
            (
                f"voices/{stranger}",
                str(voices[stranger]),
                "not on the roster; a release ships the roster and nothing else",
            )
        )

    for names, kind, tool in (
        (SYNTHESIS_ONNX, "onnx", "tools/export_onnx.py"),
        (ENROLL_ONNX, "onnx", "tools/export_enroll_onnx.py"),
    ):
        for name in names:
            need_file(f"{kind}/{name}", onnx_dir / name, f"export it: {tool}")
    for names, kind, tool in (
        (SYNTHESIS_COREML, "coreml", "tools/export_coreml.py"),
        (ENROLL_COREML, "coreml", "tools/export_enroll_coreml.py"),
    ):
        for name in names:
            need_dir(f"{kind}/{name}", coreml_dir / name, f"export it: {tool}")

    for requirement in (*DOCUMENTS, BRANDING, *SAMPLES):
        source, name, *_metadata = requirement
        need_file(name, REPO / source, f"it is {source} in this repository")

    # A strict build is only strict if its closing gate can run, and half of
    # that gate is six CoreML packages that open on Apple platforms and
    # nowhere else. Publishing from Linux would ship 589 MB of packages that
    # nothing ever opened — the exact defect this profile exists to prevent —
    # so the refusal is here rather than a skip at the end. `full-0.1` refuses
    # `--skip-verify`, so the gate always runs under this profile and the
    # platform is always its business.
    if sys.platform != "darwin":
        missing.append(
            (
                "Apple platform",
                f"sys.platform is {sys.platform!r}",
                "the six CoreML packages only open on macOS; cut the release there",
            )
        )

    return missing


def _skip_verify_refusal(skip_verify: bool) -> list[tuple[str, str, str]]:
    """``--skip-verify`` and ``full-0.1`` are a contradiction, so it is refused.

    The flag turns off the load-and-speak gate, and the bundle it wrote still
    said ``"profile": "full-0.1"`` and still exited 0. The release checklist
    reads that string as proof of readiness, so an unverified bundle was
    indistinguishable from a verified one by the one thing anybody checks.
    Under ``lenient`` the flag stays: that bundle is already marked
    unreleasable.
    """
    if not skip_verify:
        return []
    return [
        (
            "--skip-verify",
            f"the load-and-speak gate does not run under {STRICT}",
            "a full-0.1 bundle is the claim that it loads and speaks; drop the "
            "flag, or build with --profile lenient",
        )
    ]


def _refuse(problems: Sequence[tuple[str, str, str]], profile: str) -> int:
    width = max(len(what) for what, _, _ in problems)
    print(
        f"\nREFUSING to build a {profile} release: {len(problems)} problem(s).\n",
        file=sys.stderr,
    )
    for what, where, how in problems:
        print(f"  {what:{width}s}  {where}", file=sys.stderr)
        print(f"  {'':{width}s}  -> {how}", file=sys.stderr)
    print(
        "\nNothing was assembled. Fix what is listed, or build a partial\n"
        "bundle with --profile lenient. A lenient bundle is not releasable.",
        file=sys.stderr,
    )
    return 1


# ------------------------------------------------------------- the allowlist


def _allowlist(
    *, roster: Sequence[str], ships_onnx: bool, ships_coreml: bool
) -> tuple[set[str], tuple[str, ...]]:
    """Exactly what a ``full-0.1`` bundle holds.

    Returns ``(paths, package_prefixes)``. The paths are exact; the prefixes
    are the six CoreML packages, whose internal layout belongs to
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
    prefixes = (
        tuple(f"coreml/{name}/" for name in SYNTHESIS_COREML + ENROLL_COREML)
        if ships_coreml
        else ()
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
    * a ``full-0.1`` bundle records ``verified: true``;
    * the bundle contains no symlink, because a bundle is bytes and a link is an
      address that can point outside it;
    * every ``SHA256SUMS`` line hashes to the bytes on disk (``sha256sum -c``,
      in effect), and every file on disk has a line, ``SHA256SUMS`` excepted;
    * ``release.json`` and ``SHA256SUMS`` name the same files with the same
      digests, plus the one line for ``release.json`` itself;
    * a ``full-0.1`` bundle matches the profile's allowlist exactly.
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
    if profile not in PROFILES:
        problems.append(f"release.json names no known profile: {profile!r}")
    if strict and manifest.get("verified") is not True:
        problems.append(
            "release.json says full-0.1 without verified: true; the profile "
            "is the claim that the gate ran and passed"
        )

    entries, bad_lines = _sums_entries(sums_path)
    problems += bad_lines
    problems += _disk_agreement(out, entries)
    problems += _manifest_agreement(manifest, entries)
    if strict:
        problems += _allowlist_agreement(out)
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
                    CHECKPOINT_NAME: out / CHECKPOINT_NAME,
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
    from loudkit.checkpoint import _tensor_payload_sha256

    problems: list[str] = []
    for name in (CHECKPOINT_NAME, ENROLLMENT_CHECKPOINT_NAME):
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
            actual = _tensor_payload_sha256(path)
        except Exception as exc:  # noqa: BLE001 - any reader failure is a finding
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


def _verify_only(out: Path) -> int:
    """``--verify-only``: judge an assembled bundle in place, building nothing.

    The same :func:`check_bundle` the build runs after its closing gate, so an
    operator, or a pre-upload CI step, holds a bundle to exactly the standard
    the builder held it to before the rename. It does not load the model; the
    load-and-speak gate belongs to the build; this is about whether the bytes
    on disk are the bytes the manifests vouch for.
    """
    out = out.resolve()
    if not out.is_dir():
        print(f"FAILED: {out} is not a directory", file=sys.stderr)
        return 1
    print(f"checking {out}")
    problems = check_bundle(out)
    if problems:
        print(
            f"\nFAILED: {len(problems)} problem(s):\n  " + "\n  ".join(problems),
            file=sys.stderr,
        )
        return 1
    manifest = json.loads((out / "release.json").read_text(encoding="utf-8"))
    entries, _ = _sums_entries(out / "SHA256SUMS")
    print(
        f"  profile {manifest['profile']}, verified {manifest['verified']}, "
        f"{len(entries)} files checksummed and re-hashed from disk"
    )
    return 0


# ------------------------------------------------------------------ assembly


def _copy_voices(
    voices: dict[str, Path], roster: Sequence[str], out: Path, *, strict: bool
) -> list[dict[str, object]]:
    """The roster under ``full-0.1``; whatever is present under ``lenient``."""
    names = [f"{name}.safetensors" for name in roster] if strict else sorted(voices)
    copied = [
        copy(voices[name], out / "voices" / name, out, f"voice {Path(name).stem}")
        for name in names
    ]
    if not copied:
        # `full-0.1` refuses before reaching here; under `lenient` a bundle
        # with no voice cannot be used from its own quickstart, which is the
        # only instruction most people will follow.
        print("  WARNING: no voice profiles found — the quickstart will not work")
    return copied


def _assemble(
    out: Path,
    *,
    profile: str,
    ckpt: Path,
    voice_encoder: Path | None,
    voices: dict[str, Path],
    roster: Sequence[str],
    onnx_dir: Path,
    coreml_dir: Path,
    onnx_flag: Path | None,
    coreml_flag: Path | None,
) -> dict[str, Any]:
    """Copy every piece into ``out`` and return the ``release.json`` body."""
    strict = profile == STRICT
    files: dict[str, Any] = {}
    # The profile that built it. Without this a lenient bundle and a release
    # are indistinguishable by machine, and the difference is exactly the one
    # a CI job needs to know.
    files["profile"] = profile
    files["checkpoint"] = copy(ckpt, out / ckpt.name, out, "checkpoint")

    # The enrollment half. `full-0.1` refused this build before a byte moved if
    # it is absent or if the pair does not hold together, so here it is a copy.
    # Under `lenient` it ships when it is there and is noted when it is not.
    enrollment = ckpt.parent / ENROLLMENT_CHECKPOINT_NAME
    if enrollment.is_file():
        files["enrollment_checkpoint"] = copy(
            enrollment, out / ENROLLMENT_CHECKPOINT_NAME, out, "enrollment checkpoint"
        )
    else:
        print(f"  note: no {ENROLLMENT_CHECKPOINT_NAME} beside the checkpoint, cannot enroll")

    files["manifest"] = copy(
        ckpt.parent / "manifest.json", out / "manifest.json", out, "manifest"
    )
    files["tokenizer"] = copy(
        ckpt.parent / "tokenizer.json", out / "tokenizer.json", out, "tokenizer"
    )
    if voice_encoder is not None:
        files["voice_encoder"] = copy(
            voice_encoder, out / VOICE_ENCODER_NAME, out, "voice encoder"
        )

    files["voices"] = _copy_voices(voices, roster, out, strict=strict)

    # The exported graphs, when they sit beside the checkpoint, which is where
    # every backend looks for them and where the export tools write them.
    #
    # Under `full-0.1` the graphs and packages are copied **by name**. Copying
    # the source directories wholesale made the bundle's contents whatever the
    # export directory happened to hold — a stale graph, a `.tmp.mlpackage`
    # left by an interrupted export, an editor's backup file — all of it
    # shipped and checksummed as part of the release.
    if strict:
        files["onnx"] = [
            copy(onnx_dir / name, out / "onnx" / name, out, f"onnx/{name}")
            for name in SYNTHESIS_ONNX + ENROLL_ONNX
        ]
        coreml: list[dict[str, object]] = []
        for name in SYNTHESIS_COREML + ENROLL_COREML:
            coreml.extend(
                copy_tree(coreml_dir / name, out / "coreml" / name, out, f"coreml/{name}")
            )
        files["coreml"] = coreml
        return _with_documents(files, out)

    for kind, source, flag in (
        ("onnx", onnx_dir, onnx_flag),
        ("coreml", coreml_dir, coreml_flag),
    ):
        if source.is_dir():
            files[kind] = copy_tree(source, out / kind, out, kind)
        elif flag is not None:
            raise FileNotFoundError(f"{kind} directory missing: {source}")
        else:
            print(f"  note: no {kind}/ beside the checkpoint — not shipping it")
    return _with_documents(files, out)


def _with_documents(files: dict[str, Any], out: Path) -> dict[str, Any]:
    """The documents, brand image and samples, all covered by the manifests.

    The documents used to be copied and then dropped on the floor: the return
    value went nowhere, so LICENSE, NOTICE, README and RESPONSIBLE_USE shipped
    with no checksum. The samples deliberately take the same route: a model
    card whose player depends on bytes outside its release is not portable.
    """
    for source, name, key in DOCUMENTS:
        files[key] = copy(REPO / source, out / name, out, name)
    source, name, key = BRANDING
    files[key] = copy(REPO / source, out / name, out, name)
    files["samples"] = [copy(REPO / source, out / name, out, name) for source, name in SAMPLES]
    return files


def _write_manifests(
    out: Path, files: dict[str, Any], *, verified: bool
) -> list[dict[str, object]]:
    """``release.json`` then ``SHA256SUMS``, in that order.

    The order is the point. `release.json` carries the profile and the
    verified flag, which is what a consumer reads to decide whether a bundle
    is trustworthy, so it is written first and `SHA256SUMS` covers it.
    Written the other way round it was the one file nothing vouched for.

    `verified` says the closing gate ran and passed. The gate needs the
    assembled bundle, so a build writes both manifests once with
    ``verified: false``, runs the gate, and writes them again.

    Checksum paths are relative to the release root, matching `release.json`
    and the layout a stranger verifies from: `cd dist/loudr-1 && sha256sum -c
    SHA256SUMS`. Using only the basename breaks the voices subdirectory and
    silently skips them.

    `newline=""` keeps the "\\n" written here as the byte written to disk.
    Python's text mode otherwise translates it to CRLF on Windows, and
    `sha256sum -c` treats the CR as part of the filename: every line fails as
    `'loudr-1.safetensors'$'\\r': No such file or directory`, on a release
    whose files are all present and correct.
    """
    files["verified"] = verified
    manifest = out / "release.json"
    manifest.write_text(json.dumps(files, indent=1) + "\n", encoding="utf-8", newline="")
    checksummed = [
        *_checksum_entries(files),
        {
            "path": "release.json",
            "sha256": sha256(manifest),
            "bytes": manifest.stat().st_size,
        },
    ]
    (out / "SHA256SUMS").write_text(
        "".join(f"{e['sha256']}  {e['path']}\n" for e in checksummed),
        encoding="utf-8",
        newline="",
    )
    return checksummed


# ------------------------------------------------------------- the rename


def _staging_dir(out: Path) -> Path:
    """A sibling of the target, so the closing rename is same-filesystem.

    Dot-prefixed and process-stamped: it is not a release, it must not look
    like one to a glob or to a person, and two builds must not collide.
    """
    return out.parent / f".{out.name}.staging-{os.getpid()}"


def _alive(pid: int) -> bool:
    """Whether that pid is still running, including one owned by somebody else.

    On POSIX, ``EPERM`` means the process exists and is not ours to signal.
    Windows is queried through a process handle because its ``os.kill`` is not
    a harmless existence probe.
    """
    if os.name == "nt":
        # Windows' os.kill is not POSIX kill(2): non-console signals call
        # TerminateProcess, so using signal 0 as a probe is not a harmless
        # existence check there. Ask the process handle whether it is signaled
        # instead. Any uncertainty is treated as alive, because this answer
        # gates deletion of another build's staging directory.
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        synchronize = 0x00100000
        wait_object_0 = 0x00000000
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            # ERROR_INVALID_PARAMETER means there is no such PID. Access
            # denied means there is a process, just not one this user may
            # inspect. Unknown errors stay conservative and keep the tree.
            return int(ctypes.get_last_error()) != 87  # type: ignore[attr-defined]
        try:
            state = int(kernel32.WaitForSingleObject(handle, 0))
            return state != wait_object_0
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sweep_stale(out: Path) -> None:
    """Remove the staging trees of builds that died beside this target.

    ``main``'s ``finally`` covers every ending the interpreter lives through
    and nothing else. A SIGSEGV or a SIGKILL leaves the whole 4.6 GB sitting
    beside the target under a dot name, and three killed runs leave three of
    them, on a machine that has to hold two more releases. Signal handlers do
    not close that hole: SIGKILL takes no handler, and a handler running
    inside an already-faulted process is not the thing to hand an ``rmtree``
    to. The pid in the staging name is the mechanism instead: the next run
    reads it, asks whether that process is still running, and reclaims only
    what nobody is filling, so two concurrent builds leave each other alone.

    A ``.previous-`` tree is swept on the same terms, but only when the
    release itself is in place: ``_commit`` moves the old bundle there and
    removes it once the new one has landed, so a leftover *with* a release
    beside it is a duplicate, and a leftover *without* one is the only
    surviving copy of the last release and is left for a person to look at.
    """
    survivors = [
        (stray, stray.name.rsplit("-", 1)[-1])
        for pattern in (f".{out.name}.staging-*", f".{out.name}.previous-*")
        for stray in out.parent.glob(pattern)
    ]
    for stray, pid in survivors:
        if not pid.isdigit() or _alive(int(pid)):
            continue
        if ".previous-" in stray.name and not out.exists():
            continue
        print(f"  reclaiming {stray.name}, left by a build that is gone")
        shutil.rmtree(stray, ignore_errors=True)


def _swap(a: Path, b: Path) -> bool:
    """Atomically exchange ``a`` and ``b``, where the platform can.

    Darwin's ``renamex_np(RENAME_SWAP)`` exchanges two paths in one filesystem
    operation, so there is never an instant at which either name is absent.
    Strict builds are cut on macOS (``_preflight`` requires it), which makes
    this the common case, not the lucky one. ``False`` means the platform or
    the filesystem cannot, and the caller falls back to the two renames whose
    window its docstring describes.
    """
    if sys.platform != "darwin":
        return False
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renamex_np = libc.renamex_np
    except AttributeError:  # pragma: no cover - every macOS since 10.12 has it
        return False
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    rename_swap = 0x2  # RENAME_SWAP, from Darwin's <stdio.h>
    return int(renamex_np(os.fsencode(a), os.fsencode(b), rename_swap)) == 0


def _commit(staging: Path, out: Path) -> None:
    """Rename the staging directory into place, keeping the old one until then.

    Where the filesystem can exchange two directories atomically (macOS APFS,
    via :func:`_swap`), the new bundle replaces the old one with no instant at
    which ``out`` is absent, and the old bundle, now under the staging name,
    is moved to ``.previous-`` and removed.

    On the fallback path the previous bundle is moved aside first and removed
    only after the new one has landed, so an interrupted commit leaves either
    the old release or the new one, never a merge of the two. The window that
    path cannot close: between the two ``os.replace`` calls, ``out`` names
    nothing. A crash inside it leaves no ``out`` and the old bundle intact
    under ``.{out}.previous-{pid}``, which ``_sweep_stale`` refuses to
    reclaim precisely because no release sits beside it, and recovery is one
    rename by a person. A concurrent reader in that window sees "no such
    directory", never a half-written bundle.
    """
    previous = out.parent / f".{out.name}.previous-{os.getpid()}"
    shutil.rmtree(previous, ignore_errors=True)
    if out.exists():
        if _swap(staging, out):
            # `out` is the new bundle; `staging` now holds the old one. Under
            # the `.previous-` name a crash before the rmtree leaves something
            # `_sweep_stale` knows how to judge.
            os.replace(staging, previous)
            shutil.rmtree(previous, ignore_errors=True)
            return
        os.replace(out, previous)
    try:
        os.replace(staging, out)
    except OSError:
        if previous.exists():
            os.replace(previous, out)
        raise
    shutil.rmtree(previous, ignore_errors=True)


# ----------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--voices", type=Path, help="directory of voice profiles")
    ap.add_argument(
        "--voice-encoder",
        type=Path,
        help="ve.safetensors (voice cloning); required for a cloning-capable release",
    )
    ap.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="exported ONNX graphs; defaults to onnx/ beside the checkpoint",
    )
    ap.add_argument(
        "--coreml",
        type=Path,
        default=None,
        help="exported CoreML packages; defaults to coreml/ beside the checkpoint",
    )
    ap.add_argument("--out", type=Path, default=REPO / "dist" / "loudr-1")
    ap.add_argument(
        "--profile",
        choices=PROFILES,
        default=STRICT,
        help=(
            "full-0.1 (default) requires every piece a release ships and refuses "
            "without it; lenient assembles whatever is present and is not releasable"
        ),
    )
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="assemble without the load-and-speak check; refused under full-0.1",
    )
    ap.add_argument(
        "--verify-only",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "judge an assembled bundle in place (re-hash every file, match the "
            "inventory against both manifests, re-audit the allowlist) and "
            "build nothing. The same audit a build runs before its rename."
        ),
    )
    args = ap.parse_args()

    if args.verify_only is not None:
        return _verify_only(args.verify_only)
    if args.checkpoint is None:
        ap.error("--checkpoint is required (or pass --verify-only DIR)")

    strict = args.profile == STRICT
    ckpt = args.checkpoint.resolve()
    out = args.out.resolve()
    voice_encoder = args.voice_encoder.resolve() if args.voice_encoder else None
    voice_dir = (args.voices or ckpt.parent / "voices").resolve()
    onnx_dir = (args.onnx or ckpt.parent / "onnx").resolve()
    coreml_dir = (args.coreml or ckpt.parent / "coreml").resolve()

    try:
        # A collision breaks `SHA256SUMS` under either profile, so it is
        # checked under either profile.
        voices = _voice_sources(voice_dir)
        roster = roster_names() if strict else ()
        if strict:
            missing = _skip_verify_refusal(args.skip_verify)
            missing += _preflight(
                ckpt=ckpt,
                voice_encoder=voice_encoder,
                voices=voices,
                roster=roster,
                onnx_dir=onnx_dir,
                coreml_dir=coreml_dir,
            )
            if missing:
                raise BuildRefusedError(missing)
    except BuildRefusedError as refused:
        return _refuse(refused.problems, args.profile)

    out.parent.mkdir(parents=True, exist_ok=True)
    _sweep_stale(out)
    staging = _staging_dir(out)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    print(f"assembling {out}  (profile {args.profile})")
    try:
        code = _build(
            staging,
            args=args,
            ckpt=ckpt,
            voice_encoder=voice_encoder,
            voices=voices,
            roster=roster,
            onnx_dir=onnx_dir,
            coreml_dir=coreml_dir,
        )
        if code == 0:
            _commit(staging, out)
            print(f"\n  {out}")
        return code
    finally:
        # Whatever happened — a refusal, a failed check, an exception, a
        # Ctrl-C — the staging directory goes. A failed run must leave nothing
        # that looks publishable.
        shutil.rmtree(staging, ignore_errors=True)


def _build(
    staging: Path,
    *,
    args: argparse.Namespace,
    ckpt: Path,
    voice_encoder: Path | None,
    voices: dict[str, Path],
    roster: Sequence[str],
    onnx_dir: Path,
    coreml_dir: Path,
) -> int:
    strict = args.profile == STRICT
    files = _assemble(
        staging,
        profile=args.profile,
        ckpt=ckpt,
        voice_encoder=voice_encoder,
        voices=voices,
        roster=roster,
        onnx_dir=onnx_dir,
        coreml_dir=coreml_dir,
        onnx_flag=args.onnx,
        coreml_flag=args.coreml,
    )
    checksummed = _write_manifests(staging, files, verified=False)
    total = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
    print(f"\n  total {total / 1e9:.2f} GB")

    # Every file in the bundle has a checksum line, or this is not a bundle
    # that vouches for itself. The published 0.1.0 candidate listed 24 files
    # and shipped 3.7 GB, because the graphs and the documents were copied
    # outside `files` and the downloader's checksum pass skips what the
    # manifest does not name.
    uncovered = _uncovered(staging, checksummed)
    if uncovered:
        print(
            "\nFAILED: SHA256SUMS does not cover every file in the bundle:\n  "
            + "\n  ".join(uncovered),
            file=sys.stderr,
        )
        return 1
    files_shipped = len(checksummed) + len(UNCHECKSUMMED)
    print(f"  {files_shipped} files, {len(checksummed)} checksummed")

    if strict:
        paths, prefixes = _allowlist(roster=roster, ships_onnx=True, ships_coreml=True)
        drift = _audit(staging, paths, prefixes)
        if drift:
            print(
                "\nFAILED: the bundle is not what the full-0.1 profile names:\n  "
                + "\n  ".join(drift),
                file=sys.stderr,
            )
            return 1
        print(f"  matches the full-0.1 allowlist ({len(roster)} voices)")

    if args.skip_verify:
        # `full-0.1` refused this flag before anything was copied, so this is
        # a lenient bundle, and it records `"verified": false`.
        print("\nskipped the load-and-speak check — not a releasable build")
        return _closing_audit(staging)

    code = verify(staging, profile=args.profile)
    if code:
        return code

    # The gate ran and passed, so the claim goes into the bundle. Both
    # manifests are written again from the digests taken before the gate:
    # `release.json` gains the flag, and `SHA256SUMS` covers the file it is
    # now in. The digests are deliberately the pre-gate ones: writing them
    # from a fresh hash of the tree would launder whatever the gate did to it,
    # and the audit below exists to catch exactly that.
    _write_manifests(staging, files, verified=True)
    print("  release.json records verified: true")

    # The gate imported the bundle's own code and ran it, which is the one
    # step of this build that is not a copy under this tool's control. So the
    # bundle is judged again, from disk alone: every file re-hashed against
    # the manifests just written, the inventory matched both ways, the
    # allowlist re-audited. A gate that mutated a byte or added a file yields
    # a refusal here, not a bundle stamped verified.
    return _closing_audit(staging)


def _closing_audit(staging: Path) -> int:
    problems = check_bundle(staging)
    if problems:
        print(
            "\nFAILED: the assembled bundle does not survive its own audit:\n  "
            + "\n  ".join(problems),
            file=sys.stderr,
        )
        return 1
    print("  re-hashed from disk: the bundle is what the manifests say")
    return 0


# --------------------------------------------------------------- the closing gate


def verify(out: Path, *, profile: str = STRICT) -> int:
    """Load the assembled release from its own paths and speak.

    Deliberately uses only what is inside ``out``: a release that quietly
    depends on a file left over on the build machine passes every other check
    and fails for everyone else.

    ``full-0.1`` loads what it shipped, and nothing it did not: the
    checkpoint, ``ve.safetensors``, every voice on the roster, the nine ONNX
    graphs and the six CoreML packages. Checking torch alone is what let a
    bundle with no enrollment graphs pass — the torch path never opens them,
    so the piece most users receive was the piece nothing exercised.
    """
    print("\nverifying — loading the release as a stranger would")
    # A build machine with an export directory on either assets variable would
    # otherwise verify somebody else's graphs.
    os.environ.pop("LOUDKIT_ONNX_ASSETS", None)
    os.environ.pop("LOUDKIT_COREML_ASSETS", None)
    sys.path.insert(0, str(REPO / "python"))
    try:
        import loudkit
    except ImportError as exc:
        print(f"  cannot import loudkit: {exc}", file=sys.stderr)
        return 1

    strict = profile == STRICT
    ckpt = _release_checkpoint(out, strict=strict)
    voices = sorted((out / "voices").glob("*.safetensors"))
    if ckpt is None:
        return 1
    if not voices:
        print("  no voice to speak with — cannot verify", file=sys.stderr)
        return 1

    # Each step returns 0 or 1 and prints its own reason. Listed rather than
    # chained so the strict half is one visible block: what a release is
    # checked for, beyond what any bundle is checked for.
    steps: list[Callable[[], int]] = [lambda: _verify_torch(loudkit, ckpt, voices[0])]
    if strict:
        steps += [
            lambda: _verify_voices(loudkit, voices),
            lambda: _verify_voice_encoder(loudkit, out, ckpt),
            lambda: _verify_onnx(loudkit, ckpt, voices[0]),
            lambda: _verify_coreml(ckpt, voices[0]),
        ]
    for step in steps:
        code = step()
        if code:
            return code
    return 0


def _release_checkpoint(out: Path, *, strict: bool) -> Path | None:
    """The bundle's checkpoint, by its canonical name or by ``release.json``.

    Under ``full-0.1`` the name is fixed, so the lookup is a constant. Under
    ``lenient`` it is whatever ``release.json`` recorded — not "any
    .safetensors at the root", since ``ve.safetensors`` is one too.
    """
    if strict:
        return out / CHECKPOINT_NAME
    manifest = json.loads((out / "release.json").read_text(encoding="utf-8"))
    entry = manifest.get("checkpoint")
    if not isinstance(entry, dict):
        print("  release.json names no checkpoint", file=sys.stderr)
        return None
    return out / str(entry["path"])


def _verify_torch(loudkit: ModuleType, ckpt: Path, voice_path: Path) -> int:
    """Speak on the torch path, twice on one seed.

    Determinism is checked here rather than anywhere else because it is the
    property a caller is most likely to build on and least likely to test:
    the same seed and the same text give byte-identical audio, or the release
    does not keep the promise its documents make.
    """
    try:
        engine = loudkit.load(str(ckpt), device="cpu")
        voice = loudkit.VoiceProfile.load(voice_path)
        result = engine.synthesize("The release is assembled.", voice, seed=7)
        again = engine.synthesize("The release is assembled.", voice, seed=7)
    except Exception as exc:  # noqa: BLE001 — any failure here fails the release
        print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"  {engine.describe()}")
    print(f"  spoke {result.duration:.2f}s with {voice_path.stem}, {len(result.tokens)} tokens")
    if not (again.audio == result.audio).all():
        print("  FAILED: same seed produced different audio — determinism is broken")
        return 1
    print("  same seed, byte-identical audio")
    return 0


def _verify_voices(loudkit: ModuleType, voices: Sequence[Path]) -> int:
    """Every shipped profile opens and carries the tensors a voice needs.

    The old gate loaded one voice and spoke with it, so nineteen of the twenty
    were shipped, checksummed and never opened. A truncated or mis-packed
    profile is a per-file defect, and a per-file defect needs a per-file check.
    """
    print(f"  voices: {len(voices)}")
    for path in voices:
        try:
            profile = loudkit.VoiceProfile.load(path)
        except Exception as exc:  # noqa: BLE001 — any failure here fails the release
            print(f"  FAILED to load {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if profile.speaker_embedding.shape != (256,) or profile.flow_embedding.shape != (192,):
            print(
                f"  FAILED: {path.name} carries "
                f"speaker {profile.speaker_embedding.shape}, "
                f"flow {profile.flow_embedding.shape}",
                file=sys.stderr,
            )
            return 1
    print(f"    all {len(voices)} load, embeddings the right shape")
    return 0


def _verify_voice_encoder(loudkit: ModuleType, out: Path, ckpt: Path) -> int:
    """Clone a voice with the shipped ``ve.safetensors``.

    ``ve.safetensors`` was copied and checksummed and never opened, so a
    release could ship voice cloning that does not clone. This enrolls the
    fixture's reference recording through the bundle's own voice encoder and
    checks the two embeddings against the fixture, which is the same spec
    every port is held to.
    """
    import numpy as np

    fixture = _fixture()
    if fixture is None:
        return 1
    print("  voice encoder:")
    audio = np.frombuffer((fixture / "ref_audio.f32").read_bytes(), dtype=np.float32)
    try:
        cloned = loudkit.enroll(
            audio,
            str(ckpt),
            name="release-check",
            voice_encoder_weights=str(out / VOICE_ENCODER_NAME),
        )
    except Exception as exc:  # noqa: BLE001 — any failure here fails the release
        print(f"  FAILED to enroll: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for label, got, want_name in (
        ("speaker", cloned.speaker_embedding, "speaker_embedding.f32"),
        ("flow", cloned.flow_embedding, "flow_embedding.f32"),
    ):
        want = np.frombuffer((fixture / want_name).read_bytes(), dtype=np.float32)
        c = _cos(np.asarray(got, dtype=np.float32).ravel(), want)
        if c <= 0.999:
            print(f"  FAILED: cloned {label} cosine {c:.6f} <= 0.999", file=sys.stderr)
            return 1
        print(f"    {label:8s} cosine {c:.6f}")
    return 0


def _verify_onnx(loudkit: ModuleType, ckpt: Path, voice_path: Path) -> int:
    """Speak on the six synthesis graphs, then enroll on the three others."""
    print("  onnx path:")
    try:
        engine = loudkit.load(str(ckpt), device="onnx")
        voice = loudkit.VoiceProfile.load(voice_path)
        result = engine.synthesize("The release is assembled.", voice, seed=7)
    except Exception as exc:  # noqa: BLE001 — any failure here fails the release
        print(f"  FAILED on the onnx path: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"    {engine.describe()}")
    print(f"    spoke {result.duration:.2f}s, {len(result.tokens)} tokens")
    return _enrollment_gate("onnx", ckpt.parent / "onnx", _run_onnx_enrollment)


def _verify_coreml(ckpt: Path, voice_path: Path) -> int:
    """Speak on the three synthesis packages, then enroll on the three others.

    The six packages were shipped and checksummed and never opened. They are
    the only artefact in the bundle that a non-Apple machine cannot open at
    all, which is why ``_preflight`` refuses a strict build off macOS rather
    than skipping this quietly at the end. The skip below is named and
    printed rather than silent, and is reachable only by a caller of
    :func:`verify` on a bundle it did not build; a ``full-0.1`` build is
    refused before it gets here.
    """
    if sys.platform != "darwin":
        print("  coreml path: SKIPPED — not an Apple platform (sys.platform != 'darwin')")
        return 0
    print("  coreml path:")
    spoke = _speak_coreml(ckpt, voice_path)
    if spoke is None:
        return 1
    print(f"    {spoke['describe']}")
    print(f"    spoke {spoke['duration']:.2f}s, {spoke['tokens']} tokens")
    return _enrollment_gate("coreml", ckpt.parent / "coreml", _run_coreml_enrollment)


# Speaking on the coreml device used to kill whatever process did it, about a
# second after the last predict and from a thread that process does not own:
# CoreML lets its execution stream linger, then resets it on a dispatch queue
# (`-[MLE5ExecutionStream resetAfterLingering:]`), and the `MLFeatureValue`
# deallocs underneath freed the Python objects coremltools handed them,
# calling `_PyObject_Free` on a background thread without the GIL. The
# backend's `_PinnedInputs` fixes that by keeping one interpreter-owned buffer
# per input, so CoreML never holds the last reference (see
# `loudkit.backends.coreml_backend` and apple/coremltools#2827).
#
# The child below is the regression detector for that fix, so it must do the
# opposite of what its first version did. That version escaped through
# `os._exit(0)` milliseconds after the predict, which was the only way to
# survive the teardown before the fix, and is exactly the wrong thing now:
# a clean, ordinary exit after the linger window is the property under test.
# So the child
#
#   - runs several create / synthesize / destroy cycles, because a single
#     render never reuses the pinned buffers and a single teardown never
#     follows a reuse;
#   - outlives the measured one-second linger by a five-second margin after
#     **every** render, with the engine still alive, because that is when the
#     reset queue frees the inputs of a finished predict. Both halves of that
#     were measured to matter on this fault: tearing the engine down first
#     deallocs the model on the main thread and hides it, and a second render
#     issued inside the first one's linger also hides it, so back-to-back
#     renders with one wait at the end detect nothing. The wait churns small
#     allocations, because the fault needs pymalloc in use when an off-thread
#     free corrupts it, and sleeping is a weaker detector;
#   - then drops the engine and collects, and churns a moment longer, because
#     releasing the model must tear the stream down here, on the main thread,
#     not on the reset queue;
#   - exits through the interpreter's normal shutdown.
_COREML_LINGER_SECONDS = 5.0
_COREML_CYCLES = 3
_COREML_TIMEOUT_SECONDS = 1800.0
"""Hard ceiling on each CoreML child process.

A first CoreML load compiles the packages, which can take minutes; a healthy
gate run is comfortably inside half an hour. Without a ceiling a child hung
inside a predict hangs the whole build with no verdict, which is worse than a
failure: nothing says the gate did not pass. A timeout is a refusal like any
other.
"""
_COREML_SPEAK = """
import gc, json, sys, time
sys.path.insert(0, sys.argv[1])
import loudkit
voice = loudkit.VoiceProfile.load(sys.argv[3])
cycles, linger = int(sys.argv[4]), float(sys.argv[5])

def churn(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        [object() for _ in range(2000)]
        time.sleep(0.01)

for _ in range(cycles):
    engine = loudkit.load(sys.argv[2], device="coreml")
    for _ in range(2):  # the second render is the one that reuses the buffers
        result = engine.synthesize("The release is assembled.", voice, seed=7)
        print("SPOKE " + json.dumps({
            "describe": engine.describe(),
            "duration": result.duration,
            "tokens": len(result.tokens),
        }), flush=True)
        churn(linger)   # the engine is alive; the lingering stream resets now
    del engine, result  # teardown must happen here, on the main thread
    gc.collect()
    churn(1.0)
print("SURVIVED", flush=True)
"""


def _speak_coreml(ckpt: Path, voice_path: Path) -> dict[str, Any] | None:
    """Load the bundle on the coreml device and speak, in a child process.

    Same load and same call a caller makes, on the packages the bundle ships.
    The child process is not an escape hatch any more; it is the measurement.
    The pass condition is threefold, and the exit status comes first: a child
    that printed ``SPOKE`` and then took a signal is precisely the regression
    the backend's ``_PinnedInputs`` exists to prevent, and its stdout looks
    like success up to the last line. Correct audio from a process that then
    dies is a failing release, not a passing one.
    """
    import subprocess

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _COREML_SPEAK,
                str(REPO / "python"),
                str(ckpt),
                str(voice_path),
                str(_COREML_CYCLES),
                str(_COREML_LINGER_SECONDS),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=_COREML_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # A hung child is a failing gate, not a stalled build: kill it (the
        # exception handler has already done so) and refuse with a verdict.
        print(
            f"  FAILED on the coreml path: the speak process was still running "
            f"after {_COREML_TIMEOUT_SECONDS:.0f}s and was killed",
            file=sys.stderr,
        )
        return None

    def fail(reason: str) -> None:
        print(
            f"  FAILED on the coreml path: {reason}\n"
            + "\n".join(f"    {line}" for line in proc.stderr.strip().splitlines()[-20:]),
            file=sys.stderr,
        )

    if proc.returncode != 0:
        # `returncode` is named because a signal shows up here as a negative
        # number and nothing else in the output would say so.
        signal = f" (signal {-proc.returncode})" if proc.returncode < 0 else ""
        fail(
            f"the speak process exited {proc.returncode}{signal} instead of "
            f"surviving {_COREML_LINGER_SECONDS:.0f}s past its last predict"
        )
        return None
    spoken = next(
        (line for line in reversed(proc.stdout.splitlines()) if line.startswith("SPOKE ")),
        None,
    )
    if spoken is None:
        fail("the speak process exited 0 without speaking")
        return None
    lines = proc.stdout.splitlines()
    if not lines or lines[-1] != "SURVIVED":
        fail("the speak process never reported SURVIVED after the linger window")
        return None
    result: dict[str, Any] = json.loads(spoken[len("SPOKE ") :])
    return result


# ----------------------------------------------------------- shared enrollment

_EnrollRunner = Callable[[Path, "dict[str, dict[str, np.ndarray]]"], "dict[str, np.ndarray]"]


def _fixture() -> Path | None:
    fixture = REPO / "tests" / "data" / "enrollment"
    if not (fixture / "manifest.json").is_file():
        print(
            f"  FAILED: the enrollment fixture is missing: {fixture}\n"
            "    rebuild it with tools/make_enrollment.py",
            file=sys.stderr,
        )
        return None
    return fixture


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _partials(mel: np.ndarray) -> np.ndarray:
    """``_VoiceEncoder.embed``'s partial windows: stride 77, window 160.

    Zero-padded so the last one is full. Host orchestration in every port,
    which is why it is not inside the graph.
    """
    import numpy as np

    step, window = 77, 160
    n_wins, remainder = divmod(max(len(mel) - window + step, 0), step)
    if n_wins == 0 or (remainder + (window - step)) / window >= 0.8:
        n_wins += 1
    target = window + step * (n_wins - 1)
    if target > len(mel):
        mel = np.concatenate([mel, np.zeros((target - len(mel), mel.shape[1]), np.float32)])
    return np.stack([mel[i * step : i * step + window] for i in range(n_wins)]).astype(
        np.float32
    )


def _enrollment_feeds(fixture: Path) -> dict[str, dict[str, np.ndarray]]:
    """The three graphs' inputs, read from the fixture once.

    One enrollment, three graphs. The old gate opened the fixture manifest,
    reshaped its arrays and built a runtime session once **per graph** — three
    passes over the same enrollment, which is slower and, more to the point,
    is not what an enrollment is: the three graphs are stages of one
    operation, and running them as one is the closer imitation of a caller.
    """
    import numpy as np

    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))

    def read(name: str, dtype: type) -> np.ndarray:
        shape = manifest["files"][name]["shape"]
        flat: np.ndarray = np.frombuffer((fixture / name).read_bytes(), dtype=dtype)
        return np.asarray(flat.reshape(shape))

    return {
        "s3_tokenizer": {"mel": read("tokenizer_mel.f32", np.float32)[None].astype(np.float32)},
        "camp": {"fbank": read("kaldi_fbank.f32", np.float32).T[None].astype(np.float32)},
        "voice_encoder": {"partials": _partials(read("voiceenc_mel.f32", np.float32))},
    }


def _enrollment_gate(label: str, assets: Path, runner: _EnrollRunner) -> int:
    """Run one enrollment through the three graphs and check it against the fixture.

    The fixture (``tests/data/enrollment``) is the spec every port is held to,
    so the graphs are checked against real DSP inputs and the shipped outputs:
    the tokenizer exactly, the two encoders to cosine > 0.9999. Present but
    broken is the failure this profile exists to prevent, and a graph nobody
    runs is indistinguishable from a graph that is not there.
    """
    import numpy as np

    fixture = _fixture()
    if fixture is None:
        return 1
    feeds = _enrollment_feeds(fixture)
    print(f"    enrollment graphs ({label}):")
    try:
        got = runner(assets, feeds)
    except Exception as exc:  # noqa: BLE001 — any failure here fails the release
        print(f"  FAILED ({label}): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))

    def read(name: str, dtype: type) -> np.ndarray:
        shape = manifest["files"][name]["shape"]
        flat: np.ndarray = np.frombuffer((fixture / name).read_bytes(), dtype=dtype)
        return np.asarray(flat.reshape(shape))

    want_tokens = read("prompt_tokens.i64", np.int64)
    if not np.array_equal(got["s3_tokenizer"].astype(np.int64).ravel(), want_tokens):
        print(
            f"  FAILED ({label}): s3_tokenizer does not reproduce the fixture tokens",
            file=sys.stderr,
        )
        return 1
    print(f"      s3_tokenizer   {len(want_tokens)} tokens, exact")

    camp = got["camp"].astype(np.float32).ravel()
    pooled = got["voice_encoder"].astype(np.float32).mean(0)
    pooled = pooled / np.linalg.norm(pooled)
    for graph, vector, want in (
        ("camp", camp, read("flow_embedding.f32", np.float32)),
        ("voice_encoder", pooled, read("speaker_embedding.f32", np.float32)),
    ):
        c = _cos(vector, want)
        if c <= 0.9999:
            print(f"  FAILED ({label}): {graph} cosine {c:.6f} <= 0.9999", file=sys.stderr)
            return 1
        print(f"      {graph:14s} cosine {c:.6f}")
    return 0


def _run_onnx_enrollment(
    assets: Path, feeds: dict[str, dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    import numpy as np
    import onnxruntime as ort

    out: dict[str, np.ndarray] = {}
    for stem, feed in feeds.items():
        session = ort.InferenceSession(
            str(assets / f"{stem}.onnx"), providers=["CPUExecutionProvider"]
        )
        out[stem] = np.asarray(session.run(None, feed)[0])
    return out


# coremltools' in-process ``MLModel.predict`` segfaults on these graphs — the
# same crash ``docs/platforms/apple.md`` records for the T3 decode loop and
# ``tools/export_enroll_coreml.py`` works around the same way. One subprocess
# runs the whole enrollment, so the three packages still cost one process.
_COREML_ENROLL = """
import sys, numpy as np, coremltools as ct
assets, in_npz, out_npz = sys.argv[1:4]
feed = dict(np.load(in_npz, allow_pickle=False))
out = {}
for stem in ("s3_tokenizer", "camp", "voice_encoder"):
    # Swift's Enroller and the exporter both declare these graphs CPU-only.
    # The release gate must validate that shipped placement, not Core ML's
    # default ALL placement, which may try and fail to build an ANE plan.
    model = ct.models.MLModel(
        f"{assets}/{stem}.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY
    )
    inputs = {k.split(".", 1)[1]: v for k, v in feed.items() if k.startswith(stem + ".")}
    out[stem] = np.asarray(next(iter(model.predict(inputs).values())))
np.savez(out_npz, **out)
"""


def _run_coreml_enrollment(
    assets: Path, feeds: dict[str, dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    import subprocess
    import tempfile

    import numpy as np

    # `dict[str, Any]`: numpy types `savez`'s second positional as `allow_pickle`,
    # so a `**` splat of concrete arrays does not type-check against it.
    flat: dict[str, Any] = {
        f"{stem}.{k}": v for stem, feed in feeds.items() for k, v in feed.items()
    }
    with tempfile.TemporaryDirectory() as tmp:
        in_npz, out_npz = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(in_npz, **flat)
        # Same ceiling as the speak gate: a child hung inside a predict must
        # end the build with a verdict, not hold it open.
        subprocess.run(
            [sys.executable, "-c", _COREML_ENROLL, str(assets), str(in_npz), str(out_npz)],
            check=True,
            timeout=_COREML_TIMEOUT_SECONDS,
        )
        with np.load(out_npz) as z:
            got = {k: np.asarray(z[k]) for k in z.files}
    # The base-3 FSQ fold is on the host for the CoreML export: coremltools'
    # int64 output segfaults in-process, so `s3_tokenizer.mlpackage` returns
    # the 8 float dims and the port folds them. `tools/export_enroll_coreml.py`
    # does the same, and this mirrors it rather than inventing a second rule.
    powers: np.ndarray = 3 ** np.arange(8, dtype=np.float32)
    got["s3_tokenizer"] = (np.squeeze(got["s3_tokenizer"], 0) * powers).sum(-1).astype(np.int64)
    return got


if __name__ == "__main__":
    raise SystemExit(main())
