"""What a release is, and how to find your way around one on disk.

One repository serves every backend, so what varies is the fetch, not the
layout. This module holds the names a release ships its files under, the fetch
plan per backend (:func:`release_patterns`), the inventory a fetched directory
must add up to (:func:`verify_release_inventory`), the resolvers that find the
synthesis and enrollment artefacts inside a directory by canonical name and by
the ``artifact_role`` their manifests declare, and the release record: which
``release.json`` profiles count as a release under ``loudreader/``. It reads
files that are already here and never fetches; :mod:`loudkit.hub` fetches, and
:mod:`loudkit.checksums` hashes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint import (
    ASSET_PREFIX,
    TOKENIZER_FILENAME,
    decode_mode,
    read_manifest,
    require_decode_support,
)
from .errors import VoiceNotFoundError

BACKENDS = ("torch", "onnx", "coreml")
"""The backends a release can be fetched for. :func:`release_patterns` refuses
anything outside it rather than falling back to torch, whose set is a strict
subset of the other two and would answer a typo with a plan that fetches no
graphs."""

CHECKPOINT_GLOB = "*.safetensors"
"""What a checkpoint looks like inside a released repo."""

CHECKPOINT_NAME = "loudr-1.safetensors"
"""The synthesis artefact's canonical name in a release. Resolution is by this
name and by the manifest's ``artifact_role``, never by counting files."""

TURBO_CHECKPOINT_NAME = "loudr-1-turbo.safetensors"
"""The synthesis artefact's canonical name in a ``loudr-1-turbo`` release. It
differs from :data:`CHECKPOINT_NAME` so two models unpacked into one directory
get "name the one you mean" rather than whichever landed last."""

CHECKPOINT_NAMES = (CHECKPOINT_NAME, TURBO_CHECKPOINT_NAME)
"""Every name a release ships its synthesis artefact under, one per model."""

ENROLLMENT_NAME = "loudr-1-enrollment.safetensors"
"""The enrollment artefact: the two modules synthesis never opens. Fetched only
when the caller asked for the cloning capability."""

SYNTHESIS_ROLE = "synthesis"
"""``manifest["artifact_role"]`` of the file :func:`_only_checkpoint_in` returns.
A manifest with no role is a pre-split checkpoint holding every tensor, which
still loads: the field is read to refuse a file, never to require one."""

ENROLLMENT_ROLE = "enrollment"
"""``manifest["artifact_role"]`` of the file :func:`_enrollment_in` returns."""

VOICE_DIR = "voices"
"""Where voices live inside a release."""

VOICE_SUFFIX = ".safetensors"
"""A voice file's extension. The same container as the checkpoint; only the
directory tells them apart."""

VOICE_ENCODER_NAME = "ve.safetensors"
"""The utterance voice encoder, at a release's root. A derived work with its
own attribution (see NOTICE), so it ships beside the checkpoint."""

_ONNX_SYNTHESIS = (
    "onnx/t3_cond.onnx",
    "onnx/t3_prefill.onnx",
    "onnx/t3_step.onnx",
    "onnx/flow_encoder.onnx",
    "onnx/flow_estimator.onnx",
    "onnx/vocoder.onnx",
    # The record that says these six came from one export. Fetched with the
    # graphs it describes, so the backend's mixed-set check is live for
    # everyone who fetched the ordinary way.
    "onnx/export.json",
)
_ONNX_ENROLL = ("onnx/s3_tokenizer.onnx", "onnx/camp.onnx", "onnx/voice_encoder.onnx")
_COREML_SYNTHESIS = (
    "coreml/flow_encoder.mlpackage/*",
    "coreml/flow_estimator.mlpackage/*",
    "coreml/vocoder.mlpackage/*",
)
EXPORT_RECORD = "export.json"
"""What each exporter writes beside its graphs, naming the checkpoint they came
from. Fetched with them, required by nobody."""

_COREML_RECORD = f"coreml/{EXPORT_RECORD}"
"""Kept out of `_COREML_SYNTHESIS`, which is also the package inventory."""
_COREML_ENROLL = (
    "coreml/s3_tokenizer.mlpackage/*",
    "coreml/camp.mlpackage/*",
    "coreml/voice_encoder.mlpackage/*",
)
_RELEASE_CORE = (
    # `*.safetensors` is the checkpoint and, nested, every voice; ve.safetensors
    # and the enrollment artefact are carved back out below when cloning was not
    # asked for. The JSON names are spelled out rather than `*.json`, which also
    # matches the small manifests inside every CoreML package.
    "*.safetensors",
    "manifest.json",
    TOKENIZER_FILENAME,
    "release.json",
    "voices/*",
    "SHA256SUMS",
)

_OFFICIAL_ORG = "loudreader"
"""The org whose releases this library vouches for. Everything under it is
built by ``tools/build_release.py``, which always writes ``SHA256SUMS`` and
``release.json``."""

_STRICT_PROFILES = frozenset({"full-0.1", "turbo-0.1"})
"""The profiles a builder stamps on a releasable bundle: ``full-0.1`` for
``loudr-1`` and ``turbo-0.1`` for ``loudr-1-turbo``. Neither is written until
the builder's load-and-speak gate passes."""


def backend_for_device(device: str | None) -> str:
    """Which release set a device needs. ``cuda:1`` and the rest are torch
    devices, so only the stem is read."""
    base = (device or "").split(":", 1)[0]
    return base if base in ("onnx", "coreml") else "torch"


def release_patterns(
    backend: str, *, cloning: bool = False
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(allow_patterns, ignore_patterns)`` for a selective release fetch.

    Every set carries the synthesis checkpoint, the tokenizer and manifest,
    all the voices and the release record; ``onnx`` and ``coreml`` add their
    exported graphs. ``cloning`` adds what that backend enrols with: torch
    uncovers the enrollment artefact and ``ve.safetensors``, the graph
    backends add their three enrollment graphs and keep those two ignored,
    because the ports carry their own enrollers.

    Raises:
        ValueError: for a backend outside :data:`BACKENDS`.
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"{backend!r} is not a backend this release ships. Pass one of "
            + ", ".join(BACKENDS)
            + "."
        )
    allow = list(_RELEASE_CORE)
    if backend == "onnx":
        allow += (*_ONNX_SYNTHESIS, "onnx/t3_pair_step.onnx", "onnx/t3_head2.onnx")
        if cloning:
            allow += _ONNX_ENROLL
    elif backend == "coreml":
        allow += [*_COREML_SYNTHESIS, _COREML_RECORD, "coreml/t3_*.mlpackage/*"]
        if cloning:
            allow += _COREML_ENROLL
    if cloning and backend == "torch":
        ignore: tuple[str, ...] = ()
    else:
        ignore = (VOICE_ENCODER_NAME, ENROLLMENT_NAME)
    return tuple(allow), ignore


def verify_release_inventory(
    root: Path,
    backend: str,
    *,
    cloning: bool = False,
    require_voices: bool = False,
) -> None:
    """The inventory :func:`release_patterns` promised must be on disk.

    ``allow_patterns`` is a request, not a receipt: the hub client fetches
    whatever subset the repo holds, so a release missing its graphs comes
    back looking like one that has them. Every backend needs a synthesis
    checkpoint, ``manifest.json`` and ``tokenizer.json`` (unless the
    checkpoint packs it); ``coreml`` packages must have the structure of one.
    ``require_voices`` is the caller's claim that a voiceless fetch is a
    failure, true for ``loudkit download`` and for official releases.

    Raises:
        FileNotFoundError: naming every absent piece.
        ValueError: for a backend outside :data:`BACKENDS`, or a checkpoint
            whose decode loop this build does not run.
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"{backend!r} is not a backend this release ships. Pass one of "
            + ", ".join(BACKENDS)
            + "."
        )
    missing: list[str] = []
    expected: list[str] = ["manifest.json"]
    try:
        checkpoint: Path | None = _only_checkpoint_in(root)
    except FileNotFoundError:
        checkpoint = None
        missing.append(" or ".join(CHECKPOINT_NAMES))
    if checkpoint is not None:
        _refuse_unrunnable_decode(checkpoint, backend)
    if checkpoint is None or not _packs_tokenizer(checkpoint):
        expected.append(TOKENIZER_FILENAME)
    mode = decode_mode(_manifest_of(checkpoint) or {}) if checkpoint is not None else "single"
    extra_expected, absent_packages = _backend_extras(root, backend, cloning=cloning, mode=mode)
    expected += extra_expected
    missing += absent_packages
    # The same condition release_patterns fetches under: requiring what the
    # plan did not fetch turns a correct download into an error.
    if cloning and backend == "torch":
        expected.append(VOICE_ENCODER_NAME)
        try:
            _enrollment_in(root)
        except FileNotFoundError:
            missing.append(ENROLLMENT_NAME)
    missing += [rel for rel in expected if not (root / rel).is_file()]
    if require_voices and not any(
        p.is_file() for p in (root / VOICE_DIR).glob(f"*{VOICE_SUFFIX}")
    ):
        missing.append(f"{VOICE_DIR}/*{VOICE_SUFFIX}")
    if missing:
        raise FileNotFoundError(
            f"{root}: this fetch does not add up to a usable {backend} set: "
            "missing: " + ", ".join(sorted(missing)) + ". The release does not "
            "carry these files, or the fetch was interrupted; retry, or pin a "
            "revision that ships them."
        )


def _backend_extras(
    root: Path, backend: str, *, cloning: bool, mode: str
) -> tuple[list[str], list[str]]:
    """What one backend needs beyond the basics, as ``(expected, missing)``.

    ONNX adds filenames, judged with everything else at the end; CoreML adds
    packages, whose absence is judged by looking at the tree, so those are
    settled here.
    """
    if backend == "onnx":
        # The export record is fetched with the graphs and not required: a
        # set exported before the record existed keeps working.
        expected = [g for g in _ONNX_SYNTHESIS if not g.endswith(EXPORT_RECORD)]
        if mode == "fusion_mtp2":
            expected.remove("onnx/t3_step.onnx")
            expected += ["onnx/t3_pair_step.onnx", "onnx/t3_head2.onnx"]
        return expected + list(_ONNX_ENROLL if cloning else ()), []
    if backend == "coreml":
        packages = _COREML_SYNTHESIS + (_COREML_ENROLL if cloning else ())
        if mode == "fusion_mtp2" or any((root / "coreml").glob("t3_*.mlpackage")):
            stages = (
                ("cond", "prefill", "pair_step", "head2")
                if mode == "fusion_mtp2"
                else ("cond", "prefill", "step")
            )
            packages += tuple(f"coreml/t3_{stage}.mlpackage/*" for stage in stages)
        rels = [pattern.removesuffix("/*") for pattern in packages]
        return [], [rel for rel in rels if not _is_mlpackage(root / rel)]
    return [], []


def _refuse_unrunnable_decode(checkpoint: Path, backend: str) -> None:
    """Refuse a fetch for a checkpoint whose decode loop this build does not
    run, with the sentence :func:`~loudkit.backends.build_engine` says on a
    local path, so a user who hits both doors gets one diagnosis. The mode is
    the whole test; ``backend`` only names the door."""
    manifest = _manifest_of(checkpoint) or {}
    require_decode_support(
        decode_mode(manifest),
        backend,
        name=str(manifest.get("name") or checkpoint.name),
    )


def _is_mlpackage(node: Path) -> bool:
    """Whether ``node`` has the structure of a CoreML package: a
    ``Manifest.json`` beside a non-empty ``Data/``. "A directory with
    anything in it" passes for a package whose weights never arrived."""
    if not node.is_dir():
        return False
    data = node / "Data"
    return (node / "Manifest.json").is_file() and data.is_dir() and any(data.iterdir())


def _packs_tokenizer(checkpoint: Path) -> bool:
    """Whether ``checkpoint`` carries ``tokenizer.json`` as a packed asset.
    ``False`` for anything unreadable, so it is asked for the sibling."""
    try:
        from safetensors import safe_open

        with safe_open(str(checkpoint), framework="numpy") as f:
            return f"{ASSET_PREFIX}tokenizer.json" in f.keys()  # noqa: SIM118
    except Exception:  # not a readable checkpoint; require the sibling
        return False


def _artifact_role(path: Path) -> str | None:
    """``manifest["artifact_role"]`` for ``path``, or ``None`` for no claim: a
    pre-split checkpoint, or a file this process cannot read. A missing claim
    never promotes a file to a role it did not ask for."""
    role = (_manifest_of(path) or {}).get("artifact_role")
    return role if isinstance(role, str) else None


def _manifest_of(path: Path) -> dict[str, Any] | None:
    """``path``'s embedded manifest, or ``None`` if it has none this can read."""
    try:
        return read_manifest(path)
    except Exception:  # unreadable is "no claim", never a role
        return None


def _refuse_role(path: Path, expected: str) -> None:
    """Refuse a file whose manifest declares it to be the other artefact."""
    role = _artifact_role(path)
    if role is not None and role != expected:
        raise FileNotFoundError(
            f"{path}: this is a release's {role} artefact, and the {expected} "
            "artefact is what was asked for. Pass the release directory, or "
            "the repo id, and let the resolver pick."
        )


def _split_source(path: Path) -> str | None:
    """``manifest["split"]["source_payload_sha256"]``, or ``None``: the digest
    of the packed original the splitting tool stamps into both halves."""
    split = (_manifest_of(path) or {}).get("split")
    if not isinstance(split, dict):
        return None
    source = split.get("source_payload_sha256")
    return source if isinstance(source, str) else None


def _refuse_mismatched_pair(synthesis: Path, enrollment: Path) -> None:
    """Refuse two halves that came from different packing runs.

    Mismatched halves both load and the voice is wrong, with no error to show
    for it. Silent when either file makes no claim.

    Raises:
        ValueError: a bad pair, not a missing file, so the inventory check
            reports it rather than folding it into "did not come".
    """
    left, right = _split_source(synthesis), _split_source(enrollment)
    if left is not None and right is not None and left != right:
        raise ValueError(
            f"{synthesis.name} and {enrollment.name} are halves of different "
            "packing runs (their manifests record different "
            f"split.source_payload_sha256: {left[:12]}... and {right[:12]}...). "
            "Enrolling from a mismatched pair produces a voice that is wrong "
            "with no error to show for it. Fetch both halves from one release."
        )


def _is_official(repo: str | None) -> bool:
    """Whether ``repo`` names a release this project publishes."""
    return repo is not None and repo.split("/", 1)[0].lower() == _OFFICIAL_ORG


def _require_releasable(root: Path, where: str) -> None:
    """An official snapshot must be a strict, gate-verified release.

    Checksums say the bytes arrived intact and nothing about what they are: a
    development bundle carries a perfectly valid ``SHA256SUMS``.
    """
    manifest = root / "release.json"
    if not manifest.is_file():
        raise ValueError(
            f"{where}: no release.json. Every {_OFFICIAL_ORG} release records "
            "its profile and its verified flag there, so this snapshot cannot "
            "prove it is a release. Retry the download, and pin a revision you "
            "trust."
        )
    _check_release_record(manifest, where)


def _check_release_record(manifest: Path, where: str) -> None:
    """``release.json`` must name a release profile and record ``verified: true``."""
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(
            f"{where}: release.json is unreadable ({exc}). Delete the cached "
            "snapshot and retry, or pin a revision you trust."
        ) from exc
    profile = record.get("profile") if isinstance(record, dict) else None
    if profile not in _STRICT_PROFILES:
        raise ValueError(
            f"{where}: release.json says profile {profile!r}, and an "
            f"{_OFFICIAL_ORG} release is one of "
            f"{', '.join(sorted(_STRICT_PROFILES))}. This is a development "
            "bundle, not the release. Fetch a published revision, or pass a "
            "local path if the bundle is your own."
        )
    if not isinstance(record, dict) or record.get("verified") is not True:
        raise ValueError(
            f"{where}: release.json does not record verified: true, so the "
            "bundle never passed the builder's load-and-speak gate. Rebuild it "
            "with tools/build_release.py, or fetch a published revision."
        )


def _root_checkpoints(directory: Path) -> list[Path]:
    """Every root-level file that could be a checkpoint, sorted.

    Non-recursive, so voices under ``voices/`` are not candidates. The voice
    encoder is a fixed name and excluded by it. A dot-prefixed file is not
    something ``build_release`` can produce, so it is not a candidate either.
    """
    return sorted(
        p
        for p in directory.glob(CHECKPOINT_GLOB)
        if p.is_file() and p.name != VOICE_ENCODER_NAME and not p.name.startswith(".")
    )


def _only_checkpoint_in(directory: Path) -> Path:
    """The synthesis artefact in ``directory``, or a useful complaint.

    Three rules, in order: exactly one canonical name; otherwise the file that
    declares the synthesis role; otherwise exactly one candidate, with anything
    declaring the enrollment role set aside first.
    """
    canonical = [directory / name for name in CHECKPOINT_NAMES if (directory / name).is_file()]
    if len(canonical) == 1:
        _refuse_role(canonical[0], SYNTHESIS_ROLE)
        return canonical[0]
    if canonical:
        names = ", ".join(p.name for p in canonical)
        raise FileNotFoundError(
            f"{directory}: {len(canonical)} models here ({names}): name the one "
            "you mean. They are different releases with different decode loops, "
            "so there is no right one to pick."
        )
    found = _root_checkpoints(directory)
    roles = {p: _artifact_role(p) for p in found}
    declared = [p for p in found if roles[p] == SYNTHESIS_ROLE]
    candidates = declared or [p for p in found if roles[p] != ENROLLMENT_ROLE]
    if len(candidates) == 1:
        return candidates[0]
    if not found:
        raise FileNotFoundError(
            f"{directory}: no {CHECKPOINT_GLOB} here. A loudkit release is a "
            "synthesis checkpoint and a voices/ directory, with the enrollment "
            "checkpoint beside them when it ships."
        )
    if not candidates:
        raise FileNotFoundError(
            f"{directory}: the only checkpoint here is a release's enrollment "
            f"artefact ({', '.join(p.name for p in found)}). It carries the two "
            "modules a clone needs and nothing synthesis reads. Fetch the "
            f"release's {CHECKPOINT_NAME} beside it."
        )
    names = ", ".join(p.name for p in candidates)
    raise FileNotFoundError(
        f"{directory}: {len(candidates)} checkpoints ({names}): name the one you mean"
    )


def _enrollment_in(directory: Path) -> Path:
    """The enrollment artefact in ``directory``, or a useful complaint.

    Canonical name, then declared role, then the one rule that makes this a
    separate resolver: a pre-split checkpoint, which declares no role, holds
    the enrollment tensors too and answers for both.
    """
    named = directory / ENROLLMENT_NAME
    if named.is_file():
        _refuse_role(named, ENROLLMENT_ROLE)
        return _paired_with_the_checkpoint_here(named, directory)
    declared = [p for p in _root_checkpoints(directory) if _artifact_role(p) == ENROLLMENT_ROLE]
    if len(declared) == 1:
        return _paired_with_the_checkpoint_here(declared[0], directory)
    if declared:
        names = ", ".join(p.name for p in declared)
        raise FileNotFoundError(
            f"{directory}: {len(declared)} enrollment artefacts ({names}): "
            "name the one you mean"
        )
    return _enrollment_from_presplit(_only_checkpoint_in(directory), directory)


def _paired_with_the_checkpoint_here(enrollment: Path, directory: Path) -> Path:
    """``enrollment``, once it is known to match the checkpoint beside it, when
    there is one: the enrollment half alone is a legitimate thing to enroll
    from."""
    try:
        synthesis = _only_checkpoint_in(directory)
    except FileNotFoundError:
        return enrollment
    _refuse_mismatched_pair(synthesis, enrollment)
    return enrollment


def _enrollment_from_presplit(checkpoint: Path, where: Path | str) -> Path:
    """``checkpoint`` itself, when it is a pre-split file that holds everything.

    Raises:
        FileNotFoundError: when it declares :data:`SYNTHESIS_ROLE`, so the
            enrollment tensors are in the artefact that did not come.
    """
    if _artifact_role(checkpoint) is None:
        return checkpoint
    raise FileNotFoundError(
        f"{where}: no {ENROLLMENT_NAME}, and {checkpoint.name} declares itself "
        "the synthesis artefact, so the enrollment tensors are not in it "
        "either. This is a synthesis-only set: it can speak but not clone. "
        "Fetch the release with cloning (`loudkit download --with-cloning`)."
    )


def _release_tree(repo: str | None) -> Path | None:
    """``repo`` as a directory on disk, or ``None`` when it names a remote repo.

    Anything that exists on disk is a path, always, which is why a path that
    exists and is not a directory is refused here rather than handed to the
    network.

    Raises:
        FileNotFoundError: when ``repo`` exists but is not a release directory.
    """
    if repo is None:
        return None
    path = Path(repo)
    if path.is_dir():
        return path
    if path.exists():
        raise FileNotFoundError(
            f"{repo}: this is a file, not a release directory. A release is one "
            f"{CHECKPOINT_GLOB} beside a {VOICE_DIR}/ directory: pass the "
            "directory that contains them, or a Hugging Face repo id."
        )
    return None


def _enrollment_for_file(checkpoint: Path) -> Path:
    """The enrollment artefact for a checkpoint named by hand: the file itself
    when it declares the role, else its sibling, else the pre-split rule."""
    if _artifact_role(checkpoint) == ENROLLMENT_ROLE:
        return checkpoint
    sibling = checkpoint.parent / ENROLLMENT_NAME
    if sibling.is_file():
        _refuse_role(sibling, ENROLLMENT_ROLE)
        _refuse_mismatched_pair(checkpoint, sibling)
        return sibling
    return _enrollment_from_presplit(checkpoint, checkpoint.parent)


def _voice_names(tree: Path) -> tuple[str, ...]:
    """The voice names under ``tree/voices``, sorted."""
    found = (tree / VOICE_DIR).glob(f"*{VOICE_SUFFIX}")
    return tuple(sorted(p.stem for p in found if p.is_file()))


def release_confinement(directory: Path) -> Path:
    """The directory a file named inside ``directory`` may resolve into.

    Normally ``directory`` itself. A release the Hub cache holds is a snapshot
    of symlinks, one per file, into the sibling ``blobs/`` of the same cache
    entry (``models--<org>--<name>/``), and those blobs are the release's own
    bytes, so there the entry is the boundary. A link that leaves the entry,
    or leaves a plain directory, is still an escape.
    """
    resolved = directory.resolve()
    parts = resolved.parts
    for i in range(len(parts) - 1, 0, -1):
        if parts[i] == "snapshots" and parts[i - 1].startswith(("models--", "datasets--")):
            return Path(*parts[:i])
    return resolved


def _voice_in_tree(tree: Path, ref: str, name: str) -> Path:
    """The voice file ``name`` under ``tree/voices``, or a complaint listing
    what is there. Checked after ``resolve()`` so a symlink out of the release
    is caught too; the Hub cache's own links, into its blob store, are the
    release."""
    voices_dir = (tree / VOICE_DIR).resolve()
    candidate = (voices_dir / name).resolve()
    if not candidate.is_relative_to(release_confinement(tree / VOICE_DIR)):
        raise VoiceNotFoundError(
            f"{ref}: a voice is named, not addressed: {name!r} escapes "
            f"{voices_dir}. Pass a bare name, or a full path as the first "
            "argument with no `repo=`.",
            ref=ref,
            available=_voice_names(tree),
        )
    if candidate.is_file():
        return candidate
    raise VoiceNotFoundError(
        f"{ref}: no such voice in {tree / VOICE_DIR}", ref=ref, available=_voice_names(tree)
    )


def _voice_encoder_beside(checkpoint: Path) -> Path:
    """The utterance voice encoder beside a checkpoint named by hand."""
    sibling = checkpoint.parent / VOICE_ENCODER_NAME
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        f"{sibling} is missing: cloning needs the utterance voice encoder, "
        f"which a release ships as {VOICE_ENCODER_NAME} beside the "
        "checkpoint. Point at a cloning-capable release, or pass "
        "voice_encoder_weights= explicitly."
    )


def _voice_encoder_in(tree: Path, ref: str) -> Path:
    """The utterance voice encoder at a release tree's root."""
    candidate = tree / VOICE_ENCODER_NAME
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"{candidate} is missing: {ref} is a synthesis-only release, with "
        "no utterance voice encoder to clone with."
    )
