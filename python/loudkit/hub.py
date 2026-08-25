"""Resolving a checkpoint by name instead of by path.

``loudkit.load("loudreader/loudr-1")`` should work on a machine that has
never seen the model, the same way every other library in this ecosystem works.
Before this, the documented first step was a manual ``hf download`` into a
directory the caller then had to name correctly — two commands and a path to get
wrong, in front of the one thing a new user came to do, which is hear a voice.

The rule is deliberately dumb, because a resolver that guesses is worse than one
that does not:

* anything that exists on disk is a path, always. A local file wins, so nothing
  starts reaching for the network because a directory happened to be named like
  a repo.
* ``org/name`` with no path separator beyond the one is a Hugging Face repo id.
* anything else is a path that does not exist, and the error says so.

Downloads land in the standard Hugging Face cache, so a second process, a second
virtualenv and a second project share one copy of a 747 MB file.
"""

from __future__ import annotations

import contextlib
import json
import os
import posixpath
import re
from pathlib import Path
from typing import Any

from .errors import VoiceNotFoundError

__all__ = [
    "BACKENDS",
    "CHECKPOINT_GLOB",
    "CHECKPOINT_NAME",
    "ENROLLMENT_NAME",
    "ENROLLMENT_ROLE",
    "SYNTHESIS_ROLE",
    "VOICE_DIR",
    "VOICE_SUFFIX",
    "is_repo_id",
    "list_voices",
    "release_patterns",
    "resolve_checkpoint",
    "resolve_enrollment_checkpoint",
    "resolve_voice",
    "resolve_voice_encoder",
    "verify_release_inventory",
]

BACKENDS = ("torch", "onnx", "coreml")
"""The backends a release can be fetched for.

Named here because :func:`release_patterns` refuses anything outside it. It
used to treat an unknown name as ``torch`` — the one backend whose set is a
strict subset of every other — so ``release_patterns("tourch")`` answered with
a plan that fetches no graphs at all, and the caller who asked for a graph
backend got a snapshot that loads on torch and nothing else, with nothing said.
"""

CHECKPOINT_GLOB = "*.safetensors"
"""What a checkpoint looks like inside a released repo."""

CHECKPOINT_NAME = "loudr-1.safetensors"
"""The synthesis artefact's canonical name in a release.

The packed checkpoint is split in two: this file carries ``t3``, ``flow`` and
``mel2wav`` — everything synthesis reads — and :data:`ENROLLMENT_NAME` carries
the two enrollment modules beside it. Resolution is by *this name* and by the
``artifact_role`` in the manifest, never by globbing and counting, because the
counting rule answered a perfectly ordinary two-artefact release with
"2 checkpoints — name the one you mean".
"""

ENROLLMENT_NAME = "loudr-1-enrollment.safetensors"
"""The enrollment artefact's canonical name: ``s3gen.tokenizer`` and
``s3gen.speaker_encoder``, the ~40% of the old packed file that synthesis never
opens. Fetched only when the caller asked for the cloning capability."""

SYNTHESIS_ROLE = "synthesis"
"""``manifest["artifact_role"]`` of the file :func:`resolve_checkpoint` returns.

A manifest with **no** ``artifact_role`` is a pre-split checkpoint: one file
holding every tensor. That is not an error and must keep loading, and it is also
the only thing that lets the enrollment resolver answer for a release built
before the split, since such a file does carry the enrollment tensors. So the
field is read to *refuse* a file, never to require one: absence is an older
release, and older releases still work.
"""

ENROLLMENT_ROLE = "enrollment"
"""``manifest["artifact_role"]`` of the file
:func:`resolve_enrollment_checkpoint` returns."""

VOICE_DIR = "voices"
"""Where voices live inside a released repo, and inside a local release tree.

Named rather than spelled out at each use because three functions now agree
about it — :func:`resolve_voice` downloads one file from it, :func:`list_voices`
enumerates it, and :func:`resolve_checkpoint`'s non-recursive glob exists
precisely to *avoid* it.
"""

VOICE_SUFFIX = ".safetensors"
"""A voice file's extension. The same container as the checkpoint; only the
directory tells them apart."""

VOICE_ENCODER_NAME = "ve.safetensors"
"""The utterance voice encoder, at a release's root.

Not inside the packed checkpoint: it is a 5.7 MB derived work of the upstream
Chatterbox weights with its own attribution (see NOTICE), so it ships beside
the checkpoint rather than folded into it. `tools/build_release.py` writes it
to the release root, which is where :func:`resolve_voice_encoder` looks.
"""

_ONNX_SYNTHESIS = (
    "onnx/t3_cond.onnx",
    "onnx/t3_prefill.onnx",
    "onnx/t3_step.onnx",
    "onnx/flow_encoder.onnx",
    "onnx/flow_estimator.onnx",
    "onnx/vocoder.onnx",
)
_ONNX_ENROLL = ("onnx/s3_tokenizer.onnx", "onnx/camp.onnx", "onnx/voice_encoder.onnx")
_COREML_SYNTHESIS = (
    "coreml/flow_encoder.mlpackage/*",
    "coreml/flow_estimator.mlpackage/*",
    "coreml/vocoder.mlpackage/*",
)
_COREML_ENROLL = (
    "coreml/s3_tokenizer.mlpackage/*",
    "coreml/camp.mlpackage/*",
    "coreml/voice_encoder.mlpackage/*",
)
_RELEASE_CORE = (
    # `*.safetensors` is the checkpoint and, nested, every voice; ve.safetensors
    # and the enrollment artefact are carved back out below when cloning was not
    # asked for. The three JSON names are spelled out rather than `*.json`, which
    # also matches the small manifests *inside* every CoreML package.
    "*.safetensors",
    "manifest.json",
    "tokenizer.json",
    "release.json",
    "voices/*",
    "SHA256SUMS",
)


def backend_for_device(device: str | None) -> str:
    """Which release set a device needs.

    One definition, because every entry point that fetches needs the same
    answer: ``load``, the three transports, and the CLI seam that resolves
    ``--revision``. A caller that skipped this and let the default stand
    fetched the torch set for ``device="onnx"``, which is a snapshot holding no
    graphs at all -- the backend then failed on a release that had been
    downloaded correctly for a backend nobody asked for.

    ``cuda:1`` and the rest are torch devices, so only the stem is read.
    """
    base = (device or "").split(":", 1)[0]
    return base if base in ("onnx", "coreml") else "torch"


def release_patterns(
    backend: str, *, cloning: bool = False
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(allow_patterns, ignore_patterns)`` for a selective release fetch.

    One repository is the source of truth for every backend and every port, so
    what varies is the fetch, not the layout: every set carries the synthesis
    checkpoint (all three backends read tensors from it), the tokenizer and
    manifest, all twenty voices (3 MB does not earn an axis) and the release
    manifest; ``onnx`` and ``coreml`` add their exported graphs. ``cloning``
    adds what that backend enrols with, and only that: ``torch`` uncovers the
    enrollment artefact and ``ve.safetensors``, which is what
    ``build_torch_enroller`` reads; ``onnx`` and ``coreml`` add their three
    enrollment graphs and keep those two ignored, because the ports carry
    their own enrollers and never open them.

    The two ignored names are what makes the split worth having: a synthesis
    fetch that still pulled :data:`ENROLLMENT_NAME` would move the same bytes
    the packed checkpoint moved, and the caller who never clones would pay for
    the two modules they never open.

    This table lives here rather than in the CLI because it is a capability
    plan, not a flag translation: :func:`resolve_checkpoint` fetches by it,
    :func:`verify_release_inventory` judges the fetch against it, and the CLI
    and the transports delegate to it, so none of them can drift apart.

    Raises:
        ValueError: for a backend outside :data:`BACKENDS`. Not a nicety: the
            torch set is a strict subset of the other two, so falling back to
            it silently answered a typo with a plan that fetches no graphs and
            a fetch that then passes its own inventory check.
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"{backend!r} is not a backend this release ships. Pass one of "
            + ", ".join(BACKENDS)
            + "."
        )
    allow = list(_RELEASE_CORE)
    if backend == "onnx":
        allow += _ONNX_SYNTHESIS
        if cloning:
            allow += _ONNX_ENROLL
    elif backend == "coreml":
        allow += _COREML_SYNTHESIS
        if cloning:
            allow += _COREML_ENROLL
    # The torch enrollment weights are torch's, not everyone's. Python enrols
    # through `build_torch_enroller` and nothing else, so `--for torch` with
    # cloning needs them; the graph ports carry their own enrollers and clone
    # through the enrollment graphs added above. Sending 528 MB of torch
    # weights to an ONNX or CoreML caller is exactly the overfetch the split
    # was for. A Python caller who asked for the graph backend and then calls
    # `enroll()` still works: `resolve_enrollment_checkpoint` fetches the one
    # file on demand rather than dragging it into every snapshot.
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
    whatever subset of the patterns the repo actually holds and answers with
    the same snapshot path either way, so a release missing its ``onnx/``
    graphs used to come back looking exactly like one that has them. A
    shortfall here is an error naming what is absent, because the caller asked
    for a *usable* set for one backend, and the alternative was a printed path
    to a directory that does not exist.

    ``require_voices`` is the caller's claim that a voiceless fetch is a
    failure — true for ``loudkit download`` (the next printed step is
    ``--voice <name>``) and for official releases, which always ship voices;
    a stranger's bare checkpoint upload keeps the lenient path.

    **The basics are checked before the backend's extras**, because this
    function used to judge only what varies. It asked after the exported
    graphs, the voice encoder and the voices, and said nothing about the
    checkpoint, the tokenizer or the manifest — so a fetch that came back
    without the weights, or without the tokenizer that decides what the
    weights are asked to say, passed here and printed as a usable set. Four
    things are now required of every backend:

    * a **synthesis checkpoint** that :func:`resolve_checkpoint` will find,
      which after the split means :data:`CHECKPOINT_NAME` or one file that
      resolves as the synthesis artefact;
    * ``manifest.json``, the human-readable mirror of the manifest the
      checkpoint embeds;
    * ``tokenizer.json``, unless the checkpoint carries it as a packed asset —
      the resolution order :meth:`Checkpoint.resolve_asset` documents, so a
      self-contained checkpoint is not asked for a sibling it does not need;
    * for ``coreml``, that each ``.mlpackage`` has **the structure of one** —
      a ``Manifest.json`` beside a non-empty ``Data/`` — rather than merely
      being a directory with something in it. A pattern fetch that brought
      back one stray file made an empty-shaped package look complete.

    With ``cloning``, the enrollment tensors must be reachable too: the
    enrollment artefact, or a pre-split checkpoint that still holds them.

    Raises:
        FileNotFoundError: naming every absent piece, and the backend whose
            set is short.
        ValueError: for a backend outside :data:`BACKENDS`.
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
        missing.append(CHECKPOINT_NAME)
    if checkpoint is None or not _packs_tokenizer(checkpoint):
        expected.append("tokenizer.json")
    if backend == "onnx":
        expected += _ONNX_SYNTHESIS
        if cloning:
            expected += _ONNX_ENROLL
    elif backend == "coreml":
        # The pattern names the package's files; what must exist is a package,
        # and an .mlpackage is a directory tree with a fixed shape.
        packages = _COREML_SYNTHESIS + (_COREML_ENROLL if cloning else ())
        for pattern in packages:
            rel = pattern.removesuffix("/*")
            if not _is_mlpackage(root / rel):
                missing.append(rel)
    # The same condition release_patterns fetches under, and it has to be:
    # requiring what the plan did not fetch turns a correct download into an
    # error naming files the caller was right not to have. Torch enrols from
    # these two; the graph ports enrol from the enrollment graphs checked
    # above and never open them.
    if cloning and backend == "torch":
        expected.append(VOICE_ENCODER_NAME)
        try:
            resolve_enrollment_checkpoint(str(root))
        except FileNotFoundError:
            missing.append(ENROLLMENT_NAME)
    missing += [rel for rel in expected if not (root / rel).is_file()]
    if require_voices and not any(
        p.is_file() for p in (root / VOICE_DIR).glob(f"*{VOICE_SUFFIX}")
    ):
        missing.append(f"{VOICE_DIR}/*{VOICE_SUFFIX}")
    if missing:
        raise FileNotFoundError(
            f"{root}: this fetch does not add up to a usable {backend} set — "
            "missing: " + ", ".join(sorted(missing)) + ". The release does not "
            "carry these files, or the fetch was interrupted; retry, or pin a "
            "revision that ships them."
        )


def _is_mlpackage(node: Path) -> bool:
    """Whether ``node`` has the structure of a CoreML package.

    An ``.mlpackage`` is a directory tree, not a file: a ``Manifest.json`` at
    its root beside a ``Data/`` holding the model and its weights. The check
    used to be "a directory with anything in it", which passes for a package
    whose weights never arrived — the one shortfall a pattern fetch actually
    produces, since ``coreml/vocoder.mlpackage/*`` brings back whatever subset
    the repo holds.
    """
    if not node.is_dir():
        return False
    data = node / "Data"
    return (node / "Manifest.json").is_file() and data.is_dir() and any(data.iterdir())


def _packs_tokenizer(checkpoint: Path) -> bool:
    """Whether ``checkpoint`` carries ``tokenizer.json`` as a packed asset.

    ``False`` for anything this process cannot read as a checkpoint, so an
    unreadable file is asked for the sibling rather than excused from it: the
    lenient answer here would be the one that lets a set with no tokenizer at
    all pass as usable.
    """
    try:
        from safetensors import safe_open

        from .checkpoint import ASSET_PREFIX

        with safe_open(str(checkpoint), framework="numpy") as f:
            return f"{ASSET_PREFIX}tokenizer.json" in f.keys()  # noqa: SIM118
    except Exception:  # noqa: BLE001 - not a readable checkpoint; require the sibling
        return False


def _artifact_role(path: Path) -> str | None:
    """``manifest["artifact_role"]`` for ``path``, or ``None``.

    ``None`` means *no claim*, and covers both halves of one rule: a pre-split
    checkpoint, whose manifest predates the field and which does carry every
    tensor, and a file this process cannot read as a checkpoint at all. Both
    are treated the same way — as the artefact loudkit has always accepted —
    because the field is only ever used to *refuse* (:func:`_refuse_role`) or
    to pick between two files that both declare one. A missing claim never
    promotes a file to a role it did not ask for.
    """
    role = (_manifest_of(path) or {}).get("artifact_role")
    return role if isinstance(role, str) else None


def _manifest_of(path: Path) -> dict[str, Any] | None:
    """``path``'s embedded manifest, or ``None`` if it has none this can read."""
    try:
        from .checkpoint import read_manifest

        return read_manifest(path)
    except Exception:  # noqa: BLE001 - unreadable is "no claim", never a role
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
    """``manifest["split"]["source_payload_sha256"]``, or ``None``.

    The splitting tool stamps the digest of the *packed original* into both
    halves, so two files can be proved to be halves of one packing run from
    their two headers alone — no third artefact, and no comparing filenames,
    which is the check that cannot tell a matching pair from two files that
    merely agree about what they are called.
    """
    split = (_manifest_of(path) or {}).get("split")
    if not isinstance(split, dict):
        return None
    source = split.get("source_payload_sha256")
    return source if isinstance(source, str) else None


def _refuse_mismatched_pair(synthesis: Path, enrollment: Path) -> None:
    """Refuse two halves that came from different packing runs.

    Nothing else would report this. Enrollment reads the speaker encoder and
    the speech tokenizer from the enrollment half; synthesis reads everything
    else from the other. Mismatched halves both load, both produce audio, and
    the voice is wrong — a failure with no error and no obvious symptom, which
    is exactly the kind the ``split`` block exists to catch.

    Silent when either file makes no claim: a pre-split checkpoint answers for
    both halves and has no pair to disagree with, and a third-party build may
    carry no ``split`` block at all.

    Raises:
        ValueError: a bad pair, not a missing file — which is also why it is
            not a ``FileNotFoundError``. Both halves are right here; what is
            wrong is that they do not belong together, and
            :func:`verify_release_inventory` must report that rather than fold
            it into "the enrollment artefact did not come".
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


_REPO_ID = re.compile(r"^[\w\-.]+/[\w\-.]+$")
"""``org/name``. Dots are legal in both halves, which is why the path-shaped
rejections below are needed: ``./model.safetensors`` matches this pattern
perfectly well as the repo ``.`` / ``model.safetensors``."""

_HUB_NOT_FOUND = frozenset(
    {
        "EntryNotFoundError",
        # A *subclass* of the one above, five deep past HfHubHTTPError,
        # HTTPError and OSError. Listing it is belt and braces; what actually
        # broke was comparing the exact class name, which cannot see
        # inheritance, so a real 404 from a real client escaped as a raw error
        # and the pre-split fallback failed on the published repository.
        "RemoteEntryNotFoundError",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
    }
)
"""What the hub client raises for "it is not there".

Matched by name rather than imported: the client is an optional extra, so
importing its exception types at module scope would make this module require
it. Anything outside this set — a timeout, a 500, a proxy refusing — is not a
"no such voice" and keeps its own traceback.
"""


def _is_not_found(exc: BaseException) -> bool:
    """Whether the hub said "it is not there".

    Walks the exception's own class chain rather than testing one name. That
    is the whole fix: the client's own hierarchy already says a remote 404 is
    an entry-not-found, and a comparison against one class name cannot see it.
    A subclass the client adds later is recognised without an edit here.

    The class names are the contract, not the types: see
    :data:`_HUB_NOT_FOUND` for why they are not imported.
    """
    return any(cls.__name__ in _HUB_NOT_FOUND for cls in type(exc).__mro__)


_MISSING_HUB = (
    "resolving {ref!r} needs the Hugging Face client.\n"
    '  pip install "loudkit[hub]"\n'
    "Or download the files yourself and pass a path."
)


def is_repo_id(ref: str) -> bool:
    """True for ``org/name``, false for anything that exists on disk.

    Checked in that order on purpose: a path that exists is never a repo id,
    however it is spelled.
    """
    if Path(ref).exists():
        return False
    # Path-shaped strings are paths even when they have not been created yet.
    # A user who typed a filename and misspelled it wants "no such file", not a
    # network round trip against a repo that will never exist.
    if ref.startswith((".", "/", "~")) or ref.endswith(".safetensors"):
        return False
    return bool(_REPO_ID.match(ref))


def _hub() -> Any:
    """The Hugging Face client, or a message naming the extra that carries it.

    `Any` because the client ships no type information this package can pin
    against; the three calls made through it are immediately below.
    """
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - depends on extras
        # `name=` because the CLI's `_explain_missing` reads `exc.name` to say
        # which package is absent; a `ModuleNotFoundError` raised by hand has
        # `name is None`, and the advice degrades to "loudkit needs the ''
        # package".
        raise ModuleNotFoundError(
            _MISSING_HUB.format(ref="a repo id"), name="huggingface_hub"
        ) from exc
    return huggingface_hub


def resolve_checkpoint(
    ref: str, *, revision: str | None = None, backend: str = "torch"
) -> Path:
    """A local path for the **synthesis** artefact of ``ref``.

    Always the synthesis artefact, whichever door the caller came through: a
    release is two files now (:data:`CHECKPOINT_NAME` and
    :data:`ENROLLMENT_NAME`), and this is what ``load()`` needs. The other half
    has its own resolver, :func:`resolve_enrollment_checkpoint`, and is only
    ever fetched by a caller who asked to clone.

    Args:
        ref: a path to a ``.safetensors``, a directory holding one, or a
            Hugging Face repo id such as ``loudreader/loudr-1``.
        backend: which backend's file set to fetch when ``ref`` is a repo id —
            ``torch``, ``onnx`` or ``coreml``, per :func:`release_patterns`.
            The checkpoint alone is not the release: an ONNX caller needs the
            exported graphs beside it, and a resolver that does not know the
            backend hands back a path that loads on torch and nothing else.
            Ignored for a local path, whose contents are already decided.
        revision: branch, tag or commit for the repo case. Pin it for anything
            reproducible — a moving ``main`` is a moving model, and unpinned is
            the default here only because it is what every hub client does.

            Worth being exact about what pinning buys, because the checkpoint
            looks like it already solves this and does not. A packed checkpoint
            carries ``tensor_payload_sha256``, a digest **of itself**: it
            detects a truncated or corrupted download and authenticates
            nothing, since anything that can replace the file can replace the
            digest beside it. A commit sha, or a hash checked against the
            release's ``SHA256SUMS`` obtained separately, is the part that
            says *this* is the artifact you meant.

    Raises:
        FileNotFoundError: when the path does not exist, or when the repo holds
            no checkpoint. The message names what was looked for.
    """
    path = Path(ref)
    if path.is_file():
        # A file the caller named by hand is still checked against its own
        # claim: `load("loudr-1-enrollment.safetensors")` would otherwise get
        # as far as the engine builder and fail there about missing `t3.`
        # tensors, which is a true statement about the wrong question.
        _refuse_role(path, SYNTHESIS_ROLE)
        return path
    if path.is_dir():
        return _only_checkpoint_in(path)
    if not is_repo_id(ref):
        raise FileNotFoundError(
            f"{ref}: no such file or directory, and not a Hugging Face repo id "
            "(those look like 'org/name')"
        )

    hub = _hub()
    # The backend's whole set, not one file: the tokenizer and the voices
    # travel beside the weights, and a caller who has the checkpoint but not
    # the tokenizer has a download that looks finished and an engine that will
    # not build. `release_patterns` keeps the *other* backends' exported
    # graphs — hundreds of megabytes this backend never opens — out of it,
    # and fetches this backend's own.
    allow, ignore = release_patterns(backend)
    try:
        local: str = hub.snapshot_download(
            repo_id=ref,
            revision=revision,
            allow_patterns=list(allow),
            ignore_patterns=list(ignore) or None,
        )
    except Exception as exc:  # noqa: BLE001 — mapped by name below
        friendly = _friendly_hub_error(exc, ref, revision)
        if friendly is None:
            raise
        raise friendly from exc
    root = Path(local)
    _verify_sha256sums(root, repo=ref)
    # Checked after the hashes: a set can be intact and still short, because
    # `allow_patterns` fetches whatever subset the repo holds and reports
    # nothing about the rest.
    verify_release_inventory(root, backend, require_voices=_is_official(ref))
    return _only_checkpoint_in(root)


def resolve_enrollment_checkpoint(ref: str, *, revision: str | None = None) -> Path:
    """A local path for the **enrollment** artefact of ``ref``.

    The other half of :func:`resolve_checkpoint`, and separate from it because
    the two are fetched by different callers at different times: synthesis is
    every ``load()``, enrollment is only :func:`loudkit.enroll`. A repo id here
    is one file (:data:`ENROLLMENT_NAME`) rather than a snapshot, the same way
    a voice and the utterance encoder are, so nobody who never clones moves
    those bytes.

    ``ref`` is whatever :func:`loudkit.enroll` was handed — a repo id, a
    directory holding an unpacked release, or a checkpoint file. For a file,
    the enrollment artefact is its sibling; a pre-split file, which declares no
    ``artifact_role``, answers for itself.

    Raises:
        FileNotFoundError: when the set is synthesis-only — the enrollment
            artefact is absent and the checkpoint beside it declares that it is
            not carrying those tensors either. Said in those words, because the
            alternative is the enroller's own complaint about
            ``s3gen.speaker_encoder``, a name no caller of the public API has
            heard of.
    """
    path = Path(ref)
    if path.is_file():
        if _artifact_role(path) == ENROLLMENT_ROLE:
            return path
        sibling = path.parent / ENROLLMENT_NAME
        if sibling.is_file():
            _refuse_role(sibling, ENROLLMENT_ROLE)
            _refuse_mismatched_pair(path, sibling)
            return sibling
        return _enrollment_from_presplit(path, path.parent)
    if path.is_dir():
        return _enrollment_in(path)
    if not is_repo_id(ref):
        raise FileNotFoundError(
            f"{ref}: no such file or directory, and not a Hugging Face repo id "
            "(those look like 'org/name')"
        )

    hub = _hub()
    try:
        downloaded: str = hub.hf_hub_download(
            repo_id=ref, filename=ENROLLMENT_NAME, revision=revision
        )
    except Exception as exc:  # noqa: BLE001 — the client raises its own hierarchy
        friendly = _friendly_hub_error(exc, ref, revision)
        if friendly is not None:
            raise friendly from exc
        if not _is_not_found(exc):
            raise
        # The repo answered and holds no enrollment artefact. That is a
        # pre-split release — which is what the published repo is until the
        # new bundle is uploaded — so the synthesis checkpoint still carries
        # the enrollment tensors, and `_enrollment_from_presplit` is what
        # decides whether it does or whether this set simply cannot clone.
        return _enrollment_from_presplit(resolve_checkpoint(ref, revision=revision), ref)
    resolved = Path(downloaded)
    _verify_against_release_sums(hub, ref, revision, ENROLLMENT_NAME, resolved)
    return resolved


def _friendly_hub_error(
    exc: Exception, ref: str, revision: str | None
) -> FileNotFoundError | None:
    """A one-line diagnosis for the hub's own "it is not there" errors.

    The raw ``RepositoryNotFoundError`` ends in "Invalid username or password",
    which diagnoses an auth problem the user does not have. This is the error
    a stranger's very first command produces when the release is not published
    yet or they are offline — it deserves its own sentence.
    """
    if type(exc).__name__ == "RepositoryNotFoundError":
        return FileNotFoundError(
            f"{ref}: repository not found or not public (revision "
            f"{revision or 'default'}). Is the release published, and are you "
            "online? Run `loudkit doctor` to check this machine."
        )
    if type(exc).__name__ == "RevisionNotFoundError":
        return FileNotFoundError(f"{ref}: revision {revision!r} not found in the repository.")
    # Offline with nothing cached. The client's own message is a paragraph
    # about connection errors and `HF_HUB_OFFLINE`; what the caller can act on
    # is that this machine has never fetched this release.
    if type(exc).__name__ == "LocalEntryNotFoundError":
        return FileNotFoundError(
            f"{ref}: not in the local cache and the hub cannot be reached. "
            "Connect once to fetch it, or pass a path to a release directory."
        )
    return None  # not one of ours; caller re-raises


_SUMS_LINE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")

_DRIVE_LETTER = re.compile(r"^[A-Za-z]:/")
"""``C:/`` at the very start: a Windows absolute path wearing POSIX separators.

The backslash rejection above catches ``C:\\weights\\evil.safetensors`` by its
separators, but Windows accepts forward slashes just as well and ``C:/weights/
evil.safetensors`` is a name relative to nothing. It is refused for the same
reason a leading ``/`` is.

Deliberately narrow: one ASCII letter, a colon, a slash, anchored. A colon is
a legal character in a POSIX filename, so ``voices/a:b.safetensors`` and even
``c:b.safetensors`` stay legal names for real files a release could contain.
Only the drive-plus-separator prefix is a claim about a filesystem root.
"""
_VERIFIED_MARKER = ".loudkit-verified"

_OFFICIAL_ORG = "loudreader"
"""The org whose releases this library vouches for.

Everything under it is built by ``tools/build_release.py``, which always writes
``SHA256SUMS``; a snapshot of one of these repos arriving without a manifest is
therefore not "an old-fashioned upload", it is a release that lost the one file
that says what it should contain — stripped in transit, served by a mirror that
does not carry it, or fetched from a repo that only looks like ours. That is
the case worth failing on, and it is the only case this module can tell apart
from a stranger's bare checkpoint upload.
"""

_UNMANIFESTED = frozenset({"SHA256SUMS", _VERIFIED_MARKER})
"""Files a correct release holds that its own manifest cannot list.

Two, and the reason is the same both times: nothing can hash a file that does
not exist yet. ``SHA256SUMS`` cannot contain its own digest, and
``.loudkit-verified`` is written by this module after the release was built.

``release.json`` is deliberately not here. It carries the profile a bundle was
built from and whether the checkpoint was verified, which is exactly the claim
an unchecksummed file cannot make, so ``tools/build_release.py`` writes it
first and covers it. Exempting it would suppress the report on a bundle whose
``release.json`` is *not* covered, which is the bundle worth hearing about.

Everything else in a snapshot is either listed or reported by
:func:`_verify_sha256sums`.
"""

_HUB_BOOKKEEPING = frozenset({".gitattributes"})
"""Hub furniture at a snapshot's root, named rather than pattern-matched.

The hub client materialises the repo's ``.gitattributes`` (the LFS rules) into
every snapshot, and ``tools/build_release.py`` does not hash it, so it has to
be exempt or every release warns about a file nobody loads.

Exempt **by name**, because the exemption that read "anything starting with a
dot" also covered ``.hidden.safetensors``. That name is matched by the
checkpoint loader's glob like any other, which made the one class of file
whose bytes matter most the one class that could skip the manifest entirely.
"""

_HUB_CACHE_DIR = ".cache"
"""The client's download bookkeeping, when it writes any.

``snapshot_download(local_dir=...)`` puts etags and partial-download metadata
in ``.cache/huggingface/`` at the root of the tree it fills. It is the client's
own scratch, regenerated on demand, and it is never bytes loudkit opens; it is
pruned at the root only, so a ``.cache/`` deeper in a release would still be
inventoried and reported.
"""


def _is_official(repo: str | None) -> bool:
    """Whether ``repo`` names a release this project publishes."""
    return repo is not None and repo.split("/", 1)[0].lower() == _OFFICIAL_ORG


def _missing_manifest(where: str) -> ValueError:
    return ValueError(
        f"{where}: no SHA256SUMS. Every {_OFFICIAL_ORG} release ships one, so a "
        "download without it cannot be checked against anything and will not be "
        "used. Retry the download, and pass a revision you trust."
    )


_STRICT_PROFILE = "full-0.1"
"""The one profile ``tools/build_release.py`` will stamp on a releasable
bundle. The builder refuses to write it without running its closing gate, so
the pair ``profile == "full-0.1", verified == true`` is the machine-readable
claim that a bundle is the release and that it loaded and spoke."""


def _require_releasable(root: Path, where: str) -> None:
    """An official snapshot must be a strict, gate-verified release.

    Checksums say the bytes arrived intact; they say nothing about what the
    bytes are. A ``lenient`` development bundle carries a perfectly valid
    ``SHA256SUMS``, so without this check an unreleasable bundle pushed to an
    official repo verifies and loads like the release. The builder records
    the difference in ``release.json`` for exactly this reader: the profile
    it was built from, and whether the load-and-speak gate ran and passed.
    Third-party repos and local directories are not held to it: there is no
    claim there to check.
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
    """``release.json`` must say ``profile: full-0.1`` and ``verified: true``.

    The body of :func:`_require_releasable`, shared with the single-file path
    (:func:`_verify_against_release_sums`), which holds the same official
    repos to the same claim but fetches the record on its own rather than
    finding it in a snapshot.
    """
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(
            f"{where}: release.json is unreadable ({exc}). Delete the cached "
            "snapshot and retry, or pin a revision you trust."
        ) from exc
    profile = record.get("profile") if isinstance(record, dict) else None
    if profile != _STRICT_PROFILE:
        raise ValueError(
            f"{where}: release.json says profile {profile!r}, and an "
            f"{_OFFICIAL_ORG} release is {_STRICT_PROFILE!r}. This is a "
            "development bundle, not the release. Fetch a published revision, "
            "or pass a local path if the bundle is your own."
        )
    if not isinstance(record, dict) or record.get("verified") is not True:
        raise ValueError(
            f"{where}: release.json does not record verified: true, so the "
            "bundle never passed the builder's load-and-speak gate. Rebuild it "
            "with tools/build_release.py, or fetch a published revision."
        )


def _stat_record(path: Path) -> dict[str, Any]:
    """One inventory entry: what ``path`` is, and the stat of what it opens.

    ``size``/``mtime_ns`` follow the link when there is one, because those are
    the bytes a reader opens — a blob rewritten in place moves them even
    though the link itself never changed. ``kind`` and, for a link, ``target``
    and the link's own ``lstat`` mtime are recorded besides, so that swapping
    a file for a link, or repointing a link at different bytes of the same
    size and mtime, is a change the marker sees rather than one it seals in.
    """
    info = path.stat()
    record: dict[str, Any] = {
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "kind": "file",
    }
    if path.is_symlink():
        record["kind"] = "link"
        record["target"] = os.readlink(path)
        record["lmtime_ns"] = os.lstat(path).st_mtime_ns
    return record


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    """What the snapshot holds right now: every in-scope file, with its stat.

    In scope is every regular file under ``root``, named by its POSIX path
    relative to ``root``, except two exemptions, both by exact name at the
    root:

    * the names in :data:`_UNMANIFESTED`, which a correct release holds and
      its own manifest cannot cover;
    * :data:`_HUB_BOOKKEEPING` and the :data:`_HUB_CACHE_DIR` tree, which are
      the hub client's furniture rather than the release's contents.

    Named exactly, and never by shape. A dot prefix is not a licence: a file
    called ``.hidden.safetensors`` is matched by the checkpoint loader's glob
    like any other, so exempting it from the inventory would exempt weights
    from the manifest. Everything the two lists do not name is in scope, dot
    or no dot, and is either listed by the manifest or reported.

    The cache directory is pruned rather than walked, so the cost is the
    snapshot's own tree and not the client's metadata beside it.
    """
    out: dict[str, dict[str, Any]] = {}
    for parent, dirs, files in os.walk(root):
        if Path(parent) == root:
            dirs[:] = [d for d in dirs if d != _HUB_CACHE_DIR]
        for name in files:
            path = Path(parent, name)
            if not path.is_file():
                continue  # a broken symlink is not bytes anything can open
            rel = path.relative_to(root).as_posix()
            if rel in _UNMANIFESTED or rel in _HUB_BOOKKEEPING:
                continue
            out[rel] = _stat_record(path)
    return out


def _unverified_files(
    inventory: dict[str, dict[str, Any]], entries: dict[str, str]
) -> list[str]:
    """Files present in the snapshot that the manifest says nothing about."""
    return sorted(name for name in inventory if name not in entries)


def _rejected_name(name: str) -> str | None:
    """Why ``name`` is not addressable inside a snapshot, or ``None``.

    A manifest name is joined onto the snapshot root and then read, so it has
    to be a normalised relative POSIX path and nothing else. Nested names —
    ``onnx/t3_step.onnx``, ``voices/joe.safetensors`` — are the ordinary case
    and stay legal; what is refused is a name that can address bytes the
    snapshot does not contain, and a name a reader cannot compare against what
    ``shasum -c`` would do with the same file.
    """
    if "\\" in name:
        return "is not a POSIX path (it contains a backslash)"
    # Both spellings of "rooted at a filesystem, not at this snapshot": a
    # leading `/`, and a Windows drive letter (see `_DRIVE_LETTER`).
    if name.startswith("/") or _DRIVE_LETTER.match(name):
        return "is absolute, and a manifest name is relative to the snapshot root"
    parts = name.split("/")
    if ".." in parts:
        return "escapes the snapshot root with '..'"
    if "" in parts or "." in parts:
        return "is not normalised (an empty or '.' path component)"
    # Belt and braces: whatever the component checks let through must also be
    # a fixed point of normalisation, so that the name this parser returns is
    # the name the filesystem will resolve.
    if posixpath.normpath(name) != name:
        return f"is not normalised (it names {posixpath.normpath(name)!r})"
    return None


def _hub_blobs_dir(root: Path) -> Path | None:
    """The ``blobs/`` directory of the cache entry ``root`` belongs to.

    The standard hub cache lays a repo out as ``models--org--name/`` holding
    ``blobs/`` (content-addressed bytes) beside ``snapshots/<revision>/``
    (names, each a symlink into ``blobs/``). ``None`` for anything else — an
    unpacked directory, a ``local_dir`` download — where a symlink has no
    cache to point into and stays refused. The ``blobs`` directory itself must
    be a real directory: a symlink there would relocate every "confined"
    target in one move.
    """
    repo_dir = root.parent.parent
    if (
        root.parent.name == "snapshots"
        and repo_dir.name.startswith("models--")
        and not (root.is_symlink() or root.parent.is_symlink())
    ):
        blobs = repo_dir / "blobs"
        if blobs.is_dir() and not blobs.is_symlink():
            return blobs
    return None


def _blob_symlink_ok(root: Path, node: Path) -> bool:
    """Whether a final-file symlink is the hub cache addressing its own blob.

    True only when ``root`` is a hub-cache snapshot and ``node`` resolves —
    through however many hops — to a regular file *directly inside* the same
    repo's ``blobs/`` directory. That is the one symlink the cache is made of;
    anything else a link reaches is bytes the snapshot does not contain.
    """
    blobs = _hub_blobs_dir(root)
    if blobs is None:
        return False
    try:
        resolved = node.resolve(strict=True)
    except OSError:
        return False
    return resolved.parent == blobs.resolve() and resolved.is_file()


def _refuse_symlinked(root: Path, name: str) -> None:
    """Refuse a manifest name that reaches outside the snapshot via a symlink.

    :func:`_rejected_name` refuses traversal spelled in the name; a directory
    symlink inside the snapshot performs the same escape with a perfectly
    legal name: ``voices/joe.safetensors`` addresses whatever ``voices``
    points at, and a verifier that follows it hashes, vouches for and hands
    back bytes the snapshot does not contain. So every component of the
    joined path is asked what it *is*, with ``lstat`` semantics
    (``Path.is_symlink`` never follows), from the first component down to the
    file itself. An absent component ends the walk: a name that resolves to
    nothing addresses nothing.

    One symlink is legitimate, because the hub cache is built out of it: the
    **final** component, when it resolves into the ``blobs/`` directory of the
    same cache entry this snapshot belongs to (:func:`_blob_symlink_ok`). A
    directory component is never that — the cache links files, not
    directories — so a symlinked directory stays refused everywhere.
    """
    node = root
    parts = name.split("/")
    for index, part in enumerate(parts):
        node = node / part
        if node.is_symlink():
            if index == len(parts) - 1 and _blob_symlink_ok(root, node):
                return
            raise ValueError(
                f"{root}: {name} reaches through a symlink "
                f"({node.relative_to(root).as_posix()}), which can address "
                "bytes outside the snapshot; refusing to verify it. Delete "
                "the cached snapshot and retry."
            )
        if not node.exists():
            return


def _parse_sha256sums(sums: Path) -> dict[str, str]:
    """Parse a ``SHA256SUMS`` file, refusing lines it cannot understand.

    A mangled manifest must fail loudly rather than verify nothing: every
    non-empty line has to be a 64-hex digest, two spaces, and a normalised
    relative POSIX path — the exact format ``tools/build_release.py`` writes.
    Entries whose files are absent from disk are returned all the same; the
    caller decides whether that is expected (a partial fetch) or fatal.

    Three shapes are refused beyond "does not match the format", because each
    of them makes the manifest describe something other than this snapshot:

    * ``..`` traversal and absolute paths, which address bytes outside the
      root the manifest is a manifest of;
    * un-normalised names, so that the name checked here is the name the
      filesystem resolves;
    * a name listed twice, because then the manifest disagrees with itself and
      whichever entry the parser happens to keep decides whether verification
      passes. A release with a duplicate can pass here and fail ``shasum -c``.

    Every refusal names the manifest line, since the reader is usually the
    person who built the release.
    """
    out: dict[str, str] = {}
    for number, raw_line in enumerate(sums.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.rstrip("\r")
        if not line.strip():
            continue
        match = _SUMS_LINE.match(line)
        if match is None:
            raise ValueError(
                f"{sums}: malformed SHA256SUMS line {number}: {line!r} — the "
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
                f"{sums}: line {number}: duplicate entry for {name!r} — the "
                "manifest disagrees with itself about one file, so what it "
                "verifies depends on which entry is read. Rebuild the release."
            )
        out[name] = match.group(1)
    assert out, f"{sums}: no checksum entries"
    return out


def _verify_sha256sums(root: Path, *, repo: str | None = None) -> None:
    """Check every downloaded file against the release's own ``SHA256SUMS``.

    A release built by ``tools/build_release.py`` ships the manifest, and the
    freshly downloaded bytes are hashed against it — in 1 MB blocks, because
    the checkpoint is 747 MB and a verification that reads it whole into
    memory costs more than the load it guards — before anything is returned.
    A truncated or substituted file fails ``load`` instead of loading.

    Three rules decide what "verified" means here, because a checksum file
    that is merely *consulted* proves nothing:

    * **A missing manifest is fatal for an official release.** Any repo under
      ``loudreader/`` ships one; arriving without it means the release is not
      the one it claims to be. Anything else — a stranger's repo, a bare
      checkpoint upload, an unpacked directory verified by hand — keeps the
      lenient path and skips the check, since there is nothing to check it
      against and no expectation to violate.
    * **A listed file that was not fetched is skipped**, since ``load()`` asks
      for a subset of the release (``allow_patterns``) and the entries for the
      exported graphs describe files it never downloads.
    * **A fetched file the manifest does not list is never trusted.**
      ``*.safetensors`` is refused outright under any repo: those are the
      bytes loudkit opens, and an unlisted one is weights nothing vouches
      for. For an official release *everything* uncovered is refused, because
      ``tools/build_release.py`` checksums every file it ships, so an uncovered
      file inside a loudreader snapshot did not come from the release, and a
      file another tool fetched from the same repo and revision is by the
      same rule in the manifest. A third-party snapshot gets a warning naming
      the file instead, since no builder promised coverage there. The files a
      correct snapshot holds that its manifest cannot cover (``SHA256SUMS``
      itself, the hub client's furniture) are exempt by exact name, not by
      shape.

    An official snapshot is additionally required to *be* a release:
    ``release.json`` must say ``profile: full-0.1`` and ``verified: true``
    (:func:`_require_releasable`), because a lenient development bundle
    carries a perfectly valid manifest and would otherwise verify and load
    like the real thing.

    The manifest authenticates the download, not the publisher: anything that
    can replace a file can replace the manifest beside it. Pin ``revision=``
    for that.

    A passing run leaves ``.loudkit-verified`` beside the manifest, recording
    the manifest's digest and, for every in-scope file in the snapshot
    (:func:`_inventory` defines in-scope), its size and mtime, what the
    directory entry *is*, and a symlink's target (:func:`_stat_record`), so a
    cached snapshot is hashed once and not on every ``load()``.

    **Trust-on-first-use, stated plainly:** on later calls the file bytes are
    not read again. The marker's record catches the cheap failures a sealed
    cache otherwise hides — a file truncated, replaced or rewritten in place
    after verification — because any of those moves the size or the mtime,
    and either one sends the snapshot back through a full re-hash. A file
    swapped for a symlink, or a link repointed, moves the recorded kind or
    target the same way — and manifest names are re-confined against
    :func:`_refuse_symlinked` before the marker is even consulted. The record
    is an inventory rather than a list of what happened to be there once: a
    file that *appears* after the marker was written invalidates it just as a
    moved size does, because the hub cache is shared per repo and revision and
    loudkit fetches in stages, so a checkpoint fetch can seal the marker and a
    later ``onnx/`` fetch add files it never saw. What the marker cannot catch
    is decay that preserves the whole inventory and every stat: silent bit rot
    in the storage layer, or an edit careful enough to restore the timestamp.
    Delete the marker to force a full re-verification.
    """
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        if _is_official(repo):
            raise _missing_manifest(f"{repo} ({root})")
        return

    from .checkpoint import file_sha256

    entries = _parse_sha256sums(sums)
    # Checked on every load, marker or no marker: the claim is one small JSON
    # read, and a snapshot that stops being a strict release must stop loading
    # the moment it does, not the next time something re-hashes.
    if _is_official(repo):
        _require_releasable(root, f"{repo} ({root})")
    # Confinement first, before the marker can answer: a name that reaches
    # through a symlink is refused on *every* load, because a link swapped in
    # after verification can be aimed at outside bytes whose size and mtime
    # match the record — the one substitution the marker's stat comparison was
    # designed not to have to catch.
    for name in entries:
        _refuse_symlinked(root, name)
    marker = root / _VERIFIED_MARKER
    sums_digest = file_sha256(sums)
    # Taken before the marker is consulted, because the marker's claim is now
    # about this inventory: a walk and a stat per file, against the ~4 s it
    # takes to re-hash a snapshot: 747 MB for synthesis, 1.27 GB once cloning
    # has pulled the enrollment half too.
    inventory = _inventory(root)
    if _marker_still_holds(marker, sums_digest, inventory):
        return

    bad: list[str] = []
    hashed: list[str] = []
    for name, digest in sorted(entries.items()):
        target = root / name
        if not target.is_file():
            continue  # not part of the patterns fetched; nothing to check
        if file_sha256(target) != digest:
            bad.append(name)
        else:
            hashed.append(name)
            # Re-stat after the read rather than reusing the walk's record, so
            # a file rewritten while it was being hashed is recorded as it is
            # now and re-hashed next time instead of being sealed in.
            inventory[name] = _stat_record(target)
    if bad:
        raise ValueError(
            f"{root}: downloaded file(s) failed the release checksum: "
            + ", ".join(sorted(bad))
            + " — delete the cached snapshot and retry, or pin a revision you trust."
        )
    if not hashed:
        raise ValueError(
            f"{root}: SHA256SUMS lists {len(entries)} file(s) but none of them "
            "was fetched — the manifest and the download disagree."
        )

    _judge_unlisted(root, repo, _unverified_files(inventory, entries))

    # A read-only cache still gets the verification; only the marker is lost.
    with contextlib.suppress(OSError):
        marker.write_text(
            json.dumps({"sha256sums": sums_digest, "files": inventory}) + "\n",
            encoding="utf-8",
        )


def _judge_unlisted(root: Path, repo: str | None, unlisted: list[str]) -> None:
    """The verdict on fetched files the manifest says nothing about.

    Weights are refused under any repo: those are the bytes loudkit opens.
    For an official release everything uncovered is refused, because the
    builder checksums every file a release ships, so an uncovered file inside
    a loudreader snapshot did not come from the release, and a file another
    tool fetched from the same repo and revision is by the same rule in the
    manifest. A third-party snapshot gets a warning naming the file, since
    no builder promised coverage there.
    """
    weights = [name for name in unlisted if name.endswith(VOICE_SUFFIX)]
    if weights:
        raise ValueError(
            f"{root}: SHA256SUMS does not list " + ", ".join(weights) + " — these are "
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


def _marker_still_holds(
    marker: Path, sums_digest: str, inventory: dict[str, dict[str, Any]]
) -> bool:
    """Whether a previous verification of *this* manifest still describes disk.

    The manifest digest alone would say "verified once"; the recorded inventory
    is what makes that claim about the files rather than about the bookkeeping.
    The comparison is exact in both directions — a recorded file whose size or
    mtime moved fails it, and so does a file present now that the marker never
    saw, since the marker has to describe *the* snapshot rather than a snapshot
    that was once a prefix of this one.

    Anything unreadable, unparseable or short of an exact match sends the
    caller back through the full hash — the expensive answer is the safe one.
    """
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict) or record.get("sha256sums") != sums_digest:
        return False
    files = record.get("files")
    if not isinstance(files, dict) or not files:
        return False
    # JSON round-trips the inventory's dict-of-dicts exactly, so one
    # comparison covers presence, absence, every recorded stat, each entry's
    # kind and a link's target — a repointed or swapped-in symlink is a
    # mismatch here, not a fast path.
    return bool(files == inventory)


def _verify_against_release_sums(
    hub: Any, repo: str, revision: str | None, name: str, path: Path
) -> None:
    """Hash one downloaded file against the release's ``SHA256SUMS``.

    The checkpoint path verifies its whole snapshot; single-file fetches —
    voices and the voice encoder — are exactly the artefacts worth guarding
    (a profile is derived from a recording of a person), so each is checked on
    download. The manifest itself is fetched once per ``(repo, revision)`` and
    cached.

    Same two rules as :func:`_verify_sha256sums`, arriving by the other door:
    an official release without a manifest is refused rather than skipped, and
    a manifest that does not list the file just fetched does not get to pass it
    as verified. The second is a refusal here where the snapshot path only
    warns, because this call names one file and asks for one answer about it:
    there is no correct release in which the voice or the encoder ``load`` is
    about to open is absent from the release's own manifest.
    """

    try:
        sums_path = hub.hf_hub_download(repo_id=repo, filename="SHA256SUMS", revision=revision)
    except Exception as exc:  # noqa: BLE001 — the client raises its own hierarchy
        if not _is_not_found(exc):
            raise  # a timeout or proxy 500 must not ship the file unverified
        if _is_official(repo):
            raise _missing_manifest(repo) from exc
        return  # a third-party upload has nothing to check against
    entries = _parse_sha256sums(Path(sums_path))
    # The snapshot path holds an official repo to `release.json` saying
    # `profile: full-0.1, verified: true` before any file from it is trusted;
    # a voice or the encoder fetched alone was skipping that claim entirely,
    # so the same repo answered strictly through one door and leniently
    # through the other.
    if _is_official(repo):
        _require_verified_release(hub, repo, revision, entries)
    expected = entries.get(name)
    if expected is None:
        raise ValueError(
            f"{repo}: SHA256SUMS does not list {name} — the file was downloaded "
            "and nothing vouches for it. Pin a revision you trust, or pass a "
            "local path."
        )
    from .checkpoint import file_sha256

    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{path}: downloaded file failed the release checksum — delete the "
            "cached file and retry, or pin a revision you trust."
        )


def _require_verified_release(
    hub: Any, repo: str, revision: str | None, entries: dict[str, str]
) -> None:
    """Fetch an official repo's ``release.json`` and hold it to its claim.

    The record is itself checked against the manifest before it is believed —
    ``build_release`` writes it first and covers it, so an official
    ``SHA256SUMS`` that does not list it, or a record whose bytes do not
    match, is the same event as a missing one: nothing vouches that this
    snapshot is a release.
    """
    try:
        fetched: str = hub.hf_hub_download(
            repo_id=repo, filename="release.json", revision=revision
        )
    except Exception as exc:  # noqa: BLE001 — the client raises its own hierarchy
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
    from .checkpoint import file_sha256

    if file_sha256(record) != expected:
        raise ValueError(
            f"{record}: release.json failed the release checksum — delete the "
            "cached file and retry, or pin a revision you trust."
        )
    _check_release_record(record, repo)


def _root_checkpoints(directory: Path) -> list[Path]:
    """Every root-level file that could be a checkpoint, sorted.

    Voices are safetensors too, and they live in `voices/`; a non-recursive
    glob is what distinguishes them from the weights.

    The voice encoder does not: `build_release` writes `ve.safetensors` beside
    the checkpoint, so every cloning-capable release had two files here and the
    counting rule below refused all of them — `loudkit.load("loudreader/loudr-1")`,
    the line in the README, answered "2 checkpoints — name the one you mean".
    It is a fixed name, which is why `VOICE_ENCODER_NAME` exists; excluding it by
    that name keeps this a question about the weights rather than about how many
    safetensors a release happens to carry.

    `Path.glob` is not a shell: `*.safetensors` matches `.hidden.safetensors`
    too. The inventory now covers such a file, so a downloaded snapshot could
    not smuggle one past the manifest either way. But these functions are also
    the whole of `resolve_checkpoint("./some-dir")`, where there is no
    manifest and nothing verifies anything. A dot-prefixed checkpoint is not
    something `build_release` can produce, so it is not a candidate here.
    """
    return sorted(
        p
        for p in directory.glob(CHECKPOINT_GLOB)
        if p.is_file() and p.name != VOICE_ENCODER_NAME and not p.name.startswith(".")
    )


def _only_checkpoint_in(directory: Path) -> Path:
    """The synthesis artefact in ``directory``, or a useful complaint.

    Three rules, in order, and the first is the one the split made necessary:

    1. **:data:`CHECKPOINT_NAME`, if it is there.** A release is two files now,
       and a directory holding both is the ordinary case rather than an
       ambiguity to complain about.
    2. **Otherwise the file that declares the synthesis role.** A release
       someone unpacked and renamed still says what each half is, in its own
       manifest.
    3. **Otherwise the old rule: exactly one candidate**, with anything
       declaring the *enrollment* role set aside first. This is the pre-split
       case and the hand-assembled directory, where one file carries every
       tensor and no manifest makes any claim at all.
    """
    named = directory / CHECKPOINT_NAME
    if named.is_file():
        _refuse_role(named, SYNTHESIS_ROLE)
        return named
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
        f"{directory}: {len(candidates)} checkpoints ({names}) — name the one you mean"
    )


def _enrollment_in(directory: Path) -> Path:
    """The enrollment artefact in ``directory``, or a useful complaint.

    Same shape as :func:`_only_checkpoint_in` — canonical name, then declared
    role — with one extra rule at the end, which is the whole reason this
    resolver is separate rather than a flag:

    **A pre-split checkpoint answers for both.** A directory a user assembled
    by hand may hold one older file carrying every tensor, including the two
    enrollment modules. Its manifest declares no ``artifact_role``, and that
    absence is the evidence: nothing has claimed the file is synthesis-only,
    and before the split nothing could be. A file that *does* declare
    :data:`SYNTHESIS_ROLE` is the opposite claim, and gets the synthesis-only
    error rather than a confusing failure inside the enroller about tensors
    that are not there.
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
            f"{directory}: {len(declared)} enrollment artefacts ({names}) — "
            "name the one you mean"
        )
    return _enrollment_from_presplit(_only_checkpoint_in(directory), directory)


def _paired_with_the_checkpoint_here(enrollment: Path, directory: Path) -> Path:
    """``enrollment``, once it is known to match the checkpoint beside it.

    Only when there *is* one: a directory holding the enrollment half alone is
    a legitimate thing to enroll from, and there is nothing to disagree with.
    """
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

    A release is the same shape whether it was downloaded or unpacked by hand —
    one checkpoint beside a ``voices/`` directory — so a caller who already has
    it locally should be able to say ``repo="./loudr-1"`` and get the same
    answers without a network round trip. Same rule as everywhere else in this
    module: anything that exists on disk is a path, always.

    Which is why a path that exists and is **not** a directory is refused here
    rather than returned as ``None``. Returning ``None`` meant "treat it as a
    repo id", so ``repo="./loudr-1.safetensors"`` — a real file, on this
    disk, that the caller obviously meant locally — fell through to the network
    and asked a remote host about it. That contradicts the rule this docstring
    states, and it is the one failure mode that leaves the machine.

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
            f"{CHECKPOINT_GLOB} beside a {VOICE_DIR}/ directory — pass the "
            "directory that contains them, or a Hugging Face repo id."
        )
    return None


def resolve_voice(ref: str, *, repo: str | None = None, revision: str | None = None) -> Path:
    """A local path for a voice, by path or by name from a released repo.

    ``resolve_voice("kathleen", repo="loudreader/loudr-1")`` fetches the one
    ~150 KB file rather than the whole release, because picking a second voice
    should not re-download a gigabyte. ``repo`` may also be a directory holding
    an unpacked release, which resolves against ``voices/`` inside it.
    """
    path = Path(ref)
    if path.is_file():
        return path
    if repo is None:
        # `available` is left empty: the only place to look would be a remote
        # repo the caller has not named, and an error message is not worth a
        # network round trip.
        raise VoiceNotFoundError(
            f"{ref}: no such file. Pass a path, or `repo=` to fetch a voice by name.",
            ref=ref,
        )
    name = ref if ref.endswith(VOICE_SUFFIX) else f"{ref}{VOICE_SUFFIX}"
    # Both branches below turn this name into a path: the local one by joining,
    # the remote one by handing it to `hf_hub_download`, which joins it into the
    # cache tree. The separator check lives here, once, so neither branch has
    # to trust the caller: a name like "../../id_rsa" must not escape the
    # release tree on either path.
    if not name or name.startswith(".") or "/" in name or "\\" in name:
        raise VoiceNotFoundError(
            f"{ref}: a voice is named, not addressed. Pass a bare name, or a "
            "full path as the first argument with no `repo=`.",
            ref=ref,
        )
    local = _release_tree(repo)
    if local is not None:
        voices_dir = (local / VOICE_DIR).resolve()
        candidate = (voices_dir / name).resolve()
        # A name is a name, not a path. Without this, `resolve_voice("../../id_rsa",
        # repo=tree)` reads a file outside the release the caller named — the
        # same class of hole `VoiceLibrary.load` closes for the server, arriving
        # by a different door because this branch builds its path by joining
        # rather than by looking a name up in a listing. Checked after
        # `resolve()` so that a symlink out of the tree is caught too.
        if not candidate.is_relative_to(voices_dir):
            raise VoiceNotFoundError(
                f"{ref}: a voice is named, not addressed — {name!r} escapes "
                f"{voices_dir}. Pass a bare name, or a full path as the first "
                "argument with no `repo=`.",
                ref=ref,
                available=list_voices(repo=repo, revision=revision),
            )
        if candidate.is_file():
            return candidate
        # Listing is a directory read here, so the alternatives cost nothing —
        # which is the condition `VoiceNotFoundError.available` documents. The
        # remote branch below still cannot afford it.
        available = list_voices(repo=repo, revision=revision)
        raise VoiceNotFoundError(
            f"{ref}: no such voice in {local / VOICE_DIR}",
            ref=ref,
            available=available,
        )
    hub = _hub()
    try:
        downloaded: str = hub.hf_hub_download(
            repo_id=repo, filename=f"{VOICE_DIR}/{name}", revision=revision
        )
    except Exception as exc:  # noqa: BLE001 — the client raises its own hierarchy
        # Translated rather than propagated. `loudkit.voice("nope", repo=...)`
        # ended in a `huggingface_hub` traceback naming a cache path and an HTTP
        # status, from a call whose whole job is "find me this voice". The
        # library has an error for "no such voice"; a network answer to that
        # question is still an answer to that question.
        #
        # But only when the repo answered. A missing or unreachable repo used
        # to be reported as "kathleen: no voice by that name in org/typo",
        # which sends the reader hunting for a misspelt voice in the one word
        # they got right — and offers `available=`, a listing that would fail
        # the same way. The repo-level diagnosis comes first.
        friendly = _friendly_hub_error(exc, repo, revision)
        if friendly is not None:
            raise friendly from exc
        if not _is_not_found(exc):
            raise
        raise VoiceNotFoundError(f"{ref}: no voice by that name in {repo}", ref=ref) from exc
    path = Path(downloaded)
    _verify_against_release_sums(hub, repo, revision, f"{VOICE_DIR}/{name}", path)
    return path


def resolve_voice_encoder(ref: str, *, revision: str | None = None) -> Path:
    """A local path for the utterance voice encoder a release ships.

    ``ref`` is whatever the caller handed :func:`loudkit.enroll` — a repo id, a
    directory holding an unpacked release, or a checkpoint file. In the last two
    cases the encoder is its sibling at the release root, which is where
    ``tools/build_release.py`` writes it; for a repo id it is one 5.7 MB
    download rather than the whole release.

    Raises:
        FileNotFoundError: the release has no encoder, which means it is a
            synthesis-only release. Said in those words, because the alternative
            was the enroller's own complaint about a ``voice_encoder_weights``
            argument that :func:`loudkit.enroll` does not take — a remedy no
            caller of the public API could act on.
    """
    path = Path(ref)
    if path.is_file():
        sibling = path.parent / VOICE_ENCODER_NAME
        if sibling.is_file():
            return sibling
        raise FileNotFoundError(
            f"{sibling} is missing: cloning needs the utterance voice encoder, "
            f"which a release ships as {VOICE_ENCODER_NAME} beside the "
            "checkpoint. Point at a cloning-capable release, or pass "
            "voice_encoder_weights= explicitly."
        )
    local = _release_tree(ref)
    if local is not None:
        candidate = local / VOICE_ENCODER_NAME
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"{candidate} is missing: {ref} is a synthesis-only release, with "
            "no utterance voice encoder to clone with."
        )
    hub = _hub()
    downloaded: str = hub.hf_hub_download(
        repo_id=ref, filename=VOICE_ENCODER_NAME, revision=revision
    )
    path = Path(downloaded)
    _verify_against_release_sums(hub, ref, revision, VOICE_ENCODER_NAME, path)
    return path


def list_voices(*, repo: str, revision: str | None = None) -> tuple[str, ...]:
    """The voice names a release holds, sorted.

    Reads the repo's file list rather than downloading anything: a caller
    choosing a voice wants the menu, not a gigabyte. A directory holding an
    unpacked release is read straight off disk, which is also what makes this
    testable without a network.

    Sorted because the answer is shown to people and compared in tests, and
    neither the hub's listing order nor a filesystem's directory order is
    stable enough to be either.
    """
    local = _release_tree(repo)
    if local is not None:
        found = (local / VOICE_DIR).glob(f"*{VOICE_SUFFIX}")
        return tuple(sorted(p.stem for p in found if p.is_file()))

    hub = _hub()
    files: list[str] = hub.list_repo_files(repo_id=repo, revision=revision)
    prefix = f"{VOICE_DIR}/"
    return tuple(
        sorted(
            name[len(prefix) : -len(VOICE_SUFFIX)]
            for name in files
            # `count("/") == 1` keeps a nested `voices/archive/old.safetensors`
            # out: a name with a separator in it is not something the rest of
            # this module would accept back, and half a path is worse than an
            # omission.
            if name.startswith(prefix) and name.endswith(VOICE_SUFFIX) and name.count("/") == 1
        )
    )
