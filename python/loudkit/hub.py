"""Resolving a checkpoint by name instead of by path.

The rule is deliberately dumb: anything that exists on disk is a path, always;
``org/name`` is a Hugging Face repo id; anything else is a path that does not
exist, and the error says so. This module holds that rule, the hub client and
everything that talks to it (the resolvers, the listing, the cache), and the
receipt ``.loudkit-release.json`` that :func:`download` writes beside a
``local_dir`` fetch so a later run can tell a verified directory from a pile
of files. What a release is lives in :mod:`loudkit.release`; the ``SHA256SUMS``
rules live in :mod:`loudkit.checksums`. Their names are re-exported here,
because callers and tests learned them at this door.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from . import checksums, release
from .checksums import (
    _is_not_found,
    _release_sums,
    _require_verified_release,
    _verify_against_release_sums,
    _verify_sha256sums,
)
from .errors import VoiceNotFoundError
from .release import (
    BACKENDS,
    CHECKPOINT_GLOB,
    CHECKPOINT_NAME,
    CHECKPOINT_NAMES,
    ENROLLMENT_NAME,
    ENROLLMENT_ROLE,
    SYNTHESIS_ROLE,
    TURBO_CHECKPOINT_NAME,
    VOICE_DIR,
    VOICE_ENCODER_NAME,
    VOICE_SUFFIX,
    _enrollment_for_file,
    _enrollment_from_presplit,
    _enrollment_in,
    _is_official,
    _only_checkpoint_in,
    _refuse_role,
    _release_tree,
    _voice_encoder_beside,
    _voice_encoder_in,
    _voice_in_tree,
    _voice_names,
    backend_for_device,
    release_patterns,
    verify_release_inventory,
)

__all__ = [
    "BACKENDS",
    "CHECKPOINT_GLOB",
    "CHECKPOINT_NAME",
    "CHECKPOINT_NAMES",
    "TURBO_CHECKPOINT_NAME",
    "ENROLLMENT_NAME",
    "ENROLLMENT_ROLE",
    "RECEIPT_NAME",
    "Receipt",
    "SYNTHESIS_ROLE",
    "VOICE_DIR",
    "VOICE_ENCODER_NAME",
    "VOICE_SUFFIX",
    "backend_for_device",
    "download",
    "is_repo_id",
    "list_voices",
    "read_receipt",
    "receipt_hit",
    "release_patterns",
    "resolve_checkpoint",
    "resolve_enrollment_checkpoint",
    "resolve_voice",
    "resolve_voice_encoder",
    "verify_release_inventory",
    "write_receipt",
]

# Private names the CLI and the tests still reach through this module.
_artifact_role = release._artifact_role
_manifest_of = release._manifest_of
_root_checkpoints = release._root_checkpoints
_parse_sha256sums = checksums._parse_sha256sums

_LOG = logging.getLogger("loudkit.hub")
"""Where this module's notices go. A library writing straight to stderr gives
a host application no way to route or silence them; a logger at WARNING still
reaches stderr through ``logging.lastResort`` on an unconfigured process, so
the notice a user must see is not lost."""

_REPO_ID = re.compile(r"^[\w\-.]+/[\w\-.]+$")  # org/name; dots are legal in both

_MISSING_HUB = (
    "resolving {ref!r} needs the Hugging Face client.\n"
    '  pip install "loudkit[hub]"\n'
    "Or download the files yourself and pass a path."
)


def is_repo_id(ref: str) -> bool:
    """True for ``org/name``, false for anything that exists on disk, checked
    in that order. Path-shaped strings are paths even before they exist."""
    if Path(ref).exists():
        return False
    if ref.startswith((".", "/", "~")) or ref.endswith(".safetensors"):
        return False
    return bool(_REPO_ID.match(ref))


def _hub() -> Any:
    """The Hugging Face client, or a message naming the extra that carries it."""
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - depends on extras
        # `name=` because the CLI reads `exc.name` to say which package is absent.
        raise ModuleNotFoundError(
            _MISSING_HUB.format(ref="a repo id"), name="huggingface_hub"
        ) from exc
    return huggingface_hub


def _not_a_ref(ref: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"{ref}: no such file or directory, and not a Hugging Face repo id "
        "(those look like 'org/name')"
    )


def resolve_checkpoint(
    ref: str, *, revision: str | None = None, backend: str = "torch"
) -> Path:
    """A local path for the synthesis artefact of ``ref``: a ``.safetensors``,
    a directory holding one, or a repo id such as ``loudreader/loudr-1``.
    ``backend`` picks which set to fetch for a repo id, because the checkpoint
    alone is not the release; ``revision`` pins it, since a moving ``main`` is
    a moving model. Raises ``FileNotFoundError`` for no such path."""
    path = Path(ref)
    if path.is_file():
        _refuse_role(path, SYNTHESIS_ROLE)
        return path
    if path.is_dir():
        return _only_checkpoint_in(path)
    if not is_repo_id(ref):
        raise _not_a_ref(ref)

    hub = _hub()
    allow, ignore = release_patterns(backend)
    cached = _cached_snapshot(hub, ref, revision, backend, allow, ignore)
    if cached is not None and _current(hub, ref, revision, cached):
        return _only_checkpoint_in(cached)
    try:
        _refuse_before_fetching(hub, ref, revision, backend)
        local: str = hub.snapshot_download(
            repo_id=ref,
            revision=revision,
            allow_patterns=list(allow),
            ignore_patterns=list(ignore) or None,
        )
    except Exception as exc:  # mapped by name below
        friendly = _friendly_hub_error(exc, ref, revision)
        if friendly is None:
            raise
        raise friendly from exc
    root = Path(local)
    _verify_sha256sums(root, repo=ref)
    # After the hashes: a set can be intact and still short.
    verify_release_inventory(root, backend, require_voices=_is_official(ref))
    return _only_checkpoint_in(root)


def resolve_enrollment_checkpoint(ref: str, *, revision: str | None = None) -> Path:
    """A local path for the enrollment artefact of ``ref``. Only
    :func:`loudkit.enroll` asks, so a repo id here is one file, not a snapshot.
    Raises ``FileNotFoundError`` when the set is synthesis-only."""
    path = Path(ref)
    if path.is_file():
        return _enrollment_for_file(path)
    if path.is_dir():
        return _enrollment_in(path)
    if not is_repo_id(ref):
        raise _not_a_ref(ref)

    hub = _hub()
    try:
        downloaded: str = hub.hf_hub_download(
            repo_id=ref, filename=ENROLLMENT_NAME, revision=revision
        )
    except Exception as exc:  # the client raises its own hierarchy
        friendly = _friendly_hub_error(exc, ref, revision)
        if friendly is not None:
            raise friendly from exc
        if not _is_not_found(exc):
            raise
        # The repo holds no enrollment artefact: a pre-split release, whose
        # one checkpoint may still carry the enrollment tensors.
        return _enrollment_from_presplit(resolve_checkpoint(ref, revision=revision), ref)
    resolved = Path(downloaded)
    _verify_against_release_sums(hub, ref, revision, ENROLLMENT_NAME, resolved)
    return resolved


def _friendly_hub_error(
    exc: Exception, ref: str, revision: str | None
) -> FileNotFoundError | None:
    """A one-line diagnosis for the hub's own "it is not there" errors. The raw
    ``RepositoryNotFoundError`` ends in "Invalid username or password"."""
    if type(exc).__name__ == "RepositoryNotFoundError":
        return FileNotFoundError(
            f"{ref}: repository not found or not public (revision "
            f"{revision or 'default'}). Is the release published, and are you "
            "online? Run `loudkit doctor` to check this machine."
        )
    if type(exc).__name__ == "RevisionNotFoundError":
        return FileNotFoundError(f"{ref}: revision {revision!r} not found in the repository.")
    if type(exc).__name__ == "LocalEntryNotFoundError":
        return FileNotFoundError(
            f"{ref}: not in the local cache and the hub cannot be reached. "
            "Connect once to fetch it, or pass a path to a release directory."
        )
    return None  # not one of ours; caller re-raises


def _refuse_before_fetching(hub: Any, repo: str, revision: str | None, backend: str) -> None:
    """What a repo must say about itself before a byte of weights moves: an
    official ``release.json`` held to its profile, and for a graph backend the
    listing's graphs, so a torch-only release is refused before 1.2 GB."""
    if _is_official(repo):
        entries = _release_sums(hub, repo, revision)
        assert entries is not None  # an official repo without one raised above
        _require_verified_release(hub, repo, revision, entries)
    if backend == "torch":
        return
    files: list[str] = hub.list_repo_files(repo_id=repo, revision=revision)
    if not any(name.startswith(f"{backend}/") for name in files):
        raise ValueError(
            f"{repo} ships no {backend}/ directory, so the {backend} backend "
            "cannot run it in this version of loudkit: pass device='cpu', 'cuda' "
            f"or 'mps' to run it on torch, or use loudreader/loudr-1 for {backend}."
        )


def _cached_snapshot(
    hub: Any,
    repo: str,
    revision: str | None,
    backend: str,
    allow: tuple[str, ...],
    ignore: tuple[str, ...],
) -> Path | None:
    """The complete, already fetched snapshot for ``repo``, or ``None``. A load
    of what is already here neither fetches nor re-hashes."""
    try:
        local: str = hub.snapshot_download(
            repo_id=repo,
            revision=revision,
            allow_patterns=list(allow),
            ignore_patterns=list(ignore) or None,
            local_files_only=True,
        )
        root = Path(local)
        verify_release_inventory(root, backend, require_voices=_is_official(repo))
    except Exception:  # not cached, or cached short: fetch
        return None
    return root


def _cached_file(hub: Any, repo: str, revision: str | None, name: str) -> Path | None:
    """One release file already in the cache, at the commit ``revision``
    names today, or ``None``."""
    try:
        local: str = hub.hf_hub_download(
            repo_id=repo, filename=name, revision=revision, local_files_only=True
        )
    except Exception:  # not cached: fetch
        return None
    path = Path(local)
    return path if _current(hub, repo, revision, path) else None


_RESOLVED: dict[tuple[str, str | None], str | None] = {}
"""What each ``(repo, revision)`` resolved to in this process, ``None`` for a
hub that could not be reached: one call and one stderr line per pair, so a
load followed by two voices asks once."""


def _current(hub: Any, repo: str, revision: str | None, cached: Path) -> bool:
    """Whether ``cached``, a path inside the hub cache, is at the commit
    ``revision`` names today. A revision that is a commit is that commit
    and needs no call; otherwise one ``model_info`` call per process
    answers, and a hub that cannot be reached leaves the cache in use, with
    a line on stderr."""
    if revision is not None and _COMMIT.fullmatch(revision):
        return True
    key = (repo, revision)
    if key not in _RESOLVED:
        _RESOLVED[key] = _resolve_commit(hub, repo, revision)
        if _RESOLVED[key] is None:
            _LOG.warning(
                "loudkit: the hub cannot be reached; using the cached %s at %s",
                repo,
                _snapshot_commit(cached),
            )
    commit = _RESOLVED[key]
    return commit is None or _snapshot_commit(cached) == commit


def _snapshot_commit(path: Path) -> str | None:
    """The commit a hub-cache path sits under: ``.../snapshots/<commit>/...``,
    the last such component, since the cache root may hold the word too."""
    parts = path.parts
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == "snapshots":
            return parts[index + 1]
    return None


def resolve_voice(ref: str, *, repo: str | None = None, revision: str | None = None) -> Path:
    """A local path for a voice, by path or by name from a released repo.

    ``resolve_voice("kathleen", repo="loudreader/loudr-1")`` fetches the one
    ~150 KB file rather than the whole release. ``repo`` may also be a
    directory holding an unpacked release.
    """
    path = Path(ref)
    if path.is_file():
        return path
    if repo is None:
        raise VoiceNotFoundError(
            f"{ref}: no such file. Pass a path, or `repo=` to fetch a voice by name.",
            ref=ref,
        )
    name = ref if ref.endswith(VOICE_SUFFIX) else f"{ref}{VOICE_SUFFIX}"
    # A name is a name, not a path: "../../id_rsa" must not escape the release
    # tree on either branch below. NUL is here for a different reason: no path
    # can hold one, so the tree branch's `resolve()` raises `ValueError: lstat:
    # embedded null character in path`, an OS-layer sentence naming neither the
    # field it came from nor what a caller should send instead.
    if not name or name.startswith(".") or "/" in name or "\\" in name or "\x00" in name:
        raise VoiceNotFoundError(
            f"{ref}: a voice is named, not addressed. Pass a bare name, or a "
            "full path as the first argument with no `repo=`.",
            ref=ref,
        )
    local = _release_tree(repo)
    if local is not None:
        return _voice_in_tree(local, ref, name)
    hub = _hub()
    cached = _cached_file(hub, repo, revision, f"{VOICE_DIR}/{name}")
    if cached is not None:
        return cached
    try:
        downloaded: str = hub.hf_hub_download(
            repo_id=repo, filename=f"{VOICE_DIR}/{name}", revision=revision
        )
    except Exception as exc:  # the client raises its own hierarchy
        # The repo-level diagnosis first: a missing repo is not a misspelt voice.
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
    """A local path for the utterance voice encoder a release ships: beside
    the checkpoint for a file or a directory, one 5.7 MB download for a repo.

    Raises:
        FileNotFoundError: the release is synthesis-only.
    """
    path = Path(ref)
    if path.is_file():
        return _voice_encoder_beside(path)
    local = _release_tree(ref)
    if local is not None:
        return _voice_encoder_in(local, ref)
    hub = _hub()
    cached = _cached_file(hub, ref, revision, VOICE_ENCODER_NAME)
    if cached is not None:
        return cached
    try:
        downloaded: str = hub.hf_hub_download(
            repo_id=ref, filename=VOICE_ENCODER_NAME, revision=revision
        )
    except Exception as exc:
        # The repo-level diagnosis first: a missing repo is not a release that
        # ships no encoder, and the raw client error for it ends in "Invalid
        # username or password".
        friendly = _friendly_hub_error(exc, ref, revision)
        if friendly is not None:
            raise friendly from exc
        if not _is_not_found(exc):
            raise
        raise FileNotFoundError(
            f"{ref}: the release ships no {VOICE_ENCODER_NAME}, so it is a "
            "synthesis-only set: it can speak but not clone. Point at a "
            "cloning-capable release, or pass voice_encoder_weights= explicitly."
        ) from exc
    path = Path(downloaded)
    _verify_against_release_sums(hub, ref, revision, VOICE_ENCODER_NAME, path)
    return path


def list_voices(*, repo: str, revision: str | None = None) -> tuple[str, ...]:
    """The voice names a release holds, sorted. Reads the repo's file list
    rather than downloading anything; a directory is read off disk."""
    local = _release_tree(repo)
    if local is not None:
        return _voice_names(local)
    hub = _hub()
    try:
        files: list[str] = hub.list_repo_files(repo_id=repo, revision=revision)
    except Exception as exc:  # mapped by name, like every other fetch
        friendly = _friendly_hub_error(exc, repo, revision)
        if friendly is None:
            raise
        raise friendly from exc
    prefix = f"{VOICE_DIR}/"
    return tuple(
        sorted(
            name[len(prefix) : -len(VOICE_SUFFIX)]
            for name in files
            # A nested `voices/archive/old.safetensors` is not a voice name.
            if name.startswith(prefix) and name.endswith(VOICE_SUFFIX) and name.count("/") == 1
        )
    )


# ------------------------------------------------------------- the receipt

RECEIPT_NAME = ".loudkit-release.json"
"""What a verified download directory carries, in every port: ``revision`` as
asked, ``commit`` as the hub resolved it, ``sha256sums`` (``null`` when the
release ships none) and ``fetched_at`` in UTC. :func:`read_receipt` hands one
back only when it vouches for the repo asked; a receipt naming the commit the
asked revision resolves to today is a hit and hashes nothing; any other, or
none, means fetch and verify again."""

RECEIPT_FIELDS = ("repo", "revision", "commit", "sha256sums", "fetched_at")

_RECEIPT_LIMIT = 1 << 20
"""The most a receipt file is read: five short fields. A file past it is not
a receipt, and is not read."""

_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Receipt:
    """A receipt :func:`read_receipt` accepted."""

    repo: str
    revision: str
    commit: str
    sha256sums: str | None
    fetched_at: str


def _refuse_a_cache_that_is_not_the_pin(
    local_dir: Path, receipt: Receipt, revision: str | None
) -> None:
    """Offline, answer a pinned request with those bytes or with nothing.

    A caller asking for one revision and getting another is the failure a pin
    exists to prevent, and a warning is not a refusal: a build that pins its
    weights would run on different ones and report success.

    Both halves of the pin are already on disk. A forty-hex request names the
    commit, which the receipt records, so it is answered exactly. A tag or a
    branch cannot be resolved with no hub, but the receipt also records the ref
    it was fetched for, and asking for a different one cannot be satisfied from
    this directory either.
    """
    if not revision:
        return
    if _COMMIT.fullmatch(revision):
        if revision == receipt.commit:
            return
        raise ValueError(
            f"{local_dir}: holds {receipt.repo} at {receipt.commit}, and the hub "
            f"cannot be reached to fetch {revision}. A pinned revision is not a "
            "preference: rerun with the hub reachable, or point --local-dir at a "
            "copy of that commit."
        )
    if revision == receipt.revision:
        return
    raise ValueError(
        f"{local_dir}: was fetched for {receipt.revision!r} ({receipt.commit}), the "
        f"hub cannot be reached, and {revision!r} cannot be resolved without it. "
        "Rerun with the hub reachable, or ask for the revision this copy holds."
    )


def _parse_receipt(root: Path) -> Receipt | None:
    """The file under ``root`` as a :class:`Receipt`, or ``None`` when it is
    unreadable, not an object, or has a field missing or of the wrong type
    (``null`` is the wrong type for all but ``sha256sums``; fields it does
    not know are ignored). Judged by nothing else; :func:`read_receipt` is
    the one caller."""
    path = root / RECEIPT_NAME
    try:
        if path.stat().st_size > _RECEIPT_LIMIT:
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or any(f not in record for f in RECEIPT_FIELDS):
        return None
    typed = all(isinstance(record[f], str) for f in RECEIPT_FIELDS if f != "sha256sums")
    if not typed or not isinstance(record["sha256sums"], (str, type(None))):
        return None
    return Receipt(*(record[f] for f in RECEIPT_FIELDS))


def read_receipt(
    root: Path, repo: str, *, backend: str = "torch", cloning: bool = False
) -> Receipt | None:
    """The receipt under ``root`` when it vouches for ``repo``, else ``None``:
    every field present with its type, ``repo`` the one asked, ``commit``
    forty lowercase hex, ``sha256sums`` the digest of the ``SHA256SUMS`` on
    disk (``None`` for none), and every file it lists that the plan selects
    present. No weight is hashed: the receipt is checked, not the release.
    This is the only reader of the file."""
    # Looked up per call rather than bound at import: the cache-hit cases in
    # tests/test_hub.py replace `checkpoint.file_sha256` to prove that a hit
    # hashes no weight, and a module-scope binding would outlive the patch.
    from .checkpoint import file_sha256

    receipt = _parse_receipt(root)
    if receipt is None or receipt.repo != repo or not _COMMIT.fullmatch(receipt.commit):
        return None
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        return receipt if receipt.sha256sums is None else None
    try:
        if receipt.sha256sums != file_sha256(sums):
            return None
        listed = _parse_sha256sums(sums)
    except (OSError, ValueError):
        return None
    allow, ignore = release_patterns(backend, cloning=cloning)
    present = all(
        (root / name).is_file()
        for name in listed
        if any(fnmatch(name, a) for a in allow) and not any(fnmatch(name, i) for i in ignore)
    )
    return receipt if present else None


def write_receipt(root: Path, *, repo: str, revision: str | None, commit: str) -> None:
    """Record that ``root`` holds ``repo`` at ``commit``, verified."""
    # The same call-time lookup as `read_receipt`, for the same reason.
    from .checkpoint import file_sha256

    sums = root / "SHA256SUMS"
    receipt = {
        "repo": repo,
        "revision": revision if revision is not None else "main",
        "commit": commit,
        "sha256sums": file_sha256(sums) if sums.is_file() else None,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # `newline="\n"`, not the platform's. Text mode translates on Windows, so
    # the same receipt written there would carry CRLF and hash differently
    # from the one the four ports are held to. A receipt is a record other
    # implementations read; its bytes are not a local matter.
    (root / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=1) + "\n", encoding="utf-8", newline="\n"
    )


def receipt_hit(
    root: Path, repo: str, commit: str, *, backend: str = "torch", cloning: bool = False
) -> bool:
    """Whether ``root`` already holds ``repo`` at ``commit``: a receipt
    :func:`read_receipt` accepts, naming that commit. An empty commit never
    hits: nothing was resolved."""
    receipt = read_receipt(root, repo, backend=backend, cloning=cloning)
    return bool(commit) and receipt is not None and receipt.commit == commit


def _resolve_commit(hub: Any, repo: str, revision: str | None) -> str | None:
    """The commit ``revision`` names on the hub today, or ``None`` when the
    hub cannot be reached. An answer is never an outage: a repo or revision
    that is not there, any other HTTP status and a record naming no commit
    are errors."""
    try:
        info = hub.model_info(repo_id=repo, revision=revision)
    except Exception as exc:  # the client raises its own hierarchy
        friendly = _friendly_hub_error(exc, repo, revision)
        if friendly is not None and _is_not_found(exc):
            raise friendly from exc
        if _is_not_found(exc) or _hub_answered(exc):
            raise
        return None
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not sha:
        raise ValueError(
            f"{repo} at {revision or 'main'}: the hub named no commit for this revision"
        )
    return sha


def _hub_answered(exc: BaseException) -> bool:
    """Whether the hub answered with an HTTP status, as opposed to the
    transport failing. Matched by name: the client is an optional extra."""
    return any(cls.__name__ in ("HfHubHTTPError", "HTTPError") for cls in type(exc).__mro__)


def download(
    repo: str,
    *,
    revision: str | None = None,
    backend: str = "torch",
    cloning: bool = False,
    local_dir: Path | None = None,
) -> Path:
    """Fetch what ``backend`` needs from ``repo``, verified, and answer with the
    directory it landed in: the hub cache, which is keyed by commit already, or
    ``local_dir``, which then carries :data:`RECEIPT_NAME`. A second run whose
    receipt names today's commit, over a directory that still holds the set,
    fetches and hashes nothing; a run that cannot reach the hub uses the
    receipt and says so."""
    # Judged here, as `resolve_checkpoint` judges its own argument and as all
    # four ports judge theirs. Handing a path-shaped string straight to the
    # client answers with the client's sentence about `repo_type`, which names
    # a concept loudkit does not have and a fix that is not the fix.
    if not is_repo_id(repo):
        raise ValueError(
            f"{repo}: not a Hugging Face repo id (those look like 'org/name', "
            "such as 'loudreader/loudr-1'). This fetches from the hub; a release "
            "already on disk is a path, and every command that needs one takes it."
        )
    # Before the fetch, and before the receipt is read, because both of those
    # go through this directory. `unlink(missing_ok=True)` below swallows a
    # missing file, which is not the same as a `local_dir` that cannot hold
    # one: unchecked, that surfaces from inside the fetch as advice about a
    # truncated download that never started.
    if local_dir is not None and local_dir.exists() and not local_dir.is_dir():
        raise ValueError(
            f"{local_dir}: --local-dir is where the release lands, so it has to "
            "be a directory (or a name that does not exist yet); this is a file."
        )
    hub = _hub()
    allow, ignore = release_patterns(backend, cloning=cloning)
    commit: str | None = None
    if local_dir is not None:
        commit = _resolve_commit(hub, repo, revision)
        receipt = read_receipt(local_dir, repo, backend=backend, cloning=cloning)
        if receipt is not None and _complete(local_dir, backend, cloning=cloning):
            if commit is None:
                _refuse_a_cache_that_is_not_the_pin(local_dir, receipt, revision)
                _LOG.warning(
                    "loudkit: the hub cannot be reached; using %s, which holds "
                    "%s at %s (fetched %s)",
                    local_dir,
                    receipt.repo,
                    receipt.commit,
                    receipt.fetched_at,
                )
                return local_dir
            if receipt.commit == commit:
                return local_dir
        # A stale receipt must not outlive the files it vouched for.
        (local_dir / RECEIPT_NAME).unlink(missing_ok=True)
    # What the repo says about itself is read before a byte of weights moves:
    # the same order the four ports keep, and the one place in Python that
    # still fetched first and judged after.
    _refuse_before_fetching(hub, repo, revision, backend)
    try:
        root = Path(
            hub.snapshot_download(
                repo_id=repo,
                revision=revision,
                allow_patterns=list(allow),
                ignore_patterns=list(ignore) or None,
                local_dir=str(local_dir) if local_dir else None,
            )
        )
    except Exception as exc:  # mapped by name below
        friendly = _friendly_hub_error(exc, repo, revision)
        if friendly is None:
            raise
        raise friendly from exc
    _verify_sha256sums(root, repo=repo)
    verify_release_inventory(root, backend, cloning=cloning, require_voices=True)
    if local_dir is not None and commit is not None:
        write_receipt(root, repo=repo, revision=revision, commit=commit)
    return root


def _complete(root: Path, backend: str, *, cloning: bool) -> bool:
    """Whether ``root`` holds the inventory ``backend`` runs on, by name. The
    receipt vouches for what ``SHA256SUMS`` lists; this is the rest."""
    try:
        verify_release_inventory(root, backend, cloning=cloning, require_voices=True)
    except (OSError, ValueError):
        return False
    return True
