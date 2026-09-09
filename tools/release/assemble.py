"""Copy every piece into a staging directory and write the two manifests.

``release.json`` is written before ``SHA256SUMS`` so the file that names the
profile and the verified flag is covered by the checksums. The staging
directory is a sibling of ``--out`` and is renamed into place by ``_commit``
only after every check passed; a run that dies leaves a pid-stamped tree the
next run reclaims.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import (
    BRANDING,
    DOCUMENTS,
    ENROLL_COREML,
    ENROLL_ONNX,
    ENROLLMENT_CHECKPOINT_NAME,
    EXPORT_RECORD,
    REPO,
    SAMPLES,
    STRICT,
    SYNTHESIS_COREML,
    SYNTHESIS_ONNX,
    TURBO_CHECKPOINT_NAME,
    TURBO_DOCUMENTS,
    TURBO_PROFILE,
    VOICE_ENCODER_NAME,
    sha256,
    synthesis_files,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from collections.abc import Sequence


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

    A release either vouches for what it ships or it does not. The graphs and
    the packages ship, so every file in them is hashed.

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
        rel = f.relative_to(src)
        # The audit allowlists a package by *prefix*, because an `.mlpackage`'s
        # internal layout belongs to coremltools rather than to this tool, so
        # anything sitting inside one ships and is checksummed, and the release
        # is cut on macOS, where `.DS_Store` is near-certain. A dotfile is not
        # part of a package coremltools wrote.
        if any(part.startswith(".") for part in rel.parts):
            raise ValueError(
                f"{label}: {rel} is a dotfile inside an exported package. The "
                f"audit allowlists a package by prefix, so this would ship and "
                f"be checksummed as part of the release. Remove it and re-cut."
            )
        # `shutil.copy2` dereferences a link, so the bundle-level symlink check
        # never sees one: the target's *bytes* arrive under the link's name.
        if f.is_symlink():
            raise ValueError(
                f"{label}: {rel} is a symlink. Copying follows it, so whatever "
                f"it points at would be published under this name."
            )
        entries.append(copy(f, dst / rel, root, f"{label}/{f.name}"))
    if not entries:
        raise FileNotFoundError(f"{label} is empty: {src}")
    return entries


def _checksum_entries(files: dict[str, Any]) -> list[dict[str, object]]:
    """Flatten release.json entries into the ordered list SHA256SUMS names.

    ``files`` also carries scalar metadata, ``profile`` and ``verified``,
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
        print("  WARNING: no voice profiles found: the quickstart will not work")
    return copied


def loudr1(
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
    # export directory happened to hold: a stale graph, a `.tmp.mlpackage`
    # left by an interrupted export, an editor's backup file. All of it
    # shipped and checksummed as part of the release.
    if strict:
        # `export.json` travels with the graphs it describes. Requiring it in
        # the export directory and then leaving it behind made the runtime
        # check dead on arrival: every load of a correct release found no
        # record and warned about it, which is the worst of both: a warning
        # nobody can act on, and no check where one was wanted.
        files["onnx"] = [
            copy(onnx_dir / name, out / "onnx" / name, out, f"onnx/{name}")
            for name in (*SYNTHESIS_ONNX, *ENROLL_ONNX, EXPORT_RECORD)
        ]
        coreml: list[dict[str, object]] = []
        for name in SYNTHESIS_COREML + ENROLL_COREML:
            coreml.extend(
                copy_tree(coreml_dir / name, out / "coreml" / name, out, f"coreml/{name}")
            )
        coreml.append(
            copy(
                coreml_dir / EXPORT_RECORD,
                out / "coreml" / EXPORT_RECORD,
                out,
                f"coreml/{EXPORT_RECORD}",
            )
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
            print(f"  note: no {kind}/ beside the checkpoint: not shipping it")
    return _with_documents(files, out)


def _with_documents(files: dict[str, Any], out: Path) -> dict[str, Any]:
    """The documents, brand image and samples, all covered by the manifests.

    Each copy's entry goes into ``files``, so LICENSE, NOTICE, README and
    RESPONSIBLE_USE carry a checksum line the way the weights do. The samples
    take the same route: a model
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
    Written the other way round it is the one file nothing vouches for.

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


def turbo(
    out: Path,
    *,
    ckpt: Path,
    tokenizer: Path,
    voice_encoder: Path,
    voices: dict[str, Path],
    roster: Sequence[str],
    enrollment: Path,
    onnx_dir: Path,
    coreml_dir: Path,
    samples: Path,
) -> dict[str, Any]:
    """Copy every piece into ``out`` and return the ``release.json`` body."""
    files: dict[str, Any] = {"profile": TURBO_PROFILE}
    files["checkpoint"] = copy(ckpt, out / TURBO_CHECKPOINT_NAME, out, "checkpoint")

    # The mirror is written from the manifest the checkpoint embeds, not
    # copied from a sibling the packer happened to leave beside it. The
    # embedded copy is the authority (`checkpoint.read_manifest`), so a mirror
    # taken from anywhere else is a file that can disagree with the weights it
    # claims to describe.
    from loudkit.checkpoint import read_manifest

    mirror = out / "manifest.json"
    mirror.write_text(
        json.dumps(read_manifest(str(ckpt)), indent=1) + "\n", encoding="utf-8", newline=""
    )
    files["manifest"] = {
        "path": "manifest.json",
        "sha256": sha256(mirror),
        "bytes": mirror.stat().st_size,
    }
    print(f"  {'manifest':22s} {mirror.stat().st_size / 1e6:8.1f} MB  written from the weights")

    files["tokenizer"] = copy(tokenizer, out / "tokenizer.json", out, "tokenizer")
    files["voice_encoder"] = copy(voice_encoder, out / VOICE_ENCODER_NAME, out, "voice encoder")
    files["voices"] = _copy_voices(voices, roster, out, strict=True)

    for source, name, key in TURBO_DOCUMENTS:
        files[key] = copy(REPO / source, out / name, out, name)
    source, name, key = BRANDING
    files[key] = copy(REPO / source, out / name, out, name)
    files["enrollment"] = copy(enrollment, out / ENROLLMENT_CHECKPOINT_NAME, out, "enrollment")
    files["onnx"] = [
        copy(onnx_dir / name, out / "onnx" / name, out, f"onnx/{name}")
        for name in (*synthesis_files("fusion_mtp2", ".onnx"), *ENROLL_ONNX, EXPORT_RECORD)
    ]
    coreml: list[dict[str, object]] = []
    for name in synthesis_files("fusion_mtp2", ".mlpackage") + ENROLL_COREML:
        coreml.extend(
            copy_tree(coreml_dir / name, out / "coreml" / name, out, f"coreml/{name}")
        )
    coreml.append(
        copy(
            coreml_dir / EXPORT_RECORD, out / "coreml" / EXPORT_RECORD, out, "coreml provenance"
        )
    )
    files["coreml"] = coreml
    files["samples"] = [
        copy(samples / Path(name).name, out / name, out, name) for _, name in SAMPLES
    ]
    return files


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
    if os.name != "posix" or sys.platform != "darwin":
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

    Refuses when a ``.previous-`` tree under this build's own pid is already
    there with no release beside it. That is the recovery case above, and pids
    come round again after a reboot, so a new build can be handed the same
    name; clearing it here would take the copy ``_sweep_stale`` deliberately
    kept for a person to look at.
    """
    previous = out.parent / f".{out.name}.previous-{os.getpid()}"
    if previous.exists() and not out.exists():
        raise FileExistsError(
            f"{previous} is the only surviving copy of the last release: no "
            f"{out.name} sits beside it, so an earlier build was interrupted "
            f"between the two renames. Rename it back to {out.name}, or move "
            f"it away, before building here again."
        )
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
