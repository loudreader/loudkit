"""The packed checkpoint: one file, one manifest, no re-guessing.

See ``docs/design/runtime-notes.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ASSET_PREFIX",
    "Checkpoint",
    "TOKENIZER_FILENAME",
    "decode_mode",
    "file_sha256",
    "read_header",
    "read_manifest",
    "require_decode_support",
    "resolve_dtype",
]

TOKENIZER_FILENAME = "tokenizer.json"
"""The text tokenizer's name, beside the checkpoint or packed inside it.

A release-layout fact, so it lives in the leaf every backend and the release
reader already import rather than being spelled again in each of them. The
manifest records its digest under ``tokenizer_sha256``.
"""

_SAFETENSORS_TORCH_DTYPES: dict[str, tuple[str, int]] = {
    "BOOL": ("torch.bool", 1),
    "U8": ("torch.uint8", 1),
    "I8": ("torch.int8", 1),
    "U16": ("torch.uint16", 2),
    "I16": ("torch.int16", 2),
    "F16": ("torch.float16", 2),
    "BF16": ("torch.bfloat16", 2),
    "U32": ("torch.uint32", 4),
    "I32": ("torch.int32", 4),
    "F32": ("torch.float32", 4),
    "U64": ("torch.uint64", 8),
    "I64": ("torch.int64", 8),
    "F64": ("torch.float64", 8),
    "F8_E4M3": ("torch.float8_e4m3fn", 1),
    "F8_E5M2": ("torch.float8_e5m2", 1),
}


def read_header(path: str | Path) -> tuple[dict[str, Any], int]:
    """A safetensors file's raw header and where its payload begins.

    Eight bytes of little-endian length, then that many bytes of JSON. Nothing
    here loads a tensor, so asking a checkpoint what it contains is two short
    reads whatever the file weighs.

    The header comes back as safetensors wrote it, ``__metadata__`` included:
    callers that want the embedded manifest read it from there, and callers
    that want the tensors skip that one key.

    Raises:
        ValueError: not a safetensors container, or an unreadable header.
    """
    source = Path(path)
    with source.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("too short to be a safetensors file")
        (header_length,) = struct.unpack("<Q", raw_length)
        if not 0 < header_length <= 100 << 20:
            raise ValueError(f"implausible safetensors header length {header_length}")
        try:
            header = json.loads(stream.read(header_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"unreadable safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("the safetensors header is not an object")
    return cast("dict[str, Any]", header), 8 + header_length


def payload_sha256(path: str | Path) -> str:
    """Hash a safetensors payload by the checkpoint manifest's recipe.

    The recipe predates the runtime split and records PyTorch dtype spellings,
    but verifying it does not require PyTorch. Safetensors already stores the
    dtype, shape and byte offsets in its header; reading the payload directly
    makes verification available to the base package and keeps memory bounded
    to one 1 MiB block instead of materialising the whole checkpoint.
    """
    source = Path(path)
    header, payload_start = read_header(source)
    with source.open("rb") as stream:
        payload_size = source.stat().st_size - payload_start
        digest = hashlib.sha256()
        for name in sorted(k for k in header if k != "__metadata__"):
            entry = header[name]
            if not isinstance(entry, dict):
                raise ValueError(f"tensor {name!r} has no metadata object")
            dtype = entry.get("dtype")
            if not isinstance(dtype, str):
                raise ValueError(f"tensor {name!r} has no string dtype, got {dtype!r}")
            try:
                torch_dtype, width = _SAFETENSORS_TORCH_DTYPES[dtype]
            except KeyError:
                raise ValueError(f"tensor {name!r} has unsupported dtype {dtype!r}") from None
            shape = entry.get("shape")
            offsets = entry.get("data_offsets")
            if (
                not isinstance(shape, list)
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(n, bool) or not isinstance(n, int) for n in offsets)
            ):
                raise ValueError(f"tensor {name!r} has invalid shape or data offsets")
            begin, end = offsets
            expected = math.prod(shape) * width
            if begin < 0 or end < begin or end > payload_size or end - begin != expected:
                raise ValueError(
                    f"tensor {name!r} has invalid byte range {offsets!r} for shape {shape!r}"
                )

            digest.update(name.encode())
            digest.update(torch_dtype.encode())
            digest.update(str(tuple(shape)).encode())
            stream.seek(payload_start + begin)
            remaining = end - begin
            while remaining:
                block = stream.read(min(1 << 20, remaining))
                if not block:
                    raise ValueError(f"tensor {name!r} ends before its declared byte range")
                digest.update(block)
                remaining -= len(block)
    return digest.hexdigest()


ASSET_PREFIX = "assets."
"""Tensor-name prefix for text artefacts carried inside the checkpoint.

See :meth:`Checkpoint.asset`. Chosen so `tensors("t3.")` and `tensors("s3gen.")`,
which take everything under a prefix, cannot pick these up as weights.
"""

CHECKPOINT_FORMAT = "loudkit-checkpoint"
"""Value of ``manifest["format"]`` this loader understands."""

SUPPORTED_FORMAT_VERSIONS = (1, 2)
"""Layouts this build reads.

Version 2 adds the two-token decode of ``decode_mode="fusion_mtp2"``: a second
speech head and a fusion MLP under ``t3.``, and a ``decode`` block in the
manifest that says so. It is a version rather than an optional field because a
version-1 reader finds weights it knows, runs the loop it knows, and emits
fluent nonsense, the failure this format exists to make loud.
"""

DECODE_FORMAT_VERSION = {"fusion_mtp2": 2}
"""The lowest ``format_version`` each decode mode may be declared under.

See ``docs/design/runtime-notes.md``.
"""


def decode_mode(manifest: Mapping[str, object]) -> str:
    """The decode loop ``manifest`` declares. ``"single"`` when it declares none.

    Absent means single-token, the way :func:`_check_decode_version` reads it.
    Unlike ``AlgorithmConfig``, this does not validate the name: the caller is
    deciding whether a backend can run the file, and an unknown mode is one
    nothing here can run either.
    """
    block = manifest.get("decode")
    if not isinstance(block, Mapping):
        return "single"
    return str(block.get("mode", "single"))


_KNOWN_DECODE_MODES = ("single", "fusion_mtp2")
"""The decode loops this build runs.

Spelled here rather than imported from ``config.DecodeMode`` because this
module is a leaf every backend reads before any config exists. A test pins the
two lists equal, in both directions.
"""


def require_decode_support(mode: str, target: str, *, name: str = "") -> None:
    """Refuse, in one sentence, a decode loop this build does not run.

    Every backend runs both known modes, so ``target`` names the door the
    caller came through rather than narrowing what is refused; an unknown mode
    is one nothing here runs.

    See ``docs/design/runtime-notes.md``.
    """
    if mode not in _KNOWN_DECODE_MODES:
        raise ValueError(
            f"the {target} backend cannot run {name or 'this checkpoint'}: "
            f"unsupported decode mode {mode!r}"
        )


def read_manifest(path: str | Path) -> dict[str, object]:
    """Read the manifest embedded in a packed checkpoint.

    See ``docs/design/runtime-notes.md``.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        meta: Mapping[str, str] | None = f.metadata()
    if not meta or "manifest" not in meta:
        raise ValueError(f"{path}: no embedded manifest: not a loudkit checkpoint")
    manifest = json.loads(meta["manifest"])
    # Raised, not asserted, and checked before the first `.get`: the manifest
    # is external data, `python -O` strips asserts, and a JSON list here would
    # otherwise surface as an AttributeError from inside a getter.
    if not isinstance(manifest, dict):
        raise ValueError(
            f"{path}: manifest is a {type(manifest).__name__}, expected a JSON object"
        )
    fmt = manifest.get("format")
    # `int()` over a manifest value, which is external data. `null` and a list
    # raise `TypeError`, so a caller catching the `ValueError` this module
    # documents got an unhandled crash instead of a refusal. Only the type of
    # the refusal changes here: every value `int()` accepts still loads, so
    # which checkpoints open is exactly what it was.
    declared = manifest.get("format_version", -1)
    try:
        version = int(declared)
    except (TypeError, ValueError):
        raise ValueError(
            f"{path}: manifest['format_version'] is {declared!r}, expected a version number"
        ) from None
    if fmt != CHECKPOINT_FORMAT or version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"{path}: format {fmt!r} version {version}, "
            f"this build reads {CHECKPOINT_FORMAT!r} versions {SUPPORTED_FORMAT_VERSIONS}"
        )
    _check_decode_version(path, manifest, version)
    return manifest


def _check_decode_version(
    path: str | Path, manifest: Mapping[str, object], version: int
) -> None:
    """Refuse a file whose declared version does not cover its decode mode.

    See ``docs/design/runtime-notes.md``.
    """
    block = manifest.get("decode")
    if block is None:
        return
    if not isinstance(block, Mapping):
        raise ValueError(f"{path}: manifest['decode'] must be an object")
    mode = str(block.get("mode", "single"))
    needs = DECODE_FORMAT_VERSION.get(mode)
    if needs is not None and version < needs:
        raise ValueError(
            f"{path}: manifest declares decode.mode {mode!r} under format_version "
            f"{version}, which needs {needs}. Version {version} is the one-token "
            f"loop, and the Rust, Go, TypeScript and Swift engines gate on that "
            f"number alone: under it they would run the loop they know over "
            f"these weights and produce fluent nonsense. Repack with "
            f"format_version {needs}."
        )


def resolve_dtype(name: str, dtype_map: Mapping[str, str]) -> str | None:
    """Longest-prefix match of a tensor name into the manifest's dtype map.

    Longest-prefix so that ``s3gen.flow.decoder.estimator`` (fp16) can override
    ``s3gen.flow`` (fp32), the estimator
    tolerates half precision at mel corr 0.999999 while the encoder under the
    same treatment collapses to 0.619.
    """
    best: str | None = None
    best_len = -1
    for prefix, dt in dtype_map.items():
        if (name == prefix or name.startswith(prefix + ".")) and len(prefix) > best_len:
            best, best_len = dt, len(prefix)
    return best


@dataclass(frozen=True)
class Checkpoint:
    """A packed checkpoint, opened lazily.

    See ``docs/design/runtime-notes.md``.
    """

    path: Path
    manifest: dict[str, object] = field(repr=False)

    @classmethod
    def open(cls, path: str | Path) -> Checkpoint:
        path = Path(path)
        return cls(path=path, manifest=read_manifest(path))

    @cached_property
    def file_digest(self) -> str:
        """SHA-256 of the checkpoint file exactly as it sits on disk.

        See ``docs/design/runtime-notes.md``.
        """
        return file_sha256(self.path)

    @property
    def dtype_map(self) -> Mapping[str, str]:
        dm = self.manifest.get("dtype_map") or {}
        if not isinstance(dm, Mapping):
            raise ValueError(
                f"{self.path}: manifest 'dtype_map' is a {type(dm).__name__}, "
                "expected a JSON object"
            )
        return cast(Mapping[str, str], dm)

    def keys(self) -> list[str]:
        """All tensor names in the file, sorted."""
        from safetensors import safe_open

        with safe_open(str(self.path), framework="numpy") as f:
            return sorted(f.keys())

    def shapes(self, prefix: str = "") -> dict[str, tuple[int, ...]]:
        """Tensor shapes under ``prefix``, read from the header only.

        See ``docs/design/runtime-notes.md``.
        """
        from safetensors import safe_open

        with safe_open(str(self.path), framework="numpy") as f:
            return {
                name[len(prefix) :]: tuple(f.get_slice(name).get_shape())
                for name in f.keys()  # noqa: SIM118 - see `tensors`
                if name.startswith(prefix)
            }

    def tensors(self, prefix: str) -> dict[str, NDArray[np.generic]]:
        """All tensors under ``prefix``, with the prefix stripped.

        Storage dtype is preserved (fp16 stays fp16); deciding the compute
        dtype is the backend's job, driven by ``ExecutionConfig.precision``.
        """
        from safetensors import safe_open

        out: dict[str, NDArray[np.generic]] = {}
        with safe_open(str(self.path), framework="numpy") as f:
            # `.keys()` is not redundant here: safetensors' handle exposes
            # keys() but is not itself iterable, so ruff's SIM118 rewrite of
            # this line raises `'builtins.safe_open' object is not iterable`.
            for name in f.keys():  # noqa: SIM118
                if name.startswith(prefix):
                    out[name[len(prefix) :]] = f.get_tensor(name)
        if not out:
            raise KeyError(f"{self.path.name}: no tensors under prefix {prefix!r}")
        return out

    def iter_names(self, prefix: str = "") -> Iterator[str]:
        for name in self.keys():
            if name.startswith(prefix):
                yield name

    def sibling(self, filename: str) -> Path:
        """A file distributed next to the checkpoint (e.g. the text tokenizer)."""
        return self.path.parent / filename

    def verified_sibling(self, filename: str, *, manifest_key: str) -> Path:
        """A sibling artefact, checked against the digest the manifest records.

        See ``docs/design/runtime-notes.md``.
        """
        path = self.sibling(filename)
        expected = self.manifest.get(manifest_key)
        if expected is None:
            return path
        if not isinstance(expected, str):
            raise ValueError(
                f"{self.path.name}: manifest[{manifest_key!r}] must be a hex digest, "
                f"got {type(expected).__name__}"
            )
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; the checkpoint's manifest records a "
                f"{manifest_key} for it, so it is part of this release"
            )
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"{path.name} does not belong to this checkpoint.\n"
                f"  manifest[{manifest_key!r}]: {expected}\n"
                f"  file on disk:               {actual}\n"
                "A different tokenizer or graph reads the same text as different "
                "tokens while the algorithm fingerprint stays identical, so nothing "
                "downstream would report the mismatch."
            )
        return path

    # -- packed assets ------------------------------------------------------

    def asset(self, name: str) -> bytes | None:
        """A text artefact carried *inside* the checkpoint, or ``None``.

        See ``docs/design/runtime-notes.md``.
        """
        from safetensors import safe_open

        key = f"{ASSET_PREFIX}{name}"
        with safe_open(str(self.path), framework="numpy") as f:
            if key not in f.keys():  # noqa: SIM118 - the handle is not iterable
                return None
            return cast(NDArray[np.uint8], f.get_tensor(key)).tobytes()

    def resolve_asset(self, filename: str, *, manifest_key: str) -> bytes:
        """The packed copy if there is one, else the verified sibling.

        One resolution order for every caller, so "where did this tokenizer come
        from" has a single answer. A packed checkpoint is self-contained; an
        older one still works and is still digest-checked when its manifest says
        what to expect.
        """
        packed = self.asset(filename)
        if packed is not None:
            return packed
        return self.verified_sibling(filename, manifest_key=manifest_key).read_bytes()

    def __repr__(self) -> str:
        name = self.manifest.get("name", self.path.stem)
        return f"Checkpoint({name!r}, {self.path.name})"


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
