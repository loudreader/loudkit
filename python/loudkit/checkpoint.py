"""The packed checkpoint: one file, one manifest, no re-guessing.

A loudkit checkpoint is a single ``safetensors`` file whose tensors live in two
namespaces — ``t3.*`` for the token generator and ``s3gen.*`` for everything
downstream — with a JSON manifest embedded in the file's metadata and mirrored
in a ``manifest.json`` beside it.

The manifest is not documentation, it is authority. Values that are properties
of the weights — silence-token ids, Euler step count, vocabulary bounds, the
per-module dtype map — are read from it and nowhere else, because the worst
The divergence class this format exists to prevent is defaults re-guessed by a second
implementation (see ``AlgorithmConfig``'s module docstring).

Two facts about the tensor payload that loading code must know:

* precision is **mixed per module** and recorded in ``manifest["dtype_map"]``
  by longest-prefix match. The packed dtype is the *storage* dtype; a backend
  may upcast (fp16 -> fp32 is exact) but must consult its own
  ``ExecutionConfig.precision`` for the compute dtype.
* the vocoder's weight-norm reparametrisation is already folded — the packed
  weights are plain ``weight`` tensors, bit-exactly equal to what the
  parametrised forward would have computed.

This module is deliberately torch-free: it reads numpy arrays, so a future
runtime-only backend (ONNX, CoreML) can load the same file without dragging
torch in.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ASSET_PREFIX",
    "Checkpoint",
    "file_sha256",
    "read_manifest",
    "resolve_dtype",
]

ASSET_PREFIX = "assets."
"""Tensor-name prefix for text artefacts carried inside the checkpoint.

See :meth:`Checkpoint.asset`. Chosen so `tensors("t3.")` and `tensors("s3gen.")`
— which take everything under a prefix — cannot pick these up as weights.
"""

CHECKPOINT_FORMAT = "loudkit-checkpoint"
"""Value of ``manifest["format"]`` this loader understands."""

SUPPORTED_FORMAT_VERSIONS = (1,)


def read_manifest(path: str | Path) -> dict[str, object]:
    """Read the manifest embedded in a packed checkpoint.

    The embedded copy is authoritative — the sibling ``manifest.json`` is a
    convenience for humans and can drift if someone edits it, so it is never
    read here.

    Raises:
        ValueError: if the file carries no manifest or declares a format this
            build does not read. Failing loudly beats loading a checkpoint
            under wrong assumptions about what its numbers mean.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        meta: Mapping[str, str] | None = f.metadata()
    if not meta or "manifest" not in meta:
        raise ValueError(f"{path}: no embedded manifest — not a loudkit checkpoint")
    manifest = json.loads(meta["manifest"])
    # Raised, not asserted, and checked before the first `.get`: the manifest
    # is external data, `python -O` strips asserts, and a JSON list here would
    # otherwise surface as an AttributeError from inside a getter.
    if not isinstance(manifest, dict):
        raise ValueError(
            f"{path}: manifest is a {type(manifest).__name__}, expected a JSON object"
        )
    fmt = manifest.get("format")
    version = int(manifest.get("format_version", -1))
    if fmt != CHECKPOINT_FORMAT or version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"{path}: format {fmt!r} version {version}, "
            f"this build reads {CHECKPOINT_FORMAT!r} versions {SUPPORTED_FORMAT_VERSIONS}"
        )
    return manifest


def resolve_dtype(name: str, dtype_map: Mapping[str, str]) -> str | None:
    """Longest-prefix match of a tensor name into the manifest's dtype map.

    Longest-prefix so that ``s3gen.flow.decoder.estimator`` (fp16) can override
    ``s3gen.flow`` (fp32) — the estimator
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

    Tensors are pulled on demand rather than loaded wholesale because the two
    stages of the engine may live in different processes or devices, and
    enrollment (~40% of the payload) is not needed for synthesis at all.

    Example:
        >>> ckpt = Checkpoint.open("loudr-1.safetensors")
        >>> t3 = ckpt.tensors("t3.")            # generator weights, prefix stripped
        >>> ckpt.manifest["n_cfm_timesteps"]
        2
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

        This is the value a release's ``SHA256SUMS`` lists and the value
        provenance manifests carry as ``checkpoint_sha256`` — the digest that
        names *which artefact* rendered a waveform. It is not
        ``tensor_payload_sha256``, which lives inside the file and can only
        say the payload survived the download. Computed on first use and
        cached: one chunked read of the file, once per opened checkpoint.
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

        No tensor data is touched, so this is cheap on a 747 MB file and — the
        reason it exists — it is available *before* anything is allocated. A
        manifest declares the architecture and the architecture decides how much
        memory the model constructor asks the allocator for, so a manifest that
        nothing checks is a 20 kB file that can demand gigabytes. These shapes
        are the other half of the same checkpoint and cannot be inflated without
        inflating the file, which makes them the thing to check the manifest
        against.
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

        A checkpoint is not self-contained: the tokenizer, and on the graph
        backends the exported ONNX or CoreML packages, are separate files
        resolved by name from the checkpoint's directory. Nothing tied them to
        the weights. Swapping ``tokenizer.json`` for another valid one changes
        the text ids, the speech, and potentially where EOS lands — and
        ``AlgorithmConfig.fingerprint()`` does not move, because the tokenizer
        is not part of the algorithm config and ``TextFrontend`` carries no
        config for ``Engine._assert_one_algorithm`` to compare. The result is
        two different readings reporting the same identity, which is the one
        thing this library promises cannot happen.

        Enforced when the manifest records the digest and skipped when it does
        not, because packs predating the field are still loadable — a checkpoint
        that cannot state what it expects cannot have its expectation checked.
        `tools/build_release.py` records the digests it ships.
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

        The tokenizer and the Polish respelling lexicon are not weights, but
        they decide what the weights are asked to say: a different
        ``tokenizer.json`` reads the same text as different ids, and a different
        lexicon reads embedded English a different way. Shipped beside the file
        they are two more things to keep in step, and this project has now spent
        five copies of the
        lexicon, a sibling bound only by a digest the shipping manifest does not
        carry, and three ports that disagreed about the funnel.

        Carried as a ``uint8`` tensor under ``assets.`` rather than in a new
        container format, because **every port already has a safetensors
        reader**. Nothing new to write in five languages, and the bytes are
        covered by the same file the weights live in.

        Returns ``None`` when the checkpoint predates the convention, which is
        what :meth:`resolve_asset` falls back on.
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
    """Hex SHA-256 of a file, read in chunks (the graphs are hundreds of MB)."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
