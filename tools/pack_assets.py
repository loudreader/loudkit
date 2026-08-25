"""Carry the tokenizer and the respelling lexicon *inside* the checkpoint.

A loudkit release is one 1.2 GB weights file plus a 68 KB `tokenizer.json`
beside it, plus a 6.3 MB `pl_en_respell.json` compiled into each of five ports.
The weights are content-addressed and immutable; the two text files are neither.
That gap has produced, across two review rounds: a tokenizer bound to the
checkpoint only by a digest the shipping manifest does not actually carry, five
copies of the lexicon kept in step by discipline, and three ports whose speech
funnels had silently diverged.

Packing closes it. The artefacts go in as `uint8` tensors under `assets.`, so
they travel with the weights, are covered by the same file, and — the reason
this is not a new container format — **every port already has a safetensors
reader**. Nothing new to implement in five languages.

    python tools/pack_assets.py \\
        --checkpoint loudr-1.safetensors \\
        --out loudr-1-packed.safetensors

Writes a new file and refuses to overwrite the input: a checkpoint is what
every measurement in this repository is stated against, and rewriting one in
place changes what past results mean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# What travels with the weights, and where the canonical copy lives.
#
# `manifest_key` is the digest field the loader already checks for a *sibling*
# copy (`Checkpoint.verified_sibling`). Recording it for the packed copy too
# means an older runtime pointed at the sibling and a newer one reading the
# packed bytes are checking the same value.
ASSETS: dict[str, tuple[Path, str]] = {
    "tokenizer.json": (Path("tokenizer.json"), "tokenizer_sha256"),
    "pl_en_respell.json": (
        REPO / "python" / "loudkit" / "models" / "data" / "pl_en_respell.json",
        "pl_en_respell_sha256",
    ),
}


def _load(name: str, source: Path, checkpoint: Path) -> bytes:
    """Read one asset, resolving a relative source beside the checkpoint."""
    path = source if source.is_absolute() else checkpoint.parent / source
    if not path.is_file():
        raise FileNotFoundError(f"{name}: not found at {path}")
    return path.read_bytes()


def pack(checkpoint: Path, out: Path, *, only: set[str] | None = None) -> dict[str, str]:
    """Copy `checkpoint` to `out` with the assets embedded. Returns the digests."""
    from safetensors import safe_open
    from safetensors.numpy import save_file

    if out.resolve() == checkpoint.resolve():
        raise ValueError("refusing to overwrite the input checkpoint; pass a new --out")

    from loudkit.checkpoint import ASSET_PREFIX, read_manifest

    manifest = read_manifest(checkpoint)
    tensors: dict[str, np.ndarray] = {}
    with safe_open(str(checkpoint), framework="numpy") as f:
        metadata = dict(f.metadata() or {})
        for key in f.keys():  # noqa: SIM118 - the handle is not iterable
            if key.startswith(ASSET_PREFIX):
                continue  # replaced below, so re-packing is idempotent
            tensors[key] = f.get_tensor(key)

    digests: dict[str, str] = {}
    for name, (source, manifest_key) in ASSETS.items():
        if only is not None and name not in only:
            continue
        payload = _load(name, source, checkpoint)
        # uint8, not a metadata string: safetensors metadata is a string map
        # that a reader must parse in full before it can touch a single tensor,
        # and 6.3 MB of JSON in the header would be paid by every load.
        tensors[f"{ASSET_PREFIX}{name}"] = np.frombuffer(payload, dtype=np.uint8)
        digest = hashlib.sha256(payload).hexdigest()
        manifest[manifest_key] = digest
        digests[name] = digest

    manifest["packed_assets"] = sorted(digests)
    metadata["manifest"] = json.dumps(manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=metadata)
    return digests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(ASSETS),
        help="pack a subset (default: everything in ASSETS)",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    out = Path(args.out)
    try:
        digests = pack(checkpoint, out, only=set(args.only) if args.only else None)
    except (FileNotFoundError, ValueError) as exc:
        print(f"pack failed: {exc}", file=sys.stderr)
        return 1

    size = out.stat().st_size / (1 << 20)
    print(f"wrote {out} ({size:.1f} MiB)")
    for name, digest in sorted(digests.items()):
        print(f"  {name}  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
