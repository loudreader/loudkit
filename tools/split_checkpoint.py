"""Split the packed checkpoint into the two artefacts a release ships.

Until now one file carried everything: the T3 estimator, the flow, the
vocoder, *and* the two enrollment towers (the S3 speech tokenizer and the
speaker encoder). A caller who only ever loads a shipped voice downloads the
523 MB of enrollment weights they will never open, and every port that reads
the graph exports carries them too.

So the release ships two files, and this tool is what makes them:

  loudr-1.safetensors             t3, s3gen.flow, s3gen.mel2wav
                                  full torch synthesis, plus what the graph
                                  ports read. artifact_role "synthesis".
  loudr-1-enrollment.safetensors  s3gen.tokenizer, s3gen.speaker_encoder
                                  what enrolling a new voice from audio
                                  needs, and nothing else. artifact_role
                                  "enrollment".

There is no third variant. An ONNX/CoreML-only checkpoint would save the graph
ports about 390 MB and cost another artefact, another resolver branch and
another test matrix.

**The split is proved, not assumed.** Three properties, each refused by name:

*Disjoint*: no tensor lands in both outputs. A tensor matching two routing
rules is a rule collision, and the resolution is a decision about where those
weights belong, not a rerun.

*Complete*: every tensor lands in one output. A tensor matching no rule is a
group this tool has never seen, and dropping it silently is how a checkpoint
loses a tower.

*Unchanged*: the bytes are copied, not re-encoded. Each source tensor gets a
digest over (name, dtype, shape, raw bytes); every output is re-read from disk
after writing and each tensor re-hashed against the source's digest, and each
side's payload digest is checked against the payload hash of the source
tensors routed to it. The source's own ``tensor_payload_sha256`` is verified
before anything is written, so a file that is not the checkpoint its manifest
describes is refused rather than split.

Both manifests keep what the source manifest carried and still applies
(``dtype_map`` and ``dtype_rationale`` are filtered to the groups actually
present, since those keys *are* tensor groups) and gain:

  artifact_role           "synthesis" or "enrollment"
  tensor_payload_sha256   recomputed over this file's own tensors, so the
                          existing payload check keeps meaning what it meant
  tensor_count            how many tensors this file holds
  split                   what the pair is: the source's payload digest, a
                          digest over the source's full sorted tensor-name
                          list, the source tensor count, and the two canonical
                          filenames by role

That ``split`` block is what lets a consumer prove the pair from headers
alone: hash the sorted union of both files' tensor names and compare, with no
access to the packed original. ``tools/build_release.py`` does exactly that
before it copies a byte.

Usage:
  .venv/bin/python tools/split_checkpoint.py \
      --checkpoint release-dir/loudr-1.safetensors \
      --out-dir dist/split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - annotations only
    import torch

# The two names the release ships, and the role each carries in its manifest.
# `build_release` and `hub` name the same two constants; a file under any other
# name is a different artefact wearing the same directory.
SYNTHESIS_NAME = "loudr-1.safetensors"
ENROLLMENT_NAME = "loudr-1-enrollment.safetensors"
SYNTHESIS_ROLE = "synthesis"
ENROLLMENT_ROLE = "enrollment"

# Routing, by tensor group. A group is matched exactly or as a dotted prefix,
# so `t3` catches `t3.tfmr.*` and would catch a bare `t3` tensor, and nothing
# else. Written as data rather than as an if-chain because the disjointness
# check below is a property *of this table*: it asks how many rules each tensor
# matched, and both 0 and 2 are refusals.
ROUTING: dict[str, tuple[str, ...]] = {
    SYNTHESIS_ROLE: ("t3", "s3gen.flow", "s3gen.mel2wav"),
    ENROLLMENT_ROLE: ("s3gen.tokenizer", "s3gen.speaker_encoder"),
}

ROLE_FILENAMES = {SYNTHESIS_ROLE: SYNTHESIS_NAME, ENROLLMENT_ROLE: ENROLLMENT_NAME}

# Manifest keys that describe the packed whole and do not survive the split.
# `tensor_payload_sha256` is not dropped but replaced: each side recomputes it
# over its own tensors, so a loader's existing payload check keeps working and
# keeps meaning "these are the bytes this manifest describes".
REPLACED_KEYS = frozenset({"tensor_payload_sha256"})

# Manifest keys whose *values* are keyed by tensor group, so a half-checkpoint
# carries only the entries for the groups it actually holds. Everything else is
# carried across untouched: a recipe value, a window, a licence chain describes
# the weights wherever they live.
GROUP_KEYED = ("dtype_map", "dtype_rationale")


def matches(name: str, group: str) -> bool:
    """Whether a tensor belongs to a group: exactly, or under its dot."""
    return name == group or name.startswith(group + ".")


_HEX64 = re.compile(r"[0-9a-f]{64}")
"""A sha256, lowercase hex. What a recorded digest has to look like."""


def payload_sha256(tensors: dict[str, torch.Tensor]) -> str:
    """Identical recipe to tools/pack_checkpoint.py in the research repo:
    sha256 over (name, dtype, shape, raw bytes) in sorted key order."""
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    """One tensor's digest, over the same four things the payload hash uses.

    The payload hash proves a *set* of tensors is unchanged; this proves each
    tensor individually, which is what turns "the halves add up" into "this
    tensor is the tensor it was" and lets a mismatch name the tensor.
    """
    t = tensor.contiguous()
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def names_sha256(names: list[str]) -> str:
    """A digest over a sorted tensor-name list, newline-joined.

    This is the completeness witness. Both outputs carry it, so the pair can
    be proved complete from the two headers alone: sort the union of their
    tensor names, hash, compare. No access to the packed original required,
    which matters because by then the original is not what anybody has.
    """
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def route(names: list[str]) -> dict[str, list[str]]:
    """Tensor names by role, refusing anything not routed exactly once.

    Raises:
        SystemExit: naming every tensor that matched no rule (the split would
            silently drop it) or more than one (the rules disagree about where
            it belongs). Both are decisions for a person, not a rerun.
    """
    by_role: dict[str, list[str]] = {role: [] for role in ROUTING}
    unrouted: list[str] = []
    contested: list[tuple[str, list[str]]] = []
    for name in names:
        hits = [
            role for role, groups in ROUTING.items() if any(matches(name, g) for g in groups)
        ]
        if not hits:
            unrouted.append(name)
        elif len(hits) > 1:
            contested.append((name, hits))
        else:
            by_role[hits[0]].append(name)

    problems: list[str] = []
    if unrouted:
        problems.append(
            f"{len(unrouted)} tensor(s) match no rule, so the split would not be "
            "complete. Every tensor must land somewhere:\n    "
            + "\n    ".join(unrouted[:20])
            + ("\n    ..." if len(unrouted) > 20 else "")
        )
    if contested:
        problems.append(
            f"{len(contested)} tensor(s) match more than one rule, so the split "
            "would not be disjoint:\n    "
            + "\n    ".join(f"{n} -> {', '.join(r)}" for n, r in contested[:20])
            + ("\n    ..." if len(contested) > 20 else "")
        )
    empty = sorted(role for role, got in by_role.items() if not got)
    if empty:
        problems.append(
            f"nothing routed to {', '.join(empty)}: this is not the checkpoint "
            "this tool splits, and an empty half is not an artefact"
        )
    if problems:
        raise SystemExit("REFUSING to split:\n  " + "\n  ".join(problems))
    return by_role


def side_manifest(
    source: dict[str, object],
    *,
    role: str,
    names: list[str],
    payload: str,
    split_block: dict[str, object],
) -> dict[str, object]:
    """The manifest one half ships: everything that still applies, plus the split."""
    manifest = {k: v for k, v in source.items() if k not in REPLACED_KEYS}
    for key in GROUP_KEYED:
        value = manifest.get(key)
        if isinstance(value, dict):
            kept = {g: v for g, v in value.items() if any(matches(n, g) for n in names)}
            manifest[key] = kept
    manifest["artifact_role"] = role
    manifest["tensor_payload_sha256"] = payload
    manifest["tensor_count"] = len(names)
    manifest["split"] = split_block
    return manifest


def _write_side(
    target: Path,
    *,
    role: str,
    tensors: dict[str, torch.Tensor],
    digests: dict[str, str],
    source_manifest: dict[str, object],
    split_block: dict[str, object],
) -> dict[str, object]:
    """Write one half and prove it, or leave nothing at ``target``.

    Written to a ``.splitting`` sibling and renamed only after the file on
    disk has been re-read and every tensor in it re-hashed against the
    source's digest. Re-reading rather than trusting the dict just handed to
    ``save_file`` is the point: what ships is the file, and everything between
    the tensors in memory and the bytes on disk is this tool's responsibility.
    """
    from safetensors.torch import load_file, save_file

    names = sorted(tensors)
    payload = payload_sha256(tensors)
    manifest = side_manifest(
        source_manifest, role=role, names=names, payload=payload, split_block=split_block
    )
    tmp = target.with_suffix(".safetensors.splitting")
    save_file(
        {k: tensors[k] for k in names},
        str(tmp),
        metadata={"manifest": json.dumps(manifest, sort_keys=True)},
    )

    check = load_file(str(tmp))
    if sorted(check) != names:
        tmp.unlink()
        raise SystemExit(f"{target.name}: the tensors written are not the tensors routed")
    wrong = [n for n in names if tensor_sha256(n, check[n]) != digests[n]]
    if wrong:
        tmp.unlink()
        raise SystemExit(
            f"{target.name}: {len(wrong)} tensor(s) changed across the copy:\n    "
            + "\n    ".join(wrong[:20])
        )
    if payload_sha256(check) != payload:
        tmp.unlink()
        raise SystemExit(f"{target.name}: payload digest changed across the copy")
    tmp.replace(target)

    size = target.stat().st_size
    print(f"\n{target.name}")
    print(f"  role     {role}")
    print(f"  tensors  {len(names)}")
    print(f"  bytes    {size} ({size / 1e6:.1f} MB)")
    print(f"  payload  {payload}")
    print(f"  file     {file_sha256(target)}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help=(
            "where the two files are written. Must not be the checkpoint's own "
            "directory: the packed original is not overwritten by this tool"
        ),
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite outputs that are already there",
    )
    args = ap.parse_args()

    from safetensors import safe_open
    from safetensors.torch import load_file

    path = args.checkpoint.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir == path.parent:
        raise SystemExit(
            f"--out-dir is the checkpoint's own directory ({out_dir}); the packed "
            "original and its manifest.json stay as they are. Write the split "
            "somewhere new"
        )
    outputs = {role: out_dir / name for role, name in ROLE_FILENAMES.items()}
    existing = sorted(p.name for p in outputs.values() if p.exists())
    if existing and not args.force:
        raise SystemExit(f"already there: {', '.join(existing)}. Pass --force to overwrite")

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if not meta or "manifest" not in meta:
        raise SystemExit(f"{path} carries no embedded manifest; not splitting it")
    source_manifest = json.loads(meta["manifest"])
    if source_manifest.get("artifact_role") is not None:
        raise SystemExit(
            f"{path.name} already carries artifact_role="
            f"{source_manifest['artifact_role']!r}. This is a half, not the packed "
            "checkpoint, and splitting a half again is not a thing"
        )

    print(f"loading {path} ...")
    tensors = load_file(str(path))
    source_names = sorted(tensors)

    # Refuse before writing anything if this file is not what its manifest
    # describes: splitting a checkpoint whose payload already disagrees would
    # produce two halves that faithfully carry the wrong bytes.
    source_payload = payload_sha256(tensors)
    recorded = source_manifest.get("tensor_payload_sha256")
    # Absent, wrong type or wrong shape is a refusal, not a pass. `if recorded`
    # let a checkpoint with no recorded digest through, which is the one case
    # where nothing at all was being checked -- the tool would then stamp both
    # halves with a source digest it had never verified.
    if not isinstance(recorded, str) or not _HEX64.fullmatch(recorded):
        raise SystemExit(
            f"the manifest records tensor_payload_sha256={recorded!r}, which is not "
            "a sha256. Nothing here can vouch for these bytes, and both halves would "
            "carry a source digest nobody checked; not splitting it"
        )
    if source_payload != recorded:
        raise SystemExit(
            f"payload hash mismatch BEFORE the split ({source_payload[:12]}… != "
            f"{recorded[:12]}…). This file is not the checkpoint its manifest "
            "describes; not splitting it"
        )

    by_role = route(source_names)
    covered = sorted(n for names in by_role.values() for n in names)
    # The routing loop cannot produce a duplicate or a gap, so these are checks
    # of the loop rather than of the checkpoint. They cost one comparison and
    # they are the difference between proving the split and trusting it.
    if covered != source_names:
        raise SystemExit("internal: the routed tensors are not the source tensors")

    split_block: dict[str, object] = {
        "source_payload_sha256": source_payload,
        "source_tensor_names_sha256": names_sha256(source_names),
        "source_tensor_count": len(source_names),
        "roles": dict(ROLE_FILENAMES),
    }

    digests = {name: tensor_sha256(name, tensors[name]) for name in source_names}

    # Created only now: a run that refuses leaves no directory behind that an
    # operator has to wonder about.
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, object]] = {}
    for role, names in by_role.items():
        written[role] = _write_side(
            outputs[role],
            role=role,
            tensors={name: tensors[name] for name in names},
            digests=digests,
            source_manifest=source_manifest,
            split_block=split_block,
        )

    # The sibling `manifest.json` a release ships, rewritten for the half it
    # now ships beside. Copying the packed original's would put a
    # `tensor_payload_sha256` in the bundle that describes 2574 tensors next to
    # a file holding 1533 of them, and a loader that checks the payload against
    # the sibling would refuse a correct release. The enrollment manifest is
    # not written here: it travels inside its own file, which is where a
    # consumer of that file already is.
    sibling = out_dir / "manifest.json"
    sibling.write_text(
        json.dumps(written[SYNTHESIS_ROLE], sort_keys=True, indent=1) + "\n", encoding="utf-8"
    )
    print(f"\nmanifest {sibling} (the synthesis half's own)")

    print(f"source payload sha256 {source_payload}")
    print(f"source tensors        {len(source_names)}")
    print("disjoint and complete: every tensor landed in exactly one output")
    print(f"written to {out_dir}")
    print(f"the packed original is untouched at {path}")


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks: the digest SHA256SUMS carries."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    main()
