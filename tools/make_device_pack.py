#!/usr/bin/env python3
"""Stage the on-device asset pack for an iOS app bundle.

``--out`` is required and takes the app's asset directory: the demo app this
was written for is not tracked here (``/Examples/`` is gitignored), so a
default pointing at it staged a 356 MB pack into a directory that exists on
one machine.

The full checkpoint is 1.27 GB; a phone doing *synthesis* needs 356 MB of it:

  t3.*                                  355.7 MB  the token generator (fp16)
  s3gen.flow.spk_embed_affine_layer.*     0.1 MB  the 192->80 speaker projection
                                                  (lives in neither CoreML graph)

Left out, and why that is safe:

  s3gen.tokenizer        495 MB  S3 speech tokenizer — enrollment only; voices
                                 ship as ~165 KB profiles enrolled on a Mac
  s3gen.speaker_encoder   28 MB  enrollment only, same reason
  s3gen.flow (rest)      307 MB  baked into flow_encoder/flow_estimator.mlmodelc
  s3gen.mel2wav           83 MB  baked into vocoder.mlmodelc

The renderer weights travel as the *precompiled* .mlmodelc directories, so the
phone never pays the on-device CoreML compile (or the duplicate .mlpackage
bytes). The embedded manifest is copied verbatim — the algorithm fingerprint
must stay identical to the full pack — plus a `subset` metadata key naming
what was dropped, so a subset pack can never masquerade as the full one.

Usage:
    python3 tools/make_device_pack.py \
        --checkpoint \
            ~/Developer/chatterbox-apple/checkpoints/loudr-1/\
            loudr-1.safetensors \
        --out Examples/LoudKitDemo/Assets
"""

import argparse
import json
import shutil
import struct
from pathlib import Path

KEEP_PREFIXES = ("t3.", "s3gen.flow.spk_embed_affine_layer.")

REPO = Path(__file__).resolve().parent.parent


def read_header(path: Path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def write_subset(src: Path, dst: Path) -> None:
    header, payload_base = read_header(src)
    metadata = header.pop("__metadata__", {})
    keep = {k: v for k, v in header.items() if k.startswith(KEEP_PREFIXES)}
    if not keep:
        raise SystemExit(f"{src}: no tensors match {KEEP_PREFIXES}")

    metadata = dict(metadata)
    metadata["subset"] = (
        "synthesis-only: t3.* + s3gen.flow.spk_embed_affine_layer.*; "
        "tokenizer/speaker_encoder (enrollment) and flow/mel2wav torch weights "
        "(shipped as CoreML graphs) removed by tools/make_device_pack.py"
    )

    # repack contiguously, ordered by original offset
    ordered = sorted(keep.items(), key=lambda kv: kv[1]["data_offsets"][0])
    new_header = {"__metadata__": metadata}
    cursor = 0
    for name, info in ordered:
        b, e = info["data_offsets"]
        size = e - b
        new_header[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size

    header_bytes = json.dumps(new_header, separators=(",", ":")).encode()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(struct.pack("<Q", len(header_bytes)))
        fout.write(header_bytes)
        for _name, info in ordered:
            b, e = info["data_offsets"]
            fin.seek(payload_base + b)
            remaining = e - b
            while remaining:
                chunk = fin.read(min(remaining, 1 << 24))
                fout.write(chunk)
                remaining -= len(chunk)
    print(f"  {dst.name}: {len(ordered)} tensors, {cursor / 1e6:.1f} MB payload")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="the app's asset directory")
    args = ap.parse_args()

    src_dir = args.checkpoint.parent
    out = args.out
    ckpt_dir = out / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("device pack ->", out)
    write_subset(args.checkpoint, ckpt_dir / "loudr-1.safetensors")
    shutil.copy2(src_dir / "tokenizer.json", ckpt_dir / "tokenizer.json")

    coreml = ckpt_dir / "coreml"
    coreml.mkdir(exist_ok=True)
    for stage in ("flow_encoder", "flow_estimator", "vocoder"):
        src = src_dir / "coreml" / f"{stage}.mlmodelc"
        if not src.exists():
            raise SystemExit(f"missing {src} — compile the .mlpackage first")
        copy_tree(src, coreml / f"{stage}.mlmodelc")
        print(f"  coreml/{stage}.mlmodelc")

    # conformance fixture + the reference voice it names, flattened so the
    # app resolves the voice by basename inside one bundled directory
    conf_src = REPO / "tests/data/conformance"
    conf_dst = out / "conformance"
    copy_tree(conf_src, conf_dst)
    shutil.copy2(
        REPO / "tests/data/reference/testvoice.voice.safetensors",
        conf_dst / "testvoice.voice.safetensors",
    )
    print("  conformance fixture + reference voice")

    voices = out / "voices"
    voices.mkdir(exist_ok=True)
    shutil.copy2(
        REPO / "tests/data/reference/testvoice.voice.safetensors",
        voices / "testvoice.safetensors",
    )
    print("  voices/testvoice.safetensors")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"total pack: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
