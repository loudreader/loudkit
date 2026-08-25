"""Move the engine-borne algorithm values into the checkpoint manifest.

The window recipe (query 255 / prompt 238, silence-unit padding) and the EOS
floor (``max(10, 1.2 x text tokens)``) were, until this amendment, the only
production algorithm values that lived in *code* (``loudkit/backends/__init__``)
rather than in the checkpoint. ``docs/design/parity.md`` names that as a known gap:
the manifest is supposed to be the authority a future backend cannot re-guess
against, and two of the most defect-prone values (the window recipe *is* the
entire measured ANE-vs-torch mel deviation) were not in it.

This tool rewrites the checkpoint's embedded manifest — tensors untouched,
proven by re-hashing the payload against ``tensor_payload_sha256`` before and
after — adding:

  guidance / guidance_rate     "single_path" / 0.0 (EXP-016: the estimator is
                               guidance-distilled; stating it in the manifest
                               is what stops a loader from defaulting to CFG)
  window                       the static-window recipe
  eos_floor                    the len-prior gate
  chunking                     where the reader breathes, and the prefix carry
                               that keeps the pitch contour continuous across
                               a join
  tokenizer_sha256             the digest of the tokenizer.json shipped beside
                               these weights, so a swapped tokenizer is refused
                               at load instead of quietly reading the same text
                               as different tokens
  sampling_defaults.max_new_tokens  255 — coupled to the window: one static
                               window carries 255 tokens, so a longer free run
                               would be silently truncated by the renderer

Values already present and equal are left alone; present-and-different is an
error (a manifest is not a place to lose an argument silently). That includes
``recipe_version``: a checkpoint amended before the loudkit-1 bump will refuse
this tool, and the resolution is a deliberate decision about which recipe those
weights belong to — not a rerun with a bigger hammer.

Usage:
  .venv/bin/python tools/amend_manifest.py \
      --checkpoint /path/to/loudr-1.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

AMENDMENTS: dict[str, object] = {
    "guidance": "single_path",
    "guidance_rate": 0.0,
    # 0 -> 6) altered a hashed algorithm value and re-based the goldens, so the
    # contract version moved with it. A checkpoint amended with the old string
    # under one name, on a checkpoint whose whole purpose is to be the
    # authority on which recipe is in force.
    # loudkit-1 adds the postprocess block below. Those detectors *remove
    # tokens*, so the same weights under the same values now produce shorter
    # identity contract, and the version moves with it for the same reason it
    # moved for the prefix carry.
    "recipe_version": "loudkit-1",
    # The artifact detectors. Stated in the manifest rather than left to a
    # shipping default for the same reason the window and the joins are: a
    # backend that re-guesses where a chunk ended cuts somewhere else, and the
    # difference is a hallucinated word that either does or does not reach a
    # listener. Every constant's provenance is in docs/reference/postprocess.md.
    "postprocess": {
        "mode": "trim",
        "ceiling_speech_per_text_token": 4.0,
        "ceiling_slack_tokens": 40,
        "trailing_filler_threshold": 0.7,
        "trailing_silence_run_tokens": 12,
        "filler_min_eos_probability": 0.05,
        "filler_max_speech_after_run": 10,
        "desperation_speech_per_text_token": 4.5,
        "desperation_min_text_tokens": 10,
        "ended_tail_silence_run": 6,
        "ended_tail_blip_max": 2,
        "ended_tail_word_max": 10,
        "ended_tail_keep": 5,
        "echo_strong_eos_probability": 0.1,
        "echo_strong_max_tail": 30,
        "echo_strong_min_position_pct": 68,
        "echo_weak_eos_probability": 0.003,
        "echo_weak_max_tail": 16,
        "echo_weak_min_position_pct": 85,
    },
    "window": {
        "max_speech_tokens": 255,
        "static_length": 255,
        "pad_token_id": 4254,
        "static_prompt_tokens": 238,
    },
    "eos_floor": {
        "min_tokens_floor": 10,
        "min_tokens_text_ratio": 1.2,
    },
    # Where the reader breathes is an algorithm value, and the loader now reads
    # this block instead of silently building defaults for it. Stated here so a
    # future backend cannot re-guess the joins any more than it can re-guess
    # the window.
    "chunking": {
        "enabled": True,
        "max_tokens": 255,
        "prefix_tokens": 6,
        "split_on": [". ", "! ", "? ", "; ", ", "],
    },
}


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


def _file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _apply(manifest: dict, amendments: dict, bump: tuple[str, str] | None) -> bool:
    """Fold the amendments in, refusing any that would lose an argument.

    Returns whether anything moved. Present-and-equal is a no-op;
    present-and-different is an error, except for the one deliberate move
    --bump-recipe names.
    """
    changed = False
    for key, value in amendments.items():
        if key in manifest:
            if manifest[key] != value:
                if key == "recipe_version" and bump == (manifest[key], value):
                    # The one key whose difference is expected rather than a
                    # mistake: a checkpoint packed under an earlier recipe is
                    # not wrong about its own history, it is simply older than
                    # the values this tool writes. Moving it is a claim about
                    # which recipe these weights ship under, so --bump-recipe
                    # makes the operator name both ends. Naming the old one is
                    # the part that matters: it cannot be typed by somebody who
                    # has not looked at the manifest.
                    print(f"recipe_version: {manifest[key]} -> {value} (asked for)")
                    manifest[key] = value
                    changed = True
                    continue
                extra = (
                    "\n  the weights are unchanged; what moved is the recipe around "
                    f"them. If they ship as {value!r}, say so:\n"
                    f"    --bump-recipe {manifest[key]}:{value}"
                    if key == "recipe_version"
                    else ""
                )
                raise SystemExit(
                    f"manifest already carries {key}={manifest[key]!r}, refusing to "
                    f"overwrite with {value!r} — resolve deliberately, not by rerun" + extra
                )
            continue
        manifest[key] = value
        changed = True
    return changed


def _parse_bump(value: str | None) -> tuple[str, str] | None:
    """``OLD:NEW`` for --bump-recipe, or None.

    Both ends are required so the move cannot be made by somebody who has not
    read the manifest, and so a rerun against an already-moved checkpoint
    refuses instead of repeating.
    """
    if value is None:
        return None
    if value.count(":") != 1:
        raise SystemExit("--bump-recipe takes OLD:NEW, one colon")
    old, new = value.split(":")
    if not old or not new:
        raise SystemExit("--bump-recipe takes OLD:NEW, both ends non-empty")
    return (old, new)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--bump-recipe",
        metavar="OLD:NEW",
        default=None,
        help=(
            "move recipe_version from OLD to NEW. Both ends are named so the "
            "move cannot be made by somebody who has not read the manifest, "
            "and so a rerun on an already-moved checkpoint refuses rather than "
            "repeating"
        ),
    )
    args = ap.parse_args()
    bump = _parse_bump(args.bump_recipe)

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    path = Path(args.checkpoint)
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    manifest = json.loads(meta["manifest"])

    amendments = dict(AMENDMENTS)

    # The tokenizer is a separate file resolved by name from the checkpoint's
    # directory, and swapping it for another valid one changes the text ids and
    # therefore the speech — with the algorithm fingerprint unmoved, because a
    # tokenizer is not part of the algorithm config. Recording its digest here
    # is what lets `Checkpoint.verified_sibling` refuse the mismatch at load.
    # Computed rather than constant: it is a property of the file that shipped.
    tokenizer = path.parent / "tokenizer.json"
    if tokenizer.exists():
        amendments["tokenizer_sha256"] = _file_sha256(tokenizer)
    else:
        print(f"note: no tokenizer.json beside {path.name}; not recording its digest")

    changed = _apply(manifest, amendments, bump)

    sampling = manifest.setdefault("sampling_defaults", {})
    if "max_new_tokens" not in sampling:
        sampling["max_new_tokens"] = 255
        changed = True
    elif sampling["max_new_tokens"] != 255:
        raise SystemExit(
            f"sampling_defaults.max_new_tokens={sampling['max_new_tokens']!r} != 255"
        )

    if not changed:
        print("manifest already carries every amendment; nothing to do")
        return

    print("loading tensors ...")
    tensors = load_file(str(path))
    pay = payload_sha256(tensors)
    recorded = manifest.get("tensor_payload_sha256")
    if recorded and pay != recorded:
        raise SystemExit(
            f"payload hash mismatch BEFORE rewrite ({pay[:12]}… != {str(recorded)[:12]}…) "
            "— this file is not the checkpoint its manifest describes; not touching it"
        )

    tmp = path.with_suffix(".safetensors.amending")
    save_file(
        {k: tensors[k] for k in sorted(tensors)},
        str(tmp),
        metadata={"manifest": json.dumps(manifest, sort_keys=True)},
    )

    # verify the rewrite before replacing anything
    check = load_file(str(tmp))
    if payload_sha256(check) != pay:
        tmp.unlink()
        raise SystemExit("payload hash changed across rewrite — aborted, original intact")
    os.replace(tmp, path)

    sibling = path.parent / "manifest.json"
    with open(sibling, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, sort_keys=True, indent=1) + "\n")

    print(f"amended  {path}")
    print(f"manifest {sibling}")
    print(f"payload sha256 unchanged: {pay}")


if __name__ == "__main__":
    main()
