"""Move the engine-borne algorithm values into the checkpoint manifest.

The window recipe (query 255 / prompt 238, silence-unit padding) and the EOS
floor (``max(10, 1.2 x text tokens)``) were, until this amendment, the only
production algorithm values that lived in *code* (``loudkit/backends/__init__``)
rather than in the checkpoint. The manifest is supposed to be the authority a
future backend cannot re-guess against, and two of the most defect-prone values
(the window recipe *is* the entire measured ANE-vs-torch mel deviation) were
not in it.

This tool rewrites the checkpoint's embedded manifest, tensors untouched,
proven by re-hashing the payload against ``tensor_payload_sha256`` before and
after, adding the ten values below:

  guidance / guidance_rate     "single_path" / 0.0 (EXP-016: the estimator is
                               guidance-distilled; stating it in the manifest
                               is what stops a loader from defaulting to CFG)
  window                       the static-window recipe
  eos_floor                    the len-prior gate
  chunking                     where the reader breathes, and the prefix carry
                               that keeps the pitch contour continuous across
                               a join
  edge_fade_seconds            the ramp on both edges of every rendered window
                               (docs/design/postprocess.md)
  postprocess                  the artifact detectors and their constants
                               (docs/design/postprocess.md)
  recipe_version               the recipe name these weights ship under
  tokenizer_sha256             the digest of the tokenizer.json shipped beside
                               these weights, so a swapped tokenizer is refused
                               at load instead of quietly reading the same text
                               as different tokens
  sampling_defaults.max_new_tokens  255, coupled to the window: one static
                               window carries 255 tokens, so a longer free run
                               would be silently truncated by the renderer

Values already present and equal are left alone; present-and-different is an
error (a manifest is not a place to lose an argument silently). That includes
``recipe_version``: a checkpoint amended before the loudkit-1 bump will refuse
this tool, and the resolution is a deliberate decision about which recipe those
weights belong to, not a rerun with a bigger hammer.

Usage:
  .venv/bin/python tools/amend_manifest.py \
      --checkpoint /path/to/loudr-1.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from split_checkpoint import payload_refusal

from loudkit.checkpoint import file_sha256, payload_sha256, read_manifest

AMENDMENTS: dict[str, object] = {
    "edge_fade_seconds": 0.02,
    "guidance": "single_path",
    "guidance_rate": 0.0,
    # The recipe name in force: loudkit-1 is the one that carries the
    # postprocess block below, and those detectors remove tokens, so the same
    # weights under the same values produce a shorter render than the recipe
    # before it. A checkpoint packed under an earlier name is moved only
    # through ``--bump-recipe OLD:NEW``, which ``_apply`` documents; nothing
    # here overwrites a name silently.
    "recipe_version": "loudkit-1",
    # The artifact detectors. Stated in the manifest rather than left to a
    # shipping default for the same reason the window and the joins are: a
    # backend that re-guesses where a chunk ended cuts somewhere else, and the
    # difference is a hallucinated word that either does or does not reach a
    # listener. Every constant's provenance is in docs/design/postprocess.md.
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
        # And where the reader does *not* breathe: a period that closes a
        # title, or that the funnel left behind when it folded an ellipsis, is
        # not a sentence end. Written out for the same reason as the rest of
        # the block, since a manifest that declares the joins and leaves this
        # implicit is a manifest a future backend can still re-guess.
        "abbreviations": [
            "A",
            "B",
            "Cpn",
            "D",
            "Dr",
            "F",
            "H",
            "Hr",
            "I",
            "J",
            "K",
            "M",
            "Mr",
            "Mrs",
            "R",
            "S",
            "St",
            "T",
            "V",
            "Vors",
            "dr",
            "mrs",
            "prof",
            "\u015bw",
        ],
        "mid_sentence_period": "hold",
    },
}


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


def _parse_only(value: str | None) -> list[str] | None:
    """``KEY[,KEY]`` for --only, or None for the whole list. Every name must be
    an amendment this tool knows, so a typo refuses instead of writing nothing."""
    if value is None:
        return None
    keys = [k.strip() for k in value.split(",") if k.strip()]
    unknown = [k for k in keys if k not in AMENDMENTS]
    if unknown:
        raise SystemExit(f"--only names no amendment: {', '.join(unknown)}")
    if not keys:
        raise SystemExit("--only takes at least one amendment name")
    return keys


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
    ap.add_argument(
        "--only",
        metavar="KEY[,KEY]",
        default=None,
        help=(
            "write only the named amendments and leave every other key as the "
            "manifest has it; no tokenizer digest is recorded either. 0.1.1 "
            "stamped edge_fade_seconds alone this way, because the full list "
            "also rewrites chunking over a shorter block the loader defaults "
            "identically, and the release digests depend on nothing else moving"
        ),
    )
    args = ap.parse_args()
    bump = _parse_bump(args.bump_recipe)
    only = _parse_only(args.only)

    from safetensors.torch import load_file, save_file

    path = Path(args.checkpoint)
    # The runtime's read, so a file with no metadata, a manifest that is not an
    # object, or a format this build does not know is one refusal sentence
    # rather than a TypeError traceback out of `meta["manifest"]`. This tool
    # rewrites the same file `split_checkpoint` does and owes the same door.
    try:
        manifest = dict(read_manifest(path))
    except Exception as exc:  # every failure to read it is the same refusal
        raise SystemExit(f"{path}: not a readable loudkit checkpoint: {exc}") from exc

    amendments = {k: AMENDMENTS[k] for k in only} if only is not None else dict(AMENDMENTS)

    # The tokenizer is a separate file resolved by name from the checkpoint's
    # directory, and swapping it for another valid one changes the text ids and
    # therefore the speech, with the algorithm fingerprint unmoved, because a
    # tokenizer is not part of the algorithm config. Recording its digest here
    # is what lets `Checkpoint.verified_sibling` refuse the mismatch at load.
    # Computed rather than constant: it is a property of the file that shipped.
    tokenizer = path.parent / "tokenizer.json"
    if only is not None:
        pass  # --only means only: the tokenizer digest is an amendment like the rest
    elif tokenizer.exists():
        amendments["tokenizer_sha256"] = file_sha256(tokenizer)
    else:
        print(f"note: no tokenizer.json beside {path.name}; not recording its digest")

    changed = _apply(manifest, amendments, bump)

    # Checked rather than assumed: the manifest is a file on disk, and a
    # `sampling_defaults` that is not an object would otherwise reach the
    # indexing below as a traceback instead of a sentence.
    sampling = manifest.setdefault("sampling_defaults", {})
    if not isinstance(sampling, dict):
        raise SystemExit(
            f"sampling_defaults is a {type(sampling).__name__}, expected a JSON object"
        )
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

    # The same check `split_checkpoint` runs before it writes, and for the same
    # reason: rewriting the manifest of a file whose payload already disagrees
    # stamps a fresh, confident record onto bytes nobody vouched for.
    pay = payload_sha256(path)
    problem = payload_refusal(manifest, pay)
    if problem is not None:
        raise SystemExit(f"{problem}; not touching it")

    print("loading tensors ...")
    tensors = load_file(str(path))
    tmp = path.with_suffix(".safetensors.amending")
    save_file(
        {k: tensors[k] for k in sorted(tensors)},
        str(tmp),
        metadata={"manifest": json.dumps(manifest, sort_keys=True)},
    )

    # verify the rewrite before replacing anything
    if payload_sha256(tmp) != pay:
        tmp.unlink()
        raise SystemExit("payload hash changed across rewrite — aborted, original intact")
    os.replace(tmp, path)

    # `newline="\n"`: a bundle member, and the bundle's digests are pinned.
    sibling = path.parent / "manifest.json"
    with open(sibling, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(manifest, sort_keys=True, indent=1) + "\n")

    print(f"amended  {path}")
    print(f"manifest {sibling}")
    print(f"payload sha256 unchanged: {pay}")


if __name__ == "__main__":
    main()
