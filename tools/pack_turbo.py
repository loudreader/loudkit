"""Pack a turbo (two-head fusion) research checkpoint into a loudkit checkpoint.

`loudr-1` is a one-token-per-forward model: the packed file carries `t3.*` and
`s3gen.*`, and a reader that finds those weights knows how to run them. Turbo
emits **two speech tokens per transformer forward**, a second head predicts the
pair's other token, and the KV slot for the pair carries a learned fusion of both
embeddings instead of the second token's embedding alone. Those are extra weights
*and* a different decode loop, so a reader that loaded turbo under `loudr-1`
assumptions would produce fluent-sounding nonsense rather than fail.

That is why this writes **format_version 2**: a reader that does not implement
the loop refuses the file, which is the format working, not a bug to route around.

What version 2 adds, and nothing else:

    t3.head2.weight        [speech_vocab, 2 * hidden]   pair's second token from
                                                        [h ; emb(first token)]
    t3.fuse.0.{weight,bias}  [hidden, 2 * hidden]       slot = mean(ea, eb)
    t3.fuse.2.{weight,bias}  [hidden, hidden]                  + fuse([ea ; eb])
    manifest["decode"]     the loop contract, spelled out for the runtime

Policy (the silence lists, the sampling defaults, the dtype map) is inherited
from a template manifest, normally `loudr-1`'s. The postprocess preset is
written out as turbo's own block, with loudr-1's values and
`"calibrated_on": "loudr-1"`, until it is measured on turbo.
`--show-inherited` prints the split.

Runs where the research checkpoints live (the GPU box), because it needs torch
and the chatterbox modules to fold weight-norm parametrisations the way the
format requires.

    python tools/pack_turbo.py \
        --t3 /path/to/MIMI_onpolicy_r1/best.pt \
        --s3gen ~/chatterbox/models/s3gen.safetensors \
        --estimator ~/chatterbox/distill/ckpt_s3gen_stepFull/best.pt \
        --n-cfm-timesteps 2 \
        --template-manifest loudr-1-manifest.json \
        --out dist/loudr-1-turbo.safetensors

Or, preferred, from a build recipe that names the same inputs by hash:

    python tools/pack_turbo.py --recipe tools/recipes/loudr-1-turbo.json --root ~/chatterbox

A recipe is the answer to "which weights is this build made of", written down
once instead of retyped per invocation. Every input carries its sha256; a file
that no longer hashes to what the recipe says stops the build, because the
alternative is a checkpoint whose provenance is a shell command in somebody's
scrollback. On a first build, leave the hashes empty and the script prints the
ones it measured, to be pasted in and reviewed like any other value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from distill_meta import estimator_k
from split_checkpoint import payload_sha256 as tensor_payload_sha256

from loudkit.checkpoint import file_sha256, payload_sha256, resolve_dtype

# Inherited from the template because each was measured, not chosen: widening or
# narrowing any of them is a separate experiment with its own evidence.
INHERITED_KEYS = (
    "chunking",
    "dtype_map",
    "dtype_rationale",
    "eos_floor",
    "quiet_render_ids",
    "quiet_render_ids_source",
    "runtime_requirements",
    "sample_rate",
    "sampling_defaults",
    "silence_render_ids",
    "silence_render_ids_source",
    "silence_token_ids",
    "silence_token_ids_source",
    "silence_token_ids_alternate_rejected",
    "speech_tokens",
    "t3_config",
    "t3_llama_config_registered_as",
    "tokenizer_sha256",
    "window",
)


def fold_weight_norm(module) -> int:
    """Materialise weight-norm parametrisations, as `weight_norm.method` records.

    A parametrised module stores `weight_g`/`weight_v` and recomputes `weight` on
    every forward; the packed file carries the product instead, so a runtime that
    knows nothing about parametrisation loads plain weights.
    """
    import torch.nn.utils.parametrize as P

    folded = 0
    for sub in module.modules():
        if P.is_parametrized(sub, "weight"):
            P.remove_parametrizations(sub, "weight", leave_parametrized=True)
            folded += 1
    return folded


_KEYS_IN_REFUSAL = 5
"""How many key names a refusal prints before it counts the rest."""


def refuse_partial_load(source: Path, missing: list[str], unexpected: list[str]) -> None:
    """Refuse a renderer load that did not place every tensor.

    A missing key is the dangerous half: `load_state_dict` leaves that module's
    random initialisation in place and this tool packs it as weights. Nothing
    downstream can see it. The release gate checks determinism, not
    correctness, and the conformance fixture is generated from the same packed
    file, so the five ports would agree on the wrong weights and the only
    remaining defence is listening. An unexpected key is refused for the
    matching reason: a source carrying names this renderer does not have is
    not the renderer the build thinks it is packing.

    The two checkpoints that were built were compared tensor by tensor against
    their sources and nothing was random, so this is about what the tool
    allows, not about what shipped.
    """
    if not missing and not unexpected:
        return
    parts = []
    for label, keys in (("missing", missing), ("unexpected", unexpected)):
        if not keys:
            continue
        shown = ", ".join(sorted(keys)[:_KEYS_IN_REFUSAL])
        rest = len(keys) - _KEYS_IN_REFUSAL
        if rest > 0:
            shown += f", +{rest} more"
        parts.append(f"{len(keys)} {label} ({shown})")
    raise SystemExit(
        f"{source} is not the state dict this renderer expects: "
        + "; ".join(parts)
        + ". A missing tensor would pack as the module's random initialisation, "
        "and nothing downstream reads weights for correctness, so pack the "
        "s3gen file these weights were trained with rather than the nearest one."
    )


def build_t3_tensors(ckpt: dict) -> tuple[dict, dict]:
    """`t3.*` tensors plus the architecture facts the manifest must record."""
    student = ckpt["student_sd"]
    tensors = {f"t3.{k}": v for k, v in student.items()}

    head2 = ckpt.get("head2_sd")
    fuse = ckpt.get("fuse_sd")
    if head2 is None or fuse is None:
        raise SystemExit(
            "checkpoint carries no head2_sd/fuse_sd — this is a one-head model, "
            "pack it as format_version 1 with the loudr-1 packer instead"
        )
    for k, v in head2.items():
        tensors[f"t3.head2.{k}"] = v
    for k, v in fuse.items():
        tensors[f"t3.fuse.{k}"] = v

    hidden = student["speech_head.weight"].shape[1]
    vocab = student["speech_head.weight"].shape[0]
    h2_out, h2_in = tensors["t3.head2.weight"].shape
    if h2_in != 2 * hidden or h2_out != vocab:
        raise SystemExit(
            f"head2 is {h2_out}x{h2_in}, expected {vocab}x{2 * hidden} — "
            "the second head reads [hidden ; embedding] and scores the same vocab"
        )
    facts = {
        "llama_config": ckpt["llama_config"],
        "speech_vocab_size": int(vocab),
        "hidden_size": int(hidden),
    }
    if ckpt.get("keep_layers") is not None:
        facts["t3_keep_layers"] = list(ckpt["keep_layers"])
    return tensors, facts


def build_s3gen_tensors(
    s3gen_path: Path, estimator_path: Path | None, torch, n_cfm: int | None
) -> tuple[dict, int, int]:
    """`s3gen.*` tensors: stock renderer, optional distilled estimator, folded.

    Returns the tensors, the fold count, and **the Euler step count this build
    must declare**, the estimator's own where it records one, the caller's
    `--n-cfm-timesteps` otherwise. The two must agree where both exist, and the
    resolved value is handed back rather than recomputed, so the manifest
    cannot end up declaring a different number from the one checked here.

    The check is the point: a step-distilled estimator run at any other K is a
    different algorithm, and this is the last place where the two are still
    separable, once the estimator is inside the packed checkpoint, every later
    guard reads K from the manifest this build wrote and finds it consistent
    with itself.

    `--n-cfm-timesteps` has no default, because a default is a guess: it packs
    a K=1 estimator whose file records nothing into a checkpoint declaring some
    other K, which then passes everything downstream including the CoreML
    exporter's own guard. An unrecorded K has to be stated, the way
    `export_coreml.py` already requires.
    """
    # The estimator is read and checked before anything heavy is imported or
    # loaded: a K disagreement is the build's fault, not the model's, and
    # discovering it after 1.3 GB of weights are in memory helps nobody.
    if n_cfm is not None and n_cfm < 1:
        # The manifest carries this number and `AlgorithmConfig` refuses
        # `euler_steps < 1` at load, so a build that accepted 0 or a negative
        # wrote a checkpoint the runtime would rightly refuse, a failure moved
        # from the build, where it can be fixed, to whoever downloads it.
        raise SystemExit(
            f"--n-cfm-timesteps must be at least 1: {n_cfm}. It is the number of "
            f"Euler steps the renderer runs, and the runtime refuses a manifest "
            f"declaring fewer than one."
        )
    blob = None
    declared: int | None = None
    if estimator_path is None and n_cfm is None:
        # No estimator to read the number off, and nobody stated it. The
        # manifest has to carry one, so there is nothing to write.
        raise SystemExit(
            "--n-cfm-timesteps is required when no --estimator is packed: the "
            "manifest declares the Euler step count and there is no file here "
            "to read it from."
        )
    if estimator_path is not None:
        blob = torch.load(str(estimator_path), map_location="cpu", weights_only=False)
        try:
            declared = estimator_k(blob)
        except ValueError as exc:
            raise SystemExit(f"{estimator_path}: {exc}") from None
        if declared is None and n_cfm is None:
            # The hole this whole function exists to close, and it stayed open
            # while `--n-cfm-timesteps` had a default: a metadata-less K=1
            # estimator packed without saying so produced a manifest declaring
            # 2, and every later guard, including the CoreML exporter's,
            # reads K from that manifest and finds it consistent with itself.
            # `export_coreml.py` refuses the same case for the same reason; a
            # default here was the one place left that would guess.
            raise SystemExit(
                f"{estimator_path} does not record the Euler step count it was "
                f"distilled for. Pass --n-cfm-timesteps to state it: this "
                f"number goes into the manifest, and everything downstream "
                f"believes the manifest, so guessing it here is a checkpoint "
                f"that lies about its own algorithm."
            )
        if declared is not None and n_cfm is not None and declared != n_cfm:
            raise SystemExit(
                f"{estimator_path} was distilled for K={declared}; this build "
                f"declares n_cfm_timesteps={n_cfm}. The number ships beside the "
                f"weights because running the estimator at another K is a "
                f"different algorithm — pass --n-cfm-timesteps {declared}, or "
                f"pack the estimator that belongs at {n_cfm}."
            )
        print(
            f"estimator: K={declared if declared is not None else n_cfm}"
            + ("" if declared is not None else " (declared, not recorded in the file)")
        )

    from chatterbox.models.s3gen.s3gen import S3Token2Wav
    from safetensors.torch import load_file

    model = S3Token2Wav()
    # `strict=False` so the two key lists can be read and named in the refusal
    # instead of arriving as one `RuntimeError` string. Nothing is tolerated:
    # see `refuse_partial_load`.
    incompatible = model.load_state_dict(load_file(str(s3gen_path)), strict=False)
    refuse_partial_load(
        s3gen_path, list(incompatible.missing_keys), list(incompatible.unexpected_keys)
    )
    if blob is not None:
        # `ema_sd` is the shipped half of a step-distill run; `student_sd` is the
        # live one and lags it. Prefer EMA, fall back only if the run predates it.
        state = blob.get("ema_sd") or blob["student_sd"]
        model.flow.decoder.estimator.load_state_dict(state)
    folded = fold_weight_norm(model)
    tensors = {f"s3gen.{k}": v for k, v in model.state_dict().items()}
    resolved = declared if declared is not None else n_cfm
    assert resolved is not None, "the refusals above leave no third case"
    return tensors, folded, resolved


def turbo_postprocess_block() -> dict:
    """Turbo's own postprocess preset: loudr-1's values written out in full and
    labelled, so the manifest says what it runs and the owner can measure and
    replace them without a code change."""
    from dataclasses import fields

    from loudkit.postprocess import PostprocessConfig

    default = PostprocessConfig()
    block = {
        f.name: getattr(default, f.name)
        for f in fields(PostprocessConfig)
        if not isinstance(getattr(default, f.name), tuple)  # the censuses ride the top level
    }
    block["calibrated_on"] = "loudr-1"
    return block


RECIPE_INPUTS = ("t3", "s3gen", "estimator", "template_manifest")
"""The files a build consumes, in the order a reader wants to see them."""


def apply_recipe(args: argparse.Namespace) -> None:
    """Fill `args` from the recipe file and verify every input's hash.

    The per-input hashes end up in the manifest's `sources` block, which
    every build writes anyway; what the recipe adds is that they were declared
    in advance and checked, rather than merely observed after the fact.
    """
    recipe = json.loads(Path(args.recipe).read_text())
    root = Path(args.root).expanduser() if args.root else Path(".")
    for key in ("name", "recipe_version", "n_cfm_timesteps", "out"):
        if key in recipe and getattr(args, key.replace("-", "_"), None) in (None, ""):
            setattr(args, key.replace("-", "_"), recipe[key])
    if "n_cfm_timesteps" in recipe:
        args.n_cfm_timesteps = int(recipe["n_cfm_timesteps"])

    measured: dict[str, str] = {}
    for key in RECIPE_INPUTS:
        entry = recipe.get("inputs", {}).get(key)
        if entry is None:
            continue
        declared = Path(entry["path"])
        path = declared if declared.is_absolute() else root / declared
        path = path.expanduser()
        if not path.exists():
            raise SystemExit(f"recipe input {key!r}: {path} does not exist")
        digest = file_sha256(path)
        want = entry.get("sha256") or ""
        if want and want != digest:
            raise SystemExit(
                f"recipe input {key!r}: {path}\n  recipe says {want}\n  file is   {digest}"
            )
        if not want:
            measured[key] = digest
        setattr(args, key, str(path))

    if measured:
        print("recipe inputs with no sha256 — measured now, paste into the recipe:")
        for key, digest in measured.items():
            print(f'  "{key}": {{"sha256": "{digest}"}}')


def main() -> None:  # noqa: PLR0915 - argparse builders read better flat
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default=None, help="build recipe JSON naming the inputs")
    ap.add_argument(
        "--root", default=None, help="directory the recipe's relative paths hang off"
    )
    ap.add_argument("--t3", help="turbo research ckpt (student+head2+fuse)")
    ap.add_argument("--s3gen", help="stock s3gen.safetensors")
    ap.add_argument("--estimator", default=None, help="distilled flow estimator ckpt")
    ap.add_argument(
        "--n-cfm-timesteps",
        type=int,
        default=None,
        help=(
            "Euler steps the estimator was distilled for. Taken from the "
            "estimator checkpoint when it records one; required when it does "
            "not, and when no estimator is packed"
        ),
    )
    ap.add_argument(
        "--template-manifest", help="JSON manifest to inherit policy from (loudr-1's)"
    )
    ap.add_argument("--name", default="loudkit-v0.1-turbo")
    # Still `loudkit-1`: the recipe gate stays strict, and what distinguishes
    # turbo is `format_version 2` plus the fingerprinted `decode` block,
    # both of which an old reader refuses rather than misreads.
    ap.add_argument("--recipe-version", default="loudkit-1")
    ap.add_argument("--out")
    ap.add_argument("--show-inherited", action="store_true")
    args = ap.parse_args()

    if args.recipe:
        apply_recipe(args)
    missing = [n for n in ("t3", "s3gen", "template_manifest", "out") if not getattr(args, n)]
    if missing:
        raise SystemExit("missing inputs: " + ", ".join(sorted(missing)))

    import torch
    from safetensors.torch import save_file

    out = Path(args.out)
    template = json.loads(Path(args.template_manifest).read_text())
    if template.get("format") != "loudkit-checkpoint":
        raise SystemExit(f"{args.template_manifest}: not a loudkit manifest")

    t3_ckpt = torch.load(args.t3, map_location="cpu", weights_only=False)
    t3_tensors, facts = build_t3_tensors(t3_ckpt)
    s3_tensors, folded, n_cfm_timesteps = build_s3gen_tensors(
        Path(args.s3gen),
        Path(args.estimator) if args.estimator else None,
        torch,
        int(args.n_cfm_timesteps) if args.n_cfm_timesteps is not None else None,
    )
    tensors = {**t3_tensors, **s3_tensors}

    dtype_map = template["dtype_map"]
    torch_dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    for name, tensor in list(tensors.items()):
        if not tensor.is_floating_point():
            continue
        want = resolve_dtype(name, dtype_map)
        if want:
            tensors[name] = tensor.to(torch_dtypes[want])
    tensors = {k: v.contiguous() for k, v in tensors.items()}

    build_recipe = None
    if args.recipe:
        build_recipe = {"path": args.recipe, "sha256": file_sha256(Path(args.recipe))}

    sources = {}
    for label, path in (("t3", args.t3), ("s3gen", args.s3gen), ("estimator", args.estimator)):
        if path:
            sources[str(path)] = {
                "bytes": Path(path).stat().st_size,
                "sha256": file_sha256(Path(path)),
                "role": label,
            }

    manifest = {k: template[k] for k in INHERITED_KEYS if k in template}
    manifest["postprocess"] = turbo_postprocess_block()
    manifest.update(
        {
            "format": "loudkit-checkpoint",
            # Two tokens per forward is not a flag a v1 reader can ignore: it would
            # run the loop it knows and emit plausible garbage. New version, loud door.
            "format_version": 2,
            "name": args.name,
            "recipe_version": args.recipe_version,
            "identity_contract_version": 1,
            "edge_fade_seconds": 0.02,
            "guidance": "single_path",
            "guidance_rate": 0.0,
            # The number `build_s3gen_tensors` resolved and checked, not the
            # raw flag: the manifest is what everything downstream believes, so
            # it must carry the value the estimator was verified against.
            "n_cfm_timesteps": n_cfm_timesteps,
            "llama_config": facts["llama_config"],
            "speech_vocab_size": facts["speech_vocab_size"],
            "decode": {
                "mode": "fusion_mtp2",
                "head1": "speech_head(h) -> first token of the pair",
                "head2": "head2([h ; speech_emb(first)]) -> second token of the pair",
                "slot": "0.5 * (ea + eb) + fuse([ea ; eb]), ea/eb = speech_emb of the pair",
                "position_clock": "one position per PAIR (not per token)",
                "stop": "either head emitting the stop token ends the sequence",
                "sampler": "unchanged from v1, including the silence-token exemptions",
            },
            "weight_norm": {
                "folded": True,
                "method": "torch.nn.utils.parametrize.remove_parametrizations("
                "mod, 'weight', leave_parametrized=True)",
                "modules": folded,
            },
            "sources": sources,
            "torch_version_at_pack": torch.__version__,
            "build_recipe": build_recipe,
        }
    )
    if "t3_keep_layers" in facts:
        manifest["t3_keep_layers"] = facts["t3_keep_layers"]

    out.parent.mkdir(parents=True, exist_ok=True)
    # Written once. The payload hash belongs in the manifest that ships, and
    # reading it back off a first write costs a second write of the whole
    # checkpoint. `tensor_payload_sha256` computes the same recipe over the
    # tensors still in memory, and the file form below proves the two agree.
    manifest["tensor_payload_sha256"] = tensor_payload_sha256(tensors)
    save_file(tensors, str(out), metadata={"manifest": json.dumps(manifest)})
    if payload_sha256(out) != manifest["tensor_payload_sha256"]:
        raise SystemExit(f"{out.name}: the payload written is not the payload hashed")
    # `newline="\n"`: a bundle member, and the bundle's digests are pinned.
    Path(str(out) + ".manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8", newline="\n"
    )

    n_t3 = sum(1 for k in tensors if k.startswith("t3."))
    n_s3 = sum(1 for k in tensors if k.startswith("s3gen."))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(
        f"  t3 {n_t3} tensors (incl. head2 + fuse), s3gen {n_s3}, "
        f"weight-norm folded on {folded} modules"
    )
    print(
        f"  layers {manifest['llama_config']['num_hidden_layers']}, "
        f"vocab {manifest['speech_vocab_size']}, "
        f"n_cfm_timesteps {manifest['n_cfm_timesteps']}"
    )
    print(f"  payload sha256 {manifest['tensor_payload_sha256'][:16]}…")
    print("  format_version 2 — readers must implement decode.mode=fusion_mtp2")
    if args.show_inherited:
        print(
            "  inherited from template: "
            + ", ".join(k for k in INHERITED_KEYS if k in template)
        )


if __name__ == "__main__":
    main()
