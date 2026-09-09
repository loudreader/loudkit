"""Export a checkpoint's generator and renderer as CoreML packages.

The manifest selects t3_cond/t3_prefill/t3_step for single-token decoding, or
t3_cond/t3_prefill/t3_pair_step/t3_head2 for fusion_mtp2. Generator packages
use fp32 CPU compute with explicit, dynamically sized KV tensors. Prefill
accepts embeddings; the runtime builds speech-prefix slots.

The renderer is flow_encoder (fp32 CPU), flow_estimator (fp16 CPU+ANE for
single-token decoding, fp32 for fusion_mtp2, see the estimator block), and
vocoder (fp32 CPU). Its geometry and Euler steps come from AlgorithmConfig.
Randomness remains an explicit graph input.

Generator stages are exported together and must pass fp32 logit comparisons
and exact sampled-token gates before promotion. Renderer stages retain their
component gates. Failed conversion or parity gates leave existing outputs unchanged.

Usage:
    python tools/export_coreml.py --checkpoint PATH [--out DIR] [--stages ...]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from distill_meta import estimator_k, file_sha256
from export_generator import (
    GENERATOR_STAGES,
    export_generator,
    exporter_parser,
    gate_generator,
    generator_filename,
    install_ct_shims,
    load_generator,
    promote,
    require_decode,
    selected_stages,
    staging_directory,
    write_provenance,
)
from export_renderer import (
    RENDERER_INPUTS,
    RENDERER_STAGES,
    load_mel_decoder,
    renderer_geometry,
    renderer_stage,
)

from loudkit.backends import production_algorithm
from loudkit.checkpoint import Checkpoint

ENCODER_NAME = "flow_encoder.mlpackage"
ESTIMATOR_NAME = "flow_estimator.mlpackage"
VOCODER_NAME = "vocoder.mlpackage"

STAGE_PACKAGES = {
    **{
        stage: generator_filename(stage, "coreml")
        for stage in ("cond", "prefill", "step", "pair_step", "head2")
    },
    "encoder": ENCODER_NAME,
    "estimator": ESTIMATOR_NAME,
    "vocoder": VOCODER_NAME,
}
"""The stages `--stages` accepts, and the package each one writes."""


def _convert_and_gate(
    module: nn.Module,
    example: tuple,
    input_names: list[str],
    int_inputs: bool,
    out_path: Path,
    *,
    fp16: bool,
    compute_units: str,
    reference: torch.Tensor,
    corr_gate: float,
):
    import coremltools as ct

    install_ct_shims()
    traced = torch.jit.trace(module, example, strict=False)
    if int_inputs:
        inputs = [
            ct.TensorType(name=n, shape=tuple(t.shape), dtype=int)
            for n, t in zip(input_names, example, strict=True)
        ]
    else:
        inputs = [
            ct.TensorType(name=n, shape=tuple(t.shape))
            for n, t in zip(input_names, example, strict=True)
        ]
    mlm = ct.convert(
        traced,
        inputs=inputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
        compute_units=getattr(ct.ComputeUnit, compute_units),
    )
    tmp = out_path.with_name(out_path.stem + ".tmp.mlpackage")
    shutil.rmtree(tmp, ignore_errors=True)
    mlm.save(str(tmp))

    feed = {
        n: (t.numpy().astype(np.int32) if int_inputs else t.numpy().astype(np.float32))
        for n, t in zip(input_names, example, strict=True)
    }
    got = torch.from_numpy(np.asarray(next(iter(mlm.predict(feed).values())), dtype=np.float32))
    ref = reference.flatten().double()
    corr = torch.corrcoef(torch.stack([ref, got.flatten().double()]))[0, 1].item()
    max_err = (got.flatten().double() - ref).abs().max().item()
    # Reported, not gated, and the ONNX twin gates on both. One function
    # converts every package here, and the set spans fp32 CPU and an fp16
    # estimator on the ANE, whose worst element is orders away from the fp32
    # bound while the shape it produces is not: `corr_gate` already carries
    # that split (0.999 against 0.999999) and a single element bound cannot.
    # Gating the fp32 packages on their own bound is 0.1.2 work; it needs the
    # bound measured on a real conversion first, not guessed here.
    ok = corr >= corr_gate
    print(
        f"  {out_path.name}: corr {corr:.7f} (gate {corr_gate}), max|err| {max_err:.3e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"{out_path.name}: parity gate failed (corr {corr:.7f} < {corr_gate})")
    shutil.rmtree(out_path, ignore_errors=True)
    tmp.rename(out_path)
    return corr, max_err


def _check_estimator_k(blob: dict, algo, args) -> int:
    """Refuse an estimator distilled for a step count this export will not run.

    The recipe is explicit that a step-distilled estimator "reproduces a
    six-step stock rollout in one Euler step" and that "running it at any other
    K is a different algorithm, which is why this number ships beside the
    weights instead of being a runtime flag". This flag traces the weights
    against whatever checkpoint it was handed, and the loop count comes from
    *that* checkpoint's manifest, so pointing it at a stock `loudr-1` (K=2)
    with a K=1 estimator exported a package that runs the distilled estimator
    twice, printed `euler=2`, and said nothing. The CoreML package carries no K
    of its own: the graph is one step and the count lives in the engine's
    config, so nothing downstream could catch it either.

    Refused rather than reconciled, like every other manifest disagreement
    here. If the checkpoint file does not record its K, the operator states it
    with `--estimator-k`, the same shape as `pack_turbo.py`, which takes
    `n_cfm_timesteps` from a declared recipe rather than guessing.
    """
    try:
        declared = estimator_k(blob)
    except ValueError as exc:
        raise SystemExit(f"{args.estimator_ckpt}: {exc}") from None
    if declared is None and args.estimator_k is None:
        raise SystemExit(
            f"{args.estimator_ckpt} does not record the Euler step count it was "
            f"distilled for, and this export runs {algo.euler_steps} "
            f"(from {Path(args.checkpoint).name}). Pass --estimator-k to state it: "
            f"an estimator run at a K it was not distilled for is a different "
            f"algorithm, not the same one at a different speed."
        )
    if declared is not None and args.estimator_k is not None and declared != args.estimator_k:
        raise SystemExit(
            f"--estimator-k {args.estimator_k} contradicts the checkpoint, which "
            f"records {declared}."
        )
    k = declared if declared is not None else args.estimator_k
    if k != algo.euler_steps:
        raise SystemExit(
            f"{args.estimator_ckpt} was distilled for K={k}; "
            f"{Path(args.checkpoint).name} declares euler_steps={algo.euler_steps}. "
            f"Export from a checkpoint packed at K={k} (see tools/pack_turbo.py), "
            f"or point this at the estimator that belongs to this manifest."
        )
    print(f"[estimator] K={k}, which is the manifest's")
    return k


def _write_provenance(out_dir, stages, ckpt, algo, args, estimator_k_value) -> None:
    """The shared record, plus the two things only a CoreML export knows.

    A swapped-in estimator is weights this run traced that the checkpoint digest
    does not cover, and the precision each package was converted at is what the
    reader compares an ANE-resident estimator against.
    """
    extra: dict[str, object] = {}
    if args.estimator_ckpt:
        extra["estimator_sha256"] = file_sha256(args.estimator_ckpt)
        extra["estimator_k"] = estimator_k_value
    write_provenance(
        out_dir,
        stages,
        ckpt,
        algo,
        backend="coreml",
        stage_files=STAGE_PACKAGES,
        extra=extra,
        per_stage=lambda stage: {
            "precision": "fp16" if stage == "estimator" and algo.decode == "single" else "fp32"
        },
    )


def main() -> None:
    ap = exporter_parser(__doc__, "coreml")
    ap.add_argument(
        "--estimator-ckpt",
        default=None,
        help="distilled estimator weights (torch .pt with student_sd/ema_sd) "
        "to swap in before tracing; the rest of the flow stays as packed",
    )
    ap.add_argument(
        "--estimator-k",
        type=int,
        default=None,
        help="the Euler step count --estimator-ckpt was distilled for, when the "
        "checkpoint file does not record it. Must equal the manifest's.",
    )
    args = ap.parse_args()

    ckpt = Checkpoint.open(args.checkpoint)
    algo = production_algorithm(ckpt)
    require_decode(algo.decode)
    stages = selected_stages(args, algo)
    destination = Path(args.out) if args.out else ckpt.path.parent / "coreml"
    out_dir = staging_directory(destination)

    t_mel, hift_frames = renderer_geometry(ckpt, algo)

    torch.manual_seed(0)

    if stages & set(GENERATOR_STAGES[algo.decode]):
        gen = load_generator(ckpt, algo)
        runs = export_generator(gen, out_dir, stages, "coreml")
        gate_generator(
            gen, runs, Path(__file__).resolve().parent.parent / "tests/data/conformance"
        )

    # -- torch reference modules, straight from the packed weights ----------
    decoder = load_mel_decoder(ckpt, algo)
    estimator_declared_k: int | None = None
    if args.estimator_ckpt:
        # A step-distilled estimator is trained apart from the packed weights, so
        # it arrives as a bare submodule state dict. EMA is the eval copy, and that
        # is what the melGEN number was measured on.
        blob = torch.load(args.estimator_ckpt, map_location="cpu", weights_only=False)
        estimator_declared_k = _check_estimator_k(blob, algo, args)
        sd = blob.get("ema_sd") or blob["student_sd"]
        # `strict=True` is the check; a name that does not belong raises here.
        decoder.decoder.estimator.load_state_dict(sd, strict=True)
        print(
            f"[estimator] swapped in {args.estimator_ckpt} (step {blob.get('step')}, "
            f"score {blob.get('score')})"
        )

    decoder = decoder.float().eval()  # trace in fp32; ct re-quantizes the estimator

    for stage in RENDERER_STAGES:
        if stage not in stages:
            continue
        print(f"[{stage}]")
        module, example, reference = renderer_stage(
            stage, decoder, ckpt, algo, t_mel, hift_frames
        )
        # Turbo amplifies fp16 mel error into audible-wave phase drift.
        # Keep its one-step estimator in fp32; preserve the base export recipe.
        fp16 = stage == "estimator" and algo.decode == "single"
        _convert_and_gate(
            module,
            example,
            RENDERER_INPUTS[stage],
            stage == "encoder",
            out_dir / STAGE_PACKAGES[stage],
            fp16=fp16,
            compute_units="CPU_AND_NE" if stage == "estimator" else "CPU_ONLY",
            reference=reference,
            corr_gate=0.999 if fp16 else 0.999999,
        )

    _write_provenance(out_dir, stages, ckpt, algo, args, estimator_declared_k)
    promote(out_dir, destination)
    print(f"done -> {destination}")


if __name__ == "__main__":
    main()
