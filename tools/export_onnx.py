"""Export a checkpoint's generator and renderer as ONNX graphs.

The signatures and call order of what this writes are in
``docs/design/onnx-graphs.md``, read off the published files.

The manifest selects t3_cond/t3_prefill/t3_step for single-token decoding, or
t3_cond/t3_prefill/t3_pair_step/t3_head2 for fusion_mtp2. Prefill accepts
embeddings, built by the runtime, and every step takes explicit past KV.
The fused graphs also expose the hidden state needed by the second head.

The renderer is flow_encoder, flow_estimator, and vocoder. Its geometry and
Euler steps come from AlgorithmConfig. Randomness is an explicit graph input.

Generator stages are exported together and must pass fp32 logit comparisons
and exact sampled-token gates before promotion. Renderer stages retain their
component gates. Failed conversion or parity gates leave existing outputs unchanged.

Usage:
    python tools/export_onnx.py --checkpoint PATH [--out DIR] [--stages ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_generator import (
    GENERATOR_STAGES,
    export_generator,
    exporter_parser,
    gate_generator,
    generator_filename,
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

# The exported graph set, in the order the backend loads them.
ENCODER_NAME = "flow_encoder.onnx"
ESTIMATOR_NAME = "flow_estimator.onnx"
VOCODER_NAME = "vocoder.onnx"

STAGE_GRAPHS = {
    **{
        stage: generator_filename(stage, "onnx")
        for stage in ("cond", "prefill", "step", "pair_step", "head2")
    },
    "encoder": ENCODER_NAME,
    "estimator": ESTIMATOR_NAME,
    "vocoder": VOCODER_NAME,
}
"""The stages `--stages` accepts, and the graph each one writes."""


def _export_and_gate(
    module: nn.Module,
    example: tuple,
    input_names: list[str],
    out_path: Path,
    *,
    int_inputs: set[str] | bool,
    dynamic_axes: dict[str, dict[int, str]] | None,
    reference: torch.Tensor,
    corr_gate: float,
    max_abs_gate: float,
):
    import onnxruntime as ort

    if isinstance(int_inputs, bool):
        int_inputs = set(input_names) if int_inputs else set()

    tmp = out_path.with_name(out_path.stem + ".tmp.onnx")
    with torch.no_grad():
        torch.onnx.export(
            module,
            example,
            str(tmp),
            input_names=input_names,
            output_names=[f"{Path(out_path.stem)}_out"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    feed = {
        n: (t.numpy().astype(np.int64) if n in int_inputs else t.numpy().astype(np.float32))
        for n, t in zip(input_names, example, strict=True)
    }
    got = np.asarray(sess.run(None, feed)[0], dtype=np.float32)
    ref = reference.numpy().astype(np.float32)
    max_err = float(np.abs(got - ref).max())
    corr = float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
    ok = corr >= corr_gate and max_err <= max_abs_gate
    print(
        f"  {out_path.name}: corr {corr:.7f} (gate {corr_gate}), "
        f"max|err| {max_err:.3e} (gate {max_abs_gate:.0e}) -> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{out_path.name}: ONNX parity gate failed")
    # replace, not rename: the target is a file, and rename would fail on an
    # existing one on Windows
    tmp.replace(out_path)
    return corr, max_err


def main() -> None:
    args = exporter_parser(__doc__, "onnx").parse_args()

    ckpt = Checkpoint.open(args.checkpoint)
    algo = production_algorithm(ckpt)
    require_decode(algo.decode)
    stages = selected_stages(args, algo)
    destination = Path(args.out) if args.out else ckpt.path.parent / "onnx"
    out_dir = staging_directory(destination)

    t_mel, hift_frames = renderer_geometry(ckpt, algo)

    torch.manual_seed(0)
    if stages & set(GENERATOR_STAGES[algo.decode]):
        gen = load_generator(ckpt, algo)
        runs = export_generator(gen, out_dir, stages, "onnx")
        gate_generator(
            gen, runs, Path(__file__).resolve().parent.parent / "tests/data/conformance"
        )

    # -- renderer graphs -----------------------------------------------------
    decoder = load_mel_decoder(ckpt, algo).float().eval()

    for stage in RENDERER_STAGES:
        if stage not in stages:
            continue
        print(f"[{stage}]")
        module, example, reference = renderer_stage(
            stage, decoder, ckpt, algo, t_mel, hift_frames
        )
        _export_and_gate(
            module,
            example,
            RENDERER_INPUTS[stage],
            out_dir / STAGE_GRAPHS[stage],
            int_inputs=stage == "encoder",
            dynamic_axes=None,
            reference=reference,
            corr_gate=0.999999,
            max_abs_gate=1e-3,
        )

    write_provenance(out_dir, stages, ckpt, algo, backend="onnx", stage_files=STAGE_GRAPHS)
    promote(out_dir, destination)
    print(f"done -> {destination}")


if __name__ == "__main__":
    main()
