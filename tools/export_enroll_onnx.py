"""Export the enrollment stage graphs from the packed checkpoint.

Three graphs, all fp32, each gated against the torch modules loaded from the
*same* checkpoint before anything is saved. The DSP — the two resamplers and
the four filterbanks — is deliberately *not* in these graphs: it lives on the
host in every port (``EnrollmentDSP.swift`` is the reference), so a graph
exported with the mel inside it would be a second, silently different copy of
the same arithmetic. What the graphs carry is only what cannot be written
portably by hand:

  s3_tokenizer.onnx   mel [1,128,F] -> speech tokens [T]       (dynamic F)
  camp.onnx           kaldi fbank [1,80,F] -> x-vector [192]   (dynamic F)
  voice_encoder.onnx  partials [n,160,40] -> per-partial [n,256]

The voice encoder's weights are *not* in the packed checkpoint — pass
``--voice-encoder ve.safetensors``, the same file the enroller needs.

Gates: the tokenizer emits discrete tokens, so it is gated on exact token
agreement against torch over the fixture-shaped inputs. The two encoders emit
continuous vectors and are gated on correlation and max|err| like the synthesis
graphs. A failed gate raises and leaves nothing half-exported.

Usage (from the loudkit repo root):
  .venv/bin/python tools/export_enroll_onnx.py \\
      --checkpoint /path/to/loudr-1.safetensors \\
      --voice-encoder /path/to/ve.safetensors [--out DIR]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from loudkit.checkpoint import Checkpoint  # noqa: E402
from loudkit.models.enroll import (  # noqa: E402
    _CAMLayer,
    _CAMPPlus,
    _S3Tokenizer,
    _VoiceEncoder,
)

TOKENIZER_NAME = "s3_tokenizer.onnx"
CAMP_NAME = "camp.onnx"
VOICE_ENC_NAME = "voice_encoder.onnx"

CORR_GATE = 0.999999
MAX_ABS_GATE = 1e-3


def _seg_pool_fixed(x: torch.Tensor, seg_len: int = 100) -> torch.Tensor:
    """``_CAMLayer._seg_pool`` in a form ONNX's AveragePool does not mangle.

    The reference averages ``avg_pool1d(..., ceil_mode=True)``, whose last
    window is a partial segment averaged over its *available* elements. The
    TorchScript ONNX exporter lowers that to AveragePool with ``ceil_mode`` and
    ``count_include_pad`` set so the last window divides by the full kernel
    width instead — the last segment comes out diluted by zero padding (measured
    0.1458 -> 0.0802 on the fixture geometry). Pool the sums and the counts
    separately and divide: the last window divides by exactly the elements it
    has, matching the reference to the last ulp of the summation.
    """
    t = x.shape[-1]
    n_seg = (t + seg_len - 1) // seg_len
    padded = n_seg * seg_len
    pad = padded - t
    xp = torch.nn.functional.pad(x, (0, pad))
    ones = torch.nn.functional.pad(torch.ones_like(x), (0, pad))
    sums = torch.nn.functional.avg_pool1d(xp, seg_len, seg_len) * seg_len
    counts = torch.nn.functional.avg_pool1d(ones, seg_len, seg_len) * seg_len
    means = sums / counts
    # Repeat each segment mean seg_len times with expand+reshape, not
    # repeat_interleave: the latter lowers to a shape ONNX Runtime cannot infer
    # inside the dense blocks (Concat "axis must be in [-rank, rank-1]").
    shape = means.shape
    expanded = means.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
    return expanded[..., :t]


class _TokenizerModel(nn.Module):
    """mel [1,128,F] -> speech tokens [T], the part of tokenize() past the mel."""

    def __init__(self, tok: _S3Tokenizer) -> None:
        super().__init__()
        self.encoder = tok.encoder
        self.codebook = tok.quantizer._codebook  # noqa: SLF001

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(mel)
        return self.codebook.encode(hidden)[0]


class _CAMPPModel(nn.Module):
    """kaldi fbank [1,80,F] -> x-vector [192], past the fbank and the mean-removal."""

    def __init__(self, spk: _CAMPPlus) -> None:
        super().__init__()
        self.head = spk.head
        self.xvector = spk.xvector

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        h = self.head(fbank)
        return self.xvector(h)[0]


class _VoiceEncModel(nn.Module):
    """partials [n,160,40] -> per-partial [n,256], past the trim, mel and pooling."""

    def __init__(self, ve: _VoiceEncoder) -> None:
        super().__init__()
        self.lstm = ve.lstm
        self.proj = ve.proj

    def forward(self, partials: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(partials)
        raw = F.relu(self.proj(hidden[-1]))
        return raw / torch.linalg.norm(raw, dim=1, keepdim=True)


def _export_float_graph(  # type: ignore[no-untyped-def]
    module: nn.Module,
    example: tuple,
    input_names: list[str],
    out_path: Path,
    *,
    dynamic_axes: dict[str, dict[int, str]] | None,
    reference: torch.Tensor,
    corr_gate: float = CORR_GATE,
):
    import onnxruntime as ort

    tmp = out_path.with_name(out_path.stem + ".tmp.onnx")
    with torch.no_grad():
        torch.onnx.export(
            module,
            example,
            str(tmp),
            input_names=input_names,
            output_names=["out"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    feed = {n: t.numpy().astype(np.float32) for n, t in zip(input_names, example, strict=True)}
    got = np.asarray(sess.run(None, feed)[0], dtype=np.float32)
    ref = reference.numpy().astype(np.float32)
    max_err = float(np.abs(got - ref).max())
    corr = float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
    ok = corr >= corr_gate and max_err <= MAX_ABS_GATE
    print(
        f"  {out_path.name}: corr {corr:.7f} (gate {corr_gate}), "
        f"max|err| {max_err:.3e} (gate {MAX_ABS_GATE:.0e}) -> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{out_path.name}: ONNX parity gate failed")
    shutil.rmtree(out_path, ignore_errors=True)
    tmp.rename(out_path)


def _load_spk_weights(ckpt: Checkpoint) -> dict[str, torch.Tensor]:
    tensors = ckpt.tensors("s3gen.speaker_encoder.")
    return {k: torch.from_numpy(v.copy()) for k, v in tensors.items()}


def main() -> None:  # noqa: PLR0915 — one export per stage, linear and logged
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--voice-encoder", required=True, help="ve.safetensors")
    ap.add_argument("--out", default=None, help="output dir (default: <ckpt dir>/onnx)")
    args = ap.parse_args()

    ckpt = Checkpoint.open(args.checkpoint)
    out_dir = Path(args.out) if args.out else ckpt.path.parent / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = _S3Tokenizer()
    tok.load_state_dict(
        {k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors("s3gen.tokenizer.").items()}
    )
    tok = tok.float().eval()

    spk = _CAMPPlus()
    spk.load_state_dict(_load_spk_weights(ckpt))
    spk = spk.float().eval()

    from safetensors.torch import load_file

    ve = _VoiceEncoder()
    ve.load_state_dict(load_file(args.voice_encoder))
    ve = ve.float().eval()

    torch.manual_seed(0)

    # The fixture-shaped inputs: 512 mel frames is the shipped prompt, 510
    # fbank frames is the reference clip's Kaldi count, 160x40 is one partial.
    print("[s3_tokenizer]")
    tmod = _TokenizerModel(tok).eval()
    mel = torch.randn(1, 128, 512)
    with torch.no_grad():
        ref = tmod(mel)
    tmp = out_dir / TOKENIZER_NAME
    tmp_t = tmp.with_name(tmp.stem + ".tmp.onnx")
    with torch.no_grad():
        torch.onnx.export(
            tmod,
            (mel,),
            str(tmp_t),
            input_names=["mel"],
            output_names=["tokens"],
            dynamic_axes={"mel": {2: "mel_frames"}, "tokens": {0: "tokens"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(tmp_t), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"mel": mel.numpy()})[0].astype(np.int64)
    agree = int((got == ref.numpy()).sum())
    ok = agree == ref.numel()
    print(
        f"  {TOKENIZER_NAME}: tokens {agree}/{ref.numel()} exact -> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        tmp_t.unlink(missing_ok=True)
        raise SystemExit(f"{TOKENIZER_NAME}: token parity gate failed")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp_t.rename(tmp)

    print("[camp]")
    fbank = torch.randn(1, 80, 510)
    with torch.no_grad():
        camp_ref = _CAMPPModel(spk)(fbank)
    # The reference uses avg_pool1d(ceil_mode=True), which ONNX lowers wrong.
    # Compute the reference first, then swap in the ONNX-safe seg_pool and
    # build a second model from the same weights to export.
    _CAMLayer._seg_pool = staticmethod(_seg_pool_fixed)  # type: ignore[method-assign]
    spk_export = _CAMPPlus()
    spk_export.load_state_dict(_load_spk_weights(ckpt))
    spk_export = spk_export.float().eval()
    cmod = _CAMPPModel(spk_export).eval()
    _export_float_graph(
        cmod,
        (fbank,),
        ["fbank"],
        out_dir / CAMP_NAME,
        dynamic_axes={"fbank": {2: "fbank_frames"}},
        reference=camp_ref,
    )

    print("[voice_encoder]")
    vmod = _VoiceEncModel(ve).eval()
    partials = torch.randn(4, 160, 40)
    with torch.no_grad():
        ref = vmod(partials)
    _export_float_graph(
        vmod,
        (partials,),
        ["partials"],
        out_dir / VOICE_ENC_NAME,
        dynamic_axes={"partials": {0: "n_partials"}},
        reference=ref,
        # The LSTM lowers to ONNX's LSTM op, whose gate layout is equivalent
        # but not bit-identical to torch's — measured corr 0.9999988. The
        # embedding is L2-normalised downstream, so what matters is direction,
        # and the enrollment fixture gates the shipped embedding on cosine
        # > 0.9999. A corr bar of 0.9999 is the same tolerance, not a weaker one.
        corr_gate=0.9999,
    )

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
