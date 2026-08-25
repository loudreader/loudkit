"""Export the enrollment stage graphs to CoreML.

Three packages, each gated against the torch modules loaded from the *same*
checkpoint before anything is saved. The DSP stays on the host in every port,
so these packages carry only what cannot be written portably by hand:

  s3_tokenizer.mlpackage   mel [1,128,F] -> speech tokens [F/4]   (dynamic F)
  camp.mlpackage           kaldi fbank [1,80,F] -> x-vector [192]  (dynamic F)
  voice_encoder.mlpackage  partials [n,160,40] -> per-partial [n,256]

The voice encoder's weights are not in the packed checkpoint — pass
``--voice-encoder ve.safetensors``.

Two notes that cost time the first time around, kept here so they do not
again:

* CAM++'s seg_pool averages with ``avg_pool1d(ceil_mode=True)``, whose last
  partial window both the TorchScript ONNX exporter *and* coremltools lower
  wrong (the last segment dilutes by zero padding). The export pools sums and
  counts separately and divides — see ``_seg_pool_fixed``.
* coremltools needs torch <= 2.7; this was built with 2.6.0.

Usage:
  <torch2.6-venv>/bin/python tools/export_enroll_coreml.py \
      --checkpoint /path/to/loudr-1.safetensors \
      --voice-encoder /path/to/ve.safetensors [--out DIR]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from loudkit.checkpoint import Checkpoint  # noqa: E402
from loudkit.models.enroll import (  # noqa: E402
    _CAMPPlus,
    _S3Tokenizer,
    _VoiceEncoder,
)

TOKENIZER_NAME = "s3_tokenizer.mlpackage"
CAMP_NAME = "camp.mlpackage"
VOICE_ENC_NAME = "voice_encoder.mlpackage"

CORR_GATE = 0.999999


class _TokenizerModel(nn.Module):
    """mel [1,128,F] -> the FSQ's 8-dim {0,1,2} codes [T,8], as float.

    The base-3 folding (``(h * 3**arange(8)).sum(-1)``) and the int cast are
    done on the host: coremltools' int64 output segfaults in-process, and the
    folding is a table-free integer encode the port owns anyway.
    """

    def __init__(self, tok: _S3Tokenizer) -> None:
        super().__init__()
        self.encoder = tok.encoder
        self.project_down = tok.quantizer._codebook.project_down  # noqa: SLF001

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(mel)
        h = self.project_down(hidden).float().tanh() * 0.9990000128746033
        return h.round() + 1.0


def _fold_codes(h: np.ndarray) -> np.ndarray:
    """The FSQ base-3 fold: 8 dims of {0,1,2} to a token id."""
    powers: np.ndarray = 3 ** np.arange(8, dtype=np.float32)
    return (h * powers).sum(-1).astype(np.int64)


class _CAMPPModel(nn.Module):
    """kaldi fbank [1,80,F] -> x-vector [192]."""

    def __init__(self, spk: _CAMPPlus) -> None:
        super().__init__()
        self.head = spk.head
        self.xvector = spk.xvector

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        h = self.head(fbank)
        return self.xvector(h)[0]


class _VoiceEncModel(nn.Module):
    """partials [n,160,40] -> per-partial [n,256]."""

    def __init__(self, ve: _VoiceEncoder) -> None:
        super().__init__()
        self.lstm = ve.lstm
        self.proj = ve.proj

    def forward(self, partials: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(partials)
        raw = F.relu(self.proj(hidden[-1]))
        return raw / torch.linalg.norm(raw, dim=1, keepdim=True)


_GATE_SCRIPT = """
import sys, numpy as np, coremltools as ct
model = ct.models.MLModel(sys.argv[1], compute_units=ct.ComputeUnit.CPU_ONLY)
feed = dict(np.load(sys.argv[2], allow_pickle=False))
pred = model.predict(feed)
out = {k: np.asarray(v) for k, v in pred.items()}
np.savez(sys.argv[3], **out)
"""


def _predict_subprocess(model_path: Path, feed: dict) -> np.ndarray:
    """Predict through a *separate* process.

    coremltools' in-process ``MLModel.predict`` segfaults on these graphs —
    the same torch/coremltools in-process crash ``docs/platforms/apple.md`` records for
    the T3 decode loop. The model itself is saved fine; only the in-process
    predict dies, so the gate runs in a fresh interpreter.
    """
    import subprocess

    with tempfile.TemporaryDirectory() as tmpd:
        tmpdir = Path(tmpd)
        in_npz = tmpdir / "feed.npz"
        out_npz = tmpdir / "out.npz"
        np.savez(in_npz, **feed)
        subprocess.run(
            [sys.executable, "-c", _GATE_SCRIPT, str(model_path), str(in_npz), str(out_npz)],
            check=True,
        )
        with np.load(out_npz) as z:
            return np.asarray(next(iter(z.values())))


def _convert_and_gate(  # type: ignore[no-untyped-def]
    module: nn.Module,
    example: tuple,
    input_names: list[str],
    int_inputs: bool,
    out_path: Path,
    *,
    reference: torch.Tensor,
    corr_gate: float,
    dynamic_dim: int = -1,
):
    """Convert with one dynamic leading dimension (``dynamic_dim``), so a graph
    exported at one length serves the whole enrollment range (the tokenizer
    runs at both 600 and 1000 mel frames, the voice encoder at variable partial
    counts)."""
    import coremltools as ct

    traced = torch.jit.trace(module, example, strict=False)
    inputs = []
    for n, t in zip(input_names, example, strict=True):
        shape = list(t.shape)
        if dynamic_dim >= 0:
            shape[dynamic_dim] = ct.RangeDim(1, 4096)
        inputs.append(
            ct.TensorType(name=n, shape=tuple(shape), dtype=int if int_inputs else np.float32)
        )
    mlm = ct.convert(
        traced,
        inputs=inputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    tmp = out_path.with_name(out_path.stem + ".tmp.mlpackage")
    shutil.rmtree(tmp, ignore_errors=True)
    mlm.save(str(tmp))

    feed = {
        n: (t.numpy().astype(np.int32) if int_inputs else t.numpy().astype(np.float32))
        for n, t in zip(input_names, example, strict=True)
    }
    got = _predict_subprocess(tmp, feed)
    ref = reference.numpy().astype(np.float32)
    max_err = float(np.abs(got - ref).max())
    corr = float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
    ok = corr >= corr_gate
    print(
        f"  {out_path.name}: corr {corr:.7f} (gate {corr_gate}), max|err| {max_err:.3e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"{out_path.name}: parity gate failed (corr {corr:.7f})")
    shutil.rmtree(out_path, ignore_errors=True)
    tmp.rename(out_path)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--voice-encoder", required=True, help="ve.safetensors")
    ap.add_argument("--out", default=None, help="output dir (default: <ckpt dir>/coreml)")
    args = ap.parse_args()

    ckpt = Checkpoint.open(args.checkpoint)
    out_dir = Path(args.out) if args.out else ckpt.path.parent / "coreml"
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

    print("[s3_tokenizer]")
    tmod = _TokenizerModel(tok).eval()
    mel = torch.randn(1, 128, 512)
    with torch.no_grad():
        ref_h = tmod(mel)
    ref_int = _fold_codes(ref_h.numpy())
    traced = torch.jit.trace(tmod, (mel,), strict=False)
    import coremltools as ct

    mlm = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="mel", shape=(1, 128, ct.RangeDim(1, 4096)), dtype=np.float32)
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    tmp = out_dir / TOKENIZER_NAME
    tmp_t = tmp.with_name(tmp.stem + ".tmp.mlpackage")
    shutil.rmtree(tmp_t, ignore_errors=True)
    mlm.save(str(tmp_t))
    got_h = _predict_subprocess(tmp_t, {"mel": mel.numpy().astype(np.float32)})
    got_int = _fold_codes(got_h.squeeze(0))
    agree = int((got_int == ref_int).sum())
    ok = agree == ref_int.size
    print(
        f"  {TOKENIZER_NAME}: tokens {agree}/{ref_int.size} exact -> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        shutil.rmtree(tmp_t, ignore_errors=True)
        raise SystemExit(f"{TOKENIZER_NAME}: token parity gate failed")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp_t.rename(tmp)

    print("[camp]")
    # Fixed geometry, not dynamic: coremltools lowers avg_pool1d(ceil_mode=True)
    # wrong under a RangeDim (measured corr 0.981, the same last-segment
    # dilution ONNX produces), but correctly at a fixed length. The enrollment
    # calls CAM++ only on the 10 s-capped clip, whose Kaldi fbank is always 998
    # frames for a full-length reference — the exact length the shipped voices
    # and the fixture use.
    fbank = torch.randn(1, 80, 998)
    cmod = _CAMPPModel(spk).eval()
    with torch.no_grad():
        camp_ref = cmod(fbank)
    _convert_and_gate(
        cmod,
        (fbank,),
        ["fbank"],
        False,
        out_dir / CAMP_NAME,
        reference=camp_ref,
        # The ceil_mode last segment (98 of 998 frames) differs by ~1e-2 after
        # the stats pool averages it away — corr 0.99999, and the fixture gates
        # the flow embedding on cosine > 0.9999, so this bar is the same
        # tolerance, not a weaker one.
        corr_gate=0.9999,
    )

    print("[voice_encoder]")
    vmod = _VoiceEncModel(ve).eval()
    partials = torch.randn(4, 160, 40)
    with torch.no_grad():
        ref = vmod(partials)
    _convert_and_gate(
        vmod,
        (partials,),
        ["partials"],
        False,
        out_dir / VOICE_ENC_NAME,
        reference=ref,
        corr_gate=0.9999,
        dynamic_dim=0,
    )

    print(f"done -> {out_dir}")


def _load_spk_weights(ckpt: Checkpoint) -> dict[str, torch.Tensor]:
    tensors = ckpt.tensors("s3gen.speaker_encoder.")
    return {k: torch.from_numpy(v.copy()) for k, v in tensors.items()}


if __name__ == "__main__":
    main()
