"""Generate the enrollment DSP tables the ports load.

The enrollment filterbanks need the same mel filters and windows in all five
languages, and the Python reference computes them through three libraries
(librosa's Slaney mel, torch's Hann windows, torchaudio's Kaldi mel) with
conventions that differ per path. A filterbank built from the wrong window
does not fail to build — it returns numbers and a voice comes out, quietly
worse, with nothing to point at. So the tables are materialised once, here,
and every port loads the same float32 values.

Dumped (all raw little-endian float32, shapes in the manifest):

  s3_mel_filters.f32      (128, 201)  the tokenizer's filters, from the checkpoint
  s3_hann400.f32          (400,)      torch.hann_window(400) — periodic
  matcha_mel_filters.f32  (80, 961)   librosa Slaney mel @ 24 kHz
  matcha_hann1920.f32     (1920,)     torch.hann_window(1920) — periodic
  voiceenc_mel_filters.f32 (40, 201)  librosa Slaney mel @ 16 kHz
  voiceenc_hann400.f32    (400,)      scipy symmetric hann — librosa.stft's default
  kaldi_mel_filters.f32   (80, 257)   torchaudio's Kaldi mel banks
  kaldi_povey400.f32      (400,)      torch.hann_window(400, periodic=False) ** 0.85

The two 400-point Hann windows differ on purpose and are dumped separately:
the tokenizer runs torch's *periodic* hann while the voice encoder runs
librosa's *symmetric* hann, and a port that reuses one for both produces a
different mel for the other path.

Usage:
  .venv/bin/python tools/gen_dsp_assets.py --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from loudkit.checkpoint import Checkpoint  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "python" / "loudkit" / "models" / "data" / "dsp"


def _write(name: str, values: np.ndarray) -> None:
    np.asarray(values, dtype=np.float32).tofile(OUT / name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import torch
    from torchaudio.compliance import kaldi as tkaldi

    ck = Checkpoint.open(args.checkpoint)
    tensors = ck.tensors("s3gen.tokenizer.")
    s3_filters = tensors["_mel_filters"]
    s3_window = tensors["window"]

    import librosa

    matcha_filters = librosa.filters.mel(  # type: ignore[attr-defined]
        sr=24_000, n_fft=1920, n_mels=80, fmin=0, fmax=8000
    )
    voiceenc_filters = librosa.filters.mel(  # type: ignore[attr-defined]
        sr=16_000, n_fft=400, n_mels=40, fmin=0, fmax=8000
    )

    matcha_hann = torch.hann_window(1920).numpy()

    from scipy.signal import get_window

    voiceenc_hann = get_window("hann", 400, fftbins=False).astype(np.float32)

    kaldi_filters = tkaldi.get_mel_banks(80, 512, 16_000, 20, 8000, 100, -500, 1.0)[0].numpy()
    kaldi_povey = torch.hann_window(400, periodic=False).pow(0.85).numpy()

    _write("s3_mel_filters.f32", s3_filters)
    _write("s3_hann400.f32", s3_window)
    _write("matcha_mel_filters.f32", matcha_filters)
    _write("matcha_hann1920.f32", matcha_hann)
    _write("voiceenc_mel_filters.f32", voiceenc_filters)
    _write("voiceenc_hann400.f32", voiceenc_hann)
    _write("kaldi_mel_filters.f32", kaldi_filters)
    _write("kaldi_povey400.f32", kaldi_povey)

    # Shape *and* where the numbers came from. The tables ship in all five
    # packages and NOTICE names their upstreams (librosa, PyTorch, SciPy, Kaldi
    # via torchaudio, Chatterbox); librosa and Kaldi ask that attribution travel
    # with the values, and a bare shape does not carry it. Written here so a
    # regenerated manifest cannot quietly drop the provenance again.
    sources = {
        "s3_mel_filters.f32": (
            "Chatterbox (MIT)",
            "S3 tokenizer mel filterbank, read from the upstream checkpoint",
        ),
        "s3_hann400.f32": ("PyTorch (BSD-3-Clause)", "torch.hann_window(400)"),
        "matcha_mel_filters.f32": (
            "librosa (ISC)",
            "librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000)",
        ),
        "matcha_hann1920.f32": ("PyTorch (BSD-3-Clause)", "torch.hann_window(1920)"),
        "voiceenc_mel_filters.f32": (
            "librosa (ISC)",
            "librosa.filters.mel(sr=16000, n_fft=400, n_mels=40, fmin=0, fmax=8000)",
        ),
        "voiceenc_hann400.f32": (
            "SciPy (BSD-3-Clause)",
            "scipy.signal.get_window('hann', 400, fftbins=False)",
        ),
        "kaldi_mel_filters.f32": (
            "Kaldi via torchaudio (Apache-2.0)",
            "torchaudio.compliance.kaldi.get_mel_banks("
            "80, 512, 16000, 20, 8000, 100, -500, 1.0)",
        ),
        "kaldi_povey400.f32": (
            "PyTorch (BSD-3-Clause)",
            "torch.hann_window(400, periodic=False) ** 0.85",
        ),
    }
    shapes = {
        "s3_mel_filters.f32": list(s3_filters.shape),
        "s3_hann400.f32": list(s3_window.shape),
        "matcha_mel_filters.f32": list(matcha_filters.shape),
        "matcha_hann1920.f32": list(matcha_hann.shape),
        "voiceenc_mel_filters.f32": list(voiceenc_filters.shape),
        "voiceenc_hann400.f32": list(voiceenc_hann.shape),
        "kaldi_mel_filters.f32": list(kaldi_filters.shape),
        "kaldi_povey400.f32": list(kaldi_povey.shape),
    }
    manifest = {
        "about": (
            "Precomputed DSP tables, so five implementations multiply the same "
            "numbers. Generated by tools/gen_dsp_assets.py; see NOTICE, section "
            "'DSP filterbank and window tables', for the licences these values "
            "carry."
        ),
        "tables": {
            name: {
                "shape": shape,
                "source": sources[name][0],
                "generated_by": sources[name][1],
            }
            for name, shape in shapes.items()
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"dsp tables written to {OUT}")


if __name__ == "__main__":
    main()
