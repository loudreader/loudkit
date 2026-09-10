"""Enrollment adapters shared by the ONNX and CoreML enrollment exporters.

Two of the three wrappers are the same in both backends, and so is the load of
the three torch models they wrap. Only the tokenizer differs: the ONNX graph
folds the FSQ codes to token ids in-graph, the CoreML package returns the eight
float dims and the host folds them, so each exporter keeps its own tokenizer
wrapper. Each also keeps its own conversion and parity gates.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import nn

from loudkit.checkpoint import Checkpoint
from loudkit.models.enroll import _CAMPPlus, _S3Tokenizer, _VoiceEncoder


def enrollment_parser(description: str | None, backend: str) -> argparse.ArgumentParser:
    """The command line both enrollment exporters take, spelled once."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--voice-encoder", required=True, help="ve.safetensors")
    ap.add_argument("--out", default=None, help=f"output dir (default: <ckpt dir>/{backend})")
    return ap


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


def load_spk_weights(ckpt: Checkpoint) -> dict[str, torch.Tensor]:
    """CAM++ weights off the packed checkpoint, as torch tensors."""
    tensors = ckpt.tensors("s3gen.speaker_encoder.")
    return {k: torch.from_numpy(v.copy()) for k, v in tensors.items()}


def load_enrollment_models(
    ckpt: Checkpoint, voice_encoder: str
) -> tuple[_S3Tokenizer, _CAMPPlus, _VoiceEncoder]:
    """The three torch models each gate compares its graph against.

    The voice encoder's weights are not in the packed checkpoint, so its file
    is named separately; the other two come off the checkpoint being exported,
    which is what makes the gate a comparison against *this* checkpoint.
    """
    from safetensors.torch import load_file

    tok = _S3Tokenizer()
    tok.load_state_dict(
        {k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors("s3gen.tokenizer.").items()}
    )

    spk = _CAMPPlus()
    spk.load_state_dict(load_spk_weights(ckpt))

    ve = _VoiceEncoder()
    ve.load_state_dict(load_file(voice_encoder))

    return tok.float().eval(), spk.float().eval(), ve.float().eval()
