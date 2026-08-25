"""The exported enrollment graphs reproduce the fixture, stage by stage.

The enrollment fixture (``tests/data/enrollment``) is the spec every port is
held to. These tests run the three exported ONNX graphs against the fixture's
*real* DSP inputs — not synthetic probes — and require the same agreement the
ports will have to show:

* the tokenizer reproduces the prompt tokens **exactly** (they are discrete,
  so close is not a thing);
* the two encoders reproduce the shipped embeddings to cosine > 0.9999, the
  tolerance the embeddings are used at (both are directions, L2-normalised
  downstream, so what matters is where they point).

This is the foundation check: if these pass, the ONNX graphs are faithful
substitutes for torch, and a port's remaining job is its own DSP and
orchestration — which the fixture's DSP files pin separately.

Skips when onnxruntime or the exported graphs are absent, with a named reason;
``LOUDKIT_REQUIRE_ASSETS=1`` turns that into a failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from .assets import asset, needs_module, requires, skip_or_fail

CKPT = asset("checkpoint")
FIXTURE = Path(__file__).parent / "data" / "enrollment"

_COS_GATE = 0.9999

pytestmark = [
    pytest.mark.slow,
    requires("checkpoint"),
]


def _graphs_dir() -> Path:
    """The directory holding the three enrollment graphs, or a named skip."""
    import os

    if env := os.environ.get("LOUDKIT_ENROLL_ONNX"):
        return Path(env)
    candidate = CKPT.parent / "onnx"
    names = ("s3_tokenizer.onnx", "camp.onnx", "voice_encoder.onnx")
    if all((candidate / n).exists() for n in names):
        return candidate
    return skip_or_fail(
        f"enrollment ONNX graphs not found in {candidate} (set LOUDKIT_ENROLL_ONNX); "
        "run tools/export_enroll_onnx.py to create them"
    )


def _read(name: str, dtype: np.dtype) -> np.ndarray:
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    shape = manifest["files"][name]["shape"]
    raw = (FIXTURE / name).read_bytes()
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _session(name: str):
    needs_module("onnxruntime")
    import onnxruntime as ort

    return ort.InferenceSession(str(_graphs_dir() / name), providers=["CPUExecutionProvider"])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _voice_encoder_partials(mel: np.ndarray, *, step: int, window: int) -> np.ndarray:
    """The 1.6 s partial windows the utterance voice encoder reads.

    Mirrors ``_VoiceEncoder.embed``: partials strided at ``step`` frames, the
    mel zero-padded so the last partial is full. This is the orchestration the
    ports reimplement; the test reconstructs it to feed the graph the fixture's
    own mel, so the graph is checked against real data rather than a probe.
    """
    n_wins, remainder = divmod(max(len(mel) - window + step, 0), step)
    if n_wins == 0 or (remainder + (window - step)) / window >= 0.8:
        n_wins += 1
    target = window + step * (n_wins - 1)
    if target > len(mel):
        mel = np.concatenate([mel, np.zeros((target - len(mel), mel.shape[1]), np.float32)])
    return np.stack([mel[i * step : i * step + window] for i in range(n_wins)])


class TestTokenizerGraph:
    def test_prompt_tokens_match_exactly(self) -> None:
        sess = _session("s3_tokenizer.onnx")
        mel = _read("tokenizer_mel.f32", np.float32)
        got = sess.run(None, {"mel": mel[None].astype(np.float32)})[0].astype(np.int64)
        want = _read("prompt_tokens.i64", np.int64)
        np.testing.assert_array_equal(got, want)

    def test_cond_tokens_match_exactly(self) -> None:
        sess = _session("s3_tokenizer.onnx")
        # The conditioning prompt is a different resample: librosa, truncated
        # to 6 s — its own mel, not the prompt path's.
        mel = _read("tokenizer_mel_cond.f32", np.float32)
        got = sess.run(None, {"mel": mel[None].astype(np.float32)})[0].astype(np.int64)
        want = _read("cond_prompt_tokens.i64", np.int64)
        np.testing.assert_array_equal(got, want)


class TestEncoderGraphs:
    def test_flow_embedding_matches(self) -> None:
        sess = _session("camp.onnx")
        fbank = _read("kaldi_fbank.f32", np.float32)
        got = sess.run(None, {"fbank": fbank.T[None].astype(np.float32)})[0].astype(np.float32)
        want = _read("flow_embedding.f32", np.float32)
        assert _cos(got.ravel(), want) > _COS_GATE

    def test_speaker_embedding_matches(self) -> None:
        sess = _session("voice_encoder.onnx")
        mel = _read("voiceenc_mel.f32", np.float32)
        partials = _voice_encoder_partials(mel, step=77, window=160)
        got = sess.run(None, {"partials": partials.astype(np.float32)})[0].astype(np.float32)
        pooled = got.mean(0)
        pooled = pooled / np.linalg.norm(pooled)
        want = _read("speaker_embedding.f32", np.float32)
        assert _cos(pooled, want) > _COS_GATE
