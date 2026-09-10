"""`lk.enroll(path)` hands the enroller the file's own samples and rate.

The enroller resamples with the one Hann-sinc law every port ships. A second
resampler in front of it (the librosa/soxr read this path used to do) gave a
44.1 or 48 kHz recording a different profile from Python than from any port.
No weights: the enroller is a recorder.
"""

from __future__ import annotations

import numpy as np
import pytest

import loudkit

from .conftest import fake_voice

pytest.importorskip("soundfile")


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, int]] = []

    def enroll(self, samples: np.ndarray, sample_rate: int, *, name: str = "") -> object:
        self.calls.append((np.asarray(samples), sample_rate))
        return fake_voice()


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    from loudkit import hub
    from loudkit.backends import torch_backend

    rec = _Recorder()
    monkeypatch.setattr(loudkit, "_require_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(torch_backend, "build_torch_enroller", lambda *_a, **_k: rec)
    monkeypatch.setattr(hub, "resolve_enrollment_checkpoint", lambda *_a, **_k: tmp_path / "e")
    monkeypatch.setattr(hub, "resolve_voice_encoder", lambda *_a, **_k: tmp_path / "ve")
    return rec


def test_a_file_is_read_at_its_native_rate(recorder, tmp_path) -> None:
    import soundfile as sf

    rng = np.random.default_rng(1)
    wav = (0.5 * rng.standard_normal(48_000)).astype(np.float32)
    path = tmp_path / "me.wav"
    sf.write(str(path), wav, 48_000, subtype="FLOAT")

    loudkit.enroll(str(path), str(tmp_path), name="me")

    ((samples, rate),) = recorder.calls
    assert rate == 48_000, "the rate must reach the enroller; it resamples, not the reader"
    assert samples.dtype == np.float32
    np.testing.assert_array_equal(samples, wav)


def test_stereo_is_averaged_not_dropped(recorder, tmp_path) -> None:
    import soundfile as sf

    left = np.full(2400, 0.5, np.float32)
    right = np.full(2400, -0.25, np.float32)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), np.stack([left, right], axis=1), 24_000, subtype="FLOAT")

    loudkit.enroll(str(path), str(tmp_path), name="me")

    ((samples, rate),) = recorder.calls
    assert rate == 24_000
    np.testing.assert_allclose(samples, np.full(2400, 0.125, np.float32))


def test_samples_are_taken_at_the_rate_the_caller_names(recorder, tmp_path) -> None:
    wav = np.zeros(16_000, np.float32)
    loudkit.enroll(wav, str(tmp_path), name="me", sample_rate=16_000)
    ((_, rate),) = recorder.calls
    assert rate == 16_000


def test_a_missing_file_is_named(tmp_path) -> None:
    with pytest.raises(loudkit.AudioNotFoundError, match="nope.wav"):
        loudkit.enroll(str(tmp_path / "nope.wav"), str(tmp_path), name="me")
