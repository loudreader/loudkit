"""The clone input contract, stated once and enforced before any model runs.

The prompt is built from the first 10 seconds of the reference and the speaker
embedding reads the whole clip. A five-minute recording therefore produced a
voice mostly shaped by audio the docs said was ignored. The preflight makes
the contract enforceable: 5 to 10 seconds recommended, more than 30 refused,
silence refused, NaN or Inf refused, and a clip too short to enroll from
refused — each with a message that says what a good input looks like.

These tests call :func:`loudkit.models.enroll.validate_reference_audio`
directly (no weights, no models) and, once, through ``TorchVoiceEnroller``
with untrained modules to prove the enroller runs the same preflight.
"""

from __future__ import annotations

import numpy as np
import pytest

from loudkit.models.enroll import validate_reference_audio

_SR = 24_000


def _speech_like(seconds: float, sr: int = _SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(0)
    return (0.5 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.normal(size=len(t))).astype(
        np.float32
    )


class TestGoodInputPasses:
    @pytest.mark.parametrize("seconds", [1.5, 5.0, 10.0, 29.9])
    def test_clean_audio_within_the_contract_is_accepted(self, seconds: float) -> None:
        validate_reference_audio(_speech_like(seconds), _SR)

    def test_the_recommended_length_is_squarely_inside(self) -> None:
        """5 to 10 seconds is what every message recommends; it must never be
        near either refusal line."""
        for seconds in (5.0, 7.5, 10.0):
            validate_reference_audio(_speech_like(seconds), _SR)


class TestRefusals:
    def test_audio_longer_than_thirty_seconds_is_refused(self) -> None:
        with pytest.raises(ValueError, match="30") as excinfo:
            validate_reference_audio(_speech_like(31.0), _SR)
        # The message states the contract, not just the refusal.
        assert "5 to 10 seconds" in str(excinfo.value)
        assert "first 10 s" in str(excinfo.value)

    def test_a_clip_too_short_to_enroll_from_is_refused(self) -> None:
        with pytest.raises(ValueError, match="too short") as excinfo:
            validate_reference_audio(_speech_like(0.5), _SR)
        assert "5 to 10 seconds" in str(excinfo.value)

    def test_an_empty_clip_is_refused(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            validate_reference_audio(np.zeros(0, np.float32), _SR)

    def test_silence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="silent") as excinfo:
            validate_reference_audio(np.zeros(5 * _SR, np.float32), _SR)
        assert "5 to 10 seconds" in str(excinfo.value)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_samples_are_refused(self, bad: float) -> None:
        clip = _speech_like(5.0)
        clip[1234] = bad
        with pytest.raises(ValueError, match="NaN or Inf") as excinfo:
            validate_reference_audio(clip, _SR)
        assert "5 to 10 seconds" in str(excinfo.value)

    def test_one_nan_in_a_long_clip_does_not_report_length_first(self) -> None:
        """Finiteness is judged before any statistic, so the NaN is what gets
        named — a length complaint about a poisoned clip would send the user
        trimming a file that will fail again."""
        clip = _speech_like(31.0)
        clip[0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_reference_audio(clip, _SR)

    def test_a_non_positive_sample_rate_keeps_its_sentence(self) -> None:
        """Go's sentence, shared by all four implementations."""
        with pytest.raises(ValueError, match="sample rate must be positive"):
            validate_reference_audio(_speech_like(5.0), 0)

    def test_non_mono_audio_keeps_its_sentence(self) -> None:
        with pytest.raises(ValueError, match="mono"):
            validate_reference_audio(np.zeros((2, _SR), np.float32), _SR)


class TestTheEnrollerRunsThePreflight:
    def test_enroll_refuses_before_touching_any_model(self) -> None:
        """The refusal must come from the preflight, not from a model choking:
        untrained modules on the CPU would happily process 31 seconds."""
        import torch

        from loudkit.models.enroll import TorchVoiceEnroller, _CAMPPlus, _S3Tokenizer

        enroller = TorchVoiceEnroller(
            _S3Tokenizer().eval(),
            _CAMPPlus().eval(),
            None,
            device=torch.device("cpu"),
        )
        with pytest.raises(ValueError, match="30"):
            enroller.enroll(_speech_like(31.0), _SR)
        with pytest.raises(ValueError, match="silent"):
            enroller.enroll(np.zeros(5 * _SR, np.float32), _SR)
