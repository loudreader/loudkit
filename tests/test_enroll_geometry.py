"""The torch-free enrollment frontend's numerology, stated once.

``backends/graph_enroll.py`` is a NumPy mirror of ``models/enroll.py``, which
is itself a mirror of three upstream frontends. Its window sizes, hops, FFT
lengths and filter counts used to be bare literals: a value could be retuned by
one digit and nothing but an asset-backed fixture would notice, and those skip
on a machine without the checkpoint.

So these run without any asset. They pin the numbers to what the graphs were
traced against, and pin the mirror to the original wherever the original names
the same quantity, which is the pairing a rename could otherwise break in one
file only.
"""

from __future__ import annotations

import pytest

from loudkit.backends import graph_enroll as dsp


class TestTheRatesAndCuts:
    def test_the_two_sample_rates(self) -> None:
        assert dsp._S3_SR == 16_000
        assert dsp._MEL_SR == 24_000

    def test_the_prompt_is_ten_seconds_and_the_conditioning_six(self) -> None:
        assert dsp._PROMPT_SAMPLES == 10 * dsp._MEL_SR
        assert dsp._COND_SAMPLES == 6 * dsp._S3_SR
        # The same six seconds, counted in tokenizer mel frames.
        assert dsp._COND_MEL_FRAMES == dsp._COND_SAMPLES // dsp._S3_HOP


class TestTheThreeSpectrogramGeometries:
    def test_the_tokenizer_mel_is_whispers(self) -> None:
        assert (dsp._S3_WIDTH, dsp._S3_HOP, dsp._S3_PAD) == (400, 160, 200)
        assert dsp._S3_MELS == 128
        # Centring, so the pad is half the window.
        assert dsp._S3_PAD == dsp._S3_WIDTH // 2

    def test_the_prompt_mel_is_matchas(self) -> None:
        assert (dsp._FLOW_WIDTH, dsp._FLOW_HOP, dsp._FLOW_PAD) == (1920, 480, 720)
        assert dsp._FLOW_MELS == 80
        assert dsp._FLOW_PAD == (dsp._FLOW_WIDTH - dsp._FLOW_HOP) // 2

    def test_the_fbank_is_kaldis(self) -> None:
        # 25 ms framed, shifted 10 ms, at 16 kHz.
        assert dsp._KALDI_WIDTH == dsp._S3_SR * 25 // 1000
        assert dsp._KALDI_HOP == dsp._S3_SR * 10 // 1000
        assert dsp._KALDI_NFFT == 512
        assert dsp._KALDI_WIDTH <= dsp._KALDI_NFFT < 2 * dsp._KALDI_WIDTH
        assert dsp._KALDI_MELS == 80
        assert dsp._KALDI_PREEMPHASIS == 0.97


class TestThePartialGeometry:
    """160, 77 and 83 are one geometry seen from three sides."""

    def test_the_step_is_the_rate_the_encoder_was_trained_at(self) -> None:
        assert dsp._PARTIAL_FRAMES == 160
        assert dsp._VOICEENC_MELS == 40
        assert dsp._PARTIAL_STEP == round((dsp._S3_SR / 1.3) / dsp._PARTIAL_FRAMES) == 77
        # The 83 this file used to carry as its own number.
        assert dsp._PARTIAL_FRAMES - dsp._PARTIAL_STEP == 83
        assert dsp._PARTIAL_MIN_COVERAGE == 0.8

    def test_the_trim_is_librosas_defaults_at_twenty_decibels(self) -> None:
        assert (dsp._TRIM_WIDTH, dsp._TRIM_HOP) == (2048, 512)
        assert dsp._TRIM_PAD == dsp._TRIM_WIDTH // 2
        assert pytest.approx(10 ** (-20 / 20)) == dsp._TRIM_THRESHOLD


class TestTheMirrorAgreesWithTheOriginal:
    """The pairs both files name. A rename in one is a divergence in both."""

    def test_the_voice_encoder_partials_match(self) -> None:
        pytest.importorskip("torch")
        from loudkit.models.enroll import _VoiceEncoder

        assert dsp._PARTIAL_FRAMES == _VoiceEncoder.PARTIAL_FRAMES
        assert dsp._VOICEENC_MELS == _VoiceEncoder.NUM_MELS

    def test_the_sample_rates_match(self) -> None:
        pytest.importorskip("torch")
        from loudkit.models import enroll

        assert dsp._S3_SR == enroll._S3_SR
        assert dsp._MEL_SR == enroll._MEL_SR
        assert int(enroll._MAX_REF_SECONDS * enroll._MEL_SR) == dsp._PROMPT_SAMPLES


def test_the_coreml_tokenizer_emits_eight_ternary_digits() -> None:
    """Base 3 to the eighth covers the 6561-id speech vocabulary exactly."""
    assert dsp._COREML_TOKEN_DIGITS == 8
    assert 3**dsp._COREML_TOKEN_DIGITS == 6561
