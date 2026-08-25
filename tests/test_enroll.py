"""Enrollment — determinism, validation and the missing-encoder error, without weights.

The parity suite already proves that a real enrollment reproduces the
reference voice bit for bit (``test_parity.py::TestEnrollmentParity``); that
needs the checkpoint, the voice encoder and a reference clip. What these tests
pin without any of that is the *contract*:

* enrollment is deterministic — no sampling anywhere, so the same clip always
  yields the same profile, which is what makes comparing profiles across
  machines meaningful at all;
* the profile it returns carries every field the renderer reads, with the
  shapes the renderer expects;
* the validation and error paths are real: non-mono audio is refused, and
  building an enroller without the 256-d utterance encoder fails with an error
  that names the argument to pass.

The modules are randomly initialised here — the tests assert the pipeline
shape and determinism, never the quality of the embedding.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from loudkit.models.enroll import TorchVoiceEnroller, _CAMPPlus, _S3Tokenizer, _VoiceEncoder

pytest.importorskip("torchaudio")
pytest.importorskip("librosa")


@pytest.fixture
def enroller() -> TorchVoiceEnroller:
    import torch

    return TorchVoiceEnroller(
        _S3Tokenizer().eval(),
        _CAMPPlus().eval(),
        _VoiceEncoder().eval(),
        device=torch.device("cpu"),
    )


@pytest.fixture
def clip() -> np.ndarray:
    """1.5 s of 220 Hz tone plus low noise at 24 kHz — enough signal for the
    FSMN tokenizer and both encoders to produce finite, non-degenerate
    features."""
    sr = 24_000
    t = np.arange(int(1.5 * sr)) / sr
    rng = np.random.default_rng(0)
    return (0.5 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.normal(size=len(t))).astype(
        np.float32
    )


class TestDeterminism:
    def test_same_clip_same_profile(self, enroller, clip) -> None:  # type: ignore[no-untyped-def]
        a = enroller.enroll(clip, 24_000, name="voice")
        b = enroller.enroll(clip, 24_000, name="voice")
        for field in (
            "speaker_embedding",
            "flow_embedding",
            "prompt_tokens",
            "prompt_mel",
            "cond_prompt_tokens",
        ):
            assert np.array_equal(getattr(a, field), getattr(b, field)), field


class TestProfileShape:
    def test_profile_carries_every_renderer_field(self, enroller, clip) -> None:  # type: ignore[no-untyped-def]
        profile = enroller.enroll(clip, 24_000, name="voice")
        assert profile.name == "voice"
        assert profile.speaker_embedding.shape == (256,)
        assert profile.flow_embedding.shape == (192,)
        assert profile.prompt_tokens.ndim == 1
        assert profile.cond_prompt_tokens.ndim == 1
        assert profile.prompt_mel.ndim == 2
        assert profile.prompt_mel.shape[0] == 80
        # tokens and mel frames stay aligned at exactly 2 frames per token
        assert profile.prompt_mel.shape[1] == 2 * len(profile.prompt_tokens)

    def test_profile_is_small_enough_to_be_a_file(self, enroller, clip) -> None:  # type: ignore[no-untyped-def]
        profile = enroller.enroll(clip, 24_000)
        assert profile.n_bytes < 1_000_000, "a voice is a file, not a model"


class TestValidation:
    def test_non_mono_audio_is_refused(self, enroller) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="mono"):
            enroller.enroll(np.zeros((2, 100), np.float32), 24_000)

    def test_enrollment_without_voice_encoder_names_the_argument(self, clip) -> None:  # type: ignore[no-untyped-def]
        import torch

        enroller = TorchVoiceEnroller(
            _S3Tokenizer().eval(),
            _CAMPPlus().eval(),
            None,
            device=torch.device("cpu"),
        )
        with pytest.raises(RuntimeError, match="voice_encoder_weights"):
            enroller.enroll(clip, 24_000)

    def test_source_sample_rate_is_recorded(self, enroller, clip) -> None:  # type: ignore[no-untyped-def]
        profile = enroller.enroll(clip, 16_000)
        assert profile.source_sample_rate == 16_000

    def test_read_only_audio_never_reaches_torch(self, enroller, clip) -> None:  # type: ignore[no-untyped-def]
        read_only = clip.copy()
        read_only.setflags(write=False)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message="The given NumPy array is not writable",
                category=UserWarning,
            )
            profile = enroller.enroll(read_only, 24_000)

        assert profile.prompt_mel.shape[0] == 80
        assert not read_only.flags.writeable


class TestDevicePlacement:
    def test_voice_encoder_runs_on_its_own_device(self) -> None:
        """The partials tensor must follow the module, not assume the CPU.

        ``build_torch_enroller`` accepts any device and moves the encoder
        there, but ``embed`` built its input from numpy — always CPU — and fed
        it straight to the LSTM, so ``enroll()`` on CUDA or MPS died with a
        device mismatch. Every enrollment fixture is CPU-only, which is exactly
        why nothing caught it.

        Exercised on torch's ``meta`` device: it needs no accelerator, it is
        genuinely *not* the CPU, and it reproduces the original failure
        verbatim — "Input and parameter tensors are not at the same device".
        """
        import torch

        encoder = _VoiceEncoder().to("meta")
        out = encoder.embed(np.zeros(3 * 16_000, np.float32))
        assert out.device == torch.device("meta")
