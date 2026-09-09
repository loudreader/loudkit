"""The CoreML renderer is the ONNX renderer with its session calls replaced.

The two used to be separate copies of the same framing, the same speaker
affine, the same Euler loop and the same 510-frame vocoder pad. They are now
one implementation with four seams, so this file states what the seams are
allowed to change: the token dtype the exported artifacts declare, the names
each stage's feed carries, and nothing else. Anything the seams do *not* change
is asserted here as byte equality between the two paths, because a renderer
that drifts produces plausible audio rather than an error.

No asset is loaded. The stages are stubs that record their feed and hand back a
fixed array, which is what makes "the same window through both paths" a
comparison of loudkit's arithmetic rather than of two exported graphs.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from loudkit.backends.onnx_backend import (
    ONNXMelDecoder,
    ONNXVocoder,
    _require_static_window,
)
from loudkit.config import DEFAULT_ALGORITHM, AlgorithmConfig
from loudkit.contracts import MEL_BINS
from loudkit.voice import VoiceProfile

coreml_backend = pytest.importorskip("loudkit.backends.coreml_backend")

STATIC = replace(
    DEFAULT_ALGORITHM,
    window=replace(
        DEFAULT_ALGORITHM.window,
        static_length=255,
        static_prompt_tokens=238,
        pad_token_id=4254,
    ),
)
"""The framing both exports are static at."""

RAGGED = DEFAULT_ALGORITHM
"""No static window: the ONNX decoder halves the row, CoreML refuses."""

SEED = 1234
TOKENS = [int(t) for t in np.random.default_rng(11).integers(0, 6561, size=64)]


def _voice() -> VoiceProfile:
    rng = np.random.default_rng(0)
    return VoiceProfile(
        name="seam",
        speaker_embedding=rng.normal(size=256).astype(np.float32),
        flow_embedding=rng.normal(size=192).astype(np.float32),
        prompt_tokens=rng.integers(0, 6561, size=180).astype(np.int64),
        prompt_mel=rng.normal(size=(MEL_BINS, 300)).astype(np.float32),
        cond_prompt_tokens=rng.integers(0, 6561, size=150).astype(np.int64),
    )


def _affine() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    return (
        rng.normal(size=(MEL_BINS, 192)).astype(np.float32),
        rng.normal(size=MEL_BINS).astype(np.float32),
    )


class _Stage:
    """One renderer stage, under either calling convention.

    Returns the same array whatever it is fed, so a difference in the two
    outputs can only come from loudkit's own arithmetic. What it *was* fed is
    kept, because the dtypes and the feed names are the seam under test.
    """

    def __init__(self, reply: np.ndarray) -> None:
        self.reply = reply
        self.positional: list[list[np.ndarray]] = []
        self.named: list[dict[str, np.ndarray]] = []

    def run_positional(self, values: Any) -> list[np.ndarray]:
        self.positional.append([np.asarray(v) for v in values])
        return [self.reply]

    def predict(self, data: Any) -> dict[str, np.ndarray]:
        self.named.append({k: np.asarray(v) for k, v in data.items()})
        return {"out": self.reply}


def _reply(config: AlgorithmConfig, voice: VoiceProfile) -> np.ndarray:
    w = config.window
    row = (w.static_length or len(TOKENS)) + (
        w.static_prompt_tokens or len(voice.prompt_tokens)
    )
    return np.random.default_rng(5).normal(size=MEL_BINS * 2 * row).astype(np.float32)


def _onnx_decoder(config: AlgorithmConfig, stage: _Stage) -> ONNXMelDecoder:
    """An ONNXMelDecoder whose two sessions are the stub, built without assets."""
    decoder = ONNXMelDecoder.__new__(ONNXMelDecoder)
    decoder.config = config
    decoder._encoder = stage
    decoder._estimator = stage
    decoder.attach_speaker_affine(*_affine())
    return decoder


def _coreml_decoder(config: AlgorithmConfig, stage: _Stage) -> Any:
    decoder = coreml_backend.CoreMLMelDecoder(config, encoder=stage, estimator=stage)
    decoder.attach_speaker_affine(*_affine())
    return decoder


class TestTheTwoRenderersAgree:
    def test_the_same_window_decodes_to_the_same_bytes(self) -> None:
        """The one assertion the merge exists to keep true."""
        voice = _voice()
        reply = _reply(STATIC, voice)
        onnx_stage, coreml_stage = _Stage(reply), _Stage(reply)

        onnx_mel = _onnx_decoder(STATIC, onnx_stage).decode(TOKENS, voice, seed=SEED)
        coreml_mel = _coreml_decoder(STATIC, coreml_stage).decode(TOKENS, voice, seed=SEED)

        assert onnx_mel.dtype == coreml_mel.dtype == np.float32
        assert onnx_mel.tobytes() == coreml_mel.tobytes()

    def test_the_same_mel_synthesises_to_the_same_bytes(self) -> None:
        voice = _voice()
        frames = 2 * STATIC.window.max_speech_tokens
        wav = np.random.default_rng(9).normal(size=frames * 480).astype(np.float32)
        mel = np.random.default_rng(13).normal(size=(MEL_BINS, 96)).astype(np.float32)
        onnx_stage, coreml_stage = _Stage(wav), _Stage(wav)

        vocoder = ONNXVocoder.__new__(ONNXVocoder)
        vocoder.config = STATIC
        vocoder._hift = onnx_stage
        onnx_wav = vocoder.synthesize(mel, voice, seed=SEED)
        coreml_wav = coreml_backend.CoreMLVocoder(STATIC, coreml_stage).synthesize(
            mel, voice, seed=SEED
        )

        assert onnx_wav.tobytes() == coreml_wav.tobytes()
        # Both padded to the static window and cropped back to the real mel.
        assert len(onnx_wav) == 96 * 480
        assert onnx_stage.positional[0][0].shape == (1, MEL_BINS, frames)


class TestWhatTheSeamsMayChange:
    """The differences the merge had to keep, named one at a time."""

    def test_the_token_dtype_is_the_one_each_export_declares(self) -> None:
        """int64 graphs, int32 packages. The only dtype the seam moves."""
        voice = _voice()
        reply = _reply(STATIC, voice)
        onnx_stage, coreml_stage = _Stage(reply), _Stage(reply)
        _onnx_decoder(STATIC, onnx_stage).decode(TOKENS, voice, seed=SEED)
        _coreml_decoder(STATIC, coreml_stage).decode(TOKENS, voice, seed=SEED)

        prompt, query = onnx_stage.positional[0]
        assert prompt.dtype == query.dtype == np.int64
        named = coreml_stage.named[0]
        assert named["prompt_token"].dtype == named["speech_tokens"].dtype == np.int32
        # Same numbers under either dtype: only the storage differs.
        assert (prompt == named["prompt_token"]).all()
        assert (query == named["speech_tokens"]).all()
        assert ONNXMelDecoder._TOKEN_DTYPE is np.int64
        assert coreml_backend.CoreMLMelDecoder._TOKEN_DTYPE is np.int32

    def test_the_packages_are_fed_by_name(self) -> None:
        voice = _voice()
        stage = _Stage(_reply(STATIC, voice))
        _coreml_decoder(STATIC, stage).decode(TOKENS, voice, seed=SEED)

        assert list(stage.named[0]) == ["prompt_token", "speech_tokens"]
        assert list(stage.named[1]) == ["x", "mu", "t", "spks", "cond"]

        wav = np.random.default_rng(9).normal(size=510 * 480).astype(np.float32)
        hift = _Stage(wav)
        mel = np.zeros((MEL_BINS, 8), dtype=np.float32)
        coreml_backend.CoreMLVocoder(STATIC, hift).synthesize(mel, voice, seed=SEED)
        assert list(hift.named[0]) == ["mel", "phase", "noise"]

    def test_a_ragged_window_is_halved_on_onnx_and_refused_on_coreml(self) -> None:
        """The packages are traced at one geometry; the graphs are not."""
        voice = _voice()
        stage = _Stage(_reply(RAGGED, voice))
        _onnx_decoder(RAGGED, stage).decode(TOKENS, voice, seed=SEED)
        prompt, query = stage.positional[0]
        # Half the ragged row, which is not where the prompt actually ends.
        assert prompt.shape[1] == (len(voice.prompt_tokens) + len(TOKENS)) // 2
        assert prompt.shape[1] + query.shape[1] == len(voice.prompt_tokens) + len(TOKENS)

        with pytest.raises(ValueError, match="static at query 255"):
            _coreml_decoder(RAGGED, stage).decode(TOKENS, voice, seed=SEED)
        with pytest.raises(ValueError, match="static at query 255"):
            coreml_backend.CoreMLVocoder(RAGGED, stage)

    def test_each_refusal_names_its_own_export(self) -> None:
        """One message, two words: a user must not be sent to re-export ONNX
        graphs because a CoreML package was the one framed wrong."""
        with pytest.raises(ValueError, match="the exported ONNX graphs"):
            _require_static_window(RAGGED)
        with pytest.raises(ValueError, match="the exported CoreML graphs"):
            coreml_backend._require_static_window(RAGGED)

    def test_the_builder_named_in_the_affine_refusal_is_the_right_one(self) -> None:
        stage = _Stage(np.zeros(1, dtype=np.float32))
        onnx = ONNXMelDecoder.__new__(ONNXMelDecoder)
        onnx.config = STATIC
        onnx._spk_weight = onnx._spk_bias = None
        with pytest.raises(RuntimeError, match="build_onnx_engine"):
            onnx.decode(TOKENS, _voice(), seed=SEED)
        with pytest.raises(RuntimeError, match="build_coreml_engine"):
            coreml_backend.CoreMLMelDecoder(STATIC, encoder=stage, estimator=stage).decode(
                TOKENS, _voice(), seed=SEED
            )


class TestTheClassesAreOneImplementation:
    def test_the_coreml_renderers_subclass_the_onnx_ones(self) -> None:
        """Stated as a test because the alternative -- two copies that happen to
        agree today -- is what this file was written to end."""
        assert issubclass(coreml_backend.CoreMLMelDecoder, ONNXMelDecoder)
        assert issubclass(coreml_backend.CoreMLVocoder, ONNXVocoder)

    def test_only_the_seams_are_overridden(self) -> None:
        seams = {"__init__", "_prompt_length", "_encode", "_estimate", "_vocode"}
        for cls in (coreml_backend.CoreMLMelDecoder, coreml_backend.CoreMLVocoder):
            overridden = {
                name
                for name, value in vars(cls).items()
                if inspect.isfunction(value) and not name.startswith("__")
            }
            assert overridden <= seams, f"{cls.__name__} overrides more than the seams"

    def test_decode_and_synthesize_are_not_reimplemented(self) -> None:
        assert coreml_backend.CoreMLMelDecoder.decode is ONNXMelDecoder.decode, (
            "the CoreML decode loop is a copy again"
        )
        assert coreml_backend.CoreMLVocoder.synthesize is ONNXVocoder.synthesize, (
            "the CoreML vocoder pad is a copy again"
        )
