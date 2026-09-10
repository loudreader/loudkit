"""NumPy DSP counterpart of models/enroll.py for torch-free graph enrollment.

The graph enrollment cases in tests/test_models.py pin tokens and embeddings
to the same enrollment fixture as the torch implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ..contracts import TOKEN_MEL_RATIO
from ..models.enrollment_audio import validate_reference_audio
from ..models.resample import resample
from ..voice import VoiceProfile

_TABLES = Path(__file__).parents[1] / "models/data/dsp"

# Mirrored from models/enroll.py, which mirrors the three upstream frontends
# the graphs were traced from. Derived where it is derived: a window and its
# padding are one fact and are written as one.
_S3_SR = 16_000  # Hz, what the tokenizer, CAM++ and the voice encoder read
_MEL_SR = 24_000  # Hz, what the flow's prompt mel is computed at
_PROMPT_SAMPLES = 10 * _MEL_SR  # the prompt cut, enroll.py's _MAX_REF_SECONDS
_COND_SAMPLES = 6 * _S3_SR  # the generator's conditioning window

# S3 tokenizer mel: Whisper's geometry, a 400-sample Hann hopped 160, centred.
_S3_WIDTH, _S3_HOP = 400, 160  # samples
_S3_PAD = _S3_WIDTH // 2  # centring pad, what torch.stft(center=True) adds
_S3_MELS = 128
_COND_MEL_FRAMES = _COND_SAMPLES // _S3_HOP  # those 6 s as tokenizer frames
_S3_POWER_FLOOR = 1e-10  # below this a bin is silence, before the log10
_S3_DECADES = 8.0  # kept below the peak; quieter bins clamp to the floor
_S3_LOG_SHIFT = _S3_LOG_SCALE = 4.0  # log10 -> roughly [-1, 1]

# Flow prompt mel: matcha's geometry, a 1920-sample Hann hopped 480, centred.
_FLOW_WIDTH, _FLOW_HOP = 1920, 480  # samples
_FLOW_PAD = (_FLOW_WIDTH - _FLOW_HOP) // 2  # matcha centres on the overlap
_FLOW_MELS = 80
_FLOW_MAGNITUDE_EPS = 1e-9  # inside the sqrt: a silent bin must stay finite
_FLOW_LOG_FLOOR = 1e-5

# CAM++ fbank: Kaldi's defaults at 16 kHz, a 25 ms frame shifted 10 ms.
_KALDI_WIDTH, _KALDI_HOP = 400, 160  # samples
_KALDI_NFFT = 512  # Kaldi rounds the window up to a power of two
_KALDI_MELS = 80
_KALDI_PREEMPHASIS = 0.97

# Voice encoder partials: 160-frame windows at 1.3 per second over a 40-mel
# power spectrogram, the Real-Time-Voice-Cloning recipe the weights saw.
_VOICEENC_MELS = 40
_PARTIAL_FRAMES = 160
_PARTIAL_STEP = 77  # round((_S3_SR / 1.3) / _PARTIAL_FRAMES)
_PARTIAL_MIN_COVERAGE = 0.8  # how much real mel a last, short window must hold

# What librosa.effects.trim(top_db=20) uses, mirrored because the torch path
# trims before it windows and the graphs were traced on trimmed audio.
_TRIM_WIDTH, _TRIM_HOP = 2048, 512  # samples
_TRIM_PAD = _TRIM_WIDTH // 2
_TRIM_THRESHOLD = 0.1  # 20 dB below the loudest frame

_COREML_TOKEN_DIGITS = 8  # that tokenizer returns base-3 digits, not an id


def _table(name: str, rows: int = 1) -> NDArray[np.float32]:
    return np.fromfile(_TABLES / (name + ".f32"), dtype="<f4").reshape(rows, -1)


def _frames(
    audio: NDArray[np.float32], width: int, hop: int, pad: int = 0
) -> NDArray[np.float32]:
    audio = np.pad(audio, (pad, pad), mode="reflect") if pad else audio
    return np.lib.stride_tricks.sliding_window_view(audio, width)[::hop]


def _spectra(
    audio: NDArray[np.float32], window: str, width: int, hop: int, pad: int
) -> NDArray[np.complex128]:
    frames = _frames(audio, width, hop, pad).astype(np.float64)
    spectra: NDArray[np.complex128] = np.fft.rfft(frames * _table(window), axis=-1).T
    return spectra


def _token_mel(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    spec = _spectra(audio, "s3_hann400", _S3_WIDTH, _S3_HOP, _S3_PAD)[:, :-1]
    mel = (_table("s3_mel_filters", _S3_MELS).astype(np.float64) @ (np.abs(spec) ** 2)).astype(
        np.float32
    )
    log = np.log10(np.maximum(mel, np.float32(_S3_POWER_FLOOR)))
    floored: NDArray[np.float32] = (
        np.maximum(log, log.max() - np.float32(_S3_DECADES)) + np.float32(_S3_LOG_SHIFT)
    ) / np.float32(_S3_LOG_SCALE)
    return floored


def _prompt_mel(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    spec = _spectra(audio, "matcha_hann1920", _FLOW_WIDTH, _FLOW_HOP, _FLOW_PAD)
    mel = (
        _table("matcha_mel_filters", _FLOW_MELS).astype(np.float64)
        @ np.sqrt(spec.real**2 + spec.imag**2 + _FLOW_MAGNITUDE_EPS)
    ).astype(np.float32)
    logged: NDArray[np.float32] = np.log(np.maximum(mel, np.float32(_FLOW_LOG_FLOOR)))
    return logged


def _fbank(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    frames = _frames(audio, _KALDI_WIDTH, _KALDI_HOP).astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    frames = frames - _KALDI_PREEMPHASIS * np.concatenate(
        (frames[:, :1], frames[:, :-1]), axis=1
    )
    spec = np.abs(np.fft.rfft(frames * _table("kaldi_povey400"), n=_KALDI_NFFT, axis=1)) ** 2
    filters = _table("kaldi_mel_filters", _KALDI_MELS)
    mel = (filters.astype(np.float64) @ spec[:, : filters.shape[1]].T).astype(np.float32)
    mel = np.log(np.maximum(mel, np.finfo(np.float32).eps))
    centred: NDArray[np.float32] = mel - mel.mean(
        axis=1, keepdims=True, dtype=np.float64
    ).astype(np.float32)
    return centred


def _partials(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    rms = np.sqrt(
        np.mean(
            _frames(np.pad(audio, (_TRIM_PAD, _TRIM_PAD)), _TRIM_WIDTH, _TRIM_HOP).astype(
                np.float64
            )
            ** 2,
            axis=1,
        )
    )
    active = np.flatnonzero(rms > _TRIM_THRESHOLD * rms.max())
    if len(active):
        audio = audio[active[0] * _TRIM_HOP : min((active[-1] + 1) * _TRIM_HOP, len(audio))]
    spec = _spectra(audio, "voiceenc_hann400", _S3_WIDTH, _S3_HOP, _S3_PAD)
    mel = (
        _table("voiceenc_mel_filters", _VOICEENC_MELS).astype(np.float64) @ np.abs(spec) ** 2
    ).T.astype(np.float32)
    # How many strided windows the mel affords, and whether the last, short one
    # covers enough to keep. The three numbers are one geometry: a window is
    # _PARTIAL_FRAMES long, the next starts _PARTIAL_STEP later, so the overlap
    # a short window inherits is the difference between them.
    overlap = _PARTIAL_FRAMES - _PARTIAL_STEP
    count, remainder = divmod(max(len(mel) - overlap, 0), _PARTIAL_STEP)
    if count == 0 or (remainder + overlap) / _PARTIAL_FRAMES >= _PARTIAL_MIN_COVERAGE:
        count += 1
    target = _PARTIAL_FRAMES + _PARTIAL_STEP * (count - 1)
    mel = np.pad(mel, ((0, max(0, target - len(mel))), (0, 0)))
    return np.stack(
        [mel[i * _PARTIAL_STEP : i * _PARTIAL_STEP + _PARTIAL_FRAMES] for i in range(count)]
    )


class _PositionalRunner(Protocol):
    """One enrollment graph as ONNX Runtime holds it: inputs by position."""

    def run_positional(
        self, values: Sequence[NDArray[np.generic]]
    ) -> list[NDArray[np.generic]]: ...


class _NamedRunner(Protocol):
    """One enrollment graph as CoreML holds it: inputs and outputs by name."""

    def predict(self, inputs: Mapping[str, NDArray[np.float32]]) -> Mapping[str, object]: ...


class GraphEnroller:
    """Enrollment graphs with the shared, fixture-checked NumPy audio frontend."""

    def __init__(self, assets: Path, device: str) -> None:
        self.models: Sequence[_PositionalRunner] | Sequence[_NamedRunner]
        self.device = device
        if device == "onnx":
            from ..config import ExecutionConfig
            from .onnx_backend import _load_session

            self.models = [
                _load_session(assets, name + ".onnx", ExecutionConfig(onnx_provider="cpu"))
                for name in ("s3_tokenizer", "camp", "voice_encoder")
            ]
        else:
            from .coreml_backend import _load_model

            self.models = [
                _load_model(assets / (name + ".mlpackage"), "CPU_ONLY")
                for name in ("s3_tokenizer", "camp", "voice_encoder")
            ]

    def _run(self, index: int, value: NDArray[np.float32]) -> NDArray[Any]:
        """One graph, one array in, its first output out.

        The dtype is the graph's: token ids from the tokenizer, embeddings from
        the other two, so only "a numpy array" is promised here.
        """
        # `device` chose which runner `__init__` built, and a string does not
        # narrow a union, so each branch names the half it is holding.
        value = np.asarray(value, dtype=np.float32)
        if self.device == "onnx":
            onnx = cast("Sequence[_PositionalRunner]", self.models)
            return onnx[index].run_positional([value])[0]
        coreml = cast("Sequence[_NamedRunner]", self.models)
        return cast(
            "NDArray[Any]",
            next(
                iter(
                    coreml[index].predict({("mel", "fbank", "partials")[index]: value}).values()
                )
            ),
        )

    def _tokens(self, mel: NDArray[np.float32]) -> NDArray[np.int64]:
        output = self._run(0, mel[None])
        if self.device == "coreml":
            # The CoreML tokenizer returns the id as base-3 digits, least
            # significant first; the ONNX graph returns the id itself.
            output = (output * (3 ** np.arange(_COREML_TOKEN_DIGITS))).sum(axis=-1)
        return np.asarray(output, dtype=np.int64).reshape(-1)

    def enroll(
        self, audio: NDArray[np.float32], sample_rate: int, *, name: str = ""
    ) -> VoiceProfile:
        """Validate reference audio and encode a portable speaker profile."""
        wav = np.asarray(audio, dtype=np.float32)
        validate_reference_audio(wav, sample_rate)
        full = resample(wav, sample_rate, _MEL_SR)
        prompt = full[:_PROMPT_SAMPLES]
        flow = resample(prompt, _MEL_SR, _S3_SR)
        speaker = resample(full, _MEL_SR, _S3_SR)
        mel = _prompt_mel(prompt)
        tokens = self._tokens(_token_mel(flow))
        n = min(len(tokens), mel.shape[1] // TOKEN_MEL_RATIO)
        cond = self._tokens(_token_mel(speaker[:_COND_SAMPLES])[:, :_COND_MEL_FRAMES])
        embedding = np.asarray(self._run(2, _partials(speaker)), dtype=np.float32).mean(axis=0)
        embedding /= np.linalg.norm(embedding)
        return VoiceProfile(
            name=name or "enrolled",
            speaker_embedding=embedding,
            flow_embedding=np.asarray(
                self._run(1, _fbank(flow)[None]), dtype=np.float32
            ).reshape(-1),
            prompt_tokens=tokens[:n],
            prompt_mel=mel[:, : TOKEN_MEL_RATIO * n],
            cond_prompt_tokens=cond,
            source_sample_rate=sample_rate,
        )


def build_graph_enroller(
    checkpoint: str, *, device: str, revision: str | None
) -> GraphEnroller:
    """Resolve a complete enrollment graph set for the selected runtime."""
    from ..hub import download, is_repo_id, resolve_checkpoint

    if is_repo_id(checkpoint):
        root = download(checkpoint, revision=revision, backend=device, cloning=True)
    else:
        root = resolve_checkpoint(checkpoint, revision=revision, backend=device).parent
    return GraphEnroller(root / device, device)
