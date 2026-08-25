"""The CoreML backend: the Apple renderer graphs, driven from Python.

This backend runs the CoreML stage packages exported by
``tools/export_coreml.py`` — ``flow_encoder`` (CPU, fp32),
``flow_estimator`` (CPU + Neural Engine, fp16 pipeline), ``vocoder`` (CPU,
fp32) — behind the loudkit ``MelDecoder`` and ``Vocoder`` protocols. The
graph geometry is exactly the one the iOS app ships (query 255 / prompt 238,
T986 mel, 510-frame HiFT); the weights are re-exported from the packed
checkpoint so provenance is one file, not archaeology. It exists so that
"matches the shipped engine" is a table produced by this repo rather than a
belief.

**The token generator stays on torch (CPU).** The app's T3 runs through a
stateful multi-function CoreML package whose Python-side validation was
explicitly not achieved (torch/CoreML same-process instability, documented in
the 2026-07-26 sample wall under "Not covered: T3 on the ANE"); a backend row
produced from an unvalidated export would be worse than no row. The renderer
is the part where the ANE recipe questions live, and it is fully covered.

The static graphs make the window recipe non-negotiable: this backend refuses
an :class:`AlgorithmConfig` whose window does not match the exported geometry,
because "pad differently and hope" is precisely the defect class
(mel corr 0.975–0.993) the recipe moved into configuration to end.

Asset resolution: a ``coreml/`` directory beside the checkpoint, or the
``LOUDKIT_COREML_ASSETS`` environment variable. Missing assets fail with the
expected filenames, not with a fallback to a different engine.

Every predict goes through :class:`_PinnedInputs`, which is what keeps this
backend from killing its host a second after it succeeds; its docstring carries
the mechanism.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ..checkpoint import Checkpoint
from ..config import AlgorithmConfig, ExecutionConfig
from ..contracts import Mel, SpeechTokens, Waveform
from ..engine import Engine
from ..models.flow import FLOW_NOISE_STREAM, frame_windows, time_grid
from ..models.noise import gaussian_field, symmetric_uniforms
from ..models.vocoder import VOCODER_NOISE_STREAM, VOCODER_PHASE_STREAM
from ..voice import VoiceProfile
from . import register_backend

__all__ = ["CoreMLMelDecoder", "CoreMLVocoder", "build_coreml_engine"]

ASSETS_ENV = "LOUDKIT_COREML_ASSETS"
ENCODER_PACKAGE = "flow_encoder.mlpackage"
ESTIMATOR_PACKAGE = "flow_estimator.mlpackage"
HIFT_PACKAGE = "vocoder.mlpackage"

_MEL_BINS = 80
_N_HARMONICS = 9
_UPSAMPLE_PER_FRAME = 480


class _MLModelLike(Protocol):
    """The one method this backend uses from coremltools' MLModel.

    coremltools ships no type information (see the pyproject override), so
    this protocol states the contract the code below actually relies on; the
    output values are numpy arrays, typed Any because that is what crosses
    the untyped boundary.
    """

    def predict(self, data: Mapping[str, Any]) -> dict[str, Any]: ...


class _PinnedInputs:
    """A model whose input arrays live as long as it does.

    coremltools wraps each input array without copying: ``PybindCompatibleArray``
    (``coremlpython/CoreMLPythonArray.mm``) builds the ``MLMultiArray`` over the
    caller's numpy buffer and keeps the ``py::array`` as an Objective-C ivar.
    CoreML does not drop that reference when ``predict`` returns. The MLE5
    execution stream lingers and resets itself about a second later on
    ``com.apple.coreml.MLE5ExecutionStream.resetQueue``, and *that* is where
    ``-[MLFeatureValue dealloc]`` runs. The compiler-generated ``.cxx_destruct``
    then releases the ``py::array`` on a dispatch thread that holds no GIL and
    has no thread state; if the interpreter's reference was the last one, the
    release reaches ``_PyObject_Free`` and corrupts pymalloc's arenas. The host
    process dies about a second after a perfectly successful synthesis, inside
    whatever it does next (upstream: apple/coremltools#2827, open and unfixed
    through 9.0).

    So the interpreter keeps a reference of its own, for the model's lifetime,
    and every predict copies into that same buffer. CoreML's release then only
    ever takes the count from two to one, and the actual free happens here, on a
    thread that holds the GIL. The buffers are bounded rather than leaked
    because the exported graphs are static (:func:`_require_static_window`):
    one buffer per input, reallocated never. Reusing them is safe because
    ``predict`` is synchronous: the lingering stream holds the reference, not
    the data. Measured: waveform bit-identical to the unpinned path, and the
    copy does not show above run-to-run noise (2.09 s against 2.06 s median of
    four, one 4.96 s sentence on an M3 Pro).

    The lock is what makes reuse safe under ``Engine.stream``, whose renderer
    runs on its own thread; two callers sharing one buffer would otherwise
    interleave a fill with a predict.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._buffers: dict[str, NDArray[Any]] = {}
        self._lock = threading.Lock()

    def predict(self, data: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            pinned: dict[str, Any] = {}
            for name, value in data.items():
                arr = np.ascontiguousarray(value)
                buf = self._buffers.get(name)
                if buf is None or buf.shape != arr.shape or buf.dtype != arr.dtype:
                    # A replaced buffer is *retained*, under a new key, rather
                    # than dropped: a stream lingering over the old one would
                    # free it on the reset queue, which is the crash this class
                    # exists to prevent. The static geometry means this branch
                    # runs once per input and never again.
                    if buf is not None:
                        self._buffers[f"{name}#{len(self._buffers)}"] = buf
                    buf = np.empty(arr.shape, dtype=arr.dtype)
                    self._buffers[name] = buf
                np.copyto(buf, arr)
                pinned[name] = buf
            return cast("dict[str, Any]", self._model.predict(pinned))


def _load_model(path: Path, compute_units: str) -> _MLModelLike:
    import coremltools as ct

    units = getattr(ct.ComputeUnit, compute_units)
    # the wrapper is the single place coremltools' untyped MLModel enters typed
    # code; _MLModelLike is the promise the rest of this module leans on, and
    # _PinnedInputs is why a raw MLModel is never handed out (see its docstring)
    return _PinnedInputs(ct.models.MLModel(str(path), compute_units=units))


def _first_output(prediction: Mapping[str, Any]) -> NDArray[np.float32]:
    return np.asarray(next(iter(prediction.values())), dtype=np.float32)


def _require_static_window(config: AlgorithmConfig) -> tuple[int, int]:
    w = config.window
    if w.static_length != 255 or w.static_prompt_tokens != 238:
        raise ValueError(
            "the exported CoreML graphs are static at query 255 / prompt 238; "
            f"this AlgorithmConfig frames {w.static_length}/{w.static_prompt_tokens}. "
            "A different window is a different algorithm — re-export the graphs "
            "rather than silently reframing here."
        )
    return w.static_length, w.static_prompt_tokens


class CoreMLMelDecoder:
    """``MelDecoder`` over the shipped encoder + estimator packages."""

    def __init__(
        self, config: AlgorithmConfig, encoder: _MLModelLike, estimator: _MLModelLike
    ) -> None:
        if config.guidance != "single_path":
            raise ValueError(
                "the exported estimator is the guidance-distilled student; "
                "cfg_dual_path would apply guidance twice (EXP-016)"
            )
        self.config = config
        self._encoder = encoder
        self._estimator = estimator
        self._spk_weight: NDArray[np.float32] | None = None
        self._spk_bias: NDArray[np.float32] | None = None

    def attach_speaker_affine(
        self, weight: NDArray[np.float32], bias: NDArray[np.float32]
    ) -> None:
        """The 192->80 speaker projection is part of the torch flow module and
        was baked into neither exported graph; the backend hands its weights
        over so the CoreML path computes the identical ``spks``."""
        self._spk_weight = weight
        self._spk_bias = bias

    def decode(self, tokens: SpeechTokens, voice: VoiceProfile, *, seed: int) -> Mel:
        if self._spk_weight is None or self._spk_bias is None:
            raise RuntimeError("speaker affine not attached; build via build_coreml_engine")
        _, prompt_len = _require_static_window(self.config)
        row, cond, prompt_frames, n = frame_windows(self.config, tokens, voice)
        t_mel = 2 * row.shape[1]
        prompt = row[:, :prompt_len].astype(np.int32)
        query = row[:, prompt_len:].astype(np.int32)

        mu = _first_output(
            self._encoder.predict({"prompt_token": prompt, "speech_tokens": query})
        ).reshape(1, _MEL_BINS, t_mel)

        emb = np.asarray(voice.flow_embedding, dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        spks = (self._spk_weight @ emb + self._spk_bias)[None].astype(np.float32)

        grid = time_grid(self.config)
        x = gaussian_field(seed, FLOW_NOISE_STREAM, _MEL_BINS, t_mel)[None]
        for t0, t1 in zip(grid[:-1], grid[1:], strict=False):
            v = _first_output(
                self._estimator.predict(
                    {
                        "x": x,
                        "mu": mu,
                        "t": np.array([t0], dtype=np.float32),
                        "spks": spks,
                        "cond": cond,
                    }
                )
            ).reshape(1, _MEL_BINS, t_mel)
            # np.float32 keeps the step in fp32 (identical arithmetic — NEP 50
            # rounds a weak python float to the array dtype anyway) and keeps
            # numpy's stubs from promoting the whole state to float64
            x = x + np.float32(t1 - t0) * v
        return x[0, :, prompt_frames : prompt_frames + 2 * n].astype(np.float32)


class CoreMLVocoder:
    """``Vocoder`` over the shipped fp32 HiFT package (static 510-frame mel)."""

    def __init__(self, config: AlgorithmConfig, hift: _MLModelLike) -> None:
        _require_static_window(config)
        self.config = config
        self._hift = hift

    def synthesize(self, mel: Mel, voice: VoiceProfile, *, seed: int) -> Waveform:
        del voice  # timbre already lives in the mel; see TorchVocoder
        frames = 2 * self.config.window.max_speech_tokens
        n_frames = min(int(mel.shape[1]), frames)
        padded = np.zeros((1, _MEL_BINS, frames), dtype=np.float32)
        padded[0, :, :n_frames] = mel[:, :n_frames]

        n_samples = frames * _UPSAMPLE_PER_FRAME
        phase = np.zeros((1, _N_HARMONICS, 1), dtype=np.float32)
        phase[0, 1:, 0] = symmetric_uniforms(
            seed, VOCODER_PHASE_STREAM, _N_HARMONICS - 1, math.pi
        )
        noise = gaussian_field(seed, VOCODER_NOISE_STREAM, _N_HARMONICS, n_samples)[None]

        wav = _first_output(
            self._hift.predict({"mel": padded, "phase": phase, "noise": noise})
        ).reshape(-1)
        return wav[: n_frames * _UPSAMPLE_PER_FRAME].astype(np.float32)


def _assets_dir(ckpt: Checkpoint) -> Path:
    """The first candidate directory holding **all three** packages.

    Testing only for the estimator accepted a partial export and stopped
    looking, so a stale or half-written directory named by the environment
    variable shadowed a complete one beside the checkpoint — and the failure
    surfaced later, as a missing encoder or vocoder, naming a file rather than
    the directory choice that caused it. The ONNX backend has always required
    the full set; this matches it, and reports what each candidate was missing.
    """
    required = (ENCODER_PACKAGE, ESTIMATOR_PACKAGE, HIFT_PACKAGE)
    env = os.environ.get(ASSETS_ENV)
    candidates = [Path(env)] if env else []
    candidates.append(ckpt.path.parent / "coreml")

    report: list[str] = []
    for cand in candidates:
        missing = [name for name in required if not (cand / name).exists()]
        if not missing:
            return cand
        report.append(f"  {cand}: missing {', '.join(missing)}")
    raise FileNotFoundError(
        "CoreML assets not found. Every candidate directory is incomplete:\n"
        + "\n".join(report)
        + f"\n(override the search with ${ASSETS_ENV}.) "
        "Run tools/export_coreml.py to create them."
    )


@register_backend("coreml")
def build_coreml_engine(
    ckpt: Checkpoint, execution: ExecutionConfig, algorithm: AlgorithmConfig
) -> Engine:
    """Torch T3 on the CPU, the shipped CoreML graphs for the renderer.

    The mixed build is the honest one: it is exactly the split the sample wall
    validated (renderer end-to-end, T3 export not yet validated from Python).
    """
    from .torch_backend import build_torch_frontend_and_generator

    assets = _assets_dir(ckpt)
    cpu_exec = ExecutionConfig(
        device="cpu",
        precision=execution.precision,
        deterministic=execution.deterministic,
        num_threads=execution.num_threads,
    )
    # Only the stages this backend actually keeps. Building a whole torch
    # engine here loaded the mel decoder and vocoder — several hundred MB —
    # and then discarded them for the CoreML packages below, on the device
    # where CoreML is supposed to be the lightweight option.
    frontend, generator = build_torch_frontend_and_generator(ckpt, cpu_exec, algorithm)

    mel_decoder = CoreMLMelDecoder(
        algorithm,
        _load_model(assets / ENCODER_PACKAGE, "CPU_ONLY"),
        _load_model(assets / ESTIMATOR_PACKAGE, "CPU_AND_NE"),
    )
    # The 192->80 speaker affine is read from the checkpoint, not fished out
    # of torch_engine.mel_decoder: the MelDecoder protocol deliberately does
    # not expose module internals (mypy confirmed the hole), and the packed
    # file is the same provenance the torch module loaded these weights from.
    affine = ckpt.tensors("s3gen.flow.spk_embed_affine_layer.")
    mel_decoder.attach_speaker_affine(
        np.asarray(affine["weight"], dtype=np.float32),
        np.asarray(affine["bias"], dtype=np.float32),
    )
    vocoder = CoreMLVocoder(algorithm, _load_model(assets / HIFT_PACKAGE, "CPU_ONLY"))

    return Engine(
        frontend=frontend,
        token_generator=generator,
        mel_decoder=mel_decoder,
        vocoder=vocoder,
        algorithm=algorithm,
        execution=execution,
        backend="coreml",
        checkpoint_sha256=ckpt.file_digest,
    )
