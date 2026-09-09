"""The CoreML backend: native generation and rendering, driven from Python.

See ``docs/design/execution-config.md``.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ..checkpoint import TOKENIZER_FILENAME, Checkpoint
from ..config import AlgorithmConfig, ExecutionConfig
from ..contracts import TokenGenerator
from ..engine import Engine
from ..frontend.text import GraphemeTextFrontend
from ..release import EXPORT_RECORD
from . import register_backend
from .onnx_backend import (
    _F32,
    ONNXMelDecoder,
    ONNXTokenGenerator,
    ONNXVocoder,
    _GraphSession,
    _Tokens,
    graph_names,
)
from .onnx_backend import _require_static_window as require_static_window

__all__ = ["CoreMLMelDecoder", "CoreMLVocoder", "build_coreml_engine"]

ASSETS_ENV = "LOUDKIT_COREML_ASSETS"
ENCODER_PACKAGE = "flow_encoder.mlpackage"
ESTIMATOR_PACKAGE = "flow_estimator.mlpackage"
HIFT_PACKAGE = "vocoder.mlpackage"

_COMPUTE_UNITS = {
    ENCODER_PACKAGE: "CPU_ONLY",
    ESTIMATOR_PACKAGE: "CPU_AND_NE",
    HIFT_PACKAGE: "CPU_ONLY",
}
"""Which processors each renderer package is allowed to run on.

The placement ``docs/design/execution-config.md`` documents and the export was
measured against: the encoder and the vocoder are fp32 on the CPU, the
estimator is the fp16 pipeline the Neural Engine runs. Changing a unit here
changes both the speed and the numerics of a render, so it is written once
rather than three times at the call sites.
"""


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

    See ``docs/design/execution-config.md``.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._buffers: dict[str, NDArray[Any]] = {}
        self._storage: dict[str, NDArray[Any]] = {}
        self._views: dict[tuple[str, int, tuple[int, ...]], NDArray[Any]] = {}
        self._lock = threading.Lock()

    def predict(self, data: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            pinned: dict[str, Any] = {}
            for name, value in data.items():
                arr = np.ascontiguousarray(value)
                buf = self._buffers.get(name)
                if buf is None or buf.shape != arr.shape or buf.dtype != arr.dtype:
                    # Retain the array headers CoreML may still reference. Views
                    # share geometric storage, so growing KV caches stay linear.
                    storage = self._storage.get(name)
                    if storage is None or storage.size < arr.size or storage.dtype != arr.dtype:
                        capacity = 1 << max(0, arr.size - 1).bit_length()
                        storage = np.empty(capacity, dtype=arr.dtype)
                        self._storage[name] = storage
                    key = (name, id(storage), arr.shape)
                    buf = self._views.get(key)
                    if buf is None:
                        buf = storage[: arr.size].reshape(arr.shape)
                        self._views[key] = buf
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


class _GeneratorSession:
    def __init__(self, path: Path) -> None:
        import coremltools as ct

        self._model = _load_model(path, "CPU_ONLY")

        spec = ct.utils.load_spec(str(path))
        names_in = {item.name for item in spec.description.input}
        primary = {
            "t3_cond": ["speaker_emb", "prompt_tokens", "emotion"],
            "t3_prefill": ["embeds", "positions"],
            "t3_step": ["embeds", "position"],
            "t3_pair_step": ["pair_ids", "speech_position", "position"],
            "t3_head2": ["hidden", "first_id"],
        }[path.stem]
        self._inputs = primary + sorted(
            names_in - set(primary),
            key=lambda name: (int(name.rsplit("_", 1)[1]), "_v_" in name),
        )
        names = {item.name for item in spec.description.output}
        ordered = [name for name in ("logits", "hidden") if name in names]
        cache = names - set(ordered)
        # Cache names use paired layer indices, independent of protobuf ordering.
        ordered += (
            sorted(cache, key=lambda name: (int(name.rsplit("_", 1)[1]), "_v_" in name))
            if ordered
            else [item.name for item in spec.description.output]
        )
        self._outputs = ordered

    def run_positional(self, values: Any) -> list[NDArray[Any]]:
        feed = {
            name: np.asarray(
                value, dtype=np.int32 if np.asarray(value).dtype.kind in "iu" else np.float32
            )
            for name, value in zip(self._inputs, values, strict=True)
        }
        result = self._model.predict(feed)
        return [np.asarray(result[name]) for name in self._outputs]


class CoreMLTokenGenerator(ONNXTokenGenerator):
    @staticmethod
    def _load_session(directory: Path, name: str, execution: ExecutionConfig) -> _GraphSession:
        del execution
        return _GeneratorSession(directory / name.replace(".onnx", ".mlpackage"))


def _first_output(prediction: Mapping[str, Any]) -> NDArray[np.float32]:
    return np.asarray(next(iter(prediction.values())), dtype=np.float32)


def _require_static_window(config: AlgorithmConfig) -> tuple[int, int]:
    """Refuse a window the exported packages were not built for.

    See ``docs/design/execution-config.md``.
    """
    return require_static_window(config, "CoreML")


class CoreMLMelDecoder(ONNXMelDecoder):
    """``MelDecoder`` over the shipped encoder + estimator packages.

    The framing, the speaker affine and the Euler loop are the ONNX renderer's;
    only the token dtype and the two session calls are CoreML's.
    """

    _TOKEN_DTYPE = np.int32
    _BUILDER = "build_coreml_engine"

    def __init__(
        self, config: AlgorithmConfig, encoder: _MLModelLike, estimator: _MLModelLike
    ) -> None:
        if config.guidance != "single_path":
            raise ValueError(
                "the exported estimator is the guidance-distilled student; "
                "cfg_dual_path would apply guidance twice (EXP-016)"
            )
        self.config = config
        # Named apart from the base class's ONNX sessions rather than shadowing
        # them: a loaded package and a loaded graph are different objects with
        # different feeds, and only the seams below ever touch either.
        self._encoder_package = encoder
        self._estimator_package = estimator
        self._spk_weight: NDArray[np.float32] | None = None
        self._spk_bias: NDArray[np.float32] | None = None

    def _prompt_length(self, row: NDArray[np.int64]) -> int:
        # Not `static_prompt_tokens or half the row`: the packages are traced
        # at one geometry, so a window they were not built for is refused here
        # rather than reframed.
        del row
        return _require_static_window(self.config)[1]

    def _encode(self, prompt: _Tokens, query: _Tokens) -> _F32:
        return _first_output(
            self._encoder_package.predict({"prompt_token": prompt, "speech_tokens": query})
        )

    def _estimate(self, x: _F32, mu: _F32, t: _F32, spks: _F32, cond: _F32) -> _F32:
        return _first_output(
            self._estimator_package.predict(
                {"x": x, "mu": mu, "t": t, "spks": spks, "cond": cond}
            )
        )


class CoreMLVocoder(ONNXVocoder):
    """``Vocoder`` over the shipped fp32 HiFT package (static 510-frame mel)."""

    def __init__(self, config: AlgorithmConfig, hift: _MLModelLike) -> None:
        _require_static_window(config)
        self.config = config
        self._hift_package = hift

    def _vocode(self, padded: _F32, phase: _F32, noise: _F32) -> _F32:
        return _first_output(
            self._hift_package.predict({"mel": padded, "phase": phase, "noise": noise})
        )


def _assets_dir(ckpt: Checkpoint, algorithm: AlgorithmConfig | None = None) -> Path:
    """The first candidate directory holding the required packages.

    See ``docs/design/execution-config.md``.
    """
    required = (
        tuple(name.replace(".onnx", ".mlpackage") for name in graph_names(algorithm))
        if algorithm is not None
        else (ENCODER_PACKAGE, ESTIMATOR_PACKAGE, HIFT_PACKAGE)
    )
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


EXPORT_PROVENANCE = EXPORT_RECORD
"""What `tools/export_coreml.py` writes beside the packages it traced."""


def _check_provenance(assets: Path, ckpt: Checkpoint, algorithm: AlgorithmConfig) -> None:
    """The export record beside the packages must name this checkpoint."""
    from . import check_export_record

    check_export_record(
        assets,
        ckpt,
        algorithm,
        block="packages",
        members=tuple(name.replace(".onnx", ".mlpackage") for name in graph_names(algorithm)),
        tool="tools/export_coreml.py",
    )


@register_backend("coreml")
def build_coreml_engine(
    ckpt: Checkpoint, execution: ExecutionConfig, algorithm: AlgorithmConfig
) -> Engine:
    """Native graphs, or torch generation explicitly placed on the CPU."""
    precision = execution.precision_map()["token_generator"]
    if precision not in ("fp32", "fp16"):
        raise ValueError("CoreML generation supports native fp32 or PyTorch fp16")
    use_torch = execution.generator_device == "cpu"
    try:
        assets = _assets_dir(ckpt, None if use_torch else algorithm)
    except FileNotFoundError:
        # A renderer-only release ships no generator packages.
        # Never mask a partially installed generator or a missing turbo export.
        assets = _assets_dir(ckpt)
        if algorithm.decode != "single" or any(assets.glob("t3_*.mlpackage")):
            raise
        record = assets / EXPORT_PROVENANCE
        if record.is_file():
            import json

            try:
                packages = json.loads(record.read_text())["packages"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{record}: unreadable export record") from exc
            if any(name.startswith("t3_") for name in packages):
                raise
        use_torch = True
    generator: TokenGenerator
    if use_torch:
        from . import check_export_record

        try:
            from .torch_backend import build_torch_frontend_and_generator
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            raise ImportError(
                "fp16 or renderer-only CoreML generation needs PyTorch; "
                "install loudkit[torch] or use a complete native CoreML release"
            ) from exc

        check_export_record(
            assets,
            ckpt,
            algorithm,
            block="packages",
            members=(ENCODER_PACKAGE, ESTIMATOR_PACKAGE, HIFT_PACKAGE),
            tool="tools/export_coreml.py",
        )
        frontend, generator = build_torch_frontend_and_generator(
            ckpt, replace(execution, generator_device="cpu"), algorithm
        )
    else:
        if precision != "fp32":
            raise ValueError(
                "Native CoreML generation requires fp32; set generator_device='cpu' "
                "to use PyTorch with a different precision."
            )
        _check_provenance(assets, ckpt, algorithm)
        frontend = GraphemeTextFrontend(
            ckpt.resolve_asset(TOKENIZER_FILENAME, manifest_key="tokenizer_sha256")
        )
        generator = CoreMLTokenGenerator(algorithm, ckpt, assets, execution=execution)

    mel_decoder = CoreMLMelDecoder(
        algorithm,
        _load_model(assets / ENCODER_PACKAGE, _COMPUTE_UNITS[ENCODER_PACKAGE]),
        _load_model(assets / ESTIMATOR_PACKAGE, _COMPUTE_UNITS[ESTIMATOR_PACKAGE]),
    )
    # The speaker projection is outside the renderer graphs.
    affine = ckpt.tensors("s3gen.flow.spk_embed_affine_layer.")
    mel_decoder.attach_speaker_affine(
        np.asarray(affine["weight"], dtype=np.float32),
        np.asarray(affine["bias"], dtype=np.float32),
    )
    vocoder = CoreMLVocoder(
        algorithm, _load_model(assets / HIFT_PACKAGE, _COMPUTE_UNITS[HIFT_PACKAGE])
    )

    return Engine(
        frontend=frontend,
        token_generator=generator,
        mel_decoder=mel_decoder,
        vocoder=vocoder,
        algorithm=algorithm,
        execution=replace(execution, generator_device="cpu" if use_torch else "coreml"),
        backend="coreml",
        checkpoint_sha256=ckpt.file_digest,
        checkpoint_path=str(ckpt.path),
    )
