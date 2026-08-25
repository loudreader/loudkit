"""Backend registry: one algorithm, several ways to execute it.

A backend is a function that turns ``(checkpoint, ExecutionConfig,
AlgorithmConfig)`` into an :class:`~loudkit.engine.Engine`. Backends declare
execution choices — dtype maps, kernels, device placement — and inherit every
algorithm value unchanged; a backend that *decides* an algorithm value is the
bug class this library exists to end.

Amended checkpoints (``tools/amend_manifest.py``) carry the static window
recipe and the EOS floor in their manifest, and the manifest is the authority.
The constants below exist only as the fallback for checkpoints packed before
the amendment — the values are the same ones, stated once more so an old pack
still runs the shipped algorithm rather than a ragged-window guess.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping

from ..checkpoint import Checkpoint
from ..config import (
    AlgorithmConfig,
    ExecutionConfig,
    ExecutionOverrides,
    PostprocessConfig,
    Precision,
    WindowConfig,
)
from ..engine import Engine

__all__ = [
    "build_engine",
    "register_backend",
    "production_algorithm",
    "PRODUCTION_WINDOW",
    "PRODUCTION_EOS_FLOOR",
    "PRODUCTION_EOS_TEXT_RATIO",
]

PRODUCTION_WINDOW = WindowConfig(
    max_speech_tokens=255,  # one static window is 10.2 s; longer text chunks upstream
    static_length=255,
    pad_token_id=4254,  # a silence unit; padding with token 0 bleeds +3 dB HF into the tail
    static_prompt_tokens=238,
)
"""The shipped static-window recipe (ChatterboxMelSynthesizer.swift)."""

PRODUCTION_EOS_FLOOR = 10
PRODUCTION_EOS_TEXT_RATIO = 1.2
"""The shipped EOS floor (ChatterboxT3Runner: ``max(10, textIds * 6/5)``)."""


def production_algorithm(checkpoint: Checkpoint) -> AlgorithmConfig:
    """The shipping algorithm for this checkpoint.

    ``AlgorithmConfig.from_manifest`` reads everything an amended checkpoint
    carries — including the window recipe and the EOS floor. For a checkpoint
    packed before the amendment, the production constants above fill exactly
    those two gaps and nothing else. Guidance stays whatever the manifest
    declares (``single_path`` for the packed student — EXP-016).
    """
    base = AlgorithmConfig.from_manifest(checkpoint.manifest)
    manifest = checkpoint.manifest
    if "window" not in manifest:
        base = base.with_(window=PRODUCTION_WINDOW)
    if "postprocess" not in manifest:
        # The artifact detectors are a shipping default rather than a manifest
        # field. A manifest that does not mention them still gets them, because
        # they are part of the one recipe this library has.
        base = base.with_(postprocess=PostprocessConfig())
    if "eos_floor" not in manifest:
        from dataclasses import replace

        # replace() rather than a hand-written reconstruction: a new field on
        # SamplingConfig must not silently reset to its default for every
        # non-amended checkpoint. The EOS floor is the only override here.
        base = base.with_(
            sampling=replace(
                base.sampling,
                min_tokens_floor=PRODUCTION_EOS_FLOOR,
                min_tokens_text_ratio=PRODUCTION_EOS_TEXT_RATIO,
            )
        )
    return base


_Builder = Callable[[Checkpoint, ExecutionConfig, AlgorithmConfig], Engine]
_REGISTRY: dict[str, _Builder] = {}


def register_backend(*devices: str) -> Callable[[_Builder], _Builder]:
    """Class of decorator that claims device names for a builder.

    Kept trivially small on purpose: the registry exists so that a future
    ONNX or CoreML backend is an entry here rather than a fork of
    ``Engine.from_checkpoint``.
    """

    def deco(fn: _Builder) -> _Builder:
        for device in devices:
            _REGISTRY[device] = fn
        return fn

    return deco


def require_backend(device: str) -> None:
    """Raise unless some backend claims ``device``.

    Split out of :func:`build_engine` so a caller can ask the cheap question
    first. ``loudkit.load`` does: a typo\'d device is answerable from a dict,
    while resolving the checkpoint may mean downloading 747 MB — and a call that
    could never have worked should not cost that.

    Raises:
        ValueError: naming the device and listing the ones that exist.
    """
    base = device.split(":", 1)[0]

    # Optional graph backends register themselves when their runtime is
    # present: onnxruntime + exported graphs, coremltools + exported packages.
    with contextlib.suppress(ImportError):
        from . import onnx_backend  # noqa: F401
    with contextlib.suppress(ImportError):
        from . import coreml_backend  # noqa: F401

    # The torch backend is registered lazily and only when asked for: a
    # ``loudkit[onnx]`` install synthesises with no torch in the process.
    if base in ("cpu", "cuda", "mps"):
        from . import torch_backend  # noqa: F401

    if base not in _REGISTRY:
        raise ValueError(f"no backend for device {device!r}; known: {sorted(_REGISTRY)}")


def build_engine(
    path: str,
    *,
    device: str = "cpu",
    execution: ExecutionConfig | ExecutionOverrides | None = None,
    algorithm: AlgorithmConfig | None = None,
) -> Engine:
    """Build an engine from a packed checkpoint (``Engine.from_checkpoint``'s
    implementation).

    Args:
        path: packed ``.safetensors`` checkpoint.
        device: which backend runs it. ``cpu`` / ``cuda`` / ``mps`` select the
            torch backend on that device; ``onnx`` and ``coreml`` (when their
            exported assets are present) select the graph backends.
        execution: ``None`` for the manifest's shipping dtype map on the chosen
            device, an :class:`~loudkit.config.ExecutionOverrides` to change
            named fields and inherit the rest, or a full
            :class:`~loudkit.config.ExecutionConfig` to specify everything.
            See :func:`_resolve_execution` for why those last two are separate
            types rather than one.
        algorithm: algorithm override. Defaults to the checkpoint's shipping
            algorithm; a different value is a deliberately different engine
            and its fingerprint will say so.
    """
    require_backend(device)
    base = device.split(":", 1)[0]

    ckpt = Checkpoint.open(path)
    algo = algorithm or production_algorithm(ckpt)
    defaults = _default_execution(ckpt, device)
    execu = _resolve_execution(defaults, execution)
    _warn_if_static_cache(execu)
    return _REGISTRY[base](ckpt, execu, algo)


def _warn_if_static_cache(execution: ExecutionConfig) -> None:
    """Warn when the static KV cache is active.

    The static cache (``cuda_graphs`` / ``compile_model``) runs the decode over
    a padded buffer that switches the cuBLAS kernel at large widths; measured
    logit drift ~2e-4/layer at a 750-token prefill flips a sampled token on long
    windows. The identity contract classifies it as ``equivalent`` — same
    distribution, not the same stream — and the user deserves to know they are
    off the bit-identical path before the output differs from eager.
    """
    if execution.cuda_graphs or execution.compile_model:
        import warnings

        warnings.warn(
            "cuda_graphs/compile_model uses a static KV cache: logits drift "
            "~2e-4/layer on long windows and a sampled token may differ from "
            "the eager path (identity contract: 'equivalent', not bit-exact). "
            "Fine for throughput; not for byte-for-byte agreement.",
            RuntimeWarning,
            stacklevel=3,
        )


def _resolve_execution(
    defaults: ExecutionConfig, execution: ExecutionConfig | ExecutionOverrides | None
) -> ExecutionConfig:
    """Decide the execution config from the caller's request and the manifest.

    Three cases, and the distinction between the last two is the whole point:

    * ``None`` — the manifest's shipping map on the requested device.
    * :class:`~loudkit.config.ExecutionOverrides` — a patch. Named fields win,
      everything else inherits, ``precision`` merges per module. This is what a
      caller means by "the shipping engine plus CUDA graphs".
    * :class:`~loudkit.config.ExecutionConfig` — a complete configuration, used
      verbatim. This is what a caller means by "run exactly this", and it is
      the case the old merge could not express: it compared each field against
      the dataclass default and dropped anything that matched, so an explicit
      all-fp32 map was silently replaced by the manifest's fp16 one and a
      conformance run measured a precision it had not asked for.
    """
    if execution is None:
        return defaults
    if isinstance(execution, ExecutionOverrides):
        return execution.applied_to(defaults)
    return execution


def _default_execution(ckpt: Checkpoint, device: str) -> ExecutionConfig:
    """The manifest's shipping dtype map, on the requested device.

    fp16 where it is measured safe (the generator: median KL 1.3e-06; the flow
    estimator: mel corr 0.999999), fp32 where it is measured fatal (the flow
    encoder, the vocoder). The same map on every device — precision is an
    execution choice, but a *declared* one, and the default should be the
    configuration the parity tables were measured in.

    **The onnx device is the exception, and it is a hard one.** The exported
    graphs are fp32 (EXP-015: fp16 not worth a second artifact; int8 blocked),
    so an ONNX engine *cannot* run the manifest's fp16 map — the graphs do not
    exist in fp16. The default is therefore all-fp32, which is also the gate's
    measurement configuration.
    """
    if device.split(":", 1)[0] == "onnx":
        return ExecutionConfig(
            device=device,  # type: ignore[arg-type]
            precision={
                "token_generator": "fp32",
                "mel_decoder.estimator": "fp32",
                "mel_decoder.encoder": "fp32",
                "vocoder": "fp32",
            },
        )
    # On Apple silicon the two stages live on different hardware, and since the
    # streaming pipeline the reason is parallelism, not per-stage speed: the
    # renderer renders window k on the GPU while the generator computes window
    # k+1 on the CPU, and the two genuinely overlap. Putting both on MPS makes
    # them contend for one device — measured on an M3 Pro (torch 2.13), a
    # six-window passage runs at RTF 3.37 split against 2.15 all-MPS, with the
    # all-MPS generator slowed ~40% by the renderer sharing its queue. (For a
    # single window there is no pipeline and all-MPS is mildly faster, 3.07
    # against 2.45; the default favours the multi-window paths, which are the
    # ones a reader or a server actually runs hot.)
    generator_device: str | None = "cpu" if device.startswith("mps") else None

    dtype_map: Mapping[str, str] = ckpt.dtype_map
    # storage-dtype names (safetensors) -> ExecutionConfig's Precision literals
    to_prec: dict[str, Precision] = {"float16": "fp16", "float32": "fp32"}
    return ExecutionConfig(
        # str, not Device: callers pass forms like "cuda:0"/"coreml" that the
        # Literal does not cover; the registry lookup above already vetted it
        device=device,  # type: ignore[arg-type]
        generator_device=generator_device,  # type: ignore[arg-type]
        precision={
            "token_generator": to_prec.get(str(dtype_map.get("t3", "float32")), "fp32"),
            "mel_decoder.estimator": to_prec.get(
                str(dtype_map.get("s3gen.flow.decoder.estimator", "float32")), "fp32"
            ),
            "mel_decoder.encoder": "fp32",
            "vocoder": "fp32",
        },
    )
