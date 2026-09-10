"""Backend registry: one algorithm, several ways to execute it.

A backend turns ``(checkpoint, ExecutionConfig, AlgorithmConfig)`` into an
:class:`~loudkit.engine.Engine`. It declares execution choices and inherits
every algorithm value; a backend that decides an algorithm value is the bug
class this library exists to end.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from ..checkpoint import Checkpoint, require_decode_support
from ..config import AlgorithmConfig, DecodeMode, ExecutionConfig, Precision, WindowConfig
from ..engine import Engine
from ..postprocess import PostprocessConfig
from ..release import EXPORT_RECORD

__all__ = [
    "build_engine",
    "register_backend",
    "production_algorithm",
    "check_export_record",
    "PRODUCTION_WINDOW",
    "PRODUCTION_EOS_FLOOR",
    "PRODUCTION_EOS_TEXT_RATIO",
]

PRODUCTION_WINDOW = WindowConfig(
    max_speech_tokens=255,
    static_length=255,
    pad_token_id=4254,  # a silence unit; token 0 bleeds into the tail
    static_prompt_tokens=238,
)
"""The shipped static-window recipe, for a checkpoint packed before its manifest carried one."""

PRODUCTION_EOS_FLOOR = 10
PRODUCTION_EOS_TEXT_RATIO = 1.2
"""The shipped EOS floor, ``max(10, textIds * 6/5)``, for the same checkpoints."""


def production_algorithm(checkpoint: Checkpoint) -> AlgorithmConfig:
    """The manifest's algorithm; the production constants fill what an old pack omits."""
    base = AlgorithmConfig.from_manifest(checkpoint.manifest)
    manifest = checkpoint.manifest
    if "window" not in manifest:
        base = base.with_(window=PRODUCTION_WINDOW)
    if "postprocess" not in manifest:
        # replace(), so the render censuses read from the top level survive.
        base = base.with_(
            postprocess=replace(
                PostprocessConfig(),
                silence_render_ids=base.postprocess.silence_render_ids,
                quiet_render_ids=base.postprocess.quiet_render_ids,
            )
        )
    if "eos_floor" not in manifest:
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
    """Decorator claiming device names for a builder."""

    def deco(fn: _Builder) -> _Builder:
        for device in devices:
            _REGISTRY[device] = fn
        return fn

    return deco


def require_backend(device: str) -> None:
    """Raise unless some backend claims ``device``. Cheap, so it runs before any download."""
    base = device.split(":", 1)[0]
    with contextlib.suppress(ImportError):
        from . import onnx_backend  # noqa: F401
    with contextlib.suppress(ImportError):
        from . import coreml_backend  # noqa: F401
    # torch is registered only when asked for, so an onnx-only install never imports it.
    if base in ("cpu", "cuda", "mps"):
        from . import torch_backend  # noqa: F401
    if base not in _REGISTRY:
        raise ValueError(f"no backend for device {device!r}; known: {sorted(_REGISTRY)}")


def build_engine(
    path: str,
    *,
    device: str | None = None,
    execution: ExecutionConfig | None = None,
    algorithm: AlgorithmConfig | None = None,
) -> Engine:
    """``Engine.from_checkpoint``'s implementation.

    ``execution`` names the fields to change; unset fields take the manifest's
    defaults for ``device``. ``algorithm`` overrides the manifest's, and its
    fingerprint says so.
    """
    device = _agreed_device(device, execution) or "cpu"
    require_backend(device)
    base = device.split(":", 1)[0]
    ckpt = Checkpoint.open(path)
    algo = algorithm or production_algorithm(ckpt)
    require_decode_support(algo.decode, device, name=str(ckpt.manifest.get("name", "")))
    defaults = _default_execution(ckpt, device, decode=algo.decode)
    execu = defaults if execution is None else execution.resolved(defaults)
    _warn_if_static_cache(execu)
    return _REGISTRY[base](ckpt, execu, algo)


def _agreed_device(device: str | None, execution: ExecutionConfig | None) -> str | None:
    """The one device from two places that can each name one; ``None`` when neither does."""
    named = getattr(execution, "device", None)
    if device is None:
        return named
    if named is not None and named != device:
        raise ValueError(
            f"device={device!r} and execution.device={named!r} disagree. The "
            f"first chooses the backend (and, through loudkit.load, the "
            f"snapshot that is fetched); the second places the tensors. Name "
            f"one of them."
        )
    return device


def _warn_if_static_cache(execution: ExecutionConfig) -> None:
    """The static KV cache is the identity contract's ``equivalent`` class, not bit-exact."""
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


def _default_execution(ckpt: Checkpoint, device: str, *, decode: DecodeMode) -> ExecutionConfig:
    """The manifest's shipping dtype map on ``device``, every field resolved.

    Graph backends use their exported precision rather than checkpoint storage
    dtypes. On MPS the generator stays on the CPU so the two stages overlap.

    ``decode`` is passed in rather than read here: the caller already resolved
    it, and re-reading the manifest would parse it twice and would ignore an
    ``algorithm=`` override for this one choice.
    """
    if device.split(":", 1)[0] in ("onnx", "coreml"):
        precision: dict[str, Precision] = {}
        if device.split(":", 1)[0] == "coreml" and decode == "single":
            precision["mel_decoder.estimator"] = "fp16"
        return ExecutionConfig(device=device, precision=precision).resolved()  # type: ignore[arg-type]
    generator_device: str | None = "cpu" if device.startswith("mps") else None
    dtype_map: Mapping[str, str] = ckpt.dtype_map
    to_prec: dict[str, Precision] = {"float16": "fp16", "float32": "fp32"}
    return ExecutionConfig(
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
    ).resolved()


def _packed_estimator_digest(ckpt: Checkpoint) -> str | None:
    """The sha256 of the estimator the checkpoint was packed from, if ``sources`` says."""
    sources = ckpt.manifest.get("sources")
    if not isinstance(sources, dict):
        return None
    for entry in sources.values():
        if isinstance(entry, dict) and entry.get("role") == "estimator":
            digest = entry.get("sha256")
            return digest if isinstance(digest, str) else None
    return None


_RECORD_KEYS = ("checkpoint_sha256", "algorithm_fingerprint", "euler_steps", "estimator_sha256")


def check_export_record(
    assets: Path,
    ckpt: Checkpoint,
    algorithm: AlgorithmConfig,
    *,
    block: str,
    members: Sequence[str],
    tool: str,
) -> None:
    """Refuse an exported set whose members did not come from one export of this checkpoint.

    ``export.json`` beside the graphs records, per member, the checkpoint
    digest, the fingerprint, the step count and the estimator digest. Every
    member must agree with the others and with the checkpoint being loaded.
    Absent is a warning, because sets exported before the record exist.
    """
    import json
    import warnings

    path = assets / EXPORT_RECORD
    if not path.is_file():
        warnings.warn(
            f"{assets} carries no {EXPORT_RECORD}, so nothing says its {len(members)} "
            f"{block} came from one export of one checkpoint. Re-export with "
            f"{tool} to record it.",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))[block]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: unreadable export record ({exc})") from None

    def well_formed(entry: object) -> bool:
        if not isinstance(entry, dict):
            return False
        return all(
            isinstance(entry.get(key), (str, int, type(None)))
            and not isinstance(entry.get(key), bool)
            for key in _RECORD_KEYS
        )

    if not isinstance(entries, dict) or not all(well_formed(e) for e in entries.values()):
        raise ValueError(
            f"{path}: the {block!r} block is not a mapping of name to its export "
            f"record. Re-export the set with {tool}."
        )
    seen = {name: tuple(entry.get(k) for k in _RECORD_KEYS) for name, entry in entries.items()}
    for name in members:
        if name not in seen:
            raise ValueError(
                f"{path} does not record {name}, so it came from some other run "
                f"than the ones it does record. Re-export the set."
            )
    distinct = set(seen.values())
    if len(distinct) > 1:
        rows = "\n".join(
            f"  {n}: {dict(zip(_RECORD_KEYS, v, strict=True))}" for n, v in sorted(seen.items())
        )
        raise ValueError(
            f"{assets} is a mixed {block[:-1]} set: its members were exported from "
            f"different inputs:\n{rows}\nRe-export every stage together."
        )
    agreed = dict(zip(_RECORD_KEYS, distinct.pop(), strict=True))
    engine_keys = ("checkpoint_sha256", "algorithm_fingerprint", "euler_steps")
    got = tuple(agreed[k] for k in engine_keys)
    want = (ckpt.file_digest, algorithm.fingerprint(), algorithm.euler_steps)
    if got != want:
        raise ValueError(
            f"{assets} was exported from a different engine than the one loading "
            f"it:\n  {block}: {dict(zip(engine_keys, got, strict=True))}\n"
            f"  checkpoint: {dict(zip(engine_keys, want, strict=True))}\n"
            f"Re-export against {ckpt.path.name}."
        )
    # A set traced from a swapped-in estimator agrees with itself and
    # describes a renderer the checkpoint does not; compared only when both
    # sides record one.
    recorded = agreed["estimator_sha256"]
    packed = _packed_estimator_digest(ckpt)
    if recorded is not None and packed is not None and recorded != packed:
        raise ValueError(
            f"{assets} was traced with an estimator the checkpoint was not "
            f"packed from:\n  traced:  {recorded}\n  packed:  {packed}\n"
            f"That is a different renderer than {ckpt.path.name} describes, and "
            f"its fingerprint does not cover the difference. Re-export without "
            f"--estimator-ckpt, or pack the estimator you traced."
        )
