"""How a backend runs the algorithm. Never what it computes.

Every field defaults to ``None``, which means "the checkpoint's own default
for this backend". Name a field and it wins, even when the value equals what
the default would have been. :meth:`ExecutionConfig.resolved` fills the rest.
The measurements behind each knob are in ``docs/design/execution-config.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, NoReturn, get_args

__all__ = ["Device", "ExecutionConfig", "ONNXProvider", "ONNX_PROVIDERS", "Precision"]

Device = Literal["cpu", "cuda", "mps", "onnx", "coreml"]
Precision = Literal["fp32", "fp16", "bf16"]

ONNXProvider = Literal["auto", "cpu", "cuda", "coreml", "directml"]
"""The five spellings every port accepts for an onnxruntime execution provider."""

ONNX_PROVIDERS: tuple[ONNXProvider, ...] = get_args(ONNXProvider)

Attention = Literal["auto", "eager", "sdpa"]

_DEVICES: tuple[Device, ...] = get_args(Device)
_PRECISIONS: tuple[Precision, ...] = get_args(Precision)
_ATTENTIONS: tuple[Attention, ...] = get_args(Attention)

FP32_PRECISION: Mapping[str, Precision] = {
    "token_generator": "fp32",
    "mel_decoder.estimator": "fp32",
    "mel_decoder.encoder": "fp32",
    "vocoder": "fp32",
}
"""Every module in fp32: the reference datapath, and the ONNX backend's only one."""


def _refuse(field: str, value: object, allowed: Sequence[str]) -> NoReturn:
    """One message shape for every closed field, so the field is always named."""
    raise ValueError(f"unknown {field} {value!r}; expected one of {', '.join(allowed)}")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Device placement, precision and kernels. ``None`` means the backend's default."""

    device: Device | None = None
    generator_device: Device | None = None
    renderer_device: Device | None = None
    """Per-stage placement. The generator and the renderer want different hardware."""

    precision: Mapping[str, Precision] | None = None
    """Per-module dtype. A partial map merges over the default map, module by module."""

    compile_model: bool | None = None
    cuda_graphs: bool | None = None
    """Capture the decode step as a CUDA graph over a static KV cache.

    Both flags run the same capture. The static cache is the identity
    contract's ``equivalent`` class, not bit-identical with eager decode.
    """

    vocoder_ragged: bool | None = None
    """Vocode the frames a chunk has plus the receptive field, on a CUDA renderer only."""

    attention: Attention | None = None
    onnx_provider: ONNXProvider | None = None
    num_threads: int | None = None
    allow_tf32: bool | None = None
    deterministic: bool | None = None

    def __post_init__(self) -> None:
        """Refuse a value no backend can honour, where the field is still named.

        Every closed field is a ``Literal``, which nothing checks at runtime, so
        a typo used to travel: ``attention='bogus'`` reached the kernel picker
        and ``precision={'token_generator': 'int8'}`` surfaced three layers down
        as ``KeyError('int8')``, naming neither the field nor the config.
        """
        for field in ("device", "generator_device", "renderer_device"):
            value: str | None = getattr(self, field)
            # By stem: an ordinal such as "cuda:1" names the same backend, and
            # which ordinals exist is a question about this machine, asked once
            # the device is known to be real. See `loudkit._require_device`.
            if value is not None and value.split(":", 1)[0] not in _DEVICES:
                _refuse(field, value, _DEVICES)
        if self.attention is not None and self.attention not in _ATTENTIONS:
            _refuse("attention", self.attention, _ATTENTIONS)
        if self.onnx_provider is not None and self.onnx_provider not in ONNX_PROVIDERS:
            _refuse("onnx_provider", self.onnx_provider, ONNX_PROVIDERS)
        for module, precision in (self.precision or {}).items():
            # The module names stay open: a backend may carry stages this
            # package does not know about, and an unknown key is inert.
            if precision not in _PRECISIONS:
                _refuse(f"precision[{module!r}]", precision, _PRECISIONS)
        if self.num_threads is not None and self.num_threads <= 0:
            raise ValueError(
                f"num_threads must be positive, got {self.num_threads}; "
                "leave it None for the runtime's own default"
            )

    def resolved(self, defaults: ExecutionConfig | None = None) -> ExecutionConfig:
        """Every field filled: from this config, then ``defaults``, then the shipping
        fallbacks."""
        base = _FALLBACK if defaults is None else defaults.resolved()
        changes: dict[str, object] = {}
        for name in _FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if name == "precision":
                merged = dict(base.precision or FP32_PRECISION)
                merged.update(value)
                value = merged
            changes[name] = value
        return replace(base, **changes)  # type: ignore[arg-type]

    def resolved_device(self) -> str:
        return self.device or "cpu"

    def resolved_generator_device(self) -> str:
        return self.generator_device or self.resolved_device()

    def resolved_renderer_device(self) -> str:
        return self.renderer_device or self.resolved_device()

    def precision_map(self) -> Mapping[str, Precision]:
        return {**FP32_PRECISION, **(self.precision or {})}

    def resolved_vocoder_ragged(self) -> bool:
        """Whether the vocoder will pad by the receptive field: a CUDA renderer on torch
        only."""
        if self.vocoder_ragged is False:
            return False
        if self.resolved_device().split(":", 1)[0] in ("onnx", "coreml"):
            return False
        return self.resolved_renderer_device().split(":", 1)[0] == "cuda"

    def resolved_attention(self) -> Literal["eager", "sdpa"]:
        """``auto`` is ``eager`` on MPS and on pre-Ampere CUDA, ``sdpa`` elsewhere."""
        if self.attention is not None and self.attention != "auto":
            return self.attention
        gen = self.resolved_generator_device()
        if gen == "mps":
            return "eager"
        if gen.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    idx = gen.split(":", 1)
                    index = int(idx[1]) if len(idx) > 1 else 0
                    if torch.cuda.get_device_capability(index)[0] < 8:
                        return "eager"
            except (ImportError, RuntimeError, AssertionError, ValueError, IndexError) as exc:
                import warnings

                warnings.warn(
                    f"could not read the CUDA capability of {gen!r} ({exc}); "
                    "assuming SDPA is supported",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return "sdpa"

    def describe(self) -> str:
        """One line naming what runs. Logged on every run; recorded by every benchmark."""
        prec = ",".join(
            f"{k.split('.')[-1]}={v}" for k, v in sorted(self.precision_map().items())
        )
        placement = self.resolved_device()
        if self.generator_device or self.renderer_device:
            placement = (
                f"gen={self.resolved_generator_device()}/"
                f"render={self.resolved_renderer_device()}"
            )
        flags = [
            placement,
            f"attn={self.resolved_attention()}",
            f"prec[{prec}]",
            f"tf32={'on' if self.allow_tf32 else 'off'}",
        ]
        provider = self.onnx_provider or "auto"
        if self.resolved_device().split(":", 1)[0] == "onnx" or self.onnx_provider not in (
            None,
            "auto",
        ):
            flags.append(f"provider={provider}")
        if self.compile_model:
            flags.append("compiled")
        if self.cuda_graphs:
            flags.append("graphs")
        if self.resolved_vocoder_ragged():
            flags.append("ragged-vocoder")
        if self.deterministic is not False:
            flags.append("deterministic")
        if self.num_threads is not None:
            flags.append(f"threads={self.num_threads}")
        return "exec[" + " ".join(flags) + "]"


_FIELDS = tuple(ExecutionConfig.__dataclass_fields__)

_FALLBACK = ExecutionConfig(
    device="cpu",
    precision=FP32_PRECISION,
    compile_model=False,
    cuda_graphs=False,
    vocoder_ragged=True,
    attention="auto",
    onnx_provider="auto",
    allow_tf32=False,
    deterministic=True,
)
"""What an unset field means when no checkpoint default says otherwise."""
