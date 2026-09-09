"""The torch backend: one implementation, parameterised by device.

See ``docs/design/execution-config.md``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch

from ..checkpoint import TOKENIZER_FILENAME, Checkpoint
from ..config import AlgorithmConfig, ExecutionConfig, Precision
from ..engine import Engine
from ..frontend.text import GraphemeTextFrontend
from ..models.enroll import TorchVoiceEnroller
from ..models.flow import TorchMelDecoder
from ..models.generator import TorchTokenGenerator
from ..models.vocoder import TorchVocoder
from . import register_backend

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

__all__ = [
    "build_torch_engine",
    "build_torch_enroller",
    "build_torch_frontend_and_generator",
    "pin_determinism",
]

_DTYPES: dict[Precision, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

_PINNED: tuple[bool, bool, int | None] | None = None
"""The ``(allow_tf32, deterministic, num_threads)`` this process last pinned.

See ``docs/design/execution-config.md``.
"""

_PIN_LOCK = threading.Lock()
"""Serialises the read-compare-write on ``_PINNED``. See above."""


def pin_determinism(execution: ExecutionConfig) -> None:
    """Apply the identity contract's backend flags (I-2).

    See ``docs/design/execution-config.md``.
    """
    execution = execution.resolved()
    # Module-global on purpose: it mirrors torch's own process-global flags,
    # and there is exactly one process to track.
    global _PINNED  # noqa: PLW0603
    # `num_threads` belongs here with the other two.
    tf32 = bool(execution.allow_tf32)
    deterministic = execution.deterministic is not False
    pins = (tf32, deterministic, execution.num_threads)
    # Read, compare and write under one lock: two threads building engines at
    # once could otherwise both see `None`, both pin, and the second never be
    # reported as the contradiction it is.
    with _PIN_LOCK:
        previous = _PINNED
        _PINNED = pins
    if previous is not None and pins != previous:
        # These flags are process-global, so the second engine silently re-pins them
        # under the first one's feet: the first keeps reporting its own `describe()`.
        import warnings

        was_tf32, was_det, was_threads = previous
        warnings.warn(
            "a torch engine in this process was already pinned with "
            f"allow_tf32={was_tf32}, deterministic={was_det}, "
            f"num_threads={was_threads}; building one with "
            f"allow_tf32={execution.allow_tf32}, "
            f"deterministic={execution.deterministic}, "
            f"num_threads={execution.num_threads} "
            "re-pins those process-global flags for BOTH. The older engine's "
            "describe() no longer matches what it runs. Use one execution "
            "configuration per process, or run them in separate processes.",
            RuntimeWarning,
            stacklevel=3,
        )

    # TF32 is pinned from the config on EVERY path, not only the deterministic
    # one. Left alone it rides on PyTorch's defaults -- cudnn on, matmul off --
    # so a non-deterministic engine would silently run TF32 convolutions and
    # nothing would say so. That is the exact trap that made a measured fp32
    # baseline 5% too fast and not bit-exact with itself.
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    # Symmetric on purpose. Only setting these under `deterministic` meant a
    # later non-deterministic engine inherited `cudnn.deterministic=True` from
    # an earlier one and ran ~5% slower than its own config describes.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if execution.num_threads is not None:
        torch.set_num_threads(execution.num_threads)


def _tensors(ckpt: Checkpoint, prefix: str) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors(prefix).items()}


def _check_precision(execution: ExecutionConfig) -> dict[str, torch.dtype]:
    prec = dict(execution.precision_map())
    for module in ("mel_decoder.encoder", "vocoder"):
        if prec.get(module, "fp32") != "fp32":
            raise ValueError(
                f"precision[{module!r}] = {prec[module]!r} is not a tuning knob: "
                "measured fp16 there gives mel corr 0.619 / an audible Nyquist "
                "tone. Only the token generator and the flow estimator tolerate "
                "reduced precision."
            )
    return {name: _DTYPES[p] for name, p in prec.items()}


def build_torch_frontend_and_generator(
    ckpt: Checkpoint, execution: ExecutionConfig, algorithm: AlgorithmConfig
) -> tuple[GraphemeTextFrontend, TorchTokenGenerator]:
    """The first two stages only, text to speech tokens.

    See ``docs/design/execution-config.md``.
    """
    pin_determinism(execution)
    # The generator's own device: an autoregressive step is a few hundred tiny
    # dispatches, so it often wants different hardware from the renderer. They
    # can be split at all only because the components pass arrays to each
    # other, never live tensors.
    gen_device = torch.device(execution.resolved_generator_device())
    dtypes = _check_precision(execution)

    # Packed copy first, sibling second, see `Checkpoint.resolve_asset`. A
    # packed checkpoint carries its own tokenizer, so there is no second file
    # to swap, mislay, or bind by a digest the manifest never recorded.
    frontend = GraphemeTextFrontend(
        ckpt.resolve_asset(TOKENIZER_FILENAME, manifest_key="tokenizer_sha256")
    )

    llama_config = ckpt.manifest.get("llama_config")
    if not isinstance(llama_config, dict):
        # Raised, not asserted: `python -O` strips asserts, and a manifest
        # without this would then fail inside the module constructor with a
        # message naming neither the manifest nor the missing key.
        raise ValueError(
            f"{ckpt.path.name}: manifest is missing 'llama_config', which the "
            "token generator's architecture is read from"
        )
    _check_architecture_against_weights(ckpt, llama_config)
    generator = TorchTokenGenerator(
        algorithm,
        llama_config,
        attention=execution.resolved_attention(),
        cuda_graphs=bool(execution.cuda_graphs),
        compile_model=bool(execution.compile_model),
    )
    generator.load_state_dict(_tensors(ckpt, "t3."))
    gen_dtype = dtypes.get("token_generator", torch.float32)
    if gen_dtype is not torch.float32:
        generator = generator.to(gen_dtype)
    generator = generator.to(gen_device).eval()
    for p in generator.parameters():
        p.requires_grad_(False)
    return frontend, generator


def _check_architecture_against_weights(
    ckpt: Checkpoint, llama_config: Mapping[str, object]
) -> None:
    """Refuse a manifest whose architecture the weights do not corroborate.

    See ``docs/design/execution-config.md``.
    """
    shapes = ckpt.shapes("t3.")
    embed = shapes.get("tfmr.embed_tokens.weight")
    if embed is None or len(embed) != 2:  # an embedding is a matrix
        raise ValueError(
            f"{ckpt.path.name}: no 't3.tfmr.embed_tokens.weight' matrix to check "
            "the manifest's architecture against"
        )
    layers = {
        int(name.split(".")[2])
        for name in shapes
        if name.startswith("tfmr.layers.") and name.split(".")[2].isdigit()
    }

    def _shape(name: str) -> tuple[int, int]:
        """The *whole* shape of a layer-zero projection. Required, not optional.

        See ``docs/design/execution-config.md``.
        """
        shape = shapes.get(f"tfmr.layers.0.{name}.weight")
        if shape is None or len(shape) != 2:  # a projection is a matrix
            raise ValueError(
                f"{ckpt.path.name}: no 't3.tfmr.layers.0.{name}.weight' matrix. The "
                "manifest's architecture cannot be checked against weights that are "
                "not there, and a layer without it cannot be built either."
            )
        return (shape[0], shape[1])

    hidden = embed[1]
    raw_heads = llama_config.get("num_attention_heads")
    heads = raw_heads if isinstance(raw_heads, int) else None
    raw_kv = llama_config.get("num_key_value_heads", heads)
    kv_heads = raw_kv if isinstance(raw_kv, int) else None
    # `head_dim` is `hidden_size // num_attention_heads`, so a head count that does not
    # divide the hidden size describes no architecture at all.
    if heads is not None and (heads <= 0 or hidden % heads):
        raise ValueError(
            f"{ckpt.path.name}: manifest says num_attention_heads={heads!r}, which "
            f"does not divide hidden_size={hidden}. Refusing before building a model "
            "the checkpoint cannot fill."
        )
    head_dim = hidden // heads if heads is not None and heads > 0 else None

    # (field, value compared, value shown). The last is there because the two
    # head rows compare a *product*, a message reporting "4096" to someone who
    # wrote `num_key_value_heads: 1024` names a number they never typed.
    checks: list[tuple[str, object, object]] = [
        ("vocab_size", llama_config.get("vocab_size", 8), None),
        ("hidden_size", llama_config.get("hidden_size"), None),
        ("num_hidden_layers", llama_config.get("num_hidden_layers"), None),
    ]
    actuals: list[object] = [embed[0], hidden, len(layers)]
    # Whole shapes from here down. A projection in this architecture is always
    # `(something, hidden_size)`, so the declared row count and the known column
    # count together are the entire tensor, and checking only the rows let a
    # `(16_000_000, 0)` matrix of almost no bytes stand in for one of 65 GB.
    checks.append(("intermediate_size", (llama_config.get("intermediate_size"), hidden), None))
    actuals.append(_shape("mlp.gate_proj"))
    if head_dim is not None and heads is not None:
        checks.append(
            (
                "num_attention_heads",
                (heads * head_dim, hidden),
                f"{heads} x head_dim {head_dim}, hidden {hidden}",
            )
        )
        actuals.append(_shape("self_attn.q_proj"))
    if head_dim is not None and kv_heads is not None:
        checks.append(
            (
                "num_key_value_heads",
                (kv_heads * head_dim, hidden),
                f"{kv_heads} x head_dim {head_dim}, hidden {hidden}",
            )
        )
        actuals.append(_shape("self_attn.k_proj"))

    for (field_name, declared, shown), actual in zip(checks, actuals, strict=True):
        if declared != actual:
            raise ValueError(
                f"{ckpt.path.name}: manifest says {field_name}="
                f"{shown if shown is not None else repr(declared)} and the weights in the "
                f"same file say {actual}. Refusing before building a "
                "model the checkpoint cannot fill."
            )


@register_backend("cpu", "cuda", "mps")
def build_torch_engine(
    ckpt: Checkpoint, execution: ExecutionConfig, algorithm: AlgorithmConfig
) -> Engine:
    """Assemble the four synthesis components from a packed checkpoint.

    Weights load with ``strict=True`` throughout: the modules mirror the
    checkpoint's tensor names, and a mismatch means the architecture drifted:
    a load-time error, not a render-time mystery.
    """
    execution = execution.resolved()
    frontend, generator = build_torch_frontend_and_generator(ckpt, execution, algorithm)
    render_device = torch.device(execution.resolved_renderer_device())
    dtypes = _check_precision(execution)

    mel_decoder = TorchMelDecoder(
        algorithm,
        estimator_dtype=dtypes.get("mel_decoder.estimator", torch.float32),
        attention=execution.resolved_attention(),
    )
    mel_decoder.load_state_dict(_tensors(ckpt, "s3gen.flow."))
    mel_decoder = mel_decoder.to(render_device).eval()
    if mel_decoder.estimator_dtype is not torch.float32:
        mel_decoder.decoder.estimator.to(mel_decoder.estimator_dtype)

    # CUDA only. The measurement behind it is a CUDA one, and the CPU path is
    # the reference the conformance fixture is taken on, the same reason the
    # device-drawn noise stays off there. The condition lives on the config as
    # `resolved_vocoder_ragged` rather than here, so that `describe()` cannot
    # report a mode this line declined to build.
    vocoder = TorchVocoder(algorithm, ragged=execution.resolved_vocoder_ragged())
    vocoder.load_state_dict(_tensors(ckpt, "s3gen.mel2wav."))
    vocoder = vocoder.to(render_device).eval()

    for module in (mel_decoder, vocoder):
        for p in module.parameters():
            p.requires_grad_(False)

    return Engine(
        frontend=frontend,
        token_generator=generator,
        mel_decoder=mel_decoder,
        vocoder=vocoder,
        algorithm=algorithm,
        execution=execution,
        backend="torch",
        checkpoint_sha256=ckpt.file_digest,
        checkpoint_path=str(ckpt.path),
    )


def build_torch_enroller(
    path: str, *, device: str = "cpu", voice_encoder_weights: str | None = None
) -> TorchVoiceEnroller:
    """Build the enrollment pipeline from the enrollment checkpoint.

    See ``docs/design/execution-config.md``.
    """
    ckpt = Checkpoint.open(path)
    return TorchVoiceEnroller.from_checkpoint(
        ckpt, device=torch.device(device), voice_encoder_weights=voice_encoder_weights
    )
