"""The torch backend: one implementation, parameterised by device.

cpu, cuda and mps are one code path with three execution profiles — the model
modules are identical, and the differences (attention implementation,
determinism pinning) are declared through :class:`ExecutionConfig`, never
decided here. Two device facts are honoured rather than rediscovered:

* **MPS**: the fused scaled-dot-product path aborts the whole interpreter —
  ``LLVM ERROR: Failed to infer result type(s)`` from ``mps_matmul``, no
  Python traceback. ``ExecutionConfig.resolved_attention()`` returns
  ``eager`` there and this backend passes that straight to the generator.
* **CUDA determinism**: with ``ExecutionConfig.deterministic`` the backend
  pins ``cudnn.deterministic=True``, ``benchmark=False`` and both TF32 flags
  off. Without the first, the vocoder's conv stack drifts ~5e-06 between
  identical runs; without the TF32 pins, "fp32" silently is not (cudnn's flag
  defaults *on*).

Precision is applied per module from ``ExecutionConfig.precision`` and
validated against what measurement allows: the flow *encoder* and the vocoder
refuse fp16 (mel corr 0.619 / an audible Nyquist tone respectively — see the
module docstrings), the generator and flow estimator accept it.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch

from ..checkpoint import Checkpoint
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

TOKENIZER_FILENAME = "tokenizer.json"
"""The text tokenizer ships beside the checkpoint under this name."""

_PINNED: tuple[bool, bool] | None = None
"""The ``(allow_tf32, deterministic)`` this process last pinned, or None.

Kept so a second engine built with contradictory flags is reported rather than
applied in silence — see :func:`pin_determinism`.

**Process-global and never restored.** These are torch's own global switches;
pinning them changes every model in the interpreter, including ones this library
did not build, and nothing puts them back. A library reaching into a process's
global state is worth stating rather than discovering: an application that pins
determinism for a render has pinned it for whatever else it does afterwards.

Guarded by a lock because two threads building engines concurrently could both
read ``None`` and both pin, and the second would then never be reported as the
contradiction it is.
"""

_PIN_LOCK = threading.Lock()
"""Serialises the read-compare-write on ``_PINNED``. See above."""


def pin_determinism(execution: ExecutionConfig) -> None:
    """Apply the identity contract's backend flags (I-2).

    Costs ~5% end to end and buys "same seed, same build, bit-identical
    waveform". The flags are pinned even on non-CUDA hosts: they are
    process-global, harmless there, and pinning unconditionally means the
    recorded configuration is the running one.

    **This mutates process-global torch state.** Unrelated code in the same
    process inherits `cudnn.deterministic` and the TF32 setting, and building a
    second engine does not restore what the first one changed. That is a
    deliberate trade -- an engine running under someone else's flags
    could not honour the identity contract at all -- but it is a side effect,
    and callers who embed loudkit in a larger torch program should know it
    happens.
    """
    # Module-global on purpose: it mirrors torch's own process-global flags,
    # and there is exactly one process to track.
    global _PINNED  # noqa: PLW0603
    pins = (execution.allow_tf32, execution.deterministic)
    # Read, compare and write under one lock: two threads building engines at
    # once could otherwise both see `None`, both pin, and the second never be
    # reported as the contradiction it is.
    with _PIN_LOCK:
        previous = _PINNED
        _PINNED = pins
    if previous is not None and pins != previous:
        # These flags are process-global, so the second engine silently
        # re-pins them under the first one's feet: the first keeps reporting
        # its own `describe()` while computing under someone else's flags.
        # loudkit cannot make two contradictory engines both correct in one
        # process, but it can refuse to let the contradiction be invisible —
        # a recorded configuration that is not the running one is the failure
        # this library was built to end.
        import warnings

        was_tf32, was_det = previous
        warnings.warn(
            "a torch engine in this process was already pinned with "
            f"allow_tf32={was_tf32}, deterministic={was_det}; building one with "
            f"allow_tf32={execution.allow_tf32}, deterministic={execution.deterministic} "
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
    torch.backends.cuda.matmul.allow_tf32 = execution.allow_tf32
    torch.backends.cudnn.allow_tf32 = execution.allow_tf32
    # Symmetric on purpose. Only setting these under `deterministic` meant a
    # later non-deterministic engine inherited `cudnn.deterministic=True` from
    # an earlier one and ran ~5% slower than its own config describes.
    torch.backends.cudnn.deterministic = execution.deterministic
    torch.backends.cudnn.benchmark = not execution.deterministic
    if execution.num_threads is not None:
        torch.set_num_threads(execution.num_threads)


def _tensors(ckpt: Checkpoint, prefix: str) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors(prefix).items()}


def _check_precision(execution: ExecutionConfig) -> dict[str, torch.dtype]:
    prec = dict(execution.precision)
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
    """The first two stages only — text to speech tokens.

    Split out for the CoreML backend, which supplies its own renderer and used
    to obtain these by building a *whole* torch engine and discarding the mel
    decoder and vocoder it had just loaded. On the device where CoreML is the
    lightweight option, that was several hundred megabytes of weights read from
    disk, materialised, and freed — startup time and peak RSS spent on modules
    that never ran, and a plausible OOM on a phone.
    """
    pin_determinism(execution)
    # The generator's own device: an autoregressive step is a few hundred tiny
    # dispatches, so it often wants different hardware from the renderer. They
    # can be split at all only because the components pass arrays to each
    # other, never live tensors.
    gen_device = torch.device(execution.resolved_generator_device())
    dtypes = _check_precision(execution)

    # Packed copy first, sibling second — see `Checkpoint.resolve_asset`. A
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
        cuda_graphs=execution.cuda_graphs,
        compile_model=execution.compile_model,
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

    The manifest is data from outside the process, and three of its numbers --
    `vocab_size`, `hidden_size`, `num_hidden_layers` -- decide how much memory
    the model constructor asks for, before a single tensor is read. Unchecked,
    a twenty-kilobyte file that carries a manifest and almost nothing else can
    name an architecture large enough to exhaust the machine; `load_state_dict`
    would reject it a moment later, which is a moment too late.

    Bounds would be a weaker answer than this one. Any ceiling is either low
    enough to reject a legitimate future checkpoint or high enough to still be
    worth an attacker's while. The weights are the honest limit: they are in the
    same file, their shapes are in the header, and they cannot be inflated
    without inflating the file. A manifest claiming 100 000 layers beside 16
    layers' worth of tensors is refused for the reason that makes it wrong.

    Every field that drives allocation is checked, including the ones a state
    dict would appear to cover later: the *model constructor* builds the MLP
    before any state dict is loaded, so `intermediate_size` is spent the moment
    it is believed — "checked by `load_state_dict`" is too late for a field
    that sizes an allocation. A manifest naming
    16 000 000 beside 2100 layers' worth of tensors asked for 196 GB of
    `gate/up/down` weights and got past this function without a word.

    The corroboration is the same one, from the same header: `gate_proj` is
    `(intermediate_size, hidden_size)`, `q_proj` is
    `(num_attention_heads x head_dim, hidden_size)` and `k_proj` the same with
    the key-value count. Layer zero is enough -- the layers are uniform, and a
    file whose layers disagree fails `load_state_dict` without having allocated
    anything the header did not already promise.
    """
    shapes = ckpt.shapes("t3.")
    embed = shapes.get("tfmr.embed_tokens.weight")
    if embed is None or len(embed) != 2:  # noqa: PLR2004 - an embedding is a matrix
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

        That distinction is the whole check: making the comparison conditional
        on the tensor being present leaves a door open — a crafted file that
        *omits* `gate_proj` skips the `intermediate_size` check, so a
        twenty-six kilobyte checkpoint claiming `intermediate_size = 16_000_000`
        walks past preflight and asks the constructor for 197 GB. The bypass was
        one deleted tensor, in the function written to stop exactly this.

        There is nothing to be compatible with. A transformer layer without a
        gate projection or a query projection is not a layer this engine can
        build, and `load_state_dict` would say so afterwards -- after the
        allocation this function exists to prevent.

        Both dimensions, because comparing only the first was a hole of its
        own: a `(16_000_000, 0)` tensor weighs almost nothing on disk and
        satisfied an `intermediate_size` of sixteen million, after which the
        constructor asked for 197 GB. A projection's columns are the hidden
        size in every case here, so there is no reason not to check them.
        """
        shape = shapes.get(f"tfmr.layers.0.{name}.weight")
        if shape is None or len(shape) != 2:  # noqa: PLR2004 - a projection is a matrix
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
    # `head_dim` is `hidden_size // num_attention_heads`, so a head count that
    # does not divide the hidden size describes no architecture at all. Refused
    # here, by name, rather than through the product below: integer division
    # would silently make `head_dim` zero and the message would report a
    # mismatch of `0` against the weights, naming neither the field the caller
    # got wrong nor the value they wrote.
    if heads is not None and (heads <= 0 or hidden % heads):
        raise ValueError(
            f"{ckpt.path.name}: manifest says num_attention_heads={heads!r}, which "
            f"does not divide hidden_size={hidden}. Refusing before building a model "
            "the checkpoint cannot fill."
        )
    head_dim = hidden // heads if heads is not None and heads > 0 else None

    # (field, value compared, value shown). The last is there because the two
    # head rows compare a *product* — a message reporting "4096" to someone who
    # wrote `num_key_value_heads: 1024` names a number they never typed.
    checks: list[tuple[str, object, object]] = [
        ("vocab_size", llama_config.get("vocab_size", 8), None),
        ("hidden_size", llama_config.get("hidden_size"), None),
        ("num_hidden_layers", llama_config.get("num_hidden_layers"), None),
    ]
    actuals: list[object] = [embed[0], hidden, len(layers)]
    # Whole shapes from here down. A projection in this architecture is always
    # `(something, hidden_size)`, so the declared row count and the known column
    # count together are the entire tensor — and checking only the rows let a
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
    checkpoint's tensor names, and a mismatch means the architecture drifted —
    a load-time error, not a render-time mystery.
    """
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

    vocoder = TorchVocoder(algorithm)
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
    )


def build_torch_enroller(
    path: str, *, device: str = "cpu", voice_encoder_weights: str | None = None
) -> TorchVoiceEnroller:
    """Build the enrollment pipeline from the enrollment checkpoint.

    Separate from :func:`build_torch_engine` because enrollment loads the two
    tensor groups (speech tokenizer + speaker encoder) that synthesis never
    touches, and because its extra dependency surface (torchaudio's Kaldi
    fbank, librosa's mel filters) should not tax anyone who only synthesizes.

    Args:
        path: the enrollment checkpoint, or a pre-split one carrying its
            tensors.
        device: torch device for the enrollment models.
        voice_encoder_weights: the 256-d utterance voice encoder is *not* part
            of either checkpoint; pass its ``ve.safetensors`` here to
            enroll the token generator's speaker embedding. Without it,
            enrollment fails with an error naming this parameter.
    """
    ckpt = Checkpoint.open(path)
    return TorchVoiceEnroller.from_checkpoint(
        ckpt, device=torch.device(device), voice_encoder_weights=voice_encoder_weights
    )
