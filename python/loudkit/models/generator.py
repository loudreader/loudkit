"""The token generator: a Llama-architecture decoder that writes speech.

See ``docs/design/models-notes.md``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn

from ..config import AlgorithmConfig
from ..contracts import Sampler, SpeechTokens
from ..errors import CancelledError
from ..voice import EMOTION_NEUTRAL, VoiceProfile
from .windowing import START_TEXT_TOKEN, STOP_TEXT_TOKEN, eos_floor

# Typing note: torch types nn.Module.__call__ as Any, so submodule calls in a
# forward pass propagate Any. Where the callee's forward provably returns a
# Tensor, the return is wrapped in cast(Tensor, ...), an assertion about
# torch's contract, not a guess. See docs/design/typing.md.

__all__ = ["TorchTokenGenerator", "eos_floor"]

_ATTN = {"eager", "sdpa"}

_GRAPH_BUCKET = 64
"""KV buffer lengths are rounded up to this, so similar chunks share a graph."""


def _bucket(needed: int) -> int:
    """``needed`` slots rounded up to a whole :data:`_GRAPH_BUCKET`.

    The rounding is what makes a captured graph reusable, so all three slot
    builders have to round the same way: two chunks that round apart get two
    graphs and two sets of buffers where one would have done.
    """
    return -(-needed // _GRAPH_BUCKET) * _GRAPH_BUCKET


class _DeviceSampler(Protocol):
    """The device form of an injected sampler, as this module drives it.

    Structural rather than an import of :class:`~loudkit.sampler.DeviceSamplerV1`:
    ``models`` does not import ``sampler`` (``tests/test_import_graph.py`` holds
    the edge), and the decode loop accepts any sampler offering these members.

    ``host`` is opaque here. The loop never reads it, it only carries it from
    the sampler it was handed to :meth:`rebind` on the sampler a cached graph
    holds, so naming its type would buy nothing and cost the import.
    """

    @property
    def law(self) -> tuple[object, ...]:
        """What a captured graph baked in, as part of the slot's cache key."""

    @property
    def host(self) -> object:
        """The host sampler this device form draws with."""

    def rebind(self, host: object) -> None: ...

    def prepare(
        self,
        width: int,
        seen: NDArray[np.bool_],
        *,
        floor: int,
        step: int,
        stop_token: int | None = None,
    ) -> None: ...

    def select(self, logits: Tensor, *, valid: Tensor | None = None) -> Tensor: ...

    def advance_to(self, step: int, *, draws: int = 1) -> None: ...

    def flush_peak(self) -> None: ...


@runtime_checkable
class _OnDevice(Protocol):
    """A sampler that can hand back a capturable device form.

    ``Sampler`` does not carry ``on_device``, so the graph decode asks each
    injected sampler whether it has one. Runtime-checkable because the ask is
    the check: a sampler without the method keeps the host loop.

    ``device`` is annotated as the decode passes it. ``LRSamplerV1.on_device``
    declares ``str`` and then calls ``torch.device(device)``, which takes
    either, so the declaration is narrower than the code. That is a
    ``sampler.py`` matter, recorded rather than fixed here.
    """

    def on_device(
        self, device: torch.device, *, block_steps: int | None = None
    ) -> _DeviceSampler: ...


@dataclass
class _GraphSlot:
    """A captured decode step and every buffer whose address it holds.

    Kept together because that is the actual invariant: a CUDA graph records
    addresses, so reusing one means reusing *these* tensors, and a slot handed
    out with any of them replaced is a graph reading memory nobody writes.
    """

    max_len: int
    runner: Callable[[], None]
    buffers: dict[str, Tensor]
    keepalive: Callable[[], None] | None = None
    """The step closure. Not decoration: ``_capture_runner`` returns the
    graph's ``replay``, which holds no reference to the function that was
    captured, so the closure, and every tensor only it names, such as the KV
    write's index grid, is otherwise collected the moment the builder returns.
    The graph then replays over memory the allocator has handed to somebody
    else, and reports it as an index out of bounds nowhere near the cause."""

    extra: _DeviceSampler | None = None
    """The device sampler a ``fused_dev`` graph holds the buffers of. ``None``
    on every other slot, whose decode draws on the host."""


@dataclass(frozen=True, slots=True)
class _Decode:
    """One utterance as every decode loop takes it: the ten they all agree on.

    The loops differ in how they step, never in what they are decoding, so
    these ten are the same ten each time and :meth:`_static_decode` relays
    them without reading most. Naming them once means an eleventh shared thing
    is one field rather than five signature changes, and that no loop can be
    handed a ``floor`` that disagrees with the ``seen`` it was measured with.

    Frozen names the bindings, not the memory behind them: ``seen`` is the
    array the draw marks and ``cache`` holds the prefill's tensors, both
    written through as before. ``logits`` is the row the loop *opens* on; each
    loop takes a local from it and rebinds that local as it steps.

    Each loop unpacks what it uses and then reads as the arithmetic it is.
    """

    cap: int
    """The budget, counted in tokens, so one pair can carry a loop past it."""
    floor: int
    """Tokens that must exist before the stop token becomes sampleable."""
    stop: int
    seen: NDArray[np.bool_]
    """What has been drawn, the repetition penalty's memory. Written through."""
    prefill_len: int
    cache: list[tuple[Tensor, Tensor]]
    """The eager prefill's KV, which every static loop seeds its buffers from."""
    prefix_len: int
    """Speech tokens carried in from the window before, already in ``seen``."""
    sampler: Sampler
    logits: NDArray[np.float32]
    should_cancel: Callable[[], bool] | None

    def cancelled(self) -> bool:
        """Whether barge-in has fired, polled once per call.

        Every loop asks on every decode step, so an interrupt is honoured
        within one forward pass rather than at the next chunk boundary, some
        ten seconds of speech later. The token that was about to be sampled is
        discarded. A caller who passed nothing is never cancelled.
        """
        return self.should_cancel is not None and self.should_cancel()


def _cfg_int(cfg: Mapping[str, object], key: str, default: int | None = None) -> int:
    """Read an integer out of a JSON-shaped architecture dict, loudly.

    The manifest's ``llama_config`` arrives as ``dict[str, object]`` (it is
    parsed JSON); this narrows one value at a time instead of sprinkling
    ``type: ignore`` over every ``int(...)`` call.
    """
    value = cfg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"config[{key!r}] should be a number, got {value!r}")
    return int(value)


def _cfg_float(cfg: Mapping[str, object], key: str, default: float) -> float:
    value = cfg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"config[{key!r}] should be a number, got {value!r}")
    return float(value)


# --------------------------------------------------------------------- llama


class _RMSNorm(nn.Module):
    """Llama RMSNorm: normalise in fp32, scale in the module dtype.

    The fp32 round-trip is not an optimisation choice, it is what the weights
    were trained under, and in fp16 the variance of a 1024-wide activation
    genuinely overflows half precision.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


def _llama3_inv_freq(
    head_dim: int,
    theta: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_len: int,
) -> Tensor:
    """RoPE inverse frequencies with the llama3 wavelength-dependent rescale.

    Short wavelengths keep their frequency, wavelengths beyond the original
    training context are slowed by ``factor``, and the band between is blended
    smoothly, verbatim the published llama3 rule, kept in fp64 until the end
    so two implementations of "the same formula" cannot round differently.
    """
    exponents = np.arange(0, head_dim, 2, dtype=np.float64) / head_dim
    inv_freq = 1.0 / (theta**exponents)
    wavelen = 2.0 * np.pi / inv_freq
    low_wavelen = original_max_len / low_freq_factor
    high_wavelen = original_max_len / high_freq_factor
    smooth = (original_max_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    blended = (1.0 - smooth) * inv_freq / factor + smooth * inv_freq
    out = np.where(
        wavelen < high_wavelen,
        inv_freq,
        np.where(wavelen > low_wavelen, inv_freq / factor, blended),
    )
    return torch.from_numpy(out).float()


def _position(index: int, device: torch.device) -> Tensor:
    """``[index]`` on ``device``, built there rather than copied there.

    ``torch.tensor([index], device=...)`` stages a host buffer and copies it
    across, which on CUDA is a synchronising transfer for eight bytes. The
    range is a device-side kernel with the same values.
    """
    return torch.arange(index, index + 1, device=device)


def _row(weight: Tensor, index: int) -> Tensor:
    """Row ``index`` of ``weight``, as a view rather than a lookup.

    The decode reads embedding rows this way to skip the host-to-device copy
    an index tensor costs per step. Indexing keeps the upper bound the lookup
    had, where a slice would not: ``weight[n : n + 1]`` past the end returns no
    rows and broadcasts away into a silently empty answer. It does not keep the
    lower bound, since ``weight[-1]`` is the last row where the lookup raised,
    so that end is checked here.
    """
    if index < 0:
        raise IndexError(
            f"index {index} is out of bounds for dimension 0 with size {weight.shape[0]}"
        )
    return weight[index]


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _Attention(nn.Module):
    """Grouped-query attention: 16 query heads sharing 4 KV heads."""

    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: tuple[Tensor, Tensor] | None,
        *,
        causal: bool,
        attention: str,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)

        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        if attention == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal and t > 1)
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            if causal and t > 1:
                mask = torch.full(
                    (t, k.shape[2]), float("-inf"), device=x.device, dtype=scores.dtype
                ).triu(k.shape[2] - t + 1)
                scores = scores + mask
            out = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype) @ v
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out), new_cache

    def forward_static(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        k_buf: Tensor,
        v_buf: Tensor,
        grid: tuple[Tensor, Tensor, Tensor],
        pos: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """One decode step against a fixed-shape KV buffer (graph-capturable).

        ``mask`` is the buffer's padding, built once per step by the stack: it
        depends on the step and the buffer length, not on the layer.

        See ``docs/design/models-notes.md``.
        """
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        batch_idx, kv_idx, head_idx = grid
        k_buf.index_put_(
            (batch_idx, kv_idx, pos.expand_as(batch_idx), head_idx),
            k[0, :, 0, :].reshape(-1),
        )
        v_buf.index_put_(
            (batch_idx, kv_idx, pos.expand_as(batch_idx), head_idx),
            v[0, :, 0, :].reshape(-1),
        )

        rep = self.n_heads // self.n_kv_heads
        kk = k_buf.repeat_interleave(rep, dim=1)
        vv = v_buf.repeat_interleave(rep, dim=1)

        scores = q @ kk.transpose(-2, -1) / math.sqrt(self.head_dim)
        out = torch.softmax(scores + mask, dim=-1, dtype=torch.float32).to(q.dtype) @ vv
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return cast(Tensor, self.o_proj(out))


class _MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x)). Intermediate width 2100."""

    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        intermediate: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.self_attn = _Attention(hidden, n_heads, n_kv_heads, head_dim)
        self.mlp = _MLP(hidden, intermediate)
        self.input_layernorm = _RMSNorm(hidden, norm_eps)
        self.post_attention_layernorm = _RMSNorm(hidden, norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: tuple[Tensor, Tensor] | None,
        *,
        causal: bool,
        attention: str,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attn_out, new_cache = self.self_attn(
            self.input_layernorm(x), cos, sin, cache, causal=causal, attention=attention
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache

    def forward_static(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        k_buf: Tensor,
        v_buf: Tensor,
        grid: tuple[Tensor, Tensor, Tensor],
        pos: Tensor,
        mask: Tensor,
    ) -> Tensor:
        attn_out = self.self_attn.forward_static(
            self.input_layernorm(x), cos, sin, k_buf, v_buf, grid, pos, mask
        )
        x = x + attn_out
        return cast(Tensor, x + self.mlp(self.post_attention_layernorm(x)))


class LlamaDecoder(nn.Module):
    """The bare decoder stack (checkpoint namespace ``t3.tfmr``).

    Consumes pre-built input embeddings, token/positional embedding is the
    conditioning layer's business, and returns final hidden states. The KV
    cache is a plain list of per-layer ``(k, v)`` tensors: explicit, portable,
    and free of library cache classes whose semantics shift between releases.
    """

    inv_freq: Tensor  # registered buffer; annotated so access is not Tensor | Module

    def __init__(self, cfg: dict[str, object]) -> None:
        super().__init__()
        hidden = _cfg_int(cfg, "hidden_size")
        head_dim = _cfg_int(cfg, "head_dim", 64)
        self.n_layers = _cfg_int(cfg, "num_hidden_layers")
        self.head_dim = head_dim
        rope = cfg.get("rope_scaling") or {}
        # Raised, not asserted: `python -O` strips asserts, and a checkpoint
        # whose `rope_scaling` is a list or a string would then fail on the next
        # line with an `AttributeError` about `.get`, naming neither the field
        # nor the file. A malformed checkpoint is data from outside the process.
        if not isinstance(rope, dict):
            raise ValueError(
                f"checkpoint config: rope_scaling must be an object, got {type(rope).__name__}"
            )
        self.embed_tokens = nn.Embedding(_cfg_int(cfg, "vocab_size", 8), hidden)
        self.layers = nn.ModuleList(
            _DecoderLayer(
                hidden,
                _cfg_int(cfg, "num_attention_heads"),
                _cfg_int(cfg, "num_key_value_heads"),
                head_dim,
                _cfg_int(cfg, "intermediate_size"),
                _cfg_float(cfg, "rms_norm_eps", 1e-5),
            )
            for _ in range(self.n_layers)
        )
        self.norm = _RMSNorm(hidden, _cfg_float(cfg, "rms_norm_eps", 1e-5))
        inv_freq = _llama3_inv_freq(
            head_dim,
            _cfg_float(cfg, "rope_theta", 500_000.0),
            _cfg_float(rope, "factor", 8.0),
            _cfg_float(rope, "low_freq_factor", 1.0),
            _cfg_float(rope, "high_freq_factor", 4.0),
            _cfg_int(rope, "original_max_position_embeddings", 8192),
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """cos/sin for the given absolute positions, computed in fp32.

        fp32 here regardless of module dtype: at position ~500 the angle spans
        hundreds of radians and fp16 cannot hold it without visible phase error.
        """
        freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]

    def forward(
        self,
        inputs_embeds: Tensor,
        positions: Tensor,
        cache: list[tuple[Tensor, Tensor]] | None,
        *,
        attention: str,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        if attention not in _ATTN:
            # Raised, not asserted: `python -O` strips an assert, and the
            # next line would then read a kernel this stack does not have.
            raise ValueError(f"attention must be one of {sorted(_ATTN)}: {attention!r}")
        cos, sin = self._rope(positions, inputs_embeds.dtype)
        x = inputs_embeds
        new_cache: list[tuple[Tensor, Tensor]] = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(
                x,
                cos,
                sin,
                cache[i] if cache else None,
                causal=cache is None,
                attention=attention,
            )
            new_cache.append(kv)
        return self.norm(x), new_cache

    def forward_static(
        self,
        inputs_embeds: Tensor,
        positions: Tensor,
        k_bufs: Tensor,
        v_bufs: Tensor,
        grid: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """One decode step into preallocated ``[n_layers, 1, n_kv, max_len, hd]``
        KV buffers. Writes layer ``i``'s key/value into column ``positions[0]``
        in place via ``index_put_`` with the shared preallocated ``grid``.
        Fixed addresses and shapes throughout, so this callable can be
        captured by a CUDA graph, the ``cuda_graphs`` execution flag.
        """
        cos, sin = self._rope(positions, inputs_embeds.dtype)
        # One mask for the stack. It is a function of the step and the buffer
        # length, not of the layer, so building it inside the layer built the
        # same small tensor once per layer per decode step, and a captured
        # graph replayed every copy.
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        mask = torch.where(
            torch.arange(k_bufs.shape[3], device=device) > positions,
            torch.full((), float("-inf"), device=device, dtype=dtype),
            torch.zeros((), device=device, dtype=dtype),
        )
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x = cast(_DecoderLayer, layer).forward_static(
                x, cos, sin, k_bufs[i], v_bufs[i], grid, positions, mask
            )
        return cast(Tensor, self.norm(x))


# ------------------------------------------------------------- conditioning


class _PerceiverAttention(nn.Module):
    """One shared attention block used for both perceiver passes.

    Both inputs go through the *same* LayerNorm and the same q/k/v
    projections: one block, called twice, so the weights exist once in the
    checkpoint and both passes have to read them under the same names.
    """

    def __init__(self, dim: int, n_heads: int, attention: str = "sdpa") -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.attention = attention
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        b, t, _ = x1.shape
        q = self.to_q(self.norm(x1)).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        x2n = self.norm(x2)
        k = self.to_k(x2n).view(b, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x2n).view(b, -1, self.n_heads, self.head_dim).transpose(1, 2)
        # SDPA lowers to flash-attention on CUDA, which needs Ampere or newer
        # (compute >= 8.0). On older GPUs the fused path raises mid-decode; the
        # eager matmul path below is the same mathematics, written portably:
        # the mirror of _Attention's eager branch, and non-causal here so no
        # mask is needed.
        if self.attention == "eager":
            scale = float(self.head_dim) ** -0.5
            attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
            out = attn @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return x1 + cast(Tensor, self.proj_out(out))


class _Perceiver(nn.Module):
    """Resample the ~150-token speech prompt to 32 learned query slots:
    one cross-attention pass from the queries into the prompt, then one
    self-attention pass over the result."""

    def __init__(
        self, dim: int = 1024, n_query: int = 32, n_heads: int = 4, attention: str = "sdpa"
    ) -> None:
        super().__init__()
        # zeros, not `torch.empty`: this is the one parameter in the file with no
        # initialiser of its own, and uninitialised memory would make the module's
        # output depend on whatever the allocator last wrote there.
        self.pre_attention_query = nn.Parameter(torch.zeros(1, n_query, dim))
        self.attn = _PerceiverAttention(dim, n_heads, attention)

    def forward(self, prompt_emb: Tensor) -> Tensor:
        query = self.pre_attention_query.expand(prompt_emb.shape[0], -1, -1)
        resampled = self.attn(query, prompt_emb)
        return cast(Tensor, self.attn(resampled, resampled))


class _CondEncoder(nn.Module):
    """Non-text conditioning (checkpoint namespace ``t3.cond_enc``):
    ``[speaker (1), resampled prompt (32), emotion (1)]`` -> 34 slots."""

    def __init__(self, hidden: int, speaker_dim: int, attention: str = "sdpa") -> None:
        super().__init__()
        self.spkr_enc = nn.Linear(speaker_dim, hidden)
        self.emotion_adv_fc = nn.Linear(1, hidden, bias=False)
        self.perceiver = _Perceiver(hidden, attention=attention)

    def forward(self, speaker_emb: Tensor, prompt_emb: Tensor, emotion: Tensor) -> Tensor:
        spkr = self.spkr_enc(speaker_emb.view(1, 1, -1))
        prompt = self.perceiver(prompt_emb)
        emo = self.emotion_adv_fc(emotion.view(1, 1, 1))
        return torch.cat((spkr, prompt, emo), dim=1)


# ----------------------------------------------------------------- generator


@dataclass(frozen=True, slots=True)
class DecodeGeometry:
    """The KV cache's shape, for a caller sizing buffers of its own.

    See ``docs/design/models-notes.md``.
    """

    n_layers: int
    n_kv_heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype


def check_manifest_sizes(
    config: AlgorithmConfig, *, speech_vocab: int, start_token: int
) -> None:
    """Refuse a manifest whose speech vocabulary disagrees with the weights.

    See ``docs/design/models-notes.md``.
    """
    declared_vocab = config.speech_vocab_size
    declared_start = config.start_speech_token
    if declared_vocab != speech_vocab or declared_start != start_token:
        raise ValueError(
            f"manifest declares speech_vocab_size={declared_vocab}, "
            f"start_speech_token={declared_start}; these weights carry "
            f"{speech_vocab} and {start_token}. A different vocabulary is a "
            "different checkpoint: re-export rather than re-declare."
        )


class _LearnedPositions(nn.Module):
    """Learned absolute positions (GPT-2 style), one table per segment kind."""

    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(max_len, dim)

    def range(self, length: int, device: torch.device, *, start: int = 0) -> Tensor:
        return cast(Tensor, self.emb(torch.arange(start, start + length, device=device)))

    def at(self, position: int) -> Tensor:
        """One row as ``(1, 1, dim)``. A read-only view, for the reason in :func:`_row`."""
        return _row(self.emb.weight, position)[None, None]

    def at_buf(self, pos: Tensor) -> Tensor:
        """Same lookup from a buffer, the graph-capturable variant of :meth:`at`."""
        return cast(Tensor, self.emb(pos.view(1, 1)))


class TorchTokenGenerator(nn.Module):
    """``TokenGenerator`` implementation on torch (cpu / cuda / mps).

    See ``docs/design/models-notes.md``.
    """

    # The shipped weights' dimensions, and a manifest that disagrees is refused rather
    # than silently overridden, see `check_manifest_sizes`.
    SPEECH_VOCAB = 8194
    START_SPEECH = 6561
    """The speech-start marker in these weights, and the ceiling for a prompt
    token: prompts index the codebook below it, conditioning the whole vocabulary."""
    TEXT_VOCAB = 2454
    MAX_TEXT_POSITIONS = 2050
    MAX_SPEECH_POSITIONS = 4100

    def __init__(
        self,
        config: AlgorithmConfig,
        llama_config: dict[str, object],
        *,
        attention: str = "sdpa",
        speaker_dim: int = 256,
        cuda_graphs: bool = False,
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        if attention not in _ATTN:
            raise ValueError(f"attention must be one of {sorted(_ATTN)}: {attention!r}")
        check_manifest_sizes(
            config, speech_vocab=self.SPEECH_VOCAB, start_token=self.START_SPEECH
        )
        self.config = config
        self.attention = attention
        self.cuda_graphs = cuda_graphs
        self.compile_model = compile_model
        self._cond_cache: dict[tuple[bytes, bytes], Tensor] = {}
        """Conditioning rows by :meth:`VoiceProfile.cond_key`.

        The row is a pure function of the profile's speaker embedding and
        conditioning tokens, and every chunk of every request in the same
        voice recomputes it, a speaker projection plus two perceiver passes.
        Execution-layer memoisation: the cached tensor is what the computation
        would have produced, bit for bit, on this device and dtype. Capped
        small; an engine rarely sees more than a couple of voices at once.
        """
        self._cond_lock = threading.Lock()
        """Serialises the read-promote-evict-insert on `_cond_cache`.

        See ``docs/design/models-notes.md``.
        """
        self._graphs: dict[tuple[object, ...], _GraphSlot] = {}
        """Captured decode graphs, by kind and buffer length.

        See ``docs/design/models-notes.md``.
        """
        self._decoding = threading.Lock()
        """Held for the length of one static-cache ``generate``.

        See ``docs/design/models-notes.md``.
        """

        hidden = _cfg_int(llama_config, "hidden_size")

        self.tfmr = LlamaDecoder(llama_config)
        self.cond_enc = _CondEncoder(hidden, speaker_dim, attention=attention)
        self.text_emb = nn.Embedding(self.TEXT_VOCAB, hidden)
        self.speech_emb = nn.Embedding(self.SPEECH_VOCAB, hidden)
        self.text_pos_emb = _LearnedPositions(self.MAX_TEXT_POSITIONS, hidden)
        self.speech_pos_emb = _LearnedPositions(self.MAX_SPEECH_POSITIONS, hidden)
        self.text_head = nn.Linear(hidden, self.TEXT_VOCAB, bias=False)
        self.speech_head = nn.Linear(hidden, self.SPEECH_VOCAB, bias=False)

        # Built only for `fusion_mtp2`, and named to match the packed tensors,
        # so a mode/weights disagreement is a loud `load_state_dict` error in
        # either direction: weights without the mode are unexpected keys, the
        # mode without weights is missing ones.
        if config.decode == "fusion_mtp2":
            self.head2 = nn.Linear(2 * hidden, self.SPEECH_VOCAB, bias=False)
            self.fuse = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )

    # -- assembly ------------------------------------------------------------

    @property
    def _device(self) -> torch.device:
        return self.speech_head.weight.device

    @property
    def _dtype(self) -> torch.dtype:
        return self.speech_head.weight.dtype

    def _speech_row(self, token: int) -> Tensor:
        """Token ``token``'s embedding as ``(1, dim)``, straight out of the table.

        A row view rather than ``speech_emb(tensor([[token]]))``: the index
        tensor is built on the host, so the lookup costs a host-to-device copy
        per decode step. Same values, and on a device whose host is slow that
        copy costs more than the lookup it feeds. See :func:`_row` for the
        bounds. Read only: it aliases the parameter.
        """
        return _row(self.speech_emb.weight, token)[None].to(self._dtype)

    def _speech_token_embed(self, token: int, position: int) -> Tensor:
        return self._speech_row(token)[None] + self.speech_pos_emb.at(position)

    def _fuse_pair(self, e_a: Tensor, e_b: Tensor) -> Tensor:
        """Two speech embeddings as the one slot they share under ``fusion_mtp2``.

        Their mean plus what ``fuse`` reads off the pair, and the arithmetic
        every fused path has to agree on: the two prefix builders, the eager
        pair step, both captured pair steps, and teacher forcing. Only the
        trailing dimension is the pair's, so a single slot and a whole row of
        them both come through here.
        """
        return 0.5 * (e_a + e_b) + cast(Tensor, self.fuse(torch.cat((e_a, e_b), dim=-1)))

    def _pair_slot_embed(self, first: int, second: int, position: int) -> Tensor:
        """The KV slot a decoded pair occupies under ``fusion_mtp2``.

        See ``docs/design/models-notes.md``.
        """
        e_a, e_b = self._speech_row(first)[None], self._speech_row(second)[None]
        return self._fuse_pair(e_a, e_b) + self.speech_pos_emb.at(position)

    def _pair_second_logits(self, hidden: Tensor, first: int) -> NDArray[np.float32]:
        """Logits for a pair's second token, read from ``[hidden ; e(first)]``."""
        both = torch.cat((hidden, self._speech_row(first)), dim=-1)
        return cast(NDArray[np.float32], self.head2(both).float().cpu().numpy())

    def decode_geometry(self) -> DecodeGeometry:
        """The KV cache's shape, for a caller sizing buffers of its own.

        See :class:`DecodeGeometry`: published so a benchmark can allocate a
        cache without reading three levels into this module's internals.
        """
        # `nn.ModuleList` indexes as `Tensor | Module`, so the attention block
        # is named explicitly for the type checker; the cast documents what the
        # decoder actually holds.
        first = cast(_DecoderLayer, self.tfmr.layers[0])
        return DecodeGeometry(
            n_layers=self.tfmr.n_layers,
            n_kv_heads=first.self_attn.n_kv_heads,
            head_dim=self.tfmr.head_dim,
            device=self._device,
            dtype=self._dtype,
        )

    def prefill_embeds(self, text_tokens: NDArray[np.int64], voice: VoiceProfile) -> Tensor:
        """The conditioning + text embedding row a decode starts from.

        Public for the same reason as :meth:`decode_geometry`: a benchmark needs
        the real prefill to measure anything meaningful, and calling
        ``_prefill_embeds`` from a tool made the underscore a lie.
        """
        return self._prefill_embeds(text_tokens, voice)

    def _prefill_embeds(self, text_tokens: NDArray[np.int64], voice: VoiceProfile) -> Tensor:
        """``[cond | START text STOP | speech START]`` as one embedding row."""
        device, dtype = self._device, self._dtype

        key = voice.cond_key()
        with self._cond_lock:
            cond = self._cond_cache.get(key)
            if cond is not None:
                self._cond_cache[key] = self._cond_cache.pop(key)  # true LRU
        if cond is None:
            speaker = torch.from_numpy(np.asarray(voice.speaker_embedding)).to(device, dtype)
            emotion = torch.tensor(EMOTION_NEUTRAL, device=device, dtype=dtype)
            cond_tokens = torch.from_numpy(np.asarray(voice.cond_prompt_tokens)).to(device)[
                None
            ]
            prompt_emb = self.speech_emb(cond_tokens) + self.speech_pos_emb.range(
                cond_tokens.shape[1], device
            )
            # Computed outside the lock: it is a forward pass, and holding a
            # lock across one would serialise two engines that share nothing
            # but a cache. Two threads may compute the same row and the second
            # insert wins, which costs one recomputation and is right either
            # way, the row is a pure function of the key.
            cond = self.cond_enc(speaker, prompt_emb, emotion)
            with self._cond_lock:
                # Looked up again under the lock.
                existing = self._cond_cache.get(key)
                if existing is not None:
                    cond = existing
                else:
                    if len(self._cond_cache) >= 8:
                        self._cond_cache.pop(next(iter(self._cond_cache)))
                    self._cond_cache[key] = cond

        framed = np.concatenate(([START_TEXT_TOKEN], text_tokens, [STOP_TEXT_TOKEN]))
        tt = torch.from_numpy(framed.astype(np.int64)).to(device)[None]
        if tt.shape[1] > self.MAX_TEXT_POSITIONS:
            # Checked before the embedding rather than discovered inside it. The
            # table has `MAX_TEXT_POSITIONS` rows, so a longer prompt came back
            # as a bare `IndexError` from a positional-embedding lookup, a
            # message naming neither the text, the limit, nor the method that
            # exists precisely for this case.
            raise ValueError(
                f"{tt.shape[1]} text tokens exceeds the "
                f"{self.MAX_TEXT_POSITIONS}-position table; use "
                "`Engine.synthesize`, which splits at sentence boundaries"
            )
        text = self.text_emb(tt) + self.text_pos_emb.range(tt.shape[1], device)

        bos = self._speech_token_embed(self.config.start_speech_token, 0)
        return torch.cat((cond, text, bos), dim=1).to(dtype)

    def _head_logits(self, hidden: Tensor) -> NDArray[np.float32]:
        """Speech logits in fp32 on the host, the sampler's world is numpy."""
        # Tensor.numpy() is untyped in torch's stubs; .float() guarantees fp32.
        return cast(NDArray[np.float32], self.speech_head(hidden).float().cpu().numpy())

    @staticmethod
    def _draw(
        logits: NDArray[np.float32],
        out: list[int],
        sampler: Sampler,
        *,
        floor: int,
        stop: int,
        seen: NDArray[np.bool_],
    ) -> int:
        """Mask the floor, draw one token onto ``out``, and hand it back.

        What the five decode loops do between draws is what makes them five
        loops; the draw itself they agree on completely. Below the floor the
        stop token is out of reach, the sampler is asked at the step ``out``
        has reached, and a token that did not end the sequence is marked seen.

        The stop test stays with the caller, because what follows it is the
        part that differs: the eager loops break, the pair loops have a second
        token to decide about, and the on-device loop has a graph to seed.
        """
        if len(out) < floor:
            logits[stop] = -np.inf
        token = sampler(logits, step=len(out), seen=seen)
        out.append(token)
        if token != stop:
            seen[token] = True
        return token

    # -- contract ------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        text_tokens: NDArray[np.int64],
        voice: VoiceProfile,
        *,
        sampler: Sampler,
        max_new_tokens: int | None = None,
        prefix: SpeechTokens = (),
        should_cancel: Callable[[], bool] | None = None,
    ) -> SpeechTokens:
        """Autoregressive decode to the stop token or the cap.

        See ``docs/design/models-notes.md``.
        """
        # `is None`, not `or`: the parameter is `int | None`, so None is the
        # only "unset". Under `or`, a caller asking for 0 got the manifest's
        # couple of thousand instead. Go, Rust, JS and Swift all test the
        # option rather than the value.
        cap = self.config.sampling.max_new_tokens if max_new_tokens is None else max_new_tokens
        floor = eos_floor(len(text_tokens), self.config)
        stop = self.config.stop_speech_token

        embeds = self._prefill_embeds(text_tokens, voice)
        seen = np.zeros(self.SPEECH_VOCAB, dtype=bool)

        prefix = [int(t) for t in prefix]
        fused = self.config.decode == "fusion_mtp2"
        if prefix and fused:
            # One slot per pair, so an odd carry-over cannot be represented.
            prefix = prefix[: len(prefix) - len(prefix) % 2]
        if prefix:
            toks = torch.tensor([prefix], device=self._device, dtype=torch.long)
            if fused:
                pairs = cast(Tensor, self.speech_emb(toks)).to(self._dtype)
                spe = self._fuse_pair(pairs[:, 0::2], pairs[:, 1::2])
                spe = spe + self.speech_pos_emb.range(len(prefix) // 2, self._device, start=1)
            else:
                spe = cast(Tensor, self.speech_emb(toks)) + self.speech_pos_emb.range(
                    len(prefix), self._device, start=1
                )
            embeds = torch.cat((embeds, spe.to(self._dtype)), dim=1)
            seen[prefix] = True

        prefill_len = embeds.shape[1]
        positions = torch.arange(prefill_len, device=self._device)
        hidden, cache = self.tfmr(embeds, positions, None, attention=self.attention)
        logits = self._head_logits(hidden[:, -1])[0]

        plan = _Decode(
            cap=cap,
            floor=floor,
            stop=stop,
            seen=seen,
            prefill_len=prefill_len,
            cache=cache,
            prefix_len=len(prefix),
            sampler=sampler,
            logits=logits,
            should_cancel=should_cancel,
        )

        if self.cuda_graphs or self.compile_model:
            # See `_decoding`: from here down the buffers are the generator's,
            # not this call's.
            if not self._decoding.acquire(blocking=False):
                raise RuntimeError(
                    "this generator is already decoding. With `cuda_graphs` or "
                    "`compile_model` the KV cache, the positions and the "
                    "sampler's noise are per-generator buffers a captured graph "
                    "holds by address, so a second concurrent `generate` would "
                    "overwrite the first's state and both callers would get "
                    "fluent speech that is not their text. Use one engine per "
                    "thread, or serialise the calls as the shipped transports do."
                )
            try:
                return self._static_decode(plan, hidden, fused=fused)
            finally:
                self._decoding.release()

        if fused:
            return self._generate_fused(plan, hidden)

        out: list[int] = []
        for step in range(cap):
            if plan.cancelled():
                # Token-level barge-in: the partial row is discarded, not
                # returned. `docs/reference/errors.md` says a cancelled
                # `synthesize` raises, and the four ports raise here. Returning
                # short instead left the caller to poll the flag a second time,
                # which a flag that is true once and consumed by its first
                # reader does not survive: the truncated row went on to the
                # detectors, the retry ladder fired, and the call returned
                # different speech rather than refusing.
                raise CancelledError(f"at decode step {step}: cancelled")
            token = self._draw(logits, out, sampler, floor=floor, stop=stop, seen=seen)
            if token == stop:
                break
            emb = self._speech_token_embed(token, len(prefix) + step + 1).to(self._dtype)
            pos = _position(prefill_len + step, self._device)
            hidden, cache = self.tfmr(emb, pos, cache, attention=self.attention)
            logits = self._head_logits(hidden[:, -1])[0]
        return out

    def _static_decode(self, plan: _Decode, hidden: Tensor, *, fused: bool) -> SpeechTokens:
        """Pick the static-cache loop this build and this sampler can run.

        See ``docs/design/models-notes.md``.
        """
        if not fused:
            return self._generate_static(plan)
        on_device = self._device_sampler(plan.sampler)
        if on_device is None:
            return self._generate_static_fused(plan, hidden)
        return self._generate_static_fused_ondevice(plan, hidden, device_sampler=on_device)

    def _generate_fused(self, plan: _Decode, hidden: Tensor) -> SpeechTokens:
        """Two tokens per forward: ``speech_head``, then ``head2``, then fuse.

        See ``docs/design/models-notes.md``.
        """
        cap, floor, stop, seen = plan.cap, plan.floor, plan.stop, plan.seen
        sampler, prefill_len, prefix_len = plan.sampler, plan.prefill_len, plan.prefix_len
        logits, cache = plan.logits, plan.cache

        out: list[int] = []
        pair = 0
        while len(out) < cap:
            if plan.cancelled():
                break
            first = self._draw(logits, out, sampler, floor=floor, stop=stop, seen=seen)
            if first == stop:
                break

            if len(out) >= cap:
                break
            second_logits = self._pair_second_logits(hidden[:, -1], first)[0]
            second = self._draw(second_logits, out, sampler, floor=floor, stop=stop, seen=seen)
            if second == stop:
                break

            slot = self._pair_slot_embed(first, second, prefix_len // 2 + pair + 1)
            pos = _position(prefill_len + pair, self._device)
            hidden, cache = self.tfmr(slot, pos, cache, attention=self.attention)
            logits = self._head_logits(hidden[:, -1])[0]
            pair += 1
        return out

    def _static_kv(self, max_len: int) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        """Static KV buffers for a decode of at most ``max_len`` slots.

        The grid is a preallocated index triple for the in-place KV write:
        one fixed (batch, kv, head) set shared by every layer and every step,
        so the decode step allocates nothing and can be captured by a graph.
        """
        n_kv = cast(_DecoderLayer, self.tfmr.layers[0]).self_attn.n_kv_heads
        head_dim = self.tfmr.head_dim
        device, dtype = self._device, self._dtype
        shape = (self.tfmr.n_layers, 1, n_kv, max_len, head_dim)
        k_bufs = torch.zeros(shape, device=device, dtype=dtype)
        grid = (
            torch.zeros(n_kv * head_dim, dtype=torch.long, device=device),
            torch.arange(n_kv, device=device).repeat_interleave(head_dim).contiguous(),
            torch.arange(head_dim, device=device).repeat(n_kv).contiguous(),
        )
        return k_bufs, torch.zeros_like(k_bufs), grid

    @staticmethod
    def _seed_static_kv(
        slot: _GraphSlot, cache: list[tuple[Tensor, Tensor]], prefill_len: int
    ) -> None:
        """Write the eager prefill's KV into a slot's static buffers.

        Every static loop does this once, after fetching its slot and never
        before: capture (warm-up plus graph) runs the step a few times with
        pos=0, which writes garbage into column 0 of the buffers, so the real
        prefill KV is seeded afterwards. At replay the write column comes from
        the ``rope_pos`` device value, not the value frozen at capture time.
        """
        k_bufs, v_bufs = slot.buffers["k"], slot.buffers["v"]
        for i, (k, v) in enumerate(cache):
            k_bufs[i, 0, :, :prefill_len, :].copy_(k[0])
            v_bufs[i, 0, :, :prefill_len, :].copy_(v[0])

    def _single_slot(self, needed: int) -> _GraphSlot:
        """A captured single-token step, cached by buffer length.

        Same reasoning as :meth:`_fused_device_slot`, and the same reason it is
        worth having: capture is a fixed cost per ``generate`` call, and a long
        read should not pay it once per chunk. Every single-token checkpoint
        decodes on this path, so the saving is not turbo's alone.
        """
        bucket = _bucket(needed)
        key = ("single", bucket)
        slot = self._graphs.get(key)
        if slot is not None:
            return slot

        device = self._device
        k_bufs, v_bufs, grid = self._static_kv(bucket)
        token_buf = torch.zeros(1, 1, dtype=torch.long, device=device)
        emb_pos_buf = torch.zeros(1, dtype=torch.long, device=device)
        rope_pos_buf = torch.zeros(1, dtype=torch.long, device=device)
        logits_buf = torch.zeros(1, self.SPEECH_VOCAB, dtype=torch.float32, device=device)

        def step() -> None:
            emb = self.speech_emb(token_buf) + self.speech_pos_emb.at_buf(emb_pos_buf)
            emb = emb.to(self._dtype)
            hidden = self.tfmr.forward_static(emb, rope_pos_buf, k_bufs, v_bufs, grid)
            logits_buf.copy_(self.speech_head(hidden[:, -1]).float())

        slot = _GraphSlot(
            max_len=bucket,
            runner=self._capture_runner(step),
            keepalive=step,
            buffers={
                "k": k_bufs,
                "v": v_bufs,
                "token": token_buf,
                "emb_pos": emb_pos_buf,
                "rope_pos": rope_pos_buf,
                "logits": logits_buf,
                "grid_batch": grid[0],
                "grid_kv": grid[1],
                "grid_head": grid[2],
            },
        )
        self._graphs[key] = slot
        return slot

    def _generate_static(self, plan: _Decode) -> SpeechTokens:
        """The static-cache decode loop (``cuda_graphs``/``compile_model``).

        KV buffers are sized to this utterance's ``prefill_len + cap``, seeded
        from the eager prefill's ``cache``, then each step writes one token's
        key/value in place and reads logits back. The step is either a captured
        CUDA graph, a compiled function, or, on machines without CUDA, the
        same function run eagerly, so the math is identical in all three.
        """
        cap, floor, stop, seen = plan.cap, plan.floor, plan.stop, plan.seen
        sampler, prefill_len, prefix_len = plan.sampler, plan.prefill_len, plan.prefix_len
        logits, cache = plan.logits, plan.cache

        slot = self._single_slot(prefill_len + cap + 2)
        token_buf = slot.buffers["token"]
        emb_pos_buf = slot.buffers["emb_pos"]
        rope_pos_buf = slot.buffers["rope_pos"]
        logits_buf = slot.buffers["logits"]
        runner = slot.runner
        self._seed_static_kv(slot, cache, prefill_len)

        out: list[int] = []
        for step_idx in range(cap):
            if plan.cancelled():
                break
            token = self._draw(logits, out, sampler, floor=floor, stop=stop, seen=seen)
            if token == stop:
                break
            token_buf.fill_(token)
            emb_pos_buf.fill_(prefix_len + step_idx + 1)
            rope_pos_buf.fill_(prefill_len + step_idx)
            runner()
            logits = cast(NDArray[np.float32], logits_buf.cpu().numpy()[0])
        return out

    def _fused_host_slot(self, needed: int) -> _GraphSlot:
        """A captured pair-step whose logits come back to the host to sample."""
        bucket = _bucket(needed)
        key = ("fused_host", bucket)
        slot = self._graphs.get(key)
        if slot is not None:
            return slot

        device, dtype = self._device, self._dtype
        k_bufs, v_bufs, grid = self._static_kv(bucket)
        pair_buf = torch.zeros(1, 2, dtype=torch.long, device=device)
        emb_pos_buf = torch.zeros(1, dtype=torch.long, device=device)
        rope_pos_buf = torch.zeros(1, dtype=torch.long, device=device)
        logits_buf = torch.zeros(1, self.SPEECH_VOCAB, dtype=torch.float32, device=device)
        hidden_buf = torch.zeros(1, self.speech_head.in_features, device=device, dtype=dtype)

        def step() -> None:
            pair = self.speech_emb(pair_buf).to(dtype)
            fused_slot = self._fuse_pair(pair[:, :1], pair[:, 1:])
            fused_slot = fused_slot + self.speech_pos_emb.at_buf(emb_pos_buf)
            out = self.tfmr.forward_static(fused_slot, rope_pos_buf, k_bufs, v_bufs, grid)
            hidden_buf.copy_(out[:, -1])
            logits_buf.copy_(self.speech_head(out[:, -1]).float())

        slot = _GraphSlot(
            max_len=bucket,
            runner=self._capture_runner(step),
            keepalive=step,
            buffers={
                "k": k_bufs,
                "v": v_bufs,
                "pair": pair_buf,
                "emb_pos": emb_pos_buf,
                "rope_pos": rope_pos_buf,
                "logits": logits_buf,
                "hidden": hidden_buf,
                "grid_batch": grid[0],
                "grid_kv": grid[1],
                "grid_head": grid[2],
            },
        )
        self._graphs[key] = slot
        return slot

    def _generate_static_fused(self, plan: _Decode, hidden: Tensor) -> SpeechTokens:
        """:meth:`_generate_static` for ``fusion_mtp2``: one graph per *pair*.

        See ``docs/design/models-notes.md``.
        """
        cap, floor, stop, seen = plan.cap, plan.floor, plan.stop, plan.seen
        sampler, prefill_len, prefix_len = plan.sampler, plan.prefill_len, plan.prefix_len
        logits, cache = plan.logits, plan.cache

        max_pairs = cap // 2 + 1
        slot_ = self._fused_host_slot(prefill_len + max_pairs + 2)
        pair_buf = slot_.buffers["pair"]
        emb_pos_buf = slot_.buffers["emb_pos"]
        rope_pos_buf = slot_.buffers["rope_pos"]
        logits_buf = slot_.buffers["logits"]
        hidden_buf = slot_.buffers["hidden"]
        runner = slot_.runner
        self._seed_static_kv(slot_, cache, prefill_len)

        # The first pair's second head reads the eager prefill's hidden state;
        # every later one reads `hidden_buf`, which the graph refills.
        head_state = hidden[:, -1]

        out_tokens: list[int] = []
        for pair_idx in range(max_pairs):
            # `cap` counts tokens, and a pair can carry the count past a bound
            # checked only per pair, the eager loop's `while len(out) < cap`,
            # spelled out here because this loop is indexed by slot.
            if len(out_tokens) >= cap:
                break
            if plan.cancelled():
                break
            first = self._draw(logits, out_tokens, sampler, floor=floor, stop=stop, seen=seen)
            if first == stop:
                break

            if len(out_tokens) >= cap:
                break
            second_logits = self._pair_second_logits(head_state, first)[0]
            second = self._draw(
                second_logits, out_tokens, sampler, floor=floor, stop=stop, seen=seen
            )
            if second == stop:
                break

            pair_buf[0, 0] = first
            pair_buf[0, 1] = second
            emb_pos_buf.fill_(prefix_len // 2 + pair_idx + 1)
            rope_pos_buf.fill_(prefill_len + pair_idx)
            runner()
            head_state = hidden_buf
            logits = cast(NDArray[np.float32], logits_buf.cpu().numpy()[0])
        return out_tokens

    def _device_sampler(self, sampler: Sampler) -> _DeviceSampler | None:
        """The injected sampler as device buffers, or ``None`` to stay on the host.

        Three things have to hold, and each failure is a fall back rather than
        an error: the decode must actually be running as captured graphs on a
        CUDA device, and the sampler must offer a device form. A caller who
        injected a sampler of their own keeps the loop that calls it.
        """
        if not self.cuda_graphs or self._device.type != "cuda":
            return None
        if not torch.cuda.is_available():
            return None
        if not isinstance(sampler, _OnDevice):
            return None
        return sampler.on_device(self._device)

    def _fused_device_slot(
        self,
        needed: int,
        device_sampler: _DeviceSampler,
        *,
        seen: NDArray[np.bool_],
        floor: int,
        stop: int,
    ) -> tuple[_GraphSlot, _DeviceSampler]:
        """A captured pair-step for a KV buffer at least ``needed`` long.

        See ``docs/design/models-notes.md``.
        """
        cfg = device_sampler.law
        bucket = _bucket(needed)
        key = ("fused_dev", bucket, cfg)
        slot = self._graphs.get(key)
        if slot is not None:
            # The cached slot's own sampler, not the one passed in: the graph
            # holds that sampler's buffer addresses, so the host is rebound
            # onto it and the caller decodes with it.
            cached = slot.extra
            if cached is None:
                raise RuntimeError(
                    f"graph slot {key!r} carries no device sampler; its captured "
                    "step reads buffers nothing would write"
                )
            cached.rebind(device_sampler.host)
            return slot, cached

        device, dtype = self._device, self._dtype
        k_bufs, v_bufs, grid = self._static_kv(bucket)
        pair_buf = torch.zeros(1, 2, dtype=torch.long, device=device)
        emb_pos_buf = torch.zeros(1, dtype=torch.long, device=device)
        rope_pos_buf = torch.zeros(1, dtype=torch.long, device=device)
        token_out = torch.zeros(2, dtype=torch.long, device=device)
        one = torch.ones((), dtype=torch.long, device=device)
        stop_t = torch.full((), stop, dtype=torch.long, device=device)
        # Whether the host has room for this pair's *second* token. Written
        # before each replay, because only the host knows how many tokens are
        # already out; combined in the graph with "the first token was not the
        # stop token", which only the graph knows. See `select(valid=...)`.
        want_second = torch.ones((), dtype=torch.bool, device=device)
        device_sampler.prepare(self.SPEECH_VOCAB, seen, floor=floor, step=0, stop_token=stop)

        def step() -> None:
            pair = self.speech_emb(pair_buf).to(dtype)
            fused_slot = self._fuse_pair(pair[:, :1], pair[:, 1:])
            fused_slot = fused_slot + self.speech_pos_emb.at_buf(emb_pos_buf)
            out = self.tfmr.forward_static(fused_slot, rope_pos_buf, k_bufs, v_bufs, grid)
            state = out[:, -1]

            first = device_sampler.select(self.speech_head(state))
            e_first = self.speech_emb(first.reshape(1, 1))[:, 0].to(dtype)
            # The second draw happens either way, a captured graph has one
            # shape, but the host will discard it if the first token ended the
            # sequence or the cap closed. A discarded draw must leave no state
            # behind in `seen` or the EOS peak, which postprocess compares
            # against thresholds.
            keeps_second = want_second & (first != stop_t)
            second = device_sampler.select(
                self.head2(torch.cat((state, e_first), dim=-1)), valid=keeps_second
            )

            token_out[0].copy_(first)
            token_out[1].copy_(second)
            # The pair this replay produced is the slot the next one consumes,
            # written after the read above and therefore in that order on replay.
            pair_buf[0, 0].copy_(first)
            pair_buf[0, 1].copy_(second)
            emb_pos_buf.add_(one)
            rope_pos_buf.add_(one)

        slot = _GraphSlot(
            max_len=bucket,
            runner=self._capture_runner(step),
            keepalive=step,
            buffers={
                "k": k_bufs,
                "v": v_bufs,
                "pair": pair_buf,
                "emb_pos": emb_pos_buf,
                "rope_pos": rope_pos_buf,
                "token_out": token_out,
                "want_second": want_second,
                "grid_batch": grid[0],
                "grid_kv": grid[1],
                "grid_head": grid[2],
                "one": one,
            },
            extra=device_sampler,
        )
        self._graphs[key] = slot
        return slot, device_sampler

    # One decode loop, and it stays one: splitting it would put the buffers a
    # captured graph holds at fixed addresses in one function and the capture
    # in another. The branches are the pair boundary, a stop and a cap asked
    # twice because a pair is two tokens.
    def _generate_static_fused_ondevice(
        self, plan: _Decode, hidden: Tensor, *, device_sampler: _DeviceSampler
    ) -> SpeechTokens:
        """The whole pair in one captured graph: forward, both heads, both draws.

        See ``docs/design/models-notes.md``.
        """
        cap, floor, stop, seen = plan.cap, plan.floor, plan.stop, plan.seen
        sampler, prefill_len, prefix_len = plan.sampler, plan.prefill_len, plan.prefix_len
        logits, cache = plan.logits, plan.cache

        max_pairs = cap // 2 + 1
        slot, device_sampler = self._fused_device_slot(
            prefill_len + max_pairs + 2, device_sampler, seen=seen, floor=floor, stop=stop
        )
        pair_buf = slot.buffers["pair"]
        emb_pos_buf = slot.buffers["emb_pos"]
        rope_pos_buf = slot.buffers["rope_pos"]
        token_out = slot.buffers["token_out"]
        want_second = slot.buffers["want_second"]
        runner = slot.runner
        self._seed_static_kv(slot, cache, prefill_len)

        # The opening pair is drawn on the host, from the prefill's own hidden
        # state, there is no preceding slot for the graph to read.
        out_tokens: list[int] = []
        # Polled before the first draw, exactly as the other two loops do: the
        # cancellation boundary is a contract, and an interrupt that has already
        # fired must not pay for an opening pair.
        if plan.cancelled():
            return out_tokens
        first = self._draw(logits, out_tokens, sampler, floor=floor, stop=stop, seen=seen)
        if first != stop and cap > 1:
            second_logits = self._pair_second_logits(hidden[:, -1], first)[0]
            self._draw(second_logits, out_tokens, sampler, floor=floor, stop=stop, seen=seen)

        if len(out_tokens) < 2 or out_tokens[-1] == stop:
            return out_tokens

        # After capture, not before: the warm-up replays advanced the step, the
        # positions and `seen`, and wrote garbage into column 0 of the KV.
        device_sampler.prepare(
            self.SPEECH_VOCAB, seen, floor=floor, step=len(out_tokens), stop_token=stop
        )
        pair_buf[0, 0] = out_tokens[0]
        pair_buf[0, 1] = out_tokens[1]
        emb_pos_buf.fill_(prefix_len // 2 + 1)
        rope_pos_buf.fill_(prefill_len)

        for _ in range(1, max_pairs):
            if len(out_tokens) >= cap:
                break
            if plan.cancelled():
                break
            device_sampler.advance_to(len(out_tokens), draws=2)
            # Room for the pair's second token, which only the host knows. The
            # graph ANDs this with "the first token was not the stop token" and
            # suppresses the discarded draw's `seen` and EOS-peak updates.
            want_second.fill_(len(out_tokens) + 1 < cap)
            runner()
            drawn = token_out.tolist()  # the one synchronisation per pair

            out_tokens.append(int(drawn[0]))
            if drawn[0] == stop:
                break
            if len(out_tokens) >= cap:
                break
            out_tokens.append(int(drawn[1]))
            if drawn[1] == stop:
                break

        device_sampler.flush_peak()
        return out_tokens

    def _capture_runner(self, step: Callable[[], None]) -> Callable[[], None]:
        """Return a zero-arg callable that runs the decode step.

        See ``docs/design/models-notes.md``.
        """
        if not (self.cuda_graphs or self.compile_model):
            return step
        if not torch.cuda.is_available() or self._device.type != "cuda":
            # Same math, no launch benefit, never silently pretend otherwise.
            return step
        # CUDA graphs need sm_70+ (Volta). On older cards (compute
        # 6.x) capture fails with "operation failed during capture"; the static
        # path still runs eagerly with identical tokens, just no launch win.
        if torch.cuda.get_device_capability(self._device)[0] < 7:
            return step

        # Warm up once so cuDNN/cuBLAS pick kernels with the final shapes, then
        # capture. Without the warm-up the graph bakes in algorithm choices
        # made for unknown shapes.
        for _ in range(3):
            step()
        torch.cuda.synchronize()

        # Called through an `Any`-typed alias rather than directly: whether
        # `CUDAGraph.__init__` carries annotations varies by torch release, so
        # a direct call is a `no-untyped-call` error under `strict` on some
        # versions and clean on others, and a `type: ignore` for it is an
        # `unused-ignore` error on the rest. This form is correct on every one.
        make_graph: Any = torch.cuda.CUDAGraph
        graph = make_graph()
        # thread_local, not the default global mode: capture happens per generate()
        # call, and under the streaming pipeline the render worker is legitimately
        # issuing CUDA work for the previous window while this thread captures.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            step()
        return cast("Callable[[], None]", graph.replay)

    @torch.inference_mode()
    def teacher_forced_logits(
        self,
        text_tokens: NDArray[np.int64],
        voice: VoiceProfile,
        forced: SpeechTokens,
    ) -> NDArray[np.float32]:
        """Per-step logits with the speech stream pinned to ``forced``.

        One causal forward over the whole sequence: position ``k`` of the
        result is the distribution the model held *before* seeing
        ``forced[k]``, which is what makes two backends comparable step by
        step without free-running chaos.
        """
        if self.config.decode == "fusion_mtp2":
            return self._teacher_forced_fused(text_tokens, voice, forced)

        embeds = self._prefill_embeds(text_tokens, voice)
        speech_start = embeds.shape[1] - 1  # index of the speech START slot
        if len(forced) > 0:
            toks = torch.tensor([list(forced)], device=self._device, dtype=torch.long)
            spe = self.speech_emb(toks) + self.speech_pos_emb.range(
                len(forced), self._device, start=1
            )
            embeds = torch.cat((embeds, spe.to(self._dtype)), dim=1)
        positions = torch.arange(embeds.shape[1], device=self._device)
        hidden, _ = self.tfmr(embeds, positions, None, attention=self.attention)
        return self._head_logits(hidden[0, speech_start:])

    def _teacher_forced_fused(
        self,
        text_tokens: NDArray[np.int64],
        voice: VoiceProfile,
        forced: SpeechTokens,
    ) -> NDArray[np.float32]:
        """:meth:`teacher_forced_logits` under ``fusion_mtp2``.

        See ``docs/design/models-notes.md``.
        """
        forced = list(forced)
        embeds = self._prefill_embeds(text_tokens, voice)
        speech_start = embeds.shape[1] - 1
        # Slots exist only for complete pairs; a trailing odd token is
        # predicted from the last complete slot but never occupies one.
        paired = forced[: len(forced) - len(forced) % 2]
        if paired:
            toks = torch.tensor([paired], device=self._device, dtype=torch.long)
            pairs = cast(Tensor, self.speech_emb(toks)).to(self._dtype)
            slots = self._fuse_pair(pairs[:, 0::2], pairs[:, 1::2])
            slots = slots + self.speech_pos_emb.range(len(paired) // 2, self._device, start=1)
            embeds = torch.cat((embeds, slots.to(self._dtype)), dim=1)
        positions = torch.arange(embeds.shape[1], device=self._device)
        hidden, _ = self.tfmr(embeds, positions, None, attention=self.attention)

        # The START slot plus one per pair: every state that predicts a first
        # token, which is exactly the even rows of the result.
        states = hidden[0, speech_start:]
        rows = np.empty((len(forced) + 1, self.SPEECH_VOCAB), dtype=np.float32)
        rows[0::2] = self._head_logits(states)

        first_ids = forced[0::2]
        if first_ids:
            ids = torch.tensor([first_ids], device=self._device)
            e_first = cast(Tensor, self.speech_emb(ids))
            both = torch.cat((states[: len(first_ids)], e_first[0].to(self._dtype)), dim=-1)
            rows[1::2] = cast(NDArray[np.float32], self.head2(both).float().cpu().numpy())
        return rows
