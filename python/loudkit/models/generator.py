"""The token generator: a Llama-architecture decoder that writes speech.

This is the T3 student — 16 decoder layers, hidden 1024, 16 query heads of 64
with 4 KV heads (GQA), SwiGLU MLPs, RoPE with the llama3 long-context scaling,
and a speech head over an 8194-token vocabulary. Implemented directly in torch
rather than through ``transformers`` so that the arithmetic is visible, the
attention implementation is selectable (the fused path aborts the interpreter
on MPS — no traceback, just ``LLVM ERROR`` from ``mps_matmul``), and the
dependency surface stays small. Module attribute names mirror the packed
checkpoint (``t3.*``), so weights load with ``strict=True``.

Sequence layout, identical to the shipped engine's runner::

    [ cond (34) | [START] text [STOP] | speech: START, s0, s1, ... ]

where cond = speaker projection (1) + perceiver-resampled speech prompt (32) +
emotion (1). Text and speech carry *learned* positional embeddings on top of
their token embeddings, each restarting at zero — RoPE handles relative order
inside the transformer, the learned tables tell it which segment it is in.

Two behaviours are deliberate and worth naming:

* **The EOS floor is applied here**, before the sampler sees the logits: the
  stop token is masked to −inf until the configured minimum length (production:
  ``max(10, 1.2 x text tokens)``). Sampler exemptions cannot express "forbid
  one token for a while", and the floor is an EOS *policy*, which the algorithm
  layer owns.
* ``teacher_forced_logits`` runs one causal forward over the whole forced
  sequence rather than replaying the decode loop. Same mathematics, different
  reduction shapes — an *equivalent*-class difference, which is exactly the
  tolerance the comparisons that use it are designed to absorb.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn

from ..config import AlgorithmConfig
from ..contracts import Sampler, SpeechTokens
from ..voice import EMOTION_NEUTRAL, VoiceProfile
from .windowing import START_TEXT_TOKEN, STOP_TEXT_TOKEN, eos_floor

# Typing note, which holds for every model module in this package: torch types
# ``nn.Module.__call__`` as returning ``Any`` (a module can return anything),
# so every submodule call site would otherwise propagate Any. Where a module's
# own ``forward`` provably returns a Tensor, the call is wrapped in
# ``cast(Tensor, ...)`` — an assertion about torch's contract, not a guess.
# See docs/reference/typing.md.

__all__ = ["TorchTokenGenerator", "eos_floor"]

_ATTN = {"eager", "sdpa"}


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

    The fp32 round-trip is not an optimisation choice — it is what the weights
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
    smoothly — verbatim the published llama3 rule, kept in fp64 until the end
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
    ) -> Tensor:
        """One decode step against a fixed-shape KV buffer (graph-capturable).

        ``k_buf``/``v_buf`` are ``[1, n_kv_heads, max_len, head_dim]`` and the
        new key/value is written into column ``pos`` in place via
        ``index_put_`` with a **preallocated** index grid ``grid`` — so no
        tensor is created on the critical path, every address is fixed, and
        the whole step can be captured by a CUDA graph (the ``cuda_graphs``
        execution flag). ``grid`` is ``(batch_idx, kv_idx, head_idx)``, all
        ``[n_kv_heads * head_dim]``, built once per generator; ``pos`` is a
        0-dim device buffer updated between replays.

        The query attends over the whole padded buffer with a causal mask, so
        positions beyond ``pos`` contribute exactly zero. This changes the
        reduction order relative to the dynamic ``torch.cat`` path — an
        *equivalent*-class difference (deterministic, logit drift ~1e-6, same
        tokens in practice), which is the identity contract's own
        classification for a static KV cache. It is opt-in via
        ``ExecutionConfig.cuda_graphs``; the default path stays bit-identical.
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
        pad = torch.arange(kk.shape[2], device=x.device) > pos
        mask = torch.where(
            pad,
            torch.full((), float("-inf"), device=x.device, dtype=scores.dtype),
            torch.zeros((), device=x.device, dtype=scores.dtype),
        )
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
    ) -> Tensor:
        attn_out = self.self_attn.forward_static(
            self.input_layernorm(x), cos, sin, k_buf, v_buf, grid, pos
        )
        x = x + attn_out
        return cast(Tensor, x + self.mlp(self.post_attention_layernorm(x)))


class LlamaDecoder(nn.Module):
    """The bare decoder stack (checkpoint namespace ``t3.tfmr``).

    Consumes pre-built input embeddings — token/positional embedding is the
    conditioning layer's business — and returns final hidden states. The KV
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
        assert attention in _ATTN, attention
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
        captured by a CUDA graph — the ``cuda_graphs`` execution flag.
        """
        cos, sin = self._rope(positions, inputs_embeds.dtype)
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x = cast(_DecoderLayer, layer).forward_static(
                x, cos, sin, k_bufs[i], v_bufs[i], grid, positions
            )
        return cast(Tensor, self.norm(x))


# ------------------------------------------------------------- conditioning


class _PerceiverAttention(nn.Module):
    """One shared attention block used for both perceiver passes.

    Both inputs go through the *same* LayerNorm and the same q/k/v projections
    — that is how the original was built (one block, called twice), and the
    weights only exist once in the checkpoint.
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
        # eager matmul path below is the same mathematics, written portably —
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
        # zeros, not `torch.empty`: this is the one parameter in the file with
        # no initialiser, so `torch.empty` handed it whatever was in the
        # recycled allocation. The checkpoint overwrites it, which is why
        # nothing noticed — but constructing the model was not deterministic,
        # so `torch.manual_seed(0)` did not actually pin a build, and a heap
        # that had recently held NaNs produced a model whose logits were NaN.
        # That is exactly what it did: TestStaticCacheDecode failed roughly one
        # run in three with `drift = nan`, and passed when run alone on a clean
        # heap — a flake that reads as numerical trouble in the static-cache
        # path rather than as an uninitialised read.
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

    Published because the batch benchmark needed it and was reading it out of
    ``gen.tfmr.layers[0].self_attn.n_kv_heads`` — a tool reaching three levels
    into a module's internals, which then breaks the first time the decoder is
    refactored and says nothing about why. Benchmarking a new target is a real
    use, so the numbers a benchmark needs are part of the surface.

    Attributes:
        n_layers: decoder layers, and so the first axis of a KV cache.
        n_kv_heads: key/value heads per layer — fewer than query heads under
            grouped-query attention, which is why this cannot be derived from
            the head count.
        head_dim: width of one head.
        device: where the weights live.
        dtype: what they are stored as.
    """

    n_layers: int
    n_kv_heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype

    def cache_shape(self, batch: int, max_length: int) -> tuple[int, int, int, int, int]:
        """The shape of one of the two (key, value) buffers."""
        return (self.n_layers, batch, self.n_kv_heads, max_length, self.head_dim)


def check_manifest_sizes(config: object, *, speech_vocab: int, start_token: int) -> None:
    """Refuse a manifest whose speech vocabulary disagrees with the weights.

    The ONNX backend reads ``speech_vocab_size`` and ``start_speech_token`` from
    the manifest and shapes its inputs accordingly; the torch path has them as
    constants, because they are the dimensions of *these* tables. A manifest
    carrying different values therefore loaded on both backends under one
    fingerprint and behaved differently on each — the divergence class this
    library exists to prevent, arriving through data rather than code.

    Refused rather than reconciled: whichever side is wrong, a checkpoint with
    other dimensions needs other weights, and guessing which to believe is how
    one of them starts speaking wrongly with nothing flagging it.
    """
    declared_vocab = getattr(config, "speech_vocab_size", speech_vocab)
    declared_start = getattr(config, "start_speech_token", start_token)
    if declared_vocab != speech_vocab or declared_start != start_token:
        raise ValueError(
            f"manifest declares speech_vocab_size={declared_vocab}, "
            f"start_speech_token={declared_start}; these weights carry "
            f"{speech_vocab} and {start_token}. A different vocabulary is a "
            "different checkpoint — re-export rather than re-declare."
        )


class TorchTokenGenerator(nn.Module):
    """``TokenGenerator`` implementation on torch (cpu / cuda / mps).

    Args:
        config: the algorithm. Never copied, never defaulted — the engine
            checks its fingerprint against every other component's.
        llama_config: architecture dict from the checkpoint manifest.
        attention: ``"eager"`` or ``"sdpa"``, from
            ``ExecutionConfig.resolved_attention()``. On MPS this must be
            eager; the fused kernel kills the interpreter with no traceback.
    """

    # The shipped weights' dimensions, and a manifest that disagrees is refused
    # rather than silently overridden — see `check_manifest_sizes`.
    #
    # The ONNX backend reads `speech_vocab_size` and `start_speech_token` from
    # the manifest; this path did not, so a manifest carrying different values
    # loaded on both backends under one fingerprint and they behaved
    # differently. These are the tables in *these* weights: a checkpoint with
    # other dimensions needs other weights, not another constant — so a manifest
    # that disagrees is now refused at load rather than half-honoured.
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
        voice recomputes it — a speaker projection plus two perceiver passes.
        Execution-layer memoisation: the cached tensor is what the computation
        would have produced, bit for bit, on this device and dtype. Capped
        small; an engine rarely sees more than a couple of voices at once.
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

    # -- assembly ------------------------------------------------------------

    @property
    def _device(self) -> torch.device:
        return self.speech_head.weight.device

    @property
    def _dtype(self) -> torch.dtype:
        return self.speech_head.weight.dtype

    def _speech_token_embed(self, token: int, position: int) -> Tensor:
        t = torch.tensor([[token]], device=self._device)
        return cast(Tensor, self.speech_emb(t)) + self.speech_pos_emb.at(position)

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
            cond = self.cond_enc(speaker, prompt_emb, emotion)
            if len(self._cond_cache) >= 8:
                self._cond_cache.pop(next(iter(self._cond_cache)))
            self._cond_cache[key] = cond

        framed = np.concatenate(([START_TEXT_TOKEN], text_tokens, [STOP_TEXT_TOKEN]))
        tt = torch.from_numpy(framed.astype(np.int64)).to(device)[None]
        if tt.shape[1] > self.MAX_TEXT_POSITIONS:
            # Checked before the embedding rather than discovered inside it. The
            # table has `MAX_TEXT_POSITIONS` rows, so a longer prompt came back
            # as a bare `IndexError` from a positional-embedding lookup — a
            # message naming neither the text, the limit, nor the method that
            # exists precisely for this case.
            raise ValueError(
                f"{tt.shape[1]} text tokens exceeds the "
                f"{self.MAX_TEXT_POSITIONS}-position table; use "
                "`Engine.synthesize_long`, which splits at sentence boundaries"
            )
        text = self.text_emb(tt) + self.text_pos_emb.range(tt.shape[1], device)

        bos = self._speech_token_embed(self.config.start_speech_token, 0)
        return torch.cat((cond, text, bos), dim=1).to(dtype)

    def _head_logits(self, hidden: Tensor) -> NDArray[np.float32]:
        """Speech logits in fp32 on the host — the sampler's world is numpy."""
        # Tensor.numpy() is untyped in torch's stubs; .float() guarantees fp32.
        return cast(NDArray[np.float32], self.speech_head(hidden).float().cpu().numpy())

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

        The sampler owns the law; this loop owns only the EOS floor and the
        ``seen`` bookkeeping. The stop token, when it fires naturally, is
        *included* in the returned sequence so the engine can distinguish a
        natural ending from a cap hit.

        ``prefix`` holds speech tokens from the preceding chunk. They are fed
        through the model to build context and then dropped from the result:
        the caller asked for this chunk, not the previous one. They also seed
        the repetition-penalty state, since a token repeated across a join is
        just as repeated as one repeated within a chunk.

        With ``cuda_graphs`` or ``compile_model`` set, the per-token decode
        step runs over a **static KV cache**: preallocated buffers written in
        place, so the step has fixed addresses and can be captured as a CUDA
        graph (or compiled). That changes the attention reduction order — an
        *equivalent*-class difference (deterministic, logit drift ~1e-6, same
        tokens in practice), exactly what the identity contract sanctions for a
        static cache. The default path below is unchanged and bit-identical.
        """
        cap = max_new_tokens or self.config.sampling.max_new_tokens
        floor = eos_floor(len(text_tokens), self.config)
        stop = self.config.stop_speech_token

        embeds = self._prefill_embeds(text_tokens, voice)
        seen = np.zeros(self.SPEECH_VOCAB, dtype=bool)

        prefix = [int(t) for t in prefix]
        if prefix:
            toks = torch.tensor([prefix], device=self._device, dtype=torch.long)
            spe = self.speech_emb(toks) + self.speech_pos_emb.range(
                len(prefix), self._device, start=1
            )
            embeds = torch.cat((embeds, spe.to(self._dtype)), dim=1)
            seen[prefix] = True

        prefill_len = embeds.shape[1]
        positions = torch.arange(prefill_len, device=self._device)
        hidden, cache = self.tfmr(embeds, positions, None, attention=self.attention)
        logits = self._head_logits(hidden[:, -1])[0]

        if self.cuda_graphs or self.compile_model:
            return self._generate_static(
                cap,
                floor,
                stop,
                seen,
                prefill_len,
                cache,
                prefix_len=len(prefix),
                sampler=sampler,
                logits=logits,
                should_cancel=should_cancel,
            )

        out: list[int] = []
        for step in range(cap):
            # Token-level cancellation for barge-in: polled on every decode
            # step, so an interrupt is honoured within one forward pass rather
            # than only at a chunk boundary (~10 s of speech). The token that
            # was about to be sampled is discarded.
            if should_cancel is not None and should_cancel():
                break
            if len(out) < floor:
                logits[stop] = -np.inf
            token = sampler(logits, step=step, seen=seen)
            out.append(token)
            if token == stop:
                break
            seen[token] = True
            emb = self._speech_token_embed(token, len(prefix) + step + 1).to(self._dtype)
            pos = torch.tensor([prefill_len + step], device=self._device)
            hidden, cache = self.tfmr(emb, pos, cache, attention=self.attention)
            logits = self._head_logits(hidden[:, -1])[0]
        return out

    def _generate_static(
        self,
        cap: int,
        floor: int,
        stop: int,
        seen: NDArray[np.bool_],
        prefill_len: int,
        cache: list[tuple[Tensor, Tensor]],
        *,
        prefix_len: int,
        sampler: Sampler,
        logits: NDArray[np.float32],
        should_cancel: Callable[[], bool] | None = None,
    ) -> SpeechTokens:
        """The static-cache decode loop (``cuda_graphs``/``compile_model``).

        KV buffers are sized to this utterance's ``prefill_len + cap``, seeded
        from the eager prefill's ``cache``, then each step writes one token's
        key/value in place and reads logits back. The step is either a captured
        CUDA graph, a compiled function, or — on machines without CUDA — the
        same function run eagerly, so the math is identical in all three.
        """
        max_len = prefill_len + cap + 2
        n_layers = self.tfmr.n_layers
        first = cast(_DecoderLayer, self.tfmr.layers[0])
        n_kv = first.self_attn.n_kv_heads
        head_dim = self.tfmr.head_dim
        device, dtype = self._device, self._dtype

        k_bufs = torch.zeros(n_layers, 1, n_kv, max_len, head_dim, device=device, dtype=dtype)
        v_bufs = torch.zeros_like(k_bufs)

        # Preallocated index grid for the in-place KV write: one fixed
        # (batch, kv, head) triple, shared by every layer and every step, so
        # the decode step creates no tensors and can be captured by a graph.
        grid = (
            torch.zeros(n_kv * head_dim, dtype=torch.long, device=device),
            torch.arange(n_kv, device=device).repeat_interleave(head_dim).contiguous(),
            torch.arange(head_dim, device=device).repeat(n_kv).contiguous(),
        )

        token_buf = torch.zeros(1, 1, dtype=torch.long, device=device)
        emb_pos_buf = torch.zeros(1, dtype=torch.long, device=device)
        rope_pos_buf = torch.zeros(1, dtype=torch.long, device=device)
        logits_buf = torch.zeros(1, self.SPEECH_VOCAB, dtype=torch.float32, device=device)

        def step() -> None:
            emb = self.speech_emb(token_buf) + self.speech_pos_emb.at_buf(emb_pos_buf)
            emb = emb.to(self._dtype)
            hidden = self.tfmr.forward_static(emb, rope_pos_buf, k_bufs, v_bufs, grid)
            logits_buf.copy_(self.speech_head(hidden[:, -1]).float())

        # Capture (warm-up + graph) runs `step` a few times with pos=0; that
        # writes garbage into column 0 of the buffers, so the real prefill KV
        # must be seeded *after* capture. At replay the write column comes from
        # the rope_pos_buf device value, not the value frozen at capture time.
        runner = self._capture_runner(step)
        for i, (k, v) in enumerate(cache):
            k_bufs[i, 0, :, :prefill_len, :].copy_(k[0])
            v_bufs[i, 0, :, :prefill_len, :].copy_(v[0])

        out: list[int] = []
        for step_idx in range(cap):
            # Token-level cancellation (barge-in), same as the eager path.
            if should_cancel is not None and should_cancel():
                break
            if len(out) < floor:
                logits[stop] = -np.inf
            token = sampler(logits, step=step_idx, seen=seen)
            out.append(token)
            if token == stop:
                break
            seen[token] = True
            token_buf.fill_(token)
            emb_pos_buf.fill_(prefix_len + step_idx + 1)
            rope_pos_buf.fill_(prefill_len + step_idx)
            runner()
            logits = cast(NDArray[np.float32], logits_buf.cpu().numpy()[0])
        return out

    def _capture_runner(self, step: Callable[[], None]) -> Callable[[], None]:
        """Return a zero-arg callable that runs the decode step.

        ``cuda_graphs`` and ``compile_model`` both want the same thing — the
        per-token decode as one captured graph instead of ~1442 kernel launches
        — and both run over the same static KV cache, so they share the manual
        ``torch.cuda.CUDAGraph`` capture here (``torch.compile``'s own capture
        hits an inductor mask-alignment bug on this model, and its intent is
        identical). With neither flag, ``step`` itself is the runner. A graph
        needs the buffers to stay at fixed addresses and the input values to be
        pushed through ``.fill_`` before each replay, which ``_generate_static``
        does.
        """
        if not (self.cuda_graphs or self.compile_model):
            return step
        if not torch.cuda.is_available() or self._device.type != "cuda":
            # Same math, no launch benefit — never silently pretend otherwise.
            return step
        # CUDA graphs need sm_70+ (Volta). On Pascal (GTX 1080 Ti, compute
        # 6.1) capture fails with "operation failed during capture"; the static
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
        # versions and clean on others — and a `type: ignore` for it is an
        # `unused-ignore` error on the rest. This form is correct on every one.
        make_graph: Any = torch.cuda.CUDAGraph
        graph = make_graph()  # noqa: F841 - held by `torch.cuda.graph` below
        # thread_local, not the default global mode: capture happens per
        # generate() call, and under the streaming pipeline the render worker
        # is legitimately issuing CUDA work for the *previous* window while
        # this thread captures. Global mode forbids any concurrent CUDA call
        # process-wide ("operation not permitted when stream is capturing");
        # thread_local scopes the restriction to the capturing thread, and
        # only this thread's stream is recorded into the graph either way.
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


class _LearnedPositions(nn.Module):
    """Learned absolute positions (GPT-2 style), one table per segment kind."""

    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(max_len, dim)

    def range(self, length: int, device: torch.device, *, start: int = 0) -> Tensor:
        return cast(Tensor, self.emb(torch.arange(start, start + length, device=device)))

    def at(self, position: int) -> Tensor:
        return cast(Tensor, self.emb(torch.tensor([[position]], device=self.emb.weight.device)))

    def at_buf(self, pos: Tensor) -> Tensor:
        """Same lookup from a buffer — the graph-capturable variant of :meth:`at`."""
        return cast(Tensor, self.emb(pos.view(1, 1)))
