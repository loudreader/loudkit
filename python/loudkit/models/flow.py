"""The mel decoder: speech tokens to mel, by conditional flow matching.

See ``docs/design/models-notes.md``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import NamedTuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Typing note: torch types nn.Module.__call__ as Any, so submodule calls in a
# forward pass propagate Any. Where the callee's forward provably returns a
# Tensor, the return is wrapped in cast(Tensor, ...), an assertion about
# torch's contract, not a guess. See docs/design/typing.md.
from ..config import AlgorithmConfig
from ..contracts import MEL_BINS, TOKEN_MEL_RATIO, Mel, SpeechTokens
from ..voice import VoiceProfile
from .noise import gaussian_field, gaussian_field_torch
from .windowing import FLOW_NOISE_STREAM, frame_windows, pad_token_id, time_grid

__all__ = [
    "TorchMelDecoder",
    "FLOW_NOISE_STREAM",
    "frame_windows",
    "time_grid",
    "pad_token_id",
]

_ENCODER_DIM = 512


class _DeviceWindow(NamedTuple):
    """:class:`~loudkit.models.windowing.FramedWindow` with its two arrays
    already on the decoder's device."""

    row: Tensor
    cond: Tensor
    prompt_frames: int
    n: int


# ----------------------------------------------------------------- encoder


class _EspnetRelPositionalEncoding(nn.Module):
    """Relative positional table, symmetric around the current frame.

    Returns ``pe[center-T+1 : center+T]``, ``2T−1`` vectors covering every
    possible key−query offset, and scales the input by ``sqrt(d)``. No
    parameters; regenerated on demand rather than stored in the checkpoint.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self._pe = torch.zeros(1, 0, d_model)
        # Not a buffer or a parameter: it is a cache, and `_extend` replaces it.
        # See there for why the lock exists.
        self._pe_lock = threading.Lock()

    def _extend(self, length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Grow the cached encoding table if this call needs more of it.

        See ``docs/design/models-notes.md``.
        """
        with self._pe_lock:
            self._extend_locked(length, device, dtype)
            # Returned from inside the lock, not read from `self._pe` after it.
            return self._pe

    def _extend_locked(self, length: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._pe.shape[1] >= 2 * length - 1 and self._pe.device == device:
            self._pe = self._pe.to(dtype=dtype)
            return
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10_000.0) / self.d_model)
        )
        pos = torch.zeros(length, self.d_model)
        neg = torch.zeros(length, self.d_model)
        pos[:, 0::2] = torch.sin(position * div)
        pos[:, 1::2] = torch.cos(position * div)
        neg[:, 0::2] = torch.sin(-position * div)
        neg[:, 1::2] = torch.cos(-position * div)
        pe = torch.cat([torch.flip(pos, [0]).unsqueeze(0), neg[1:].unsqueeze(0)], dim=1)
        self._pe = pe.to(device=device, dtype=dtype)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        t = x.shape[1]
        pe = self._extend(t, x.device, x.dtype)
        center = pe.shape[1] // 2
        return x * self.xscale, pe[:, center - t + 1 : center + t]


class _LinearEmbed(nn.Module):
    """Input projection (``embed`` / ``up_embed``): Linear + LayerNorm, then
    the relative positional table. The Dropout slot in the original Sequential
    is inference-inert and carries no weights, so it is simply absent."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim, eps=1e-5))
        self.pos_enc = _EspnetRelPositionalEncoding(dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return cast("tuple[Tensor, Tensor]", self.pos_enc(self.out(x)))


class _PreLookahead(nn.Module):
    """Three frames of future context before the causal stack (residual)."""

    def __init__(self, dim: int, lookahead: int = 3) -> None:
        super().__init__()
        self.pre_lookahead_len = lookahead
        self.conv1 = nn.Conv1d(dim, dim, lookahead + 1)
        self.conv2 = nn.Conv1d(dim, dim, 3)

    def forward(self, x: Tensor) -> Tensor:
        h = x.transpose(1, 2)
        h = F.leaky_relu(self.conv1(F.pad(h, (0, self.pre_lookahead_len))))
        h = self.conv2(F.pad(h, (2, 0)))
        return cast(Tensor, h.transpose(1, 2) + x)


class _RelPosAttention(nn.Module):
    """Transformer-XL style attention with relative positions (8 heads x 64)."""

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.h = n_heads
        self.d_k = dim // n_heads
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.linear_out = nn.Linear(dim, dim)
        self.linear_pos = nn.Linear(dim, dim, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_k))

    @staticmethod
    def _rel_shift(x: Tensor) -> Tensor:
        """Turn the (query, 2T−1 offsets) score matrix into (query, key)."""
        b, h, t, _ = x.shape
        zero = torch.zeros((b, h, t, 1), device=x.device, dtype=x.dtype)
        padded = torch.cat([zero, x], dim=-1).view(b, h, x.shape[3] + 1, t)
        return padded[:, :, 1:].view_as(x)[..., : x.shape[3] // 2 + 1]

    def forward(self, x: Tensor, pos_emb: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = self.linear_q(x).view(b, t, self.h, self.d_k)
        k = self.linear_k(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(1, -1, self.h, self.d_k).transpose(1, 2)

        q_u = (q + self.pos_bias_u).transpose(1, 2)
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_ac = q_u @ k.transpose(-2, -1)
        matrix_bd = self._rel_shift(q_v @ p.transpose(-2, -1))
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, -1)
        return cast(Tensor, self.linear_out(out))


class _FeedForward(nn.Module):
    """Position-wise feed-forward with Swish, the encoder was built with
    ``activation_type="swish"``, and ReLU here costs a full mel point."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w_1 = nn.Linear(dim, hidden)
        self.w_2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.w_2(F.silu(self.w_1(x))))


class _ConformerLayer(nn.Module):
    """Pre-norm attention + feed-forward (no macaron, no conv module, this
    encoder was built plain and its checkpoint has no such weights)."""

    def __init__(self, dim: int, n_heads: int, ff_hidden: int) -> None:
        super().__init__()
        self.self_attn = _RelPosAttention(dim, n_heads)
        self.feed_forward = _FeedForward(dim, ff_hidden)
        self.norm_mha = nn.LayerNorm(dim, eps=1e-12)
        self.norm_ff = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, x: Tensor, pos_emb: Tensor) -> Tensor:
        x = x + self.self_attn(self.norm_mha(x), pos_emb)
        return x + cast(Tensor, self.feed_forward(self.norm_ff(x)))


class _Upsample(nn.Module):
    """Nearest-neighbour x2 in time, left-pad 4, then a k=5 conv."""

    def __init__(self, dim: int, stride: int = 2) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(dim, dim, stride * 2 + 1)

    def forward(self, x: Tensor) -> Tensor:
        h = F.interpolate(x, scale_factor=float(self.stride), mode="nearest")
        return cast(Tensor, self.conv(F.pad(h, (self.stride * 2, 0))))


class _UpsampleConformerEncoder(nn.Module):
    """Token embeddings (25 Hz) -> encoder states at mel rate (50 Hz)."""

    def __init__(
        self,
        dim: int = _ENCODER_DIM,
        n_heads: int = 8,
        ff_hidden: int = 2048,
        n_blocks: int = 6,
        n_up_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.embed = _LinearEmbed(dim)
        self.pre_lookahead_layer = _PreLookahead(dim)
        self.encoders = nn.ModuleList(
            _ConformerLayer(dim, n_heads, ff_hidden) for _ in range(n_blocks)
        )
        self.up_layer = _Upsample(dim)
        self.up_embed = _LinearEmbed(dim)
        self.up_encoders = nn.ModuleList(
            _ConformerLayer(dim, n_heads, ff_hidden) for _ in range(n_up_blocks)
        )
        self.after_norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x: Tensor) -> Tensor:
        x, pos = self.embed(x)
        x = self.pre_lookahead_layer(x)
        for layer in self.encoders:
            x = layer(x, pos)
        x = self.up_layer(x.transpose(1, 2)).transpose(1, 2)
        x, pos = self.up_embed(x)
        for layer in self.up_encoders:
            x = layer(x, pos)
        return cast(Tensor, self.after_norm(x))


# --------------------------------------------------------------- estimator


class _CausalConv1d(nn.Conv1d):
    """k-1 frames of left padding: the stack the students were trained in."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int) -> None:
        super().__init__(in_ch, out_ch, kernel)
        self._left_pad = kernel - 1

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.pad(x, (self._left_pad, 0)))


class _CausalBlock(nn.Module):
    """CausalConv -> LayerNorm (over channels) -> Mish.

    The container is a Sequential purely to reproduce the original parameter
    indices (0 = conv, 2 = norm; the Transpose modules at 1 and 3 carry no
    weights and are expressed inline in :meth:`forward`).
    """

    def __init__(self, dim: int, dim_out: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _CausalConv1d(dim, dim_out, 3), nn.Identity(), nn.LayerNorm(dim_out)
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.block[0](x)
        h = self.block[2](h.transpose(1, 2)).transpose(1, 2)
        return F.mish(h)


class _CausalResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_dim, dim_out))
        self.block1 = _CausalBlock(dim, dim_out)
        self.block2 = _CausalBlock(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        h = self.block1(x)
        h = h + self.mlp(t).unsqueeze(-1)
        h = self.block2(h)
        return cast(Tensor, h + self.res_conv(x))


class _SelfAttention(nn.Module):
    """Diffusers-style attention: 8 heads x 64 over a 256-wide stream (the
    inner width 512 exceeds the stream width; the out projection folds back)."""

    def __init__(self, dim: int, n_heads: int, head_dim: int, attention: str = "sdpa") -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.attention = attention
        inner = n_heads * head_dim
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim)])

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = self.to_q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        # Same portability branch as the generator's attention: SDPA lowers to
        # flash-attention (Ampere+ only), so pre-Ampere GPUs need the eager
        # matmul form. Non-causal here, so no mask.
        if self.attention == "eager":
            scale = float(self.head_dim) ** -0.5
            attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
            out = attn @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        return cast(Tensor, self.to_out[0](out.transpose(1, 2).contiguous().view(b, t, -1)))


class _GeluProj(nn.Module):
    """Diffusers' GELU module: a linear projection followed by exact gelu."""

    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.proj(x))


class _FFShell(nn.Module):
    """Namespace so the feed-forward weights live at ``ff.net.*`` exactly as
    the checkpoint stores them; the annotation gives the container a type."""

    net: nn.ModuleList


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, attention: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = _SelfAttention(dim, n_heads, head_dim, attention)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = _FFShell()
        self.ff.net = nn.ModuleList(
            [_GeluProj(dim, dim * 4), nn.Identity(), nn.Linear(dim * 4, dim)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn1(self.norm1(x)) + x
        h = self.ff.net[2](self.ff.net[0](self.norm3(x)))
        return cast(Tensor, h + x)


class _SinusoidalTime(nn.Module):
    """Timestep t -> 320-d sinusoidal embedding (scale 1000). No parameters."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor, scale: float = 1000.0) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(math.log(10_000.0) / (half - 1))
        ).to(t.dtype)
        arg = scale * t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat((arg.sin(), arg.cos()), dim=-1)


class _TimeMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def forward(self, t: Tensor) -> Tensor:
        return cast(Tensor, self.linear_2(F.silu(self.linear_1(t))))


def _transformer_stack(
    dim: int, n_blocks: int, n_heads: int, head_dim: int, attention: str = "sdpa"
) -> nn.ModuleList:
    return nn.ModuleList(
        _TransformerBlock(dim, n_heads, head_dim, attention) for _ in range(n_blocks)
    )


class _Estimator(nn.Module):
    """The velocity field v(x, t | mu, spks, cond): channels 320 -> 80.

    See ``docs/design/models-notes.md``.
    """

    def __init__(
        self,
        in_ch: int = 320,
        out_ch: int = MEL_BINS,
        dim: int = 256,
        n_blocks: int = 4,
        n_mid: int = 12,
        n_heads: int = 8,
        head_dim: int = 64,
        attention: str = "sdpa",
    ) -> None:
        super().__init__()
        time_dim = dim * 4
        self.time_embeddings = _SinusoidalTime(in_ch)
        self.time_mlp = _TimeMLP(in_ch, time_dim)
        self.down_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _CausalResnetBlock(in_ch, dim, time_dim),
                        _transformer_stack(dim, n_blocks, n_heads, head_dim, attention),
                        _CausalConv1d(dim, dim, 3),
                    ]
                )
            ]
        )
        self.mid_blocks = nn.ModuleList(
            nn.ModuleList(
                [
                    _CausalResnetBlock(dim, dim, time_dim),
                    _transformer_stack(dim, n_blocks, n_heads, head_dim, attention),
                ]
            )
            for _ in range(n_mid)
        )
        self.up_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _CausalResnetBlock(dim * 2, dim, time_dim),
                        _transformer_stack(dim, n_blocks, n_heads, head_dim, attention),
                        _CausalConv1d(dim, dim, 3),
                    ]
                )
            ]
        )
        self.final_block = _CausalBlock(dim, dim)
        self.final_proj = nn.Conv1d(dim, out_ch, 1)

    @staticmethod
    def _run_transformers(x: Tensor, blocks: nn.ModuleList) -> Tensor:
        h = x.transpose(1, 2).contiguous()
        for block in blocks:
            h = block(h)
        return h.transpose(1, 2).contiguous()

    def forward(self, x: Tensor, mu: Tensor, t: Tensor, spks: Tensor, cond: Tensor) -> Tensor:
        t_emb = self.time_mlp(self.time_embeddings(t))
        spks_t = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        h = torch.cat((x, mu, spks_t, cond), dim=1)

        # Indexing a ModuleList is typed as plain Module, so each nested stage
        # is cast back to the container type __init__ actually built there.
        down_resnet, down_tf, down_conv = cast(nn.ModuleList, self.down_blocks[0])
        h = down_resnet(h, t_emb)
        h = self._run_transformers(h, cast(nn.ModuleList, down_tf))
        skip = h  # taken before the tail conv, as trained
        h = down_conv(h)

        for mid_stage in self.mid_blocks:
            mid_resnet, mid_tf = cast(nn.ModuleList, mid_stage)
            h = mid_resnet(h, t_emb)
            h = self._run_transformers(h, cast(nn.ModuleList, mid_tf))

        up_resnet, up_tf, up_conv = cast(nn.ModuleList, self.up_blocks[0])
        h = torch.cat((h[:, :, : skip.shape[-1]], skip), dim=1)
        h = up_resnet(h, t_emb)
        h = self._run_transformers(h, cast(nn.ModuleList, up_tf))
        h = up_conv(h)

        return cast(Tensor, self.final_proj(self.final_block(h)))


# -------------------------------------------------------------- mel decoder


class _DecoderShell(nn.Module):
    """Nothing but a namespace: the checkpoint stores the estimator under
    ``decoder.estimator`` (the upstream CFM wrapper owned it), and mirroring
    that path keeps the tensor names 1:1."""

    def __init__(self, estimator: _Estimator) -> None:
        super().__init__()
        self.estimator = estimator


class TorchMelDecoder(nn.Module):
    """``MelDecoder`` implementation on torch (cpu / cuda / mps).

    See ``docs/design/models-notes.md``.
    """

    def __init__(
        self,
        config: AlgorithmConfig,
        *,
        estimator_dtype: torch.dtype = torch.float32,
        flow_embedding_dim: int = 192,
        vocab_size: int = 6561,
        attention: str = "sdpa",
    ) -> None:
        super().__init__()
        self.config = config
        self.estimator_dtype = estimator_dtype
        self.input_embedding = nn.Embedding(vocab_size, _ENCODER_DIM)
        self.spk_embed_affine_layer = nn.Linear(flow_embedding_dim, MEL_BINS)
        self.encoder = _UpsampleConformerEncoder()
        self.encoder_proj = nn.Linear(_ENCODER_DIM, MEL_BINS)
        self.decoder = _DecoderShell(_Estimator(attention=attention))

    @property
    def _device(self) -> torch.device:
        return self.encoder_proj.weight.device

    # -- windowing -----------------------------------------------------------

    def _frame(self, tokens: SpeechTokens, voice: VoiceProfile) -> _DeviceWindow:
        """The shared window recipe (:func:`frame_windows`), as device tensors."""
        row, cond, prompt_frames, n = frame_windows(self.config, tokens, voice)
        return _DeviceWindow(
            torch.from_numpy(row).to(self._device),
            torch.from_numpy(cond).to(self._device),
            prompt_frames,
            n,
        )

    # -- integration ---------------------------------------------------------

    def _velocity(self, x: Tensor, mu: Tensor, t: float, spks: Tensor, cond: Tensor) -> Tensor:
        est = self.decoder.estimator
        t_row = torch.tensor([t], device=x.device, dtype=x.dtype)
        if self.config.guidance == "single_path":
            return cast(Tensor, est(x, mu, t_row, spks, cond))
        # cfg_dual_path: conditional and unconditional velocities, combined
        # (1+w)·v_cond − w·v_uncond. Teacher mode only: running it on a
        # guidance-distilled student applies guidance twice, a 0.979
        # mel-correlation defect no output check caught. See
        # docs/design/models-notes.md.
        rate = self.config.guidance_rate
        x2 = torch.cat([x, x], dim=0)
        # One t per row of x2, so a batch of utterances lines up with its
        # doubled prior. At one utterance this is the pair it always was.
        t2 = t_row.repeat(2 * x.shape[0])
        mu2 = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks2 = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond2 = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        v = est(x2, mu2, t2, spks2, cond2)
        v_cond, v_uncond = v.chunk(2, dim=0)
        return cast(Tensor, (1.0 + rate) * v_cond - rate * v_uncond)

    def _prior(self, seed: int, t_mel: int) -> Tensor:
        """The CFM prior for one utterance, ``(80, t_mel)`` on the device.

        Same stream either way, see `gaussian_field_torch`. Far smaller than
        the vocoder's excitation (80 x frames against 9 x samples), so the
        device draw is for consistency rather than for the milliseconds.
        """
        if self._device.type == "cuda":
            return gaussian_field_torch(seed, FLOW_NOISE_STREAM, MEL_BINS, t_mel, self._device)
        z = gaussian_field(seed, FLOW_NOISE_STREAM, MEL_BINS, t_mel)
        return torch.from_numpy(z).to(self._device)

    def _integrate(self, row: Tensor, cond: Tensor, spks: Tensor, x: Tensor) -> Tensor:
        """Encode the row and walk the Euler grid from ``x`` to the mel.

        The whole of the flow that does not depend on how many utterances are
        in flight, so :meth:`decode` and :meth:`decode_batch` share it rather
        than each keeping a copy of the encoder call, the dtype cast and the
        loop. What differs between them is the prior, which the caller builds
        and passes in, and the cutting afterwards.
        """
        h = self.encoder(self.input_embedding(row))
        mu = self.encoder_proj(h).transpose(1, 2).contiguous()

        dt_ = self.estimator_dtype
        x, mu, spks, cond = (a.to(dt_) for a in (x, mu, spks, cond))
        grid = time_grid(self.config)
        for t0, t1 in zip(grid[:-1], grid[1:], strict=True):
            x = x + (t1 - t0) * self._velocity(x, mu, t0, spks, cond)
        return x.float()

    # -- contract ------------------------------------------------------------

    @torch.inference_mode()
    def decode(self, tokens: SpeechTokens, voice: VoiceProfile, *, seed: int) -> Mel:
        """Integrate the flow to a mel for ``tokens`` in ``voice``.

        The returned mel covers only the real speech region, the prompt
        reconstruction (a coarse, audibly "underwater" render of the reference)
        and any static padding are cut here, not left for the vocoder to
        stumble over.
        """
        row, cond, prompt_frames, n = self._frame(tokens, voice)
        t_mel = TOKEN_MEL_RATIO * row.shape[1]

        emb = torch.from_numpy(np.asarray(voice.flow_embedding, dtype=np.float32))
        emb = F.normalize(emb[None], dim=1).to(self._device)
        spks = self.spk_embed_affine_layer(emb)

        x = self._integrate(row, cond, spks, self._prior(seed, t_mel)[None])
        mel = x[0, :, prompt_frames : prompt_frames + TOKEN_MEL_RATIO * n]
        return mel.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def decode_batch(
        self,
        tokens: Sequence[SpeechTokens],
        voices: Sequence[VoiceProfile],
        *,
        seeds: Sequence[int],
    ) -> list[Mel]:
        """:meth:`decode` for several utterances at once, one row each.

        A prototype beside the contract, not on it. Nothing in the engine calls
        it: measured, batching this stage buys at most 9% on the GPU for eight
        times the latency, and 15% to 22% on CPU, where the renderer is
        latency-bound anyway. See ``docs/design/models-notes.md``, and
        ``research/bench_render.py`` to re-measure on another device.

        Every utterance must frame to the same window length. The conformer
        attends over the whole row with no mask, so a row padded out to a
        longer neighbour would change that neighbour's mel.
        """
        rows = len(tokens)
        if rows == 0 or len(voices) != rows or len(seeds) != rows:
            raise ValueError(
                f"decode_batch needs one voice and one seed per utterance: "
                f"{rows} token sequences, {len(voices)} voices, {len(seeds)} seeds"
            )
        framed = [self._frame(t, v) for t, v in zip(tokens, voices, strict=True)]
        widths = sorted({f.row.shape[1] for f in framed})
        if len(widths) != 1:
            raise ValueError(
                f"decode_batch needs one window length for the whole batch, got "
                f"{widths}; group the utterances by length, or use a static window"
            )
        row = torch.cat([f.row for f in framed])
        cond = torch.cat([f.cond for f in framed])
        t_mel = TOKEN_MEL_RATIO * row.shape[1]

        emb = torch.from_numpy(
            np.stack([np.asarray(v.flow_embedding, dtype=np.float32) for v in voices])
        )
        emb = F.normalize(emb, dim=1).to(self._device)
        spks = self.spk_embed_affine_layer(emb)

        prior = torch.stack([self._prior(seed, t_mel) for seed in seeds])
        out = self._integrate(row, cond, spks, prior)
        return [
            out[i, :, f.prompt_frames : f.prompt_frames + TOKEN_MEL_RATIO * f.n]
            .cpu()
            .numpy()
            .astype(np.float32)
            for i, f in enumerate(framed)
        ]
