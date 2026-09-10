"""Enrollment: reference audio to a :class:`~loudkit.voice.VoiceProfile`.

See ``docs/design/models-notes.md``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn

# Typing note: torch types nn.Module.__call__ as Any, so submodule calls in a
# forward pass propagate Any. Where the callee's forward provably returns a
# Tensor, the return is wrapped in cast(Tensor, ...), an assertion about
# torch's contract, not a guess. See docs/design/typing.md.
from ..checkpoint import Checkpoint
from ..contracts import TOKEN_MEL_RATIO
from ..voice import VoiceProfile
from .enrollment_audio import validate_reference_audio
from .resample import resample
from .windowing import UPSAMPLE_PER_FRAME

__all__ = ["TorchVoiceEnroller", "validate_reference_audio"]

_S3_SR = 16_000
_MEL_SR = 24_000
_COND_PROMPT_TOKENS = 150  # the token generator's speech_cond_prompt_len
_MAX_REF_SECONDS = 10.0

_MEL_WINDOW = 1920
"""STFT window and transform size for the conditioning mel: 80 ms at 24 kHz."""

_FSQ_TANH_SCALE = 0.9990000128746033
"""float32(0.999), the trained quantiser's constant.

Written to all its digits rather than as ``0.999`` because the graph was traced
with the float32 value and a float64 0.999 rounds elsewhere."""


# ------------------------------------------------------------- mel extractor


_mel_basis_cache: dict[tuple[int, int, int, int, int], Tensor] = {}


def _slaney_mels(sr: int, n_fft: int, n_mels: int, fmin: int, fmax: int) -> Tensor:
    key = (sr, n_fft, n_mels, fmin, fmax)
    if key not in _mel_basis_cache:
        # The submodule, not the package. `import librosa` does not bind
        # `librosa.filters` by contract, it works only because something else
        # imported it first, and librosa 1.0 says so in its own type
        # information.
        import librosa.filters

        # `type: ignore` because librosa's stubs are skipped wholesale (see the
        # librosa override in pyproject.toml): mypy therefore knows the package
        # but not that it re-exports its submodules. The import above is what
        # makes the attribute exist at runtime.
        basis = librosa.filters.mel(  # type: ignore[attr-defined]
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        _mel_basis_cache[key] = torch.from_numpy(basis).float()
    return _mel_basis_cache[key]


def _matcha_mel(wav: Tensor) -> Tensor:
    """(80, frames) log-mel at 24 kHz, the flow's conditioning features."""
    n_fft, hop, win = _MEL_WINDOW, UPSAMPLE_PER_FRAME, _MEL_WINDOW
    pad = (n_fft - hop) // 2
    y = F.pad(wav[None, None], (pad, pad), mode="reflect")[0]
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop,
        win_length=win,
        window=torch.hann_window(win),
        center=False,
        return_complex=True,
    )
    mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    mel = _slaney_mels(_MEL_SR, n_fft, 80, 0, 8000) @ mag
    return torch.log(torch.clamp(mel, min=1e-5))[0]


# ---------------------------------------------------------- speech tokenizer


def _rotary_tables(head_dim: int, max_len: int) -> tuple[Tensor, Tensor]:
    inv = 1.0 / (10_000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len).float(), inv)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def _rotate(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _FSMNAttention(nn.Module):
    """Whisper-style attention plus a depthwise FSMN memory over the values."""

    def __init__(self, n_state: int, n_head: int, kernel: int = 31) -> None:
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
        self.fsmn_block = nn.Conv1d(n_state, n_state, kernel, groups=n_state, bias=False)
        self._pad = ((kernel - 1) // 2, kernel - 1 - (kernel - 1) // 2)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, d = x.shape
        q = self.query(x).view(b, t, self.n_head, -1)
        k = self.key(x).view(b, t, self.n_head, -1)
        v = self.value(x).view(b, t, self.n_head, -1)

        q = q * cos[:t, None] + _rotate(q) * sin[:t, None]
        k = k * cos[:t, None] + _rotate(k) * sin[:t, None]

        memory = v.reshape(b, t, d).transpose(1, 2)
        memory = self.fsmn_block(F.pad(memory, self._pad)).transpose(1, 2) + v.reshape(b, t, d)

        scale = (d // self.n_head) ** -0.25
        qh = q.permute(0, 2, 1, 3) * scale
        kh = k.permute(0, 2, 3, 1) * scale
        vh = v.permute(0, 2, 1, 3)
        attn = torch.softmax((qh @ kh).float(), dim=-1).to(qh.dtype)
        mixed = (attn @ vh).permute(0, 2, 1, 3).flatten(2)
        return cast(Tensor, self.out(mixed) + memory)


class _TokenizerBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int) -> None:
        super().__init__()
        self.attn = _FSMNAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_state * 4), nn.GELU(), nn.Linear(n_state * 4, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.attn_ln(x), cos, sin)
        return x + cast(Tensor, self.mlp(self.mlp_ln(x)))


class _TokenizerEncoder(nn.Module):
    def __init__(
        self, n_mels: int = 128, n_state: int = 1280, n_head: int = 20, n_layer: int = 6
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList(_TokenizerBlock(n_state, n_head) for _ in range(n_layer))
        cos, sin = _rotary_tables(64, 2048)
        self.register_buffer("_rope_cos", cos, persistent=False)
        self.register_buffer("_rope_sin", sin, persistent=False)

    def forward(self, mel: Tensor) -> Tensor:
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x, self._rope_cos, self._rope_sin)
        return x


class _FSQCodebook(nn.Module):
    """Finite scalar quantisation: project to 8 dims, tanh, round to {0,1,2},
    read the result as a base-3 number, 3^8 = 6561 codes."""

    def __init__(self, dim: int = 1280, level: int = 3) -> None:
        super().__init__()
        self.project_down = nn.Linear(dim, 8)
        self.level = level

    def encode(self, x: Tensor) -> Tensor:
        h = self.project_down(x).float().tanh() * _FSQ_TANH_SCALE
        h = h.round() + 1
        powers = torch.pow(
            torch.tensor(float(self.level)), torch.arange(8, dtype=h.dtype, device=h.device)
        )
        return cast(Tensor, (h * powers).sum(-1).long())


class _Quantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._codebook = _FSQCodebook()


class _S3Tokenizer(nn.Module):
    """16 kHz audio -> 25 Hz speech tokens (namespace ``s3gen.tokenizer``)."""

    # registered buffers; annotated so access is not Tensor | Module
    _mel_filters: Tensor
    window: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _TokenizerEncoder()
        self.quantizer = _Quantizer()
        self.register_buffer("_mel_filters", torch.zeros(128, 201))
        self.register_buffer("window", torch.hann_window(400))

    def _log_mel(self, wav: Tensor) -> Tensor:
        stft = torch.stft(wav, 400, 160, window=self.window, return_complex=True)
        mag = stft[..., :-1].abs() ** 2
        mel = self._mel_filters @ mag
        log = torch.clamp(mel, min=1e-10).log10()
        log = torch.maximum(log, log.max() - 8.0)
        return (log + 4.0) / 4.0

    @torch.inference_mode()
    def tokenize(self, wav: Tensor, *, max_tokens: int | None = None) -> Tensor:
        """One clip in, one token row out. Clips are enrollment-sized (<=10 s);
        the upstream long-audio segmentation is deliberately not carried."""
        mel = self._log_mel(wav[None])
        if max_tokens is not None:
            mel = mel[..., : max_tokens * 4]
        hidden = self.encoder(mel)
        return self.quantizer._codebook.encode(hidden)[0]


# ------------------------------------------------------------ CAM++ x-vector


def _bn_relu(channels: int, affine: bool = True) -> nn.Sequential:
    seq = nn.Sequential()
    seq.add_module("batchnorm", nn.BatchNorm1d(channels, affine=affine))
    if affine:
        seq.add_module("relu", nn.ReLU(inplace=True))
    return seq


class _FCMBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.shortcut(x))


class _FCM(nn.Module):
    """2-D front-end: 80 mel bins -> 32 x 10 channel-frequency features."""

    def __init__(self, m_channels: int = 32, feat_dim: int = 80) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, m_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = nn.Sequential(
            _FCMBlock(m_channels, m_channels, 2), _FCMBlock(m_channels, m_channels, 1)
        )
        self.layer2 = nn.Sequential(
            _FCMBlock(m_channels, m_channels, 2), _FCMBlock(m_channels, m_channels, 1)
        )
        self.conv2 = nn.Conv2d(m_channels, m_channels, 3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        h = self.layer2(self.layer1(h))
        h = F.relu(self.bn2(self.conv2(h)))
        return h.reshape(h.shape[0], -1, h.shape[3])


class _TDNN(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, dilation: int = 1
    ) -> None:
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_ch, out_ch, kernel, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.nonlinear = _bn_relu(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.nonlinear(self.linear(x)))


class _CAMLayer(nn.Module):
    def __init__(self, bn_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.linear_local = nn.Conv1d(
            bn_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False
        )
        self.linear1 = nn.Conv1d(bn_ch, bn_ch // 2, 1)
        self.linear2 = nn.Conv1d(bn_ch // 2, out_ch, 1)

    def forward(self, x: Tensor) -> Tensor:
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self._seg_pool(x)
        context = F.relu(self.linear1(context))
        return cast(Tensor, y * torch.sigmoid(self.linear2(context)))

    @staticmethod
    def _seg_pool(x: Tensor, seg_len: int = 100) -> Tensor:
        seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        return seg[..., : x.shape[-1]]


class _CAMDenseLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bn_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.nonlinear1 = _bn_relu(in_ch)
        self.linear1 = nn.Conv1d(in_ch, bn_ch, 1, bias=False)
        self.nonlinear2 = _bn_relu(bn_ch)
        self.cam_layer = _CAMLayer(bn_ch, out_ch, kernel, dilation)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.cam_layer(self.nonlinear2(self.linear1(self.nonlinear1(x)))))


class _CAMDenseBlock(nn.ModuleList):
    def __init__(
        self, n_layers: int, in_ch: int, growth: int, bn_ch: int, kernel: int, dilation: int
    ) -> None:
        super().__init__()
        for i in range(n_layers):
            self.add_module(
                f"tdnnd{i + 1}",
                _CAMDenseLayer(in_ch + i * growth, growth, bn_ch, kernel, dilation),
            )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class _Transit(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.nonlinear = _bn_relu(in_ch)
        self.linear = nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.linear(self.nonlinear(x)))


class _Dense(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.linear = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.nonlinear = _bn_relu(out_ch, affine=False)

    def forward(self, x: Tensor) -> Tensor:
        squeeze = x.dim() == 2
        h = self.linear(x.unsqueeze(-1) if squeeze else x)
        if squeeze:
            h = h.squeeze(-1)
        return cast(Tensor, self.nonlinear(h))


class _StatsPool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x.mean(-1), x.std(-1, unbiased=True)], dim=-1)


class _CAMPPlus(nn.Module):
    """Kaldi-fbank frames -> 192-d x-vector (namespace ``s3gen.speaker_encoder``)."""

    def __init__(
        self, embedding_size: int = 192, growth: int = 32, init_channels: int = 128
    ) -> None:
        super().__init__()
        self.head = _FCM()
        channels = self.head.out_channels
        xv: OrderedDict[str, nn.Module] = OrderedDict()
        xv["tdnn"] = _TDNN(channels, init_channels, 5, stride=2)
        channels = init_channels
        for i, (n_layers, kernel, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2), strict=True)
        ):
            xv[f"block{i + 1}"] = _CAMDenseBlock(
                n_layers, channels, growth, 4 * growth, kernel, dilation
            )
            channels += n_layers * growth
            xv[f"transit{i + 1}"] = _Transit(channels, channels // 2)
            channels //= 2
        xv["out_nonlinear"] = _bn_relu(channels)
        xv["stats"] = _StatsPool()
        xv["dense"] = _Dense(channels * 2, embedding_size)
        self.xvector = nn.Sequential(xv)

    @torch.inference_mode()
    def embed(self, wav16: Tensor) -> Tensor:
        """One 16 kHz clip -> (192,) x-vector."""
        from torchaudio.compliance import kaldi

        fbank = kaldi.fbank(wav16[None], num_mel_bins=80)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        h = self.head(fbank[None].permute(0, 2, 1))
        return cast(Tensor, self.xvector(h)[0])


# --------------------------------------------------------- utterance encoder


class _VoiceEncoder(nn.Module):
    """The 256-d utterance speaker encoder the token generator reads.

    A 3-layer LSTM over 160-frame partials of a 40-mel power spectrogram,
    partials strided at ``rate`` windows per second, mean-pooled and
    L2-normalised, the Real-Time-Voice-Cloning recipe the model was trained
    with. Weights come from ``ve.safetensors``, which the packed checkpoint
    does not carry.
    """

    NUM_MELS = 40
    PARTIAL_FRAMES = 160

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(self.NUM_MELS, 256, num_layers=3, batch_first=True)
        self.proj = nn.Linear(256, 256)
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]))

    def _mel(self, wav16: NDArray[np.float32]) -> NDArray[np.float32]:
        import librosa

        spec = (
            np.abs(
                librosa.stft(
                    wav16, n_fft=400, hop_length=160, win_length=400, pad_mode="reflect"
                )
            )
            ** 2
        )
        basis = _slaney_mels(_S3_SR, 400, self.NUM_MELS, 0, 8000).numpy()
        # Tensor.numpy() is untyped in torch's stubs, so the product is Any.
        return cast(NDArray[np.float32], (basis @ spec).T.astype(np.float32))  # (frames, 40)

    @torch.inference_mode()
    def embed(
        self, wav16: NDArray[np.float32], *, rate: float = 1.3, min_coverage: float = 0.8
    ) -> Tensor:
        import librosa.effects

        trimmed, _ = librosa.effects.trim(wav16, top_db=20)  # type: ignore[attr-defined]
        mel = self._mel(trimmed)
        step = int(np.round((_S3_SR / rate) / self.PARTIAL_FRAMES))
        n_wins, remainder = divmod(max(len(mel) - self.PARTIAL_FRAMES + step, 0), step)
        if (
            n_wins == 0
            or (remainder + (self.PARTIAL_FRAMES - step)) / self.PARTIAL_FRAMES >= min_coverage
        ):
            n_wins += 1
        target = self.PARTIAL_FRAMES + step * (n_wins - 1)
        if target > len(mel):
            mel = np.concatenate(
                [mel, np.zeros((target - len(mel), self.NUM_MELS), np.float32)]
            )
        # On the module's own device, not the CPU. `build_torch_enroller`
        # accepts any device and moves the encoder there, but this tensor was
        # built from numpy, always CPU, and fed straight to the LSTM, so
        # `enroll()` on CUDA or MPS died with a device mismatch. The fixture is
        # CPU-only, which is why the tests never saw it.
        device = next(self.parameters()).device
        partials = torch.from_numpy(
            np.stack([mel[i * step : i * step + self.PARTIAL_FRAMES] for i in range(n_wins)])
        ).to(device)
        _, (hidden, _) = self.lstm(partials)
        raw = F.relu(self.proj(hidden[-1]))
        per_partial = raw / torch.linalg.norm(raw, dim=1, keepdim=True)
        pooled = per_partial.mean(0)
        return cast(Tensor, pooled / torch.linalg.norm(pooled))


# ---------------------------------------------------------------- enroller


class TorchVoiceEnroller:
    """``VoiceEnroller`` implementation on torch.

    Build via :func:`loudkit.backends.torch_backend.build_torch_enroller`.
    """

    def __init__(
        self,
        tokenizer: _S3Tokenizer,
        speaker_encoder: _CAMPPlus,
        voice_encoder: _VoiceEncoder | None,
        *,
        device: torch.device,
    ) -> None:
        self._tokenizer = tokenizer.to(device).eval()
        self._speaker_encoder = speaker_encoder.to(device).eval()
        self._voice_encoder = voice_encoder.to(device).eval() if voice_encoder else None
        self._device = device

    @classmethod
    def from_checkpoint(
        cls,
        ckpt: Checkpoint,
        *,
        device: torch.device,
        voice_encoder_weights: str | None = None,
    ) -> TorchVoiceEnroller:
        tokenizer = _S3Tokenizer()
        tokenizer.load_state_dict(
            {k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors("s3gen.tokenizer.").items()}
        )
        speaker = _CAMPPlus()
        speaker.load_state_dict(
            {
                k: torch.from_numpy(v.copy())
                for k, v in ckpt.tensors("s3gen.speaker_encoder.").items()
            }
        )
        voice_encoder = None
        if voice_encoder_weights is not None:
            from safetensors.torch import load_file

            voice_encoder = _VoiceEncoder()
            voice_encoder.load_state_dict(load_file(voice_encoder_weights))
        return cls(tokenizer, speaker, voice_encoder, device=device)

    def enroll(
        self, audio: NDArray[np.float32], sample_rate: int, *, name: str = ""
    ) -> VoiceProfile:
        """Derive a voice from a short reference recording.

        See ``docs/design/models-notes.md``.
        """
        # The whole preflight before any model runs, including the sample-rate check: a
        # non-positive rate reaches the resampler as a division by zero, and the four
        # implementations refuse it at the entry point with one sentence.
        wav = np.asarray(audio, dtype=np.float32)
        validate_reference_audio(wav, sample_rate)
        if self._voice_encoder is None:
            raise RuntimeError(
                "enrollment needs the 256-d utterance voice encoder, whose "
                "weights are not part of the packed checkpoint: pass "
                "voice_encoder_weights=... (ve.safetensors) when building the "
                "enroller"
            )

        # torch.from_numpy requires writable storage and rejects negative
        # strides. Audio decoded from a bytes buffer is commonly read-only, so
        # make an owned C-order copy only when the caller's array needs one.
        if not wav.flags.writeable or not wav.flags.c_contiguous:
            wav = np.array(wav, dtype=np.float32, order="C", copy=True)
        wav24_full = wav if sample_rate == _MEL_SR else resample(wav, sample_rate, _MEL_SR)
        # the prompt is capped at ten seconds (the static window holds ~9.5 s);
        # the utterance speaker embedding reads the WHOLE clip, that is how
        # the shipped voices were enrolled, and it is the better estimate
        wav24 = wav24_full[: int(_MAX_REF_SECONDS * _MEL_SR)]
        t24 = torch.from_numpy(wav24)
        wav16_flow = resample(wav24, _MEL_SR, _S3_SR)
        wav16_t3 = resample(wav24_full, _MEL_SR, _S3_SR)

        prompt_mel = _matcha_mel(t24).numpy()
        prompt_tokens = (
            self._tokenizer.tokenize(torch.from_numpy(wav16_flow).to(self._device))
            .cpu()
            .numpy()
        )
        # keep mel and tokens aligned at exactly TOKEN_MEL_RATIO frames per token
        n_tok = min(len(prompt_tokens), prompt_mel.shape[1] // TOKEN_MEL_RATIO)
        prompt_tokens = prompt_tokens[:n_tok]
        prompt_mel = prompt_mel[:, : TOKEN_MEL_RATIO * n_tok]

        cond_tokens = (
            self._tokenizer.tokenize(
                torch.from_numpy(wav16_t3[: 6 * _S3_SR]).to(self._device),
                max_tokens=_COND_PROMPT_TOKENS,
            )
            .cpu()
            .numpy()
        )

        flow_emb = (
            self._speaker_encoder.embed(torch.from_numpy(wav16_flow).to(self._device))
            .cpu()
            .numpy()
        )
        speaker_emb = self._voice_encoder.embed(wav16_t3).cpu().numpy()

        return VoiceProfile(
            name=name or "enrolled",
            speaker_embedding=speaker_emb.astype(np.float32),
            flow_embedding=flow_emb.astype(np.float32),
            prompt_tokens=prompt_tokens.astype(np.int64),
            prompt_mel=prompt_mel.astype(np.float32),
            cond_prompt_tokens=cond_tokens.astype(np.int64),
            source_sample_rate=sample_rate,
        )
