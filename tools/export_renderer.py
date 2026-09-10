"""Renderer adapters shared by the ONNX and CoreML exporters.

The fixed convolution bases replace STFT/iSTFT with operations both graph
formats support. The probe inputs and the torch references every gate compares
against are built here too, so a changed seed, shape or tolerance reaches both
exporters. Exporters retain their own conversion and parity gates.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from loudkit.models.flow import TorchMelDecoder
from loudkit.models.noise import gaussian_field, symmetric_uniforms
from loudkit.models.vocoder import (
    VOCODER_NOISE_STREAM,
    VOCODER_PHASE_STREAM,
    TorchVocoder,
)
from loudkit.voice import VoiceProfile

_MEL_BINS = 80
_N_HARMONICS = 9
_HOP_SAMPLES = 480

RENDERER_STAGES = ("encoder", "estimator", "vocoder")
"""The renderer graphs, in the order both exporters write them."""

RENDERER_INPUTS = {
    "encoder": ["prompt_token", "speech_tokens"],
    "estimator": ["x", "mu", "t", "spks", "cond"],
    "vocoder": ["mel", "phase", "noise"],
}
"""Graph input names per stage; the examples below are fed under these."""

NO_VOICE = cast(VoiceProfile, None)
"""What the exporters hand `TorchVocoder.synthesize` for its `voice`.

That stage does not read it: the timbre is already in the mel, and the method
opens with `del voice`. The parameter stays in the signature so every renderer,
including ones that do condition there, shares one contract. Cast rather than
suppressed, because the value really is absent and the annotation really does
say `VoiceProfile`, and a checked profile does not exist at export time.
"""


class _EncoderWrapper(nn.Module):
    """prompt_token + speech_tokens -> mu. The voice arrives as data."""

    def __init__(self, decoder: TorchMelDecoder) -> None:
        super().__init__()
        self.input_embedding = decoder.input_embedding
        self.encoder = decoder.encoder
        self.encoder_proj = decoder.encoder_proj

    def forward(self, prompt_token: torch.Tensor, speech_tokens: torch.Tensor) -> torch.Tensor:
        row = torch.cat([prompt_token.long(), speech_tokens.long()], dim=1)
        h = self.encoder(self.input_embedding(row))
        return self.encoder_proj(h).transpose(1, 2).contiguous()


class _EstimatorWrapper(nn.Module):
    def __init__(self, decoder: TorchMelDecoder) -> None:
        super().__init__()
        self.est = decoder.decoder.estimator

    def forward(self, x, mu, t, spks, cond):
        return self.est(x, mu, t, spks, cond)


# The STFT/iSTFT-as-convolution rewrite, lifted from the production export.
# Math-equivalent restatements of torch.stft / torch.istft at n_fft=16, hop=4,
# hann, center=True, asserted below against the torch vocoder before export.


class _ConvSTFT(nn.Module):
    # Registered buffers, declared: nn.Module.__getattr__ returns
    # ``Tensor | Module``, which no conv takes.
    basis: torch.Tensor

    def __init__(self, n_fft: int, hop: int, window: torch.Tensor) -> None:
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        freqs = n_fft // 2 + 1
        n = torch.arange(n_fft, dtype=torch.float32)
        k = torch.arange(freqs, dtype=torch.float32).view(-1, 1)
        ang = 2 * np.pi * k * n / n_fft
        basis_r = (torch.cos(ang) * window).unsqueeze(1)
        basis_i = (-torch.sin(ang) * window).unsqueeze(1)
        self.register_buffer("basis", torch.cat([basis_r, basis_i], dim=0))
        self.freqs = freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        return F.conv1d(x, self.basis, stride=self.hop)  # [B, 2F, TT]


class _ConvISTFT(nn.Module):
    inv_basis: torch.Tensor
    ola: torch.Tensor
    env: torch.Tensor

    def __init__(self, n_fft: int, hop: int, window: torch.Tensor, frames: int) -> None:
        super().__init__()
        self.n_fft, self.hop, self.frames = n_fft, hop, frames
        freqs = n_fft // 2 + 1
        k = torch.arange(freqs, dtype=torch.float32).view(-1, 1)
        n = torch.arange(n_fft, dtype=torch.float32)
        ang = 2 * np.pi * k * n / n_fft
        wk = torch.full((freqs, 1), 2.0)
        wk[0] = 1.0
        wk[-1] = 1.0
        inv_r = wk * torch.cos(ang) / n_fft
        inv_i = -wk * torch.sin(ang) / n_fft
        inv = torch.cat([inv_r, inv_i], dim=0)
        self.register_buffer("inv_basis", inv.t().unsqueeze(-1))
        ola = torch.zeros(n_fft, 1, n_fft)
        for c in range(n_fft):
            ola[c, 0, c] = window[c]
        self.register_buffer("ola", ola)
        total = n_fft + hop * (frames - 1)
        env = torch.zeros(total)
        w2 = window * window
        for t in range(frames):
            env[t * hop : t * hop + n_fft] += w2
        self.register_buffer("env", env.clamp_min(1e-8))
        self.crop = n_fft // 2

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        spec = torch.cat([real, imag], dim=1)
        frames_t = F.conv1d(spec, self.inv_basis)
        y = F.conv_transpose1d(frames_t, self.ola, stride=self.hop)
        y = y.squeeze(1) / self.env
        return y[:, self.crop : y.shape[1] - self.crop]


class _VocoderWrapper(nn.Module):
    """TorchVocoder's forward with injected randomness and conv STFT/iSTFT."""

    def __init__(self, voc: TorchVocoder, mel_frames: int) -> None:
        super().__init__()
        self.voc = voc
        n_fft, hop = voc._N_FFT, voc._HOP
        window = voc.stft_window
        self.stft = _ConvSTFT(n_fft, hop, window)
        # probe the conv_post frame count to size the static iSTFT envelope
        with torch.no_grad():
            x = voc.conv_pre(torch.zeros(1, _MEL_BINS, mel_frames))
            for i in range(len(voc.ups)):
                x = voc.ups[i](F.leaky_relu(x, 0.1))
                if i == len(voc.ups) - 1:
                    x = F.pad(x, (1, 0), mode="reflect")
            out_frames = x.shape[-1]
        self.istft = _ConvISTFT(n_fft, hop, window, out_frames)
        self.n_fft = n_fft

    def forward(self, mel, phase, noise):
        voc = self.voc
        f0 = voc.f0_predictor(mel)
        f0_up = F.interpolate(f0[:, None], scale_factor=float(_HOP_SAMPLES), mode="nearest")
        source = voc.m_source(f0_up, phase, noise)  # [B, 1, T]

        s_stft = self.stft(source.squeeze(1))
        x = voc.conv_pre(mel)
        n_kernels = len(voc._RESBLOCK_KERNELS)
        for i in range(len(voc.ups)):
            x = voc.ups[i](F.leaky_relu(x, 0.1))
            if i == len(voc.ups) - 1:
                x = F.pad(x, (1, 0), mode="reflect")
            tap = voc.source_resblocks[i](voc.source_downs[i](s_stft))
            x = x + tap
            acc = voc.resblocks[i * n_kernels](x)
            for j in range(1, n_kernels):
                acc = acc + voc.resblocks[i * n_kernels + j](x)
            x = acc / n_kernels
        x = voc.conv_post(F.leaky_relu(x))
        freqs = self.n_fft // 2 + 1
        magnitude = torch.exp(x[:, :freqs]).clamp(max=1e2)
        phase_pred = torch.sin(x[:, freqs:])
        real = magnitude * torch.cos(phase_pred)
        imag = magnitude * torch.sin(phase_pred)
        y = self.istft(real, imag)
        return torch.clamp(y, -voc._AUDIO_LIMIT, voc._AUDIO_LIMIT)


def renderer_geometry(ckpt, algo) -> tuple[int, int]:
    """The static mel and vocoder frame counts, announced as the exporters do.

    Both graph sets are fixed-shape, so a window recipe without the two static
    numbers has nothing to export against.
    """
    w = algo.window
    assert w.static_length is not None, "static window recipe missing (static_length)"
    assert w.static_prompt_tokens is not None, (
        "static window recipe missing (static_prompt_tokens): these graphs are static-shape"
    )
    t_mel = 2 * (w.static_prompt_tokens + w.static_length)
    hift_frames = 2 * w.max_speech_tokens
    print(
        f"checkpoint {ckpt.path.name}: window {w.static_prompt_tokens}+{w.static_length} "
        f"-> estimator T{t_mel}, vocoder {hift_frames} frames"
    )
    print(f"algorithm: {algo.describe()}")
    return t_mel, hift_frames


def load_mel_decoder(ckpt, algo) -> TorchMelDecoder:
    """The packed flow weights, before the caller's own fp32 and eval pass.

    Left to the caller because the CoreML exporter swaps a distilled estimator
    in between the load and the cast.
    """
    decoder = TorchMelDecoder(algo)
    decoder.load_state_dict(
        {k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors("s3gen.flow.").items()}
    )
    return decoder


def renderer_stage(
    stage: str, decoder: TorchMelDecoder, ckpt, algo, t_mel: int, hift_frames: int
) -> tuple[nn.Module, tuple[torch.Tensor, ...], torch.Tensor]:
    """The module, its probe inputs and the torch reference to gate against.

    Seeded generators rather than the global stream, so what a stage is gated
    on does not depend on which other stages the same run exports.
    """
    if stage == "encoder":
        module: nn.Module = _EncoderWrapper(decoder).eval()
        w = algo.window
        example: tuple[torch.Tensor, ...] = (
            torch.randint(
                0, 6561, (1, w.static_prompt_tokens), generator=torch.Generator().manual_seed(3)
            ),
            torch.randint(
                0, 6561, (1, w.static_length), generator=torch.Generator().manual_seed(4)
            ),
        )
        with torch.no_grad():
            reference = module(*example)
        assert reference.shape == (1, _MEL_BINS, t_mel), reference.shape
        return module, example, reference

    if stage == "estimator":
        module = _EstimatorWrapper(decoder).eval()
        rng = torch.Generator().manual_seed(5)
        example = (
            torch.randn(1, _MEL_BINS, t_mel, generator=rng),
            torch.randn(1, _MEL_BINS, t_mel, generator=rng),
            torch.tensor([0.4]),
            torch.randn(1, _MEL_BINS, generator=rng),
            torch.randn(1, _MEL_BINS, t_mel, generator=rng),
        )
        with torch.no_grad():
            reference = module(*example)
        return module, example, reference

    return _vocoder_stage(ckpt, algo, hift_frames)


def _vocoder_stage(
    ckpt, algo, hift_frames: int
) -> tuple[nn.Module, tuple[torch.Tensor, ...], torch.Tensor]:
    voc = TorchVocoder(algo)
    voc.load_state_dict(
        {k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors("s3gen.mel2wav.").items()}
    )
    voc = voc.eval()

    # probe mel: the committed reference render, padded to the static frame
    # count: real speech statistics, not random noise (measurement rule 3)
    ref_mel_path = Path(__file__).resolve().parent.parent / "tests/data/reference/s0_mel.npy"
    if ref_mel_path.exists():
        mel_np = np.load(ref_mel_path)
    else:
        mel_np = np.random.default_rng(0).normal(-5.0, 2.0, (80, 300)).astype(np.float32)
        print("  (reference mel not found; probing with synthetic mel)")
    mel = np.zeros((1, _MEL_BINS, hift_frames), dtype=np.float32)
    n_real = min(mel_np.shape[1], hift_frames)
    mel[0, :, :n_real] = mel_np[:, :n_real]
    mel_t = torch.from_numpy(mel)

    seed = 1234
    n_samples = hift_frames * _HOP_SAMPLES
    phase: np.ndarray = np.zeros((1, _N_HARMONICS, 1), dtype=np.float32)
    phase[0, 1:, 0] = symmetric_uniforms(seed, VOCODER_PHASE_STREAM, _N_HARMONICS - 1, np.pi)
    noise = gaussian_field(seed, VOCODER_NOISE_STREAM, _N_HARMONICS, n_samples)[None]
    phase_t, noise_t = torch.from_numpy(phase), torch.from_numpy(noise)

    wrapper = _VocoderWrapper(voc, hift_frames).eval()
    with torch.no_grad():
        wav_conv = wrapper(mel_t, phase_t, noise_t)
        wav_ref = torch.from_numpy(voc.synthesize(mel_np[:, :n_real], NO_VOICE, seed=seed))
    # torch-level gate: the conv STFT/iSTFT rewrite must be equivalent before
    # conversion enters the picture
    d = (wav_conv[0, : n_real * _HOP_SAMPLES] - wav_ref).abs().max().item()
    print(f"  conv-STFT rewrite vs torch.stft/istft: max|err| {d:.3e}")
    assert d < 1e-4, "conv STFT/iSTFT rewrite is not equivalent to the torch vocoder"
    return wrapper, (mel_t, phase_t, noise_t), wav_conv
