"""The vocoder: HiFT (HiFiGAN + neural-source-filter excitation + iSTFT head).

Checkpoint namespace ``s3gen.mel2wav``, names mirrored, weight-norm already
folded at pack time — the convolutions here are plain convolutions carrying
the exact tensors the parametrised forward would have computed.

Signal path: mel -> f0 (a small conv net) -> harmonic sine excitation at
24 kHz (nine harmonics, cumulative-phase NSF source) -> STFT of the excitation
fused into the HiFiGAN upsampling stack at every scale -> predicted magnitude
and phase -> iSTFT -> waveform.

**fp32 only, enforced at construction.** The NSF source accumulates phase with
a running ``cumsum`` that reaches roughly 1400 cycles over a ten-second
render; at that magnitude fp16's resolution is coarser than the per-sample
phase increment, the excitation degenerates, and the result is an audible tone
at Nyquist (EXP-003 measured it as a ~12 kHz whine present in 80% of frames).
This is a property of the algorithm, not of any one backend, so the refusal
lives here rather than in a backend checklist.

Randomness (the harmonic phase offsets and the excitation noise) is
Philox-addressed data from :mod:`.noise` — the same seed produces the same
excitation on every device, and the noise is drawn fresh per sample (no
Box–Muller spare caching; the cached-spare variant put +5.3 dB at Nyquist).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NoReturn, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn

if TYPE_CHECKING:
    from typing_extensions import Self

# Typing note: torch types nn.Module.__call__ as Any, so submodule calls in a
# forward pass propagate Any. Where the callee's forward provably returns a
# Tensor, the return is wrapped in cast(Tensor, ...) — an assertion about
# torch's contract, not a guess. See docs/reference/typing.md.

from ..config import AlgorithmConfig
from ..contracts import Mel, Waveform
from ..voice import VoiceProfile
from .noise import gaussian_field, symmetric_uniforms
from .windowing import VOCODER_NOISE_STREAM, VOCODER_PHASE_STREAM

__all__ = ["TorchVocoder", "VOCODER_PHASE_STREAM", "VOCODER_NOISE_STREAM"]

_MEL_BINS = 80
_N_HARMONICS = 9  # harmonic_num 8 + the fundamental
_UPSAMPLE_PER_FRAME = 480  # 24 kHz / 50 Hz mel rate


class _Snake(nn.Module):
    """Snake activation ``x + sin²(αx)/α`` with a per-channel learned α."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))

    def forward(self, x: Tensor) -> Tensor:
        alpha = self.alpha[None, :, None]
        return x + torch.sin(x * alpha).pow(2) / (alpha + 1e-9)


class _ResBlock(nn.Module):
    """HiFiGAN residual block: dilated conv pairs with Snake activations."""

    def __init__(self, channels: int, kernel: int, dilations: tuple[int, ...]) -> None:
        super().__init__()

        def conv(dilation: int) -> nn.Conv1d:
            pad = (kernel * dilation - dilation) // 2
            return nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=pad)

        self.convs1 = nn.ModuleList(conv(d) for d in dilations)
        self.convs2 = nn.ModuleList(conv(1) for _ in dilations)
        self.activations1 = nn.ModuleList(_Snake(channels) for _ in dilations)
        self.activations2 = nn.ModuleList(_Snake(channels) for _ in dilations)

    def forward(self, x: Tensor) -> Tensor:
        for a1, c1, a2, c2 in zip(
            self.activations1, self.convs1, self.activations2, self.convs2, strict=False
        ):
            x = c2(a2(c1(a1(x)))) + x
        return x


class _F0Predictor(nn.Module):
    """mel -> per-frame f0 in Hz: five conv+ELU stages and a linear head."""

    def __init__(self, in_channels: int = _MEL_BINS, width: int = 512) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_channels
        for _ in range(5):
            layers += [nn.Conv1d(ch, width, 3, padding=1), nn.ELU()]
            ch = width
        self.condnet = nn.Sequential(*layers)
        self.classifier = nn.Linear(width, 1)

    def forward(self, mel: Tensor) -> Tensor:
        h = self.condnet(mel).transpose(1, 2)
        return cast(Tensor, self.classifier(h).squeeze(-1).abs())


class _SourceModule(nn.Module):
    """NSF harmonic source with *injected* randomness.

    The sine bank integrates ``cumsum(f0·k/sr mod 1)`` — the accumulator whose
    range is why this whole module is fp32-only — then merges nine harmonics
    through a 9->1 linear + tanh. Phase offsets and the per-sample noise come
    in as arguments; this module draws nothing itself.
    """

    def __init__(
        self,
        sample_rate: int,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 10.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.l_linear = nn.Linear(_N_HARMONICS, 1)

    def forward(self, f0_up: Tensor, phase: Tensor, noise_unit: Tensor) -> Tensor:
        """f0_up (B, 1, T) at sample rate; phase (B, 9, 1); noise (B, 9, T)."""
        k = torch.arange(1, _N_HARMONICS + 1, device=f0_up.device, dtype=f0_up.dtype)
        rate = f0_up * k[None, :, None] / self.sample_rate  # (B, 9, T)
        theta = 2.0 * math.pi * (torch.cumsum(rate, dim=-1) % 1.0)
        sine = self.sine_amp * torch.sin(theta + phase)
        voiced = (f0_up > self.voiced_threshold).to(f0_up.dtype)
        noise_amp = voiced * self.noise_std + (1.0 - voiced) * self.sine_amp / 3.0
        excitation = sine * voiced + noise_amp * noise_unit
        return torch.tanh(self.l_linear(excitation.transpose(1, 2))).transpose(1, 2)


class TorchVocoder(nn.Module):
    """``Vocoder`` implementation on torch (cpu / cuda / mps). fp32 only.

    Static geometry: when the window recipe is static, the mel is zero-padded
    to ``2 x max_speech_tokens`` frames before rendering and the waveform is
    trimmed back to the real region afterwards — the same framing the shipped
    HiFT graph runs, whose tail-padding effects bleed a few frames back into
    the kept audio through the conv stack and are therefore part of the
    algorithm, not an export artefact.
    """

    _UPSAMPLE_RATES = (8, 5, 3)
    _UPSAMPLE_KERNELS = (16, 11, 7)
    _SOURCE_KERNELS = (7, 7, 11)
    _RESBLOCK_KERNELS = (3, 7, 11)
    _DILATIONS = (1, 3, 5)
    _N_FFT = 16
    _HOP = 4
    _AUDIO_LIMIT = 0.99

    stft_window: Tensor  # registered buffer; annotated so access is not Tensor | Module

    def __init__(self, config: AlgorithmConfig, *, base_channels: int = 512) -> None:
        super().__init__()
        self.config = config
        n_fft, _hop = self._N_FFT, self._HOP

        self.f0_predictor = _F0Predictor()
        self.m_source = _SourceModule(config.sample_rate)
        self.conv_pre = nn.Conv1d(_MEL_BINS, base_channels, 7, padding=3)

        self.ups = nn.ModuleList()
        for i, (rate, kernel) in enumerate(
            zip(self._UPSAMPLE_RATES, self._UPSAMPLE_KERNELS, strict=False)
        ):
            self.ups.append(
                nn.ConvTranspose1d(
                    base_channels // (2**i),
                    base_channels // (2 ** (i + 1)),
                    kernel,
                    rate,
                    padding=(kernel - rate) // 2,
                )
            )

        # excitation taps: the source STFT is downsampled to each scale
        down_rates = np.cumprod([1, *self._UPSAMPLE_RATES[::-1][:-1]])[::-1]
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(down_rates, self._SOURCE_KERNELS, strict=False)):
            ch = base_channels // (2 ** (i + 1))
            if rate == 1:
                self.source_downs.append(nn.Conv1d(n_fft + 2, ch, 1))
            else:
                self.source_downs.append(
                    nn.Conv1d(n_fft + 2, ch, int(rate) * 2, int(rate), padding=int(rate) // 2)
                )
            self.source_resblocks.append(_ResBlock(ch, kernel, self._DILATIONS))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for kernel in self._RESBLOCK_KERNELS:
                self.resblocks.append(_ResBlock(ch, kernel, self._DILATIONS))

        self.conv_post = nn.Conv1d(ch, n_fft + 2, 7, padding=3)
        self.register_buffer("stft_window", torch.hann_window(n_fft), persistent=False)

    # -- precision guard -----------------------------------------------------

    def half(self) -> NoReturn:
        raise TypeError(self._FP16_REFUSAL)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        moved = super().to(*args, **kwargs)
        if any(p.dtype in (torch.float16, torch.bfloat16) for p in moved.parameters()):
            raise TypeError(self._FP16_REFUSAL)
        return moved

    _FP16_REFUSAL = (
        "the vocoder is fp32-only: its NSF source accumulates phase with a "
        "running sum that reaches ~1400 cycles, where fp16 resolution is "
        "coarser than the per-sample increment — the excitation degenerates "
        "and the render carries an audible tone at Nyquist (EXP-003). "
        "ExecutionConfig.precision['vocoder'] must stay 'fp32'."
    )

    # -- rendering -----------------------------------------------------------

    def _stft(self, x: Tensor) -> Tensor:
        spec = torch.stft(
            x,
            self._N_FFT,
            self._HOP,
            self._N_FFT,
            window=self.stft_window,
            return_complex=True,
        )
        return torch.cat([spec.real, spec.imag], dim=1)

    def _istft(self, magnitude: Tensor, phase: Tensor) -> Tensor:
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return torch.istft(
            torch.complex(real, imag),
            self._N_FFT,
            self._HOP,
            self._N_FFT,
            window=self.stft_window,
        )

    def _decode(self, mel: Tensor, source: Tensor) -> Tensor:
        s_stft = self._stft(source.squeeze(1))
        x = self.conv_pre(mel)
        n_kernels = len(self._RESBLOCK_KERNELS)
        for i in range(len(self.ups)):
            x = self.ups[i](F.leaky_relu(x, 0.1))
            if i == len(self.ups) - 1:
                x = F.pad(x, (1, 0), mode="reflect")
            tap = self.source_resblocks[i](self.source_downs[i](s_stft))
            x = x + tap
            # start the sum from kernel 0 rather than from None so the
            # accumulator is never Optional; n_kernels is a fixed constant >= 1
            acc = cast(Tensor, self.resblocks[i * n_kernels](x))
            for j in range(1, n_kernels):
                acc = acc + self.resblocks[i * n_kernels + j](x)
            x = acc / n_kernels
        x = self.conv_post(F.leaky_relu(x))
        freqs = self._N_FFT // 2 + 1
        magnitude = torch.exp(x[:, :freqs])
        phase = torch.sin(x[:, freqs:])
        return torch.clamp(self._istft(magnitude, phase), -self._AUDIO_LIMIT, self._AUDIO_LIMIT)

    @torch.inference_mode()
    def synthesize(self, mel: Mel, voice: VoiceProfile, *, seed: int) -> Waveform:
        """Render audio from a mel. ``voice`` is unused by this stage (the
        timbre already lives in the mel) but stays in the signature so every
        renderer — including ones that condition here — shares one contract.
        """
        del voice
        device = self.conv_pre.weight.device
        n_frames = int(mel.shape[1])

        window = self.config.window
        padded: NDArray[np.float32]
        if window.static_length is not None:
            frames = 2 * window.max_speech_tokens
            padded = np.zeros((_MEL_BINS, frames), dtype=np.float32)
            padded[:, :n_frames] = mel[:, :frames]
        else:
            frames = n_frames
            padded = np.asarray(mel, dtype=np.float32)
        m = torch.from_numpy(padded)[None].to(device)

        f0 = self.f0_predictor(m)
        f0_up = F.interpolate(
            f0[:, None], scale_factor=float(_UPSAMPLE_PER_FRAME), mode="nearest"
        )

        n_samples = frames * _UPSAMPLE_PER_FRAME
        phase = np.zeros((1, _N_HARMONICS, 1), dtype=np.float32)
        phase[0, 1:, 0] = symmetric_uniforms(
            seed, VOCODER_PHASE_STREAM, _N_HARMONICS - 1, math.pi
        )
        noise = gaussian_field(seed, VOCODER_NOISE_STREAM, _N_HARMONICS, n_samples)[None]

        source = self.m_source(
            f0_up, torch.from_numpy(phase).to(device), torch.from_numpy(noise).to(device)
        )
        wav = self._decode(m, source)[0]
        return wav[: n_frames * _UPSAMPLE_PER_FRAME].float().cpu().numpy()
