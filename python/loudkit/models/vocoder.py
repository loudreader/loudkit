"""The vocoder: HiFT (HiFiGAN + neural-source-filter excitation + iSTFT head).

See ``docs/design/models-notes.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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
# Tensor, the return is wrapped in cast(Tensor, ...), an assertion about
# torch's contract, not a guess. See docs/design/typing.md.

from ..config import AlgorithmConfig
from ..contracts import MEL_BINS, TOKEN_MEL_RATIO, Mel, Waveform
from ..voice import VoiceProfile
from .noise import gaussian_field, gaussian_field_torch, symmetric_uniforms
from .windowing import (
    UPSAMPLE_PER_FRAME,
    VOCODER_HARMONICS,
    VOCODER_NOISE_STREAM,
    VOCODER_PHASE_STREAM,
)

__all__ = [
    "TorchVocoder",
    "VOCODER_PHASE_STREAM",
    "VOCODER_NOISE_STREAM",
    "VOCODER_LENGTH_BUCKET",
    "VOCODER_RIGHT_CONTEXT",
]

VOCODER_LENGTH_BUCKET = 64
"""Ragged mel lengths are rounded up to this many frames.

See ``docs/design/models-notes.md``.
"""

VOCODER_RIGHT_CONTEXT = 32
"""Mel frames of right context the tail of a chunk needs, when the mel is not padded out to
the static window (``ExecutionConfig.vocoder_ragged``).

See ``docs/design/models-notes.md``.
"""


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
            self.activations1, self.convs1, self.activations2, self.convs2, strict=True
        ):
            x = c2(a2(c1(a1(x)))) + x
        return x


class _F0Predictor(nn.Module):
    """mel -> per-frame f0 in Hz: five conv+ELU stages and a linear head."""

    def __init__(self, in_channels: int = MEL_BINS, width: int = 512) -> None:
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

    The sine bank integrates ``cumsum(f0·k/sr mod 1)``, the accumulator whose
    range is why this whole module is fp32-only, then merges nine harmonics
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
        self.l_linear = nn.Linear(VOCODER_HARMONICS, 1)

    def forward(self, f0_up: Tensor, phase: Tensor, noise_unit: Tensor) -> Tensor:
        """f0_up (B, 1, T) at sample rate; phase (B, 9, 1); noise (B, 9, T)."""
        k = torch.arange(1, VOCODER_HARMONICS + 1, device=f0_up.device, dtype=f0_up.dtype)
        rate = f0_up * k[None, :, None] / self.sample_rate  # (B, 9, T)
        theta = 2.0 * math.pi * (torch.cumsum(rate, dim=-1) % 1.0)
        sine = self.sine_amp * torch.sin(theta + phase)
        voiced = (f0_up > self.voiced_threshold).to(f0_up.dtype)
        noise_amp = voiced * self.noise_std + (1.0 - voiced) * self.sine_amp / 3.0
        excitation = sine * voiced + noise_amp * noise_unit
        return torch.tanh(self.l_linear(excitation.transpose(1, 2))).transpose(1, 2)


class TorchVocoder(nn.Module):
    """``Vocoder`` implementation on torch (cpu / cuda / mps). fp32 only.

    See ``docs/design/models-notes.md``.
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

    def __init__(
        self,
        config: AlgorithmConfig,
        *,
        base_channels: int = 512,
        ragged: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.ragged = ragged
        """Pad the mel by the receptive field rather than out to the static
        window. Execution-layer: see ``ExecutionConfig.vocoder_ragged`` for what
        it costs and what it buys."""
        n_fft = self._N_FFT

        self.f0_predictor = _F0Predictor()
        self.m_source = _SourceModule(config.sample_rate)
        self.conv_pre = nn.Conv1d(MEL_BINS, base_channels, 7, padding=3)

        self.ups = nn.ModuleList()
        for i, (rate, kernel) in enumerate(
            zip(self._UPSAMPLE_RATES, self._UPSAMPLE_KERNELS, strict=True)
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
        for i, (rate, kernel) in enumerate(zip(down_rates, self._SOURCE_KERNELS, strict=True)):
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
        "coarser than the per-sample increment: the excitation degenerates "
        "and the render carries an audible tone at Nyquist. "
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

    def _pad_to_window(self, mel: Mel) -> tuple[NDArray[np.float32], int]:
        """The mel this stage actually runs, and how many frames that is.

        A mel longer than the static window is cut to the window: the graph
        takes a fixed number of frames and the frames past it have nowhere to
        go. Every other renderer does the same, so callers must cut their own
        output to ``min(mel frames, the returned count)`` rather than to the
        mel they passed in.
        """
        n_frames = int(mel.shape[1])
        window = self.config.window
        if window.static_length is None:
            return np.asarray(mel, dtype=np.float32), n_frames
        frames = TOKEN_MEL_RATIO * window.max_speech_tokens
        if self.ragged:
            # Only what the tail can see, rounded up to a shape the
            # convolutions are likely to have met before. Everything
            # downstream, the excitation noise most of all, which is drawn
            # per output sample, scales with this number rather than with
            # the window.
            want = n_frames + VOCODER_RIGHT_CONTEXT
            bucket = -(-want // VOCODER_LENGTH_BUCKET) * VOCODER_LENGTH_BUCKET
            frames = min(frames, bucket)
        padded = np.zeros((MEL_BINS, frames), dtype=np.float32)
        kept = min(n_frames, frames)
        padded[:, :kept] = mel[:, :kept]
        return padded, frames

    def _excitation(
        self, seed: int, n_samples: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Phase offsets ``(1, 9, 1)`` and per-sample noise ``(1, 9, n)``, one utterance.

        Drawn where it is consumed. The noise is 9 x n_samples standard normals,
        2.2 million for a full window, and in NumPy it was two thirds of this
        stage: computed on the host and then copied to the card it was always
        destined for. See ``docs/design/sampler-and-noise.md`` for why the
        stream is the same stream.
        """
        phase = np.zeros((1, VOCODER_HARMONICS, 1), dtype=np.float32)
        phase[0, 1:, 0] = symmetric_uniforms(
            seed, VOCODER_PHASE_STREAM, VOCODER_HARMONICS - 1, math.pi
        )
        if device.type == "cuda":
            noise_t = gaussian_field_torch(
                seed, VOCODER_NOISE_STREAM, VOCODER_HARMONICS, n_samples, device
            )[None]
        else:
            # Not everywhere: the transform runs in float64, which MPS has no
            # datapath for at all, and on CPU the NumPy path is the reference
            # this one is checked against. CUDA is where the cost was.
            noise_np = gaussian_field(seed, VOCODER_NOISE_STREAM, VOCODER_HARMONICS, n_samples)[
                None
            ]
            noise_t = torch.from_numpy(noise_np).to(device)
        return torch.from_numpy(phase).to(device), noise_t

    def _render(self, m: Tensor, phase: Tensor, noise_t: Tensor) -> Tensor:
        """Padded mel plus its excitation to a waveform, one row or many.

        Everything from the f0 prediction to the iSTFT, which does not depend
        on how many rows are in flight, so :meth:`synthesize` and
        :meth:`synthesize_batch` share it rather than each keeping a copy.
        What differs is how the excitation is drawn and where the output is
        cut, both of which stay with the caller.
        """
        f0 = self.f0_predictor(m)
        f0_up = F.interpolate(
            f0[:, None], scale_factor=float(UPSAMPLE_PER_FRAME), mode="nearest"
        )
        source = self.m_source(f0_up, phase, noise_t)
        return self._decode(m, source)

    @torch.inference_mode()
    def synthesize(self, mel: Mel, voice: VoiceProfile, *, seed: int) -> Waveform:
        """Render audio from a mel. ``voice`` is unused by this stage (the
        timbre already lives in the mel) but stays in the signature so every
        renderer, including ones that condition here, shares one contract.
        """
        del voice
        device = self.conv_pre.weight.device
        padded, frames = self._pad_to_window(mel)
        n_frames = min(int(mel.shape[1]), frames)
        m = torch.from_numpy(padded)[None].to(device)

        phase, noise_t = self._excitation(seed, frames * UPSAMPLE_PER_FRAME, device)
        wav = self._render(m, phase, noise_t)[0]
        return wav[: n_frames * UPSAMPLE_PER_FRAME].float().cpu().numpy()

    @torch.inference_mode()
    def synthesize_batch(
        self,
        mels: Sequence[Mel],
        voices: Sequence[VoiceProfile],
        *,
        seeds: Sequence[int],
    ) -> list[Waveform]:
        """:meth:`synthesize` for several mels at once, one row each.

        A prototype beside the contract, not on it. Nothing in the engine calls
        it: measured, batching this stage is slower than the serial loop on the
        GPU and buys 10% to 24% on CPU, where the renderer is latency-bound
        anyway. See ``docs/design/models-notes.md``, and
        ``research/bench_render.py`` to re-measure on another device.

        Every mel must pad to the same frame count. The excitation is drawn per
        output sample and the iSTFT tail is not padding-invariant, so a longer
        neighbour would move this row's last samples.
        """
        rows = len(mels)
        if rows == 0 or len(voices) != rows or len(seeds) != rows:
            raise ValueError(
                f"synthesize_batch needs one voice and one seed per mel: "
                f"{rows} mels, {len(voices)} voices, {len(seeds)} seeds"
            )
        device = self.conv_pre.weight.device
        padded = [self._pad_to_window(mel) for mel in mels]
        widths = sorted({p[1] for p in padded})
        if len(widths) != 1:
            raise ValueError(
                f"synthesize_batch needs one padded length for the whole batch, got "
                f"{widths}; group the mels by length, or turn off ExecutionConfig."
                f"vocoder_ragged, which is what makes the length depend on the mel"
            )
        m = torch.from_numpy(np.stack([p[0] for p in padded])).to(device)

        n_samples = widths[0] * UPSAMPLE_PER_FRAME
        drawn = [self._excitation(seed, n_samples, device) for seed in seeds]
        phase = torch.cat([d[0] for d in drawn])
        noise_t = torch.cat([d[1] for d in drawn])

        wav = self._render(m, phase, noise_t)
        return [
            wav[i, : min(int(mel.shape[1]), widths[0]) * UPSAMPLE_PER_FRAME]
            .float()
            .cpu()
            .numpy()
            for i, mel in enumerate(mels)
        ]
