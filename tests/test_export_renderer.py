"""The shared graph adapters retain torch's STFT/iSTFT computation."""

from __future__ import annotations

import torch
from tools.export_renderer import _ConvISTFT, _ConvSTFT


def test_convolution_transforms_match_torch() -> None:
    n_fft, hop = 16, 4
    window = torch.hann_window(n_fft)
    audio = torch.randn(2, 256, generator=torch.Generator().manual_seed(7))
    spectrum = torch.stft(audio, n_fft, hop, window=window, return_complex=True)
    packed = _ConvSTFT(n_fft, hop, window)(audio)
    real, imag = packed.chunk(2, dim=1)
    torch.testing.assert_close(real, spectrum.real, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(imag, spectrum.imag, atol=1e-5, rtol=1e-5)
    restored = _ConvISTFT(n_fft, hop, window, spectrum.shape[-1])(real, imag)
    expected = torch.istft(spectrum, n_fft, hop, window=window)
    torch.testing.assert_close(restored, expected, atol=1e-5, rtol=1e-5)
