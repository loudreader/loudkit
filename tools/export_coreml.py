"""Export the CoreML renderer graphs from the packed checkpoint.

The three packages this writes are the Apple execution path of the renderer —
the same graph geometry the shipped app runs (query 255 / prompt 238 → T986
mel, HiFT at 510 frames), rebuilt from ``loudr-1.safetensors`` so their
weights trace to the packed checkpoint rather than to whichever research
artifact happened to be on disk when the app was built:

  flow_encoder.mlpackage    prompt_token [1,238] + speech_tokens [1,255] -> mu [1,80,986]
                            fp32, CPU. fp16 here is measured fatal
                            (mel corr 0.619, +22 dB HF — EXP-011).
  flow_estimator.mlpackage  x/mu/t/spks/cond @ T986 -> velocity. fp16 pipeline,
                            CPU+ANE (EXP-011: mel corr 0.999999 under fp16).
  vocoder.mlpackage         mel [1,80,510] + phase [1,9,1] + noise [1,9,244800]
                            -> wav [1, 244800]. fp32, CPU. fp16 puts an audible
                            tone at Nyquist (EXP-003).

Randomness is *input* on every graph: the flow prior, the harmonic phase
offsets and the excitation noise arrive as tensors, so the packages compute a
pure function and parity is a checkable number, not a vibe.

The vocoder needs two math-equivalent rewrites to convert (coremltools has no
stft/istft): the fixed-basis conv STFT/iSTFT pair, lifted from the production
export (chatterbox-apple/export/hift_static.py) and asserted here against the
torch vocoder before anything is saved.

Every stage is gated against the torch modules loaded from the same checkpoint;
a failed gate raises and nothing half-exported is left behind (writes go to the
final name only after the gate passes).

Usage (from the loudkit repo root):

  .venv/bin/python tools/export_coreml.py \
      --checkpoint /path/to/loudr-1.safetensors [--out DIR] [--stages ...]

Default --out is a ``coreml/`` directory beside the checkpoint, which is where
the coreml backend looks first.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from loudkit.backends import production_algorithm  # noqa: E402
from loudkit.checkpoint import Checkpoint  # noqa: E402
from loudkit.models.flow import TorchMelDecoder  # noqa: E402
from loudkit.models.noise import gaussian_field, symmetric_uniforms  # noqa: E402
from loudkit.models.vocoder import (  # noqa: E402
    VOCODER_NOISE_STREAM,
    VOCODER_PHASE_STREAM,
    TorchVocoder,
)

_MEL_BINS = 80
_N_HARMONICS = 9
_HOP_SAMPLES = 480

ENCODER_NAME = "flow_encoder.mlpackage"
ESTIMATOR_NAME = "flow_estimator.mlpackage"
VOCODER_NAME = "vocoder.mlpackage"


# --------------------------------------------------------------- wrappers


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

    def forward(self, x, mu, t, spks, cond):  # type: ignore[no-untyped-def]
        return self.est(x, mu, t, spks, cond)


# The STFT/iSTFT-as-convolution rewrite, lifted from the production export
# (chatterbox-apple/export/hift_static.py). Math-equivalent restatements of
# torch.stft / torch.istft at n_fft=16, hop=4, hann, center=True — asserted
# below against the torch vocoder before export.


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

    def forward(self, mel, phase, noise):  # type: ignore[no-untyped-def]
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
            acc = None
            for j in range(n_kernels):
                r = voc.resblocks[i * n_kernels + j](x)
                acc = r if acc is None else acc + r
            x = acc / n_kernels
        x = voc.conv_post(F.leaky_relu(x))
        freqs = self.n_fft // 2 + 1
        magnitude = torch.exp(x[:, :freqs]).clamp(max=1e2)
        phase_pred = torch.sin(x[:, freqs:])
        real = magnitude * torch.cos(phase_pred)
        imag = magnitude * torch.sin(phase_pred)
        y = self.istft(real, imag)
        return torch.clamp(y, -voc._AUDIO_LIMIT, voc._AUDIO_LIMIT)


# ------------------------------------------------------------------ export


def _install_ct_shims() -> None:
    """Torch ops the coremltools frontend lacks; register before convert.

    ``view_as`` (encoder rel-pos attention) and ``broadcast_tensors`` are the
    two the production exports needed (chatterbox-apple/export/coreml_shims.py,
    found by the P1 spike 2026-07-16). Inlined so this script has no import
    outside the repo. Idempotent.
    """
    from coremltools.converters.mil import Builder as mb  # noqa: N813
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        _TORCH_OPS_REGISTRY,
        register_torch_op,
    )

    registered = getattr(_TORCH_OPS_REGISTRY, "name_to_func_mapping", {})

    if "view_as" not in registered:

        @register_torch_op
        def view_as(context, node):  # type: ignore[no-untyped-def]
            x, ref = _get_inputs(context, node, expected=2)
            context.add(mb.reshape(x=x, shape=mb.shape(x=ref), name=node.name))

    if "broadcast_tensors" not in registered:

        @register_torch_op
        def broadcast_tensors(context, node):  # type: ignore[no-untyped-def]
            tensors = _get_inputs(context, node, expected=1)[0]
            ins = list(tensors) if isinstance(tensors, (list, tuple)) else [tensors]
            target = np.broadcast_shapes(*[tuple(int(d) for d in t.shape) for t in ins])
            outs = [
                mb.broadcast_to(x=t, shape=list(target), name=f"{node.name}_{i}")
                for i, t in enumerate(ins)
            ]
            context.add(outs, torch_name=node.outputs[0])


def _convert_and_gate(  # type: ignore[no-untyped-def]
    module: nn.Module,
    example: tuple,
    input_names: list[str],
    int_inputs: bool,
    out_path: Path,
    *,
    fp16: bool,
    compute_units: str,
    reference: torch.Tensor,
    corr_gate: float,
):
    import coremltools as ct

    _install_ct_shims()
    traced = torch.jit.trace(module, example, strict=False)
    if int_inputs:
        inputs = [
            ct.TensorType(name=n, shape=tuple(t.shape), dtype=int)
            for n, t in zip(input_names, example, strict=True)
        ]
    else:
        inputs = [
            ct.TensorType(name=n, shape=tuple(t.shape))
            for n, t in zip(input_names, example, strict=True)
        ]
    mlm = ct.convert(
        traced,
        inputs=inputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
        compute_units=getattr(ct.ComputeUnit, compute_units),
    )
    tmp = out_path.with_name(out_path.stem + ".tmp.mlpackage")
    shutil.rmtree(tmp, ignore_errors=True)
    mlm.save(str(tmp))

    feed = {
        n: (t.numpy().astype(np.int32) if int_inputs else t.numpy().astype(np.float32))
        for n, t in zip(input_names, example, strict=True)
    }
    got = torch.from_numpy(np.asarray(next(iter(mlm.predict(feed).values())), dtype=np.float32))
    ref = reference.flatten().double()
    corr = torch.corrcoef(torch.stack([ref, got.flatten().double()]))[0, 1].item()
    max_err = (got.flatten().double() - ref).abs().max().item()
    ok = corr >= corr_gate
    print(
        f"  {out_path.name}: corr {corr:.7f} (gate {corr_gate}), max|err| {max_err:.3e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"{out_path.name}: parity gate failed (corr {corr:.7f} < {corr_gate})")
    shutil.rmtree(out_path, ignore_errors=True)
    tmp.rename(out_path)
    return corr, max_err


def main() -> None:  # noqa: PLR0915 — one export per stage, linear and logged
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None, help="output dir (default: <ckpt dir>/coreml)")
    ap.add_argument("--stages", default="encoder,estimator,vocoder")
    args = ap.parse_args()

    ckpt = Checkpoint.open(args.checkpoint)
    out_dir = Path(args.out) if args.out else ckpt.path.parent / "coreml"
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages.split(","))

    algo = production_algorithm(ckpt)
    w = algo.window
    assert w.static_length is not None, "static window recipe missing (static_length)"
    assert w.static_prompt_tokens is not None, (
        "static window recipe missing (static_prompt_tokens) — these graphs are static-shape"
    )
    t_mel = 2 * (w.static_prompt_tokens + w.static_length)
    hift_frames = 2 * w.max_speech_tokens
    print(
        f"checkpoint {ckpt.path.name}: window {w.static_prompt_tokens}+{w.static_length} "
        f"-> estimator T{t_mel}, vocoder {hift_frames} frames"
    )
    print(f"algorithm: {algo.describe()}")

    torch.manual_seed(0)

    # -- torch reference modules, straight from the packed weights ----------
    decoder = TorchMelDecoder(algo)
    decoder.load_state_dict(
        {k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors("s3gen.flow.").items()}
    )
    decoder = decoder.float().eval()  # trace in fp32; ct re-quantizes the estimator

    if "encoder" in stages:
        print("[encoder]")
        enc = _EncoderWrapper(decoder).eval()
        prompt = torch.randint(
            0, 6561, (1, w.static_prompt_tokens), generator=torch.Generator().manual_seed(3)
        )
        query = torch.randint(
            0, 6561, (1, w.static_length), generator=torch.Generator().manual_seed(4)
        )
        with torch.no_grad():
            mu_ref = enc(prompt, query)
        assert mu_ref.shape == (1, _MEL_BINS, t_mel), mu_ref.shape
        _convert_and_gate(
            enc,
            (prompt, query),
            ["prompt_token", "speech_tokens"],
            True,
            out_dir / ENCODER_NAME,
            fp16=False,
            compute_units="CPU_ONLY",
            reference=mu_ref,
            corr_gate=0.999999,
        )

    if "estimator" in stages:
        print("[estimator]")
        est = _EstimatorWrapper(decoder).eval()
        gen = torch.Generator().manual_seed(5)
        ex = (
            torch.randn(1, _MEL_BINS, t_mel, generator=gen),
            torch.randn(1, _MEL_BINS, t_mel, generator=gen),
            torch.tensor([0.4]),
            torch.randn(1, _MEL_BINS, generator=gen),
            torch.randn(1, _MEL_BINS, t_mel, generator=gen),
        )
        with torch.no_grad():
            v_ref = est(*ex)
        # fp16 pipeline: the gate is the EXP-011 band, not fp32 equality
        _convert_and_gate(
            est,
            ex,
            ["x", "mu", "t", "spks", "cond"],
            False,
            out_dir / ESTIMATOR_NAME,
            fp16=True,
            compute_units="CPU_AND_NE",
            reference=v_ref,
            corr_gate=0.999,
        )

    if "vocoder" in stages:
        print("[vocoder]")
        voc = TorchVocoder(algo)
        voc.load_state_dict(
            {k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors("s3gen.mel2wav.").items()}
        )
        voc = voc.eval()

        # probe mel: the committed reference render, padded to the static frame
        # count — real speech statistics, not random noise (measurement rule 3)
        ref_mel_path = (
            Path(__file__).resolve().parent.parent / "tests/data/reference/s0_mel.npy"
        )
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
        phase[0, 1:, 0] = symmetric_uniforms(
            seed, VOCODER_PHASE_STREAM, _N_HARMONICS - 1, np.pi
        )
        noise = gaussian_field(seed, VOCODER_NOISE_STREAM, _N_HARMONICS, n_samples)[None]
        phase_t, noise_t = torch.from_numpy(phase), torch.from_numpy(noise)

        wrapper = _VocoderWrapper(voc, hift_frames).eval()
        with torch.no_grad():
            wav_conv = wrapper(mel_t, phase_t, noise_t)
            wav_ref = torch.from_numpy(voc.synthesize(mel_np[:, :n_real], None, seed=seed))
        # torch-level gate: the conv STFT/iSTFT rewrite must be equivalent
        # before conversion enters the picture
        d = (wav_conv[0, : n_real * _HOP_SAMPLES] - wav_ref).abs().max().item()
        print(f"  conv-STFT rewrite vs torch.stft/istft: max|err| {d:.3e}")
        assert d < 1e-4, "conv STFT/iSTFT rewrite is not equivalent to the torch vocoder"

        _convert_and_gate(
            wrapper,
            (mel_t, phase_t, noise_t),
            ["mel", "phase", "noise"],
            False,
            out_dir / VOCODER_NAME,
            fp16=False,
            compute_units="CPU_ONLY",
            reference=wav_conv,
            corr_gate=0.999999,
        )

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
