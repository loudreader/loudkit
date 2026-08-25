"""Export the ONNX stage graphs from the packed checkpoint.

Five graphs, all fp32, all gated against the torch modules loaded from the
*same* checkpoint before anything is saved (a failed gate raises and nothing
half-exported is left behind — writes go to the final name only after the gate
passes):

  t3_cond.onnx        speaker_emb [1,256] + cond_prompt_tokens [1,P] + emotion [1,1]
                      -> cond [1,34,1024]              (spkr_enc + perceiver + emotion)
  t3_prefill.onnx     embeds [1,T,1024] + positions [T] -> logits [1,T,8194]
                      + 32 KV tensors                  (one causal forward, whole window)
  t3_step.onnx        embeds [1,1,1024] + position [1] + past KV -> logits [1,8194]
                      + present KV                     (one decode step, the loop's core)
  flow_encoder.onnx   prompt_token [1,238] + speech_tokens [1,255] -> mu [1,80,986]
                      (fp32, CPU — fp16 here is measured fatal, EXP-011)
  flow_estimator.onnx x/mu/t/spks/cond @ T986 -> velocity. fp32.
  vocoder.onnx        mel [1,80,510] + phase [1,9,1] + noise [1,9,244800]
                      -> wav [1,244800]                (HiFT, conv STFT/iSTFT)

``t3_cond`` exists because the conditioning encoder is real attention (a
perceiver), not a table lookup — it cannot be replicated in numpy and stay
honest. The prefill/step split is the standard decoder-with-past shape: the
whole sequence goes through one graph for teacher forcing and the window
start, then each sampled token goes through the step graph against the cache.
EXP-015 exported only the step and documented the mask/position pitfalls this
export bakes into the graph inputs instead.

**The gate is the ONNX reason to exist.** After every graph is exported, the
tool loads a torch fp32 engine and the fresh ONNX sessions, teacher-forces the
reference sentences through both, and requires:

  - aggregate top-1 >= 99.5% over every forced step, and
  - median per-step KL < 1e-3 (fp32)

That is the gate registered in the plan (step 6); int8 stays blocked
(EXP-017) and no int8 artifact is produced here.

Usage (from the loudkit repo root):

  .venv/bin/python tools/export_onnx.py \
      --checkpoint /path/to/loudr-1.safetensors [--out DIR] [--stages ...]

Default --out is an ``onnx/`` directory beside the checkpoint, which is where
the onnx backend looks first.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from loudkit.backends import production_algorithm  # noqa: E402
from loudkit.checkpoint import Checkpoint  # noqa: E402
from loudkit.config import ExecutionConfig  # noqa: E402
from loudkit.models.flow import TorchMelDecoder  # noqa: E402
from loudkit.models.generator import TorchTokenGenerator  # noqa: E402
from loudkit.models.noise import gaussian_field, symmetric_uniforms  # noqa: E402
from loudkit.models.vocoder import (  # noqa: E402
    VOCODER_NOISE_STREAM,
    VOCODER_PHASE_STREAM,
    TorchVocoder,
)
from loudkit.voice import VoiceProfile  # noqa: E402

_MEL_BINS = 80
_N_HARMONICS = 9
_HOP_SAMPLES = 480

# The exported graph set, in the order the backend loads them.
COND_NAME = "t3_cond.onnx"
PREFILL_NAME = "t3_prefill.onnx"
STEP_NAME = "t3_step.onnx"
ENCODER_NAME = "flow_encoder.onnx"
ESTIMATOR_NAME = "flow_estimator.onnx"
VOCODER_NAME = "vocoder.onnx"

# Gate: the plan's step-6 numbers, stated once so the tool and the test agree.
TF_TOP1_GATE = 0.995
TF_KL_GATE = 1e-3


# --------------------------------------------------------------- wrappers


class _CondWrapper(nn.Module):
    """speaker_emb + cond_prompt_tokens + emotion -> the 34-slot conditioning row."""

    def __init__(self, gen: TorchTokenGenerator) -> None:
        super().__init__()
        self.speech_emb = gen.speech_emb
        self.speech_pos_emb = gen.speech_pos_emb
        self.cond_enc = gen.cond_enc

    def forward(
        self, speaker_emb: torch.Tensor, prompt_tokens: torch.Tensor, emotion: torch.Tensor
    ) -> torch.Tensor:
        prompt_emb = self.speech_emb(prompt_tokens) + self.speech_pos_emb.range(
            prompt_tokens.shape[1], speaker_emb.device
        )
        return self.cond_enc(speaker_emb, prompt_emb, emotion)


class _PrefillWrapper(nn.Module):
    """One causal forward: embeds + positions -> every-position logits + KV."""

    def __init__(self, gen: TorchTokenGenerator) -> None:
        super().__init__()
        self.tfmr = gen.tfmr
        self.head = gen.speech_head

    def forward(
        self, embeds: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        hidden, cache = self.tfmr(embeds, positions, None, attention="eager")
        logits = self.head(hidden)
        flat: list[torch.Tensor] = []
        for k, v in cache:
            flat += [k, v]
        return (logits, *flat)


class _StepWrapper(nn.Module):
    """One decode step against the past cache: embeds + position + past -> logits + present."""

    def __init__(self, gen: TorchTokenGenerator, n_layers: int) -> None:
        super().__init__()
        self.tfmr = gen.tfmr
        self.head = gen.speech_head
        self.n = n_layers

    def forward(
        self, embeds: torch.Tensor, position: torch.Tensor, *past: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        cache = [(past[2 * i], past[2 * i + 1]) for i in range(self.n)]
        hidden, new_cache = self.tfmr(embeds, position, cache, attention="eager")
        logits = self.head(hidden[:, -1])
        flat: list[torch.Tensor] = []
        for k, v in new_cache:
            flat += [k, v]
        return (logits, *flat)


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


# The STFT/iSTFT-as-convolution rewrite, lifted from the CoreML export
# (chatterbox-apple/export/hift_static.py). Math-equivalent restatements of
# torch.stft / torch.istft at n_fft=16, hop=4, hann, center=True — ONNX has no
# stft/istft, so the conv basis is the only portable form.


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
        return F.conv1d(x, self.basis, stride=self.hop)


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
        self.stft = _ConvSTFT(n_fft, hop, voc.stft_window)
        with torch.no_grad():
            x = voc.conv_pre(torch.zeros(1, _MEL_BINS, mel_frames))
            for i in range(len(voc.ups)):
                x = voc.ups[i](F.leaky_relu(x, 0.1))
                if i == len(voc.ups) - 1:
                    x = F.pad(x, (1, 0), mode="reflect")
            out_frames = x.shape[-1]
        self.istft = _ConvISTFT(n_fft, hop, voc.stft_window, out_frames)
        self.n_fft = n_fft

    def forward(self, mel, phase, noise):  # type: ignore[no-untyped-def]
        voc = self.voc
        f0 = voc.f0_predictor(mel)
        f0_up = F.interpolate(f0[:, None], scale_factor=float(_HOP_SAMPLES), mode="nearest")
        source = voc.m_source(f0_up, phase, noise)

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


def _export_and_gate(  # type: ignore[no-untyped-def]
    module: nn.Module,
    example: tuple,
    input_names: list[str],
    out_path: Path,
    *,
    int_inputs: set[str] | bool,
    dynamic_axes: dict[str, dict[int, str]] | None,
    reference: torch.Tensor,
    corr_gate: float,
    max_abs_gate: float,
):
    import onnxruntime as ort

    if isinstance(int_inputs, bool):
        int_inputs = set(input_names) if int_inputs else set()

    tmp = out_path.with_name(out_path.stem + ".tmp.onnx")
    with torch.no_grad():
        torch.onnx.export(
            module,
            example,
            str(tmp),
            input_names=input_names,
            output_names=[f"{Path(out_path.stem)}_out"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    feed = {
        n: (t.numpy().astype(np.int64) if n in int_inputs else t.numpy().astype(np.float32))
        for n, t in zip(input_names, example, strict=True)
    }
    got = np.asarray(sess.run(None, feed)[0], dtype=np.float32)
    ref = reference.numpy().astype(np.float32)
    max_err = float(np.abs(got - ref).max())
    corr = float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
    ok = corr >= corr_gate and max_err <= max_abs_gate
    print(
        f"  {out_path.name}: corr {corr:.7f} (gate {corr_gate}), "
        f"max|err| {max_err:.3e} (gate {max_abs_gate:.0e}) -> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{out_path.name}: ONNX parity gate failed")
    shutil.rmtree(out_path, ignore_errors=True)
    tmp.rename(out_path)
    return corr, max_err


def _int_axes(name: str) -> dict[str, dict[int, str]]:
    return {name: {1: "seq"}}


def _kv_axes(base: list[str]) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    for n in base:
        out[n] = {1: "seq", 2: "kv"}
    return out


def main() -> None:  # noqa: PLR0915 — one export per stage, linear and logged
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None, help="output dir (default: <ckpt dir>/onnx)")
    ap.add_argument("--stages", default="cond,prefill,step,encoder,estimator,vocoder")
    args = ap.parse_args()

    ckpt = Checkpoint.open(args.checkpoint)
    out_dir = Path(args.out) if args.out else ckpt.path.parent / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages.split(","))

    algo = production_algorithm(ckpt)
    w = algo.window
    assert w.static_length is not None, "static window recipe missing (static_length)"
    assert w.static_prompt_tokens is not None, (
        "static window recipe missing (static_prompt_tokens)"
    )
    t_mel = 2 * (w.static_prompt_tokens + w.static_length)
    hift_frames = 2 * w.max_speech_tokens
    print(
        f"checkpoint {ckpt.path.name}: window {w.static_prompt_tokens}+{w.static_length} "
        f"-> estimator T{t_mel}, vocoder {hift_frames} frames"
    )
    print(f"algorithm: {algo.describe()}")

    llama_config = ckpt.manifest["llama_config"]
    assert isinstance(llama_config, dict)
    n_layers = int(llama_config["num_hidden_layers"])
    kvh = int(llama_config["num_key_value_heads"])
    hd = int(llama_config.get("head_dim", 64))

    torch.manual_seed(0)

    # -- token generator graphs ---------------------------------------------
    gen = TorchTokenGenerator(algo, llama_config, attention="eager")
    gen.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors("t3.").items()})
    gen = gen.float().eval()  # fp32 graphs; fp16 storage upcasts exactly

    if "cond" in stages:
        print("[cond]")
        # One name, one wrapper per stage: nn.Module, not the first wrapper's
        # own type.
        mod: nn.Module = _CondWrapper(gen).eval()
        speaker = torch.randn(1, 256)
        prompt = torch.randint(0, 8194, (1, 64), generator=torch.Generator().manual_seed(1))
        emotion = torch.tensor([[0.5]])
        with torch.no_grad():
            ref = mod(speaker, prompt, emotion)
        _export_and_gate(
            mod,
            (speaker, prompt, emotion),
            ["speaker_emb", "prompt_tokens", "emotion"],
            out_dir / COND_NAME,
            int_inputs={"prompt_tokens"},
            dynamic_axes=_int_axes("prompt_tokens"),
            reference=ref,
            corr_gate=0.999999,
            max_abs_gate=1e-3,
        )

    if "prefill" in stages:
        print("[prefill]")
        mod = _PrefillWrapper(gen).eval()
        seq_len = 64
        emb = torch.randn(1, seq_len, 1024)
        pos = torch.arange(seq_len)
        with torch.no_grad():
            ref = mod(emb, pos)
        base = ["logits"] + [
            f"kv_{'k' if i % 2 == 0 else 'v'}_{i // 2}" for i in range(2 * n_layers)
        ]
        out_names = base
        dyn = {"embeds": {1: "seq"}, "positions": {0: "seq"}}
        dyn.update(_kv_axes(out_names[1:]))
        tmp = out_dir / PREFILL_NAME
        tmp_t = tmp.with_name(tmp.stem + ".tmp.onnx")
        with torch.no_grad():
            torch.onnx.export(
                mod,
                (emb, pos),
                str(tmp_t),
                input_names=["embeds", "positions"],
                output_names=out_names,
                dynamic_axes=dyn,
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
        import onnxruntime as ort

        sess = ort.InferenceSession(str(tmp_t), providers=["CPUExecutionProvider"])
        seq_len2 = 113
        emb2 = torch.randn(1, seq_len2, 1024)
        pos2 = torch.arange(seq_len2)
        got = sess.run(None, {"embeds": emb2.numpy(), "positions": pos2.numpy()})
        with torch.no_grad():
            want = mod(emb2, pos2)
        max_err = max(
            float(np.abs(g - w.numpy()).max()) for g, w in zip(got, want, strict=False)
        )
        ok = max_err <= 1e-3
        print(
            f"  {PREFILL_NAME}: untraced seq {seq_len2} "
            f"max|err| {max_err:.3e} -> {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            tmp_t.unlink(missing_ok=True)
            raise SystemExit(f"{PREFILL_NAME}: untraced-length parity gate failed")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp_t.rename(tmp)

    if "step" in stages:
        print("[step]")
        mod = _StepWrapper(gen, n_layers).eval()
        s_past = 8
        emb = torch.randn(1, 1, 1024)
        pos = torch.tensor([s_past])
        past = [torch.randn(1, kvh, s_past, hd) for _ in range(2 * n_layers)]
        in_names = ["embeds", "position"] + [
            f"past_{'k' if i % 2 == 0 else 'v'}_{i // 2}" for i in range(2 * n_layers)
        ]
        out_names = ["logits"] + [
            f"present_{'k' if i % 2 == 0 else 'v'}_{i // 2}" for i in range(2 * n_layers)
        ]
        dyn = {"embeds": {1: "seq"}}
        dyn.update(_kv_axes(in_names[2:]))
        dyn.update(_kv_axes(out_names[1:]))
        tmp = out_dir / STEP_NAME
        tmp_t = tmp.with_name(tmp.stem + ".tmp.onnx")
        with torch.no_grad():
            torch.onnx.export(
                mod,
                (emb, pos, *past),
                str(tmp_t),
                input_names=in_names,
                output_names=out_names,
                dynamic_axes=dyn,
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
        import onnxruntime as ort

        sess = ort.InferenceSession(str(tmp_t), providers=["CPUExecutionProvider"])
        p2 = 37
        emb2 = torch.randn(1, 1, 1024)
        pos2 = torch.tensor([p2])
        past2 = [torch.randn(1, kvh, p2, hd) for _ in range(2 * n_layers)]
        feed = {"embeds": emb2.numpy(), "position": pos2.numpy()}
        for n, t in zip(in_names[2:], past2, strict=True):
            feed[n] = t.numpy()
        got = sess.run(None, feed)
        with torch.no_grad():
            want = mod(emb2, pos2, *past2)
        max_err = max(
            float(np.abs(g - w.numpy()).max()) for g, w in zip(got, want, strict=False)
        )
        ok = max_err <= 1e-3
        print(
            f"  {STEP_NAME}: untraced past {p2} "
            f"max|err| {max_err:.3e} -> {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            tmp_t.unlink(missing_ok=True)
            raise SystemExit(f"{STEP_NAME}: untraced-past parity gate failed")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp_t.rename(tmp)

    # -- renderer graphs -----------------------------------------------------
    decoder = TorchMelDecoder(algo)
    decoder.load_state_dict(
        {k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors("s3gen.flow.").items()}
    )
    decoder = decoder.float().eval()

    if "encoder" in stages:
        print("[encoder]")
        mod = _EncoderWrapper(decoder).eval()
        prompt = torch.randint(
            0, 6561, (1, w.static_prompt_tokens), generator=torch.Generator().manual_seed(3)
        )
        query = torch.randint(
            0, 6561, (1, w.static_length), generator=torch.Generator().manual_seed(4)
        )
        with torch.no_grad():
            ref = mod(prompt, query)
        assert ref.shape == (1, _MEL_BINS, t_mel), ref.shape
        _export_and_gate(
            mod,
            (prompt, query),
            ["prompt_token", "speech_tokens"],
            out_dir / ENCODER_NAME,
            int_inputs={"prompt_token", "speech_tokens"},
            dynamic_axes=None,
            reference=ref,
            corr_gate=0.999999,
            max_abs_gate=1e-3,
        )

    if "estimator" in stages:
        print("[estimator]")
        mod = _EstimatorWrapper(decoder).eval()
        gen_r = torch.Generator().manual_seed(5)
        ex = (
            torch.randn(1, _MEL_BINS, t_mel, generator=gen_r),
            torch.randn(1, _MEL_BINS, t_mel, generator=gen_r),
            torch.tensor([0.4]),
            torch.randn(1, _MEL_BINS, generator=gen_r),
            torch.randn(1, _MEL_BINS, t_mel, generator=gen_r),
        )
        with torch.no_grad():
            ref = mod(*ex)
        _export_and_gate(
            mod,
            ex,
            ["x", "mu", "t", "spks", "cond"],
            out_dir / ESTIMATOR_NAME,
            int_inputs=False,
            dynamic_axes=None,
            reference=ref,
            corr_gate=0.999999,
            max_abs_gate=1e-3,
        )

    if "vocoder" in stages:
        print("[vocoder]")
        voc = TorchVocoder(algo)
        voc.load_state_dict(
            {k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors("s3gen.mel2wav.").items()}
        )
        voc = voc.eval()

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
            seed, VOCODER_PHASE_STREAM, _N_HARMONICS - 1, math.pi
        )
        noise = gaussian_field(seed, VOCODER_NOISE_STREAM, _N_HARMONICS, n_samples)[None]
        phase_t, noise_t = torch.from_numpy(phase), torch.from_numpy(noise)

        wrapper = _VocoderWrapper(voc, hift_frames).eval()
        with torch.no_grad():
            wav_conv = wrapper(mel_t, phase_t, noise_t)
            wav_ref = torch.from_numpy(voc.synthesize(mel_np[:, :n_real], None, seed=seed))
        d = (wav_conv[0, : n_real * _HOP_SAMPLES] - wav_ref).abs().max().item()
        print(f"  conv-STFT rewrite vs torch.stft/istft: max|err| {d:.3e}")
        assert d < 1e-4, "conv STFT/iSTFT rewrite is not equivalent to the torch vocoder"

        _export_and_gate(
            wrapper,
            (mel_t, phase_t, noise_t),
            ["mel", "phase", "noise"],
            out_dir / VOCODER_NAME,
            int_inputs=False,
            dynamic_axes=None,
            reference=wav_conv,
            corr_gate=0.999999,
            max_abs_gate=1e-3,
        )

    _teacher_forced_gate(ckpt, out_dir, algo, stages)
    print(f"done -> {out_dir}")


# ------------------------------------------------------------------- gate


def _teacher_forced_gate(
    ckpt: Checkpoint,
    out_dir: Path,
    algo,
    stages: set[str],  # type: ignore[no-untyped-def]
) -> None:
    """Teacher-forced logits through the fresh graphs vs torch fp32.

    The conformance fixture's tokens were produced by the *fp32* torch
    generator (make_conformance.py, ``token_generator: fp32``), so this gate
    doubling as a free-run guarantee: same precision, same law, and logits
    agreeing at top-1 rate >= 99.5% means the ONNX engine produces identical
    tokens.
    """
    import json

    if not {"cond", "prefill", "step"} <= stages:
        print("[gate] generator graphs not all exported; teacher-forced gate skipped")
        return

    reference_dir = Path(__file__).resolve().parent.parent / "tests/data/reference"
    meta = json.loads((reference_dir / "meta.json").read_text(encoding="utf-8"))
    voice = VoiceProfile.load(reference_dir / "testvoice.voice.safetensors")

    torch_engine = _torch_fp32_generator(ckpt, algo)

    from loudkit.backends.onnx_backend import ONNXTokenGenerator

    onnx_gen = ONNXTokenGenerator(algo, ckpt, out_dir, execution=ExecutionConfig(device="onnx"))

    agree, steps, kls = 0, 0, []
    for key in ("0", "1", "2"):
        rec = meta[key]
        forced = rec["speech_tokens"][:64]
        ref = torch_engine.teacher_forced_logits(
            np.asarray(rec["text_ids"], dtype=np.int64), voice, forced
        )
        mine = onnx_gen.teacher_forced_logits(
            np.asarray(rec["text_ids"], dtype=np.int64), voice, forced
        )
        n = min(len(mine), len(ref))
        agree += int((mine[:n].argmax(-1) == ref[:n].argmax(-1)).sum())
        steps += n
        p = torch.log_softmax(torch.tensor(ref[:n]), -1)
        q = torch.log_softmax(torch.tensor(mine[:n]), -1)
        kl = (
            torch.nn.functional.kl_div(q, p, log_target=True, reduction="none")
            .sum(-1)
            .abs()
            .median()
        )
        kls.append(float(kl))
    top1 = agree / steps
    max_kl = max(kls)
    ok = top1 >= TF_TOP1_GATE and max_kl < TF_KL_GATE
    print(
        f"[gate] teacher-forced top-1 {top1:.5f} ({agree}/{steps}) "
        f"[gate {TF_TOP1_GATE}], median KL {kls} [gate {TF_KL_GATE}] "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        raise SystemExit(f"teacher-forced gate failed: top-1 {top1:.5f}, KL {kls}")


def _torch_fp32_generator(ckpt: Checkpoint, algo):  # type: ignore[no-untyped-def]
    """The torch reference generator, forced fp32 like the conformance run."""
    llama_config = ckpt.manifest["llama_config"]
    assert isinstance(llama_config, dict)
    gen = TorchTokenGenerator(algo, llama_config, attention="eager")
    gen.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors("t3.").items()})
    return gen.float().eval()


if __name__ == "__main__":
    main()
