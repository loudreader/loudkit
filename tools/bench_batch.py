#!/usr/bin/env python3
"""Benchmark batch scaling of the token generator on CUDA.

The generator is launch-latency-bound, not compute-bound: at batch 1 a CUDA
decode step is a few hundred tiny serial dispatches and the GPU idles most of
the time. Batching N utterances in lockstep amortises that overhead, so
aggregate throughput grows with N. This tool measures the effect on the real
model and records a reproducible row.

What it measures: **the token generator only** — a fixed 255-token window
decoded in lockstep for batch N, with mel+vocoder excluded. That is aggregate
throughput, not single-utterance latency; the two must not be compared (see
README "Aggregate throughput at batch N").

Usage:
  python tools/bench_batch.py <checkpoint> <voice> <device> <outdir> <batches>
  # batches: comma list, default 1,2,4,8,16,32,64
  # device: cuda or cuda:1 (indexed CUDA for multi-GPU)

The row records the exact command so any number here can be reproduced.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import torch

import loudkit
from loudkit.config import ExecutionOverrides
from loudkit.models.generator import TorchTokenGenerator

DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64)


def _sync(device: str) -> None:
    """Synchronise the CUDA device before/after timing. CPU is already
    synchronous; torch.cuda would fail on a CPU-only build."""
    if device.split(":", maxsplit=1)[0] == "cuda":
        torch.cuda.synchronize()


def _batch_list(value: str) -> list[int]:
    try:
        return [int(x) for x in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"batches must be a comma list of integers: {exc}"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("voice")
    ap.add_argument("device", help="cuda or cuda:<index> (indexed CUDA for multi-GPU)")
    ap.add_argument("outdir", type=Path)
    ap.add_argument(
        "batches",
        nargs="?",
        type=_batch_list,
        # A string default goes through `type` like a command-line value, so
        # the default and an explicit argument take exactly the same path.
        default=",".join(map(str, DEFAULT_BATCHES)),
        help="comma list of batch sizes (default: %(default)s)",
    )
    return ap


def main() -> int:  # noqa: PLR0915 — a benchmark tool is one long straight-line run
    args = _parser().parse_args()
    ckpt, voice_path, device = args.checkpoint, args.voice, args.device
    batches: list[int] = args.batches
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    # CUDA graphs need sm_70+ (Volta); check the *targeted* device's capability,
    # not device 0 — cuda:1 may be an older card (e.g. a Pascal 1080 Ti). For a
    # CPU device there is no CUDA at all, so graphs are never used.
    graphs = False
    if device.split(":")[0] == "cuda":
        device_index = int(device.split(":")[1]) if ":" in device else 0
        graphs = torch.cuda.get_device_capability(device_index)[0] >= 7

    e = loudkit.load(
        ckpt,
        device=device,
        # str, not Device: the registry accepts "cuda:1", which the Literal
        # does not cover; loudkit.load vets it.
        execution=ExecutionOverrides(device=device, cuda_graphs=graphs),  # type: ignore[arg-type]
    )
    # The decode loop below is hand-written against the torch generator's own
    # modules; an ONNX or CoreML engine satisfies the TokenGenerator protocol
    # and has none of them.
    if not isinstance(e.token_generator, TorchTokenGenerator):
        raise SystemExit(f"{device}: this benchmark needs the torch generator")
    # Any, not TorchTokenGenerator: the step below reads submodules that
    # nn.Module's __getattr__ types as `Tensor | Module`.
    gen: Any = e.token_generator
    voice = loudkit.VoiceProfile.load(voice_path)
    from loudkit.frontend.polish import speech_text

    text = (
        "The quick brown fox jumps over the lazy dog and the reader keeps its composure. "
    ) * 14
    tt = e.frontend.encode(speech_text(text, "en"), "en")
    r = e.synthesize(text, voice, seed=7)
    forced = list(r.tokens)
    cap = len(forced)
    audio_s = r.duration

    with torch.inference_mode():
        # Public surface, not internals: `prefill_embeds` and
        # `decode_geometry` exist so a benchmark can size a cache without
        # reading `gen.tfmr.layers[0].self_attn.n_kv_heads`, which broke the
        # first time the decoder moved and explained nothing when it did.
        geom = gen.decode_geometry()
        embeds = gen.prefill_embeds(tt, voice)
        prefill_len = embeds.shape[1]
        hidden, cache = gen.tfmr(
            embeds, torch.arange(prefill_len, device=geom.device), None, attention=gen.attention
        )
        n_layers, n_kv, hd = geom.n_layers, geom.n_kv_heads, geom.head_dim
        max_len = prefill_len + cap + 2
        dev, dtype = geom.device, geom.dtype

        def run_batch(batch: int) -> tuple[float, float]:  # noqa: PLR0915
            k_bufs = torch.zeros(n_layers, batch, n_kv, max_len, hd, device=dev, dtype=dtype)
            v_bufs = torch.zeros_like(k_bufs)
            for i, (k, vv) in enumerate(cache):
                k_bufs[i, :, :, :prefill_len, :] = k[0].expand(batch, -1, -1, -1)
                v_bufs[i, :, :, :prefill_len, :] = vv[0].expand(batch, -1, -1, -1)
            grid_b = torch.arange(batch, device=dev).repeat_interleave(n_kv * hd)
            grid_k = torch.arange(n_kv, device=dev).repeat_interleave(hd).repeat(batch)
            grid_h = torch.arange(hd, device=dev).repeat(n_kv).repeat(batch)
            token_buf = torch.zeros(batch, 1, dtype=torch.long, device=dev)
            emb_pos = torch.zeros(batch, 1, dtype=torch.long, device=dev)
            rope_pos = torch.zeros(batch, dtype=torch.long, device=dev)
            logits_buf = torch.zeros(batch, gen.SPEECH_VOCAB, dtype=torch.float32, device=dev)

            def step():
                emb = gen.speech_emb(token_buf) + gen.speech_pos_emb.emb(emb_pos)
                cos1, sin1 = gen.tfmr._rope(rope_pos[:1], emb.dtype)
                cos_b = cos1.expand(batch, -1, -1, -1)
                sin_b = sin1.expand(batch, -1, -1, -1)
                x = emb.to(dtype)
                for i, layer in enumerate(gen.tfmr.layers):
                    attn = layer.self_attn
                    h = layer.input_layernorm(x)
                    b_, t_, _ = h.shape
                    q = attn.q_proj(h).view(b_, t_, attn.n_heads, attn.head_dim).transpose(1, 2)
                    k_ = (
                        attn.k_proj(h)
                        .view(b_, t_, attn.n_kv_heads, attn.head_dim)
                        .transpose(1, 2)
                    )
                    v_ = (
                        attn.v_proj(h)
                        .view(b_, t_, attn.n_kv_heads, attn.head_dim)
                        .transpose(1, 2)
                    )

                    def rot(xx):
                        return (
                            xx * cos_b
                            + torch.cat((-xx[..., hd // 2 :], xx[..., : hd // 2]), dim=-1)
                            * sin_b
                        )

                    q = rot(q)
                    k_ = rot(k_)
                    flat_k = k_[:, :, 0, :].reshape(batch, -1)
                    flat_v = v_[:, :, 0, :].reshape(batch, -1)
                    pos_row = rope_pos.repeat_interleave(n_kv * hd)
                    k_bufs[i].index_put_((grid_b, grid_k, pos_row, grid_h), flat_k.reshape(-1))
                    v_bufs[i].index_put_((grid_b, grid_k, pos_row, grid_h), flat_v.reshape(-1))
                    rep = attn.n_heads // attn.n_kv_heads
                    kk = k_bufs[i].repeat_interleave(rep, dim=1)
                    vv = v_bufs[i].repeat_interleave(rep, dim=1)
                    scores = q @ kk.transpose(-2, -1) / (attn.head_dim**0.5)
                    pad = torch.arange(kk.shape[2], device=dev) > rope_pos[:, None]
                    mask = torch.where(
                        pad.unsqueeze(1).unsqueeze(1),
                        torch.full((), float("-inf"), device=dev, dtype=scores.dtype),
                        torch.zeros((), device=dev, dtype=scores.dtype),
                    )
                    o = (
                        torch.softmax(scores + mask, dim=-1, dtype=torch.float32).to(q.dtype)
                        @ vv
                    )
                    o = o.transpose(1, 2).reshape(b_, t_, -1)
                    x = x + attn.o_proj(o)
                    x = x + layer.mlp(layer.post_attention_layernorm(x))
                logits_buf.copy_(gen.speech_head(gen.tfmr.norm(x)[:, -1]).float())

            token_buf.fill_(forced[0])
            emb_pos.fill_(1)
            rope_pos.fill_(prefill_len)
            step()
            _sync(device)

            if graphs:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    step()

                def run() -> None:
                    g.replay()

            else:
                run = step

            t0 = time.perf_counter()
            for s in range(cap):
                token_buf.fill_(forced[s])
                emb_pos.fill_(s + 1)
                rope_pos.fill_(prefill_len + s)
                run()
            _sync(device)
            wall = time.perf_counter() - t0
            step_ms = wall / cap * 1000
            rtf = batch * audio_s / wall
            return step_ms, rtf

        rows = []
        for batch in batches:
            step_ms, rtf = run_batch(batch)
            rows.append(
                {"batch": batch, "ms_per_step": round(step_ms, 3), "rtf": round(rtf, 2)}
            )
            print(f"  batch {batch:3d}  {step_ms:8.3f} ms/step  RTF {rtf:8.2f}x", flush=True)

        cmd = (
            f"python tools/bench_batch.py {ckpt} {voice_path} {device} {out} "
            f"{','.join(map(str, batches))}"
        )
        row = {
            "tool": "tools/bench_batch.py",
            "command": cmd,
            "device": device,
            "cuda_graphs": graphs,
            "audio_s": audio_s,
            "tokens_per_utterance": cap,
            "precision": str(dtype),
            "checkpoint": str(ckpt),
            "voice": str(voice_path),
            "rows": rows,
        }
        (out / f"batch_{device.replace(':', '_')}.json").write_text(
            json.dumps(row, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"saved -> {out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
