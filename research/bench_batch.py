#!/usr/bin/env python3
"""Benchmark batch scaling of the token generator, in the loop the checkpoint declares.

The generator is launch-latency-bound, not compute-bound: at batch 1 a CUDA
decode step is a few hundred tiny serial dispatches and the GPU idles most of
the time. Batching N utterances in lockstep amortises that overhead, so
aggregate throughput grows with N. This tool measures the effect on the real
model and records a reproducible row.

What it measures: **the token generator only** -- a fixed 255-token window
decoded in lockstep for batch N, with mel+vocoder excluded. That is aggregate
throughput, not single-utterance latency, so both are reported and never in one
column: `rtf` is the batch's, `rtf_per_request` is one listener's.

The loop follows `AlgorithmConfig.decode`. `single` steps one token at a time.
`fusion_mtp2` fuses the pair into ONE slot (`0.5*(e_a+e_b) + fuse([e_a;e_b])`),
spends one transformer position per pair and reads the second token off `head2`,
which is what the engine's own captured step does. Driving a two-token
checkpoint through the one-token loop measures a model nobody ships and reports
about half its throughput; that is refused rather than printed.

Usage:
  python research/bench_batch.py <checkpoint> <voice> <device> <outdir> <batches>
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

import _bench
import loudkit
from loudkit.config import ExecutionConfig
from loudkit.models.generator import TorchTokenGenerator

DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64)


def _parser() -> argparse.ArgumentParser:
    return _bench.parser(
        __doc__, "cuda or cuda:<index> (indexed CUDA for multi-GPU)", DEFAULT_BATCHES
    )


def main() -> int:  # noqa: PLR0915 - a benchmark tool is one long straight-line run
    args = _parser().parse_args()
    ckpt, voice_path, device = args.checkpoint, args.voice, args.device
    batches: list[int] = args.batches
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    # CUDA graphs need sm_70+ (Volta); check the *targeted* device's capability,
    # not device 0, cuda:1 may be an older card (e.g. a Pascal 1080 Ti). For a
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
        execution=ExecutionConfig(device=device, cuda_graphs=graphs),
    )
    # The decode loop below is hand-written against the torch generator's own
    # modules; an ONNX or CoreML engine satisfies the TokenGenerator protocol
    # and has none of them.
    if not isinstance(e.token_generator, TorchTokenGenerator):
        raise SystemExit(f"{device}: this benchmark needs the torch generator")
    # Any, not TorchTokenGenerator: the step below reads submodules that
    # nn.Module's __getattr__ types as `Tensor | Module`.
    gen: Any = e.token_generator
    # The loop is the checkpoint's, not the tool's default. A two-token
    # checkpoint driven one token at a time is a model nobody ships.
    decode = e.algorithm.decode
    if decode not in ("single", "fusion_mtp2"):
        raise SystemExit(
            f"{ckpt}: decode mode {decode!r} has no loop here. Add one, or "
            f"benchmark it through tools/bench.py, which uses the engine."
        )
    fused = decode == "fusion_mtp2"
    if fused and not hasattr(gen, "fuse"):
        raise SystemExit(f"{ckpt}: declares {decode!r} but carries no fusion head")
    voice = loudkit.VoiceProfile.load(voice_path)
    from loudkit.frontend.speechtext import speech_text

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

        def run_batch(batch: int) -> tuple[float, float, float]:  # noqa: PLR0915
            k_bufs = torch.zeros(n_layers, batch, n_kv, max_len, hd, device=dev, dtype=dtype)
            v_bufs = torch.zeros_like(k_bufs)
            for i, (k, vv) in enumerate(cache):
                k_bufs[i, :, :, :prefill_len, :] = k[0].expand(batch, -1, -1, -1)
                v_bufs[i, :, :, :prefill_len, :] = vv[0].expand(batch, -1, -1, -1)
            grid_b = torch.arange(batch, device=dev).repeat_interleave(n_kv * hd)
            grid_k = torch.arange(n_kv, device=dev).repeat_interleave(hd).repeat(batch)
            grid_h = torch.arange(hd, device=dev).repeat(n_kv).repeat(batch)
            width = 2 if fused else 1
            token_buf = torch.zeros(batch, width, dtype=torch.long, device=dev)
            next_first = torch.zeros(batch, 1, dtype=torch.long, device=dev)
            emb_pos = torch.zeros(batch, 1, dtype=torch.long, device=dev)
            rope_pos = torch.zeros(batch, dtype=torch.long, device=dev)
            logits_buf = torch.zeros(batch, gen.SPEECH_VOCAB, dtype=torch.float32, device=dev)
            second_buf = torch.zeros(batch, gen.SPEECH_VOCAB, dtype=torch.float32, device=dev)

            def step():
                if fused:
                    # One slot for the pair, the engine's own arithmetic.
                    pair = gen.speech_emb(token_buf).to(dtype)
                    e_a, e_b = pair[:, :1], pair[:, 1:]
                    emb = 0.5 * (e_a + e_b) + gen.fuse(torch.cat((e_a, e_b), dim=-1))
                    emb = emb + gen.speech_pos_emb.emb(emb_pos)
                else:
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
                hidden = gen.tfmr.norm(x)[:, -1]
                logits_buf.copy_(gen.speech_head(hidden).float())
                if fused:
                    # The pair's second token, off the MTP head, as the engine
                    # reads it: [hidden ; e(first)].
                    e_first = gen.speech_emb(next_first)[:, 0].to(dtype)
                    second_buf.copy_(gen.head2(torch.cat((hidden, e_first), dim=-1)).float())

            token_buf.fill_(forced[0])
            next_first.fill_(forced[0])
            emb_pos.fill_(1)
            rope_pos.fill_(prefill_len)
            step()
            _bench.sync(device)

            if graphs:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    step()

                def run() -> None:
                    g.replay()

            else:
                run = step

            steps = cap // 2 if fused else cap
            t0 = time.perf_counter()
            for s in range(steps):
                if fused:
                    token_buf[:, 0] = forced[2 * s]
                    token_buf[:, 1] = forced[2 * s + 1]
                    next_first[:, 0] = forced[min(2 * s + 2, cap - 1)]
                else:
                    token_buf.fill_(forced[s])
                emb_pos.fill_(s + 1)
                # A fused pair takes one KV slot; a single token takes one too.
                rope_pos.fill_(prefill_len + s)
                run()
            _bench.sync(device)
            wall = time.perf_counter() - t0
            step_ms = wall / steps * 1000
            rtf = batch * audio_s / wall
            return step_ms, rtf, wall

        rows = []
        header = f"  {'batch':>5} {'ms/step':>9} {'tok/step':>8}"
        print(f"{header} {'aggregate':>10} {'per request':>12}")
        for batch in batches:
            step_ms, rtf, wall = run_batch(batch)
            per_request = audio_s / wall
            rows.append(
                {
                    "batch": batch,
                    "ms_per_step": round(step_ms, 3),
                    "tokens_per_step": 2 if fused else 1,
                    "rtf": round(rtf, 2),
                    "rtf_per_request": round(per_request, 2),
                }
            )
            print(
                f"  {batch:5d} {step_ms:9.3f} {2 if fused else 1:8d} "
                f"{rtf:9.2f}x {per_request:11.2f}x",
                flush=True,
            )

        cmd = (
            f"python research/bench_batch.py {ckpt} {voice_path} {device} {out} "
            f"{','.join(map(str, batches))}"
        )
        row = {
            "tool": "research/bench_batch.py",
            "command": cmd,
            "device": device,
            "cuda_graphs": graphs,
            "decode": decode,
            "tokens_per_step": 2 if fused else 1,
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
