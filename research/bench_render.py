#!/usr/bin/env python3
"""Benchmark the renderer alone: does batching the mel decoder and the vocoder pay?

The token generator is launch-latency-bound, so batching it in lockstep buys a
lot (`research/bench_batch.py`). The renderer is the opposite shape, one large
parallel pass per utterance, and the question is whether it is bound by the same
launch latency or by arithmetic. The answer is a ratio, not an opinion: render N
utterances one after another, render the same N in one batched call, divide.

What it measures: **the mel decoder and the vocoder only**, one real utterance's
tokens and its mel repeated N times, with the token generator excluded. A device
where the batched call costs about as much as one single call is launch-bound
and wants batching; a device where it costs N times one single call is
compute-bound and batching there is a rewrite that buys nothing.

Aggregate and per-request throughput are separate columns. They answer different
questions and were never the same number. For the serial baseline they happen to
be equal, because a request there is done as soon as its own work is: N calls in
W seconds is N*audio/W aggregate, and one call is W/N, so audio/(W/N) is the
same number. That is why only the batched path prints both: batching is exactly
the trade that pulls them apart, everyone in the batch waits for the whole batch.

The last column is how far the batched rows sit from their own single calls.
Zero is byte equality; a batch of one always gives it and a wider batch
sometimes does. Anything else is the intra-op reduction order, which the batch
changes: measured from 1e-6 to 1.6e-2 depending on device, shape and thread
count. It is printed rather than assumed, because that number is the cost of
adopting a batched renderer and it is not always small.

A wall time taken next to somebody else's build is a number about that build.
The record carries the load average before and after the run, the thread count
and the estimator precision, so a suspicious row can be thrown out on evidence.

The load average is a lagging average and it counts this process too, so it is
weak evidence on its own. The `N vs 1` column is the better check, because it
has an expected value: N serial calls each pay their own overhead, so on a quiet
machine the ratio sits at N. A row well above its own batch size was interrupted
or throttled, and the run should be repeated rather than argued with.

Usage:
  python research/bench_render.py <checkpoint> <voice> <device> <outdir> [batches]
  # batches: comma list, default 1,2,4,8
  # device: cpu, mps, cuda or cuda:<index>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import numpy as np
import torch

import _bench
import loudkit
from loudkit.config import ExecutionConfig
from loudkit.models.flow import TorchMelDecoder
from loudkit.models.vocoder import TorchVocoder

DEFAULT_BATCHES = (1, 2, 4, 8)
REPEATS = 3
"""Timed rounds per cell, median reported. The renderer's spread between rounds
is small next to the effect being measured; the warm-up below matters more."""


def _median_wall(fn: Callable[[], object], device: str) -> float:
    """Median of REPEATS timed rounds, warmed twice.

    Warmed because the renderer's first call on a cold process costs seconds
    more than its steady state: allocator, kernel compilation and, on Metal, the
    shader cache. Timing that once and calling it the renderer is how a batching
    win gets invented.
    """
    for _ in range(2):
        fn()
    _bench.sync(device)
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        _bench.sync(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _parser() -> argparse.ArgumentParser:
    return _bench.parser(__doc__, "cpu, mps, cuda or cuda:<index>", DEFAULT_BATCHES)


def _worst_deviation(single: list[Any], batched: list[Any]) -> float:
    """How far the batched rows sit from their own single calls, largest first.

    Zero is byte equality, which a batch of one always gives and a wider batch
    sometimes does. Anything else is the intra-op reduction order the batch
    changed. Reported rather than assumed, because a speedup nobody checked
    against the single call is not a speedup.
    """
    worst = 0.0
    for a, b in zip(single, batched, strict=True):
        if a.shape != b.shape:
            return float("inf")
        worst = max(worst, float(np.abs(a.astype(np.float64) - b).max()))
    return worst


def main() -> int:
    args = _parser().parse_args()
    ckpt, voice_path, device = args.checkpoint, args.voice, args.device
    batches: list[int] = args.batches
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    load_before = [round(x, 2) for x in os.getloadavg()]

    e = loudkit.load(
        ckpt,
        device=device,
        # str, not Device: the registry accepts "cuda:1", which the Literal
        # does not cover; loudkit.load vets it.
        execution=ExecutionConfig(device=device),
    )
    # decode_batch and synthesize_batch are torch prototypes; an ONNX or CoreML
    # engine satisfies the renderer protocols and has neither.
    if not isinstance(e.mel_decoder, TorchMelDecoder) or not isinstance(
        e.vocoder, TorchVocoder
    ):
        raise SystemExit(f"{device}: this benchmark needs the torch renderer")
    # Any, not the two classes: `decode_batch` and `synthesize_batch` are not on
    # the `MelDecoder` / `Vocoder` protocols the engine declares, and reading
    # them through nn.Module's __getattr__ types them as `Tensor | Module`.
    mel_decoder: Any = e.mel_decoder
    vocoder: Any = e.vocoder

    voice = loudkit.VoiceProfile.load(voice_path)
    text = "The quick brown fox jumps over the lazy dog and the reader keeps its composure."
    r = e.synthesize(text, voice, seed=7)
    tokens, audio_s = list(r.tokens), r.duration
    # One utterance's mel, reused for every row, so the vocoder's shapes never
    # move between the serial and the batched call.
    mel = mel_decoder.decode(tokens, voice, seed=7)
    print(
        f"{ckpt} on {device}: {audio_s:.2f}s audio, {len(tokens)} tokens, "
        f"mel {mel.shape[1]} frames, estimator {mel_decoder.estimator_dtype}, "
        f"{torch.get_num_threads()} threads, load {load_before[0]:.2f}"
    )
    if load_before[0] > 1.5:
        print(
            f"  WARNING: load average {load_before[0]:.2f} before the run. The "
            f"millisecond columns are about this machine's other work as much "
            f"as about the renderer. The 'N vs 1' column survives it better.",
            flush=True,
        )

    stages: dict[str, tuple[Callable[[int], list[Any]], Callable[[int], list[Any]]]] = {
        "mel": (
            lambda n: [mel_decoder.decode(tokens, voice, seed=7 + i) for i in range(n)],
            lambda n: mel_decoder.decode_batch(
                [tokens] * n, [voice] * n, seeds=[7 + i for i in range(n)]
            ),
        ),
        "vocoder": (
            lambda n: [vocoder.synthesize(mel, voice, seed=7 + i) for i in range(n)],
            lambda n: vocoder.synthesize_batch(
                [mel] * n, [voice] * n, seeds=[7 + i for i in range(n)]
            ),
        ),
    }

    rows = []
    print(
        f"  {'stage':>8} {'batch':>5} {'serial ms':>10} {'batch ms':>9} {'N vs 1':>7} "
        f"{'speedup':>8} {'agg serial':>11} {'agg batch':>10} {'per req batch':>14} "
        f"{'vs single':>10}"
    )
    for stage, (serial_fn, batch_fn) in stages.items():
        one = 0.0
        for batch in batches:
            worst = _worst_deviation(serial_fn(batch), batch_fn(batch))
            serial = _median_wall(partial(serial_fn, batch), device)
            batched = _median_wall(partial(batch_fn, batch), device)
            # The launch-bound probe, as a column rather than a division a
            # reader does by hand: N serial calls over one serial call. Near N
            # means one call already fills the device and a batch has no idle
            # to reclaim; near 1 means it does. Needs batch 1 in the list.
            if batch == 1:
                one = serial
            # None, not NaN: NaN is not JSON and the record is read by tools.
            n_vs_one = serial / one if one else None
            rows.append(
                {
                    "stage": stage,
                    "batch": batch,
                    "serial_ms": round(serial * 1000, 3),
                    "batch_ms": round(batched * 1000, 3),
                    "serial_n_vs_one": None if n_vs_one is None else round(n_vs_one, 3),
                    "speedup": round(serial / batched, 3),
                    "rtf_serial": round(batch * audio_s / serial, 2),
                    "rtf": round(batch * audio_s / batched, 2),
                    "rtf_per_request": round(audio_s / batched, 2),
                    "worst_deviation": worst,
                }
            )
            agreement = "identical" if worst == 0.0 else f"{worst:.1e}"
            # Blank rather than a number when batch 1 was not asked for: the
            # ratio has no baseline to be a ratio against.
            versus = "-" if n_vs_one is None else f"{n_vs_one:5.2f}x"
            print(
                f"  {stage:>8} {batch:5d} {serial * 1000:10.1f} {batched * 1000:9.1f} "
                f"{versus:>7} {serial / batched:7.2f}x "
                f"{batch * audio_s / serial:10.2f}x "
                f"{batch * audio_s / batched:9.2f}x {audio_s / batched:13.2f}x "
                f"{agreement:>10}",
                flush=True,
            )

    cmd = (
        f"python research/bench_render.py {ckpt} {voice_path} {device} {out} "
        f"{','.join(map(str, batches))}"
    )
    row = {
        "tool": "research/bench_render.py",
        "command": cmd,
        "device": device,
        "audio_s": audio_s,
        "tokens_per_utterance": len(tokens),
        "mel_frames": int(mel.shape[1]),
        "euler_steps": e.algorithm.euler_steps,
        "guidance": e.algorithm.guidance,
        "checkpoint": str(ckpt),
        "voice": str(voice_path),
        # The estimator's dtype, which is what the deviation column is measured
        # in: fp16 on the shipped checkpoints, so 1.6e-2 at a mel peak of 11.9
        # is two of its ulp and not a large number.
        "precision": str(mel_decoder.estimator_dtype),
        # Process-global and it moves the bytes, not only the wall time. Two
        # runs at different thread counts do not agree on CPU.
        "num_threads": torch.get_num_threads(),
        "repeats": REPEATS,
        # A CPU row taken next to somebody else's build is a row about the
        # build. Both ends, so a run that started quiet and finished loaded is
        # as visible as one that was loaded throughout.
        "load_average_before": load_before,
        "load_average_after": [round(x, 2) for x in os.getloadavg()],
        "rows": rows,
    }
    (out / f"render_{device.replace(':', '_')}.json").write_text(
        json.dumps(row, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
