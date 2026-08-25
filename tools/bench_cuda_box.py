"""Benchmark s1: run loudkit bench on a device, save JSON + all samples.

Produces leaderboard rows on a multi-GPU CUDA box (the published rows came
from an RTX 3090 + GTX 1080 Ti workstation)
with the reproducing command, plus every sample's tokens, mel and waveform
saved to disk so the run is fully re-inspectable.

Usage:
  python bench_cuda_box.py <checkpoint> <voice> <device> <outdir> [--seed N] [--cuda-graphs]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import numpy as np
import soundfile as sf

import loudkit
from loudkit import bench as bench_mod
from loudkit.config import ExecutionOverrides


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("voice")
    ap.add_argument("device", help="cpu, cuda, or cuda:<index> on a multi-GPU box")
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cuda-graphs", action="store_true")
    return ap


def main() -> int:
    args = _parser().parse_args()
    ckpt, voice_path, device = args.checkpoint, args.voice, args.device
    graphs, seed = args.cuda_graphs, args.seed
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    # str, not Device: the registry accepts "cuda:1", which the Literal does not
    # cover; loudkit.load vets it.
    execution = (
        ExecutionOverrides(device=device, cuda_graphs=graphs)  # type: ignore[arg-type]
        if graphs
        else None
    )
    engine = loudkit.load(ckpt, device=device, execution=execution)
    load_s = time.perf_counter() - t0
    voice = loudkit.VoiceProfile.load(voice_path)

    texts = list(bench_mod.DEFAULT_TEXTS)
    print(f"device={device} seed={seed} load={load_s:.2f}s", flush=True)

    # run the benchmark
    cmd = (
        f"loudkit bench --checkpoint {ckpt} --voice {voice_path} --device {device} "
        f"--seed {seed}" + (" --cuda-graphs" if graphs else "")
    )
    result = bench_mod.bench(engine, voice, texts=texts, seed=seed, load_s=load_s, command=cmd)

    # save every sample: tokens (json), mel (npy), wav (16-bit)
    for i, s in enumerate(result.samples, 1):
        r = engine.synthesize(s.text, voice, seed=seed)
        (out / f"sample{i}_tokens.json").write_text(
            json.dumps(list(r.tokens)), encoding="utf-8"
        )
        np.save(out / f"sample{i}_mel.npy", r.mel)
        sf.write(out / f"sample{i}.wav", r.audio, result.sample_rate)

    row_path = out / f"{device.replace(':', '_')}.json"
    row_path.write_text(bench_mod.to_json(result) + "\n", encoding="utf-8")
    print(bench_mod.render_table(result))
    print(f"saved -> {out} (json + {len(result.samples)} samples)")

    # determinism re-check (bench does it, but assert and report)
    first = result.samples[0]
    a = engine.synthesize(first.text, voice, seed=seed)
    b = engine.synthesize(first.text, voice, seed=seed)
    assert np.array_equal(a.audio, b.audio), "determinism broken on " + device
    print("deterministic: True (re-checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
