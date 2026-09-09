"""Measure the public stream: stream_bench.py BUNDLE VOICE TEXT DEVICE.

Run zero warms the engine; runs one through three are the measured repeats.
The same script can import an earlier checkout through PYTHONPATH.
"""

from __future__ import annotations

import json
import sys
import time

import loudkit


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: stream_bench.py BUNDLE VOICE TEXT DEVICE")
    bundle, voice_path, text, device = sys.argv[1:]
    started = time.perf_counter()
    execution = None
    runtime = device
    if device == "coreml-hybrid":
        from loudkit.config import ExecutionConfig

        device = "coreml"
        execution = ExecutionConfig(
            generator_device="cpu",
            precision={
                "token_generator": "fp16",
                "mel_decoder.estimator": "fp32" if bundle.endswith("turbo") else "fp16",
            },
        )
    engine = loudkit.load(bundle, device=device, execution=execution)
    # The checkpoint's rate, not a literal 24000: at any other rate a hardcoded
    # divisor reports the wrong audio duration, and so the wrong RTF, for every run.
    sample_rate = engine.algorithm.sample_rate
    load = time.perf_counter() - started
    voice = loudkit.voice(voice_path)
    runs = []
    for run in range(4):
        started = time.perf_counter()
        first, samples, tokens, chunks = 0.0, 0, 0, 0
        for chunk in engine.stream(text, voice, seed=7):
            if not chunks:
                first = time.perf_counter() - started
            chunks += 1
            samples += len(chunk.audio)
            tokens += len(chunk.tokens)
        runs.append(
            {
                "run": run,
                "seconds": time.perf_counter() - started,
                "ttfa_s": first,
                "audio_s": samples / sample_rate,
                "tokens": tokens,
                "chunks": chunks,
            }
        )
    print(
        json.dumps(
            {
                "runtime": "python-" + runtime,
                "bundle": bundle,
                "load_s": load,
                "execution": engine.describe(),
                "text": text,
                "seed": 7,
                "runs": runs,
            }
        )
    )


if __name__ == "__main__":
    main()
