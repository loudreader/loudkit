# 5. Benchmarking and profiling

Use `bench` to measure a complete synthesis path and `profile` to find the
stage that takes the time. Published results are in
[Benchmarks](../benchmarks.md).

## Measure one runtime

```bash
loudkit bench \
  --checkpoint loudr-1/loudr-1.safetensors \
  --voice loudr-1/voices/joe.safetensors \
  --device cuda \
  --cuda-graphs \
  --json out/bench.json
```

Change `--device` to `cpu`, `mps` or `onnx` for another path. The command
prints and saves:

- real-time factor, or RTF;
- time to first audio;
- load time and peak memory;
- time spent in the generator, mel decoder and vocoder;
- whether the repeated determinism check passed;
- the command needed to reproduce the row.

An RTF of `1.0x` is real time. `3.0x` means one minute of audio takes about
twenty seconds to produce.

## Profile one passage

```bash
loudkit profile \
  --checkpoint loudr-1/loudr-1.safetensors \
  --voice loudr-1/voices/joe.safetensors \
  --device mps \
  --runs 5 \
  -- "The quick brown fox jumps over the lazy dog."
```

`profile` reports warm-up and median stage times. Use it when a machine is
slower than expected and you need to know whether generation or rendering is
the bottleneck.

## Measure batching

The batch harness measures aggregate token-generator throughput. It does not
include mel generation or the vocoder, so do not compare its result with an
end-to-end RTF.

```bash
python tools/bench_batch.py \
  loudr-1/loudr-1.safetensors \
  loudr-1/voices/joe.safetensors \
  cuda \
  out/batch \
  1,2,4,8,16,32,64
```

## Compare results safely

Keep the text, voice, seed, build and device configuration fixed. Label CUDA
graphs and execution providers explicitly. A faster aggregate batch result does
not mean one request returns sooner, and deterministic output on one backend
does not promise byte-identical audio on another.

Next: [embed loudkit in your program](06-embedding.md).
