# Benchmarking and profiling

Use `tools/bench.py` to measure a complete synthesis path and `tools/profile_stages.py` to find the
stage that takes the time. Published results are in
[Benchmarks](../benchmarks.md).

## Measure one runtime

`tools/bench.py` does not create the directory for `--json`, so create it
first:

```bash
mkdir -p out
python tools/bench.py \
  --checkpoint loudr-1/loudr-1.safetensors \
  --voice loudr-1/voices/joe.safetensors \
  --device cuda \
  --cuda-graphs \
  --json out/bench.json
```

For another path, change `--device` to `cpu`, `mps` or `onnx` and remove
`--cuda-graphs`, which is CUDA only. The command prints and saves:

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
python tools/profile_stages.py \
  --checkpoint loudr-1/loudr-1.safetensors \
  --voice loudr-1/voices/joe.safetensors \
  --device mps \
  --runs 5 \
  -- "The quick brown fox jumps over the lazy dog."
```

`tools/profile_stages.py` reports warm-up and median stage times. Use it when
a machine is slower than expected, to find whether generation or rendering is
the bottleneck.

## Measure batching

The batch harness measures aggregate token-generator throughput. It does not
include mel generation or the vocoder, so do not compare its result with an
end-to-end RTF.

```bash
python research/bench_batch.py \
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

See also [embedding loudkit](embedding.md).

## M3 Pro comparison, 0.1.1

[The benchmarks page](../benchmarks.md) compares both models on eight paths
on an Apple M3 Pro. The raw runs are in two files:

- [the 2026-09-05 run](../measurements/wave-t-2026-09-05.json), taken during
  the release build;
- [the 2026-09-06 run](../measurements/2026-09-06-m3pro-both-models.json),
  the release candidate, one process at a time.

`tools/bench_ports/README.md` and `tools/bench.py` reproduce them.
