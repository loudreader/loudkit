# Benchmarks

Use this page to answer two questions:

1. How quickly does one synthesis finish?
2. How much speech can a server produce when work is batched?

Those are different measurements. Single-request results are end to end.
Batch results measure aggregate token-generator throughput and exclude the
renderer.

## Reading the numbers

- **RTF** is seconds of audio produced per second of wall time. `1.0x` is real
  time. Higher is faster.
- **TTFA** is time to first audio for the measured streaming request.
- **Aggregate throughput** adds the audio-equivalent output of every item in a
  batch. It is a capacity number, not the latency of one request.
- **Cold** includes the first run after loading. **Warm** is the median of three
  later runs.

Unless a row says otherwise, measurements use voice `joe`, seed `1234`, and
the third passage in the shipped benchmark set.

## Quick answer

| deployment | measured path | result |
|---|---|---:|
| NVIDIA desktop GPU | PyTorch, RTX 3090, CUDA graphs | 7.47x end to end |
| Apple Silicon | split PyTorch engine, M3 Pro | 3.43x end to end |
| Embedded NVIDIA | PyTorch, Jetson Orin Nano, CUDA graphs | 1.83x end to end |
| CPU without PyTorch | ONNX Runtime CPU provider, M3 Pro | 1.21x end to end |
| Swift on Apple Silicon | native generator plus CoreML renderer, M3 Pro | 2.28x warm |
| Batched NVIDIA workload | token generator, A100, batch 64 | 170.8x aggregate |

For one request on an Ampere-or-newer NVIDIA GPU, use CUDA graphs. For portable
CPU deployment without PyTorch, use ONNX Runtime. On Apple Silicon, the split
PyTorch path is the fastest measured Python path; Swift uses its native
generator with the CoreML renderer.

## End-to-end, batch 1

These rows include token generation and rendering.

### Current workstation and laptop measurements

| runtime | hardware | configuration | RTF | TTFA warm |
|---|---|---|---:|---:|
| PyTorch | RTX 3090 | CUDA graphs | 7.47x | 0.86s |
| PyTorch | RTX 3090 | eager CUDA | 3.39x | 1.84s |
| PyTorch | Apple M3 Pro | split CPU/MPS | 3.43x | 2.02s |
| ONNX Runtime | Apple M3 Pro | CPU provider | 1.21x | 4.58s |
| PyTorch | Apple M3 Pro | CPU reference | 0.33x | 17.2s |

The RTX 3090 figures use PyTorch 2.13 with CUDA 12.6. CUDA graphs capture the
decode step over a static KV cache. The flag is opt in because padded attention
can change sampled tokens on long windows. Use it when speed matters more than
byte identity with eager execution.

The Apple M3 Pro machine has a 10-core CPU, 18-core GPU and 36 GB of unified
memory. The split path generates on CPU while the previous window renders on
the GPU.

### Other measured NVIDIA hardware

| hardware | eager | CUDA graphs | note |
|---|---:|---:|---|
| A100 | 2.16x | 8.12x | older, shorter benchmark passage |
| L4 | 2.20x | 7.56x | older, shorter benchmark passage |
| T4 | 2.10x | 5.46x | older, shorter benchmark passage |
| GTX 1080 Ti | 1.95x | 1.89x | Pascal cannot capture the graph; the flag falls back |
| Jetson Orin Nano Super | 0.73x | 1.83x | JetPack 6, CUDA 12.6, 25 W mode |

The older A100, L4 and T4 runs used the previous 255-token version of the third
benchmark passage. They are useful device measurements but should not be
compared directly with the current RTX 3090 row.

### Swift

On an M3 Pro with an 11-core CPU, 14-core GPU and 36 GB of memory,
`Engine.synthesizeLong` measured `2.11x` cold and `2.28x` warm. The native
generator runs on CPU and the renderer runs through CoreML. This uses a Swift
harness rather than `loudkit bench`, so read it as a deployment result, not a
strict cross-runtime comparison.

## Aggregate throughput at batch N

These figures measure a fixed 255-token generator window decoded in lockstep.
Mel generation and the vocoder are excluded. They show server capacity, not
single-request latency.

| hardware | batch 1 | batch 8 | batch 16 | batch 32 | batch 64 |
|---|---:|---:|---:|---:|---:|
| RTX 3090, CUDA graphs | 20.1x | 80.8x | 110.8x | 137.3x | 153.1x |
| GTX 1080 Ti, eager | 2.16x | 16.7x | 33.5x | 43.5x | 47.3x |
| Jetson Orin Nano, CUDA | 3.82x | 9.82x | 11.08x | 11.77x | not measured |
| T4 | 8.43x | 25.21x | 29.74x | 31.86x | 31.39x |
| A100 | 15.68x | 69.41x | 103.50x | 140.90x | **170.80x** |
| L4 | 13.99x | 49.03x | 50.67x | 53.89x | 57.77x |
| Apple Silicon CPU | 2.71x | 4.43x | 4.81x | 4.97x | 4.97x |
| i7-6850K CPU | 0.87x | 0.98x | 1.01x | 1.02x | 1.17x |

The RTX 3090 rises from `20.1x` at batch 1 to `153.1x` at batch 64. The largest
measured aggregate result is `170.8x` on the A100 at batch 64. CPU throughput
barely changes because CPU decode is compute bound rather than launch bound.

Reproduce the table with:

```bash
python tools/bench_batch.py <checkpoint> <voice> <device> <outdir> 1,2,4,8,16,32,64
```

## ONNX

ONNX Runtime runs without PyTorch at inference. The release ships fp32 graphs;
fp16 did not justify a second artefact and int8 did not pass the quality gate.

`onnx_provider="auto"` selects CUDA when the installed runtime offers it and
CPU otherwise. CoreML and DirectML must be requested explicitly.

### ONNX execution providers

#### Apple M3 Pro

This table times `synthesize_long` with load excluded. It is a separate harness
from the end-to-end table above.

| port | provider | RTF | load | result |
|---|---|---:|---:|---|
| Python | CPU | 1.22x to 1.31x | 2.9s | reference |
| Python | CoreML | 2.06x to 2.17x | 113s cold, 25s warm | same tokens as CPU |
| Rust | CoreML | not timed | | supported, same token stream |
| Go | CoreML | not timed | | supported, same token stream |
| JavaScript | CoreML | unavailable | | runtime cannot persist the compile cache |

CoreML places the three renderer graphs on CoreML and keeps the token generator
on CPU. The first compile took 113 seconds and created about 1.6 GB in
`~/Library/Caches/loudkit/coreml`. Later loads took about 25 seconds. This cost
is why `auto` does not select CoreML.

#### RTX 3090

These rows compare ONNX providers on the same Linux machine. They are not the
PyTorch CUDA measurements above.

| port | CUDA provider | CPU provider | CUDA speedup | same tokens as CPU |
|---|---:|---:|---:|---|
| Python | 4.21x | 0.77x | 5.5x | yes |
| Rust | 3.60x | 0.70x | 5.1x | yes |
| Go | 2.68x | 0.67x | 4.0x | yes |
| JavaScript | 2.54x | 0.65x | 3.9x | yes |

The JavaScript row was measured with `onnxruntime-node` 1.26.0. The package
declares 1.27 or newer, whose CUDA build needs a newer NVIDIA driver than the
measurement machine had. Treat that row as evidence for the port, not a result
from the default npm installation.

DirectML has not been measured. Swift uses CoreML directly and does not expose
ONNX providers.

## Reproduce a result

Download only the runtime you want:

```bash
loudkit download loudreader/loudr-1 --for torch  --local-dir loudr-1
loudkit download loudreader/loudr-1 --for onnx   --local-dir loudr-1
loudkit download loudreader/loudr-1 --for coreml --local-dir loudr-1
```

Run the end-to-end benchmark:

```bash
loudkit bench \
  --checkpoint loudr-1/loudr-1.safetensors \
  --voice loudr-1/voices/joe.safetensors \
  --device cuda \
  --cuda-graphs \
  --json row.json
```

Use `--device mps`, `--device cpu` or `--device onnx` for the other Python
paths. The JSON contains RTF, TTFA, stage timings, peak memory, the exact command
and a determinism check.

For a stage-by-stage profile:

```bash
loudkit profile \
  --checkpoint loudr-1/loudr-1.safetensors \
  --voice loudr-1/voices/joe.safetensors \
  -- "A passage to profile."
```

See [Benchmarking](guides/05-benchmarking.md) for the command reference and
[Identity contract](reference/IDENTITY-CONTRACT.md) for which execution changes
may alter tokens or waveforms.
