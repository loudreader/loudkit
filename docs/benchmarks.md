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
- **Cold** includes the first run after loading. **Warm** means the engine has
  already run. In `tools/bench.py`, RTF comes from one complete stream per
  passage; warm TTFA is the median of three subsequent first-chunk probes.
  The native-port table reports medians of three complete warm streams.

Unless a row says otherwise, measurements use voice `joe`, seed `7`, and the
third passage in the shipped benchmark set (48 words).

## Measured on which release

Every row was measured on 0.1.1 (2026-09-06) for both models: the Apple rows on
one M3 Pro laptop, the NVIDIA rows on six parts. The one 0.1.0 leftover is the
ONNX-provider comparison on the RTX 3090 further down, which says so. The
README, the port guides and the website quote this page.

## Quick answer

| deployment | measured path | loudr-1 | loudr-1-turbo |
|---|---|---:|---:|
| Apple Silicon, Python | split PyTorch engine, M3 Pro | 3.29x | 5.77x |
| Apple Silicon, Swift | native generator plus CoreML renderer, M3 Pro | 2.49x | 3.44x |
| CPU without PyTorch | ONNX Runtime CPU provider, M3 Pro | 1.14x | 1.59x |
| NVIDIA desktop GPU | PyTorch, RTX 3090, CUDA graphs | 8.55x | 13.05x |
| Embedded NVIDIA | PyTorch, Jetson Orin Nano, CUDA graphs | 1.85x | 2.50x |
| Batched NVIDIA workload | token generator, A100, batch 64 | 85.3x aggregate | 223.6x aggregate |

For one request on an Ampere-or-newer NVIDIA GPU, use CUDA graphs. For portable
CPU deployment without PyTorch, use ONNX Runtime. On Apple Silicon the split
PyTorch path is the fastest measured path. loudr-1-turbo runs 1.4x to 2.1x
faster than loudr-1 on every end-to-end path measured, and 2.6x to 2.8x faster
in batched throughput.

## End-to-end on Apple M3 Pro, 0.1.1, both models

Apple M3 Pro (11-core CPU, 14-core GPU, 36 GB), macOS 26.1, 2026-09-06, commit
`a06cbb9`, the shipped 0.1.1 bundles (loudr-1 checkpoint `73e69a78`, loudr-1-turbo
`590dcf9e`), voice `joe`, seed 7, the third benchmark passage: 48 words, 361
speech tokens for loudr-1 and 352 for turbo, 14.4 s and 14.1 s of audio. One
process at a time, on a laptop in ordinary use. torch 2.13.0, onnxruntime
1.29.0, coremltools 9.0, Node 26.7 with onnxruntime-node 1.27.0, Swift 6.2.1.
Raw runs, with every setting and every cold run:
[2026-09-06-m3pro-both-models.json](measurements/2026-09-06-m3pro-both-models.json).

Python, `tools/bench.py`, the warm run of the passage:

| path | loudr-1 RTF | turbo RTF | loudr-1 TTFA | turbo TTFA |
|---|---:|---:|---:|---:|
| PyTorch, split CPU/MPS | 3.29x | 5.77x | 1.86s | 1.03s |
| Python, native CoreML renderer | 1.74x | 2.74x | 2.80s | 1.83s |
| ONNX Runtime, CPU provider | 1.14x | 1.59x | 4.22s | 2.97s |
| PyTorch, CPU reference | 0.29x | 0.56x | 16.71s | 8.58s |

The ports, `tools/bench_ports`, the same passage streamed four times per
process, medians of the three warm streams:

| port | loudr-1 RTF | turbo RTF | loudr-1 TTFA | turbo TTFA |
|---|---:|---:|---:|---:|
| Swift, native generator plus CoreML renderer | 2.49x | 3.44x | 1.93s | 1.38s |
| Rust, ONNX Runtime CPU | 1.21x | 1.74x | 4.04s | 2.74s |
| Go, ONNX Runtime CPU | 1.18x | 1.70x | 4.05s | 2.76s |
| TypeScript, ONNX Runtime CPU | 0.97x | 1.49x | 5.01s | 3.12s |

Every process produced the same token count on all four streams. The two
harnesses time the same passage but differ in shape, three passages once
against one passage four times, so compare within a table before across them.
Engine load is excluded from every figure: 3.6 s for the split engine, 19 s for
the native CoreML renderer (9 s for turbo), 2 s for ONNX Runtime, 11 s for
Swift (4 s for turbo).

## End-to-end on NVIDIA, 0.1.1, both models

Measured 2026-09-06 on six NVIDIA parts: an A100 SXM4 40 GB, an L4 and a T4 on
Google Cloud (a2-highgpu-1g, g2-standard-4 and n1-standard-8 on the
pytorch-2-9-cu129 image: torch 2.9.1+cu129, driver 580.173.02), an RTX 3090 in
a desktop (torch 2.11.0+cu128, driver 575.51.03), the GTX 1080 Ti beside it
(torch 2.7.1+cu126, the last builds that carry Pascal kernels) and a Jetson
Orin Nano Super at 25 W (JetPack 6, NVIDIA torch 2.5.0a0+nv24.08). Voice `joe`, seed 7, the
third benchmark passage, warm run, `tools/bench.py`; the cloud boxes ran it
from a startup script, the other two over ssh. The cloud boxes used the
shipped 0.1.1 bundles; the desktop and the board hold the release-candidate
split of 2026-09-05, the same tensors under a manifest without
`edge_fade_seconds`, which is applied after the vocoder and does not enter the
timing. CUDA graphs capture the decode step over a static KV cache; the flag
is opt in, the identity contract's "equivalent" class: deterministic, not
token-identical to eager. RTF, with the warm time to first audio in
parentheses:

| hardware | loudr-1 eager | loudr-1 CUDA graphs | turbo eager | turbo CUDA graphs |
|---|---:|---:|---:|---:|
| RTX 3090 | 2.36x (2.16s) | 8.55x (0.60s) | 4.99x (1.02s) | 13.05x (0.39s) |
| A100 40 GB | 2.19x (2.22s) | 7.68x (0.64s) | 4.68x (1.03s) | 11.87x (0.40s) |
| L4 | 2.22x (2.21s) | 7.50x (0.68s) | 4.81x (1.03s) | 11.93x (0.43s) |
| T4 | 1.89x (2.61s) | 5.16x (0.99s) | 3.95x (1.26s) | 7.95x (0.62s) |
| GTX 1080 Ti | 2.26x (2.21s) | 2.12x (2.34s)* | 4.64x (1.07s) | 4.26x (1.18s)* |
| Jetson Orin Nano Super, 25 W | 0.66x (7.07s) | 1.85x (2.74s) | 1.33x (3.80s) | 2.50x (1.98s) |

\* Pascal cannot capture a CUDA graph; with the flag on, the 1080 Ti runs the
eager path with the static cache, and the flag costs it a little.

Eager execution is launch bound, which is why an A100 and a T4 sit within 0.3x
of each other there; CUDA graphs remove the launches and the parts separate.
Turbo's advantage is larger eager (2.1x on the 3090) than under graphs (1.5x):
with the forward captured, the per-token sampler and the per-pair
synchronisation are a real share of what remains. Eager on a 1080 Ti is within
0.1x of eager on a 3090, for the same reason. Raw runs, every passage and
every setting:
[2026-09-06-nvidia-both-models.json](measurements/2026-09-06-nvidia-both-models.json).

## Aggregate throughput at batch N, 0.1.1, both models

`research/bench_batch.py` decodes a fixed 255-token generator window in
lockstep for N requests with CUDA graphs, mel decoding and the vocoder
excluded, and reports the audio-equivalent output of the whole batch per
second of wall clock. Server capacity, not the latency of one request. Turbo
emits two tokens per step, which is where its factor comes from.

| hardware | model | batch 1 | batch 8 | batch 16 | batch 32 | batch 64 |
|---|---|---:|---:|---:|---:|---:|
| A100 40 GB | loudr-1 | 14.4x | 48.4x | 63.7x | 77.4x | 85.3x |
| A100 40 GB | loudr-1-turbo | 36.8x | 125.4x | 166.2x | 202.6x | 223.6x |
| RTX 3090 | loudr-1 | 16.7x | 46.8x | 55.2x | 57.4x | 57.3x |
| RTX 3090 | loudr-1-turbo | 42.0x | 121.8x | 145.5x | 154.2x | 155.0x |
| L4 | loudr-1 | 12.5x | 20.8x | 23.2x | 25.0x | 25.9x |
| L4 | loudr-1-turbo | 29.6x | 53.8x | 59.5x | 64.9x | 67.7x |
| T4 | loudr-1 | 6.5x | 13.7x | 14.8x | 15.1x | 15.2x |
| T4 | loudr-1-turbo | 17.2x | 35.9x | 40.3x | 42.2x | 42.6x |

The 3090 saturates by batch 32 and the L4 and T4 by batch 16; the A100 is
still climbing at 64. The largest measured aggregate is 223.6x, turbo on the
A100 at batch 64. These rows replace the 0.1.0 batch table and are not
comparable with it: the harness's reporting changed in between.

Reproduce a row with:

```bash
python research/bench_batch.py <checkpoint> <voice> cuda <outdir> 1,2,4,8,16,32,64
```

## ONNX

ONNX Runtime runs without PyTorch at inference. The release ships fp32 graphs;
fp16 did not justify a second artefact and int8 did not pass the quality gate.

`onnx_provider="auto"` selects CUDA when the installed runtime offers it and
CPU otherwise. CoreML and DirectML must be requested explicitly.

### ONNX execution providers

#### Apple M3 Pro

The 0.1.1 figures for the CPU provider in every port, and for the native
CoreML renderer in Python, are in the M3 Pro tables above. CoreML places the
three renderer graphs on CoreML and keeps the token generator on CPU. Loading
takes about 20 s and the first compile creates about 1.6 GB under
`~/Library/Caches/loudkit/coreml`, which is why `auto` does not select it.

#### RTX 3090

These rows compare ONNX providers on the same Linux machine, measured on 0.1.0
with loudr-1. They are not the PyTorch CUDA measurements above.

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
python tools/bench.py \
  --checkpoint loudr-1/loudr-1.safetensors \
  --voice loudr-1/voices/joe.safetensors \
  --device cuda \
  --cuda-graphs \
  --json row.json
```

Use `--device mps`, `--device cpu` or `--device onnx` for the other Python
paths, and `--checkpoint loudr-1-turbo/loudr-1-turbo.safetensors` for turbo.
The port runners, their build commands and the shared passage are in
`tools/bench_ports/README.md`. The JSON contains RTF, TTFA, stage timings, peak memory, the exact command
and a determinism check.

For a stage-by-stage profile:

```bash
python tools/profile_stages.py \
  --checkpoint loudr-1/loudr-1.safetensors \
  --voice loudr-1/voices/joe.safetensors \
  -- "A passage to profile."
```

See [Benchmarking](design/benchmarking.md) for the command reference and
[Identity contract](reference/IDENTITY-CONTRACT.md) for which execution changes
may alter tokens or waveforms.
