# Jetson (JetPack 6 / L4T R36)

The Orin rows in [benchmarks.md](../benchmarks.md) were measured with this
setup. JetPack's Python environment needs three extra steps: NVIDIA's torch
wheel, a CUDA library that JetPack does not ship, and a loader path. Without
them the install looks broken.

## Torch

The generic PyPI wheel does not run on Jetson. Third-party Jetson package
indexes build torch against their own library stacks. Use NVIDIA's JetPack
wheel:

```bash
python3 -m venv .venv
.venv/bin/pip install "loudkit[audio]"
.venv/bin/pip install \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
.venv/bin/pip install "numpy<2"   # the NVIDIA wheel is compiled against 1.x
```

## Libraries the wheel needs but JetPack does not ship

`libcusparseLt` is not in L4T. Fetch NVIDIA's `linux-sbsa` archive of it once.
NVIDIA's PyTorch-for-Jetson guide runs a cuSPARSELt install script that fetches
the same `linux-sbsa` archive on aarch64.

```bash
mkdir -p ~/libs && cd ~/libs
curl -sL -o cslt.tar.xz https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.6.3.2-archive.tar.xz
tar xf cslt.tar.xz && cp libcusparse_lt-*/lib/libcusparseLt.so* .
```

Every run needs CUPTI and the CUDA libraries on the loader path:

```bash
export LD_LIBRARY_PATH=$HOME/libs:/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

## What to expect (Orin Nano Super, 25 W)

CUDA graphs raise streaming speed from 0.66x to 1.85x real time with loudr-1,
and from 1.33x to 2.50x with loudr-1-turbo, both measured on 0.1.1
(2026-09-06, the third benchmark passage). Under graphs, first audio arrives
after 2.7 s with loudr-1 and 2.0 s with turbo. In Python, turn graphs on with
`ExecutionConfig(cuda_graphs=True)` from `loudkit.config`:

```python
import loudkit as lk
from loudkit.config import ExecutionConfig

engine = lk.load(
    "path/to/loudr-1",  # a local release directory
    device="cuda",
    execution=ExecutionConfig(cuda_graphs=True),
)
```

The benchmark tool, `tools/bench.py`, takes `--cuda-graphs` for the same
setting.

A 96-token first chunk (`ChunkConfig.first_chunk_max_tokens = 96`) starts audio
sooner, and it changes the audio. On 0.1.0 it took first audio from 3.1 s to
2.55 s; it is not measured on 0.1.1. The rest of the first-audio time is
vocoder compute at this power budget: a vocoder captured as a CUDA graph
measured the same on this board. Pinning the clocks (`sudo jetson_clocks`) is
the one remaining setting to try.
