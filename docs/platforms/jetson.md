# Jetson (JetPack 6 / L4T R36)

The Orin rows in [benchmarks.md](../benchmarks.md) were measured with this
setup. JetPack's Python environment has three traps that all present as a
broken install. This page is the path around them.

## Torch

The generic PyPI wheel does not run on Jetson. The community index only carries
builds for its own library stack. Use NVIDIA's JetPack wheel:

```bash
python3 -m venv .venv
.venv/bin/pip install "loudkit[audio]"
.venv/bin/pip install \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
.venv/bin/pip install "numpy<2"   # the NVIDIA wheel is compiled against 1.x
```

## Libraries the wheel needs but JetPack does not ship

`libcusparseLt` is not in L4T. Fetch NVIDIA's aarch64 archive once:

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

`--cuda-graphs` is the difference between below and above real time: 0.73x to
1.83x streaming, and 2.06x for `synthesize_long`. First audio floors at ~2.3 s.
`ChunkConfig.first_chunk_max_tokens = 96` takes it from 3.1 s to 2.55 s. The
remainder is genuine vocoder compute at this power budget, not overhead: a
captured-graph vocoder was tried and measured neutral here. Pinning clocks
(`sudo jetson_clocks`) is the remaining free lever.
