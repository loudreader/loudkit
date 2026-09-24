# Troubleshooting

Each entry gives a symptom, its cause and the fix. Run `loudkit doctor` first:
it reports what this machine can run and what to install to run more.

## Install and first run

### "loudkit is installed without a runtime"

The core package installs no runtime, so `lk.load` stops before it downloads
anything. The CLI prints the same message. Install the extras:

```bash
pip install "loudkit[torch,audio]"    # CPU, CUDA or Apple GPU: the usual choice
pip install "loudkit[onnx,audio]"     # no torch; needs exported graphs (below)
```

`audio` writes WAVs. Add `hub` to load models by name, `enroll` to clone a
voice, `server` for `loudkit serve`.

### "reading an audio file needs soundfile"

`lk.enroll` reads a file path with soundfile. Install `loudkit[audio]` for
cloning with `device="onnx"` or `device="coreml"`, or `loudkit[enroll]` for the
PyTorch path. Both bring soundfile. You can also pass mono samples in
`[-1, 1]` with their `sample_rate`. `loudkit clone` checks its extras before it
starts and names the missing packages.

### The first synthesis hangs on a download

It is fetching the checkpoint and the assets required by the selected backend.
See the [download sizes](../guides/01-getting-started.md#download-sizes-for-011)
for both models. The files go to the Hugging Face cache. Every process that
uses the same cache reuses them, across projects and virtualenvs. Pin a
`revision=` in production, so the same code cannot resolve to different
weights later: [pinning a release](COMPATIBILITY.md#pinning-a-release).

## Speed and hardware

### Synthesis is slower than real time

The usual cause is PyTorch on CPU, which runs below real time (0.29x for
loudr-1 on an M3 Pro, see [benchmarks](../benchmarks.md)). Check what this
machine can run:

```bash
loudkit doctor
```

Then check what the engine chose. `exec[...]` shows the devices in use:

```python
print(engine.describe())
```

On Apple silicon the default is a split engine (`gen=cpu/render=mps`). It is
the fastest Python setup measured on an M3 Pro.

### MPS or CoreML never activates inside Docker

On macOS a container is CPU-only, whatever the host: Apple does not pass Metal
through. Install natively on a Mac.
[Docker](../platforms/docker.md) carries the details and the arm64 CUDA caveat.

### `device="onnx"` stops with "ONNX assets not found"

The ONNX backend runs exported graphs: six for loudr-1, seven for
loudr-1-turbo. `lk.load` with a repo id fetches them. A local directory fetched
with the default `--for torch` has none. Fetch the graphs by backend:

```bash
loudkit download loudreader/loudr-1 --for onnx
```

Add `--with-cloning` for the three enrollment graphs as well. To export the
synthesis graphs from a checkpoint instead, use a checkout of the repository,
because the exporter is not in the pip package. The exporter also needs the
`onnx` package:

```bash
pip install "loudkit[torch,onnx]" onnx   # torch only to export
python tools/export_onnx.py --checkpoint loudr-1.safetensors
```

The enrollment graphs come from `tools/export_enroll_onnx.py`. See
[benchmarks](../benchmarks.md#onnx) for what the export gates measure.

### "exported from a different engine"

The checkpoint and graphs come from different exports. A packed checkpoint and
its split synthesis checkpoint contain the same synthesis tensors but have
different file hashes. Every port reads `export.json` beside the graphs and
refuses a set it says came from another checkpoint, another algorithm or
another step count. Download a complete bundle at one pinned revision, or
re-export the graphs from the exact checkpoint you intend to load. Keep the
matching enrollment checkpoint beside it when cloning. Do not edit `export.json`
to suppress the check.

A graph set without `export.json` loads with a warning.

### "CoreML assets not found: missing t3_*"

The native generator needs the `t3_*` packages beside the renderer packages.
Download the complete bundle with `--for coreml`. loudkit refuses a partial set
of `t3_*` packages, and loudr-1-turbo without its own generator packages.

A loudr-1 CoreML bundle with no `t3_*` packages at all loads with the PyTorch
generator on CPU, and needs the `torch` extra.
`ExecutionConfig(device="coreml", generator_device="cpu")` selects that setup
on any loudr-1 bundle.

### `onnx_provider="coreml"` takes about two minutes on the first run

With `onnx_provider="coreml"`, CoreML compiles the three renderer graphs on the
first run. This takes about two minutes on an M3 Pro, against three seconds for
the CPU provider. Nothing is printed during the compile.

The compiled models are cached in `~/Library/Caches/loudkit/coreml`, about
1.6 GB, and later runs open in about 25 s. Set `$LOUDKIT_COREML_CACHE` to move
the directory. If you delete it, the next run compiles again.

To avoid the compile, use the `cpu` provider. `auto` never selects `coreml`.

The JS port does not accept `coreml`, because `onnxruntime-node` cannot set a
cache directory and every process would compile again.

### The first call in a long-running process is much slower

The first call pays for kernel autotune, graph capture and allocator pools.
Call `engine.warm(voice)` at startup. `loudkit serve`, gRPC and MCP already
do, before they report ready. To skip the warm-up for a faster start, set
`LOUDKIT_NO_WARM` to any non-empty value. `loudkit speak` does not warm up,
because it renders once and exits.

### Turbo cannot be downloaded or loaded

Use loudkit 0.1.1 and a release that contains the graphs for your runtime.
loudkit cannot fetch a repository or revision that is not published. Check the
repo id and the revision in the error, or load a complete local release
directory.

Do not mix a checkpoint from one bundle with graphs from another, even when
both say loudr-1. Point at one complete release directory. See
["exported from a different engine"](#exported-from-a-different-engine) and the
CoreML entry above, and [choosing a model](../guides/11-choosing-a-model.md)
for where each model runs.

## Voices

### `voice_not_found` for a name

A bare name resolves against the release the engine was loaded from:
`engine.voice("joe")`. Outside an engine, name the release:

```python
voice = lk.voice("joe", repo="loudreader/loudr-1")
```

or a path: `lk.voice("voices/joe.safetensors")`. On the CLI,
`speak --checkpoint <repo> --voice joe` works without a path, because the
checkpoint names the release.

### A cloned voice reads text in the wrong language

Every profile carries a language, and `enroll` defaults it to `"en"`. Name it
at enrollment: `lk.enroll(..., language="pl")`. Every implementation picks the
language in this order: the call's `language=`, then `voice.language`, then
`"en"`. See [text normalization](../design/preprocess.md).

### The result ends with an odd word, or is flagged `SUSPECT`

Detectors check the speech tokens of each chunk for known failure patterns.
By default the engine trims a tail they flag. When a chunk is too long for its
text and no detector can locate the fault, the engine returns the result
marked `SUSPECT` (`result.suspect`). The detectors do not transcribe the audio.
What the flags mean: [postprocess](../design/postprocess.md).

## Errors worth knowing by name

| raise | means | fix |
|---|---|---|
| `WindowOverflowError` | a single window's generation did not fit | drop `single_window=True`; plain `synthesize` splits |
| `UnsupportedLanguageError` | language off the twelve-id roster | pick a supported `language=`; `.supported` lists them |
| speed outside 0.5 to 2.0 | refused, not clamped | pass a value from `lk.MIN_SPEED` to `lk.MAX_SPEED` |
| `NumberGrammarError` | a number past the largest scale that language's grammar names | raised by the number helpers in `loudkit.frontend.numbers`; synthesis reads such a number digit by digit |

The full catalog, per transport: [errors](errors.md).

## Determinism

### Same seed, different bytes on another machine

This is expected. Audio is bit-identical only on the same build, device and
backend. Across backends, floating-point sums differ, so the waveform differs.
At matched precision the speech tokens usually match across backends: the
English test sentences match token for token between PyTorch and ONNX. Polish
text on ONNX can diverge within a few tokens. The exact edges:
[identity contract](IDENTITY-CONTRACT.md).

### Streaming saves only the last chunk

Giving every chunk the same filename keeps overwriting it. One file per chunk:

```python
result.save(f"chunk-{i:03}.wav")
```

For one waveform, use `synthesize`.

## Server

### The server is up but nothing answers through the port mapping

Two usual causes, both [Docker](../platforms/docker.md):

- the server binds `127.0.0.1` inside the container, which a port mapping
  cannot reach; pass `--host 0.0.0.0`;
- its default port is **8765**, so a mapping to 8000 needs `--port 8000`.

A non-loopback bind then requires `--allow-public` and a bearer token, and
`/health` sits behind the token like every route, so a healthcheck must send it.
