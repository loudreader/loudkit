# Troubleshooting

Symptoms first, causes second, fixes third. Every entry links the page that
carries the full behaviour. `loudkit doctor` is the first command for any of
these: it reports what this machine can run and what to install to run more.

## Install and first run

### `ModuleNotFoundError: No module named 'torch'`

The core package ships without a runtime, so `load()` can read a checkpoint but
not run one. Install the extras:

```bash
pip install "loudkit[torch,audio]"    # CPU, CUDA or Apple GPU: the usual choice
pip install "loudkit[onnx,audio]"     # no torch; needs exported graphs (below)
```

`audio` writes WAVs. Add `hub` to load models by name, `enroll` to clone a
voice, `server` for `loudkit serve`.

### Reading an audio file raises "needs the 'enroll' extra"

Reading files goes through librosa. Install `loudkit[audio]` for graph-based
cloning, or `loudkit[enroll]` for the PyTorch path. You can also pass mono samples
in `[-1, 1]` directly.

### The first synthesis hangs on a download

It is fetching the checkpoint and the assets required by the selected backend.
See the [download sizes](../guides/01-getting-started.md) for both models. They land in the
standard Hugging Face cache, so every later process on the machine, other
projects and other virtualenvs included, reuses it. Pin a `revision=` in
production so the same code cannot resolve to different weights later:
[pinning a release](COMPATIBILITY.md#pinning-a-release).

## Speed and hardware

### Synthesis is slower than real time

You are almost certainly on CPU, which renders at a fraction of real time by
design. Check what the engine chose:

```bash
loudkit doctor          # what this machine can run
print(engine.describe())  # exec[...] shows the devices actually in use
```

On Apple silicon expect a split engine (`gen=cpu/render=mps`): that is the
measured optimum, not a fallback.

### MPS or CoreML never activates inside Docker

On macOS a container is CPU-only, whatever the host: Apple does not pass Metal
through. Install natively on a Mac.
[Docker](../platforms/docker.md) carries the details and the arm64 CUDA caveat.

### `device="onnx"` refuses to start

The ONNX backend runs exported graphs. The release on the Hub carries all
nine; what it does not do is send them to a caller who asked for torch, which
is the default. Ask for them by backend:

```bash
loudkit download loudreader/loudr-1 --for onnx
```

Add `--with-cloning` for the three enrollment graphs as well. To export them
from a checkpoint instead:

```bash
pip install "loudkit[torch,onnx]"   # torch only to export
python tools/export_onnx.py --checkpoint loudr-1.safetensors
```

See [benchmarks](../benchmarks.md#onnx) for what the export gates measure.

### "exported from a different engine"

The checkpoint and graphs come from different exports. A packed checkpoint and
its split synthesis checkpoint contain the same synthesis tensors but have
different file hashes. Every port reads `export.json` beside the graphs and
refuses a set it says came from another checkpoint, another algorithm or
another step count. Download a complete bundle at one pinned revision, or
re-export the graphs from the exact checkpoint you intend to load. Keep the
matching enrollment checkpoint beside it when cloning. Do not edit `export.json`
to suppress the check.

A set carrying no `export.json` warns instead of refusing, because releases
exported before the record exist.

### "CoreML assets not found: missing t3_*"

The native generator needs the `t3_*` packages as well as the renderer. Download
the complete bundle with `--for coreml`. An older renderer-only base bundle can
use the PyTorch generator with `ExecutionConfig(device="coreml",
generator_device="cpu")` and the `torch` extra. Turbo requires its matching
generator exports.

### The first `coreml` run takes about two minutes

Not a hang. Asking for `onnx_provider="coreml"` makes CoreML compile the three
renderer graphs, which takes roughly two minutes on an M3 Pro against three
seconds for the CPU provider. Nothing is printed while it happens.

It is paid once per machine. The compiled models land in
`~/Library/Caches/loudkit/coreml`, about 1.6 GB, and later runs open in about
25 s. Set `$LOUDKIT_COREML_CACHE` to move the directory; delete it and the two
minutes come back.

If two minutes at startup is not acceptable, use `cpu`. `auto` already does:
it never selects `coreml`, for exactly this reason.

In JS the provider is refused outright, because `onnxruntime-node` cannot name a
cache directory and every process would pay the compile again.

### The first call in a long-running process is much slower

Kernel autotune, graph capture and allocator pools are paid once. Call
`engine.warm(voice)` at startup; `loudkit serve`, gRPC and MCP already do,
before they report ready. Set `LOUDKIT_NO_WARM` to any value to trade that
back for a faster start. `loudkit speak` never warms: it renders once and
exits, so there is nowhere to move the cost to.

### Turbo cannot be downloaded or loaded

Use loudkit 0.1.1 and a release containing graphs for your runtime. A repository
or revision that is not published cannot be fetched by name: use a complete
local release directory, or check the name and revision you requested.

Do not mix a checkpoint from one bundle with graphs from another. Even if
both say loudr-1, files from different bundles may not belong together.
Point at one complete release directory. The error “exported from a different
engine” means this consistency check worked; bypassing it is not a fix.

Older renderer-only CoreML releases of loudr-1 still use a PyTorch generator
in Python. Install the `torch` extra for those, or use a complete new CoreML
release to run without PyTorch. Incomplete native generator sets are refused.

See [choosing a model](../guides/11-choosing-a-model.md).

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
at enrollment: `lk.enroll(..., language="pl")`. The chain everywhere is the
call's `language=`, then `voice.language`, then `"en"`; see
[text normalization](../design/preprocess.md).

### The result ends with an odd word, or is flagged `SUSPECT`

The engine reads back the tokens it produced and cuts hallucinated tails; a
chunk that is detectably wrong but not localisable is reported instead of
returned silently. What the flags mean:
[postprocess](../design/postprocess.md).

## Errors worth knowing by name

| raise | means | fix |
|---|---|---|
| `WindowOverflowError` | a single window's generation did not fit | drop `single_window=True`; plain `synthesize` splits |
| `UnsupportedLanguageError` | language off the twelve-id roster | pick a supported `language=`; `.supported` lists them |
| speed outside 0.5 to 2.0 | refused, not clamped | clamp at the call site |
| `NumberGrammarError` | a number that language's grammar cannot say | rephrase, or split the sentence |

The full catalog, per transport: [errors](errors.md).

## Determinism

### Same seed, different bytes on another machine

Expected. Bit-identity holds per build, device and backend; across backends the
waveform differs because floating-point summation differs. The portable layer
is the **speech tokens**, identical across implementations at matched
precision. The exact edges: [identity contract](IDENTITY-CONTRACT.md).

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
