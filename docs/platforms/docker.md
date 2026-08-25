# Docker

One image, one process, three variants. The
[`Dockerfile`](../../Dockerfile) and [`compose.yaml`](../../compose.yaml) in the
repository root are the whole deployment story, and this page is what they mean.

Nothing publishes these images. There is no registry, no CI job and no manifest
list, so you build them yourself.

## Build

```bash
docker build --build-arg VARIANT=cpu  -t loudkit:cpu  .
docker build --build-arg VARIANT=cuda -t loudkit:cuda .
docker build --build-arg VARIANT=onnx -t loudkit:onnx .
```

| `VARIANT` | size | what it installs |
| --- | --- | --- |
| `cpu` | 1.45 GB | torch from PyTorch's CPU index, plus the server, hub and audio extras |
| `cuda` | ~5 GB | the default torch wheel, which carries the CUDA runtime |
| `onnx` | 779 MB | ONNX Runtime and the server's dependencies, no torch at all |

The backend is the only axis that varies, because torch-CUDA is about three
gigabytes of driver a CPU user should not download. There is no service split.
The engine is single-flight, holds one model in memory and takes a decode step
every 40 ms, so a boundary anywhere inside it would put a network hop in that
loop.

Everything else is shared: `python:3.12-slim`, `libsndfile1` for the WAV writer,
one virtualenv built in a separate stage and copied whole so pip and its build
dependencies stay out of the shipped image, an unprivileged `loudkit` user
(uid 10001), `EXPOSE 8000`, and `loudkit` as the entrypoint. The image carries
OCI labels, including `org.opencontainers.image.source`, which is what makes a
pulled image traceable back to a commit.

## Run

```bash
export LOUDKIT_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
docker run --rm -p 127.0.0.1:8000:8000 -e LOUDKIT_TOKEN="$LOUDKIT_TOKEN" \
  -v "$PWD/checkpoints:/weights:ro" -v "$PWD/voices:/voices:ro" \
  loudkit:cpu serve --checkpoint /weights/loudr-1.safetensors --voices /voices \
  --host 0.0.0.0 --port 8000 --allow-public
```

Or with compose, which passes the same flags for you:

```bash
export LOUDKIT_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
docker compose up --build              # CPU
VARIANT=onnx docker compose up --build # ONNX
```

`compose.yaml` reads four variables:

| variable | default | what it sets |
| --- | --- | --- |
| `LOUDKIT_TOKEN` | none, and the file refuses to start without it | the bearer token |
| `VARIANT` | `cpu` | the image to build, and the backend to ask for |
| `LOUDKIT_DEVICE` | the value of `VARIANT` | `--device`, when the variant name is not the right answer |
| `LOUDKIT_CHECKPOINTS`, `LOUDKIT_VOICES` | `./checkpoints`, `./voices` | the two read-only mounts |

`VARIANT` picks the image *and* the backend, because a torch-less image told to
use torch does not start: `serve` asks for the torch backend, dies at import,
and advises `pip install loudkit[torch,audio]`, which is wrong for the image the
user just built. `LOUDKIT_DEVICE` overrides the backend half on its own. Use it
for a cuda image on a box with no GPU visible (`cpu`), or for a specific card
(`cuda:1`).

## The token goes in the environment, never on argv

The server reads `LOUDKIT_TOKEN` when `--token` is absent, and that is the
supported way to pass it. A command line is world-readable: `docker inspect`,
`ps` and the daemon's own logs all print one, so a token on argv is a token
disclosed to every account on the host.

Left unset entirely, the server generates a token and prints it, which nothing
outside the container can read. The compose healthcheck then fails forever
against a server that is perfectly fine, so `compose.yaml` requires the variable
rather than defaulting it.

## Required server flags

- `--host 0.0.0.0`: a container's own loopback is not reachable through a port
  mapping, so the server's default bind publishes nothing.
- `--allow-public`: the server treats any non-loopback bind as public, and a
  public bind requires a bearer token.
- `--port 8000`: the server's own default is 8765, so the mapping would
  otherwise point at a port nothing is listening on. The service would come up
  unreachable rather than failing.

## The boundary is the port mapping

**The boundary is the `127.0.0.1:` on the left of the mapping**, not the bind
address. The server cannot see that from inside the container, which is why it
insists on the token anyway.

Publish the port more widely and the token is the only thing between the network
and every voice on the host, over plain HTTP. Put a TLS-terminating proxy in
front first. See
[the server guide](../guides/04-server-and-agents.md).

## Weights are never baked in

The synthesis checkpoint is 747 MB and is versioned separately from the code, so an image
carrying it would need rebuilding for every weight release. Mount it read-only
as above, or let `lk.load("org/name")` fetch it into the hub cache volume
`compose.yaml` declares, so a restart does not download it again.

## What compose hardens

The container runs `read_only`, with `no-new-privileges`, all capabilities
dropped, and a tmpfs on `/tmp` for the interpreter's and libsndfile's scratch
space. The process reads a checkpoint and answers on a socket: it needs no
capability and writes to exactly one path, the cache volume. A write anywhere
else is a bug, and `read_only` reports it at the write.

Memory is capped at 6 GB. A CPU render peaks around 3 GB, and the point of the
limit is to fail loudly instead of being OOM-killed halfway through a passage.

The healthcheck calls `/health`, which reports the resolved algorithm and
refuses to say "ok" while a synthesis is wedged. It sends the bearer token when
one is set, because `/health` sits behind the token like every other route.

## Architecture

Whatever you build it on. `python:3.12-slim` is published for linux/amd64 and
linux/arm64, and so are the wheels the `cpu` and `onnx` variants install, so both
build natively on either. The `cpu` variant was built and run on linux/arm64 on
2026-08-22, and `doctor` reported torch 2.13.0+cpu on Linux aarch64. That is the
only architecture measured here.

`cuda` is amd64 in practice, and it fails softly rather than loudly. The torch
wheel carrying the CUDA runtime is published for x86_64 only, while the aarch64
wheel of the same version is a CPU build, so an arm64 `cuda` image installs
cleanly, starts, and quietly has no GPU. Build that one on amd64.

## On macOS a container is CPU-only

Whatever the host. Docker runs Linux containers in a VM and Apple does not pass
Metal through, so MPS and CoreML are unreachable from inside one. A `loudkit:mps`
image builds, runs, and falls back to the CPU, which gives a benchmark row
labelled MPS that measured something else. On a Mac, install natively. See
[Apple](apple.md).
