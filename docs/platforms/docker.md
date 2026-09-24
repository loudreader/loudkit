# Docker

The [`Dockerfile`](../../Dockerfile) in the repository root builds a loudkit
server image in three variants. [`compose.yaml`](../../compose.yaml) runs it.

No prebuilt images are published. Build them yourself.

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

The variants differ only in the inference backend and its dependencies. The
`cpu` variant leaves out the CUDA runtime, which is about three gigabytes. The
`onnx` variant leaves out torch. Each container runs one server process, which
holds the model in memory.

All variants share the rest:

- the `python:3.12-slim` base, with `libsndfile1` for the audio encoders;
- one virtualenv, installed in a separate build stage and copied into the final
  image;
- an unprivileged `loudkit` user (uid 10001);
- `EXPOSE 8000`, and `loudkit` as the entrypoint.

The image carries OCI labels. `org.opencontainers.image.source` names the
repository. `org.opencontainers.image.revision` names the commit only if you
pass it at build time; otherwise it is `unknown`, and the version is `dev`:

```bash
docker build --build-arg VARIANT=cpu --build-arg REVISION=$(git rev-parse HEAD) \
  --build-arg VERSION=<version> -t loudkit:cpu .
```

## Run

Put a release in `./checkpoints` first. For the `onnx` variant, add
`--for onnx`: the server reads the ONNX graphs from `checkpoints/onnx/`.

```bash
loudkit download loudreader/loudr-1 --local-dir checkpoints
```

```bash
export LOUDKIT_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
docker run --rm -p 127.0.0.1:8000:8000 -e LOUDKIT_TOKEN \
  -v "$PWD/checkpoints:/weights:ro" -v "$PWD/checkpoints/voices:/voices:ro" \
  loudkit:cpu serve --checkpoint /weights/loudr-1.safetensors --voices /voices \
  --host 0.0.0.0 --port 8000 --allow-public
```

`-e LOUDKIT_TOKEN` with no value copies the variable from your shell. For
`loudkit:onnx`, add `--device onnx` to the `serve` flags. The download puts
the release's voices in `checkpoints/voices`, and the command mounts that
directory. To serve your own voice profiles, mount their directory at
`/voices` instead.

Or with compose, which passes the same flags for you:

```bash
export LOUDKIT_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
export LOUDKIT_VOICES=./checkpoints/voices
docker compose up --build              # CPU
VARIANT=onnx docker compose up --build # ONNX
```

`compose.yaml` reads seven variables:

| variable | default | what it sets |
| --- | --- | --- |
| `LOUDKIT_TOKEN` | none; compose refuses to start without it | the bearer token |
| `VARIANT` | `cpu` | the image to build, and the default `--device` |
| `LOUDKIT_DEVICE` | the value of `VARIANT` | `--device` |
| `LOUDKIT_CHECKPOINTS` | `./checkpoints` | the read-only mount at `/weights` |
| `LOUDKIT_VOICES` | `./voices` | the read-only mount at `/voices` |
| `LOUDKIT_MODEL_FILE` | `loudr-1.safetensors` | the checkpoint file in `/weights`; `loudr-1-turbo.safetensors` for turbo |
| `LOUDKIT_NO_WARM` | `0` | the server's warm-up switch: any non-empty value, `0` included, skips the startup warm-up |

Compose replaces an unset or empty `LOUDKIT_NO_WARM` with `0`, so the compose
service always starts without the warm-up. The first request then pays the
extra cost of the first render. With `docker run`, the warm-up runs unless you
set the variable.

`VARIANT` sets both the image and the server's `--device`. The `onnx` image has
no torch, so it must run with `--device onnx`. `LOUDKIT_DEVICE` overrides only
`--device`. Use `cpu` for a `cuda` image on a host with no GPU visible, or
`cuda:1` for a specific card.

### A GPU for the `cuda` variant

The container gets a GPU only if you ask for one, and `compose.yaml` does not.
The host needs the NVIDIA Container Toolkit. With `docker run`, add
`--gpus all`. With compose, put a device reservation in a
`compose.override.yaml` beside `compose.yaml`, which compose reads
automatically:

```yaml
services:
  loudkit:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Pass the token in the environment

The server reads `LOUDKIT_TOKEN` when `--token` is absent. Use the environment
variable. On Linux, any account on the host can read a process's command line
with `ps`, so a token given with `--token` is visible to all of them.
`docker inspect` also shows the container's environment, so anyone with access
to the Docker daemon can still read the token.

If `LOUDKIT_TOKEN` is unset, the server generates a token and prints it to
stderr, so `docker logs` shows it. The healthcheck reads only `LOUDKIT_TOKEN`
and cannot use the generated token, so it reports the container as unhealthy.
`compose.yaml` therefore requires the variable.

## Required server flags

- `--host 0.0.0.0`: a port mapping cannot reach the container's own loopback,
  so the server's default bind is unreachable from the host.
- `--allow-public`: the server treats any non-loopback bind as public, and a
  public bind requires a bearer token.
- `--port 8000`: the server's own default is 8765. Without this flag nothing
  listens on the mapped port, and the container still starts.

## Host access

The `127.0.0.1:` on the left of the port mapping publishes the port on this
machine's loopback only. The server inside the container cannot see this, so
it requires the token for its `0.0.0.0` bind.

If you publish the port on other interfaces, the token is the only access
control, and it crosses the network in plain HTTP. Put a TLS-terminating proxy
in front first. See [the server guide](../guides/04-server-and-agents.md).

## Weights are not in the image

The synthesis checkpoint is 747 MB and is versioned separately from the code.
Mount it read-only, as above. `compose.yaml` also mounts a `loudkit-hub` volume
at the Hugging Face cache path, `/home/loudkit/.cache/huggingface`.

## What compose hardens

The container runs with a read-only root filesystem, `no-new-privileges` and
all capabilities dropped. The only paths not mounted read-only are the
`loudkit-hub` volume and a tmpfs on `/tmp`, which the interpreter and libsndfile
use as scratch space.

Memory is capped at 6 GB. A CPU render peaks around 3 GB. A container that goes
over the cap is killed by the kernel's out-of-memory killer.

The healthcheck calls `/health` every 30 s. `/health` reports the resolved
settings, and it returns `503` when the engine is wedged or when one synthesis
has held it for more than 120 s. The healthcheck sends `LOUDKIT_TOKEN` as the
bearer token when it is set, because on a public bind `/health` needs the token
like every other route.

## Architecture

`python:3.12-slim` is published for linux/amd64 and linux/arm64, and so are
the wheels the `cpu` and `onnx` variants install. Both variants build natively
on either architecture. The `cpu` variant was built and run on linux/arm64 on
2026-08-22, and `doctor` reported torch 2.13.0+cpu on Linux aarch64. That is
the only architecture measured here. The 0.1.1 ONNX image also generated audio
with both 0.1.1 models on linux/arm64, with networking disabled and the model
mounts read-only.

Build the `cuda` variant on amd64. The torch wheel with the CUDA runtime is
published for x86_64 only, and the aarch64 wheel of the same version is a CPU
build. An arm64 `cuda` image therefore installs and starts without error, but
has no GPU.

## On macOS a container is CPU-only

Docker runs Linux containers in a VM, and Apple does not pass Metal through to
it. MPS and CoreML are not available inside a container. The `Dockerfile` has
no `mps` variant. On a Mac, install loudkit natively. See [Apple](apple.md).
