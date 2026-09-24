# loudkit, as a container.
#
# One server image in three variants. `--build-arg VARIANT=` takes `cpu`,
# `cuda` or `onnx`. Only the backend and its dependencies differ: the cpu
# variant leaves out the CUDA runtime (about three gigabytes), and the onnx
# variant leaves out torch.
#
# The onnx image has no torch, so run it with `--device onnx`. `compose.yaml`
# derives the flag from `VARIANT`; with `docker run`, pass it yourself.
#
# Build `cuda` on amd64. The torch wheel with the CUDA runtime is published
# for x86_64 only, and the aarch64 wheel of the same version is a CPU build,
# so an arm64 `cuda` image installs and starts without a GPU. `cpu` and `onnx`
# build natively on either architecture.
#
# On macOS every variant is CPU-only. Docker runs Linux containers in a VM,
# and Apple does not pass Metal through to it, so MPS and CoreML are not
# available in a container. On a Mac, install natively.
#
# The image does not include weights. The 747 MB synthesis checkpoint is
# versioned separately from the code. Mount it, or pass a repo id to
# `--checkpoint`.
#
# A container needs three server flags. `--host 0.0.0.0`, because a port
# mapping cannot reach the container's own loopback; `--allow-public`, because
# the server treats a non-loopback bind as public; and `--port`, because the
# server's own default is 8765.
#
# Pass the token as `-e LOUDKIT_TOKEN`, the CLI's environment fallback. Do not
# pass it as a flag: `ps` shows a command line to every account on the host.
# `docker inspect` still shows the environment to anyone with access to the
# Docker daemon. If the variable is unset, the server generates a token and prints it
# to stderr, where the healthcheck below cannot read it.
#
# The `127.0.0.1:` on the left of the port mapping limits access to this
# machine. The server cannot see that, so it still requires the token.
#
# docs/platforms/docker.md has the build and run commands and the variant
# table.

ARG VARIANT=cpu

FROM python:3.12-slim AS base
ARG VARIANT
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1
# libsndfile is the C library soundfile uses to encode and decode audio.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS build
ARG VARIANT
WORKDIR /build
COPY pyproject.toml README.md LICENSE NOTICE RESPONSIBLE_USE.md ./
COPY python ./python
# Dependencies go into one venv in this build stage; the final stage copies
# the whole venv.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# The CPU variant takes torch from PyTorch's CPU index, because the default
# wheel carries several gigabytes of CUDA runtime. The ONNX variant installs
# the server's dependencies directly: the `[server]` extra pulls `[torch]` (see
# pyproject).
RUN set -eu; \
    case "$VARIANT" in \
      cpu)  pip install --extra-index-url https://download.pytorch.org/whl/cpu \
                        ".[server,hub,audio]" ;; \
      cuda) pip install ".[server,hub,audio]" ;; \
      onnx) pip install ".[onnx,hub,audio]" "fastapi>=0.110" "uvicorn>=0.29" ;; \
      *)    echo "VARIANT must be cpu, cuda or onnx (got '$VARIANT')" >&2; exit 2 ;; \
    esac

FROM base AS final
ARG VARIANT
# Not a version literal: the version lives in the files RELEASING.md §1 lists
# and tests/test_release.py checks. Pass `--build-arg VERSION=...` and
# `--build-arg REVISION=$(git rev-parse HEAD)` for a labelled build; a local
# build without them says `dev` and `unknown`.
ARG VERSION=dev
ARG REVISION=unknown
# OCI labels, which registries and scanners read. `image.source` names the
# repository; `image.revision` names the commit when REVISION is passed.
LABEL org.opencontainers.image.title="loudkit" \
      org.opencontainers.image.description="Local text-to-speech: one engine, five language ports, no network at synthesis time." \
      org.opencontainers.image.source="https://github.com/loudreader/loudkit" \
      org.opencontainers.image.url="https://github.com/loudreader/loudkit" \
      org.opencontainers.image.documentation="https://github.com/loudreader/loudkit/blob/main/docs/guides/04-server-and-agents.md" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.base.name="docker.io/library/python:3.12-slim" \
      com.loudkit.variant="${VARIANT}"
ENV LOUDKIT_VARIANT=${VARIANT} \
    PATH="/opt/venv/bin:$PATH"
COPY --from=build /opt/venv /opt/venv
# Unprivileged: the process only reads a checkpoint and answers on a socket.
RUN useradd --create-home --uid 10001 loudkit
USER loudkit
WORKDIR /home/loudkit

EXPOSE 8000
# `/health` reports the resolved algorithm. It answers 503 when the engine is
# wedged or one synthesis has held it for more than 120 s.
#
# The probe reads `LOUDKIT_TOKEN` from the environment. A public bind requires
# a bearer token, and `/health` needs it like every other route; a probe
# without it reports a working server as unhealthy. When the variable is
# unset, the probe sends no header, which suits a loopback bind.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
t=os.environ.get('LOUDKIT_TOKEN'); \
r=urllib.request.Request('http://127.0.0.1:8000/health', \
headers={'Authorization': 'Bearer '+t} if t else {}); \
sys.exit(0 if urllib.request.urlopen(r, timeout=4).status == 200 else 1)"

ENTRYPOINT ["loudkit"]
CMD ["--help"]
