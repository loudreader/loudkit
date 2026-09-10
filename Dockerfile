# loudkit, as a container.
#
# One service, three variants. The engine is single-flight and holds one model
# in memory, so the backend is the only axis worth varying: `--build-arg
# VARIANT=` takes `cpu`, `cuda` or `onnx`, because torch-CUDA is about three
# gigabytes and a CPU user should not download it.
#
# The variant decides the backend, so `serve` has to be told which one: the
# onnx image carries no torch, and without `--device onnx` the server asks for
# the torch backend and dies at import. `compose.yaml` derives the flag from
# `VARIANT`; a bare `docker run` of `loudkit:onnx` passes `--device onnx`
# itself.
#
# `cuda` is an amd64 image in practice, and it fails softly rather than loudly:
# the torch wheel carrying the CUDA runtime is published for x86_64 only, while
# the aarch64 wheel of the same version is a CPU build, so an arm64 `cuda`
# image installs cleanly, starts, and quietly has no GPU. Build that one on
# amd64. `cpu` and `onnx` build natively on either architecture.
#
# On macOS every variant is CPU-only, whatever the host: Docker runs Linux
# containers in a VM and Apple does not pass Metal through to it, so MPS and
# CoreML are unreachable from inside a container. On a Mac, install natively.
#
# Weights are not baked in. Synthesis downloads a 747 MB checkpoint, versioned
# separately from the code, so baking it in would mean a new image for every
# weight release. Mount it, or let `loudkit.load("org/name")` fetch it.
#
# Three server flags are load-bearing in a container. `--host 0.0.0.0`, because
# a container's own loopback is not reachable through a port mapping;
# `--allow-public`, because the server treats a non-loopback bind as public;
# and `--port`, because the server's own default is 8765.
#
# The token is the one thing that is *not* a flag. `-e LOUDKIT_TOKEN` is the
# CLI's documented fallback and argv is world-readable: `docker inspect`, `ps`
# and the daemon's own logs all print a command line, so a token passed there
# is a token disclosed to every account on the host. Left unset entirely the
# server generates one and prints it, which the healthcheck below cannot read.
#
# The boundary is the `127.0.0.1:` on the left of the mapping, not the bind
# address. The server cannot see that, which is why it insists on the token.
#
# docs/platforms/docker.md carries the build and run commands, the variant
# table and the rest of the deployment story.

ARG VARIANT=cpu

FROM python:3.12-slim AS base
ARG VARIANT
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1
# libsndfile is soundfile's C library; without it every encode raises at import.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS build
ARG VARIANT
WORKDIR /build
COPY pyproject.toml README.md LICENSE NOTICE RESPONSIBLE_USE.md ./
COPY python ./python
# One venv, copied whole into the final stage: pip's resolver and its build
# dependencies stay out of the shipped image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# The CPU variant takes torch from PyTorch's CPU index — the default wheel
# carries the CUDA runtime and is several gigabytes of driver nobody on this
# path will load. The ONNX variant installs the server's dependencies directly
# rather than through `[server]`, which pulls `[torch]` on purpose (see
# pyproject) and would defeat the point of the variant.
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
# and tests/test_release.py checks. Release builds pass
# `--build-arg VERSION=...`; an unlabelled local build says `dev`.
ARG VERSION=dev
ARG REVISION=unknown
# Where this image came from, in the one vocabulary registries and scanners
# read. `image.source` is what makes a pulled image traceable to a commit.
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
# Unprivileged: the process reads a checkpoint and answers on a socket, and
# needs nothing a root user has.
RUN useradd --create-home --uid 10001 loudkit
USER loudkit
WORKDIR /home/loudkit

EXPOSE 8000
# `/health` reports the resolved algorithm and refuses to say "ok" while a
# synthesis is wedged, so it is a real readiness signal rather than a liveness
# tautology.
#
# `LOUDKIT_TOKEN` is read from the environment rather than baked in: a public
# bind requires a bearer token, `/health` is behind it like everything else, and
# a probe that cannot authenticate reports a healthy server as unhealthy for as
# long as it runs. Unset, the header is simply omitted, which is right for a
# loopback bind that needs no token.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
t=os.environ.get('LOUDKIT_TOKEN'); \
r=urllib.request.Request('http://127.0.0.1:8000/health', \
headers={'Authorization': 'Bearer '+t} if t else {}); \
sys.exit(0 if urllib.request.urlopen(r, timeout=4).status == 200 else 1)"

ENTRYPOINT ["loudkit"]
CMD ["--help"]
