# Security

loudkit is a local, offline text-to-speech engine. To report a vulnerability,
see [Reporting](#reporting).

## Threat model

### Voice cloning

loudkit renders any text in a voice enrolled from a few seconds of audio. It
does not verify that you have the right to use the voice you enrolled. That is
the caller's responsibility; see [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md). Do
not clone a recording of someone else's voice without their consent.

### The synthesis server

`loudkit serve` keeps an engine loaded and answers requests over HTTP. **It has
no sandbox.** On loopback, the default bind, it has no authentication either:
anyone who can reach the port can synthesise speech in every voice on disk.

The server refuses a non-loopback bind unless you pass `--allow-public`. A
public bind then *requires* a bearer token. If you do not supply one, the
server generates one and prints it to stderr. On a public bind, synthesis
routes are rate limited per client address. A public bind is therefore never
unauthenticated. The server is not a hardened network service.

### Voice profiles

A voice profile (`VoiceProfile`) is an enrolled speaker embedding plus a
prompt. Anyone who can read a profile file can synthesise speech in that voice.
Restrict access to voice directories. `loudkit clone` writes profiles with mode
0600 on POSIX.

### Weights

The model weights are derived from Chatterbox (see `NOTICE`). The shipped voice
profiles are enrollments of consented or openly licensed recordings (see
`docs/voices/roster/provenance.json`). Load checkpoints only from sources you
trust. The safetensors format cannot execute code, but untrusted weights control
what the engine says and how it sounds.

## Recommendations

* Keep `ExecutionConfig.deterministic` on. Bit-identical output makes a
  regression visible against golden files.
* If you embed loudkit in a service, keep the engine internal. Put your own
  authentication, rate limiting and text filtering in front of it.

## Dependencies

`pyproject.toml` declares most Python dependencies as **minimum versions** (for
example `fastapi>=0.110`). A few also have upper bounds: `numpy<2.4`, `mcp<3`
and `protobuf<8`. There is no Python lockfile, so the exact versions you get
depend on when you install and on your resolver. Treat a dependency update as a
change that can affect behaviour. Verify your install against a known-good set
of versions before a critical deployment. The server (FastAPI, uvicorn) is an
optional extra. The core runtime needs only numpy, safetensors and tokenizers.

Official ONNX Runtime builds enable Microsoft telemetry by default. loudkit sets
`ORT_DISABLE_TELEMETRY=1` before it initializes ONNX Runtime, and also uses the
binding's disable API where one is exposed. An application that initializes
ONNX Runtime before loudkit owns that process-global environment. It must apply
the same setting before its own initialization.

## Reporting

Report issues privately to the maintainers via the GitHub
security-advisory flow ("Report a vulnerability" on the repository's Security
tab). Please include:

* what you were doing;
* the loudkit version and device;
* a minimal reproduction if one exists.

## Acknowledgements

loudkit builds on Chatterbox (MIT, Resemble AI) and the Random123/Philox RNG
specification. loudkit makes no security claims for these upstream components.
[`NOTICE`](NOTICE) lists every third-party component.

## Supported versions

The latest release and its predecessor receive security fixes for 90 days.

## Model download trust

The initial HTTPS download trusts the repository owner and the supplied
checksum list (trust on first use). Checksums detect corruption and mixed
bundles; they are not an independent signature of the publisher. The local
receipt records the resolved revision and checksum-list digest. It permits
offline reuse; it does not protect against an attacker who can alter both the
local model and receipt. Pin a reviewed immutable revision for deployment.
