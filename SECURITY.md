# Security

loudkit is a local, offline text-to-speech engine. The codebase stays small and
auditable by design; this file records the threat model that follows, and how to
report a problem.

## Threat model

* **Voice cloning.** The whole point of the project is to render arbitrary text
  in a voice enrolled from a few seconds of audio. That is the feature. loudkit
  cannot verify that you own the voice you enrolled, and it does not try to —
  that is the caller's responsibility (`RESPONSIBLE_USE.md`). Do not point this
  tool at a recording of someone else's voice without their consent.
* **The synthesis server.** `loudkit serve` holds a warm engine and answers
  requests over HTTP. **There is no sandbox**, and on loopback — the default —
  no authentication either: anyone who can reach the port can synthesise speech
  in every voice on disk. A non-loopback bind is refused unless `--allow-public`
  is passed, and then it *requires* a bearer token, generated and printed to
  stderr if you do not supply one; synthesis routes are rate limited per client
  address on such a bind. None of that makes it a service — it is a way to keep
  a model warm on your own machine, and the token exists so that forgetting a
  flag cannot leave an open one on a network.
* **Voice profiles.** A `VoiceProfile` is an enrolled speaker embedding plus a
  prompt. Anyone who can read your voice files can clone that voice. Protect
  voice directories the way you would protect a recording of yourself.
* **Weights.** The model weights are derived from Chatterbox (`NOTICE`); the
  shipped voice profiles are enrollments of consented or
  openly licensed recordings (`docs/voices/roster/provenance.json`).
  Load checkpoints only from sources you trust: the format cannot execute
  code, but adversarial weights still decide what the engine says and how it
  sounds.

## Supported

* Use the deterministic flags (`ExecutionConfig.deterministic`) rather than
  disabling them: bit-identical output is also the property that makes a
  golden-file regression visible.
* If you embed loudkit in a service, put a boundary in front of it — your own
  auth, your own rate limit, your own text filter — and keep the engine
  internal.

## Dependencies

Dependencies are declared as **minimum versions** in `pyproject.toml` (e.g.
`fastapi>=0.110`) — there is no lockfile, so the exact versions you get depend
on when you install and on your resolver. This is a deliberate choice for a
library, not a claim of pinning: treat a dependency update as a change that
could affect behaviour, and verify your own install against a known-good
combination before a critical deployment. The server (FastAPI/uvicorn) is an
optional extra, never a core dependency; the core runtime needs only numpy,
safetensors and tokenizers.

Official ONNX Runtime builds enable Microsoft telemetry by default. Loudkit
sets `ORT_DISABLE_TELEMETRY=1` before it initializes ONNX Runtime and also uses
the binding's disable API where one is exposed. An application that initializes
ONNX Runtime before Loudkit owns that process-global environment and must apply
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
specification. Their security properties are their own; this file describes what
loudkit itself does and does not promise.

## Supported versions

Pre-release: only the current `main` receives security fixes. After the
first tagged release, the latest tag and its predecessor each get fixes for
90 days.
