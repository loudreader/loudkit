# Compatibility

Rules for what can change between releases, per surface. The guarantees for
the audio itself are in [the identity contract](IDENTITY-CONTRACT.md).

## The fingerprint

Every engine computes an **algorithm fingerprint**: 16 hex digits over the
algorithm configuration. The configuration covers the text rules and grammar
tables, the sampler, chunking, the detectors and the checkpoint's algorithm
settings. It does not cover the weights. The provenance manifest of a saved
WAV identifies the weights by `checkpoint_sha256`.

The five implementations compute the fingerprint independently and agree.
`engine.describe()` prints it, and a saved WAV records it. In 0.1.1 the
loudr-1 fingerprint is `7cd75498ad4e7531` and the loudr-1-turbo fingerprint is
`e5303ba243087222`.

Every audible change to the algorithm or the grammar tables moves the
fingerprint, with one exception: the Polish respelling rules
(`pl_respell_rules.json`) are not hashed into it. A test pins the bytes of that
file. An edit fails the test, and the test asks for a bump of the text-layer
version, which moves the fingerprint.

Execution settings are not part of the fingerprint. These include the device,
the precision, `cuda_graphs`, the ragged vocoder and the ONNX execution
provider. They can change the waveform, and some of them can change the speech
tokens. The identity contract classes each of them.

The fingerprint does not promise identical bytes across releases. Bit-identical
audio needs the same build, device and execution configuration, as the
identity contract states. Across releases, an unchanged fingerprint means an
unchanged algorithm, and the conformance fixture checks its cases against it.

## Versioning

From 1.0 onward, semantic versioning applies to the API, and the fingerprint
is a second axis:

| change | version effect | fingerprint |
| --- | --- | --- |
| new function, parameter with a default, response field, route | minor | unchanged |
| an audible algorithm change (sampling, windowing, joins, detectors) | minor at least | moves |
| removing or renaming anything public; changing a type or a default's meaning | major | may move |
| a bug fix that does not change output | patch | unchanged |
| faster, less memory, a new backend, better errors | patch or minor | unchanged |

### Before 1.0

Before 1.0, any `0.x` release, minor or patch, may change the public API and
may move the fingerprint. The CHANGELOG entry of the release lists every
breaking change. When a release moves the fingerprint, its CHANGELOG entry
starts with the old and the new value and names each change that moved it,
with its measurement.

## Pinning a release

Without a revision, `load` resolves the repository's default branch, so the
same code can get different weights later. In production, pin a full
40-character commit id from the model repository:

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1", revision="<40-character commit id>")
```

Every command that takes a repo id also takes `--revision`:

```bash
loudkit serve --checkpoint loudreader/loudr-1 \
  --revision "<40-character commit id>"
```

Every download is checked against the release's `SHA256SUMS` before it is
used. This check shows that the files arrived intact. The revision identifies
the release.

A directory that a download wrote (`loudkit download --local-dir`, or a port's
`download`) contains `.loudkit-release.json`. This receipt records the repo,
the requested revision, the commit it resolved to, the digest of
`SHA256SUMS` and the fetch time. A receipt is valid when all of these are
true:

- Every field is present and has its type.
- The repo is the one requested.
- The commit is forty lowercase hex digits.
- The digest is the lowercase hex SHA-256 of the `SHA256SUMS` on disk.
- Every file it lists that the download would select is present.

Validating a receipt hashes no weight file. A download into that directory,
and a port's load by repo id, then does one of these:

- The revision resolves to the commit in a valid receipt: nothing is hashed.
- Any other commit, an invalid receipt or no receipt: the load fetches
  `SHA256SUMS` again and keeps only the files that still match it.
- The Hub cannot be reached: the load uses a valid receipt and says so on
  stderr. Without a valid receipt it raises an error.

A receipt records what was verified when the directory was written. It does
not detect later local edits. A `SHA256SUMS` rewritten on disk with its digest
copied into the receipt passes, and so does a weight file edited in place.
Both pass until the revision resolves to a different commit. To verify the
files again, delete `.loudkit-release.json` and run the same
`loudkit download <repo> --revision <commit> --for <backend> --local-dir <directory>`
again. Python's `lk.load(<directory>)` reads a local directory as it is: it
reads no receipt and hashes nothing.

Python's own cache follows the same rule. A load by repo id asks the Hub once
per process which commit `main` (or the tag) resolves to. It reads the cached
snapshot only when it is that commit. Offline, it uses the cache and prints one
line on stderr. A revision that is a full commit id is not resolved on the Hub.

## Versioned formats

Each format below carries a version or a label. The voice profile and
checkpoint readers refuse a version they do not know. A new field is additive.
Removing a field, renaming it or giving it a new meaning is breaking. From 1.0
that needs a major release; before 1.0 the CHANGELOG lists it.

| format | pinned as | value |
| --- | --- | --- |
| voice profile | `VOICE_FORMAT_VERSION` | `1` |
| checkpoint manifest | `format` / `format_version` | `loudkit-checkpoint` / `1`, `2` |
| identity contract | `identity_contract_version` | `1` |
| provenance manifest | assertion label | `loudkit.provenance` |

In 0.1.1, `Result.save()` writes the provenance manifest of a WAV in an
`LKPV` RIFF chunk. Server WAV replies append it after the `data` chunk, which
is the layout of 0.1.0. The 0.1.1 reader reads both layouts. The 0.1.0 reader
raises `ProvenanceError` on a WAV that `Result.save()` wrote in 0.1.1. The
JSON inside, its assertion label and its fields are the same in both
layouts.

The error-code catalog follows the same rules. These nine codes are the
vocabulary of the error classes, and every transport reports the same code for
the same condition: `audio_not_found`, `cancelled`, `invalid_request`,
`invalid_tokens`, `number_grammar`, `provenance_invalid`,
`unsupported_language`, `voice_not_found`, `window_overflow`.

A transport adds its own codes for conditions the library never sees. It
reports `server_fault` for an exception it did not classify as a refusal. See
[errors](errors.md) for those.

A new code is additive and can come in a minor release. Branch on the codes
you know, and treat an unknown code as a refusal. Do not treat it as a parse
failure. Removing or renaming a code is breaking.

A checkpoint's `format_version` names the decode loop its weights need, so the
table lists two values. Version 1 is the one-token loop of loudr-1. Version 2
decodes two speech tokens per transformer forward
([why](../design/two-token-decode.md)). All five implementations read both
versions and dispatch on the manifest's decode mode. An engine without the
two-token loop refuses version 2.

`recipe_version` is not in this table, because it names the algorithm. The only
recipe is `loudkit-1`, and a manifest that declares any other recipe is
refused. The recipe name does not change on an audible change: the fingerprint
moves instead.

## Additive versus breaking, per surface

These rules say which changes are breaking. From 1.0, a breaking change needs a
major release. Before 1.0, see [Before 1.0](#before-10).

Python API:

- Additive: a keyword argument with a default, a field on `Result`, a class,
  or an exception subclass.
- Every exception raised from `loudkit.errors` also inherits the builtin it
  replaces, so an existing `except ValueError` keeps working. The base
  `LoudkitError` inherits `Exception` only and is never raised directly.
- Breaking: removing a name, making an optional argument required, or changing
  what a default means.

The four ports: Swift, Go, Rust and TypeScript version together with Python,
under one tag. Shared conformance fixtures hold them to the behaviour of
Python on the fixture cases. A port that adds a function Python already has is
additive. A port that behaves differently from Python on a shared, supported
behaviour has a bug. [SUPPORTED.md](../../SUPPORTED.md) lists the features
that only Python has.

HTTP routes:

- Routes are under `/v1`, except `/health`, which is unversioned.
- Additive: new routes, new request fields with defaults, new response fields
  and new headers.
- Breaking: removing a field, changing the status code for an unchanged
  condition, or changing the meaning of an existing field. From 1.0 such a
  change goes to `/v2`, and `/v1` keeps working under the deprecation rule
  below.

The SSE stream: read until the `done` event, and ignore event types you do not
know. A new event type is additive when the existing events keep their meaning.
`done` always carries `truncated`, and `error` carries `error_kind`.

The manifest and profile formats: a new key in a checkpoint manifest, a
provenance manifest or a `VoiceProfile` is additive. Readers ignore keys they
do not know, and a missing key keeps its documented default. Removing a key or
giving it a new meaning is breaking.

## Deprecation

From 1.0, nothing is removed in a minor release. An API that is going away
keeps working and warns: a `DeprecationWarning` in Python, a documented note
for the ports and the routes. The warning stays for at least one minor release
before a major release removes the API.

## What is explicitly not promised

- Bit-identical audio across backends, devices, releases or fingerprints.
  Different hardware adds floating-point numbers in different orders. The
  identity contract lists what holds.
- Reproduction of the published sample audio. The SHA-256 of every sample in
  `docs/voices/roster/provenance.json` identifies the published file. The
  same voice, seed and text on another machine can give different bytes.
- Stability of names with a leading underscore, or of modules not named here.
  `loudkit.models.*` and `loudkit.backends.*` are implementation.

  `loudkit.__all__` holds the names most callers need: the verbs, `Engine`,
  `Result`, `VoiceProfile`, the speed bounds and the error classes. Seven more
  modules have the same guarantees: `loudkit.config`, `loudkit.contracts`,
  `loudkit.errors`, `loudkit.postprocess`, `loudkit.sampler`,
  `loudkit.timing` and `loudkit.provenance`.
- Stability of `tools/` and `research/`. They have no stability guarantee.
- Wall-clock performance. A release can be slower than the previous one on
  your hardware.

## Conditions on two conformance bands

Two conformance bands have conditions that are part of the contract:

- `top-1 >= 99%` on the token generator holds with the batch shape and the
  kernel path fixed. A bf16 matmul reduces differently at a different batch
  size, so results from different batch shapes are not comparable against this
  band.
- `median KL < 1e-4` is defined over the full logit distributions, never over
  sampled counts. An empirical KL from *n* samples has a bias of about
  `(V-1)/2n` nats, which at this vocabulary size exceeds the band by orders of
  magnitude.
