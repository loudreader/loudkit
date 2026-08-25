# Compatibility

What may change between releases, and what may not. This page covers the
surfaces you call. For what the *output* is guaranteed to be, read
[the identity contract](IDENTITY-CONTRACT.md) first.

## Compatibility promise

**The same fingerprint means the same bytes, forever.** Give an engine
reporting `algorithm_fingerprint` *F* the same text, voice and seed on the same
build and device, and you get a bit-identical waveform. That holds today, and
for any future release whose fingerprint is still *F*.

The fingerprint is a hash of every audible decision, not a version number we
bump. The five implementations compute it independently, and the engine refuses
to start if two of its parts disagree about it.

An audible change therefore cannot be quiet. It moves the fingerprint, and a
moved fingerprint is a visible, checkable event. A hosted service can silently
re-tune a model between your two requests. This cannot.

## Versioning

Semantic versioning over the API surfaces, with the fingerprint as the extra
axis:

| change | version effect | fingerprint |
| --- | --- | --- |
| new function, parameter with a default, response field, route | minor | unchanged |
| an audible algorithm change (sampling, windowing, joins, detectors) | **minor at least** | **moves** |
| removing or renaming anything public; changing a type or a default's meaning | **major** | may move |
| a bug fix that does not change output | patch | unchanged |
| faster, less memory, a new backend, better errors | patch or minor | unchanged |

An audible change is never a patch, even when it is an improvement. If your
build's fingerprint changed, your audio changed, and the version must tell you
so before a listener does.

## Frozen formats

Four on-disk and on-the-wire formats are frozen at 0.1.0. Frozen means a reader
written against them keeps working: fields may be added, and nothing existing is
renamed, removed or given a new meaning without a major version.

| format | pinned as | value |
| --- | --- | --- |
| voice profile | `VOICE_FORMAT_VERSION` | `1` |
| checkpoint manifest | `format` / `format_version` | `loudkit-checkpoint` / `1` |
| identity contract | `identity_contract_version` | `1` |
| provenance manifest | C2PA label | `loudkit.provenance` |

The error-code catalog is frozen with them. These seven are the vocabulary the
error classes carry, and every transport reports the same word for the same
condition: `invalid_request`, `invalid_tokens`, `number_grammar`,
`provenance_invalid`, `unsupported_language`, `voice_not_found`,
`window_overflow`.

A transport adds codes of its own for conditions the library never sees, and
reports `server_fault` for an exception it did not classify as a refusal. See
[errors](errors.md) for those.

A new code is additive and may arrive in a minor release, so branch on the codes
you know and treat an unknown one as a refusal you did not anticipate rather
than as a failure to parse. Removing or renaming one is breaking.

`recipe_version` is not in this table. It names the algorithm, not a format.
`loudkit-1` is the only recipe, and a manifest that declares any other one is
refused. The fingerprint moves on its own: it hashes every setting that changes
what a listener hears, so an audible change moves the fingerprint while
`recipe_version` stands still. That is what
[the identity contract](IDENTITY-CONTRACT.md) is for.

## Additive versus breaking, per surface

**Python API.** Adding a keyword argument with a default, a field to `Result`, a
class, or an exception subclass is additive. Every raised exception in
`loudkit.errors` also inherits the builtin it replaces, so an existing
`except ValueError` keeps working. That is a compatibility guarantee, not an
implementation detail. The base `LoudkitError` inherits `Exception` alone and
is never raised directly. Removing a name, making an optional argument required,
or changing what a default means is breaking.

**The four ports.** Shared conformance fixtures hold Go, Rust, TypeScript and
Swift to the same behaviour as Python. They version together: one tag, one
fingerprint, five implementations. A port gaining a function Python already had
is additive. A port *diverging* in behaviour is a bug, not a version event.

**HTTP routes.** Under `/v1`, except `/health`, which is deliberately
unversioned. New routes, new request fields with defaults, new
response fields and new headers are additive. Removing a field, changing a
status code for an unchanged condition, or altering the meaning of an existing
field requires `/v2`, and `/v1` keeps working per the deprecation rule below.

**The SSE stream.** Read until the `done` event. Anything else that appears
alongside the events you know is additive by construction. `done` always
carries `truncated`, and `error` carries `error_kind` when it failed.

**The manifest and profile formats.** A new key in a checkpoint manifest, a C2PA
manifest, or a `VoiceProfile` is additive. Readers ignore what they do not know,
and a missing key keeps its documented default. Removing a key, or repurposing
one, is breaking. Removal happens before a format is released or not at all.
`VoiceProfile` dropped its dead `emotion` key while the format was pre-release,
and readers still accept files that carry it.

## Deprecation

Nothing is removed in a minor release. When something must go, it keeps working
and starts saying so: a `DeprecationWarning` in Python, a documented note for
the ports and the routes. That warning runs for at least one minor release
before a major one removes it. An upgrade must never be the thing that tells you
your code was wrong.

## What is explicitly not promised

* **Bit-identical audio across backends, devices or fingerprints.** Different
  hardware sums floating point in different orders. The sampling law and the
  voice are what hold constant; see the identity contract.
* **Reproduction of the published sample audio.** The sha256 of every sample in
  `docs/voices/roster/provenance.json` identifies the bytes that are published,
  and nothing else. The same voice, seed and text on your machine produce the
  same voice and different bytes.
* **Stability of anything under a leading underscore**, or of modules not
  re-exported from the top-level package. `loudkit.models.*` and
  `loudkit.backends.*` are implementation.
* **Stability of `tools/`.** Those are the project's own scripts, not products.
* **Wall-clock performance.** Faster is a patch. No release promises to be as
  fast as the last one on your hardware.

## Pre-1.0

While the version is `0.x`, a minor bump may break something. That is what `0.x`
means, and it is the window in which the shapes above get their last
corrections. The fingerprint promise holds regardless. It covers output rather
than API, and it has held since the first release.

## The bands' fine print

Two conformance bands carry conditions that are part of the contract:

- **`top-1 >= 99%`** on the token generator holds with the batch shape and
  kernel path pinned. A bf16 matmul reduces differently at a different batch
  size, and the divergence is real. Measured elsewhere at several percent of
  tokens flipped by batch shape alone. Comparing across batch shapes is a
  different experiment, not a failed gate.
- **`median KL < 1e-4`** is defined over the full logit distributions, never
  over sampled counts. An empirical KL from *n* samples carries roughly
  `(V-1)/2n` nats of estimator bias, which at this vocabulary size exceeds
  the band by orders of magnitude.
