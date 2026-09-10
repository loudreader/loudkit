# Compatibility

What may change between releases, and what may not. This page covers the
surfaces you call. For what the *output* is guaranteed to be, read
[the identity contract](IDENTITY-CONTRACT.md).

## The fingerprint

Every engine computes an **algorithm fingerprint**: a hash of every decision
that changes what a listener hears, from the text rules to the sampler to the
detectors, with the checkpoint's own settings folded in. The five
implementations compute it independently and agree, `engine.describe()`
prints it, and a saved WAV records it. Give an engine reporting fingerprint
*F* the same text, voice and seed on the same build, device and execution
config, and you get a bit-identical waveform, today and in any future release
whose fingerprint is still *F*. An audible change moves the fingerprint, so it
cannot be quiet. Execution choices (`cuda_graphs`, the ragged vocoder, the
ONNX provider) sit outside it: they can change the last bits of the waveform
without changing what is computed, which the identity contract calls
`equivalent` and measures.

## Versioning

Semantic versioning over the API surfaces, with the fingerprint as the extra
axis. From 1.0 onward (the `0.x` clause is below):

| change | version effect | fingerprint |
| --- | --- | --- |
| new function, parameter with a default, response field, route | minor | unchanged |
| an audible algorithm change (sampling, windowing, joins, detectors) | **minor at least** (`0.x`: see below) | **moves** |
| removing or renaming anything public; changing a type or a default's meaning | **major** | may move |
| a bug fix that does not change output | patch | unchanged |
| faster, less memory, a new backend, better errors | patch or minor | unchanged |

From 1.0, an audible change is never a patch, even when it is an improvement.

### Before 1.0, the fingerprint is the audible contract

While the version is `0.x`, the patch component tracks the **API surface** and
the **fingerprint** carries the audible one, because semantic versioning gives
`0.x` no stable public surface to promise against. A `0.x` patch release may
move the fingerprint, and when it does it must lead its changelog entry with
the move, old value to new, name every amendment that caused it with the
measurement, and keep the old value discoverable. 0.1.1 is the first release
to use this clause: it moves `79f71f5821477353` to `7cd75498ad4e7531`. The
clause expires at 1.0, when the table above becomes the whole rule.

## Pinning a release

Without a revision, `load` resolves the repository's default branch, so the
same code can return different weights later. In production, pin one:

```python
engine = lk.load("loudreader/loudr-1", revision="a1b2c3d")
```

Every command that takes a repo id takes `--revision` for the same reason:

```bash
loudkit serve --checkpoint loudreader/loudr-1 --revision a1b2c3d
```

Every download is checked against the release's `SHA256SUMS` before it is
used, which proves the files arrived intact. It does not prove which release
they are; the revision does.

A directory a download wrote (`loudkit download --local-dir`, or a port's
`download`) carries `.loudkit-release.json`: the repo, the revision asked
for, the commit it resolved to, the digest of `SHA256SUMS` and when it was
fetched. A receipt is valid when every field is present with its type, the
repo is the one asked for, the commit is forty lowercase hex, the digest is
that of the `SHA256SUMS` on disk, lowercase hex, and every file it lists
that the download would select is present. No weight is hashed. A load
whose revision still resolves to the commit a valid receipt names hashes
nothing; any other commit, an invalid receipt or none fetches `SHA256SUMS`
again and keeps what still hashes to it. A load that cannot reach the hub
uses a valid receipt and says so on stderr, and errors without one.

The receipt is trust on first use. It says the directory is what was
verified when it was written, not that it is what the hub publishes now: a
`SHA256SUMS` rewritten on disk with the digest recomputed into the receipt
passes offline, and only the commit check against the hub catches it. A
file edited in place under a matching receipt is not looked at until the
commit moves. `revision` is what you ask for; `commit` is what the receipt
holds you to.

Python's own cache follows the same rule: a load by repo id asks the hub
once per process what `main` (or the tag) resolves to and reads the cached
snapshot only when it is that commit, using the cache offline with one line
on stderr. A revision that is a commit asks nothing.

## Frozen formats

Four on-disk and on-the-wire formats are frozen at 0.1.0. Frozen means a reader
written against them keeps working: fields may be added, and nothing existing is
renamed, removed or given a new meaning without a major version.

| format | pinned as | value |
| --- | --- | --- |
| voice profile | `VOICE_FORMAT_VERSION` | `1` |
| checkpoint manifest | `format` / `format_version` | `loudkit-checkpoint` / `1`, `2` |
| identity contract | `identity_contract_version` | `1` |
| provenance manifest | assertion label | `loudkit.provenance` |

The error-code catalog is frozen with them. These nine are the vocabulary the
error classes carry, and every transport reports the same word for the same
condition: `audio_not_found`, `cancelled`, `invalid_request`,
`invalid_tokens`, `number_grammar`, `provenance_invalid`,
`unsupported_language`, `voice_not_found`, `window_overflow`.

A transport adds codes of its own for conditions the library never sees, and
reports `server_fault` for an exception it did not classify as a refusal. See
[errors](errors.md) for those.

A new code is additive and may arrive in a minor release, so branch on the codes
you know and treat an unknown one as a refusal you did not anticipate rather
than as a failure to parse. Removing or renaming one is breaking.

A checkpoint's `format_version` says which decode loop its weights need, so it
is the one entry that is a list rather than a single number. Version 1 is the
one-token loop every loudr-1 checkpoint uses. Version 2 decodes two speech
tokens per transformer forward ([why](../design/two-token-decode.md)), and an
engine that has not implemented that loop refuses the file rather than running
the loop it knows over weights that mean something else. All five engines
read both versions and dispatch on the manifest's decode mode; an older
reader written against version 1 alone refuses version 2 by name, which is
what the freeze promises.

`recipe_version` is not in this table. It names the algorithm, not a format.
`loudkit-1` is the only recipe, and a manifest that declares any other one is
refused. The fingerprint moves on its own, so an audible change moves the
fingerprint while `recipe_version` stands still.

## Additive versus breaking, per surface

**Python API.** Adding a keyword argument with a default, a field to `Result`, a
class, or an exception subclass is additive. Every raised exception in
`loudkit.errors` also inherits the builtin it replaces, so an existing
`except ValueError` keeps working. That is a compatibility guarantee, not an
implementation detail. The base `LoudkitError` inherits `Exception` alone and
is never raised directly. Removing a name, making an optional argument required,
or changing what a default means is breaking.

**The four ports.** Shared conformance fixtures hold Swift, Go, Rust and
TypeScript to the same behaviour as Python. They version together: one tag, one
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

**The manifest and profile formats.** A new key in a checkpoint manifest, a provenance
manifest, or a `VoiceProfile` is additive. Readers ignore what they do not know,
and a missing key keeps its documented default. Removing a key, or repurposing
one, is breaking. Removal happens before a format is released or not at all.

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
  named below. `loudkit.models.*` and `loudkit.backends.*` are implementation.

  `loudkit.__all__` is the set of names most callers need: the verbs, `Engine`,
  `Result`, `VoiceProfile`, the speed bounds and the error classes. Seven more
  modules are supported homes and keep the guarantees above: `loudkit.config`,
  `loudkit.contracts`, `loudkit.errors`, `loudkit.postprocess`,
  `loudkit.sampler`, `loudkit.timing` and `loudkit.provenance`.
* **Stability of `tools/` and `research/`.** Those are the project's own
  scripts, not products.
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
  size, and the divergence is real. Comparing across batch shapes is a
  different experiment, not a failed gate.
- **`median KL < 1e-4`** is defined over the full logit distributions, never
  over sampled counts. An empirical KL from *n* samples carries roughly
  `(V-1)/2n` nats of bias, which at this vocabulary size exceeds the band by
  orders of magnitude.
