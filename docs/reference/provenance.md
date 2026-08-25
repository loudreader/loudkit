# Provenance

**Every WAV `Result.save()` writes carries a C2PA manifest, by default.** If you
copied the three-line example from the README, your file has one. This page says
what is in it, how it is verified, and how to switch it off.

## What it is

A [C2PA](https://c2pa.org) *claim-only* manifest: a JSON document in a JUMBF box
appended after the WAV's data chunk. The box sits **outside the size the RIFF
header declares**, so every player that reads the declared length ignores it and
plays the audio unchanged. The bytes before the box are byte-identical to the
same synthesis saved without one.

```python
r = engine.synthesize("Hello from loudkit.", narrator, seed=7)
r.save("hello.wav")  # manifest included
r.save("bare.wav", include_provenance=False)  # audio only
```

## What it says

Two assertions. The first is the standard `c2pa.actions` one: this audio was
*created* by software, naming loudkit and its version, with a timestamp. The
second is loudkit's own:

| field | what it pins |
| --- | --- |
| `algorithm_fingerprint` | which algorithm produced it: the same 16 hex digits the engine reports and the five implementations agree on |
| `recipe_version` | the checkpoint's recipe |
| `seed` | the seed, so the render is repeatable |
| `sample_rate`, `speed` | how it was rendered |
| `voice`, `language` | labels, when the caller passed them |
| `checkpoint_sha256` | which weights spoke: the digest a release's `SHA256SUMS` lists. The fingerprint pins the algorithm; two checkpoints can share one, so the manifest names the file |
| `voice_profile_sha256` | which profile bytes voiced it. A voice *name* is a label anyone can reuse; the digest is not. Empty when the profile never touched disk |
| `backend`, `execution` | the datapath: `torch`/`onnx`/`coreml`, device placement and per-module precision. Execution never changes what is computed, but reduced precision perturbs it within measured bands |
| `audio_sha256` | the audio the manifest is bound to |
| `text_sha256` | a **hash** of the text, never the text itself |

Together with the identity contract, that makes a saved file self-describing.
Given the same checkpoint, profile, fingerprint, seed and backend you can
reproduce the audio and check `audio_sha256` yourself. No hosted service can
offer that property.

**On `text_sha256`.** The text is hashed, not stored, so sharing a file does not
disclose what you typed. A hash still confirms a guess. For a short or
predictable utterance, someone with a candidate list can test it. If that matters
for your use, pass `include_provenance=False`.

## Reading it back

```python
import loudkit as lk

info = lk.read_provenance("hello.wav")  # the manifest, or None
manifest, ok = lk.verify_provenance("hello.wav")  # does audio_sha256 still match?
```

`verify_provenance` re-hashes the audio and compares. It catches a manifest
transplanted onto different audio. Re-encoding changes the samples, so it fails
verification too.

## Trust model

**It is unsigned.** A full C2PA chain signs the manifest with a certificate, so a
verifier can tell who made the claim and that nobody edited it. This one carries
no signature: anyone can write, alter, or strip it. Treat it as **disclosure, not
proof**. It tells an honest downstream tool where a file came from, and it stops
nobody who does not want to be told on.

It is also fragile in the ordinary sense: converting to MP3, editing in an audio
tool, or re-uploading through a service that rewrites containers will usually
drop the box. Metadata travels with a file, not with the sound.

The server attaches the same manifest to its replies (`X-Loudkit-Provenance`, and
the box itself on the audio body). See
[the server guide](../guides/04-server-and-agents.md).
