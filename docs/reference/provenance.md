# Provenance

**Every WAV `Result.save()` writes carries a loudkit provenance manifest, by
default.** If you copied the three-line example from the README, your file has
one. This page says what is in it, how it is verified, and how to switch it off.

**It is not [C2PA](https://c2pa.org).** A C2PA manifest is a signed manifest
store, and this is one JSON assertion in boxes that borrow JUMBF's shape and
nothing else. C2PA tools do not read it, and it does not sit in the `C2PA` chunk
they look in, so they pass over the file rather than reporting a broken
manifest. `loudkit verify` is what reads it.

## What it is

A JSON document in JUMBF-shaped boxes, carried in a RIFF chunk named `LKPV`
after the audio. A player walks the chunks it knows and ignores the one it does
not, so the audio plays unchanged, and **the `data` chunk is byte-identical to
the same synthesis saved without a manifest**: adding provenance appends a chunk
and grows the size the RIFF header declares, and touches nothing else.

```python
r = engine.synthesize("Hello from loudkit.", narrator, seed=7)
r.save("hello.wav")  # manifest included
r.save("bare.wav", include_provenance=False)  # audio only
```

## What it says

Two assertions. The first follows the shape C2PA uses for `c2pa.actions`, and
carries IPTC's `digitalSourceType` term for media a model made: this audio was
*created* by software, naming loudkit and its version, with a timestamp. The
shape is borrowed so that a future move to real Content Credentials is
mechanical; it is not a claim that this file is one. The second is loudkit's
own:

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
from loudkit.provenance import read_provenance, verify_provenance

info = read_provenance("hello.wav")  # the manifest, or None
manifest, ok = verify_provenance("hello.wav")  # does audio_sha256 still match?
```

`verify_provenance` re-hashes the audio and compares. It catches a manifest
transplanted onto different audio. Re-encoding changes the samples, so it fails
verification too.

## Trust model

**It is unsigned.** C2PA signs a manifest with a certificate, so a verifier can
tell who made the claim and that nobody edited it. This one carries no
signature, which is most of why it is not C2PA: anyone can write, alter, or
strip it. Treat it as **disclosure, not
proof**. It tells an honest downstream tool where a file came from, and it stops
nobody who does not want to be told on.

It is also fragile in the ordinary sense: converting to MP3, editing in an audio
tool, or re-uploading through a service that rewrites containers will usually
drop the box. Metadata travels with a file, not with the sound.

The server attaches the same manifest to its replies (`X-Loudkit-Provenance`, and
the box itself on the audio body). See
[the server guide](../guides/04-server-and-agents.md).
