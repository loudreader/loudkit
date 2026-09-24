# Provenance

`Result.save()` adds a loudkit provenance manifest to every WAV by default.
Pass `include_provenance=False` to leave it out.

The manifest is not [C2PA](https://c2pa.org). A C2PA manifest is a signed
manifest store. The loudkit manifest is one unsigned JSON document in boxes
with the JUMBF layout, in a RIFF chunk that C2PA tools do not read. C2PA tools
skip the file. `loudkit verify` reads it.

## What it is

The JSON document is in JUMBF-shaped boxes, in a RIFF chunk named `LKPV` after
the audio. A player skips a chunk it does not know, so the audio plays as
usual. The `data` chunk is byte-identical to the same synthesis saved without a
manifest. Adding the manifest appends the chunk and updates the size in the
RIFF header. Nothing else in the file changes.

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
voice = engine.voice("joe")

r = engine.synthesize("Hello from loudkit.", voice, seed=7)
r.save("hello.wav")  # manifest included
r.save("bare.wav", include_provenance=False)  # audio only
```

WAV audio from the server carries the same boxes after the `data` chunk,
outside the RIFF chunks. This is also the layout of files that loudkit 0.1.0
saved. `read_provenance` reads both layouts.

## What it says

The document holds two assertions. The first uses the field layout of the C2PA
`c2pa.actions` assertion. It carries the IPTC `digitalSourceType` value for
media made by a model, the name and version of loudkit, and a timestamp. These
field names do not make the file a C2PA file. The second assertion is
loudkit's own:

| field | content |
| --- | --- |
| `algorithm_fingerprint` | the algorithm that made the audio: the 16 hex digits the engine reports, on which the five implementations agree |
| `recipe_version` | the checkpoint's recipe |
| `seed` | the seed |
| `sample_rate`, `speed` | how the audio was rendered |
| `voice`, `language` | labels, when the caller passed them |
| `checkpoint_sha256` | the weights: the digest that the release's `SHA256SUMS` lists. Two checkpoints can have the same fingerprint, so the manifest also names the file |
| `voice_profile_sha256` | the digest of the voice profile file. The voice name is only a label. Empty when no file digest is known for the profile |
| `backend`, `execution` | the backend (`torch`, `onnx` or `coreml`), the device placement and the precision per module. Execution settings can change the output; see [the identity contract](IDENTITY-CONTRACT.md) |
| `audio_sha256` | the SHA-256 of the audio the manifest is bound to |
| `text_sha256` | the SHA-256 of the text. The text itself is not stored |

With the identity contract, the manifest describes how a file was made. To
reproduce the audio and check `audio_sha256`, you need all of these:

- the same checkpoint, voice profile, fingerprint, seed and backend
- the original text, because the manifest stores only its hash
- the same build, device and execution configuration

`text_sha256` is a hash of the text, and the text is not stored in the
manifest. A hash can still confirm a guess: for a short or predictable
utterance, someone with a list of candidates can test each one. If that
matters for your use, pass `include_provenance=False`.

## Reading it back

```python
from loudkit.provenance import read_provenance, verify_provenance

info = read_provenance("hello.wav")  # the manifest, or None
manifest, ok = verify_provenance("hello.wav")  # does audio_sha256 still match?
```

`verify_provenance` hashes the audio in the `data` chunk again and compares the
result with `audio_sha256`. It detects a manifest moved onto different audio.
Re-encoding usually changes the samples, so it fails verification, or it
removes the manifest.

## Trust model

The manifest is unsigned. C2PA signs a manifest with a certificate, so a
verifier can tell who made the claim and that nobody changed it. This manifest
has no signature, so anyone can write, change or remove it. Use it as a
disclosure only. It is not proof of origin.

Converting to MP3, editing in an audio tool, or uploading through a service
that rewrites containers usually removes the manifest. The manifest is file
metadata. It is not a mark in the audio signal.

A one-shot HTTP reply from the server carries the manifest JSON in the
`X-Loudkit-Provenance` header. See
[the server guide](../guides/04-server-and-agents.md).
