---
license: apache-2.0
library_name: loudkit
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - voice-cloning
  - on-device
language:
  - en
  - es
  - de
  - pt
  - fr
  - it
  - pl
  - nl
  - sv
  - da
---

<p align="center">
  <img src="https://huggingface.co/loudreader/loudr-1-turbo/resolve/main/logo.png" alt="LoudKit" width="640">
</p>

# loudr-1-turbo

**The faster of the two loudkit models. Same 28 voices, same ten
languages, same API.**

loudr-1-turbo is [loudr-1](https://huggingface.co/loudreader/loudr-1) with two
changes to how the audio is produced, and no change to how it is used:

- the token generator writes **two speech tokens per forward** instead of one,
  so a second of audio costs half as many;
- the renderer turns those tokens into audio in **one** pass instead of
  several.

Everything else is loudr-1: the same voices, the same text handling, the same
sampling, the same seeds.

## Which one to pick

**loudr-1** is the default. Pick it when you want the reference quality.

**loudr-1-turbo** is faster on the same hardware, at a small cost in
naturalness that is easiest to hear on long, quiet or heavily punctuated
passages. Pick it for interactive reading, where latency is what the listener
notices. Listen to both on your own text before choosing, and measure both on
your own hardware; the
[benchmark page](https://loudreader.github.io/loudkit/benchmarks/) has the
commands.

## Start in Python

```bash
pip install "loudkit[torch,audio,hub]"
```

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1-turbo")
voice = engine.voice("joe")

engine.synthesize("Hello from loudkit.", voice, seed=7).save("hello.wav")
```

That is the whole difference from loudr-1: one string. The same from a shell:

```bash
loudkit speak --checkpoint loudreader/loudr-1-turbo --voice joe \
  "Hello from loudkit." -o hello.wav
```

To enroll a voice that you own or have permission to use:

```python
mine = lk.enroll("my-recording.wav", "loudreader/loudr-1-turbo", name="my-voice")
mine.save("voices/my-voice.safetensors")
```

A voice profile is interchangeable between the two models: a profile enrolled
against loudr-1 loads against loudr-1-turbo and the other way round.

## Backends

Version 0.1.1 supports both models in Python, Swift, Go, Rust and TypeScript.
Python offers PyTorch, ONNX Runtime and CoreML; Swift uses its native token
generator with CoreML rendering, and Go, Rust and TypeScript use ONNX Runtime.
Changing the model name keeps the same synthesis API and voice profile.

## Download

```bash
loudkit download loudreader/loudr-1-turbo --for onnx
```

Choose `--for torch` or `--for coreml` for another backend. Add
`--with-cloning` to prepare enrollment as well. Loading an existing voice
profile does not require enrollment assets. Prepare the chosen set once to
use it offline.

## What ships

| artefact | used by |
|---|---|
| `loudr-1-turbo.safetensors` | synthesis |
| `loudr-1-enrollment.safetensors` and `ve.safetensors` | shared PyTorch enrollment |
| `onnx/` | synthesis and enrollment through ONNX Runtime |
| `coreml/` | synthesis and enrollment through CoreML |
| `voices/` | 28 portable voice profiles |
| `tokenizer.json` | text processing |
| `samples/` | audio generated with this model |

The bundle uses the same canonical enrollment weights and graphs as loudr-1.
Clone once, then use the unchanged profile with either model. Exact file sizes
and checksums are recorded in the bundle's `release.json` and `SHA256SUMS`.

## Listen

These samples use this model, the named shipped voice and seed 7.

<audio controls src="https://huggingface.co/loudreader/loudr-1-turbo/resolve/main/samples/joe.opus"></audio>

<audio controls src="https://huggingface.co/loudreader/loudr-1-turbo/resolve/main/samples/kathleen.opus"></audio>

## Voices and consent

The 28 profiles are the ones loudr-1 ships, unchanged: ten for English and
two each for Spanish, French, German, Italian, Polish, Portuguese, Dutch, Swedish
and Danish, built from recordings donated for speech technology or from CC0
and CC-BY speech corpora. No scraped celebrity voices ship with the model.
[The full roster](https://github.com/loudreader/loudkit/blob/main/VOICES.md)
records the source, licence and consent basis for every profile.

## Model lineage

loudr-1-turbo is derived from loudr-1, which is derived from
[Chatterbox](https://github.com/resemble-ai/chatterbox), released by Resemble
AI under the MIT licence. The token generator is a student of loudr-1's,
trained to emit two tokens per forward; the renderer is loudr-1's, distilled
to reach the same audio in one pass.
The licence chain is loudr-1's, unchanged, and is recorded in
[NOTICE](https://huggingface.co/loudreader/loudr-1-turbo/blob/main/NOTICE).

How the two-token generator works is written up in
[two-token decode](https://github.com/loudreader/loudkit/blob/main/docs/design/two-token-decode.md).

## Reproducibility

For a fixed build, device and backend, the same text, voice and seed produce
the same waveform. loudr-1 and loudr-1-turbo are **different models**: the
same text, voice and seed give different audio on each, and a saved WAV
records which one spoke. The contract is in the
[identity contract](https://github.com/loudreader/loudkit/blob/main/docs/reference/IDENTITY-CONTRACT.md).

## Before you ship

- Long passages are rendered in windows of about ten seconds. Sentence joins
  can occasionally be audible, and turbo joins a little more so than loudr-1.
- Difficult punctuation, numbers and abbreviations can change pronunciation or
  prosody.
- Voice cloning requires consent. A recording being public does not grant
  permission to clone the speaker.
- Saved WAVs and server responses carry unsigned C2PA Content Credentials by
  default, recording the model, voice, seed, backend and a hash of the audio
  in a machine-readable form.

Read [Responsible use](https://huggingface.co/loudreader/loudr-1-turbo/blob/main/RESPONSIBLE_USE.md)
before exposing enrollment to other people.

## Intended use

loudr-1-turbo is intended for local narration, accessibility, localisation,
games, prototyping and speech research. It is not a voice-authentication
system and must not be used for deceptive impersonation.

## Training data

The original Chatterbox training data is controlled by Resemble AI and is not
documented by this project. The shipped voice profiles use recordings made or
released for speech-technology use; their sources and licences are listed in
the public roster.

## Licence

[Apache-2.0](https://huggingface.co/loudreader/loudr-1-turbo/blob/main/LICENSE).
Upstream attributions and component licences are listed in
[NOTICE](https://huggingface.co/loudreader/loudr-1-turbo/blob/main/NOTICE).

## Download sizes for 0.1.1

Approximate decimal sizes for the release files; backend weights are included.
Cloning adds enrollment assets only when requested. Both models ship separate
synthesis and enrollment checkpoints.

| Model | Torch | Torch + cloning | ONNX | ONNX + cloning | CoreML | CoreML + cloning |
|---|---:|---:|---:|---:|---:|---:|
| loudr-1 | 0.75 GB | 1.28 GB | 2.60 GB | 3.13 GB | 2.46 GB | 2.99 GB |
| loudr-1-turbo | 0.72 GB | 1.25 GB | 2.44 GB | 2.97 GB | 2.44 GB | 2.97 GB |
