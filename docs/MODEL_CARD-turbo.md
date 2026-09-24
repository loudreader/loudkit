---
license: apache-2.0
library_name: loudkit
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - voice-cloning
  - on-device
  - coreml
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
  <img src="https://huggingface.co/loudreader/loudr-1-turbo/resolve/main/logo.png" alt="loudkit" width="640">
</p>

# loudr-1-turbo

**The faster of the two loudkit models, with the same 28 voices, ten languages
and API as loudr-1.**

loudr-1-turbo produces audio differently from
[loudr-1](https://huggingface.co/loudreader/loudr-1):

- a smaller token generator writes two speech tokens per forward pass instead
  of one, so a second of audio needs half as many forward passes;
- the renderer turns the tokens into audio in one step instead of two.

The voices, the text handling, the sampling algorithm and the seeds work as in
loudr-1. The weights differ, so the same input gives different audio.

## Which one to pick

**loudr-1** is the default.

**loudr-1-turbo** is faster on the same hardware. Its audio is slightly less
natural, most audibly on long, quiet or heavily punctuated passages. Use it
when time to first audio matters. Listen to both on your own text, and measure
both on your own hardware. The
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

Only the repo id differs from loudr-1. The same from a shell:

```bash
loudkit speak --checkpoint loudreader/loudr-1-turbo --voice joe \
  "Hello from loudkit." -o hello.wav
```

To enroll a voice that you own or have permission to use:

```python
mine = lk.enroll("my-recording.wav", "loudreader/loudr-1-turbo", name="my-voice")
mine.save("voices/my-voice.safetensors")
```

A voice profile enrolled with either model works with both models.

## Backends

Version 0.1.1 supports both models in Python, Swift, Go, Rust and TypeScript.
Python offers PyTorch, ONNX Runtime and CoreML; Swift uses its native token
generator with CoreML rendering, and Go, Rust and TypeScript use ONNX Runtime.

## Download

```bash
loudkit download loudreader/loudr-1-turbo --for onnx
```

Choose `--for torch` or `--for coreml` for another backend. Add
`--with-cloning` to prepare enrollment as well. Loading an existing voice
profile does not require enrollment assets. Prepare the chosen set once to
use it offline.

Approximate decimal download sizes for 0.1.1, with the backend weights
included:

| Model | Torch | Torch + cloning | ONNX | ONNX + cloning | CoreML | CoreML + cloning |
|---|---:|---:|---:|---:|---:|---:|
| loudr-1 | 0.75 GB | 1.28 GB | 2.60 GB | 3.13 GB | 2.46 GB | 2.99 GB |
| loudr-1-turbo | 0.72 GB | 1.25 GB | 2.44 GB | 2.97 GB | 2.44 GB | 2.97 GB |

Both models ship separate synthesis and enrollment checkpoints. The enrollment
files come only with `--with-cloning`.

## What ships

| artefact | used by |
|---|---|
| `loudr-1-turbo.safetensors` | synthesis |
| `loudr-1-enrollment.safetensors` and `ve.safetensors` | PyTorch enrollment |
| `onnx/` | synthesis and enrollment through ONNX Runtime |
| `coreml/` | synthesis and enrollment through CoreML |
| `voices/` | 28 portable voice profiles |
| `tokenizer.json` | text processing |
| `samples/` | audio generated with this model |

Each bundle carries its own enrollment checkpoint. Use the enrollment files
from the same bundle as the synthesis checkpoint. Exact file sizes and
checksums are in the bundle's `release.json` and `SHA256SUMS`.

## Listen

These samples use this model and seed 7.

**Joe**

<audio controls src="https://huggingface.co/loudreader/loudr-1-turbo/resolve/main/samples/joe.opus"></audio>

**Kathleen**

<audio controls src="https://huggingface.co/loudreader/loudr-1-turbo/resolve/main/samples/kathleen.opus"></audio>

## Voices and consent

The 28 profiles are the same files loudr-1 ships: ten for English and two
each for Spanish, French, German, Italian, Polish, Portuguese, Dutch, Swedish
and Danish. They were built from recordings donated for speech technology or
from CC0 and CC-BY speech corpora. No audio without a stated licence ships
with the model.
[The full roster](https://github.com/loudreader/loudkit/blob/main/VOICES.md)
records the source, licence and consent basis for every profile.

## Model lineage

loudr-1-turbo is derived from loudr-1, which is derived from
[Chatterbox](https://github.com/resemble-ai/chatterbox), released by Resemble
AI under the MIT licence. The token generator is a smaller model, distilled
from loudr-1's token generator to emit two tokens per forward pass. The
renderer is loudr-1's renderer, distilled to run in one step. The licence
chain is the same as loudr-1's and is recorded in
[NOTICE](https://huggingface.co/loudreader/loudr-1-turbo/blob/main/NOTICE).

How the two-token generator works is written up in
[two-token decode](https://github.com/loudreader/loudkit/blob/main/docs/design/two-token-decode.md).

## Reproducibility

For a fixed build, device, backend and execution configuration, the same text,
voice and seed produce the same waveform. loudr-1 and loudr-1-turbo are
different models: the same text, voice and seed give different audio on each.
A WAV that Python saves records which model made it. The contract is in the
[identity contract](https://github.com/loudreader/loudkit/blob/main/docs/reference/IDENTITY-CONTRACT.md).

## Before you ship

- Long passages are rendered in windows of about ten seconds. Sentence joins
  can occasionally be audible, more often with turbo than with loudr-1.
- Difficult punctuation, numbers and abbreviations can change pronunciation or
  prosody.
- Voice cloning requires consent. A recording being public does not grant
  permission to clone the speaker.
- By default, WAV files that Python's `Result.save()` writes and WAV replies
  from the loudkit server carry an unsigned, machine-readable note. It records
  the model, voice, seed, backend and a checksum of the audio. It is not C2PA.
  The Swift, Go, Rust and TypeScript packages write WAV files without it.
  [The format](https://github.com/loudreader/loudkit/blob/main/docs/reference/provenance.md)
  has the details.

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
