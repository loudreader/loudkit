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
  <img src="https://huggingface.co/loudreader/loudr-1/resolve/main/logo.png" alt="loudkit" width="640">
</p>

# loudr-1

**Text-to-speech with 28 voices, ten languages and voice cloning.**

loudr-1 runs on your own hardware through
[loudkit](https://github.com/loudreader/loudkit). Download it once and work offline
from Python, Swift, Go, Rust or TypeScript with PyTorch, ONNX Runtime or CoreML.

[LoudReader](https://loudreader.io) uses these weights.

[**Try the online demo**](https://huggingface.co/spaces/jer3mi/loudkit) |
[**Hear all 28 voices**](https://loudreader.github.io/loudkit/demo/) |
[**Open in Colab**](https://colab.research.google.com/github/loudreader/loudkit/blob/main/notebooks/loudkit_quickstart.ipynb) |
[**GitHub**](https://github.com/loudreader/loudkit) |
[**Documentation**](https://loudreader.github.io/loudkit/)

## Listen

**Joe**

<audio controls src="https://huggingface.co/loudreader/loudr-1/resolve/main/samples/joe.opus"></audio>

**Kathleen**

<audio controls src="https://huggingface.co/loudreader/loudr-1/resolve/main/samples/kathleen.opus"></audio>

Both voices read the same passage from *Alice's Adventures in Wonderland*.
[Open the gallery](https://loudreader.github.io/loudkit/demo/) to compare every
shipped voice with the enrollment reference used to create its profile.

> The maintainers evaluated naturalness by ear in English only. They do not
> speak the other nine languages well enough to judge them. If you are a
> native speaker, please listen and report what sounds right or wrong.

## Start in Python

```bash
pip install "loudkit[torch,audio,hub]"
```

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
voice = engine.voice("joe")

engine.synthesize("Hello from loudkit.", voice, seed=7).save("hello.wav")
```

The first run downloads the 747 MB synthesis checkpoint and the voices. Later
runs use the local cache. The same path from a shell is:

```bash
loudkit speak --checkpoint loudreader/loudr-1 --voice joe \
  "Hello from loudkit." -o hello.wav
```

To enroll a voice that you own or have permission to use:

```python
mine = lk.enroll("my-recording.wav", "loudreader/loudr-1", name="my-voice")
mine.save("voices/my-voice.safetensors")
```

The reusable profile is about 150 KB. Install
`loudkit[torch,audio,enroll,hub]` for enrollment.

The Swift, Go, Rust and TypeScript packages load the same repo id and
download the files their backend needs. The
[README](https://github.com/loudreader/loudkit#swift-go-rust-and-typescript)
shows each one.

## What each runtime downloads

The repository contains the files for every supported backend.
`loudkit download --for` fetches only the files for one backend. Add
`--with-cloning` when the installation also needs enrollment.

| path | command |
|---|---|
| Python, synthesis | `loudkit download loudreader/loudr-1 --for torch` |
| Python, with cloning | `--for torch --with-cloning` |
| TypeScript, Go or Rust with ONNX Runtime | `--for onnx` |
| ONNX Runtime, with cloning | `--for onnx --with-cloning` |
| Swift or Python with CoreML | `--for coreml` |
| CoreML, with cloning | `--for coreml --with-cloning` |

Download size depends on the model, backend and release revision. The
synthesis checkpoint is 747 MB, and the ONNX and CoreML downloads also include
the weights of their graphs. Approximate decimal sizes for 0.1.1, with the
backend weights included:

| Model | Torch | Torch + cloning | ONNX | ONNX + cloning | CoreML | CoreML + cloning |
|---|---:|---:|---:|---:|---:|---:|
| loudr-1 | 0.75 GB | 1.28 GB | 2.60 GB | 3.13 GB | 2.46 GB | 2.99 GB |
| loudr-1-turbo | 0.72 GB | 1.25 GB | 2.44 GB | 2.97 GB | 2.44 GB | 2.97 GB |

Both models ship separate synthesis and enrollment checkpoints. The enrollment
files come only with `--with-cloning`.

Speed depends on the hardware and the backend.
[Benchmarks](https://loudreader.github.io/loudkit/benchmarks/) has the
measured figures, the machines and the commands.

## What ships

| artefact | size | used by |
|---|---:|---|
| `loudr-1.safetensors` | 747 MB | synthesis |
| `loudr-1-enrollment.safetensors` | 523 MB | PyTorch enrollment |
| `ve.safetensors` | 5.7 MB | PyTorch enrollment |
| `onnx/` | varies by model | synthesis and enrollment graphs |
| `coreml/` | varies by model | native generation, rendering and enrollment packages |
| `voices/` | 4.3 MB | 28 voice profiles |
| `samples/` | 168 KB | the two players above |
| `tokenizer.json` | 70 KB | text processing |

Synthesis downloads do not include the enrollment weights. PyTorch enrollment
uses `loudr-1-enrollment.safetensors` and `ve.safetensors`. ONNX and CoreML
use their own enrollment graphs. Every download is checked against the
release's `SHA256SUMS` before it is used.

[The graph signatures](https://github.com/loudreader/loudkit/blob/main/docs/design/onnx-graphs.md)
list what each graph under `onnx/` takes and returns, and the order to call
them in. Use them to run loudr-1 from a language that loudkit has no port for.

## Voices and consent

The release includes ten English profiles and two for each of Spanish, French,
German, Italian, Polish, Portuguese, Dutch, Swedish and Danish.

The profiles were built from recordings donated for speech technology or from
CC0 and CC-BY speech corpora. No audio without a stated licence ships with the
model. [The full roster](https://github.com/loudreader/loudkit/blob/main/VOICES.md) records
the source, licence and consent basis for every profile. The
[voice gallery](https://loudreader.github.io/loudkit/demo/) provides a generated
sample and enrollment preview for all 28.

The source enrollment WAVs are not redistributed in the model repository. Their
hashes, construction notes and the hashes of every shipped profile and sample
are recorded in
[the roster record](https://github.com/loudreader/loudkit/blob/main/docs/voices/roster/provenance.json).

## Model lineage

loudr-1 is derived from
[Chatterbox](https://github.com/resemble-ai/chatterbox), released by Resemble AI
under the MIT licence. loudr-1 changes the signal flow for faster local
inference, separates synthesis from enrollment, and sets its own graph
boundaries and device placement for PyTorch, ONNX Runtime and CoreML.

Release gates compare the implementations, check output length and early end of
speech, and run ASR-based checks per measured language. These checks catch
mechanical regressions. Native speakers still need to check the audio by ear.

## Reproducibility

For a fixed build, device, backend and execution configuration, the same text,
voice and seed produce the same waveform. Across devices and backends, the
conformance tests compare the token streams and the waveform correlation on
fixed test cases. Waveforms are not promised to be byte-identical across
devices or backends.

The exact contract and current measurements are in the
[identity contract](https://github.com/loudreader/loudkit/blob/main/docs/reference/IDENTITY-CONTRACT.md)
and [measured parity report](https://loudreader.github.io/loudkit/parity-measured/).

## Before you ship

- Long passages are rendered in windows of about ten seconds. Sentence joins
  can occasionally be audible.
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

Read [Responsible use](https://huggingface.co/loudreader/loudr-1/blob/main/RESPONSIBLE_USE.md)
before exposing enrollment to other people.

## Intended use

loudr-1 is intended for local narration, accessibility, localisation, games,
prototyping and speech research. It is not a voice-authentication system and
must not be used for deceptive impersonation.

## Training data

The original Chatterbox training data is controlled by Resemble AI and is not
documented by this project. The shipped voice profiles use recordings made or
released for speech-technology use; their sources and licences are listed in
the public roster.

## Licence

[Apache-2.0](https://huggingface.co/loudreader/loudr-1/blob/main/LICENSE).
Upstream attributions and component licences are listed in
[NOTICE](https://huggingface.co/loudreader/loudr-1/blob/main/NOTICE).
