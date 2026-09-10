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
  <img src="https://huggingface.co/loudreader/loudr-1/resolve/main/logo.png" alt="LoudKit" width="640">
</p>

# loudr-1

**Natural-sounding text-to-speech with 28 voices, ten languages and voice
cloning.**

loudr-1 runs on your own hardware through
[loudkit](https://github.com/loudreader/loudkit). Download it once and work offline
from Python, Swift, Go, Rust or TypeScript with PyTorch, ONNX Runtime or CoreML.

These are the weights behind [LoudReader](https://loudreader.io), a reading app
that speaks articles, PDFs and books on device. They are published here so the
engine can be used and checked on its own.

[**Try it in the browser**](https://huggingface.co/spaces/jer3mi/loudkit) |
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

> English is the only language we could evaluate ourselves by ear. We do not
> speak the other nine languages well enough to judge their naturalness
> reliably. If you do, please listen and share what sounds good or wrong.
> Feedback from native speakers is very welcome.

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

The Swift, Go, Rust and TypeScript packages load the same repo id and fetch
what they need themselves; the
[README](https://github.com/loudreader/loudkit#the-same-in-swift-go-rust-and-typescript)
shows each one.

## What each runtime downloads

The repository contains every supported format. A download takes one runtime
and leaves the rest behind. Add `--with-cloning` when the installation also
needs enrollment.

| path | command |
|---|---|
| Python, synthesis | `loudkit download loudreader/loudr-1 --for torch` |
| Python, with cloning | `--for torch --with-cloning` |
| TypeScript, Go or Rust with ONNX Runtime | `--for onnx` |
| ONNX Runtime, with cloning | `--for onnx --with-cloning` |
| Swift or Python with CoreML | `--for coreml` |
| CoreML, with cloning | `--for coreml --with-cloning` |

Download size depends on the model, backend and release revision. The
synthesis checkpoint is 747 MB; graph downloads also include backend-specific
weights.

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
| `samples/` | 108 KB | the two players above |
| `tokenizer.json` | 70 KB | text processing |

Synthesis and enrollment are separate so users who only need speech generation
do not download the enrollment weights. What each graph under `onnx/` takes and
returns, and the order to call them in, is in
[the graph signatures](design/onnx-graphs.md), which is what a runtime loudkit
has no port for needs. ONNX and CoreML use their own enrollment
graphs. Every download is checked against the release's `SHA256SUMS` before
it is used.

## Voices and consent

The release includes ten English profiles and two for each of Spanish, French,
German, Italian, Polish, Portuguese, Dutch, Swedish and Danish.

The profiles were built from recordings donated for speech technology or from
CC0 and CC-BY speech corpora. No scraped celebrity voices ship with the model.
[The full roster](https://github.com/loudreader/loudkit/blob/main/VOICES.md) records
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
under the MIT licence. We optimized it for faster local inference by profiling
the full synthesis path, changing the signal flow, separating synthesis from
enrollment, and adjusting graph boundaries and device placement for PyTorch,
ONNX Runtime and CoreML.

Release gates compare the implementations, check output length and early end of
speech, and run ASR-based checks per measured language. These checks catch
mechanical regressions. They do not replace listening by native speakers.

## Reproducibility

For a fixed build, device and backend, the same text, voice and seed produce the
same waveform. Across devices or backends, loudkit checks the token stream and
keeps waveform differences inside measured correlation bands. Floating-point
execution means that waveforms are not promised to be byte-identical across
different runtimes.

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
- Saved WAVs and server responses carry an unsigned, machine-readable note by
  default, recording the model, voice, seed, backend and a checksum of the
  audio. It is loudkit's own and is not C2PA Content Credentials.

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

## Download sizes for 0.1.1

Approximate decimal sizes for the release files; backend weights are included.
Cloning adds enrollment assets only when requested. Both models ship separate
synthesis and enrollment checkpoints.

| Model | Torch | Torch + cloning | ONNX | ONNX + cloning | CoreML | CoreML + cloning |
|---|---:|---:|---:|---:|---:|---:|
| loudr-1 | 0.75 GB | 1.28 GB | 2.60 GB | 3.13 GB | 2.46 GB | 2.99 GB |
| loudr-1-turbo | 0.72 GB | 1.25 GB | 2.44 GB | 2.97 GB | 2.44 GB | 2.97 GB |
