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

**Natural-sounding text-to-speech with twenty voices, ten languages and voice
cloning.**

loudr-1 runs on your own hardware through
[loudkit](https://github.com/loudreader/loudkit). Download it once and work offline
from Python, Swift, Go, Rust or TypeScript with PyTorch, ONNX Runtime or CoreML.

These are the weights behind [LoudReader](https://loudreader.io), a reading app
that speaks articles, PDFs and books on device. They are published here so the
engine can be used and checked on its own.

[**Try it in the browser**](https://huggingface.co/spaces/jer3mi/loudkit) |
[**Hear all 20 voices**](https://loudreader.github.io/loudkit/demo/) |
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
voice = lk.voice("joe", repo="loudreader/loudr-1")

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

## Choose your runtime

The repository contains all supported formats, but the downloader fetches only
the runtime you select. Add `--with-cloning` when the installation also needs
enrollment.

| path | command | download |
|---|---|---:|
| Python, synthesis | `loudkit download loudreader/loudr-1 --for torch` | 750 MB |
| Python, with cloning | `--for torch --with-cloning` | 1.28 GB |
| JS, Go or Rust with ONNX | `--for onnx` | 2.60 GB |
| ONNX, with cloning | `--for onnx --with-cloning` | 3.13 GB |
| Swift or Python with CoreML | `--for coreml` | 1.16 GB |
| CoreML, with cloning | `--for coreml --with-cloning` | 1.69 GB |

Add `--local-dir loudr-1` to create a portable directory instead of using the
shared cache. The synthesis checkpoint will be at
`loudr-1/loudr-1.safetensors`.

### Measured speed

| path | hardware | real-time factor |
|---|---|---:|
| PyTorch with CUDA graphs | RTX 3090 | 7.47x |
| PyTorch with CUDA graphs | Jetson Orin Nano | 1.83x |
| split PyTorch engine\* | Apple M3 Pro | 3.43x |
| ONNX Runtime, CPU provider | Apple M3 Pro | 1.21x |
| PyTorch CPU reference | Apple M3 Pro | 0.33x |

\* "Split" describes device placement, not a different model or checkpoint.
The token generator runs on the CPU while the mel and vocoder renderer runs on
the Apple GPU through MPS. Adjacent windows can overlap across the two devices.

Higher is faster, and 1.0x means real time. ONNX Runtime on the measured M3 Pro
CPU is faster than real time. The PyTorch CPU reference path on the same machine
is not.

For batched workloads, the token generator reaches 20.1x aggregate throughput
at batch 1 and 153.1x at batch 64 on the RTX 3090. The highest measured result
is 170.8x on an A100 at batch 64. These are generator-only throughput numbers,
not single-request latency or end-to-end RTF. See the
[benchmark report](https://loudreader.github.io/loudkit/benchmarks/) for commands,
hardware and caveats.

## What ships

| artefact | size | used by |
|---|---:|---|
| `loudr-1.safetensors` | 747 MB | synthesis |
| `loudr-1-enrollment.safetensors` | 523 MB | PyTorch enrollment |
| `ve.safetensors` | 5.7 MB | PyTorch enrollment |
| `onnx/` | 2.38 GB | nine graphs: six synthesis, three enrollment |
| `coreml/` | 941 MB | six packages: three synthesis, three enrollment |
| `voices/` | 3.1 MB | twenty voice profiles |
| `samples/` | 108 KB | the two players above |
| `tokenizer.json` | 70 KB | text processing |

Synthesis and enrollment are separate so users who only need speech generation
do not download the enrollment weights. ONNX and CoreML use their own enrollment
graphs. loudkit also verifies that paired model files came from the same source
checkpoint.

## Voices and consent

The release includes two profiles for each of these languages: English,
Spanish, French, German, Italian, Polish, Portuguese, Dutch, Swedish and Danish.

The profiles were built from recordings donated for speech technology or from
CC0 and CC-BY speech corpora. No scraped celebrity voices ship with the model.
[The full roster](https://github.com/loudreader/loudkit/blob/main/VOICES.md) records
the source, licence and consent basis for every profile. The
[voice gallery](https://loudreader.github.io/loudkit/demo/) provides a generated
sample and enrollment preview for all twenty.

The source enrollment WAVs are not redistributed in the model repository. Their
digests, construction notes and the digests of every shipped profile and sample
are recorded in
[provenance.json](https://github.com/loudreader/loudkit/blob/main/docs/voices/roster/provenance.json).

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
- Saved WAVs and server responses include an unsigned C2PA Content Credentials
  manifest by default. It records the model, voice, seed, backend and audio
  digest in a machine-readable form.

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
