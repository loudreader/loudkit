# 1. Getting started

This guide installs loudkit for Python and makes speech with it. The other
four languages have their own guides: [Swift](10-swift.md), [Go](08-go.md),
[Rust](09-rust.md), [JavaScript](07-js-ts.md).

## Install

```bash
pip install "loudkit[torch,audio,hub]"
```

`torch` runs the model, `audio` writes WAV files, `hub` fetches releases by
name.

## Say something

```bash
loudkit speak --voice joe "Hello from loudkit." --play -o hello.wav
```

The command saves a WAV and plays it through the system player. On Linux,
install `alsa-utils` or `pulseaudio-utils` if neither player is available.

From Python:

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
voice = engine.voice("joe")

result = engine.synthesize("Hello from loudkit.", voice, seed=7)
result.save("hello.wav")
print(result)  # Result(1.56s, 39 tokens, seed=7, RTF 1.00x)
```

Choose `"loudreader/loudr-1-turbo"` in `lk.load` to use the other model;
the voice and synthesis calls stay the same. Available backends depend on
the graphs included in the release. `synthesize` returns audio and never plays it.

The first call downloads the model (747 MB) and the 28 voices into the
Hugging Face cache. After that, loudkit runs offline. When the Hub is
reachable, `lk.load` checks once per process for a newer commit of the release
and fetches it. A `revision=` that is a full commit hash skips the check.

The Swift, Go, Rust and JS ports share one cache directory of their own:
`~/Library/Caches/loudkit/loudreader--loudr-1` on macOS and
`~/.cache/loudkit/loudreader--loudr-1` on Linux. `$LOUDKIT_CACHE` moves it.

Load the engine once and keep it: loading the checkpoint takes a few seconds.
`synthesize` takes text of any length and returns one `Result`. Its audio is a
float32 array at `result.sample_rate`. Memory use grows with the text length:
see [one waveform for a long passage](02-streaming-and-long-form.md#what-one-waveform-costs).

## Voices

`engine.voices()` lists the 28 names. Both model downloads include all 28
voices. A voice carries the language it was enrolled in, so no language
argument is needed:

```bash
loudkit speak --voice joe   "Hello from loudkit."     -o en.wav
loudkit speak --voice dave  "Hola desde loudkit."     -o es.wav
loudkit speak --voice henri "Bonjour depuis loudkit." -o fr.wav
```

[The demo page](https://loudreader.github.io/loudkit/demo/) plays every voice
and [VOICES.md](../../VOICES.md) names the source and licence of each. A voice
file of your own loads with `lk.voice("voices/mine.safetensors")`;
[Cloning a voice](03-cloning-a-voice.md) makes one.

## The seed

The same text, voice and seed give the same audio with the same model, build,
device and execution settings. A different seed gives a different reading.

## Preview what will be spoken

```bash
loudkit text 'Dr. Smith paid $12.' --language en
```

Prepared speech goes to stdout. A word diff on stderr shows replacements
and removals, such as the expansion of numbers and dates.
This loads no model and downloads nothing. See the [CLI reference](../reference/cli.md).

For long text, `engine.stream(...)` yields each chunk as it is rendered:
[Long text and streaming](02-streaming-and-long-form.md).

## Devices

`lk.load` picks a device in this order: CUDA, then Apple silicon, then CPU.
Name one to override it:

```python
engine = lk.load("loudreader/loudr-1", device="cuda")  # or "cpu", "mps"
engine = lk.load("loudreader/loudr-1", device="onnx")  # ONNX Runtime, no torch
```

`device="onnx"` needs `pip install "loudkit[onnx,audio,hub]"` and fetches the
exported graphs together with the checkpoint. `loudkit doctor` says what this
machine can run; `print(engine.describe())` says what the engine chose.

## Fetching the model yourself

`lk.load` fetches what it needs. To fetch ahead of time, or into a directory
of your own:

```bash
loudkit download loudreader/loudr-1                                  # into the shared cache
loudkit download loudreader/loudr-1 --for onnx --local-dir loudr-1   # ONNX graphs, into ./loudr-1
loudkit voices loudreader/loudr-1                                    # list the voices
```

`lk.load`, `lk.voice` and `lk.enroll` accept a release directory in place of
a repo id: `lk.load("loudr-1")`. To hold a production build to one exact
release, see
[pinning a release](../reference/COMPATIBILITY.md#pinning-a-release).

## Next

[Long text and streaming](02-streaming-and-long-form.md),
[Cloning a voice](03-cloning-a-voice.md),
[Choosing a model](11-choosing-a-model.md).

## Download sizes for 0.1.1

Approximate decimal sizes for the release files; backend weights are included.
Cloning adds enrollment assets only when requested. Both models ship separate
synthesis and enrollment checkpoints.

| Model | Torch | Torch + cloning | ONNX | ONNX + cloning | CoreML | CoreML + cloning |
|---|---:|---:|---:|---:|---:|---:|
| loudr-1 | 0.75 GB | 1.28 GB | 2.60 GB | 3.13 GB | 2.46 GB | 2.99 GB |
| loudr-1-turbo | 0.72 GB | 1.25 GB | 2.44 GB | 2.97 GB | 2.44 GB | 2.97 GB |
