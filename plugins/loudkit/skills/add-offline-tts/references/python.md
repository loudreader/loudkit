# loudkit in Python

Excerpt from the loudkit guides for loudkit 0.1.1. Full guides: https://github.com/loudreader/loudkit/blob/main/docs/guides/01-getting-started.md and https://github.com/loudreader/loudkit/blob/main/docs/guides/02-streaming-and-long-form.md

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
see [one waveform for a long passage](https://github.com/loudreader/loudkit/blob/main/docs/guides/02-streaming-and-long-form.md#what-one-waveform-costs).

## Voices

`engine.voices()` lists the 28 names. Both model downloads include all 28
voices. A voice carries the language it was enrolled in, so no language
argument is needed:

```bash
loudkit speak --voice joe   "Hello from loudkit."     -o en.wav
loudkit speak --voice dave  "Hola desde loudkit."     -o es.wav
loudkit speak --voice henri "Bonjour depuis loudkit." -o fr.wav
```

[The demo page](https://loudkit.loudreader.io/demo/) plays every voice
and [VOICES.md](https://github.com/loudreader/loudkit/blob/main/VOICES.md) names the source and licence of each. A voice
file of your own loads with `lk.voice("voices/mine.safetensors")`;
[Cloning a voice](https://github.com/loudreader/loudkit/blob/main/docs/guides/03-cloning-a-voice.md) makes one.

## The seed

The same text, voice and seed give the same audio with the same model, build,
device and execution settings. A different seed gives a different reading.

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
[pinning a release](https://github.com/loudreader/loudkit/blob/main/docs/reference/COMPATIBILITY.md#pinning-a-release).

## Streaming

The first chunk sets the time to first audio.

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
voice = engine.voice("joe")

passage = (
    "The train leaves the city at sunrise and follows the river north. "
    "After the first stop, the valley narrows and the mountains come into view. "
    "By noon we reach the village where the walk begins."
)

for i, result in enumerate(engine.stream(passage, voice, seed=7)):
    print(f"{result.duration:.2f}s chunk ready")  # start playing it here
    result.save(f"chunk-{i:03}.wav")  # one file per chunk
```

`stream` yields one `Result` per chunk, as each becomes ready. Give each chunk
its own filename: saving them all to one name keeps only the last. For one
waveform, use `synthesize` below.

Each chunk after the first is conditioned on the last speech tokens of the
chunk before it. This helps the pitch contour continue across the join. Each
later chunk also gets its own seed, derived from yours and from its position in
the text. A stream stopped early has played the same chunks as the start of
the full render. The first chunk uses your seed unchanged, so text that fits
one window gives the same audio from `stream` and from `synthesize`. All five
implementations use the same seed rule.

## Stopping a render

Pass `should_cancel`, a callable that returns a bool, to either call. The
engine checks it at every decode step and between render stages. A render
stage that has started runs to its end. The chunk in progress is discarded.

```python
import threading

interrupted = threading.Event()

for chunk in engine.stream(passage, voice, seed=7, should_cancel=interrupted.is_set):
    play(chunk)  # interrupted.set() from another thread ends the loop
```

All five implementations follow the same rule. `stream` yields the chunks that
finished before the cancel, then ends without an error. `synthesize` returns
the whole passage or raises `CancelledError` (Go `ErrCancelled`, Rust
`error::CANCELLED`, JS `CancelledError`, Swift `LoudKitError.cancelled`). It
never returns the chunks that finished. To stop playback at once, also discard
the audio your player has queued.
