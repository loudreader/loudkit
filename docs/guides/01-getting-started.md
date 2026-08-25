# 1. Getting started

The shortest path from nothing to a WAV file.

## Install

```bash
pip install "loudkit[torch,audio,hub]"
```

Use `pip install -e ".[dev]"` only to work *on* loudkit. That path is
CONTRIBUTING.md's.

The extras:

- `torch` is the reference backend.
- `audio` brings the WAV writer.
- `hub` adds `loudkit download` and by-name voice resolution.

On Apple silicon the engine picks the device split for you: token generator on
the CPU, renderer on the GPU.

## Say something

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
voice = lk.voice("joe", repo="loudreader/loudr-1")

result = engine.synthesize("Hello from loudkit.", voice, seed=7)
result.save("hello.wav")
print(result)  # Result(3.10s, 78 tokens, seed=7, RTF 1.44x)
```

That is the whole thing. The first call downloads the model (750 MB) and
everything after it runs from the cache, offline.

Load the engine **once** and keep it for the life of the process — the
checkpoint takes a few seconds to read. `Result` keeps the audio and every
intermediate, the speech tokens and the mel spectrogram. That is how two
backends get compared when they disagree (see the
[identity contract](../reference/IDENTITY-CONTRACT.md)).

## Get a checkpoint and a voice

`load` fetched what it needed above. To fetch it deliberately -- to pick a
backend, to see the voices, or to put the files somewhere of your own -- the
whole step is:

```bash
loudkit download loudreader/loudr-1     # checkpoint, tokenizer, all voices → shared cache
loudkit voices loudreader/loudr-1       # the menu
loudkit doctor                          # what this machine can run
```

`download` fetches the files one backend needs and no other files. The
default is `--for torch`. `--for onnx` adds the exported graphs the ONNX
backend and the Rust, Go and JS ports read; `--for coreml` adds the CoreML
packages Python's coreml backend and Swift read. `--with-cloning` adds what that
backend enrols with (guide 3 uses them): for `torch`, the enrollment
checkpoint and `ve.safetensors`; for `onnx` and `coreml`, their three
enrollment graphs. It does not add the torch weights to a graph fetch, because
the ports enrol through the graphs and never open them. `--local-dir DIR` materialises the files in a
directory instead of the cache. All twenty voices always come;
they weigh about 3 MB together.

Everything lands in the standard Hugging Face cache, so a second project or
virtualenv shares one copy. A path works everywhere a repo id does, and a path
that exists always wins — a directory named like a repo never triggers a fetch.
The snippets below use the repo-id form. With a local checkout, point `load` at
the checkpoint file and load `voices/joe.safetensors` beside it.

A voice is a handful of tensors, not a model, so it weighs a few hundred
kilobytes. Guide 3 makes your own.

## Reproduce a result with the seed

Run the same call again with the same seed. On the same build and device the WAV
is byte-identical. Change the seed and you get a different, equally valid
reading.

```python
a = engine.synthesize("Hello.", voice, seed=7)
b = engine.synthesize("Hello.", voice, seed=7)
assert (a.audio == b.audio).all()  # True on the same build and device
```

## Inspect the loaded engine

The engine has two stages. A **token generator** writes discrete speech tokens
at 25 Hz, autoregressively. A **renderer** turns those tokens into a waveform in
one parallel pass. Their shapes are opposite, so on Apple silicon they do not
share a device: the generator is faster on the CPU, the renderer on the GPU.
`Engine.describe()` prints what was active. Log it on every run.

```python
print(engine.describe())
# algo[<fingerprint>] loudkit-1 single_path euler=2(cosine) ... | exec[cpu gen=cpu/render=mps ...]
#
# The fingerprint is a placeholder here on purpose. It hashes the whole
# resolved algorithm config, including the shared grammar file, so it moves
# whenever any of that moves. A real value copied into prose is wrong by the
# next release. Run the line to see yours. `/health` and `describe()` report
# the same value.
```


## Straight to the speakers

`Result.audio` is a float32 array at `result.sample_rate`, so nothing has to
touch the disk. loudkit ships no audio output of its own: a playback stack
means a native binary per platform, and this project does not ship one.

```python
import sounddevice as sd          # pip install sounddevice

result = engine.synthesize("Hello from loudkit.", voice, seed=7)
sd.play(result.audio, result.sample_rate)
sd.wait()
```

For long text, `engine.stream(...)` yields each chunk as it is rendered, so
playback starts on the first sentence instead of the last. Guide 2 covers it.

## Say it in another language

A voice carries the language it was enrolled in, so no argument is needed.
Each line below writes one file:

```bash
loudkit speak --checkpoint loudreader/loudr-1 --voice joe      "Hello from loudkit."      -o en.wav
loudkit speak --checkpoint loudreader/loudr-1 --voice dave     "Hola desde loudkit."      -o es.wav
loudkit speak --checkpoint loudreader/loudr-1 --voice henri    "Bonjour depuis loudkit."  -o fr.wav
loudkit speak --checkpoint loudreader/loudr-1 --voice thorsten "Hallo von loudkit."       -o de.wav
loudkit speak --checkpoint loudreader/loudr-1 --voice dante    "Ciao da loudkit."         -o it.wav
```

Ten languages ship. [The demo page](https://loudreader.github.io/loudkit/demo/)
plays every voice, and [VOICES.md](../../VOICES.md) names the source and the
licence of each.

## In production, pin the revision

Without `revision=`, `load` resolves the repository's default branch. The same
code can then return different weights later:

```python
engine = lk.load("loudreader/loudr-1", revision="a1b2c3d")
```

Every command that takes a repo id takes `--revision` for the same reason:

```bash
loudkit serve --checkpoint loudreader/loudr-1 --revision a1b2c3d
```

The digest inside a checkpoint (`tensor_payload_sha256`) proves the file
downloaded intact. It does not prove which artefact it is. Use a commit sha, or
a hash compared against the release's `SHA256SUMS`.

## Next

Your first WAV is a single window, about ten seconds of speech. Anything longer
is the next guide's job.
