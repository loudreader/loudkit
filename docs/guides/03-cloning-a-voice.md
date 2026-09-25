# 3. Cloning a voice

A loudkit voice is a file of about 150 KB that holds a few tensors. You can
copy, ship and version it separately from the model. To make one from 5 to 10
seconds of clean audio, install the `enroll` extra:

```bash
pip install "loudkit[torch,audio,enroll,hub]"
```

Clone a voice with `loudkit clone` in the shell, or with `enroll` in any of the
five SDKs. The MCP server has no clone tool.

Enrollment runs the speaker encoder and the speech tokenizer, models that
synthesis never loads. Their weights are a second file of 523 MB beside the
747 MB synthesis checkpoint. loudkit fetches it the first time you clone, or
with `loudkit download ... --with-cloning`.

## From the shell

```bash
loudkit clone my-recording.wav --checkpoint loudreader/loudr-1 \
  --name my-voice --language en
```

That writes `voices/my-voice.safetensors` and prints the `speak` command that
reads it. The file is mode `0600` on POSIX.

The command reads **one local WAV or FLAC file**. It does not record, fetch a
URL, denoise or batch. The input is checked before any model runs. Use 5 to 10
seconds of one speaker. The command refuses more than 30 seconds, silence, NaN
or Inf samples, and a clip under one second.

By default, the command cuts at the last suitable pause in the first ten
seconds and appends 0.4 seconds of silence. A pause near the ten-second limit
leaves only part of that silence inside the prompt. If no pause fits, the
command cuts at 9.6 seconds when needed to make room for the silence. The added
silence can soften speech onsets. The 9.6-second cut can fall inside a word.
Use `--no-end-in-silence` to enroll the recording as given.

| flag | what it does |
|---|---|
| `--checkpoint` | a repo id, a release directory or a checkpoint file. Must be a cloning-capable release. |
| `--name` | what to call the voice. Carried in the profile, and the default filename. |
| `--language` | the language the voice speaks. Required: loudkit does not detect it. |
| `-o`, `--output` | where to write. Default `voices/<name>.safetensors`. |
| `--revision` | commit, tag or branch to pin `--checkpoint` to. |
| `--device` | enrollment runtime: `cpu`, `cuda`, `cuda:<index>`, `mps`, `onnx` or `coreml`. Default `cpu`. |
| `--no-end-in-silence` | keep the recording as given, without cutting and padding its ending. |
| `--force` | overwrite the output. Without it an existing file is left alone and the command exits 1. |

The language sets the text rules the voice reads with. A Polish voice cloned
with `--language en` reads its text through the English rules. For a Polish
voice, pass `--language pl`:

```bash
loudkit clone nagranie.wav --checkpoint loudreader/loudr-1 \
  --name ania --language pl
```

## From Python

`lk.enroll` takes a recording and the same reference that
[`lk.load`](01-getting-started.md#say-something) takes, and returns the
profile:

```python
import loudkit as lk

mine = lk.enroll("my-recording.wav", "loudreader/loudr-1", name="my-voice")
mine.save("voices/my-voice.safetensors")  # ~150 KB
```

The prompt uses the first ten seconds, and the speaker embedding reads the
whole clip. Recordings longer than 30 seconds are refused, not cut. Trim the
recording to its best 5 to 10 seconds.

Python leaves the recording as given by default. Pass `end_in_silence=True`
to use the shell command's pause selection and silence padding. The original
recording must pass the input checks before it is changed.

For a voice that is not English, name the language:

```python
mine = lk.enroll("nagranie.wav", "loudreader/loudr-1", name="my-voice", language="pl")
```

`lk.enroll` sets the language to `"en"` when you omit it. The engine reads text
in `VoiceProfile.language` when a call names no language.

The other arguments: `device` places the enrollment models (default `cpu`), and
`revision` pins the release, as on `lk.load`. A file path is read with
soundfile, which the `enroll` and `audio` extras bring. You can also pass mono
samples in `[-1, 1]` with their `sample_rate` (default 24000). Then no file
reader runs.

## Clone without PyTorch

Install the runtime and audio reader without the torch-based `enroll` extra:

```bash
pip install "loudkit[onnx,audio,hub]"
# On a Mac, for CoreML instead: pip install "loudkit[coreml,audio,hub]"
```

Use `device="onnx"` or `device="coreml"` with a complete release that includes
enrollment graphs. The output is the same portable profile format:

```python
mine = lk.enroll("nagranie.wav", "loudreader/loudr-1-turbo",
                 name="my-voice", language="pl", device="onnx")
mine.save("voices/my-voice.safetensors")
```

Both models use the same enrollment assets. You do not need to clone again
when you change the synthesis model.

## Use it

```python
import loudkit as lk

voice = lk.voice("voices/my-voice.safetensors")
engine = lk.load("loudreader/loudr-1")
engine.synthesize("Now it is my voice speaking.", voice, seed=1).save("mine.wav")
```

## What a voice carries

A `VoiceProfile` holds the speaker embedding for the token generator, the
conditioning tensors for the renderer, and the reference-prompt tokens and mel
the enrollment produced. The same file works on every machine and backend, and
with both models.

## Choosing the recording

Use five to ten seconds of clean, single-speaker audio without noise. The
limits in [From the shell](#from-the-shell) apply to `lk.enroll` too, and the
refusal message describes a good input. Try more than one clip.
[Choosing a reference](../design/choosing-a-reference.md) describes how the
shipped voices were chosen, and how to check a new profile before you ship it.

**Clone a voice only with permission to use it.** See
[RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md). The shipped voices are
enrollments of donated or openly licensed recordings.
[VOICES.md](../../VOICES.md) names the source and licence of each.

## Next

[Server and agents](04-server-and-agents.md): a warm server over HTTP, gRPC or
MCP.
