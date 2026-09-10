# 3. Cloning a voice

A voice in loudkit is a handful of tensors, not a model: a file of about 150 KB
you can copy, ship and version on its own. Making one from five to ten seconds
of clean audio is the `enroll` extra:

```bash
pip install "loudkit[torch,audio,enroll,hub]"
```

Two front ends, one enrollment: `loudkit clone` in the shell, and `enroll` in
each of the five SDKs. There is no MCP clone tool.

Enrollment runs the speaker encoder and the speech tokenizer, models that
synthesis never loads. Their weights are a second file beside the 747 MB
synthesis checkpoint, 523 MB, fetched the first time you clone or with
`loudkit download ... --with-cloning`.

## From the shell

```bash
loudkit clone my-recording.wav --checkpoint loudreader/loudr-1 \
  --name my-voice --language en
```

That writes `voices/my-voice.safetensors` and prints the `speak` command that
reads it. The file is mode `0600` on POSIX.

The command reads **one local WAV or FLAC file**. It does not record, fetch a
URL, denoise or batch. The input is checked before any model runs: 5 to
10 seconds of one speaker is right, more than 30 seconds is refused, and so are
silence, NaN or Inf samples, and a clip under one second.

By default, the command cuts at the last suitable pause in the first ten
seconds and appends 0.4 seconds of silence. A pause near the ten-second limit
leaves only part of that silence inside the prompt. If no pause fits, the
command cuts at 9.6 seconds when needed to make room for silence. This can help soften speech onsets, but the fallback can cut
through a word. Use `--no-end-in-silence` to enroll the recording as given.

| flag | what it does |
|---|---|
| `--checkpoint` | a repo id, a release directory or a checkpoint file. Must be a cloning-capable release. |
| `--name` | what to call the voice. Carried in the profile, and the default filename. |
| `--language` | the language the voice speaks. Stated, never guessed. |
| `-o`, `--output` | where to write. Default `voices/<name>.safetensors`. |
| `--revision` | commit, tag or branch to pin `--checkpoint` to. |
| `--device` | enrollment runtime: `cpu`, `cuda`, `cuda:<index>`, `mps`, `onnx` or `coreml`. Default `cpu`. |
| `--no-end-in-silence` | keep the recording as given, without cutting and padding its ending. |
| `--force` | overwrite the output. Without it an existing file is left alone and the command exits 1. |

Name the language. A Polish voice cloned without `--language pl` reads its
text through the English rules:

```bash
loudkit clone nagranie.wav --checkpoint loudreader/loudr-1 \
  --name gosia --language pl
```

## From Python

One call. It takes a recording and the same reference
[`lk.load`](01-getting-started.md#say-something) takes, and returns the
profile.

```python
import loudkit as lk

mine = lk.enroll("my-recording.wav", "loudreader/loudr-1", name="my-voice")
mine.save("voices/my-voice.safetensors")  # ~150 KB
```

The prompt is built from the first ten seconds and the speaker embedding reads
the whole clip, which is why anything over 30 seconds is refused rather than
cut: trim the recording to its best 5 to 10 seconds yourself.

Python leaves the recording as given by default. Pass `end_in_silence=True`
to use the shell command's pause selection and silence padding. The original
recording must pass the input checks before it is changed.

Name the language when the voice is not English:

```python
mine = lk.enroll("nagranie.wav", "loudreader/loudr-1", name="my-voice", language="pl")
```

`VoiceProfile.language` is what the engine uses when a call names no language,
so a Polish voice enrolled without this reads its text through the English
rules.

The other arguments: `device` places the enrollment models (`cpu` is enough);
`revision` pins the release, as on `lk.load`. Reading an audio file uses
soundfile, which the `enroll` extra brings; pass mono samples in `[-1, 1]`
instead and no reader is involved.

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

Both models use the canonical enrollment assets. You do not need to clone
again when changing the synthesis model.

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
the enrollment produced. A voice does not change when the machine or the backend
changes, so the file travels, and it works with both models.

## Good input, honest limits

Five to ten seconds of clean, single-speaker, noise-free audio is enough. The
hard limits: more than 30 seconds, silence, non-finite samples and clips under
one second are refused, with a message that says what a good input looks like.
The best ten seconds is rarely the first ten you try; how the shipped voices
were chosen, and how to check a new profile before shipping it, is in
[choosing a reference](../design/choosing-a-reference.md).

**Consent is yours to obtain.** Do not clone a voice you have no permission to
use. See [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md). The shipped voices are
enrollments of donated or openly licensed recordings named per voice in
[VOICES.md](../../VOICES.md), not clones of private individuals.

## Next

[Server and agents](04-server-and-agents.md): a warm server and an MCP tool
for any agent on the machine.
