# 3. Cloning a voice

A voice in loudkit is a handful of tensors, not a model: a few hundred kilobytes
you can copy, ship and version on its own. Making one from five to ten seconds
of clean audio is the `enroll` extra:

```bash
pip install "loudkit[torch,audio,enroll]"
```

Two front ends, one enrollment: `loudkit clone` in the shell, and `enroll` in
each of the five SDKs (Python, Swift, Go, Rust, TypeScript). There is no MCP
clone tool. See [SUPPORTED.md](../../SUPPORTED.md).

Enrollment runs the speaker encoder and the speech tokenizer, models that
synthesis never loads. Their weights ship as their own file,
`loudr-1-enrollment.safetensors`, 523 MB beside the 747 MB synthesis
checkpoint. A synthesis download does not carry it, so `--with-cloning` is what
fetches it, and `loudkit clone` fetches it on its own if you did not. The
`enroll` extra adds the code and dependencies that run them, which is why it is
an extra and not a core dependency. `loudkit doctor` reads a local checkpoint
and says which half is there.

## From the shell

```bash
loudkit clone my-recording.wav --checkpoint loudreader/loudr-1 \
  --name my-voice --language en
```

That writes `voices/my-voice.safetensors` and prints the `speak` command that
reads it. The file is mode `0600` on POSIX; on Windows it inherits the target
directory's ACL. A repo id needs the `hub` extra as well:
`pip install "loudkit[torch,audio,enroll,hub]"`.

The command reads **one local WAV or FLAC file**. It does not record, fetch a
URL, denoise, trim or batch. The input contract is checked before any model
runs: 5 to 10 seconds of one speaker is right, more than 30 seconds is
refused, and so are silence, NaN or Inf samples, and a clip under one second.

| flag | what it does |
|---|---|
| `--checkpoint` | packed `.safetensors`, or a repo id. Must be a cloning-capable release. |
| `--name` | what to call the voice. Carried in the profile, and the default filename. |
| `--language` | the language the voice speaks. Stated, never guessed. |
| `-o`, `--output` | where to write. Default `voices/<name>.safetensors`. |
| `--revision` | commit, tag or branch to pin `--checkpoint` to. |
| `--device` | torch device for the enrollment models: `cpu`, `cuda`, `cuda:<index>`, `mps`. Default `cpu`. |
| `--force` | overwrite the output. Without it, an existing file is left alone and the command exits 1. |

The profile is written beside the target and moved onto it, so a run that fails
halfway leaves the previous file or no file.

Name the language. A Polish voice cloned without `--language pl` reads its text
through the English funnel:

```bash
loudkit clone nagranie.wav --checkpoint loudreader/loudr-1 \
  --name gosia --language pl
```

## From Python

One call. It takes a recording and the same checkpoint reference
[`lk.load`](01-getting-started.md#synthesise) takes, and it returns the profile.

```python
import loudkit as lk

mine = lk.enroll("my-recording.wav", "loudreader/loudr-1", name="my-voice")
mine.save("voices/my-voice.safetensors")  # ~150 KB
```

The call resolves the checkpoint, builds the enrollment models, reads the file
at 24 kHz and returns a `VoiceProfile`. The prompt is built from the first ten
seconds and the speaker embedding reads the whole clip, which is why the input
contract refuses anything over 30 seconds instead of quietly truncating: trim
the recording to its best 5 to 10 seconds yourself.

Name the language when the voice is not English:

```python
mine = lk.enroll("nagranie.wav", "loudreader/loudr-1", name="my-voice", language="pl")
```

`VoiceProfile.language` is what the engine falls back to when a call omits one,
so a Polish voice enrolled without this reads its text through the English
funnel. See [text normalization](../reference/preprocess.md).

The other arguments:

- `device` places the enrollment models. `cpu` is the default and is enough.
- `revision` pins the release, as it does on `lk.load`.
- `voice_encoder_weights` points at a `ve.safetensors` somewhere else. The
  256-d utterance voice encoder is **not** inside the packed checkpoint. It sits
  beside it at the release root, and `lk.enroll` resolves it from there, so this
  argument is only for an encoder you keep elsewhere. A release that ships none
  can synthesise but not clone, and says so with `FileNotFoundError`.

Reading an audio *file* needs librosa, which the `enroll` extra brings. Pass
mono samples in `[-1, 1]` instead and no reader is involved.

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
changes, so the file travels.

## Good input, honest limits

Five to ten seconds of clean, single-speaker, noise-free audio is enough, and
it is what the preflight recommends. The hard limits: more than 30 seconds,
silence, non-finite samples and clips under one second are refused, with a
message that says what a good input looks like. The 24 to 16 kHz downsample is
one portable resampler shared by all five ports. The utterance embedding reads
the whole clip. The prompt is capped at ten seconds.

**Consent is yours to obtain.** Do not clone a voice you have no permission to
use. See [RESPONSIBLE_USE.md](../../RESPONSIBLE_USE.md) for what this library
should and should not be pointed at. The shipped voices are enrollments of
consented or openly licensed recordings, donations and CC0 / CC-BY corpora named
per voice in the roster's provenance file, not clones of private individuals.

## Choosing the reference: what ten seconds to feed it

The profile is only as good as the ten seconds it came from, and the best segment
is rarely the first one you try. This is the selection method the shipped voices
were built with.

**What the signal must have.** Reject any segment that fails the basics:

- **No clipping.** A peak at full scale distorts exactly the spectral detail the
  encoder identifies a speaker by.
- **Continuous speech, one speaker.** Aim for a speech fraction above ~0.7 of the
  window. Long pauses waste reference frames, and a second voice or music poisons
  the embedding.
- **Wide spectral band.** Check the spectral rolloff (the frequency below which
  ~85% of the energy sits) on the speech frames. A muffled recording rolls off
  low and clones muffled. The shipped references sit in the 2-5 kHz
  median-rolloff range. Bandwidth you feed in is the ceiling on bandwidth you get
  out.
- **Steady, healthy level.** Quiet references drown in the noise floor. Normalise
  to ~0.7 peak before enrolling.
- **5 to 10 seconds, natural narration.** Use flowing sentences, not word lists.
  Skip the first seconds of any session (throat-clearing, settling). If nothing
  long enough exists, concatenate two clips from the same session with a ~120 ms
  pause.

**Then sweep.** Signal statistics shortlist candidates; they do not decide. For
each of the top candidates:

1. enroll it,
2. synthesise the *same* fixed text with the same seed,
3. compute speaker similarity, the cosine between the voice encoder's embedding
   of the render and of the reference (`enroll` both and compare
   `speaker_embedding`),
4. **listen to the top few, and let the ear decide.**

Step 4 matters. Building the shipped Portuguese voice, the candidate
that measured best (similarity 0.90) lost the listening test to one that measured
0.88. Similarity ranks timbre match, but the ear also hears delivery, stability
and naturalness. The metric earns its keep by turning two hundred clips into five
worth an hour of listening. The hour still happens.

## Advanced: bulk enrollment

`lk.enroll` builds the enrollment models on every call and drops them. That is
right for a caller who clones once, because those models are the 40% of the
checkpoint synthesis never loads and holding them alive costs that memory for
the rest of the process. It is wrong for a caller enrolling a hundred clips.

Build one enroller and keep it:

```python
from pathlib import Path

import librosa

from loudkit.backends.torch_backend import build_torch_enroller

enroller = build_torch_enroller(
    "loudr-1.safetensors",
    voice_encoder_weights="loudr-1/ve.safetensors",
)

for clip in sorted(Path("clips").glob("*.wav")):
    samples, _ = librosa.load(clip, sr=24_000, mono=True)
    profile = enroller.enroll(samples, 24_000, name=clip.stem)
    profile.save(f"voices/{clip.stem}.safetensors")
```

This is the layer under `lk.enroll`, so it takes what `lk.enroll` resolves for
you: a checkpoint path rather than a repo id, and `ve.safetensors` by hand.
Without the encoder the enroller raises a `RuntimeError` naming the parameter.
`enroller.enroll` takes the sample rate positionally and does not set the
profile's language, so pass 24 kHz samples and write the language yourself with
`dataclasses.replace` if the voice is not English.

The sweep in the section above is the case this exists for: two hundred
candidates, one enroller, one load of the models.

## Next

Now serve what you made: a warm server and an MCP tool for any agent on the
machine.
