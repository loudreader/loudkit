# Choosing an enrollment reference, and gating the profile

The second half of [cloning a voice](../guides/03-cloning-a-voice.md): how the
shipped voices' reference clips were chosen, how a new profile is checked
before it ships, and how to enroll many clips with one enroller.

## Choosing the reference

The profile depends on the ten seconds it came from. Try several segments of a
recording, not only the first. The shipped voices were built with this method.

**What the signal must have.** Reject a segment that fails any of these:

- **No clipping.** A clipped peak distorts the spectral detail the encoder
  identifies a speaker by.
- **Continuous speech, one speaker.** Aim for a speech fraction above about 0.7
  of the window. Long pauses waste reference frames, and a second voice or
  music corrupts the embedding.
- **Wide spectral band.** Check the spectral rolloff (the frequency below which
  about 85% of the energy sits) on the speech frames. A muffled recording rolls
  off low and clones muffled. The shipped references sit in the 2-5 kHz
  median-rolloff range.
- **Clean, steady level.** Use a recording with a low noise floor. Normalise to
  about 0.7 peak before enrolling. Normalising changes the level, not the
  signal-to-noise ratio.
- **5 to 10 seconds, natural narration.** Use flowing sentences, not word lists.
  Cut throat-clearing and settling at the start of a session. If nothing long
  enough exists, concatenate two clips from the same session with a pause of
  about 120 ms.

**Then sweep.** Signal statistics make the shortlist; they do not decide. For
each candidate on the shortlist:

1. enroll it,
2. synthesise the *same* fixed text with the same seed,
3. compute speaker similarity, the cosine between the voice encoder's embedding
   of the render and of the reference (`enroll` both and compare
   `speaker_embedding`),
4. listen to the top few and pick by ear.

Similarity ranks timbre match. The ear also hears delivery, stability and
naturalness. For the shipped Portuguese voice, listeners preferred a candidate
at similarity 0.88 over the one at 0.90. Use similarity to cut a large pool
(two hundred clips, for example) to about five, then listen to those five.

## Gate it before you ship it

A profile can enroll cleanly and still read badly. A voice that puts its pauses
on one of the model's two silence bands tends to hold them too long, and long
passages then get holes. The stall gate measures this in about two minutes,
before anyone listens:

```bash
python tools/check_voice.py voices/my-voice.safetensors --language pl
```

It generates tokens for four short passages at one seed and never renders them
to audio. It reports how the voice spreads its pauses across the two silence
bands, with a verdict:

- **pass**: no stall signature. The profile can ship.
- **warn**: either the probe emitted too few silence tokens to judge (add
  passages with `--passages`), or the band-B share is between 0.10 and 0.30,
  the range of the roster voices that pass after reference cleaning. To clean,
  strip leading and trailing silence, cap internal silence at 100 ms,
  re-enroll and re-run. That treatment takes joe from 0.27 (warn) to 0.65
  (clean pass).
- **fail**: the voice free-runs silence. On the roster, cleaning does not fix
  this class: soren still probes at 0.03 after cleaning. Clean the reference
  and re-run once; if it still fails, re-record before shipping.

The probe defaults to the English reading set, whatever `--language` says.
Pass `--reading-set` with a corpus in the voice's own language when you have
one. Four passages at one seed is a small sample, and a marginal voice can
land on the other side of a threshold on other passages. The warn band exists
for those voices. Details and calibration:
[silence classes](silence-classes.md).

## Bulk enrollment

`lk.enroll` builds the enrollment models on every call and releases them. The
enrollment weights ship as a separate checkpoint,
`loudr-1-enrollment.safetensors`, which synthesis never loads, so a process
that clones once does not keep them in memory. A caller enrolling a hundred
clips pays that load a hundred times.

Build one enroller and keep it:

```python
from pathlib import Path

import librosa

from loudkit.backends.torch_backend import build_torch_enroller

enroller = build_torch_enroller(
    "loudr-1/loudr-1-enrollment.safetensors",
    voice_encoder_weights="loudr-1/ve.safetensors",
)

for clip in sorted(Path("clips").glob("*.wav")):
    samples, rate = librosa.load(clip, sr=None, mono=True)
    profile = enroller.enroll(samples, rate, name=clip.stem)
    profile.save(f"voices/{clip.stem}.safetensors")
```

This is the layer under `lk.enroll`, so you pass it what `lk.enroll` resolves
for you:

- A local path to the enrollment checkpoint. It takes no repo id and does not
  find the file beside the synthesis checkpoint.
- `ve.safetensors`, the 256-d utterance voice encoder. It is not inside either
  checkpoint; it sits at the release root, and `lk.enroll` resolves it from
  there. Without it, `enroller.enroll` raises a `RuntimeError` naming the
  parameter.

`enroller.enroll` takes the sample rate as its second argument and resamples to
24 kHz itself. It does not set the profile's language: for a voice that is not
English, set it with `dataclasses.replace` before you save.

The sweep in the first section is the case this layer exists for: two hundred
candidates, one enroller, one load of the models.
