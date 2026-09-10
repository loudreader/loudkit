# Choosing an enrollment reference, and gating the profile

The second half of [cloning a voice](../guides/03-cloning-a-voice.md): how the
shipped voices' reference clips were chosen, how a new profile is checked
before it ships, and how to enroll many clips with one enroller.

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

## Gate it before you ship it

A profile can enroll cleanly and still read badly: a voice whose pauses all
take one shape tends to hold them too long, and that shows up as holes in long
passages. The stall gate measures this in about two minutes, before anyone
listens:

```bash
python tools/check_voice.py voices/my-voice.safetensors --language pl
```

It renders four short passages in token space only and reports how the voice
distributes its pauses across the model's two silence bands, with a verdict:

- **pass**: ship it.
- **warn**: the voice pauses like the roster voices that were fixed by
  cleaning their reference. Strip leading and trailing silence, cap internal
  silence at 100 ms, re-enroll, re-run. That exact treatment took the worst
  warn on the roster to a clean pass.
- **fail**: the voice free-runs silence the way the roster's worst voices did
  before the engine had to be patched around them. Cleaning does not fix this
  class; a better recording does. Re-record before shipping.

The probe defaults to the English reading set; pass `--reading-set` with a
corpus in the voice's own language when you have one. The verdict is a probe,
not a census: a marginal voice can sit a line differently on other passages,
which is what the warn band is for. Details and calibration:
[silence classes](silence-classes.md).

## Bulk enrollment

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
The 256-d utterance voice encoder is not inside the packed checkpoint; it sits
beside it at the release root, and `lk.enroll` resolves it from there. Without
the encoder the enroller raises a `RuntimeError` naming the parameter.
`enroller.enroll` takes the sample rate positionally and does not set the
profile's language, so pass 24 kHz samples and write the language yourself with
`dataclasses.replace` if the voice is not English.

The sweep in the first section is the case this exists for: two hundred
candidates, one enroller, one load of the models.
