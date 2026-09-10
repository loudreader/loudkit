# Postprocess: where the sentence actually ended

The third stage of `preprocess → tts → postprocess`, and the one people expect
to be a denoiser. It is a **detector**: it reads the speech tokens a chunk
produced, decides whether the model kept talking after it was done, and returns
a verdict. It never touches a sample of audio.

This document is the whole rule set with the evidence behind every number.

---

## The failure it exists for

A listener reports: *"it finished the sentence, then there was a long gap, then
one random word."*

That is not a recording artifact and no filter can remove it. It is generated,
and the mechanism is specific:

1. The decoder is free-running. Nothing in the model guarantees it stops when
   the sentence is over. It stops when it samples the stop token, or when it
   hits a cap.
2. Silence tokens are exempt from the `min_p` cutoff. The exemption is
   measured: a pause token is the only way to pause, and removing the
   exemption nearly doubled long-form gaps (median 2.46 s to 4.64 s). They
   were exempt from the repetition penalty too, until the interior-stall
   study showed the pair of exemptions makes a silence run absorbing; the
   penalty now applies to silence like any other token. See
   `SamplingConfig.silence_token_ids` and the `stall` rule below.
3. So once the sentence is genuinely finished, silence tokens keep probability
   mass indefinitely. The decoder free-runs silence, and **any step where a
   non-silence token survives the cutoff becomes a hallucinated word**.

The shape in the token stream is therefore: *sentence, a run of silence, a short
burst of speech*. In a debug trace this prints literally,
`.` for silence and `#` for speech:

```
t3.rowTail  text=14 gen=96 ended=false bestEOS=45@0.000
            tail=####........................#####
```

## Why the token domain, not the audio domain

Three reasons, in order of how much they mattered.

**The evidence is there and nowhere else.** A hallucinated word is speech, with
the same spectrum, level and voice as the real sentence. No energy threshold
separates it from a real ending. What separates it is *position*, behind half a
second of silence at the end of a row that already ran long, and position is a
token fact.

**It is portable.** Token counts and set membership are integers. Five
implementations agree exactly rather than to a tolerance. An audio-domain rule
would need a float threshold on an energy envelope, and a rule that turns on the
last bits of a float decides differently on different hardware.

**It is cheap and it composes.** Cutting tokens is a slice. Cutting audio means
a crossfade, and a crossfade at the wrong place is its own artifact.

## Where it bites hardest

Measured: the reported artifact sentences **rendered clean interactively and
broke only when batched**, where a short text rides padded to its longest
neighbour. Batching is where this appears, which is where a server puts it.

---

## The rules

Seven evidence rules and one generation-time guard. Constants live in
`PostprocessConfig` rather than in code, so a port that quietly uses a different
number moves the fingerprint instead of silently producing different audio.

### 0. The ceiling: a guard, applied during generation

| constant | value | meaning |
|---|---|---|
| `ceiling_speech_per_text_token` | 4.0 | hard stop as a multiple of text tokens |
| `ceiling_slack_tokens` | 40 | additive slack (1.6 s of audio) |

`ceiling = min(window, int(text_tokens × 4.0) + 40)`

The clamp to `window` is what changed. The slack itself did not go away: it is
`ceiling_slack_tokens`, and at 40 it is larger than the 15 this line used to
add. What went away is a ceiling allowed to exceed the window. `frame_windows`
refuses anything past `max_speech_tokens`, so a row allowed to reach 270 was
stopped at 270 and rejected at 255. Changed in all five implementations
together with the `funnel-2` bump, because a ceiling change moves audio and has
to be visible in the fingerprint.

**Why it exists.** Before it, the decoder had a flat limit of 255 tokens
regardless of how short the text was. That is 10.2 s of audio, since the vocoder
emits exactly 40 ms per token. A three-word sentence could decode for ten
seconds.

**Where 4.0 comes from.** A device trace of the showcase render:

```
t3.overrun  gen=92 ceiling=92 bestEOS=74@0.003 floor=31
```

A chunk of ~26 text tokens stopped only because it hit the ceiling, with the
model's confidence in stopping at 0.003. It was mid-sentence and had already
needed 3.5 speech tokens per text token. Several narrators came back at 3.2–4.3 s
on a script that runs 10–14 s in the rest of the roster. Four is comfortably past
anything measured.

**Why not 2.6.** The chunker uses 2.6 speech tokens per text token, documented as
the conservative end of a measured 1.75–2.35. That is conservative *for budgeting
a chunk*, where guessing high only wastes window. Here it is the opposite:
guessing low cuts a sentence off. The two numbers face opposite directions and
must not be shared.

**What it is not.** A tempo control. It is a runaway guard. The fine cut is the
rescue below.

### 1. `dropout`: the row is incomplete

| constant | value | meaning |
|---|---|---|
| `dropout_min_tokens` | 25 | below this a row is too short to be its text (~1.0 s) |

Every other rule on this page says the *end* of the row is wrong. This one says
the row is **incomplete**. It is the only verdict here that reports without
cutting: there is nothing to remove, and removing more cannot recover the
missing content.

It is the most damaging failure in the set, because a listener **cannot hear
it**. A hallucinated word draws attention to itself. A sentence that simply
stopped halfway sounds like a sentence.

Two conditions, both required:

- **an absolute floor**: under 25 speech tokens, borrowed from the published
  criterion for a catastrophic neural-codec TTS failure rather than guessed;
- **a proportional test**: fewer speech tokens than text tokens. A read that
  produced less than one speech token per text token has not said the text under
  any pronunciation. This is what keeps a genuinely short line exempt: the
  shortest healthy reads measured across nine languages run 35 tokens.

It runs first, because nothing below it can help a row that is already too
short.

### 2. `repetition`: the decoder reading its own tail

| constant | value | meaning |
|---|---|---|
| `repetition_max_period` | 12 | longest cycle that counts as a lock-up (~0.5 s) |
| `repetition_min_cycles` | 3 | consecutive exact repeats: necessary, not sufficient |
| `repetition_min_span` | 24 | tokens the repeating region must cover (~1.0 s) |
| `repetition_silence` | acoustic | which silence family the all-silence-cycle exemption reads |
| `repetition_resume` | condemn | a loop the decoder resumed from is condemned, never cut |

Every other rule on this page reads the *end* of a chunk. This one is the reason
that is not enough: a stuck decoder repeats **inside** the row, and no amount of
looking at the tail will find it.

The mechanism is the same as the trailing hallucinated word, the model's own
output becoming its context, but the symptom differs. Every ranking of what goes
wrong with autoregressive speech models puts this failure first or second.
VALL-E's authors describe greedy search "continually generating silence codec
codes"; the Very Attentive Tacotron stress test produced 52 repetitions of a
phrase that was meant to occur nine times.

**This is the only rule here that cuts mid-sequence**, which makes it the most
destructive thing the layer can do, so it is hard to trigger:

- **A short cycle.** Above ~0.5 s a repeated block is a *phrase*, and a repeated
  phrase is rhetoric: a refrain, a stammer, "no, no, no". Below it the model is
  emitting the same fragment because it is reading itself.
- **A long repeat.** This is the condition that does the work, and getting it
  wrong is documented below.
- **Exactly.** Approximate repetition is left alone. A decoder that has genuinely
  locked up emits the same tokens, not similar ones, and a fuzzy match on a
  signal this destructive is a way to truncate real speech.
- **Not silence.** A cycle that is entirely silence is never a loop. Silence
  repeating is what silence *is*, and the tail rules already judge pauses
  against where they sit. "Silence" here means *acoustic* silence: with
  `repetition_silence` at its default `acoustic`, the exemption reads the
  union of the sampler's `silence_token_ids` and both render censuses
  (`silence_render_ids`, `quiet_render_ids`), so a pause parked on any
  silent-rendering id is covered. See the en0023 provenance below for the
  bug the sampler-only reading shipped. A cycle *mixing* silence with speech
  still counts: the word-then-pause stutter is one of the shapes this takes.

It cuts one full cycle past the loop's start, keeping the first instance. That
one is plausibly the word the sentence actually wanted.

#### How `repetition_min_span` was settled, and what the first attempt got wrong

The first version of this rule had no span condition. It asked only "is there a
short cycle repeated at least four times", which is the shape the published
*inline* guards use: VALL-E 2's repetition-aware sampling watches a 10-token
window, and MSpoofTTS scans at segment lengths 10/25/50.

**It fired on 22 of 27 healthy renders.** Every row, in every language.

The reason is specific to this checkpoint and would not have been guessed. It
winds down each utterance with a short repeated tail token, usually `6405` or
`6486`, and **those tokens are not in the 31-id silence list**, so the
"a cycle of pure silence is not a loop" exemption did not cover them. "Does it
repeat" is true of almost all real speech here.

What separates a stuck decoder is not that it repeats but that it **does not
stop**. Measured across the same 27 renders, in nine languages, three length
classes, one voice held constant:

| | longest repeating span |
|---|---|
| healthy rows (n=26) | **10 tokens** (0.4 s), median 7 |
| the one runaway in the set | 44 tokens (1.76 s) |

24 tokens sits between them with 2.4x margin over the healthy maximum. With it,
the rule fires on **0 of 27**.

**Two things this leaves honest.** The runaway in that table repeats a *silence*
token, so this rule correctly declines it and `desperation` owns that row, which
means the set contains **no observed true positive**. And the inline-guard
parameters that seeded the first attempt were designed to steer a sampler, not to
triage a finished row. Borrowing them without re-measuring is what produced the
81% false-positive rate.

So this rule is a **guard against a failure this checkpoint has not yet been
caught committing**, not a fix for one that was reported. It is calibrated to be
safe rather than sensitive. If a real loop ever turns up, its trace replaces
this section.

#### The false loop of en0023, and the silence family it settled

The span threshold protected healthy wind-downs. It could not protect a long
pause, and the study above had already named the ids that would bite: `6405`
and `6486` render true silence and are not in the sampler's 31-id list.

The specimen: kathleen, en0023, seed 1234. A mid-chunk pause parked on 6486
(x7) then 6405 (x24). Both ids are in the manifest's `silence_render_ids`,
neither is in `sampling.silence_token_ids`, and the exemption read the
sampler list, so the 24-token period-1 run passed every loop condition. The
cut kept one cycle and deleted the pause plus the six seconds of
correctly-read speech behind it: two whole sentences, verdict `repetition`,
not suspect, no retry, audibly fluent. A report-mode render proved the raw
row carried the full text. The `stall` rule would have condemned the row
into the retry ladder (31 census-silent tokens against a gate of 25), but
`repetition` outranks it and saved the row first. Systemic, not a corner:
six of the checkpoint's eight truly-silent ids sit outside the sampler list,
so the exemption was blind on most real pauses. Measured prevalence 1/200
renders, and the shape is the worst one this layer knows: inaudible content
loss shipping as clean.

The fix is one definition. With `repetition_silence` at `acoustic` (the
default) the exemption reads the union of the sampler list and both render
censuses, resolved inside the rule from the config, the same way `stall`
reads its censuses. The pause is then never a loop, the row falls through to
`stall`, and the retry ladder re-renders it. On a checkpoint without
censuses the union degenerates to the sampler list, and the resumption
guard below is what protects the row there; `sampling` names the
pre-amendment family outright for a manifest measured under it. The
`repetition_silence` fixture section pins the specimen shape, a genuine
speech loop, the census-only stutter, and both boundary cases.

**Two lists, on purpose.** Only `repetition` and `stall` read acoustic
silence, because both ask "is this dead air": one to decline a loop, one to
condemn a hole. Every tail rule (`silence_tail`, `terminal_echo`,
`desperation`'s seam, `ended_tail`) stays keyed to the sampler's
`silence_token_ids`: their thresholds were calibrated against that list, the
resulting ~0.5 s house tail measures exactly at Kokoro's level, and re-keying
them would change prosody everywhere for a defect nobody demonstrated.

#### The guard that holds without a census: a locked decoder does not resume

The census fix needs a manifest that names the silent ids, and the published
hub pack names none. On it the acoustic union degenerates to the sampler
list and the same pause fires again: kathleen, en0023, seed 1234, rendered
on the census-less pack, parks the same mid-chunk pause (a 6486 run into a
6405 run of 24), the period-1 run passes every loop condition, and the cut
keeps 52 of 206 tokens. The pause and the 131 tokens of correctly-read
speech behind it are deleted, two whole sentences, verdict `repetition`, no
retry, audibly fluent. Measured prevalence about 1 in 200 renders.

The guard reads no silence at all. A genuine lock-up is a tail pathology:
the model's own output is its context, the state is absorbing, and the
repeating region runs to the end of the row. Every firing fixture case
resumes by zero tokens. A ceiling can truncate at most one incomplete copy,
which is `period - 1` tokens, at most 11. The specimen resumes by 131. So
the law, with `repetition_resume` at `condemn` (the default): a qualifying
loop whose region ends a full period or more before the row does is a
decoder that resumed, a decoder that resumed was never locked, and the row
is condemned whole into the retry ladder, verdict `repetition`, `suspect`.
No trim is kept as a fallback, because the trim is the defect; an exhausted
ladder ships the row whole, which for a pause is the correct audio. A loop
whose region reaches within one period of the row's end cuts exactly as
before. The comparison is integer, `resume >= period`, with no constant to
tune: a matching full copy would have counted as another cycle, so a full
period of anything after the region already proves a deviation.

A discard-fraction guard was calibrated first and rejected. The measured
distributions:

| population | discard fraction | resume (tokens) |
|---|---|---|
| firing fixture cases (5: three `repetition`, two `repetition_silence`) | 0.36 to 0.58 | 0 |
| the en0023 specimen, census-less | 0.75 | 131 |

A cap of 2/3 splits the measured sets with margin on both sides. But the
split is geometry, not mechanism: the discard fraction depends on where in
the row the pause sits. A 30-token pause at 70% of a 206-token row with one
clause behind it discards 0.32, under any cap the fixture fires allow, and
the cut ships, deleting the clause. The fraction bounds the loss;
resumption removes it. That construction is a unit case, not a render, but
the pause position is a property of the text, so nothing keeps a real
render off it.

Two consequences worth naming. An applied repetition cut now only ever
removes a tail, like every other rule in this layer; the mid-sequence
anchor still outranks everything, but what it condemns it hands back whole.
And the guard is not a census-less fallback: it runs under every
configuration, so a pause parked on an id outside both censuses (the
da0028 class) is covered on amended checkpoints too. Set membership can be
incomplete; resumption cannot.

`repetition_resume: "cut"` names the pre-amendment law for a checkpoint
measured under it. The `repetition_resume` fixture section pins the
specimen at its measured proportions, a genuine loop, both sides of the
debris boundary (`period - 1` cuts, `period` condemns), the one-token
resumption of a period-1 pause, and a trailing run that still cuts.

### 3. `stall`: the decoder trapped in silence

| constant | value | meaning |
|---|---|---|
| `stall_run_tokens` | 25 | non-tail dead-air run that condemns a row (~1.0 s) |
| `silence_render_ids` | manifest | ids that render true digital silence: the run gate |
| `quiet_render_ids` | manifest | breath and decay ids: extend a run, never count toward it |

Every rule below this one reads a signal at the end of the row. A stalled
decoder does its damage in the middle, or produces nothing at all, and every
such row used to ship as `clean`.

**The mechanism**, measured at scale (1705 passages, 20 voices, ten
languages). The decoder enters a silence run at a pause point, as its own
argmax. `min_p` is relative to `p_max`, so with a silence token on top every
non-silence candidate falls below the cutoff, and the exemption re-admits the
listed silence ids. Before the sampler change the repetition penalty could not
degrade the state either, so the run had no exit: **zero escapes in 1,031
instrumented trap steps**. The consequences: 33.0% of paragraphs carried a
hole longer than one second, and 74 chunks in 1705 passages rendered no
speech at all, 11.5 minutes of text silently swallowed.

The sampler now applies the repetition penalty to silence, which closes the
trap at its source: holes 33.0% to 4.3%, mute chunks 74 to 1, cap-hit tails
over 2 s to zero, at 120 passages per arm across ten languages. This rule is
the detector for what still gets through, and for any checkpoint or
configuration where the trap re-opens.

Three triggers, integer-exact like every rule here. Any one condemns:

- **No speech at all.** Every generated token is in the silence-or-quiet
  family. Measured: mute rows are 255/255 silence tokens, a seed lottery
  (they recur at 3 of 18 re-renders), and a retry rescues 9 of 10.
- **A non-tail dead-air run** of at least `stall_run_tokens` true-silence
  tokens. Tail runs are excluded: the tail rules own the tail, and a trailing
  pause is judged against the place it sits in.
- **A ceiling overrun that is mostly silence**: `hit_ceiling` and a strict
  family majority of the row. A tail run on a cap-hit row is not a natural
  tail. The cap interrupted a stall, the rest of the text was never
  generated, and a trim would ship the missing content as if it were fixed.

**Detection is two-class, and that is essential.** The run gate counts only
true-silence ids; the run continues across the quiet family, the breath and
decay ids that render inaudible in context. Single-set counting was measured
broken: one breath token in the middle of real dead air split a 47-token run
into two below-threshold halves, and the rule missed it.

**Where the sets come from.** The manifest, top level, beside
`silence_token_ids`, because they are properties of the weights and the next
backend must not re-guess them. `silence_render_ids` is the union of two
independent render censuses (per-id median rendered energy below -80 dBFS;
both censuses agreed on 4137/4218/4299/6324/6405/6486, and 4215/6162 each
appeared in one). `quiet_render_ids` comes from per-instance RMS attribution
(at least 90% of instances quiet, at least 5 sightings), minus the silence
census. A checkpoint without the fields still loads; the rule then runs the
run trigger only, keyed to the configured `silence_token_ids`. That fallback
is degraded on purpose: 13 of that list's 31 ids render audible speech, so
whole-row membership in it proves nothing, and the whole-row and majority
triggers stay off rather than condemn real speech.

The censuses are read by the detectors and by nothing else. Widening the
sampler's `min_p` exemption to exactly the truly-silent ids was measured
harmful: pause-time share doubles. The penalty applies to everything now, so
there is no exemption list left to widen.

**How 25 was settled.** Calibrated across all ten shipping languages at 120
passages per arm: healthy interior runs top out at 13 to 19 tokens and
healthy leading runs at 11, so 25 clears everything ordinary prose produced
anywhere. 20 also clears; 25 is the shipped margin.

**Why it condemns instead of cutting.** The failure is a hole, not a tail. No
token anchor says where the missing speech should have been, and cutting at a
guess is how earlier rescues truncated whole sentences. Condemnation routes
the row into the retry ladder, exactly as `dropout` does. When the ladder
exhausts with every attempt condemned, the engine ships the best attempt
seen, fewest true-silence tokens, rather than the last: measured on the worst
voices, 30% of condemned fires exhaust the ladder, and the last attempt can
be worse than the first.

It also closes a structural hole in `desperation`: the ceiling clips a row to
4.0x its text tokens plus 40, so past 80 text tokens a cap-hit row can never
reach the 4.5x desperation threshold. The rows most certainly broken were
unreachable by the rule that means "certainly broken".

### 4. `silence_tail`: the peak-anchored rescue

| constant | value | meaning |
|---|---|---|
| `filler_min_eos_probability` | 0.05 | how confident the best stop must be before this rule is consulted |
| `trailing_filler_threshold` | 0.7 | share of the tail that must be silence |
| `trailing_silence_run_tokens` | 12 | an unbroken silence run that marks a boundary (~0.5 s) |
| `filler_max_speech_after_run` | 10 | speech past the seam that still counts as one word (~0.4 s) |

Applies to a row that **never said it was finished**. It cuts back to the step
where the model came closest to stopping. That peak is a **hint, not a verdict**,
and trusting it alone truncated whole sentences: the same showcase script that
runs 10.3 s in one narrator came back at 3.2 s in another. A voice reading a
language its tag does not match may never commit to stopping, so its best moment
of hesitation lands a third of the way in.

So the peak must be **corroborated by what it proposes to discard**. Either:

- the tail is mostly silence by share (≥ 70%), or
- the tail contains a run of ≥ 12 silent tokens **and** at most 10 speech tokens
  follow that run.

The second half of that condition prevents a false cut. Without it, a
rhetorical pause mid-tail (25 silent tokens, then 80 tokens of speech) matched
the run rule and the rescue cut the rest of the sentence off. That is
`aPauseFollowedByMoreSentenceIsNotFiller` in the fixture.

The share test alone was not enough either. A hallucinated word sits *behind*
the seam, and its burst lowered the silence ratio below 0.7, so the ugliest
tails, the audible "random word after a pause", were exactly the ones the rescue
refused to cut. Both checks are required.

Threshold 0.05 on the peak probability was set by measuring EOS-peak
distributions over the reference renders: a peak worth trusting.

### 5. `terminal_echo`: no seam to anchor on

| constant | value | path |
|---|---|---|
| `echo_strong_eos_probability` | 0.10 | strong |
| `echo_strong_max_tail` | 30 | strong (~1.2 s, two words) |
| `echo_strong_min_position_pct` | 68 | strong |
| `echo_weak_eos_probability` | 0.003 | weak |
| `echo_weak_max_tail` | 16 | weak |
| `echo_weak_min_position_pct` | 85 | weak |

A terminal chunk can end correctly and then free-run one or two extra words
**without** the silence seam rule 1 needs. Two acceptance paths:

**Strong.** Confidence ≥ 0.10, tail ≤ 30 tokens, and the peak in the last 32% of
the row. The position rule is what protects real comma and clause pauses from
being read as endings.

**Weak, and narrow.** It comes from one measured regression, *"...but a
brigand. Pass. Four."*, at `gen=124/124, bestEOS=109@0.004`. The model never
sampled a stop token, but its best (very weak) stop was just 15 tokens before
the hard ceiling. Weak confidence is not trustworthy in general. It is
trustworthy only with **all three** corroborators at once: a terminal chunk, an
actual ceiling overrun, and a tail inside the last 15% of the row.

Continuation chunks are exempt: their end is not an end.

### 6. `desperation`: when length is the evidence

| constant | value | meaning |
|---|---|---|
| `desperation_speech_per_text_token` | 4.5 | past this the row certainly contains garbage |
| `desperation_min_text_tokens` | 10 | below this the ratio means nothing |
| `desperation_min_keep_per_text_token` | 1.7 | on a cap-hit row, a cut keeping less than this per text token is condemned |

> **Measured on one voice, then confirmed across the roster.** The constants
> below come from 27 renders with the voice held constant. A sweep of 54
> renders across eighteen voice profiles puts the speech-per-text-token
> ratio at **2.07 min, 3.64 median, 12.25 max**, a wider spread than one voice
> suggested: the highest legitimate reading is 3.7 at fourteen text tokens, not
> 2.35. Every render past the 4.5 ceiling was a four-token utterance, which
> `desperation_min_text_tokens = 10` excludes by design. The two thresholds
> hold across the roster only *together*. Nineteen of the fifty-four renders
> would trip a detector that kept the ratio ceiling and dropped the minimum.

**Where 4.5 comes from.** *"It was as he expected."*, 14 text tokens, came back
as 96 speech tokens of sentence-then-dense-babble, with the stop peak at the
right **place** (45) but confidence 0.000, so every probability-gated rescue
refused. Measured real speech runs 1.75–2.35 speech tokens per text token. 4.5x
is unreachable by any legitimate read, so past it the question is no longer
*whether* to cut but *where*:

- **at the first silence run of ≥ 12 tokens that *starts* past the floor.** A
  structural boundary. On a certainly-broken row what follows needs no further
  corroboration. The run's *start* is tested rather than its end because a run
  straddling the floor belongs to the sentence, not to the tail.
- **else at the stop peak**, if it lands in a band a real read could have ended
  in: `[floor, int(desperation_band_ratio × text_tokens) + desperation_band_floor]`,
  2.6 and 12, config like every other audible constant. The band protects the
  mislabeled-language showcase row (92 generated / 26 text = 3.5x, below the
  ratio guard). A row of that kind must never be cut at a peak landing a third
  of the way in, and there such a peak fails the floor.

**Why tiny texts are exempt.** Fixed overheads, an initial breath and a final
pause, give a clean "No!" a ratio of 6+ all by itself.

**It applies to *ended* rows too.** A model that babbles past its sentence and
only then samples a stop token has forfeited the trust that stopping implies.

**The starved rescue: on a cap-hit row the cut must keep a plausible read.**
One row survived the stall fix. soren, da0028 chunk 4, seed 1234: 132 tokens
burned to the ceiling, the seam cut kept 36, and 33 of the 36 kept tokens
render near-silent through ids outside both manifest censuses. No
set-membership rule can see such a keep, so its length is the evidence. A
trim keeping fewer than `desperation_min_keep_per_text_token` speech tokens
per text token keeps less than any full read of the text, and the verdict is
condemned into the retry ladder (`suspect`), the route `stall` takes. The cut
stands as the keep: an exhausted ladder ships the trim, flagged, rather than
the untrimmed babble. The trigger needs no census, so it fires identically on
a checkpoint packed before them.

**How 1.7 was settled.** Every cap-hit desperation rescue in the
interior-stall and acceptance batteries, nine rows across four voices and
four languages, each kept chunk graded against its rendered audio. Absolute
keep length cannot separate them: a keep of 36 tokens was mute on one row (23
text tokens) and a complete short sentence on another (19). Keep per text
token can: every keep at or below 1.57 was mute or missing much of its text,
and every keep at or above 1.85 carried real speech. 1.7 sits inside that
gap, and under 1.75, the floor of the measured healthy band, so a complete
read is never condemned. Cap-hit rows only: a row that ended on its own
corroborated its trim with a stop token. The costs are asymmetric. A false
condemnation is a 2x decode on one window, and cap-hit desperation fires on
about 0.5% of windows at all. A false pass is a second and a half of silence
shipped as fixed. Zero disables the trigger.

### 7. `ended_tail`: dead air on a row that stopped properly

| constant | value | meaning |
|---|---|---|
| `ended_tail_silence_run` | 6 | silence before a blip that counts as stranding it (~0.24 s) |
| `ended_tail_blip_max` | 2 | ≤ 80 ms of "speech" is a click, not a word |
| `ended_tail_word_max` | 10 | a stray word on a terminal chunk (~0.4 s) |
| `ended_tail_keep` | 5 | pause left in place after trimming (~0.2 s) |

An ended row is trusted to have stopped where it meant to, so none of the rescues
above touch it. Three tail shapes still ship dead air. Walked backward, the
tail is `[sentence][r1 silence][burst][r2 silence]`:

- **bare dead air**: `r2 ≥ 12` (half a second) tightens to a natural pause.
- **a stranded click**: `burst ≤ 2` and `r1 ≥ 6`. A 40–80 ms click after a pause.
  The device specimen ended `.......#`, seven silence tokens and a blip. A real
  word is never 1–2 tokens, so speech above that is untouchable here.
- **a stranded word, terminal chunks only**: `burst ≤ 10` behind a full `r1 ≥ 12`
  seam. What a listener hears: the sentence finishes, half a second of silence,
  one word, stop. Prose does not resume after that much dead air with a single
  word.

Continuation chunks keep their tails. Their pauses are the sentence's rhythm.

### `suspect`: report without a cut

A row that survives every rule above but is still ≥ 4.5x its text length is
reported as `suspect`. It is **not** cut: no anchor agreed where, and cutting at
a guess is how the rescue truncated whole sentences before the corroboration
rules existed. It is not hidden either. Shipping such a row silently is how the
artifact reached listeners in the first place.

`Result.suspect` in Python; the equivalent field in every port.

---

## Does it hold in every language?

Every constant above was settled on English device traces. Voices ship for ten
languages, and two of the rules are ratios of **speech tokens per text token**,
where a text token is a grapheme. Token density per grapheme is a property of
the orthography, so an English constant is an *assumption* everywhere else. The
expensive direction of that assumption is a guard that truncates correct speech
in a language nobody measured.

So it was measured. One voice held constant across nine language tags. The
voice-to-voice spread on a single sentence is larger than the
language-to-language spread (the shipped samples run 2.3x to 6.3x on one
English line), so mixing narrators would have confounded exactly the effect
being tested. Three length classes each: a short line, a sentence, a long
sentence.

| language | short | mid | long |
|---|---|---|---|
| en | 5.60 | 2.79 | 2.06 |
| pl | 3.07 | 2.24 | 1.99 |
| de | 2.50 | 2.13 | 2.26 |
| es | **8.00** | 2.36 | 1.87 |
| fr | 2.61 | 2.09 | 1.69 |
| it | 3.55 | 2.46 | 2.02 |
| pt | 2.88 | 2.66 | 1.99 |
| nl | 3.29 | 2.50 | 2.23 |
| da | 3.57 | 2.60 | 2.52 |

**Ordinary sentences: 1.69–2.79 in every measured language**, against a
ceiling that only bites at `4.0x + 40`. Not one mid or long row comes near
either guard, and the margin is 1.4x–2.4x.

Swedish joined the roster after this sweep and has **not** been measured. It
ships on the same assumption, not on a row in this table. Its orthography sits
between Norwegian-family density and the measured range, but that is an
argument, not a measurement. Treat Swedish postprocessing as unverified until
it gets its own row.

**The one row the ceiling stopped is not a false positive.** Spanish `"No, hoy
no."`, 10 text tokens and 80 speech tokens, reached the ceiling *exactly*, which
is what "the decoder never emitted a stop token" looks like. It ran away on a
three-word phrase and the guard stopped it. An English-tuned ceiling caught a
Spanish runaway, in a language it was not tuned on.

**Short utterances are where a ratio stops meaning anything**, in every language.
Fixed overheads, an initial breath and a final pause, are a constant that a small
denominator turns into a large quotient. That is why the ceiling carries an
*additive* slack rather than being a pure ratio, and why the desperation rule
exempts texts under ten tokens outright. The short column above is the evidence
for both: it ranges 2.50–8.00 while the mid and long columns stay inside a
1.1x band of each other.

The rows are in the fixture's `language_guard` section and all five ports assert
them, so a constant that drifts, or a language that starts running hot, fails a
test rather than a listener.

## How often does it fire?

The rate is measured, not asserted. On the nine-language probe above, 27
renders, one voice, three length classes per language:

| verdict | renders | share |
|---|---|---|
| `clean`, untouched | 24 | 89% |
| `ended_tail`, dead air or a click trimmed | 2 | 7% |
| `desperation`, a runaway cut back | 1 | 4% |
| `suspect`, flagged and not cut | 0 | 0% |

Three rows in twenty-seven, on ordinary prose, from a healthy checkpoint.
**The layer is a rare-event catcher, not a filter every sample runs through.**

The `stall` rule fires on none of these 27; its measured surface is the
interior-stall study (1705 passages, 20 voices, ten languages), where the
rows it condemns are the 74 mute chunks and the interior holes that every
rule in this table reported as `clean`. With the sampler fix in place its
residual firing rate is near zero by construction; what remains is the
seed-lottery mute row, which the retry ladder rescues 9 times in 10.

Two of the three are short utterances, which is where the model has least
context and most room to add a syllable after the stop. The third is the Spanish
runaway. Nothing fired on a mid or long sentence in any of the nine languages.

This also sets the bar for a regression: if a change to the model or the
constants moves that 89% much in either direction, something happened. A rate
that climbs means either the checkpoint got worse or the rules got greedy, and
the two are told apart by which rule is doing the extra firing.

## Precedence

The order is fixed and lives in one resolver, not in each caller. A
predecessor of this layer grew five entry points, one per field bug, and left
the ordering to the call site. An order that lives in a caller is an order
the next caller gets wrong.

```
0. dropout         the row is incomplete: reported, never cut
1. repetition      exact-anchored: the only rule that knows precisely where
2. stall           condemned: dead air where speech should be, nothing to cut
3. silence_tail    peak-anchored, on a row that never stopped
4. terminal_echo   peak-anchored, no seam
5. desperation     length-anchored: the bluntest, so it goes last
6. ended_tail      only if nothing above fired, and only on an ended row
   → else suspect
```

`dropout` is numbered zero because it is not competing with the others. They all
answer "where did this row go wrong at the end", and it answers "there is not
enough row to ask that of".

The order is by **quality of evidence about where**, not by severity.
`repetition` is first because it is the only rule whose anchor is not inferred.
Every other one reads a signal that might mean something else: a stop peak the
model was unsure about, a silence run that might be a rhetorical pause, a ratio
that says the row is wrong without saying where. Peak-anchored rescues come
next, because their thresholds were settled on the bench and they at least
identify *where*. The length-anchored rule is last because it is the bluntest.

`stall` sits between the exact anchor and the inferred ones because it has no
anchor at all: it condemns without cutting, and it must run before the
rescues, not after them. A rescue that fires on a stalled row trims the tail
and ships the hole under its own reason, and on a cap-hit stall the trim
ships a row whose second half was never generated, labelled as fixed. A
repetition cut still outranks it: an exactly repeated cycle pins where the
failure began, and a cut that saves the row is cheaper than a re-render.

`desperation` can condemn as well as cut. On a cap-hit row a trim keeping
fewer than `desperation_min_keep_per_text_token` speech tokens per text token
is a starved rescue: it rides the same retry route as `stall`, with the cut
kept as the fallback if every attempt is condemned. The check runs at the
desperation slot rather than at the stall slot because it needs the cut's
position, which only the desperation rule computes.

---

## The layer boundary

| stage | question | domain |
|---|---|---|
| **preprocess** | what will the model read wrong? | text → text |
| **tts** | say it | text → tokens → mel → audio |
| **postprocess** | did what came out actually end? | tokens → verdict |

Preprocess rewrites input the model mispronounces. Postprocess inspects output
the model over-produced. Neither is a filter, and neither touches audio.

## Modes

`PostprocessConfig.mode`:

- **`trim`** (shipping): apply the cut. This changes the audio, so it is part of
  the algorithm and travels in the fingerprint like every other audible decision.
- **`report`**: run the detectors, attach the verdict, change nothing. For a
  caller who would rather hear the artifact than risk a wrong cut.
- **`off`**: skip the detectors, and with them their cost of one exponential and
  one sum over the vocabulary per decode step, for the stop-token observation.

## What the detectors need, and where it comes from

Three of the rules compare against `eos_peak_at` / `eos_peak_prob`: the step at
which the stop token came closest to winning, and how close.

It is observed **in the sampler**, not by changing `TokenGenerator.generate`.
Every backend already calls the injected sampler on every step, because the
sampler owns the RNG stream and a backend that skipped it would produce
different tokens. So the observation reaches torch, ONNX and CoreML without
touching a protocol other people have written implementations against.

The quantity is the shipped engine's, reproduced exactly: the stop token's
softmax weight over the sum of the weights that survived `min_p`. Two details
matter:

- The **numerator is taken before the cutoff**, so a step where the stop token
  was itself filtered out still reports how near it came. The number answers *how
  close was this to being the end*, not *what was the chance of stopping*. The
  first question is the one these rules need, because the rows they rescue
  are precisely the ones where stopping never won.
- The peak is recorded only **past the EOS floor**. Below it the generator masks
  the stop token to −∞, so its probability there describes the mask rather than
  the model.

Because two rules compare that probability against a threshold, it is an
*audible* value despite never feeding back into sampling. The conformance fixture
pins it for that reason.

The `stall` rule and the `repetition` exemption additionally need the two
render censuses. They come from the checkpoint manifest, top level, beside
`silence_token_ids`, and land on `PostprocessConfig.silence_render_ids` /
`quiet_render_ids`. They are hashed into the fingerprint like every other
audible value: two engines disagreeing on the census condemn different rows.

---

## Conformance

`tests/data/conformance/postprocess.json`: every case is a regression
observed in a real application or a named device trace. None was invented to
exercise a branch.

The `stall` section carries its own two-class id sets (`silence_render_ids`,
`quiet_render_ids`) and hand-built rows: a mute row, an interior run that
survives a breath token, a run one token short that must not fire, a cap-hit
majority-silence row, a pause-heavy healthy row, and the two fallback cases
for a checkpoint without the censuses. Each case pins both the bare rule and
the full resolver verdict, because the wiring is part of the contract.

The `starved_rescue` section pins the desperation floor: the da0028 survivor
shape condemned, a keep just over the floor shipping as today, the second
mute trim from the calibration set, and an ended row with the same tiny keep
shipping untouched. Its cases run through the full resolver only; the trigger
has no bare-rule function, and it runs without the censuses on purpose.

The `repetition_silence` section pins the loop exemption's acoustic family:
the en0023 shape (a period-1 run of a census-only silent id that must not
fire as a loop and must be condemned by `stall`), a genuine speech loop that
must cut exactly as before, the stutter with its pause on a census-only id,
a pause one token under the stall gate that ships whole, and a breath run
only the quiet census knows. Each case pins the bare rule and the full
resolver verdict, with the section's own censuses configured.

All five implementations run it: Python, Swift, Go, Rust and TypeScript. The
config block in the fixture is asserted to equal each port's shipping defaults
first, so the cases prove something about what actually runs rather than about a
config assembled for the test.

## Out of scope

The shipped reader also carries an **audio-domain** fallback for one specimen
where the audible seam was rendered from tokens *outside* the silence vocabulary,
so only the waveform shows it (4.22 s for a 1.8 s sentence, a 0.5 s hole at
2.08 s, babble after). That rule is not ported. It needs float thresholds on an
energy envelope, which is the parity risk this layer was designed to avoid, and
the token layer reports such a row as `suspect` rather than pretending it is
clean. If it is added later it belongs behind its own mode, with its own
tolerance-based fixture.

## Two interactions this layer has with the chunker, recorded

Both were found by reading the two campaigns against each other. The first was
recorded here as "left as it is" and then fixed in 0.1.1 — the reasoning that
kept it, and the reasoning that overturned it, are both below, because the
argument is the useful part. The second is still left as it is.

### A trim used to disarm the re-split — fixed in 0.1.1

`ChunkConfig.cap_resplit` repairs a chunk the window could not hold, and the
engine gates it on the *window* rather than on any cap. It used to read the
window off the finished row:

    hit_the_window = len(window.speech) >= window.max_speech_tokens

`window.speech` is the **trimmed** row. So a chunk that ran to the window cap
and was then cut by a rule on this page came out shorter than the window, the
gate read false, and the chunk was not re-split — the words the cap ate stayed
lost. Reachable, and not rare: a row of 255 tokens ending in a repeated cycle
takes a `repetition` verdict keeping 202, and 202 is not 255. `synthesize`'s
overflow refusal read the same trimmed length, so it did not fire either.

The engine now carries `hit_window_cap` beside `hit_token_cap`, measured on the
raw generation before any trim, and both readers use it. Postprocess can no
longer remove the evidence of an overflow it did not repair.

This was left alone once, on the argument that re-splitting rows that ship
today is audible and an audible change has to move the fingerprint. It moves
the fingerprint. What changed the answer is that the alternative is silent data
loss: text the caller wrote, never spoken, with a fluent short render and no
error. That is the one failure this project does not trade against a re-base,
and 0.1.1 re-bases the goldens anyway.

The narrower question — whether such a row lost text at all, or only
hallucinated past it — is still not measured. `hit_ceiling` says a cap fired,
not which one; `hit_window_cap` now says *which*, but not whether the text ran
out first.

### The EOS peak is a threshold, and the device computes it differently

Two rules here read `eos_peak`: `terminal_echo` compares `eos_peak_prob`
against `echo_strong_eos_probability` and `echo_weak_eos_probability`, and the
desperation fallback compares `eos_peak_at` against a position band. They are
comparisons against constants, so the quantity is audible even though it never
feeds back into sampling — which is why the conformance fixture pins it.

`DeviceSamplerV1` computes that quantity on the device, where the divisor is a
parallel sum and the host's is ordered left to right. Its docstring says so and
classifies the difference as `equivalent`, which is right about the arithmetic
and understates the consequence: a last-bit difference in a probability that is
*compared to a constant* does not stay a last-bit difference. It can flip a
verdict, and `eos_peak_at` is chosen by `prob > peak_prob`, so a tie broken the
other way moves the recorded step to a different step entirely rather than to
an adjacent value.

The blast radius is one chunk on the CUDA-graph path — the path the identity
contract already declines to call bit-exact — and `TestDeviceSampler` measures
the agreement to the last bits it can on CPU. What it cannot measure is CUDA's
reduction, which is exactly where the difference lives. A test that could would
need a card in CI. Until then this is recorded rather than claimed absent.

## Edge fade measurement

The ramp length is `AlgorithmConfig.edge_fade_seconds`, read from the manifest
key of the same name. The default is 0.02 s and is included in the canonical form as
`"edge_fade_seconds":"0.02"`. Only the historical 0.005 s is omitted for
identity compatibility. Old manifests without the field retain 5 ms; new
release manifests explicitly set 0.02, changing their fingerprint.

A 20 ms raised-cosine ramp on both ends of a rendered window. This attenuates
short onset bursts while preserving more of the speech attack than a 50 ms
ramp. It only affects window edges; it does not remove transients within speech.

Concatenated windows can meet at nonzero amplitudes, producing an audible
click. Bring both endpoints to zero while leaving the middle untouched.

A voice that starts speaking immediately can open a window mid-waveform
while the preceding window ends in silence. In the 2.5-hour book measured
at 4903016, half of 3,439 onsets after a pause carried that tap. The fade
addresses the sample discontinuity independently of how the voice was
enrolled; the later enrollment experiment at 92baeed located the relevant
silence at the end of the reference prompt, not the beginning of the clip.

**The ramp itself is pinned, not recomputed.** Python takes the cosine in
float32, off a float32 `linspace`. A port that evaluates it in double and
narrows to float32 lands within two units in the last place, under 1.2e-07 at
full scale, about -138 dBFS. That is the identity contract's `equivalent` class
rather than its bit-exact one, so both ramp lengths a release can apply, the
historical 5 ms and the shipped 20 ms, are carried in Go, Rust, TypeScript and
Swift as the reference's own float32 bits. `tests/data/conformance/edge_fade.json`
holds them, `tools/make_conformance.py --fixtures-only` writes it from
`loudkit.window._fade_edges`, and every port asserts both tables against it.
Any other length keeps the computed fallback.
