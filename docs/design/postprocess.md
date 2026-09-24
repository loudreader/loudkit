# Postprocess: rules and evidence

Postprocess is the third stage of `preprocess → tts → postprocess`. It is a
detector: it reads the speech tokens a chunk produced and returns a verdict,
which is a rule name, a keep length and a `suspect` flag. The engine applies
the cut and the retry policy. Postprocess never changes audio samples.

---

## The failure it exists for

A listener reports: *"it finished the sentence, then there was a long gap, then
one random word."*

The model generates this artifact, and an energy threshold cannot separate the
extra word from the sentence (see the next section). The mechanism:

1. The decoder is free-running. Nothing in the model guarantees that it stops
   when the sentence is over. It stops when it samples the stop token, or when
   it hits a cap.
2. Silence ids are exempt from the `min_p` cutoff, so a pause stays possible.
   Removing the exemption raised the median long-form gap from 2.46 s to
   4.64 s. The repetition penalty applies to silence ids like any other token;
   see `SamplingConfig.silence_token_ids` and the `stall` rule below.
3. After the sentence ends, silence ids stay selectable. The decoder can
   continue with silence and then sample a non-silence token, which renders as
   an extra word after a pause.

The shape in the token stream is therefore *sentence, a run of silence, a short
burst of speech*. A debug trace prints it with `.` for silence and `#` for
speech:

```
t3.rowTail  text=14 gen=96 ended=false bestEOS=45@0.000
            tail=####........................#####
```

## Why the token domain

The rules read tokens for three reasons:

- The evidence is positional. An extra word is speech, with the same spectrum,
  level and voice as the real sentence, so no energy threshold separates it
  from a real ending. Its position separates it: behind half a second of
  silence, at the end of a row that already ran long. Position is a token
  fact.
- Token counts and set membership are integers, so the five implementations
  agree exactly. An audio rule needs a float threshold on an energy envelope,
  and a rule that depends on the last bits of a float decides differently on
  different hardware. The rules that read the EOS peak are the exception; see
  [The EOS peak on the device](#the-eos-peak-on-the-device).
- Cutting tokens is a slice. Cutting audio needs a crossfade, and a crossfade
  in the wrong place is an artifact of its own.

## Batching

In the reported cases, the artifact sentences rendered clean interactively and
produced the artifact only when batched, where a short text is padded to its
longest neighbour. A server batches requests, so a server is where the artifact
appears.

---

## The rules

Seven detection rules and one generation-time guard. Every constant is a
`PostprocessConfig` field, read from the manifest and hashed into the
fingerprint. A port that loads different values reports a different
fingerprint, and the conformance fixture checks each port's defaults.

### 0. The ceiling: a guard, applied during generation

| constant | value | meaning |
|---|---|---|
| `ceiling_speech_per_text_token` | 4.0 | hard stop as a multiple of text tokens |
| `ceiling_slack_tokens` | 40 | additive slack (1.6 s of audio) |

`ceiling = min(window, int(text_tokens × 4.0) + 40)`

The clamp to `window` exists because `frame_windows` refuses a row longer than
`max_speech_tokens`: a ceiling above the window only allows work that the
renderer rejects. The ceiling is applied in `trim` and `report` modes, not in
`off`.

Without the ceiling, the only limit is the 255-token window: 10.2 s of audio at
40 ms per token, for any text length. A three-word sentence could decode for
ten seconds.

The ratio 4.0 comes from a device trace of the showcase render:

```
t3.overrun  gen=92 ceiling=92 bestEOS=74@0.003 floor=31
```

A chunk of about 26 text tokens stopped only because it hit the ceiling. Its
best stop probability was 0.003, it was mid-sentence, and it already needed 3.5
speech tokens per text token. Several narrators came back at 3.2 to 4.3 s on a
script that runs 10 to 14 s on the rest of the roster. The ratio 4.0 sits above
the 3.5 of that trace. Short texts run higher (up to 8.00 in the
[language table](#language-calibration)), and the additive slack carries them.

The ceiling is separate from the chunker's estimate. The chunker budgets
characters (`CHARS_PER_TOKEN = 0.5` characters per speech token, in
`frontend/chunking.py`) and uses its estimate as an upper bound, where guessing
high only makes chunks shorter. The ceiling faces the other way: guessing low
cuts a sentence off.

The ceiling is a runaway guard, not a tempo control. The rules below choose
the fine cut.

### 1. `dropout`: the row is incomplete

| constant | value | meaning |
|---|---|---|
| `dropout_min_tokens` | 25 | absolute gate (~1.0 s); the proportional test must also hold |

`dropout` flags a row that is too short for its text. It keeps every token
(`keep` is the full row) and sets `suspect`, because removing tokens cannot
restore missing content. `stall` and a resumed `repetition` also keep the full
row; the other rules say that the end of the row is wrong.

A truncated render can sound fluent. An extra word draws attention to itself,
but a sentence that stops halfway can sound like a complete sentence, so
listening alone does not reliably find this failure.

Two conditions, both required:

- an absolute gate: fewer than 25 speech tokens, the threshold of a published
  criterion for catastrophic neural-codec TTS failure;
- a proportional test: fewer speech tokens than text tokens. This keeps a
  short line exempt. In the nine-language probe below, every healthy read
  produced more than 1.6 speech tokens per text token, and the shortest had
  35 tokens.

It runs first, because no later rule can repair a row that is too short.

### 2. `repetition`: an exact repeated cycle

| constant | value | meaning |
|---|---|---|
| `repetition_max_period` | 12 | longest cycle that counts as a lock-up (~0.5 s) |
| `repetition_min_cycles` | 3 | consecutive exact repeats: necessary, not sufficient |
| `repetition_min_span` | 24 | tokens the repeating region must cover (~1.0 s) |
| `repetition_silence` | acoustic | which silence family the all-silence-cycle exemption reads |
| `repetition_resume` | condemn | a loop followed by a full period or more of other tokens is condemned whole |

The tail rules read the end of a row. A stuck decoder can repeat inside the
row, where a tail rule does not look.

The mechanism is the one behind the extra word: the model's own output becomes
its context. Published work reports this failure. VALL-E's authors describe
greedy search "continually generating silence codec codes", and the Very
Attentive Tacotron stress test produced 52 repetitions of a phrase meant to
occur nine times.

The anchor can sit anywhere in the row, so the conditions are strict:

- The cycle is short: at most `repetition_max_period` tokens. A longer
  repeated block is a phrase, and a repeated phrase can be rhetoric: a
  refrain, a stammer, "no, no, no".
- The repeat is long: the repeating region covers at least
  `repetition_min_span` tokens. This condition separates the classes; its
  calibration is below.
- The match is exact. Approximate repetition does not count. The rule is tuned
  against false positives, because its cut removes speech.
- The cycle is not all silence. A cycle made only of silence-family ids is not
  a loop: the tail rules and `stall` judge pauses. With `repetition_silence` at
  its default `acoustic`, the family is the union of the sampler's
  `silence_token_ids` and both render censuses (`silence_render_ids`,
  `quiet_render_ids`), so a pause on any census id is exempt. A cycle that
  mixes silence with speech still counts: the word-then-pause stutter is one
  shape of this failure.

The cut keeps one full cycle after the loop's start, so one instance of the
repeated fragment survives. Under the default `repetition_resume: condemn`, an
applied cut removes only a tail: the loop plus fewer than one period of tokens
after it. A loop followed by more is condemned whole; see
[the resumption guard](#the-resumption-guard).

#### Calibration of `repetition_min_span`

A rule without a span condition, "a short cycle repeated at least four times",
has the shape of the published inline guards: VALL-E 2's repetition-aware
sampling watches a 10-token window, and MSpoofTTS scans at segment lengths
10/25/50. On this checkpoint that rule fired on 22 of 27 healthy renders (81%).

The cause is specific to this checkpoint. It winds down each utterance with a
short repeated tail token, usually `6405` or `6486`, and those ids are not in
the 31-id `silence_token_ids`, so the all-silence exemption of that list does
not cover them. Short repetition is present in almost all real speech here.

A stuck decoder differs because its repetition does not stop. Measured across
the same 27 renders, in nine languages, three length classes, one voice held
constant:

| | longest repeating span |
|---|---|
| healthy rows (n=26) | 10 tokens (0.4 s), median 7 |
| the one runaway in the set | 44 tokens (1.76 s) |

24 tokens sits between them, with 2.4x margin over the healthy maximum. With
it, the rule fires on 0 of 27.

Limits of this calibration:

- The runaway in the table repeats a silence token, so this rule declines it
  and `desperation` handles that row. The set therefore contains no observed
  true positive.
- The inline-guard parameters were designed to steer a sampler, not to judge a
  finished row. Used without re-measuring, they gave the 81% false-positive
  rate.

The rule guards against a failure not yet observed on this checkpoint. It is
calibrated for no false positives, not for sensitivity.

#### en0023 and the silence family

With `repetition_silence: sampling`, the exemption reads only the sampler list,
and a pause on a census-only silent id fires as a loop. The specimen,
measured in August 2026: kathleen, en0023, seed 1234. A mid-chunk pause sits on
6486 (x7) then 6405 (x24). Both ids are in the manifest's
`silence_render_ids`, and neither is in `sampling.silence_token_ids`, so the
24-token period-1 run passed every loop condition. The cut kept one cycle and
deleted the pause plus the six seconds of correctly read speech behind it: two
whole sentences, verdict `repetition`, not suspect, no retry, audibly fluent. A
report-mode render showed that the raw row carried the full text. `stall` would
have condemned the row (31 census-silent tokens against a gate of 25), but
`repetition` runs first. Six of the checkpoint's eight silent-rendering ids are
outside the sampler list. Measured prevalence: 1 in 200 renders. The result is
content loss that sounds fluent and carries no flag.

With `acoustic` (the default), the exemption reads the union of the sampler
list and both render censuses. The rule resolves the union from the config,
the same way `stall` reads its censuses. The pause is then not a loop, the row
falls through to `stall`, and the retry ladder re-renders it. Without censuses
the union equals the sampler list, and [the resumption
guard](#the-resumption-guard) protects the row. `sampling` names the
sampler-list family for a manifest measured under it.

The `repetition_silence` fixture section covers the specimen shape, a genuine
speech loop, the stutter with its pause on a census-only id, a pause one token
under the stall gate, and a breath run that only the quiet census knows.

#### Two silence lists

`repetition` and `stall` read acoustic silence: one to decline a loop, one to
condemn a hole. The tail rules (`silence_tail`, the `desperation` seam and
`ended_tail`) read the sampler's `silence_token_ids`. Their thresholds were
calibrated against that list, and the resulting tail of about 0.5 s measures
at Kokoro's level. Re-keying them would change prosody on every render, and
no defect in them has been shown. `terminal_echo` reads no silence list.

#### The resumption guard

The acoustic union needs a manifest that names the silent ids. On a manifest
without censuses, the union equals the sampler list and the same pause fires
again. Measured on a census-less pack: kathleen, en0023, seed 1234, the same
mid-chunk pause (a 6486 run, x8, into a 6405 run of 24). The period-1 run
passes every loop condition, and the cut keeps 52 of 206 tokens. It deletes
the pause and the 131 tokens of correctly read speech behind it: two whole
sentences, verdict `repetition`, no retry, audibly fluent. Measured prevalence:
about 1 in 200 renders.

The guard reads no silence list. A genuine lock-up is a tail pathology: the
model's own output is its context, the state is absorbing, and the repeating
region runs to the end of the row. Every firing case in the fixture resumes by
zero tokens. A ceiling can truncate at most one incomplete copy, which is
`period - 1` tokens, at most 11. The specimen resumes by 131.

The rule, with `repetition_resume` at `condemn` (the default): a qualifying
loop whose region ends a full period or more before the row does is read as a
decoder that resumed, which a locked decoder does not do. The row is condemned
whole into the retry ladder: verdict `repetition`, `suspect`, `keep` equal to
the full row. No trim is kept as a fallback. If every attempt is condemned, the
engine ships the attempt with the fewest true-silence tokens, whole, which for
a pause is the correct audio. A loop whose region reaches within one period of
the row's end is cut as usual.

The test is the integer comparison `resume >= period`, with no constant to
tune. A matching full copy would count as another cycle, so a full period of
tokens after the region always contains a deviation.

A discard-fraction guard was calibrated and rejected. The measured
distributions:

| population | discard fraction | resume (tokens) |
|---|---|---|
| firing fixture cases (5: three `repetition`, two `repetition_silence`) | 0.36 to 0.58 | 0 |
| the en0023 specimen, census-less | 0.75 | 131 |

A cap of 2/3 separates the measured sets, but the discard fraction depends on
where the pause sits in the row. A 30-token pause at 70% of a 206-token row,
with one clause behind it, discards 0.32. That is under any cap the firing
cases allow, so the cut would delete the clause. A fraction cap limits how much
is lost; the resumption test prevents the loss. That case is a unit
construction, not a render, but the pause position depends on the text, so a
real render can have it.

Two consequences:

- Under `condemn`, an applied repetition cut removes only a tail, like every
  other rule in this layer. The loop anchor still runs first, but a loop the
  decoder resumed from is handed back whole.
- The guard runs under every configuration, not only without censuses. A pause
  on an id outside both censuses (the da0028 class) is also covered on a
  manifest that carries them. The resumption test does not depend on a
  complete silence list.

`repetition_resume: "cut"` turns the guard off, for a checkpoint measured
without it. The `repetition_resume` fixture section covers the specimen at its
measured proportions, a genuine loop, both sides of the boundary (`period - 1`
cuts, `period` condemns), the one-token resumption of a period-1 pause, and a
run to the row's end that still cuts.

### 3. `stall`: the decoder trapped in silence

| constant | value | meaning |
|---|---|---|
| `stall_run_tokens` | 25 | non-tail dead-air run that condemns a row (~1.0 s) |
| `silence_render_ids` | manifest | ids that render true digital silence: the run gate |
| `quiet_render_ids` | manifest | breath and decay ids: extend a run, never count toward it |

The rules after `stall` read the end of the row. A stalled decoder leaves a
hole in the middle of the row, or produces no speech at all. In the study
below, every such row was `clean` under the other rules.

The mechanism, measured at scale (interior-stall study, August 2026: 1705
passages, 20 voices, ten languages): the decoder enters a silence run at a
pause point, as its own argmax. `min_p` is relative to `p_max`, so with a
silence token on top by a wide margin, the non-silence candidates fall below
the cutoff (on 87% of stall steps in an instrumented trace), and the exemption
re-admits the listed silence ids. With the repetition penalty
also skipping silence, nothing degraded that state: zero escapes in 1,031
instrumented trap steps. 33.0% of paragraphs carried a hole longer than one
second, and 74 chunks in 1705 passages rendered no speech at all, 11.5 minutes
of mute audio in place of text.

The sampler applies the repetition penalty to silence. That change took holes
from 33.0% to 4.3%, mute chunks from 74 to 1, and cap-hit tails over 2 s to
zero, at 120 passages per arm across ten languages. `stall` detects what still
gets through, and covers any checkpoint or configuration where the trap
returns.

Three triggers; any one condemns the row. All three are integer tests.

- No speech at all: every generated token is in the silence-or-quiet family.
  Needs the census. Measured: mute rows are 255 of 255 silence tokens, they
  recur in 3 of 18 re-renders with other seeds, and a retry rescues 9 of 10.
- A non-tail dead-air run of at least `stall_run_tokens` true-silence tokens.
  Quiet ids extend the run without counting. A leading run counts; a tail run
  does not, because the tail rules judge a trailing pause against its
  position.
- A ceiling overrun that is mostly silence: `hit_ceiling` and a strict family
  majority of the row. Needs the census. On a cap-hit row, a silence majority
  shows that the cap interrupted a stall. A tail trim would ship the row as
  fixed while text may be missing.

Detection is two-class. The run gate counts only true-silence ids. The run
continues across the quiet family, the breath and decay ids that render
inaudible in context. With a single set, one breath token in the middle of
real dead air split a 47-token run into two halves below the threshold, and
the rule missed it.

The sets come from the manifest, at top level beside `silence_token_ids`,
because they are properties of the weights. `silence_render_ids` is the union
of two independent render censuses (per-id median rendered energy below
-80 dBFS). Both censuses agreed on 4137/4218/4299/6324/6405/6486, and
4215/6162 each appeared in one. `quiet_render_ids` comes from per-instance RMS
attribution (at least 90% of instances quiet, at least 5 sightings), minus the
silence census.

A manifest without the fields still loads. The rule then runs the run trigger
only, keyed to the configured `silence_token_ids`. The whole-row and majority
triggers stay off: 13 of that list's 31 ids render audible speech, so
whole-row membership in it does not identify a mute row, and those triggers
could condemn real speech.

The censuses feed only the detectors. The sampler keeps `silence_token_ids`
for its `min_p` exemption, and its repetition penalty applies to every token.
Using the census ids for the `min_p` exemption was measured to double the
pause-time share; see the open questions in
[silence classes](silence-classes.md#open-questions).

`stall_run_tokens` was calibrated across ten languages at 120 passages per
arm: healthy interior runs top out at 13 to 19 tokens and healthy leading runs
at 11. Both 20 and 25 clear those maxima; 25 is the shipped value.

#### Why `stall` condemns

A stall leaves a hole inside the row, and a tail cut cannot fill it. No token
anchor says where the missing speech belongs, and a cut at a guess truncates
whole sentences. The row goes to the retry ladder, as a `dropout` row does.
When every attempt is condemned, the engine ships the attempt with the fewest
true-silence tokens (earliest on a tie), not the last. Measured on the worst voices, 30% of condemned fires
exhaust the ladder, and the last attempt can be worse than the first.

The majority trigger also reaches rows that `desperation` cannot. The ceiling
clips a row to 4.0 × text tokens + 40, so past 80 text tokens a cap-hit row
cannot reach the 4.5 desperation ratio.

### 4. `silence_tail`: the peak-anchored cut

| constant | value | meaning |
|---|---|---|
| `filler_min_eos_probability` | 0.05 | the EOS peak must exceed this before the rule is consulted |
| `trailing_filler_threshold` | 0.7 | share of the tail that must be silence |
| `trailing_silence_run_tokens` | 12 | an unbroken silence run that marks a boundary (~0.5 s) |
| `filler_max_speech_after_run` | 10 | tokens allowed around the qualifying runs (~0.4 s) |

`silence_tail` applies to a terminal chunk whose row did not end on the stop
token. It cuts back to the EOS peak, the step where the model came closest to
stopping, when all of these hold:

- `eos_peak_prob > filler_min_eos_probability`;
- the peak lies after the earliest cut (`max(floor, 10)`) and before the end
  of the row;
- the tail after the peak passes the filler test.

The peak alone is not enough evidence. Cutting at the peak alone truncated
whole sentences: the same showcase script that runs 10.3 s with one narrator
came back at 3.2 s with another. A voice reading a language its tag does not
match may never sample the stop token, and its EOS peak can land a third of
the way into the text.

The filler test accepts the tail if either condition holds:

- at least 70% of the tail is silence; or
- the tail holds a silence run of at least 12 tokens, and at most 10 tokens lie
  before the first such run, between consecutive such runs and after the last
  one (measured from the point where each run reaches 12 tokens).

The distance limits prevent a false cut. Without them, a rhetorical pause
mid-tail (25 silent tokens, then 80 tokens of speech) matched the run test, and
the cut removed the rest of the sentence. The fixture case is "a pause followed
by more sentence is not filler" in the `trailing_filler` section.

The run test exists because the share test misses the target shape: an extra
word behind the seam lowers the silence share below 0.7. Either test is
sufficient.

The 0.05 threshold comes from EOS-peak distributions measured over the
reference renders.

### 5. `terminal_echo`: no seam to anchor on

| constant | value | path |
|---|---|---|
| `echo_strong_eos_probability` | 0.10 | strong |
| `echo_strong_max_tail` | 30 | strong (~1.2 s, two words) |
| `echo_strong_min_position_pct` | 68 | strong |
| `echo_weak_eos_probability` | 0.003 | weak |
| `echo_weak_max_tail` | 16 | weak |
| `echo_weak_min_position_pct` | 85 | weak |

A terminal chunk can end its sentence and then free-run one or two extra words
without the silence seam that `silence_tail` needs. `terminal_echo` cuts at the
EOS peak on terminal chunks only. The peak must lie after the earliest cut and
before the end of the row. Two acceptance paths:

- The strong path requires an EOS probability of at least 0.10, a tail of at
  most 30 tokens, and the peak in the last 32% of the row. The position rule
  keeps real comma and clause pauses from being read as endings.
- The weak path requires a ceiling hit (`hit_ceiling`), an EOS probability of
  at least 0.003, a tail of at most 16 tokens, and the peak in the last 15% of
  the row.

The weak path comes from one measured regression, *"...but a brigand. Pass.
Four."*, at `gen=124/124, bestEOS=109@0.004`. The model never sampled a stop
token, but its best (very weak) stop was 15 tokens before the hard ceiling. A
weak peak alone is not evidence; the path accepts it only with all of its
other conditions.

Continuation chunks are exempt: a continuation chunk ends mid-passage.

### 6. `desperation`: when length is the evidence

| constant | value | meaning |
|---|---|---|
| `desperation_speech_per_text_token` | 4.5 | ratio at or above which the row gets a length-anchored cut |
| `desperation_min_text_tokens` | 10 | texts shorter than this are exempt |
| `desperation_min_keep_per_text_token` | 1.7 | on a cap-hit row, a cut keeping less than this per text token is condemned |

The constants come from 27 renders with one voice. A later sweep of 54 renders
across eighteen voice profiles measured the speech-per-text-token ratio at
2.07 min, 3.64 median and 12.25 max, a wider spread than one voice showed. The
highest legitimate read in the sweep was 3.7, at fourteen text tokens. Every
render above 4.5 was a four-token utterance, which
`desperation_min_text_tokens = 10` excludes. On that sweep the two thresholds
hold only together: nineteen of the fifty-four renders would trip a detector
that kept the ratio and dropped the minimum.

The ratio 4.5 comes from *"It was as he expected."*: 14 text tokens came back
as 96 speech tokens, the sentence followed by dense babble. The stop peak was
at the right place (45) but with probability 0.000, so every
probability-gated rule declined it. One voice measured 1.75 to 2.35 speech
tokens per text token on real speech, and the sweep above reached 3.7. At or
above 4.5, with at least ten text tokens, the rule chooses where to cut:

- At the first silence run of at least 12 tokens that starts at or after the
  earliest cut (`max(floor, 10)`). The run's start is tested, not its end,
  because a run that straddles the floor belongs to the sentence.
- Otherwise, on a terminal chunk only, at the EOS peak, if it lies before the
  end of the row and within
  `[earliest cut, int(desperation_band_ratio × text_tokens) + desperation_band_floor]`,
  with 2.6 and 12 as the defaults. The band keeps a peak a third of the way
  into a mislabeled-language row, like the showcase row (92 generated / 26
  text = 3.5x), from anchoring a cut.

Texts under ten tokens are exempt. Fixed overheads, an initial breath and a
final pause, give a clean "No!" a ratio of 6 or more.

The rule also applies to rows that ended on the stop token: a row can run long
and still end with one.

#### The starved cut

On a cap-hit row, a desperation cut that keeps fewer than
`desperation_min_keep_per_text_token` speech tokens per text token is
condemned into the retry ladder (`suspect`), the route `stall` takes. The cut
stays as the keep: if every attempt is condemned and this attempt is chosen,
trim mode ships the trim, flagged, not the untrimmed row. The check is a
length test and needs no census.

The specimen (2026-08-29): soren, da0028 chunk 4, seed 1234. The row ran 132
tokens to the ceiling, the seam cut kept 36, and 33 of the 36 kept tokens
render near-silent through ids outside both manifest censuses. No
set-membership rule can see such a keep, so its length is the evidence. The
same window renders clean at seeds 7 and 99.

1.7 was calibrated on every cap-hit desperation cut in the interior-stall and
acceptance batteries: nine rows across four voices and four languages, each
kept chunk graded against its rendered audio. Absolute keep length does not
separate them: a keep of 36 tokens was mute on one row (23 text tokens) and a
complete short sentence on another (19). Keep per text token does: every keep
at or below 1.57 was mute or missing much of its text, and every keep at or
above 1.85 carried real speech. 1.7 sits inside that gap, and under 1.75, the
low end of the healthy band measured on one voice. On this set no complete
read is condemned; nine rows do not establish a general bound.

The check applies to cap-hit rows only: a row that ended on the stop token is
not checked. The costs are asymmetric. A false condemnation costs a retry of
one window (a 2x decode for it), and cap-hit desperation fires on about 0.5% of
windows. A false pass ships about a second and a half of silence as fixed.
Zero disables the check.

### 7. `ended_tail`: dead air on a row that stopped

| constant | value | meaning |
|---|---|---|
| `ended_tail_silence_run` | 6 | silence before a blip that strands it (~0.24 s) |
| `ended_tail_blip_max` | 2 | a burst of at most 2 tokens (80 ms) counts as a click |
| `ended_tail_word_max` | 10 | a stray word on a terminal chunk (~0.4 s) |
| `ended_tail_keep` | 5 | pause left in place after trimming (~0.2 s) |

`ended_tail` runs only on a row that ended on the stop token, and only when no
earlier rule chose a cut. Ending on the stop token does not exempt a row from
`repetition`, `stall`, `terminal_echo` or `desperation`; only `silence_tail`
requires a row that did not end.

It reads the tail backward as `[sentence][r1 silence][burst][r2 silence]` and
trims three shapes. Each trim leaves up to `ended_tail_keep` silence tokens.

- Bare dead air: `r2 >= 12` (half a second) is shortened.
- A stranded click: `burst <= 2` and `r1 >= 6`, a 40 to 80 ms click after a
  pause. The device specimen ended `.......#`, seven silence tokens and a blip.
  The rule treats a burst of one or two tokens as a click by duration alone.
  A longer burst is not cut on this path.
- A stranded word, on terminal chunks only: `burst <= 10` behind a full
  `r1 >= 12` seam. A listener hears the sentence finish, half a second of
  silence, one word, then the end.

On a continuation chunk only the stranded-word path is off; the dead-air and
click paths still apply.

### `suspect`: report without a cut

A row is flagged `suspect` and not cut when no rule chose a cut, the text has
at least `desperation_min_text_tokens` (10) tokens, and the row has at least
4.5 speech tokens per text token. No anchor says where to cut, and a cut at a
guess truncates whole sentences.

`dropout`, `stall`, a resumed `repetition` and a starved `desperation` cut set
`suspect` too. The engine retries every `suspect` row and every `dropout` row.
`Result.suspect` in Python; the equivalent field in every port.

---

## Language calibration

The ceiling, dropout and desperation constants were settled on English device
traces. Voices ship for ten languages, and two of the rules are ratios of
speech tokens per text token, where the text tokens come from a grapheme
tokenizer. The ratio depends on the orthography, so an English constant needs
a check in each language. A guard that is too tight truncates correct speech.

The probe: one voice, nine language tags, three length classes each (a short
line, a sentence, a long sentence), 27 renders. The voice is held constant
because the voice-to-voice spread on one sentence is larger than the
language-to-language spread: the voice samples run 2.3x to 6.3x on one English
line.

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

Mid and long sentences measure 1.69 to 2.79 in every language probed. No mid
or long row reaches the ceiling (4.0 × text tokens + 40) or the desperation
ratio (4.5). The ratio term alone leaves a margin of 1.4x to 2.4x, and the
slack adds more.

Swedish is not in this probe, so the ratio guards have no Swedish row here.
Treat the ratio guards as unverified for Swedish until it has one. The
starved-cut fixture includes a Swedish shape (`sv0003`).

The one row the ceiling stopped is Spanish *"No, hoy no."*: 10 text tokens and
80 speech tokens, which is the ceiling exactly, without a stop token. The
decoder ran away on a three-word phrase, and the guard stopped it.

Short utterances have high ratios in every language. Fixed overheads, an
initial breath and a final pause, are a constant, and a small denominator turns
them into a large quotient. The ceiling therefore carries additive slack, and
the desperation rule exempts texts under ten tokens. The short column ranges
2.50 to 8.00, while the mid and long columns stay within 1.69 to 2.79, a spread
of 1.65x.

The rows are in the fixture's `language_guard` section. All five ports assert
`ceiling_for` against them and check that each stored row is under its
ceiling, unless the row is marked as stopped by it. A change to the ceiling
constants fails these tests. A model that starts running long in a language
does not, because the rows are stored counts; detecting that needs fresh
renders.

## How often it fires

On the nine-language probe above (27 renders, one voice, three length classes
per language):

| verdict | renders | share |
|---|---|---|
| `clean`, untouched | 24 | 89% |
| `ended_tail`, dead air or a click trimmed | 2 | 7% |
| `desperation`, a runaway cut back | 1 | 4% |
| `suspect`, flagged and not cut | 0 | 0% |

Three of 27 rows were trimmed. These shares describe this probe, not a
production rate.

`stall` fires on none of these 27. Its calibration is the interior-stall study
(1705 passages, 20 voices, ten languages), where it condemns the 74 mute
chunks and the interior holes that the other rules left `clean`. Under the
current sampler, the measured residual is mute rows that recur with some
seeds, and one retry rescues 9 of 10.

Two of the three trimmed rows are short utterances, where the model has the
least context. The third is the Spanish runaway. No rule fired on a mid or
long sentence in any of the nine languages.

To compare a later run, use the same corpus and configuration. A change in the
clean share points at the checkpoint or at the rules. The rule that fires more
often, and the renders whose verdict changed, tell which.

## Precedence

`inspect` evaluates the rules in a fixed order and returns one verdict.
Callers use the resolver and do not assemble their own order.

```
1. dropout         too short for the text: reported whole, suspect
2. repetition      exact anchor: cut, or condemned whole if the decoder resumed
3. stall           no anchor: condemned whole, suspect
4. silence_tail    peak anchor, terminal row that did not end
5. terminal_echo   peak anchor, terminal row, no seam
6. desperation     length anchor, the least precise
7. ended_tail      only on an ended row, only if no rule above chose a cut
   then            suspect if no cut, text >= 10 tokens and ratio >= 4.5
```

`dropout` does not compete with the others. The other rules ask where the row
went wrong; `dropout` says the row is too short to ask.

The order ranks the evidence about where the row went wrong. It does not rank
severity. `repetition` comes first because its anchor is an exact repeated
cycle. Every other anchor is inferred: an EOS peak the model was unsure about,
a silence run that can be a rhetorical pause, a ratio that says the row is
wrong without saying where. The peak-anchored rules come next, because they
identify a position. The length-anchored rule is last, because its anchor is
the least precise.

`stall` has no anchor: it condemns without cutting. It runs before the tail
rules, because a tail rule that fired on a stalled row would trim the tail and
hide the hole under its own reason. On a cap-hit stall, that trim would ship a
row with missing text as fixed. A `repetition` cut runs before `stall`: an
exact cycle marks where the failure began, and a cut costs less than a
re-render.

`desperation` can condemn as well as cut. On a cap-hit row, a cut that keeps
fewer than `desperation_min_keep_per_text_token` speech tokens per text token
is a starved cut: it takes the retry route of `stall`, with the cut kept as
the fallback. The check runs at the desperation slot, not the stall slot,
because it needs the cut position, which only the desperation rule computes.

---

## The layer boundary

| stage | job | domain |
|---|---|---|
| preprocess | rewrites input text the model would read wrong | text → text |
| tts | generates tokens and renders speech | text → tokens → mel → audio |
| postprocess | inspects the generated tokens and returns a verdict | tokens → verdict |

Postprocess covers too little output (`dropout`, `stall`) as well as too much.
Neither preprocess nor postprocess changes audio samples.

## Modes

`PostprocessConfig.mode`:

- `trim` (shipping): run the detectors and apply the cut. Condemned rows
  (`dropout` or `suspect`) are retried. This changes the audio, so it is part
  of the algorithm and travels in the fingerprint like every other audible
  decision.
- `report`: run the detectors and attach the verdict without applying the cut.
  Condemned rows are still retried, so the shipped attempt can come from a
  derived seed. Set `retry_max_attempts` to 0 to inspect a single attempt.
- `off`: skip the detectors, the retries and the generation ceiling. The EOS
  observation is off too, which saves one exponential and one sum over the
  vocabulary per decode step.

## What the detectors need

Three rules read `eos_peak_at` / `eos_peak_prob`: the step where the stop token
came closest to being sampled, and the value there.

The sampler observes it, not `TokenGenerator.generate`. Every backend calls the
injected sampler on every step, because the sampler owns the RNG stream and a
backend that skipped it would produce different tokens. The observation
therefore reaches torch, ONNX and CoreML without a change to the
`TokenGenerator` protocol that external implementations use. Details:
[sampler and noise](sampler-and-noise.md#_observe_eos).

The value is the stop token's softmax weight divided by the sum of the weights
that survive `min_p`. Two details matter:

- The numerator is taken before the cutoff, so a step where the stop token was
  itself filtered out still reports how near it came. The value measures how
  close the step came to ending. It is not the probability of sampling the
  stop token after filtering. These rules need the first quantity, because
  they act on rows where the stop token was never sampled.
- The peak is recorded only after the EOS floor. Below it the generator masks
  the stop token to −∞, so its probability there describes the mask, not the
  model.

Two rules compare the probability against thresholds, and one compares the
step against a band. The value is therefore audible, although it never feeds
back into sampling, and the conformance fixture pins it.

The `stall` rule and the `repetition` exemption also need the two render
censuses. They come from the checkpoint manifest, top level, beside
`silence_token_ids`, and land on `PostprocessConfig.silence_render_ids` and
`quiet_render_ids`. They are hashed into the fingerprint like every other
audible value: two engines that disagree on the census condemn different rows.

---

## Conformance

`tests/data/conformance/postprocess.json` holds regression shapes from
observed renders and named device traces, plus hand-built cases for
boundaries, fallbacks and the resolver's wiring.

The `stall` section carries its own two-class id sets (`silence_render_ids`,
`quiet_render_ids`) and eight rows: a mute row, an interior run that survives a
breath token, a run one token short that must not fire, a cap-hit
majority-silence row, a pause-heavy healthy row, the two fallback cases for a
manifest without the censuses, and a mute row that ended on its own. Each case
checks both the bare rule and the full resolver verdict, because the wiring is
part of the contract.

The `starved_rescue` section covers the starved cut: the da0028 shape
(condemned), a keep slightly over the threshold (the trim ships), the `sv0003`
shape from the calibration set (condemned), and an ended row with the same
small keep (not checked). Its cases run through the full resolver only,
because the check has no separate function. They configure no censuses,
because the check is a length test.

The `repetition_silence` section covers the acoustic family of the loop
exemption: the en0023 shape (a period-1 run of a census-only silent id that
must not fire as a loop and must be condemned by `stall`), a genuine speech
loop that still cuts, the stutter with its pause on a census-only id, a pause
one token under the stall gate that ships whole, and a breath run that only
the quiet census knows. Each case checks the bare rule and the full resolver
verdict, with the section's own censuses configured.

Python, Swift, Go, Rust and TypeScript run the fixture. Each first asserts
that the fixture's config block equals its own defaults, then checks the
cases.

## Out of scope

An audio-domain detector is out of scope. One observed specimen had its
audible seam rendered from tokens outside the silence vocabulary, so only the
waveform shows it: 4.22 s for a 1.8 s sentence, a 0.5 s hole at 2.08 s, babble
after. An audio rule needs float thresholds on an energy envelope, which is the
parity risk this layer avoids. The token layer flags such a row as `suspect`
only when its length ratio qualifies; otherwise the row can pass as clean. An
audio rule added later belongs behind its own mode, with a tolerance-based
fixture.

## Interaction with chunk re-splitting

`ChunkConfig.cap_resplit` repairs a chunk that filled its window. The engine
gates it on the window, not on any cap. `hit_window_cap` and `hit_token_cap`
are both measured on the raw generation, before any trim. The re-split and the
single-window overflow refusal in `synthesize` read `hit_window_cap`, so a
postprocess cut does not hide a filled window. Example: a 255-token row that
ends in a repeated cycle takes a `repetition` verdict that keeps 202 tokens,
and the re-split still sees a full window.

The engine reads the raw row even though re-splitting such rows changes the
audio and the fingerprint. The alternative is silent data loss: text the
caller wrote goes unspoken, in a fluent render, without an error.

Open question: `hit_window_cap` says which cap stopped the row, not whether
the text was finished first. Whether such rows lose text or only run on past
it is not measured.

## The EOS peak on the device

Three rules read the EOS peak. `silence_tail` compares `eos_peak_prob` with
`filler_min_eos_probability`. `terminal_echo` compares it with
`echo_strong_eos_probability` and `echo_weak_eos_probability`. The desperation
fallback compares `eos_peak_at` with a position band. These are comparisons
against constants, so the value is audible although it never feeds back into
sampling, and the conformance fixture pins it.

`DeviceSamplerV1` computes the value on the device, where the divisor is a
parallel sum; the host sums left to right.
[Sampler and noise](sampler-and-noise.md#devicesamplerv1) classes the
difference as `equivalent`. That is right about the arithmetic, but a last-bit
difference in a value compared to a constant can flip a verdict. `eos_peak_at`
is chosen by `prob > peak_prob`, so a tie broken the other way moves the
recorded step to a different step, not to an adjacent value.

The effect is limited to the CUDA-graph path, which the identity contract does
not call bit-exact. A flipped verdict changes the cut or the retry of the chunk
concerned. `TestDeviceSampler` measures the agreement on CPU, to the last bits
it can. CUDA's reduction is where the difference arises, and no CI test runs
it: that needs a CUDA device in CI.

## Edge fade measurement

The ramp length is `AlgorithmConfig.edge_fade_seconds`, read from the manifest
key of the same name. The default is 0.02 s and is included in the canonical
form as `"edge_fade_seconds":"0.02"`. Only 0.005 s is omitted, for identity
compatibility. A manifest without the key gets 5 ms; release manifests set
0.02 explicitly, which is part of their fingerprint.

The fade is a 20 ms raised-cosine ramp on both ends of a rendered window. It
attenuates short onset bursts and keeps more of the speech attack than a 50 ms
ramp. It affects only window edges and does not remove transients within
speech.

Concatenated windows can meet at nonzero amplitudes, which produces an audible
click. The ramp brings both endpoints to zero and leaves the middle untouched.

A voice that starts speaking immediately can open a window mid-waveform while
the preceding window ends in silence. In a 2.5-hour book measured on
2026-09-04, half of 3,439 onsets after a pause carried that tap. The fade
addresses the sample discontinuity whatever the enrollment. An enrollment
experiment on the same date located the relevant silence at the end of the
reference prompt, not the beginning of the clip.

The ports carry the ramp as data. Python takes the cosine in float32, off a
float32 `linspace`. A port that evaluates it in double and narrows to float32
lands within two units in the last place, under 1.2e-07 at full scale, about
-138 dBFS. That is the identity contract's `equivalent` class, not its
bit-exact one. Both ramp lengths a release can apply, 5 ms and 20 ms, are
therefore carried in Go, Rust, TypeScript and Swift as the reference's own
float32 bits. `tests/data/conformance/edge_fade.json` holds them,
`tools/make_conformance.py --fixtures-only` writes it from
`loudkit.window._fade_edges`, and every port asserts both tables against it.
Any other length uses the computed fallback.
