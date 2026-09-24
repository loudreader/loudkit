# Silence classes

A listener reports one thing: "it stops for a second in the middle". The
measurement separates that report by where the silence sits and by what
causes it. A change can fix one class and leave the others, so rates are
reported per class. One worst-gap number per render hides which class moved.

## The classes

### By location

#### Head

Silence that opens a chunk. A chunk after the first carries a
six-token prefix (`ChunkConfig.prefix_tokens`) from its predecessor, and that
prefix can start it inside the previous sentence-final pause. A head can then
exceed a second. No rule trims a head: `Inspection.keep` counts the leading
tokens that survive, so every postprocess cut removes a tail. `stall` flags a
leading run of at least 25 true-silence tokens and sends the row to the retry
ladder. A shorter head is neither trimmed nor flagged.

#### Seam

What a join produces: the tail of one chunk plus the head of the
next. Chunks are joined by plain concatenation, so the sum describes the
joined audio exactly. `ended_tail_keep` (5 tokens, about 0.2 s) bounds only
the tail that `ended_tail` trims, and it counts only ids on the configured
silence list. The ids that carry about half of this model's real silence are
not on that list, so they pass through the bound. Measured tail p50 was 0.50
to 0.62 s. No single chunk looks wrong; the joined audio does.

#### Interior

A silence run the decoder free-ran mid-clause. `min_p` keeps a
candidate only if `p_i >= min_p * p_max`. When a silence token is `p_max` by a
wide margin, the cutoff removes the other candidates, and the `min_p`
exemption in `LRSamplerV1.__call__` re-admits the 31 configured silence ids.
13 of those ids render as audible speech, so the surviving distribution is not
all silence. The repetition penalty applies to silence ids, so a seen silence
id loses probability; the penalty is a constant factor and does not grow with
the run. `stall` flags a non-tail run of at least 25 true-silence tokens.

### By cause

Measured across the campaign roster on own-language natural prose. A voice can
have more than one cause.

| cause | voices | does cleaning the reference help |
|---|---|---|
| contaminated conditioning prompt | joe, kerstin, thorsten, tugao, darkman | yes: these five reach their language partner's baseline |
| cap-hit silent tails | soren, at 27% of windows | no |
| whole-silence rows | nils, soren, dave, dante; joe in the original arm (11 rows, 0 after cleaning) | no, except joe |
| engine base rate | every voice, controls included | no |

The causes separate only because head, tail, seam and interior are counted
apart. A single worst-gap number shows "nils has holes in 79% of paragraphs"
and points at his reference. His interior p95 is 0.80 s, no worse than joe's,
while his head p95 is 2.00 s and his tail p95 is over 3 s. The cause is in the
decoder.

A contaminated prompt is associated with the defect in the five voices above,
but it is not necessary for it. soren's conditioning prompt has 2 dead tokens
in 150, with a longest run of 1, and his original profile produced 31 mute
chunks, 300 s of mute output. nathalie's prompt is 25% dead tokens (38 of 150),
and she is among the cleanest voices on the roster. The decoder-side trap is
the part common to every case.

### Whole-silence rows

A whole-silence row is not a pause. It is a window of text that the model
did not speak, and it appears on first chunks as well as continuation chunks.
With a census configured, `stall` condemns such a row (every token in the
silence-or-quiet family) into the retry ladder. A trim would remove the gap by
deleting the sentence, so these rows are re-rolled.

### Content loss at the window

A chunk can fill its generation window before it has spoken all its text. This
loses content without a long silence, so it is a separate class.
`split_text` assigns text to chunks before anything is rendered, with a
character budget: `ChunkConfig.max_tokens` (255) × `CHARS_PER_TOKEN` (0.5), 127
characters. The window counts tokens, and the pace that converts characters to
tokens is known only after the voice has spoken. The next chunk starts at its
own text. With `cap_resplit: off`, the words that did not fit are lost. The
detectors use token counts and token patterns; they do not check that every
word was spoken.

## How they are measured

`research/chunk_silence_stats.py` renders chunk by chunk through
`Engine.stream`, so each chunk is measured before the join. It uses 20 ms RMS
frames, and a frame is silence below 0.012 RMS. Head and tail are the leading
and trailing quiet runs. Interior is the longest run strictly between them,
reported as zero below 180 ms. A seam is one chunk's tail plus the next
chunk's head. A chunk with no voiced frame is counted as mute, apart from the
pause classes, with its duration. Head percentiles exclude the first chunk of
each passage, because a first chunk carries no prefix and cannot show the
seam-head failure. Tail and interior statistics include all chunks.

Two rules for the numbers:

- Report the passage, chunk and seam counts with every rate. A one-in-twenty
  failure measured over twenty paragraphs is noise. About 120 passages give
  about 500 chunks and 400 seams, enough to separate a 25% rate from a 5% one.
- Check the silence floor. Each chunk records its own floor. An absolute
  threshold separates speech from silence only while the silence sits well
  below it: a voice that renders with audible room tone would have its pauses
  counted as speech and its gap rate reported as zero. The comparison warns
  when the median floor comes within 12 dB of the threshold.

Compare within a language. The campaign roster had two voices per language, so
each language was a controlled pair: same corpus, same frontend, only the
profile differs. Rates across languages are not comparable, because the
public-domain corpora differ in period and orthography, and that difference
lands in the rate. The current roster has more voices than the campaign
roster.

A mute chunk's duration is the length of the silent audio produced in place of
its text. It is not a measure of how long the missing speech would have been.

## The rules that act on them

Details and constants are in [postprocess](postprocess.md) and
[sampler and noise](sampler-and-noise.md).

- The sampler's repetition penalty applies to every seen token, silence
  ids included. Silence ids stay exempt from the `min_p` cutoff, with the
  31-id `silence_token_ids` list.
- `stall` condemns a row into the retry ladder and never cuts it. Three
  triggers: a row with no speech; a non-tail run of at least 25 true-silence
  tokens, where breath and decay ids extend the run without counting; a cap-hit
  row that is majority silence. The censuses (`silence_render_ids`,
  `quiet_render_ids`) are optional top-level manifest fields. Without them,
  only the run trigger fires, keyed to `silence_token_ids`.
- When every attempt is condemned, the engine ships the
  attempt with the fewest true-silence tokens (the sampler list if no census
  is configured), earliest on a tie. This minimises one count; it does not
  guarantee complete speech.
- On a cap-hit row, a desperation cut that keeps
  fewer than 1.7 speech tokens per text token is condemned as `suspect`. The
  cut stays as the fallback if every attempt is condemned.
- With `repetition_silence: acoustic`, the all-silence exemption of the
  repetition rule reads the union of the sampler list and both censuses, so a
  pause on a census-only id is not cut as a loop. The pause falls through to
  `stall`.
- With `repetition_resume: condemn`, a loop followed by a full period or more of
  other tokens is condemned whole, not cut. This guard needs no census.
- The tail rules (`silence_tail`, the `desperation` seam and `ended_tail`) stay
  keyed to the sampler list they were calibrated against. The resulting tail
  of about 0.5 s measures at Kokoro's level.
- With `chunking.cap_resplit: word`, a chunk whose generated row filled the
  window before any trim (`hit_window_cap`) is generated again as two halves,
  split at the word boundary nearest the middle (`window.windows_for_chunk`).
  A length-ceiling hit does not trigger it, and neither does a chunk with no
  word boundary. The split is not recursive: a half that fills its window
  ships as it is. `off` disables it.
- `tools/check_voice.py` probes a new voice profile for trap propensity; see
  [the gate](#the-enrollment-gate).

### Re-split seeds

Both halves keep the original chunk's index, so a re-split cannot move the
seed of any later chunk. That preserves the seeds and nothing more. Chunk k+1
is conditioned on the last `prefix_tokens` tokens of chunk k. After a re-split
those tokens come from the second half, so the audio of later chunks in that
passage can change. The change stays inside the passage that was re-split.

The second half draws from `_STREAM_RESPLIT` (4096) off the chunk's own seed,
clear of the flow at 1, the vocoder at 2 and the retry ladder from 8 up.

The split is at the nearest word break, and punctuation is not sought. In the
comparison below, splitting at the separator nearest the middle put every long
seam on a comma. The working explanation: a comma cues a pause, and the model
has no learned bound on pause length, so it overshoots.

### The enrollment gate

`tools/check_voice.py` measures one statistic from the stall forensics: the
band-B share of emitted silence. The two silence bands are one FSQ
quantizer cell split at the +2187 digit boundary: band A is
4137/4215/4218/4299 and band B is 6162/6324/6405/6486 (4137 + 2187 = 6324,
4218 + 2187 = 6405, 4299 + 2187 = 6486). In the calibration sample, voices that
spread their pauses across both bands ended a pause in 9 to 12 tokens, and
voices that concentrated on band A repeated one shape and free-ran.

The probe renders four passages in the voice's own language at seed 1234,
token phase only, no audio. It turns postprocess off and restores the old
penalty exemption for silence in a sampler subclass local to the tool, so it
measures the voice without the sampler mitigation. It takes about two minutes
on a laptop.

Verdicts, calibrated on the twenty campaign voices:

- warn first, when fewer than 40 silence tokens were emitted: too little
  evidence to judge;
- fail for a band-B share under 0.10, a true-silence run of 100 tokens or
  more, or a cap-hit row that is majority silence;
- warn for a share under 0.30;
- pass otherwise.

A warn suggests reviewing and cleaning the reference. The tool's cleaning
recipe strips leading and trailing silence and caps internal silence runs at
100 ms, with a threshold derived from the file's own statistics
(speech p90 - 18 dB). It took joe from warn to pass. For a profile that still
fails after cleaning, record a new reference.

The gate ranks trap propensity, not gap rate. Four passages at one seed can
move a borderline voice across a line. The calibration corpus was each voice's
own language, and the thresholds were not re-measured on the eight voices
added to the roster after calibration. The engine base rate and whole-silence
rows are decoder properties that every profile shares, so no enrollment gate
reaches them.

## Evidence

The campaign ran in August 2026 on the loudr-1 checkpoint of that month. The
fingerprint of each measured state is in
[the table at the end](#fingerprints-of-the-measured-states).

### The instrumented trap

Measured with an instrumented sampler on two stalled renders, under the
original sampler (silence exempt from both the penalty and `min_p`). On 87% of
stall steps, the cutoff removed every non-silence candidate, and the exit mass
outside the exemption was exactly 0.0. The audible listed ids carried a median
of about 1e-3. Across 1,031 hard-trap steps there were zero escapes. The exit
probability was effectively zero, not exactly zero, and with the penalty
exemption nothing degraded the state over time.

Under that sampler, none of the postprocess rules reached a head or an
interior run. They were all anchored on the tail or on the EOS peak, and the
repetition rule excluded all-silence cycles.

### The silence list against the weights

Method: render, align tokens to audio (one token is exactly 0.04 s, so the
mapping carries no drift), and take the median frame RMS per token id. Over
11,479 emitted tokens across four voices in three languages:

- The list misses silence. The ids that render as true dead air, at -103 dBFS
  or below, are `4137, 4215, 4218, 4299, 6324, 6405, 6486`. Only 4137 and 4218
  are listed. The five unlisted ids carry 677 of 1479 silence occurrences,
  46%, and 6405 alone appears 330 times, comparable to the most frequent listed
  id.
- The list includes speech. Of the 31 listed ids, 16 were not emitted in this
  sample. Of the 15 that were, only those same two render as silence; the
  other thirteen render as audible content, including 3377 at -19.4 dBFS, an
  ordinary speech level.

The 13 audible ids are in the `min_p` exemption, so they stay selectable on
every step, for every voice. That keeps audible candidates in the
distribution; it does not mean that each one is sampled. Under the original
sampler, the 46% of real silence on unlisted ids was already penalised, and
long runs still occurred.

Two categories stay separate. The seven ids above are digital silence, what a
token-domain trim rule may cut. A wider set of ids renders between -78 and
-39 dBFS: breath and decay. They fall under the absolute RMS gap threshold,
and a listener hears them as part of the gap. Cutting them in the token domain
would remove breath and clip word onsets.

### Reference cleaning: joe

Re-enrolling joe from a silence-cleaned reference, measured over 200 passages,
844 chunks and 644 seams per profile, same texts and seeds. "Original" is the
roster profile; "cleaned" is the re-enrolled one.

| | original | cleaned |
|---|---:|---:|
| passages with any gap over 1s | 25.0% | 4.5% |
| seams over 1s | 6.1% | 1.1% |
| interiors over 1s | 2.7% | 0.2% |
| head p95 | 0.46s | 0.22s |
| seam p95 | 1.14s | 0.80s |
| interior p95 | 0.78s | 0.46s |

The share of seams in the natural pause band, 0.4 to 0.8 s, rose from 79.6% to
92.4%, while the share past 1.5 s fell from 4.0% to 0.3%. These figures
describe a redistribution of seam durations; a change that suppressed pausing
would have thinned the band instead. Cutting the long tail while keeping the
band is the target shape for any change here.

### How silence enters the prompt

`TorchVoiceEnroller.enroll` tokenizes the raw, untrimmed first 6 seconds of the
reference into `cond_prompt_tokens`. Only the speaker-embedding path trims
(`_VoiceEncoder.embed`, through `librosa.effects.trim`). A reference with
digitally dead silence therefore puts silence tokens into the conditioning
prompt. In a component-swap experiment, swapping only `cond_prompt_tokens`
between two voices moved the failure with it; swapping only the speaker
embedding did not.

Both the prompt and the sampler were implicated. Runs began with silence as the
model's own argmax, and the sampler exemptions kept them from ending. Cleaning
some references reduced the failures, and the sampler change reduced them
further; neither alone removed every observed failure.

### Whole-silence rows on nils

In a snapshot of the nils original arm (365 chunks, 120 passages), 8 chunks
were mute, 76 s of mute output, and re-enrolling moved that to 6. Every mute
row passed the postprocess rules of the time as `clean`, because each rule
looked for where speech stopped and there was none. The campaign also recorded
his original-arm tail p95 as 3.32 s. The matched table below gives 10 mute
chunks (92 s) and a tail p95 of 3.62 s for the same arm.

### Reference cleaning at scale

Five voices, 120 passages per arm (joe 200), both arms of a voice rendered from
the same texts and seeds. "Mute" counts chunks that produced no speech, with
the total seconds of mute output.

| voice | arm | head p95 | tail p95 | seam p95 | seams over 1s | mute | passages with a gap |
|---|---|---:|---:|---:|---:|---:|---:|
| joe | original | 0.46s | 0.68s | 1.14s | 6.1% | 11 (duration not recorded) | 25% |
| joe | cleaned | 0.22s | 0.68s | 0.80s | 1.1% | 0 | 4% |
| kerstin | original | 0.36s | 0.78s | 1.00s | 4.9% | 0 | 24% |
| kerstin | cleaned | 0.26s | 0.78s | 0.94s | 2.3% | 0 | 8% |
| dave | original | 5.28s | 7.54s | 9.60s | 21% | 20 (197s) | 65% |
| dave | cleaned | 0.46s | 4.30s | 5.48s | 10% | 4 (33s) | 35% |
| nils | original | 2.00s | 3.62s | 6.66s | 34% | 10 (92s) | 79% |
| nils | cleaned | 1.80s | 0.78s | 3.94s | 23% | 6 (50s) | 63% |
| soren | original | 10.18s | 7.34s | 10.68s | 39% | 31 (300s) | 83% |
| soren | cleaned | 3.04s | 5.38s | 7.42s | 18% | 11 (102s) | 53% |

soren's 95th-percentile chunk opening is 10.18 s, a full window: in one chunk
in twenty he does not begin speaking. Cleaning his reference cut his mute
output from 300 s to 102 s.

Reference cleaning brought joe and kerstin to a healthy rate. It halved the
rates of dave and soren and left both with large residual failures. Cleaning
lowers the probability of entering a pause; it does not change the state that
the decoder gets stuck in once it is there.

### Voices without prompt contamination

Control voices with zero dead prompt tokens still produced gaps over a second
on own-language prose. One case was a run of 103 consecutive silence tokens on
ordinary narrative text. Dead prompt tokens are therefore not required for a
stall, and reference cleaning does not reach every observed failure.

### Repetition penalty on silence, original profiles

This comparison applied the repetition penalty to silence ids, kept the
`min_p` exemption, and used the original profiles without re-enrolment.
Passages with any gap over 1 s, matched pairwise against the original arm on
the passages that both arms rendered:

| voice | passages matched | original | reference cleaned | penalty on silence |
|---|---:|---:|---:|---:|
| dave | 120 | 65% | 35% | 0.8% |
| soren | 120 | 83% | 53% | 16% |
| nils | 120 | 79% | 63% | 25% |
| joe | 77 | 26% | not measured | 1.3% |

On nils and soren, 25% and 16% of paragraphs still carry a seam over a second,
against 0 to 8% for a healthy voice. On nils, interior runs over a second went
from 16 to zero, tails over two seconds from 29 to zero, and mute chunks from
10 to zero (92 s of mute output to none). The worst seam fell from 6.66 s to
1.14 s. The remaining defect is a long pause at a join.

Mute output across the original, cleaned-reference and penalty arms: dave
197 s, 33 s, 0; nils 92 s, 50 s, 0; soren 300 s, 102 s, 1 s. Chunk tails over
two seconds went to zero on dave and soren, from 54 and 78. Interior runs over
a second went to zero on dave.

The nils penalty-arm rate climbed from 14.8% at 61 passages to 19.1% at 68 and
25% at 120, monotonically. A partial run can misstate a rate, so each rate
here carries its passage count.

Prosody checks:

- A pause-time share of 0.213 before and 0.222 after came from six renders of
  two healthy voices. Its per-seed spread is wider than the difference, so it
  does not resolve a prosody effect.
- On healthy voices, matched pairs of about 30 passages each: natural-band
  pause counts held within 2% in both directions (kathleen 21.4 to 21.1 per
  minute, paola 16.3 to 16.2, nathalie 18.8 to 19.1). Quiet share and speaking
  rate were flat.
- The French control at 80 passages was unchanged to within 0.02 s on every
  percentile, and lost its one residual gap.
- On the damaged voices, the natural band grew denser. Seams under 0.3 s, the
  run-on signature, fell on every voice. Across a ten-voice sweep, postprocess
  verdicts fell: `desperation` 5 to 0, `terminal_echo` 1 to 0, suspect events
  14 to 1.

Two delivery changes:

- The silence that opens a continuation chunk shrank: the median went from
  0.16 to 0.24 s down to 0.00 to 0.08 s, and 88% of dave's continuation chunks
  opened with under 0.1 s. Seam totals stayed inside the natural band, because
  the preceding tail still carries the pause.
- henri's pauses were mildly compressed. His natural-band pauses fell from
  23.94 to 22.19 per minute, while pauses of 0.10 to 0.18 s rose from 11.06 to
  12.91. The near one-for-one transfer is consistent with shorter pauses
  crossing the counting floor. His articulation rate held (voiced words per
  minute 253.4 to 253.1), so the faster reading is silence coming out, not
  speech speeding up. kathleen shows no such transfer and freja's band holds.
  The effect is about 7% of henri's natural band. A listening review of three
  French passages, before and after, accepted it on 2026-08-29.

The penalty is a constant factor per seen token, not an accumulating one, so a
strong silence preference can still hold a run. One soren chunk under the
penalty free-ran 133 consecutive true-silence tokens to the 255-token cap,
verdict `clean`, leaving a 5.6 s hole. The third `stall` trigger condemns that
shape.

Putting the census ids into a sampler exemption instead doubled the pause-time
share, to 0.400, and degraded a healthy control voice. See the open questions
for which exemption.

### The sampler change at scale (August 2026)

Before the change, across 20 voices in ten languages (1705 passages): 33.0% of
paragraphs carried a hole over 1 s, and 74 chunks rendered no speech at all,
11.5 minutes of mute output. With the penalty change alone, at 120 passages
per arm across ten languages:

- paragraphs with a hole over 1 s fell from 33.0% to 4.3%;
- mute chunks fell from 74 to 1;
- cap-hit tails over 2 s and interior runs over 1 s fell to zero;
- natural-band pause rates stayed inside noise on 8 of 9 healthy voices;
- WER was flat or better;
- henri's natural band fell 8.5%, with a near one-for-one transfer to the
  sub-band and no change in articulation or WER.

Removing the `min_p` exemption instead raised the median long-form gap from
2.46 s to 4.64 s.

`recipe_version` stayed `loudkit-1`. The loader refuses a manifest that names
any other recipe, so a rename would make every published pack unloadable. The
moved fingerprint and the new hashed config fields already separate the two
sampling laws.

`stall` calibration: healthy interior runs top out at 13 to 19 tokens and
healthy leading runs at 11. With a single silence set, one breath token split
a 47-token run into two halves below the threshold.

On the worst voices, 30% of condemned fires exhausted the retry ladder, and the
last attempt can be worse than the first.

### The starved cut (2026-08-29)

The acceptance run left one mute chunk: soren, da0028 chunk 4, seed 1234. The
row ran 132 tokens to the ceiling, and the desperation seam cut trimmed it to
36 tokens, 1.44 s in which 33 of the 36 kept tokens render near-silent through
ids outside both censuses. No set-membership trigger could see it, and a
desperation cut was not a condemned verdict, so it shipped on attempt 1. Seeds
7 and 99 render the same window clean.

The threshold 1.7 was calibrated on all nine measured cap-hit desperation cuts,
graded against their rendered audio: bad keeps top out at 1.57 speech tokens
per text token, healthy ones start at 1.85, and 1.75 is the low end of the
healthy band measured on one voice. On this set no complete read is condemned;
nine rows do not establish a general bound.

The ports each reproduce the fingerprint from their own hand-written canonical
form, each consumes the `stall` and `starved_rescue` fixture sections, and each
port's fixture test rejects a section it does not know.

### The acoustic loop exemption

The repetition rule's all-silence exemption read only `silence_token_ids`, so
a pause on any of the six census-only silent ids read as a period-1 loop. The
specimen: kathleen, en0023, seed 1234, a mid-chunk pause on 6486 (x7) then 6405
(x24). The cut deleted the pause and the two correctly read sentences behind
it, verdict `repetition`, no retry, audibly fluent. Prevalence: 1 in 200
renders. The result is content loss that sounds fluent and carries no flag.
The fix reads the union of the sampler list and both censuses for this one
exemption (new hashed field `repetition_silence`, default `acoustic`); a
census-less checkpoint behaves as before. The ports resolve the union where
the search runs, not in the caller.

### The resumption guard

On a census-less pack the union equals the sampler list, and the en0023 cut
fired again: 52 of 206 tokens kept. A genuine lock-up runs its cycle to the end
of the row: every firing fixture case resumes by zero tokens, and a ceiling
truncates at most one incomplete copy, `period - 1` tokens. The specimen's
pause is followed by 131 tokens of speech. The guard (new hashed field
`repetition_resume`, default `condemn`) condemns a loop whose region ends a
full period or more before the row does. A discard-fraction guard was
calibrated first (fires discard 0.36 to 0.58, the specimen 0.75) and rejected:
the fraction depends on where the pause sits, and a late pause discards less
than any workable cap. The resumption test does not depend on the pause
position, and it also covers ids outside both censuses (the da0028 class). One
search (`_loop_candidate` in Python) returns the cut index and whether the
decoder resumed, so `repetition_cut` keeps its contract.

### The enrollment gate calibration

Measured on all twenty campaign voices, four own-language passages each, seed
1234. Each probe statistic was correlated against the voice's gap rate under
the original sampler: the share of 48 to 200 passages with any gap over 1 s,
from the original-arm measurements.

| predictor | Spearman vs original gap rate |
|---|---:|
| dead tokens in `cond_prompt_tokens` | +0.14 |
| reference speech level (p90 dBFS) | -0.58 |
| rank combination of those two | +0.47 |
| band-B share, probe under the current sampler | -0.84 |
| band-B share, probe under the original sampler | -0.91 |

Statistics computed from the profile's own tokens correlated weakly (dead
count +0.14, profile band composition -0.19). soren's conditioning prompt has
2 dead tokens in 150 and he is the worst voice on the roster; nathalie's has
38, and she is among the cleanest. A profile-only screen would pass soren, so
the gate renders.

The current sampler compresses the signal: the twenty voices spread 0.01 to
0.70 under the original sampler, but only 0.52 to 0.83 under the current one.
In the probe, every voice that the sampler change had to rescue free-ran 150
to 237 consecutive silence tokens, and no clean voice exceeded 39.

Where the voices landed:

- fail: soren 0.010, dave 0.026, dante 0.056, nils 0.059, plus pim and selma,
  whose probes free-ran;
- warn: thorsten 0.16, joe 0.27, kerstin 0.28, henri 0.27;
- pass: every voice with an original gap rate at or under 15%, and darkman at
  0.38, whose gaps came from his reference.

Damaging a good reference, measured by degrading kathleen and re-enrolling:

- Three 120 ms digital-silence splices moved the share from 0.59 to 0.53.
- Lowering the level to soren's -27.8 dBFS moved it to 0.54.
- Both at once gave 0.63.
- A clone from a damaged reference (1.5 s dead lead-in, a gap at every pause,
  low level) lengthened the longest run from 15 to 59 tokens, with one cap hit.

All of these passed, within the probe's own noise. None of the tested
modifications produced a fail. The test does not cover every possible
reference defect.

soren's reference, freshly enrolled, probes bit-identically to his roster
profile (0.010, six majority-silent cap rows). joe re-enrolled from his cleaned
reference moves from 0.27 to 0.65, warn to pass. soren re-enrolled from his
cleaned reference moves from 0.010 to 0.030 and still free-runs a fully mute
row: he still fails.

Limits: darkman passes at a 31% original gap rate, because his gaps entered
through the prompt and cleaning fixes them. pim measured 0.15 in the original
ten-voice census but 0.06 on this probe.

### Truncation at the window (August 2026)

Measured over 9920 chunks in ten languages, with the cap flag of the time
(`hit_token_cap`, which also counts length-ceiling hits): 54 chunks hit a cap,
and 30 were still speaking when it closed. Cap hits and cap-cuts by voice:
soren 38/21, freja 12/6, joe 1/1, kathleen 1/1, nathalie 1/1, thorsten 1/0. Of
the 9866 chunks that ended on their own, the median tail is 0.56 s, and only 5
fall under 0.10 s. A chunk the window cut off has no tail. Every cap-cut row
came out `clean` on the first attempt.

`CHARS_PER_TOKEN` is 0.5, chosen as the low end, with margin, of 0.53 to 0.64
measured on one voice in three languages. soren reads 0.481, below it; freja
reads 0.518 and thorsten 0.587. With each voice's own median pace, only 17 of
the 54 cap hits have enough characters to need more than 255 tokens. The other
37 should have fitted: the model did not emit a stop token in time. A per-voice
mean does not bound the per-chunk variance.

Three approaches were evaluated:

- A shorter budget for every chunk. Preventing the observed cases needs 92
  characters against 127, about 47% more joins across ten languages. That
  figure is a back-fit from the shortest cap-cut chunk, not a rendered arm.
  The one rendered budget arm, soren at 115 characters, produced three cap-cut
  chunks that the original chunking did not have.
- A per-voice budget from a calibrated pace. soren at 115 characters took his
  word-losing passages from 18 to 3, but his holed passages from 17 to 28 of
  120: the seam-over-a-second rate rose from 4.2% to 7.1%. Shorter chunks end
  mid-clause more often, and mid-clause is where a pause-prone voice overruns.
  selma at 154 characters shed 23% of her joins, and her holed passages went
  from 10 to 11 (2.73% to 3.55%), no improvement at ten events.
- A reactive re-split, which ships as `cap_resplit: word`.

The first comparison of re-split arms is invalid, and its seam figures must not
be used. Its harness reimplemented the rule instead of driving the engine, and
seeded the second half with `_derive(seed, _STREAM_CHUNK + index + 1)`, the
next chunk's stream off the passage seed. The engine draws
`_derive(chunk_seed, _STREAM_RESPLIT)`.

The replacement comparison drove `engine.stream` under each setting, over the
51 passages that carry a cap hit, on the loudr-1 checkpoint. The passages were
selected for a cap hit, so the table is not a roster-wide rate.

| | `off` | `word` |
|---|---|---|
| windows | 202 | 255 |
| joins | 151 | 204 |
| windows at the cap | 52 | 2 |
| still speaking at the cap | 29 | 1 |
| seams over 1 s | 4/151 (2.6%) | 6/204 (2.9%) |
| worst seam | 1.28 s | 1.42 s |
| audio | 1472.6 s | 1503.2 s |

Truncations fell from 22 to 0 for soren and from 6 to 0 for freja. Total audio
grew by 30.6 s; that includes changes in pauses as well as recovered speech.
The one row still cut is `nathalie/nl0098`, the single word `"omringden."`. It
has no space for `split_in_half` to use, and a chunk that short hits the
length-proportional ceiling, not the window, so re-splitting cannot help it.
The other row still at the cap, `soren/da0075`, ends on a 0.42 s tail: it
finished its clause. The cost is two more seams over a second, 2.6% to 2.9%.

The census above also measured the trigger's precision: of the 54 cap hits, 30
lost words, so 24 (about one in four hundred chunks) would be split without
need and gain a seam. Two things differ in the current trigger, and neither is
re-measured on the roster:

- Three of the 54 were the length ceiling, not the window. The current trigger
  reads `hit_window_cap`, so those are not in its population.
- The flag is read before any trim, so filled windows that postprocess cut
  back are in the population. That adds chunks, and the count is not known.

The seam-over-a-second rate moved from 2.6% to 2.9% across necessary and
unnecessary splits together; the two were not separated.

### Fingerprints of the measured states

Each change above moved the algorithm fingerprint. These are the pre-release
states the measurements were made under; later changes moved them again before
release, and 0.1.1 ships `7cd75498ad4e7531`.

| change | fingerprint (default manifest) | census-less default |
|---|---|---|
| before the campaign | `79f71f5821477353` | |
| penalty on silence, `stall`, censuses | `9ecb061d7228eaad` | |
| starved desperation cut | `494ae241cbac5869` | `d439a578b2075e9d` |
| `repetition_silence` | `e8de57e99b735340` | `59f2d8ac0564485f` |
| `repetition_resume` | `1628fb0fd201b784` | `7ee33ab14b7f8498` |

The re-split was measured on a separate line: `cap_resplit` moved
`f95cd1f9abae916d` to `7b59fba15ad17992` (the default `d705d7533801aa80` to
`86c9f40273db3fb0`). A text-funnel change unrelated to silence then moved
those to `baf8462effe3ba47` and `797c55169dbc3204`.

## Open questions

- Which exemption doubled the pause-time share. The 0.400 figure above was
  recorded as the corrected list applied to the penalty exemption. The
  manifest's `silence_render_ids_source` and the postprocess notes record the
  doubling for the census ids in the `min_p` exemption. The source runs are not
  in the repository, so this is not settled.
- The nils snapshots. One snapshot of the nils original arm records 8 mute
  rows (76 s) and a tail p95 of 3.32 s; the matched table records 10 (92 s)
  and 3.62 s. The figure 6.66 s is the seam p95 in the table and the worst seam
  in the penalty comparison. The snapshots are not reconciled.
- henri's compression. The prosody check gives about 7% of his natural
  band (23.94 to 22.19 per minute); the sampler-change summary gives -8.5%.
- Heads and residual seams. A head under 25 true-silence tokens is neither
  trimmed nor flagged. nils's residual is seams slightly over a second. No rule
  targets either, and a roster-wide count would decide whether one should.
- The sampler list. `silence_token_ids` still has 13 audible ids in the
  `min_p` exemption and misses six silent ids. Correcting it changes the
  fingerprint, and a measured correction doubled the pause-time share.
- The re-split trigger. The current trigger's population (pre-trim
  `hit_window_cap`, no length-ceiling hits) is not re-measured, so the share of
  re-splits that recover words is not known.
- Narrowing the trigger. A tail test separates the chunks that were still
  speaking from those that were not, but it needs audio, and the split decision
  happens when generation ends, before the renderer. `stream.pipelined` renders
  on a worker while the producer generates the next window, so waiting for a
  render before the decision would stall the producer and remove that overlap.
  With the census counts, a narrowed rule costs one extra render on the 0.54%
  of chunks that reach the cap, and saves two generations and a render on the
  0.24% that did not need a split. Those counts come from the census
  population, and the cost under the current pipeline is not measured. A
  change would also move the fingerprint in all five implementations.
