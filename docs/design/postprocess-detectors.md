# Postprocess detectors

Maintainer notes for `loudkit/postprocess.py`: the reasoning and the
measurements behind each symbol. Each heading names the symbol whose docstring
points here. [Postprocess](postprocess.md) has the rule definitions, the
calibration data and the precedence argument.


## Module


### `module`

The module is a detector. It reads the speech tokens a chunk produced and
returns a keep length, a reason and a `suspect` flag. It never changes audio.
The evidence is read in the token domain because the artifact is generated: a
free-running decoder past the end of a sentence leaves a run of silence with a
burst behind it. That is a shape in tokens and no reliable shape in the
spectrum. Token counts and set membership let the five implementations agree
exactly. The rules that read the EOS peak also depend on the sampler's float
value; see "The EOS peak on the device" in [postprocess](postprocess.md).

Seven rules in a fixed precedence: `dropout`, `repetition`, `stall`,
`silence_tail`, `terminal_echo`, `desperation`, `ended_tail`.


## Functions


### `is_trailing_filler`

`silence_tail` cuts back to the EOS peak, the step where the model came
closest to stopping. The peak alone is not enough evidence. Cutting at the peak
alone truncated whole sentences: the same showcase script that runs 10.3 s with
one narrator came back at 3.2 s with another. A voice reading a language its
tag does not match may never sample the stop token, and its EOS peak can land a
third of the way into the text.

The tail after the peak must therefore pass this test. It accepts either:

- a silence share of at least `trailing_filler_threshold`; or
- a silence run of at least `trailing_silence_run_tokens`, with at most
  `filler_max_speech_after_run` tokens before the first such run, between
  consecutive runs and after the last one. What follows the run is then a
  stray word, not a continuing clause.

The distance limits prevent a false cut. Without them, a rhetorical pause
mid-tail (25 silent tokens, then 80 tokens of speech) matched the run test, and
the cut removed the rest of the sentence. The `trailing_filler` fixture case "a
pause followed by more sentence is not filler" checks it.


### `repetition_cut`

The tail rules cannot see a loop inside the row. The mechanism is the one
behind the trailing extra word: the model's own output becomes its context.
Published work reports this failure. VALL-E's authors describe greedy search
"continually generating silence codec codes", and the Very Attentive Tacotron
stress test produced 52 repetitions of a phrase meant to occur nine times.

The anchor can sit mid-sequence, so the conditions are strict: a short cycle,
repeated at least `repetition_min_cycles` times over at least
`repetition_min_span` tokens, with exact matches only. Approximate repetition
does not count. The rule is tuned against false positives, because its cut
removes speech. Under `config.repetition_resume == "condemn"`, an applied cut
removes only a tail: `inspect` condemns a loop the decoder resumed from
instead of cutting it.

A cycle made only of silence-family ids is not a loop: the tail rules and
`stall` judge pauses. Under `config.repetition_silence == "acoustic"`, the
family is the passed ids plus both render censuses, read off the config the
same way `is_stalled` reads them. Keyed to the sampler list alone, the
exemption misses six of the eight silent-rendering ids. The en0023 specimen
(see `repetition_silence` below) is a pause on two of them that fired as a
period-1 loop. A cycle that mixes silence with speech still counts: the
word-then-pause stutter is one shape of this failure.

Returns the index one full cycle after the loop's start, so one instance of
the repeated fragment survives. Whether the index is applied is the resolver's
decision. Under `repetition_resume == "condemn"`, a loop whose region ends a
full period or more before the row does is condemned into the retry ladder,
not cut. `inspect` reads that from `_loop_candidate`.

Provenance: published inline repetition guards informed the cycle constants
(VALL-E 2's repetition-aware sampling uses a 10-token window; MSpoofTTS scans
at segment lengths 10/25/50). `repetition_min_span` was measured on this
checkpoint's renders (see below). That 27-render probe contains no observed
non-silence true positive, so the rule is calibrated for no false positives,
not for sensitivity.


### `is_stalled`

`is_stalled` detects a decoder trapped in silence: a hole inside the row, or a
row with no speech. The decoder enters a silence run at a pause point, as its
own argmax. `min_p` usually removes every non-silence candidate (on 87% of
stall steps in an instrumented trace), while the exemption re-admits the listed
silence ids. With silence also exempt from the repetition
penalty, the run had no exit: zero escapes in 1,031 instrumented trap steps.
Measured under that sampler (August 2026): 33.0% of paragraphs carried a hole
over 1 s, and 74 chunks in 1705 passages rendered no speech at all, 11.5
minutes of mute audio, all of them `clean` under the other rules. The sampler
now applies the repetition penalty to silence, which reduced these failures.
This rule detects what still gets through, and covers any checkpoint or
configuration where the trap returns.

Three triggers; any one condemns the row. All three are integer tests.

- No speech at all: every generated token is in the silence-or-quiet family.
  Needs the census. Measured: mute rows are 255 of 255 silence tokens, they
  recur in 3 of 18 re-renders, and a retry rescues 9 of 10.
- A non-tail dead-air run of at least `stall_run_tokens` true-silence tokens.
  Quiet-family ids extend a run without counting toward it (see
  `stall_run_tokens`). Tail runs are excluded, because the tail rules judge a
  trailing pause against its position.
- A ceiling overrun that is mostly silence: `hit_ceiling` and a strict family
  majority of the row. Needs the census. On a cap-hit row the dead air is a
  stall that the cap interrupted (the reference specimen is 90.2% silence, a
  210-token run to the cap). This trigger also reaches rows that `desperation`
  cannot: the ceiling clips rows to 4.0 × text tokens + 40, so past 80 text
  tokens a cap-hit row cannot reach the 4.5 desperation ratio.

Without a census (`silence_render_ids` empty), only the run trigger fires,
keyed to the configured `silence_token_ids`. 13 of that list's 31 ids render
audible speech, so whole-row membership in it does not identify a mute row,
and a whole-row trigger keyed to it could condemn real speech.

Returns `True` for a condemned row. There is nothing to cut: a tail cut cannot
repair a hole inside the row. The row goes to the retry ladder, as a
`dropout` row does. `is_stalled` has no terminal-chunk guard.


### `desperation_cut`

At or above `desperation_speech_per_text_token`, with at least
`desperation_min_text_tokens` text tokens, the rule chooses where to cut:

- At the first silence run of at least `trailing_silence_run_tokens` that
  starts at or after the earliest cut. A run that straddles the floor belongs
  to the sentence, so the run's start is tested, not its end.
- Otherwise, if `peak_allowed`, at the EOS peak, when it lies before the end of
  the row and within
  `[earliest cut, int(desperation_band_ratio × n) + desperation_band_floor]`.
  The band keeps a peak a third of the way into a mislabeled-language row
  (the showcase case, 92 generated / 26 text = 3.5x, below the ratio guard)
  from anchoring a cut.

`inspect` passes `peak_allowed = is_terminal`. A continuation chunk has no
sentence end, so its EOS-peak fallback is off. The silence-run path still
applies to it.


### `inspect`

`inspect` is the one resolver. It runs the rules in a fixed order and returns
one `Inspection`. This is the reference order, and the four ports implement
it:

1. `dropout`: the row is too short for the text. Reported whole, `suspect`.
2. `repetition`: an exact repeated cycle. First of the cuts, because its
   anchor is exact and every other anchor is inferred. A loop the decoder
   resumed from is condemned whole, because the cut would delete what
   followed it.
3. `stall`: a hole inside the row, or no speech. Condemned whole, before any
   tail rule: a tail cut cannot remove a hole in the middle, and a tail rule
   that fired here would trim the tail and hide the hole under its own reason.
4. `silence_tail`: the peak-anchored filler cut, on a terminal row that did not
   end.
5. `terminal_echo`: the peak-anchored cut without a seam, on a terminal row.
6. `desperation`: the length-anchored cut, the least precise. It also applies
   to rows that ended on the stop token: a row can run long and still end
   with one. On a cap-hit row, a starved cut is condemned.
7. `ended_tail_trim`: only on a row that ended, and only when no rule above
   chose a cut.

Last, a row with no cut, at least `desperation_min_text_tokens` text tokens
and a ratio at or above `desperation_speech_per_text_token` is flagged
`suspect`.

| parameter | meaning |
|---|---|
| `tokens` | the chunk's speech tokens, specials already stripped |
| `text_token_count` | how many text tokens produced them; the denominator of every ratio rule |
| `min_tokens` | the EOS floor this row was generated under; `silence_tail`, `terminal_echo` and `desperation` cut no earlier than `max(min_tokens, 10)`. `ended_tail` does not read it |
| `eos_peak_at` | the step with the largest EOS observation, or a negative number if none was recorded |
| `eos_peak_prob` | the observed value there (see [sampler and noise](sampler-and-noise.md#_observe_eos)). It never feeds back into sampling, but it gates `silence_tail` and `terminal_echo`, so the conformance fixture pins it. |
| `ended` | whether generation stopped at the stop token, not at a cap |
| `is_terminal` | whether this chunk ends the passage; enables the terminal-only checks listed below |
| `hit_ceiling` | whether generation stopped at its token cap |
| `silence` | the sampler's silence ids. The tail rules read exactly this list; `repetition_cut` adds the render censuses itself, and `is_stalled` reads the censuses off the config. |


### `pacing_outliers`

The long-form drift signal. Each ratio is speech tokens over text tokens for
one chunk. `pacing_outliers` returns the indices of chunks whose ratio is above
`median × pacing_tolerance` or below `median / pacing_tolerance`, and needs at
least three chunks. Report-only: the caller learns which chunks to listen to,
and nothing is cut, because an unusual pace does not say what is wrong. The
median is used, not the mean, so one broken chunk cannot move the baseline
toward itself.


### `_loop_candidate`

One search answers both questions. It returns the earliest qualifying loop as
`(cut index, resumed)`. The cut index is `repetition_cut`'s result. `resumed`
is true when the loop's repeating region ends `period` or more tokens before
the row does, computed as `n - at >= period`. A locked decoder emits its cycle
to the end of the row, and a ceiling can truncate at most one incomplete copy
(`period - 1` tokens). A full period of other tokens after the region
therefore shows that the decoder resumed. No extra scan is needed: a matching
full copy would have counted as another cycle, so `n - at >= period` already
implies a deviation.


### `ended_tail_trim`

Runs on a row that ended on the stop token. It reads the tail backward as
`[sentence][r1 silence][burst][r2 silence]` and trims three shapes, each down
to at most `ended_tail_keep` silence tokens:

- a bare silence run of at least `trailing_silence_run_tokens` (half a
  second);
- a burst of at most `ended_tail_blip_max` tokens after at least
  `ended_tail_silence_run` silence tokens: a 40 to 80 ms click after a pause
  (the device specimen ended `.......#`);
- on a terminal chunk only (`is_terminal=True`), a burst of at most
  `ended_tail_word_max` tokens behind a full `trailing_silence_run_tokens`
  seam.


### `terminal_echo_cut`

There is no silence seam here, so `is_trailing_filler` has no run to anchor
on. The EOS peak must lie after the earliest cut and before the end of the row,
and it must be strong, late and followed by a short tail. The late-position
rule keeps real comma and clause pauses from being cut. Terminal chunks only.

The second acceptance path is narrower. It exists for one measured regression
("...but a brigand. Pass. Four.": `gen=124/124, bestEOS=109@0.004`). The model
never sampled a stop token, but its best, very weak stop was 15 tokens before
the hard ceiling. The weak path accepts such a peak only with all of these: a
terminal chunk, a ceiling overrun (`hit_ceiling`), a tail of at most
`echo_weak_max_tail` tokens, and the peak in the last 15% of the row.


### `_is_dropout`

Two conditions, both required: `token_count < dropout_min_tokens`, and
`0 < text_token_count` with `token_count < text_token_count`. The absolute gate
catches a row that stopped almost at once, whatever the text. The proportional
test keeps a short line exempt: a row is suspect only when it has fewer speech
tokens than its text has text tokens. The function does not read the EOS
floor; `inspect` passes the floor only to the tail rules.


### `cut`

`Inspection.cut` is true when `reason != "clean"`: a rule fired. It does not
mean that tokens were removed. `dropout`, `stall` and a resumed `repetition`
condemn a row without cutting it: `keep` holds the whole input, and `cut` is
still true. To tell whether tokens came off, compare `keep` with the input
length. A caller that slices by `keep` gets the right answer either way.


## Configuration fields


### `ceiling_speech_per_text_token`

A guard against a three-word sentence decoding for ten seconds. It is the
proportional term in
`min(window, int(text_tokens × ratio) + ceiling_slack_tokens)`. The device
trace behind 4.0:

```
t3.overrun  gen=92 ceiling=92 bestEOS=74@0.003 floor=31
```

A chunk of about 26 text tokens stopped only because it hit the ceiling. Its
stop probability was 0.003, it was mid-sentence, and it was already at 3.5
speech tokens per text token. The ratio 4.0 sits above that. Short texts run
higher, and the additive slack carries them (see [postprocess](postprocess.md)).

It is separate from the chunker's estimate. The chunker budgets characters
(`CHARS_PER_TOKEN`, 0.5 characters per speech token) and uses its estimate as
an upper bound, where guessing high only makes chunks shorter. Here guessing
low cuts a sentence off.


### `trailing_silence_run_tokens`

An extra word at the very end sits behind a seam: silence, then a burst of
speech tokens. The burst lowers the silence share below the share test's
threshold, so the share test alone misses this tail. The unbroken-run test
catches it.


### `desperation_band_ratio`

The top of the EOS-peak band for the desperation fallback:
`int(ratio × text_token_count) + desperation_band_floor`. The peak must also be
at or after the earliest cut. One voice measured 1.75 to 2.35 speech tokens
per text token on real speech, so 2.6 reaches past those endings and stays well
under the rule's own 4.5 ratio. A peak outside the band is not used as an
anchor; the rule can still cut at a qualifying silence run.


### `desperation_speech_per_text_token`

The trace behind 4.5: "It was as he expected.", 14 text tokens, came back as 96
speech tokens, the sentence followed by dense babble. The stop peak was at the
right place (45) but with probability 0.000, so every probability-gated rule
declined it. One voice measured 1.75 to 2.35 on real speech. A later 54-render
sweep across eighteen voice profiles reached 3.7 at fourteen text tokens, and
every render above 4.5 had four text tokens, which
`desperation_min_text_tokens` excludes. With the minimum-length gate, reaching
4.5 lets the rule choose where to cut. It does not show that every token after
the anchor is wrong.


### `desperation_min_keep_per_text_token`

The specimen (2026-08-29): soren, da0028 chunk 4, seed 1234. The row ran 132
tokens to the ceiling, and the seam cut kept 36: 1.44 s of audio in which 33 of
the 36 kept tokens render near-silent through ids outside both manifest
censuses, so no set-membership rule can see them. Without this check the trim
shipped a mute chunk and gave the caller no signal to retry. The same window
renders clean at seeds 7 and 99.

Calibrated on every cap-hit desperation cut in the interior-stall and
acceptance batteries (nine rows, four voices, four languages), each kept chunk
graded against its rendered audio. Absolute keep length does not separate
them: a keep of 36 tokens was mute on one row (23 text tokens) and a complete
short sentence on another (19). Keep per text token does: every keep at or
below 1.57 was mute or missing much of its text, and every keep at or above
1.85 carried real speech. 1.7 sits inside that gap, and under 1.75, the low end
of the healthy band measured on one voice. On this set no complete read is
condemned; nine rows do not establish a general bound.

The costs are asymmetric. A false condemnation costs a retry of one window (a
2x decode for it), and cap-hit desperation fires on about 0.5% of windows. A
false pass ships about a second and a half of silence as fixed. The check
applies to cap-hit rows only: a row that ended on the stop token is not
checked. Zero disables it.


### `filler_max_speech_after_run`

A separate field from `ended_tail_word_max`, although both hold 10 and mean the
same duration. They govern different rows. This one sets the distance limits
of the filler test, which `silence_tail` runs on a terminal row that did not
end. `ended_tail_word_max` sets the stranded-word path of `ended_tail` on a
terminal row that did end. They were settled separately. With one field,
loosening the trim on ended rows would also loosen the filler test.


### `retry_max_attempts`

Published results show the shape: catastrophic-failure rates dropped from 5.8%
to zero at a single retry, on the failing rows. That figure is from the
literature, not a measurement of this engine. Retrying every row costs N×
compute. Retrying only the rows the detectors condemn cost about 1.1× at the
measured fire rates, which is why the detectors report instead of guessing a
cut.

The engine retries a row when the verdict is `dropout` or `suspect` is set:
`dropout`, `stall`, a resumed `repetition`, a starved `desperation` cut, and an
uncut row past the length ratio. A trimmed verdict without `suspect` is not
retried. Each attempt draws from a seed derived from the caller's seed and the
attempt number, so the whole ladder is a pure function of the caller's seed.
For that reason the ladder is config, not a loop a caller writes around the
engine.

Zero disables it, and `RETRY_LADDER_HEADROOM` (8) bounds it above: from 8
attempts, the derived seeds would run into the streams the chunk seeds use.


### `pacing_tolerance`

Pace is speech tokens per text token, the quantity the length rules use. A long
passage is rendered chunk by chunk, and a chunk whose pace lands far from its
neighbours' is rushing or dragging: the long-form drift that the literature
reports as prosody breakdown past the training window. Report-only: a healthy
render also has a pace, and cutting on it would be a guess.

The default 1.6 flags a ratio above 1.6 × the passage median or below the
median / 1.6. Healthy per-chunk ratios in the nine-language probe run 1.69 to
2.79, a 1.65x spread across languages. That range is context, not a
calibration within a passage; the false-positive rate within a passage is not
measured.


### `dropout_min_tokens`

The failure is early truncation. Content is missing, no trim recovers it, and a
truncated render can sound fluent. A published criterion for catastrophic
neural-codec TTS failure uses this shape: a speech-token count under 25, or an
ASR transcript of at most one word. The threshold is taken from it.

It is a gate on the row, and it fires only together with the proportional test
(fewer speech tokens than text tokens), so a short line is exempt. In the
nine-language probe the shortest healthy read was 35 tokens.


### `stall_run_tokens`

The run is measured two-class. Only true-silence ids (`silence_render_ids`)
count toward this threshold. The run continues across quiet-family ids
(`quiet_render_ids`), breath and decay tokens that render inaudible in context.
With a single set, one breath token in the middle of real dead air split a
47-token run into two short ones, and the rule missed it.

Calibrated across ten languages at 120 passages per arm: healthy interior runs
top out at 13 to 19 tokens and healthy leading runs at 11, and every measured
stall ran longer than 25. Both 20 and 25 clear the healthy maxima; 25 is the
shipped value.


### `silence_render_ids`

A property of the checkpoint, measured by rendering (per-id median energy below
-80 dBFS across two independent censuses). The manifest supplies it at top
level, beside `silence_token_ids`. Empty means that no census is configured.
The stall rule then runs only its run trigger, keyed to the configured
`silence_token_ids`.

That list cannot stand in for the census. Only 2 of its 31 ids (4137 and 4218)
are in the census, 13 render audible speech, and 16 were not emitted in the
census sample. The whole-row and majority triggers are therefore off without a
census.

It is not a sampling exemption list. The sampler keeps `silence_token_ids` for
its `min_p` exemption, and the repetition penalty applies to every token. Using
the census ids for the `min_p` exemption was measured to double the pause-time
share; see the open questions in
[silence classes](silence-classes.md#open-questions). This list exists so that the detectors find dead air where it is.


### `quiet_render_ids`

Measured by per-instance RMS attribution (at least 90% of instances quiet, at
least 5 sightings), minus the true-silence census. Dead-air runs continue
across these ids, but they never count toward the run gate: a breath inside
dead air extends the dead air, and a breath between words does not start a
run. The manifest supplies it like `silence_render_ids`. Empty means that no
quiet census is configured.


### `repetition_min_span`

Cycle count alone does not separate a stuck decoder from healthy speech. This
model winds down nearly every row with a short repeated tail token, so short
repetition is present in almost all real speech. A stuck decoder differs
because its repetition does not stop.

Measured across 27 renders, nine languages, one voice: the longest repeating
span in a healthy row is 10 tokens (0.4 s), median 7. The one runaway row in
the set, a Spanish three-word phrase that never emitted a stop token, repeats
for 44 tokens (1.76 s). It repeats a silence token, so it is not a true
positive for this rule, and the probe has none. 24 sits between them, with
2.4x margin over the healthy maximum.


### `repetition_resume`

`condemn` (the default): a qualifying loop followed by a full period or more of
other tokens is reported whole. `keep` is the full row, the verdict is
`repetition`, `suspect` is set, and the row goes into the retry ladder like
`stall`. The cut is refused because it would delete what followed the loop.

A genuine lock-up is a tail pathology: the model's own output is its context,
the state is absorbing, and the repeating region runs to the end of the row.
Every firing case in the conformance fixture resumes by zero tokens, and a
ceiling can truncate at most one incomplete copy (`period - 1` tokens, at most
11). A loop followed by a full period of other content is a different event.
On a manifest without render censuses, it is typically a pause on a
silent-rendering id that the sampler list does not name.

The specimen: kathleen, en0023, seed 1234, on a census-less pack. A pause sits
on 6486 (x8) then 6405 (x24), the period-1 run passes every loop condition,
and the cut keeps 52 of 206 tokens. It deletes the pause and the 131 tokens of
correctly read speech behind it: two sentences, verdict `repetition`, no
retry, audibly fluent. The render recorded under `repetition_silence` has 6486
x7. `repetition_silence` covers that case on a manifest that carries the
censuses. This field covers it on every manifest, including pauses on ids that
no census lists (da0028's mute keep rendered near-silent through ids outside
both censuses).

A discard-fraction guard was calibrated first and rejected. Genuine fires
discard 0.36 to 0.58 of the row and the specimen 0.748, so a cap of 2/3
separates the observed sets. A pause late in a row discards less than any such
cap, and the cut still ships: a 30-token pause at 70% of a 206-token row, with
one clause behind it, discards 0.32. A fraction cap limits the loss; the
resumption test prevents it. Fires resume by 0 tokens, the specimen by 131, and
a truncated genuine loop resumes by at most `period - 1`. The test is therefore
`resume >= period`, an integer comparison with no constant to tune. A cut that
passes it removes only a tail, like every other rule in the layer.

`cut` turns the guard off, for a checkpoint measured without it.
`repetition_cut` reports the loop either way; this field decides what the
resolver does with a loop the decoder resumed from.


### `repetition_silence`

`acoustic`: the union of the configured sampler silence ids and both render
censuses (`silence_render_ids`, `quiet_render_ids`). A cycle made only of ids
in that union is not a loop. Ids outside the union are not covered.

The specimen that settled it: kathleen, en0023, seed 1234. A
mid-chunk pause sits on 6486 (x7) then 6405 (x24). Both ids render true
silence and both are in `silence_render_ids`; neither is in the sampler's
`silence_token_ids`. With the sampler list alone, the exemption misses them.
The 24-token period-1 run fires as a loop, and the cut keeps one cycle and
deletes the pause plus the six seconds of correctly read speech behind it: two
whole sentences, verdict `repetition`, not suspect, no retry, audibly fluent.
Six of the checkpoint's eight silent-rendering ids are outside the sampler
list. Measured prevalence: 1 in 200 renders.

`sampling`: the configured sampler list alone, so that a checkpoint measured
under it can declare what it measured. A checkpoint without censuses gets this
behaviour under either value, because the union equals the sampler list.

This family feeds the loop exemption only. The tail rules (`silence_tail`,
`ended_tail`, the filler and desperation seams) read the sampler list they were
calibrated against; see "Two silence lists" in [postprocess](postprocess.md).


### `suspect`

`suspect` flags a row for retry and review. It is set by:

- `dropout` (content missing);
- `stall` (dead air where speech should be);
- `repetition` on a loop the decoder resumed from, under `condemn` (the row is
  handed back whole);
- a starved `desperation` cut: a cap-hit cut that keeps fewer than
  `desperation_min_keep_per_text_token` speech tokens per text token. Here
  `keep` holds the cut, as the fallback if every retry is also condemned;
- a row with no cut, at least `desperation_min_text_tokens` text tokens and a
  ratio at or above `desperation_speech_per_text_token`.

It is a report and the engine's signal to retry. It does not raise.


### `_validate_ranges` checks finiteness first

Every later check is a comparison, and NaN fails all of them: `nan < 0` and
`nan > 0` are both False. Without the finiteness check, a NaN passes the
validator and fails later, as `int(nan)` inside the ceiling or as a pacing
tolerance that no ratio exceeds. The manifest is data from outside the
process, so the validator rejects non-finite values first.


### `inspect`: dropout is reported, never cut

When `_is_dropout` holds, `inspect` returns
`Inspection(keep=len(tokens), reason="dropout", suspect=True)`. Removing tokens
cannot restore missing content, so nothing is cut. See `_is_dropout` and
`dropout_min_tokens` for the conditions.


### `inspect`: terminal-only checks

A continuation chunk ends mid-passage, so some checks are off for it:

- `silence_tail` requires `is_terminal`, and a row that did not end;
- `terminal_echo_cut` returns early when `is_terminal` is false;
- `desperation_cut` receives `peak_allowed = is_terminal`: only its EOS-peak
  fallback is off, and its silence-run path still applies;
- `ended_tail_trim` receives `is_terminal`: only its stranded-word path is off,
  and its dead-air and click paths still apply.

`is_stalled` has no terminal-chunk guard.


### `inspect`: a stall is condemned whole

For a `stall` verdict, `inspect` keeps the full row and sets `suspect`. No trim
is kept as a fallback, because a tail trim does not remove a hole in the middle
of the row.


### `inspect`: a starved desperation cut is condemned, and the cut stands

On a cap-hit row, no stop token corroborates the trim. A cut that keeps fewer
than `desperation_min_keep_per_text_token` speech tokens per text token is
condemned: `suspect` is set, and `keep` holds the cut. The kept audio can be
near-silence through ids that no census lists (da0028: 33 of the 36 kept
tokens), so the keep's length is the evidence. The engine retries. If every
attempt is condemned and this one is chosen, trim mode ships the trim, flagged
`suspect`, not the untrimmed row.
