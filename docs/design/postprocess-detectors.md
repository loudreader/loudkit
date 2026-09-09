# Postprocess detectors

Maintainer notes moved out of the runtime's docstrings: the reasoning and
the measurements behind each symbol. Each heading names the module and
the symbol the note belongs to. Not user documentation.


## `loudkit/postprocess.py`


### `is_trailing_filler`

The overrun rescue cuts back to where the model came closest to stopping,
and that peak is a hint, not a verdict. Trusting it alone truncated whole
sentences: the same showcase script that runs 10.3 s in one narrator came
back at 3.2 s in another, because a voice reading a language its tag does
not match may never commit to stopping, so its best moment of hesitation
lands a third of the way in.

So the peak has to be corroborated by *what it proposes to discard*. Two
forms of corroboration, either one suffices:

* the tail is mostly silence by share, or
* the tail contains a long unbroken silence run **and** what follows that
  run is a stray word rather than a continuing clause.

The second half of that second condition is not decoration. Without it a
rhetorical pause mid-tail (25 silent tokens, then 80 tokens of speech)
matched the run rule and the rescue cut the rest of the sentence off -
caught by ``TrailingFillerTests.aPauseFollowedByMoreSentenceIsNotFiller``.


### `repetition_cut`

The one failure this layer did not cover, and the one the literature puts
first or second in every ranking of what goes wrong with autoregressive
speech models. The mechanism is the same one behind the trailing hallucinated
word, the model's own output is its context, but it strikes *inside* the
row rather than after it, so no tail rule can see it. VALL-E's authors
describe greedy search "continually generating silence codec codes"; the
Very Attentive Tacotron stress test produced 52 repetitions of a phrase that
was supposed to occur nine times.

Unlike the other rules here, this one anchors **mid-sequence**, so it is
deliberately hard to trigger: a short cycle, repeated many times, exactly.
Approximate repetition is left alone. A decoder that has genuinely locked up
emits the same tokens, not similar ones, and a fuzzy match on a signal this
destructive would be a way to truncate real speech. And under
``config.repetition_resume == "condemn"`` an *applied* cut only ever
removes a tail: a loop the decoder resumed from is condemned by the
resolver instead, so the mid-sequence anchor never deletes what followed
it.

A cycle that is entirely silence is never a loop: silence repeating is what
silence *is*, and a pause is already judged by the tail rules against the
place it sits in. Under ``config.repetition_silence == "acoustic"`` the
exemption reads *acoustic* silence, the passed ids unioned with both
render censuses, the same way ``is_stalled`` reads its censuses off the
config. Keyed to the sampler list alone it was blind to six of the eight
truly-silent ids, and a long pause parked on one of them fired as a
period-1 loop whose cut deleted the pause and every correctly-read token
behind it (kathleen, en0023, seed 1234: two sentences, verdict
``repetition``, audibly fluent). A cycle mixing silence with speech still
counts, a word-then-pause stutter is one of the shapes this failure
takes.

Returns the index one full cycle past the loop's start: the first instance is
plausibly the word the sentence actually wanted, and everything after it is
the model reading its own tail.

Whether that index is *applied* is the resolver's decision, not this
rule's: under ``config.repetition_resume == "condemn"`` a loop the
decoder resumed from, one whose repeating region ends a full period or
more before the row does, is condemned into the retry ladder rather
than cut, because a decoder that resumed was never locked, and the thing
it resumed from is a pause the silence family could not name. This
function still reports the loop; :func:`inspect` reads the resumption
off :func:`_loop_candidate` and judges it.

**Provenance is different from every other rule on this page** and is stated
rather than buried: the two constants come from the published parameters of
inline repetition guards (VALL-E 2's repetition-aware sampling uses a window
of 10 tokens; MSpoofTTS scans at segment lengths 10/25/50), not from a device
trace of this model. They are calibrated to be *safe* rather than sensitive,
and the measured firing rate on real renders is reported in the docs.


### `is_stalled`

The failure the tail rules structurally cannot see. The decoder enters a
silence run at a pause point, its own argmax, and, with ``min_p``
stripping every non-silence candidate while the exemption re-admits the
listed silence ids, the run's exit probability is effectively zero
(measured: zero escapes in 1,031 instrumented trap steps). The sampler now
applies the repetition penalty to silence, which closes the trap at the
source; this rule is the detector for what still gets through, and for any
checkpoint or configuration where the trap re-opens. At scale before the
fix: 33.0% of paragraphs carried a >1 s hole and 74 chunks in 1705
passages rendered no speech at all, 11.5 minutes of text silently
swallowed, every one of them ``clean``, because all six other rules
anchor on the tail.

Three triggers, all integer-exact, any one condemns:

* **no speech at all**, every generated token is in the silence-or-quiet
  family. Measured: mute rows are 255/255 silence tokens, a seed lottery
  (they recur at 3/18 re-renders), and a retry rescues 9/10.
* **a non-tail dead-air run** of at least ``stall_run_tokens`` true-silence
  tokens. Two-class: quiet-family ids extend a run without counting toward
  it (see ``stall_run_tokens`` for why single-set counting is broken).
  Tail runs are excluded, the tail rules own the tail, and a trailing
  pause is judged against the place it sits in.
* **a ceiling overrun that is mostly silence**, ``hit_ceiling`` and the
  family holds a strict majority of the row. A tail run on a cap-hit row
  is not a natural tail: the ceiling truncated the read, so the dead air
  is a stall the cap happened to interrupt (the reference specimen is
  90.2% silence, a 210-token run to the cap). This also closes a
  structural hole: the ceiling clips rows to 4.0x text tokens + 40, so
  past 80 text tokens a cap-hit stall can never reach the 4.5x
  desperation threshold, the rule that means "certainly broken" was
  unreachable by the most broken rows this layer sees.

Without a census (``silence_render_ids`` empty, a checkpoint packed
before it) only the run trigger fires, keyed to the configured
``silence_token_ids``. That is the measured-safe subset: 13 of the
configured list's 31 ids render audible speech, so whole-row membership
in that list does not prove a mute row, and a whole-row trigger keyed to
it could condemn real speech. Degraded-but-safe beats a fallback that
lies.

Returns ``True`` for a condemned row. There is nothing to cut: the failure
is a hole, not a tail, and the fix is the retry ladder, the same route
``dropout`` takes, for the same reason.


### `desperation_cut`

Past ``desperation_speech_per_text_token`` the row is certainly broken, so
the question is where to cut, not whether:

* at the first long silence run that starts past the floor, a structural
  boundary, and on a certainly-broken row what follows it needs no further
  corroboration. A run straddling the floor belongs to the sentence, not to
  the tail, which is why the run's *start* is tested rather than its end;
* else at the stop peak, if it sits in a band a real read could have ended
  in. The band is what protects the mislabeled-language case (92 generated /
  26 text = 3.5x, below the ratio guard), a row of that kind must never be
  cut at a peak landing a third of the way in, and here such a peak fails
  the floor.

``peak_allowed`` is false for a continuation chunk: it has no sentence end,
so its stop peak means nothing.


### `inspect`

The reading app grew five entry points, one per field bug, and left the
ordering to each call site. Here they are one resolver with the precedence
written down, because an order that lives in a caller is an order the next
caller gets wrong. Written down here rather than only in the four ports,
which is where it was: this is the reference, and a contract stated only by
its implementations is not stated.

The order:

1. ``dropout``, the row is too short for the text. Reported whole, never
   cut: nothing below can help a row that is missing content.
2. ``repetition``, an exact repeated cycle. First of the cuts, because it
   is the only rule that knows *exactly* where the failure began; every
   other anchor here is inferred. A cycle the decoder came back from is
   condemned whole rather than cut, since the cut would delete what it came
   back to say.
3. ``stall``, a mid-row hole. Condemned whole, before any tail rescue: a
   tail cut cannot remove a hole in the middle, and a rescue firing here
   would trim the tail and ship the hole under its own reason.
4. ``silence_tail``, the peak-anchored filler trim.
5. ``terminal_echo``, then ``desperation``, the length-anchored one is the
   bluntest, and it applies to *ended* rows too, because a model that
   babbles past its sentence and only then samples a stop token has
   forfeited the trust that stopping implies.
6. ``ended_tail_trim``, only when nothing above fired.

Args:
    tokens: the chunk's speech tokens, specials already stripped.
    text_token_count: how many *text* tokens produced them. The denominator
        of every ratio rule.
    min_tokens: the EOS floor this row was generated under.
    eos_peak_at: step index at which the stop token was most probable, or a
        negative number if it was never observed.
    eos_peak_prob: that probability. Observational, it never feeds back
        into sampling, but it *does* gate two rules, so it is pinned by the
        conformance fixture like any other audible value.
    ended: whether generation stopped at the stop token rather than a cap.
    is_terminal: whether this chunk ends the passage. A continuation chunk
        has no sentence end, so its stop peak means nothing and its pauses
        are rhythm rather than dead air.
    hit_ceiling: whether generation was stopped by the length ceiling.
    silence: the sampler's silence token ids. The tail rules read exactly
        this list; ``repetition_cut`` widens it with the render censuses
        on its own, and ``is_stalled`` reads the censuses off the config.


## Notes moved from the runtime docstrings


## `loudkit/postprocess.py`


### `module`

A detector, not a filter. It reads the speech tokens a chunk produced,
answers where the sentence really stopped, and hands back a verdict; it never
touches audio. The evidence is read in the token domain because the artifact
is generated, not spectral: a free-running decoder past the end of a sentence
leaves a run of silence with a burst behind it, which is a shape in tokens
and no reliable shape in the spectrum. Integers also let five implementations
agree exactly.

Seven rules in a fixed precedence: dropout, repetition, stall, silence_tail,
terminal_echo, desperation, ended_tail. ``docs/design/postprocess.md``
carries each rule's provenance and every constant's measurement;
``docs/design/postprocess-detectors.md`` the precedence argument.


### `pacing_outliers`

The long-form drift signal, in the same integer-derived domain as the rest
of the layer: each ratio is speech tokens over text tokens for one chunk.
Report-only, the caller is told which chunks to listen to, and nothing is
cut, because a strange pace is evidence of *something* without saying what.

The median rather than the mean, so one broken chunk cannot drag the
baseline toward itself and hide.


### `_loop_candidate`

One search serves both questions. The cut index is
:func:`repetition_cut`'s contract, unchanged. ``resumed`` is whether the
winning loop's repeating region ends ``period`` or more tokens before
the row does: a locked decoder emits its cycle to the end of the row,
and a ceiling can truncate at most one incomplete copy (``period - 1``
tokens), so a full period of anything else after the region means the
decoder came back, which a locked decoder, by definition, does not.
No extra scan pays for it: a matching full copy would have been counted
as another cycle, so ``n - at >= period`` already implies a deviation.


### `ended_tail_trim`

An ended row is trusted to have stopped where it meant to, but three tail
shapes still ship dead air, walked backward as
``[sentence][r1 silence][burst][r2 silence]``:

* a bare silence run half a second long, tightened to a natural pause;
* a silence run with a 1–2 token blip right before the stop (a 40–80 ms
  click after a pause; the device specimen ended ``.......#``);
* on a *terminal* chunk only, a stray word up to ``ended_tail_word_max``
  behind a full silence seam.


### `terminal_echo_cut`

No silence seam here, so :func:`is_trailing_filler` has nothing to anchor
on. Instead the earlier stop candidate must be strong, late, and followed by
a short tail. The late-position rule is what protects real comma and clause
pauses.

The second acceptance path is narrower and exists for one measured
regression ("...but a brigand. Pass. Four.": ``gen=124/124,
bestEOS=109@0.004``). The model never sampled a stop token, but its best -
very weak, stop was 15 tokens before the hard ceiling. Weak confidence is
not trustworthy in general; it is trustworthy only with all three
corroborators together: a terminal chunk, an actual ceiling overrun, and a
very short tail occupying the last 15% of the row.


### `_is_dropout`

Two conditions, both required. The absolute floor catches a row that stopped
almost immediately whatever the text was. The proportional one is what keeps
a genuinely short line exempt: a row is only suspect when the text asked for
materially more speech than arrived, and the chunker's own conservative
estimate of "materially more" is the floor the sampler already generated
under.


### `cut`

`dropout` and `stall` condemn a row without cutting it, `keep` still holds
the whole input, and both make this true. A caller reading it as "tokens
were removed" and slicing on that gets the right answer anyway, because
`keep` is the length; a caller counting cuts with it over-counts by every
condemned row. "Whether anything was removed" is what this said, and it was
wrong for the two verdicts that exist precisely because nothing can be
removed.


## Notes moved from attribute docstrings and comments


### `loudkit/postprocess.py`


#### `ceiling_speech_per_text_token`

A guard against a three-word sentence decoding for ten seconds, and nothing
finer. Device trace of the showcase render::

t3.overrun  gen=92 ceiling=92 bestEOS=74@0.003 floor=31

A chunk of ~26 text tokens stopped only because it hit the ceiling, with the
model's confidence in stopping at 0.003, mid-sentence, already at 3.5
speech tokens per text token. Four is comfortably past anything measured.

NOT the chunker's 2.6. That number is the conservative end of a measured
1.75–2.35 and is conservative *for budgeting a chunk*, where guessing high
only wastes window. Here it is the opposite: guessing low cuts a sentence
off.


#### `trailing_silence_run_tokens`

A hallucinated word at the very end sits *behind* such a seam, silence,
then a burst of speech tokens. The burst lowers the silence share below
the share test's threshold, which is why this unbroken-run test exists:
without it, the audible tails are exactly the ones the rescue refuses to
cut.


#### `desperation_band_ratio`

The desperation rescue's fallback anchor cuts at the stop peak only if the
peak sits where a real read could have ended. Measured reads run
1.75–2.35 speech tokens per text token, so ``int(ratio * n)`` reaches past
every legitimate ending while staying well under the 4.5x garbage
threshold, a row whose best stop lands far below its own length has no
honest end in sight, and the seam rule handles that case instead.


#### `desperation_speech_per_text_token`

"It was as he expected.", 14 text tokens, came back as 96 speech tokens of
sentence-then-dense-babble, with the stop peak at the right *place* (45) but
confidence 0.000, so every probability-gated rescue refused. Measured real
speech runs 1.75–2.35 speech tokens per text token; 4.5x is unreachable by
any legitimate read, so past it the question is no longer whether to cut but
where.


#### `desperation_min_keep_per_text_token`

The specimen: soren, da0028 chunk 4, seed 1234. The row burned 132 tokens
to the ceiling and the seam cut kept 36, 1.44 s of audio in which 33 of
the 36 kept tokens render near-silent through ids outside both manifest
censuses, so no set-membership rule can see them. The trim shipped a mute
chunk and the caller was never told to retry; the same window renders
clean at seeds 7 and 99.

Calibrated on every cap-hit desperation rescue in the interior-stall and
acceptance batteries (nine rows, four voices, four languages), each kept
chunk graded against its rendered audio. Absolute keep length cannot
separate them, a keep of 36 tokens was mute on one row (23 text tokens)
and a complete short sentence on another (19), but keep per text token
can: every keep at or below 1.57 was mute or missing much of its text,
and every keep at or above 1.85 carried real speech. 1.7 sits inside that
gap, and deliberately under 1.75, the floor of the measured healthy band
of speech tokens per text token: a keep below the band floor cannot hold
a full read of its text under any measured pronunciation, so a complete
read is never condemned.

The costs are asymmetric and priced in: a false condemnation is a 2x
decode on one window (cap-hit desperation fires on ~0.5% of windows at
all), a false pass is a second and a half of silence shipped as fixed.
Cap-hit rows only, a row that ended on its own corroborated its trim
with a stop token. Zero disables the trigger.


#### `filler_max_speech_after_run`

Deliberately a separate field from ``ended_tail_word_max`` despite holding
the same number and meaning the same duration. They govern different rows -
this one every row with a seam, that one terminal ended rows, and were
settled separately. Folding them into one field would mean loosening the
trim on terminal chunks silently loosened the filler test everywhere.


#### `retry_max_attempts`

The literature's measured shape: catastrophic-failure rates drop from 5.8%
to zero at a single retry, on the failing rows. Retrying every row costs
N× compute; retrying only the rows the detectors condemned costs ~1.1× at
the measured fire rates, which is the whole argument for having detectors
that report rather than guess.

Only the verdicts nothing can trim retry: ``dropout`` (the content is
missing) and ``suspect`` (certainly wrong, nowhere to cut). A trimmed
verdict already has its fix. Each attempt draws from a **derived** seed, so
the whole ladder is a pure function of the caller's seed, reproducibility
is why this is config rather than a loop someone writes around the engine.

Zero disables it, and `RETRY_LADDER_HEADROOM` bounds it above.


#### `pacing_tolerance`

Pace is speech tokens per text token, the same integer-exact quantity the
length rules use. A long passage is rendered chunk by chunk, and a chunk
whose pace lands far from its neighbours' is rushing or dragging: the
long-form drift the literature reports as prosody breakdown past the
training window. Report-only by design, a pace is a property of a healthy
render too, and cutting on it would be guessing.

1.6 is calibrated from the nine-language probe: healthy per-chunk ratios
run 1.69–2.79 (a 1.65x spread across *languages*), so a chunk drifting
1.6x from its own passage's median is outside anything ordinary prose
produced in any of the nine.


#### `dropout_min_tokens`

The failure is early truncation, and it is the most damaging one in the set
because content is *missing*: no trim recovers it, and unlike a hallucinated
tail the listener cannot tell that anything went wrong. The published
criterion for a catastrophic neural-codec TTS failure uses exactly this
shape, a speech-token count under 25, or an ASR transcript of at most one
word, so the threshold is borrowed from a measurement rather than guessed.

It is a *floor on the row*, not on the text: a genuinely short line is
exempt, because the shortest legitimate reads measured across nine languages
run 35 tokens and up, and the rule only fires when the text asked for
materially more than it got.


#### `stall_run_tokens`

The run is measured two-class, and the two classes are essential: only
true-silence ids (``silence_render_ids``) count toward this threshold, but
the run *continues* across quiet-family ids (``quiet_render_ids``) -
breath and decay tokens that render inaudible in context. Single-set
counting was measured broken: one breath token in the middle of real dead
air split a 47-token run into two short ones and the rule missed it.

Calibrated across all ten shipping languages (120 passages per arm):
healthy interior runs top out at 13–19 tokens and healthy leading runs at
11, so 25 is outside anything ordinary prose produced anywhere while
sitting under every measured stall. 20 also clears the healthy maxima; 25
is the shipped margin.


#### `silence_render_ids`

A property of the checkpoint, measured by rendering (per-id median energy
below -80 dBFS across two independent censuses), and therefore supplied by
the manifest, top level, beside ``silence_token_ids``, precisely so the
next backend cannot re-guess it. Empty means the checkpoint predates the
census; the stall rule then runs its run trigger only, keyed to the
configured ``silence_token_ids``, degraded (only 8 of that list's 31 ids
actually render silent, so the whole-row and majority triggers cannot be
trusted with it) but safe.

NOT a sampling exemption list. Widening the sampler's ``min_p`` exemption
to exactly these ids was measured harmful, pause-time share doubles -
and the repetition penalty applies to every token regardless. This list
exists so the detectors read dead air where dead air actually is.


#### `quiet_render_ids`

Measured by per-instance RMS attribution (>= 90% of instances quiet,
>= 5 sightings), minus the true-silence census. Dead-air runs continue
across these ids but they never count toward the run gate, a breath
inside dead air is still dead air, and a breath between words is not.
Manifest-supplied like ``silence_render_ids``; empty when the checkpoint
predates the census.


#### `repetition_min_span`

Cycle count alone does not separate a stuck decoder from healthy speech:
this model winds down nearly every row with a short repeated tail token, so
"does it repeat" is true of almost all real speech. What separates a stuck
decoder is that the repetition *does not stop*.

Measured across 27 renders, nine languages, one voice: the longest naturally
repeating span in a healthy row is 10 tokens (0.4 s), median 7. The one row
in the set whose decoder genuinely ran away, a Spanish three-word phrase
that never emitted a stop token, repeats for 44 tokens (1.76 s). 24 sits
between them with 2.4x margin over the healthy maximum.


#### `repetition_resume`

The guard that holds without a census. A genuine lock-up is a tail
pathology: the model's own output is its context, the state is
absorbing, and the repeating region runs to the end of the row. Every
fire in the conformance fixture resumes by zero tokens, and a ceiling
can truncate at most one incomplete copy, ``period - 1`` tokens, at
most 11. A qualifying repetition followed by a full period or more of
other content is therefore a different event: a decoder that resumed
was never locked, and on a checkpoint without render censuses the thing
it resumed from is a pause parked on a silent-rendering id the sampler
list cannot name.

``condemn`` (the default): the row is reported whole, ``keep`` is the
full row, verdict ``repetition``, ``suspect``, and routed into the
retry ladder like ``stall``. The cut is refused because the cut *is*
the defect: kathleen, en0023, seed 1234 on the published census-less
pack parks a pause on 6486 (x8) then 6405 (x24), the period-1 run
passes every loop condition, and the cut keeps 52 of 206 tokens,
deleting the pause plus the 131 tokens of correctly-read speech behind
it, two sentences, verdict ``repetition``, no retry, audibly fluent.
``repetition_silence`` closes that on a manifest that carries the
censuses; this field closes it on every checkpoint, including ids no
census lists (da0028's mute keep rendered near-silent through ids
outside both).

Not a discard-fraction guard, though one was calibrated first. Measured
on every firing fixture case and the specimen: genuine fires discard
0.36–0.58 of the row, the specimen 0.748, so a cap of 2/3 splits the
observed sets, but a pause parked late in a row discards under any
such cap and the cut still ships (a 30-token pause at 70% of a
206-token row with one clause behind it discards 0.32). A fraction
bounds the loss;
resumption removes it. Fires resume by 0 tokens, the specimen by 131,
and the largest resume a truncated genuine loop can produce is
``period - 1``, so the law is ``resume >= period``, integer-exact,
no constant to tune, and a cut that survives it only ever removes a
tail, like every other rule in the layer.

``cut`` names the pre-amendment law, for a checkpoint measured under
it. The bare rule (:func:`repetition_cut`) reports the loop either
way; this field decides what the resolver does with one that resumed.


#### `repetition_silence`

``acoustic``: the union of the configured sampler silence ids and both
render censuses (``silence_render_ids``, ``quiet_render_ids``). A pause
parked on *any* silent-rendering id is never mistaken for a decoder loop.

The specimen that settled it: kathleen, en0023, seed 1234. A mid-chunk
pause parked on ids 6486 (x7) then 6405 (x24), both render true silence
and both are in ``silence_render_ids``, neither is in the sampler's
``silence_token_ids``. Keyed to the sampler list, the exemption could not
see them: the 24-token period-1 run fired as a loop, and the cut kept one
cycle and deleted the pause *plus the six seconds of correctly-read speech
behind it*, two whole sentences, verdict ``repetition``, not suspect, no
retry, audibly fluent. Systemically, six of the checkpoint's eight
truly-silent ids sit outside the sampler list, so this was the rule's
default behaviour on most real pauses; measured prevalence 1/200 renders,
and the shape is inaudible content loss shipping as clean.

``sampling``: the configured sampler list alone, the pre-amendment law,
nameable so a checkpoint measured under it can declare what it measured.
A checkpoint without censuses gets this behaviour under either value,
since the union degenerates to the sampler list.

This family feeds the loop exemption only. The tail rules
(``silence_tail``, ``ended_tail``, the filler and desperation seams) stay
keyed to the sampler list they were calibrated against; see
``docs/design/postprocess.md`` for the two-lists decision.


#### `suspect`

Set with ``dropout`` (content missing), with ``stall`` (the row is dead
air where speech should be), with ``repetition`` on a loop the decoder
resumed from (the cut would delete what it came back to say, so the row
is handed back whole), with a starved ``desperation`` cut (a cap-hit
trim that keeps less than any full read of its text, here ``keep``
still holds the cut, as the fallback if every retry is also condemned),
and on a row impossibly long for its text that dodged every token
anchor. Not an error, a report, and the engine's signal to retry.
It exists because the alternative (shipping such a row silently) is how
the artifact reached listeners.


#### `_validate_ranges` checks finiteness first

Every later check is a comparison and NaN loses all of them (`nan < 0` and
`nan > 0` are both False), so an unguarded NaN walks through the validator and
surfaces far away: `int(nan)` inside the ceiling, or a pacing tolerance that
nothing ever exceeds. The manifest is data from outside the process, so this
is where it is caught.


#### `inspect`: dropout is reported, never cut

There is nothing to cut: the row is already too short, and the missing content
cannot be recovered by removing more. It is the only verdict in the layer that
says "what you have is incomplete" rather than "the end of what you have is
wrong", and the most damaging failure in the set, because a listener cannot
hear that anything is absent. The floor is on the row against what the text
asked for, not on the row alone: a three-word line legitimately renders short,
and the shortest healthy reads measured across nine languages run 35 tokens.


#### `inspect`: every tail rule reads `is_terminal`

A continuation chunk's stop peak means nothing and its pauses are rhythm
rather than dead air, so every rule that reads those two signals holds off
when `is_terminal` is false: `terminal_echo_cut` with `if not is_terminal`,
`desperation_cut` through `peak_allowed`, `ended_tail_trim` by construction,
and the interior-stall rule by its own guard.


#### `inspect`: a stall is condemned whole

Unlike a starved desperation cut there is no trim worth keeping as a fallback,
because the trim is the defect.


#### `inspect`: a starved desperation cut is condemned, and the cut stands

On a cap-hit row the trim has no stop token corroborating it, and a cut
keeping fewer than `desperation_min_keep_per_text_token` speech tokens per
text token kept less than any full read of the text. The kept audio can be
near-silence through ids no census lists (da0028: 33 of the 36 kept tokens, a
mute chunk shipped as fixed), so the keep's length is the only evidence there
is. Condemned like `stall`, but the cut stands: if the retry ladder exhausts,
the trim ships, flagged `suspect`, rather than the untrimmed babble.
