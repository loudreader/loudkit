# The three silence defects

A listener reports one thing: "it stops for a second in the middle". Under the
measurement it is three different failures with three different causes, and a
change that fixes one can leave the others untouched. Reporting a single
worst-gap number per render hides which one moved, so this page names them and
fixes how they are counted.

## The three

**Head.** Silence opening a chunk that continues a sentence. A chunk after the
first carries a six-token prefix from its predecessor, which starts it inside a
sentence-final pause, and it can then open with over a second of silence.
Nothing trims it, and the reason is structural rather than an oversight:
`Inspection.keep` is defined as "how many leading tokens survive", so every
postprocess rule removes a tail by construction.

**Seam.** What the join produces: the tail of one chunk plus the head of the
next; the join is plain concatenation, so the arithmetic models the joined
product exactly. The head side is bounded by nothing. The tail side is bounded
*in intent* at `ended_tail_keep`, 5 tokens or about 0.2s, but not in practice:
the rule counts only ids on the configured silence list, and the ids that carry
roughly half of this model's real silence are not on it, so they are invisible
to the bound and ride through it. Measured tail p50 is 0.50 to 0.62s, not 0.2s.
No single chunk looks wrong; the joined audio does.

**Interior.** A run the decoder free-ran mid-clause. `min_p` is relative to
`p_max`, so when a silence token is itself `p_max` every other candidate falls
below the cutoff, and the min_p exemption in `LRSamplerV1.__call__` re-admits
the listed ids. Say it precisely: the exemption re-admits the 31 *configured* ids, of which
13 render as audible speech, so the surviving distribution is not literally all
silence. Measured with an instrumented sampler on two stalled renders, on 87% of
stall steps the cutoff removed every non-silence candidate and the exit mass
outside the exemption was exactly 0.0, while the audible listed ids carried a
median of about 1e-3. Across 1,031 hard-trap steps there were **zero escapes**.
Exit probability is effectively zero rather than exactly zero, and the
repetition-penalty exemption, in the same method, means nothing degrades the
state with time.

None of the postprocess rules *as they stood* could reach a head or an
interior run. They were all anchored on the tail or the EOS peak, and
`repetition_cut` explicitly excludes all-silence cycles, on the reasoning that
silence repeating is what silence is.

That is what the `stall` rule, described under "The fix as shipped" below, was
added for: it reaches an interior run, and it condemns the row rather than
cutting it, because a tail cut cannot remove a hole in the middle. The head
class is still unreached, and still for the structural reason above.

## How they are measured

`research/chunk_silence_stats.py`, rendering chunk by chunk through
`Engine.stream` so each chunk is inspected before the join hides it. 20ms RMS
frames, a frame is silence below 0.012 RMS, a run counts from 180ms. Head and
tail are the leading and trailing runs; interior is the longest run strictly
between them; a seam is one chunk's tail plus the next chunk's head.

Two rules keep the numbers honest:

- **Rates need hundreds of chunks.** A one-in-twenty failure measured over
  twenty paragraphs is noise. Roughly 120 passages give 500 chunks and 400
  seams, which separates a 25% rate from a 5% one with room to spare.
- **The measurement disqualifies itself.** Each chunk records its own silence
  floor. An absolute threshold only separates speech from silence while the
  silence sits well below it, so a voice rendering with audible room tone would
  have its pauses counted as speech and its gap rate reported as zero. The
  compare warns when the median floor comes within 12 dB of the threshold.

Compare within a language, never across one. Every language ships exactly two
voices, which makes each language a controlled pair: same corpus, same
frontend, only the profile differs. Cross-language rates are not comparable,
because the public-domain corpora differ in period and orthography and that
difference lands in the rate.

## What moved joe

Re-enrolling from a silence-cleaned reference, measured over 200 passages, 844
chunks and 644 seams per profile, same texts and seeds:

| | shipped | repaired |
|---|---:|---:|
| passages with any gap over 1s | 25.0% | 4.5% |
| seams over 1s | 6.1% | 1.1% |
| interiors over 1s | 2.7% | 0.2% |
| head p95 | 0.46s | 0.22s |
| seam p95 | 1.14s | 0.80s |
| interior p95 | 0.78s | 0.46s |

The shape of the change matters as much as the size. The natural pause band,
0.4 to 0.8 seconds, got **denser** — 79.6% of seams to 92.4% — while the tail
past 1.5s collapsed from 4.0% to 0.3%. A fix that suppressed pausing would have
thinned that band instead. Cutting the tail without touching the body is the
target shape for any future change here.

## The enrolment story does not cover the worst voice

The asymmetry below is real and it explains joe, kerstin, thorsten, tugao and
darkman. It does not explain soren, and that limit belongs next to it rather
than in a footnote: **soren's conditioning prompt carries 2 dead tokens in 150,
maximum run 1 — essentially clean — and he loses 300 seconds of text.**
Conversely nathalie's prompt is 25% dead tokens and she is among the cleanest
voices on the roster. A dirty prompt is sufficient to cause the defect and is
not necessary for it; the decoder-side trap is the part common to every case.

## Why the cause is in enrollment, for the voices it does explain

`TorchVoiceEnroller.enroll` tokenizes the raw, untrimmed first 6 seconds of the
reference into `cond_prompt_tokens`, while only the speaker-embedding path
trims (`_VoiceEncoder.embed`, through `librosa.effects.trim`). A reference
carrying digitally dead silence therefore puts
silence tokens into the conditioning prompt, and the prompt teaches the model
that this reader pauses. Swapping only `cond_prompt_tokens` between two voices
moves the failure with it; swapping only the speaker embedding does not.

The sampler exemptions are the enabler rather than the cause: runs begin with
silence as the model's own argmax, and the exemptions are why they do not end.
Both halves are needed to produce an audible hole, which is why a data fix
reaches most of it and a sampler fix would reach the rest.

## The silence list does not describe this model's silence

`SamplingConfig.silence_token_ids` is the set every silence-aware decision keys
on: the repetition-penalty exemption and the min_p exemption, both in
`LRSamplerV1.__call__`, and every postprocess rule that reasons about a run of
silence. `AlgorithmConfig.from_manifest` states that the checkpoint is the
authority on it, "precisely so they cannot be re-guessed by whoever writes the
next backend".

Measured against the weights, the shipped list is wrong in both directions.
Method: render, align tokens to audio (one token is exactly 0.04s, so the
mapping carries no drift), and take the median frame RMS per token id. Over
11,479 emitted tokens across four voices in three languages:

**It misses silence.** The ids that render as true dead air, at -103 dBFS or
below, are `4137, 4215, 4218, 4299, 6324, 6405, 6486`. Only 4137 and 4218 are
listed. The five unlisted ones carry 677 of 1479 silence occurrences, 46%, and
6405 alone appears 330 times, comparable to the most frequent listed id.

**It lists speech.** Of the 31 listed ids, 16 are never emitted at all. Of the
15 that are, only those same two render as silence; the other thirteen render
as audible content, including 3377 at -19.4 dBFS, which is an ordinary speech
level.

The second half matters beyond the gap problem. Thirteen non-silence tokens are
exempt from the repetition penalty and survive the min_p cutoff unconditionally.
[postprocess.md](postprocess.md) describes the artifact it exists
to remove as "any step where a non-silence token survives the cutoff becomes a
hallucinated word" — and these thirteen survive it by construction, on every
step, for every voice.

So the trap story holds only partly: 46% of real silence is already penalised
today and long runs still occur, while a fraction of the exemption is spent
protecting speech. Correcting the list changes the algorithm fingerprint, so it
is a recipe change like any other, but it is the only one of the candidate fixes
that makes a shipped value agree with the weights rather than adding new
behaviour.

Two categories must stay separate in any fix. The seven ids above are digital
silence and are what a token-domain trim rule should cut. A wider set of ids
renders between -78 and -39 dBFS — breath and decay — which falls under an
absolute RMS gap threshold, correctly, because a listener hears the hole; but
cutting those in the token domain would remove breath and clip word onsets.

## Four defects, not one, and they need different fixes

Measured across the roster on own-language natural prose. A voice can carry more
than one.

| defect | voices | does cleaning the reference help |
|---|---|---|
| contaminated conditioning prompt | joe, kerstin, thorsten, tugao, darkman | yes, these five reach their language partner's baseline |
| cap-hit silent tails | soren, at 27% of windows | no |
| whole-silence rows | nils, soren, dave, dante; none elsewhere | no |
| engine base rate | every voice, controls included | no |

The classes are separable only because head, tail, seam and interior are counted
apart. A single worst-gap number per render would have shown "nils has holes in
79% of paragraphs" and sent the work at his reference, which is not where most of
his problem is: his interior p95 is 0.80s, no worse than joe's, while his head is
2.00s and his tail 3.32s. Same symptom, different disease.

**A whole-silence row is not a pause.** It is a window of text the model never
spoke, it passes postprocess as `clean` because every rule looks for where speech
stopped and there was none, and it appears on first chunks as well as
continuation chunks. Measured on nils: 8 rows in 365 chunks, 76 seconds of text
never spoken across 120 passages. Re-enrolling barely touches it (8 to 6),
because the reference is not what causes it. The right response is the retry
ladder that already exists for condemned verdicts: a row that said nothing should
be re-rolled, not trimmed. A trim rule reaching this class would "fix" the gap by
deleting the sentence, which is why the leading-silence rule guards against it
explicitly.

## Closing it, at scale

Rates are quoted per seam, per chunk and per passage, on 120 passages per arm
with both arms rendered from the same texts and seeds.

Five voices, 120 passages per arm (joe 200), both arms of a voice rendered from
the same texts and seeds. "silent" counts chunks that produced no speech at all,
with the seconds of text never spoken beside it.

| voice | arm | head p95 | tail p95 | seam p95 | seams over 1s | silent | passages with a gap |
|---|---|---:|---:|---:|---:|---:|---:|
| joe | shipped | 0.46s | 0.68s | 1.14s | 6.1% | 11 (—) | 25% |
| joe | reference | 0.22s | 0.68s | 0.80s | 1.1% | 0 | 4% |
| kerstin | shipped | 0.36s | 0.78s | 1.00s | 4.9% | 0 | 24% |
| kerstin | reference | 0.26s | 0.78s | 0.94s | 2.3% | 0 | 8% |
| dave | shipped | 5.28s | 7.54s | 9.60s | 21% | 20 (197s) | 65% |
| dave | reference | 0.46s | 4.30s | 5.48s | 10% | 4 (33s) | 35% |
| nils | shipped | 2.00s | 3.62s | 6.66s | 34% | 10 (92s) | 79% |
| nils | reference | 1.80s | 0.78s | 3.94s | 23% | 6 (50s) | 63% |
| soren | shipped | 10.18s | 7.34s | 10.68s | 39% | 31 (300s) | 83% |
| soren | reference | 3.04s | 5.38s | 7.42s | 18% | 11 (102s) | 53% |

Read the head column first. soren's 95th-percentile chunk opening is 10.18
seconds, a full window: in one chunk in twenty he does not begin speaking at
all. Across 120 passages he loses five minutes of text outright, and cleaning
his reference recovers two thirds of that and leaves the rest.

The reference layer closes joe and kerstin, whose defect is the contaminated
prompt and nothing else. It halves dave and soren and leaves both broken. Its
ceiling is not a tuning failure: it lowers the probability of entering a pause,
and does not touch the state the decoder gets stuck in once it is there.

## What moves it furthest

Applying the repetition penalty to silence ids — dropping the
repetition-penalty exemption, keeping the min_p one — on the **shipped**
profiles, no
re-enrolment. It is the largest single lever, and it is not a closure:

Passages with any gap over 1s. The two right-hand columns are matched
pairwise against the shipped arm on whatever passages both arms rendered, and
that count is given, because it differs per voice: the sampler arms for joe and
nils were still filling when this was written.

| voice | passages matched | shipped | reference cleaned | penalty on silence |
|---|---:|---:|---:|---:|
| dave | 120 | 65% | 35% | 0.8% |
| soren | 120 | 83% | 53% | 16% |
| nils | 120 | 79% | 63% | 25% |
| joe | 77 | 26% | — | 1.3% |

Read nils and soren as improved rather than closed: 25% and 16% of paragraphs
still carry a seam over a second, against 0-8% for a healthy voice. What the
fix removes on them is the whole catastrophic half. On nils, interior runs over
a second go 16 to **zero**, tails over two seconds 29 to **zero**, chunks that
render no speech 10 to **zero** — 92 seconds of text never spoken becomes none
— and the worst seam falls from 6.66s to 1.14s. The defect that remains is a
long-ish pause at a join, which is the leading-silence rule's target, not this
one's.

A caution learned the hard way here: this rate climbed from 14.8% at 61
passages to 19.1% at 68 to 25% at 120, monotonically. Reading a rate off a run
that is still filling flatters it. Quote no number from this table without the
passage count beside it.

Text never spoken, the class that matters most: dave 197s to 33s to **zero**,
nils 92s to 50s to **zero**, soren 300s to 102s to **one second**. Chunk tails
over two seconds go to zero on both dave and soren, from 54 and 78. Interior
runs over a second go to zero on dave.

On the prosody cost, ignore the pause-time share figure of 0.213 to 0.222: it
comes from six renders of two healthy voices and its per-seed spread is wider
than the delta it claims to show. The evidence that does carry is larger.

On healthy voices, matched pairs of roughly 30 passages each: natural-band pause
counts hold within 2% in both directions (kathleen 21.4 to 21.1 per minute,
paola 16.3 to 16.2, nathalie 18.8 to 19.1), quiet share is flat, speaking rate is
flat. The French control at 80 passages is unchanged to within two hundredths of
a second on every percentile, and loses its one residual gap. On damaged voices
the natural band densifies rather than thins, seams under 0.3s — the run-on
signature — fall on every voice, and postprocess verdicts get *cleaner*:
desperation 5 to 0, terminal_echo 1 to 0, suspect events 14 to 1 across a
ten-voice sweep.

Two delivery changes should be disclosed rather than buried in that.

**The silence opening a continuation chunk collapses**, median 0.16-0.24s to
0.00-0.08s, with 88% of dave's continuation chunks opening under 0.1s. Seam
totals stay inside the natural band because the tail side still carries the
pause, so it reads as improvement — but it is a real change to how a chunk
begins.

**One voice has its pauses mildly compressed.** On henri, natural-band pauses
fall 23.94 to 22.19 per minute while sub-band ones, 0.10 to 0.18s, rise 11.06 to
12.91 — a near one-for-one transfer, so his shorter pauses are being tightened
past the counting floor rather than removed. His articulation rate is untouched
(voiced words per minute 253.4 to 253.1), so the faster reading is silence
coming out, not speech speeding up. Kathleen shows no such transfer and freja's
band holds, so this is henri alone, and he is the least healthy voice of the
healthy set. The effect is about 7% of his natural band. It is mild and it is
real, which is why "healthy voices untouched" overstates: the accurate claim is
that no voice degrades on intelligibility or gap rate, and one voice's pause
distribution tightens slightly.

**It prevents entry to the trap; it does not dismantle the trap.** The
repetition penalty is a one-time constant factor, not an accumulating one, so a
strong enough silence preference still holds. Demonstrated: one soren chunk
under the penalty still free-ran **133 consecutive true-silence tokens** to the
255-token cap, verdict `clean`, leaving a 5.6s hole. That is why this belongs
paired with an interior-stall detector routed to the existing retry ladder,
rather than shipped as a closure on its own.

What it costs is the fingerprint: this is a change to the sampling law, so it
carries a recipe bump and a golden re-base across five implementations.

Correcting the silence id list is **not** a substitute and must not be applied
to the penalty exemption. Measured, it doubles the pause-time share to 0.400 and
degrades a healthy control voice, because the exemption is the trap: widening
its membership strengthens the trap rather than removing it. The corrected list
belongs to the min_p exemption and to the detectors, not to the penalty.

## What a reference fix cannot reach

Voices with completely clean conditioning prompts still fail. On own-language
prose, control voices with zero dead prompt tokens produced gaps over a second,
and one such case was a run of 103 consecutive silence tokens on ordinary
narrative text. Cleaning a reference lowers the probability of entering a
pause. It does not remove the state the decoder gets trapped in, so a residual
rate survives every data fix and only a sampler-level or postprocess-level
change addresses it.

## The fix as shipped

Landed on the `postprocess-layer` branch, in three parts. Every number above
was the input; these are the decisions.

**The sampler applies the repetition penalty to silence.** The
repetition-penalty exemption is deleted; the min_p exemption stays, since
removing
that one instead was measured catastrophic (median gap 2.46s to 4.64s).
Measured effect of the penalty change alone, 120 passages per arm, ten
languages: paragraphs with a hole over 1s fall 33.0% to 4.3%, mute chunks 74 to
1, cap-hit tails over 2s and interior runs over 1s to zero everywhere,
natural-band pause rates inside noise on 8 of 9 healthy voices, WER flat or
better. This is a change to the sampling law: the algorithm fingerprint moved
to `9ecb061d7228eaad` and the goldens re-based. The "recipe bump" anticipated
above landed as the fingerprint move plus the new hashed config fields rather
than a `recipe_version` rename: the loader refuses a manifest naming any other
recipe, so a rename would brick every published pack, while the moved
fingerprint already separates the two laws on every surface that checks
identity. henri's mild pause tightening
(natural band -8.5%, near one-for-one transfer to the sub-band, articulation
and WER untouched) was the one effect no measurement could adjudicate, so it
went to a human ear on three French passages, before and after: accepted.

**A seventh postprocess rule, `stall`, catches what leaks through.** Two-class
detection over token ids: a run gate on true digital silence
(`silence_render_ids`, the union of the two censuses), continuation across the
breath/decay family (`quiet_render_ids`). Three triggers, each condemned into
the existing retry ladder, never cut: a row with no speech at all; a non-tail
run of 25 or more gated tokens (healthy interior runs top out at 13 to 19,
leading at 11); a cap-hit row that is majority silence, which also closes the
desperation hole (the ceiling clips at 4.0x text tokens + 40, under the 4.5x
desperation needs past 80 text tokens). The 133-token clean free-run above is
exactly what the third trigger now condemns. Provenance and constants:
`docs/design/postprocess.md`.

**The censuses ride the checkpoint manifest.** Top level, beside
`silence_token_ids`, as optional fields; an old pack loads and falls back to
the run trigger keyed to the configured list, degraded but safe. As warned
above, the corrected list is wired to the detectors only. The penalty now
applies to every token, so there is no exemption list left to widen, and the
min_p exemption keeps the shipped 31-id list unchanged.

The retry ladder also ships best-of rather than last: when every attempt is
condemned, the attempt with the fewest true-silence tokens is kept. Measured:
30% of condemned fires on the worst voices exhaust the ladder, and the last
attempt can be worse than the first.

**A follow-up hardening closed the one surviving row.** The acceptance run
left exactly one mute chunk: soren, da0028 chunk 4, seed 1234. The row burned
132 tokens to the ceiling and the desperation seam cut trimmed it to 36, 1.44
s in which 33 of the 36 kept tokens render near-silent through ids outside
both censuses, so every set-membership trigger was blind to it, and
desperation-with-cut was not a condemned verdict, so it shipped on attempt 1.
Seeds 7 and 99 render the same window clean. The law now holds a cap-hit
desperation cut to a floor: a keep under `desperation_min_keep_per_text_token`
(1.7) speech tokens per text token is condemned into the retry ladder as
`suspect`, with the cut kept as the fallback if every attempt is condemned.
1.7 was calibrated on all nine measured cap-hit desperation rescues, graded
against their rendered audio: bad keeps top out at 1.57 per text token,
healthy ones start at 1.85, and 1.75 is the floor of the measured healthy
band, so a complete read is never condemned. The floor is a new hashed config
field, so the fingerprint moved again, `9ecb061d7228eaad` to
`494ae241cbac5869`. All four ports carry both rules: each reproduces the
fingerprint from its own hand-written canonical form, each consumes the
`stall` and `starved_rescue` fixture sections, and each port's loader now
fails loudly on an unknown section instead of skipping it.

**A second hardening keyed the loop exemption to acoustic silence.** The
census warned above that the silence list misses most real silence, and the
repetition rule was still reading that list: its "a cycle that is entirely
silence is never a loop" exemption saw only `sampling.silence_token_ids`, so
a pause parked on any of the six census-only silent ids read as a period-1
decoder loop. The specimen is kathleen, en0023, seed 1234: a mid-chunk pause
on 6486 (x7) then 6405 (x24), cut as a loop, deleting the pause and the two
correctly-read sentences behind it, verdict repetition, no retry, audibly
fluent. Prevalence 1/200 renders, and the shape is the worst in this
document: inaudible content loss shipping as clean. The law now reads the
union of the sampler list and both censuses for this one exemption (new
hashed field `repetition_silence`, default `acoustic`; a census-less
checkpoint is unchanged), the declined loop falls through to `stall`, and
the retry ladder re-renders the chunk. The tail rules stay keyed to the
sampler list they were calibrated against: the ~0.5 s house tail sits at
Kokoro's level, and re-keying them would move prosody everywhere for a
defect nobody demonstrated. The fingerprint moved again,
`494ae241cbac5869` to `e8de57e99b735340` (the census-less default,
`d439a578b2075e9d` to `59f2d8ac0564485f`), and all four ports read the
field, each resolving the union where the search runs rather than in a
caller that the next caller would feed differently.

**A third hardening made the same defect unreachable without a census.** The
acoustic union is only as good as the manifest that carries it, and the
published hub pack carries no censuses: on it the union degenerates to the
sampler list and the en0023 cut fired again, measured on this laptop, 52 of
206 tokens kept. The guard that closes it reads no silence at all. A genuine
lock-up runs its cycle to the end of the row (every firing fixture case
resumes by zero tokens; a ceiling truncates at most one incomplete copy,
`period - 1` tokens), while the specimen's pause is followed by 131 tokens
of the speech the decoder came back to read. New hashed field
`repetition_resume`, default `condemn`: a loop whose region ends a full
period or more before the row does is condemned whole into the retry
ladder, `suspect`, no trim kept, because a decoder that resumed was never
locked and the trim is the defect. A discard-fraction guard was calibrated
first (fires discard 0.36 to 0.58, the specimen 0.75) and rejected: the
fraction depends on where the pause sits, and a late pause discards under
any workable cap. Resumption does not care where the pause sits, which is
what "unreachable, not unlikely" requires, and it also covers ids outside
both censuses on amended packs (the da0028 class). Provenance and the
distributions: `docs/design/postprocess.md`. The fingerprint moved
again, `e8de57e99b735340` to `1628fb0fd201b784` (the default,
`59f2d8ac0564485f` to `7ee33ab14b7f8498`). All four ports carry the law:
one search, `loop_candidate`, returns the cut index and whether the decoder
resumed, so `repetition_cut`'s own contract is untouched and `"cut"` still
names what a pack was built against.

## The enrollment gate: catching the next soren before it ships

The fix above protects listeners from the roster that exists. Nothing above
stops the next cloned voice from being pause-prone, and the per-voice patching
this document records is exactly the cost a gate at enrollment time avoids.
`tools/check_voice.py` is that gate, built on one statistic from the stall
forensics: the **band-B share of emitted silence**.

The two silence bands are one FSQ quantizer cell split at the +2187 digit
boundary (4137+2187=6324, 4218+2187=6405, 4299+2187=6486; 4215 and 6162
complete the census). A healthy voice spreads its pauses across both bands and
a pause ends in 9 to 12 tokens. A voice that concentrates on band A repeats
one shape and free-runs. Measured on all twenty voices, four own-language
passages each, seed 1234, and correlated against the pre-fix gap rate per
voice (share of 48-200 passages with any gap over 1 s, from the shipped-arm
chunkstats):

| predictor | Spearman vs pre-fix gap rate |
|---|---:|
| dead tokens in `cond_prompt_tokens` | +0.14 |
| reference speech level (p90 dBFS) | -0.58 |
| rank combination of those two | +0.47 |
| band-B share, probe under the shipped law | -0.84 |
| **band-B share, probe under the pre-fix law** | **-0.91** |

Two decisions fall straight out of that table.

**The profile alone does not suffice.** Every statistic computable from the
profile's own tokens is noise (dead count +0.14, profile band composition
-0.19). soren's conditioning prompt carries 2 dead tokens in 150 and he is
the worst voice on the roster; nathalie's carries 38 and she is among the
cleanest. A profile screen costs milliseconds and passes exactly the voice
the gate exists to catch, so the gate renders: about two minutes on this
laptop for four passages, token phase only, no audio.

**The probe re-enables the pre-fix exposure.** The shipped law penalises
repeated silence, which forces band variety and compresses the signal: the
same twenty voices separate 0.01-0.70 under the pre-fix law but only
0.52-0.83 under the shipped one. The gate measures the voice with the
mitigation off (postprocess `off`, the old penalty exemption restored in a
sampler subclass local to the tool), which also surfaces the raw pathology
directly: every voice the fix had to rescue free-ran 150-237 consecutive
silence tokens in the probe, while no clean voice exceeded 39.

The verdict lines, calibrated on the twenty: **fail** under 0.10 band-B
share, or any true-silence run of 100 tokens, or any cap-hit row that is
majority silence (soren 0.010, dave 0.026, dante 0.056, nils 0.059, plus pim
and selma whose probes free-ran); **warn** under 0.30 (thorsten 0.16, joe
0.27, kerstin 0.28, henri 0.27); **pass** above (every voice with a pre-fix
gap rate at or under 15%, and darkman at 0.38, whose defect was the
reference, not the trap). The warn band holds exactly the voices this
document showed are repaired by cleaning the reference, and that is the
instruction a warn carries.

What the gate answers when a new voice fails, measured by degrading kathleen
and re-enrolling:

- The mild defects the roster guide warns about do not break a healthy
  reference. Three 120 ms digital-silence splices (joe's construction) moved
  the share 0.59 to 0.53; level dropped to soren's -27.8 dBFS moved it to
  0.54; both at once, 0.63. All within the probe's own noise, all pass. Even
  a deliberately raw clone (1.5 s dead lead-in, a gap at every pause, low
  level) degraded the run structure (longest run 15 to 59 tokens, one cap
  hit) but still passed. The fail class is not producible by damaging a good
  recording.
- A genuinely bad source fails on arrival: soren's reference, freshly
  enrolled, probes bit-identically to his shipped profile (0.010, six
  majority-silent cap rows).
- The verdicts are actionable along class lines. joe re-enrolled from his
  cleaned reference moves 0.27 to 0.65, warn to a clean pass. soren
  re-enrolled from his cleaned reference moves 0.010 to 0.030 and still
  free-runs a fully mute row: fail stays fail, and the answer is a better
  recording, not a better cleanup.

Limits, so the number is not oversold. The gate ranks trap propensity, not
gap rate: darkman passes at a 31% pre-fix gap rate because his gaps entered
through the prompt and cleaning fixes them, and pim measured 0.15 in the
original ten-voice census but 0.06 on this probe, so a voice near a line can
cross it with the passages and the seed. Four passages at one seed is a
probe, not a census. The calibration corpus was each voice's own language.
And the engine fix already removed the catastrophic classes, so the gate's
value is preventing the next pause-prone clone and pricing its repair, not
protecting today's listeners from today's roster.

Out of scope here, by design: nils's residual is seams just over a second, and
whether any rule should reach them is a question for the roster-wide count, not
for this change. henri's tightening was listened to and accepted, so it is a
recorded property of the fix rather than an open item.

## A fifth defect: the window cuts a word and the words are gone

The four classes above are all about silence: too much of it, in the wrong
place, or a row that is nothing but. Verifying the fix across the whole roster
turned up a defect of the opposite kind, and it costs content rather than
patience.

`split_text` decides where to cut before anything is rendered, because it has
to: the budget is characters and the window is tokens, and nothing knows how one
converts into the other until the voice has spoken. The words that do not fit
are **lost, not deferred** — chunk texts are fixed before any of them renders,
so the next chunk begins at its own text. Every such row ships verdict `clean`
on the first attempt, because no postprocess detector is text-aware and none of
them can be.

Measured over 9920 chunks in ten languages: **54 hit the window cap and 30 were
still speaking when it closed.** By voice, cap hits over cap-cuts: soren 38/21,
freja 12/6, joe 1/1, kathleen 1/1, nathalie 1/1, thorsten 1/0. Six voices, not
two, and one of them German. A chunk that finished its sentence leaves a pause
before the join — of the 9866 that ended on their own, the median tail is 0.56 s
and only 5 fall under 0.10 s. A chunk the window cut off has no tail at all.
That separation is the detector, and it needs no ASR, no alignment and no
language model, so it reads the same in all ten languages.

**The mechanism is variance, not the constant, and the first version of this
section said otherwise.** `CHARS_PER_TOKEN` is 0.5, chosen as the low end with
margin against 0.53-0.64 measured on one voice in three languages, and soren
reads 0.481 — below it. That story is true of soren and of nobody else: freja
reads 0.518, above the constant, and thorsten 0.587. Taking each voice's own
median pace, **only 17 of the 54 cap hits have enough characters to need more
than 255 tokens.** The other 37 are chunks that should have fitted and did not,
because the model failed to emit a stop token in time. A mean cannot bound a
variance, which is also why calibrating one per voice could not have worked.

Three ways to fix it were priced.

**Shortening every chunk was measured and rejected.** Preventing the observed
cases needs a 92-character budget against today's 127, roughly 47% more joins
across ten languages to repair a defect concentrated in one. The derivation is a
back-fit from the shortest cap-cut chunk rather than a rendered arm, and the one
budget arm that was rendered — soren at 115 characters — produced three *new*
cap-cut chunks, so shortening also creates overruns.

**A per-voice budget from a calibrated pace was measured and rejected**, and it
had been the more promising idea: the pace would live in the voice profile, so
the fingerprint would move once and never again as voices are added. Rendering
soren at 115 characters took his word-losing passages from 18 to 3 and his holed
passages from 17 to 28 of 120, because the seam-over-a-second rate is not
independent of chunk length — it rose from 4.2% to 7.1%. Shorter chunks end
mid-clause more often, and mid-clause is where a pause-prone voice overruns. The
inverse was tested on the theory that longer chunks help a fast voice: selma at
154 characters shed 23% of her joins, and her holed passages went 10 to 11 —
a rate of 2.73% to 3.55%, no improvement at ten events either way.

**The law as shipped is reactive.** New hashed field `chunking.cap_resplit`,
default `word`: a chunk that hit the cap is replaced by its two halves, split at
the word boundary nearest the middle, generated in its place. It fires on
54/9920 = 0.54% of chunks, so the rest pay nothing. `"off"` names the old law.

The boundary is the nearest *word* break and punctuation is not sought. Cutting
at the separator nearest the middle put every long seam on a comma; a comma is
an instruction to pause, and this model has no pause-duration prior, so fed one
it overshoots — the same mechanism as every other defect in this document,
arriving through the chunker.

### What the arms actually measured

The first pair of arms is retracted. The harness reimplemented the law instead
of driving the engine, and gave the second half `_derive(seed, _STREAM_CHUNK +
index + 1)` — the *next chunk's* stream off the passage seed, where the engine
draws `_derive(chunk_seed, _STREAM_RESPLIT)`. Every seam figure it produced
described a configuration the engine never emits. The commit argued carefully
about which streams are safe to draw from while the harness validating it drew
from the wrong one.

Re-measured by driving `engine.stream` under each law, over the 51 passages that
carry a cap hit, on `dist/loudr-1`:

| | `off` | `word` |
|---|---|---|
| windows | 202 | 255 |
| joins | 151 | 204 |
| windows at the cap | 52 | 2 |
| still speaking at the cap | **29** | **1** |
| seams over 1 s | 4/151 (2.6%) | 6/204 (2.9%) |
| worst seam | 1.28 s | 1.42 s |
| audio | 1472.6 s | 1503.2 s |

The truncation class closes: soren 22 to 0, freja 6 to 0, and 30.6 seconds of
speech that the cap had been swallowing is spoken. The one row still cut is
`nathalie/nl0098`, the single word `"omringden."`, which has no space for
`split_in_half` to use — a chunk that short hits the length-proportional ceiling
rather than the window, so it belongs to a different class and splitting cannot
help it. The other row still at the cap, `soren/da0075`, ends on a 0.42 s tail:
it finished its clause.

The cost is two more seams over a second, on a rate that moves 2.6% to 2.9%.
That is the trade as shipped: a truncation that removes words silently, against
a pause a listener can hear with the words intact.

### What the index buys, and what it does not

Both halves keep the original chunk's index, so a re-split cannot move the seed
of any later chunk. **It preserves the seeds and nothing more.** Chunk k+1 is
conditioned on the last `prefix_tokens` tokens of chunk k, and after a repair
those come from the second half rather than from the truncated window, so the
audio of every later chunk in that passage moves. An earlier version of this
section, of the commit message and of five source comments claimed the passage
"sounds the same after", which is false. What the index buys is that the change
is confined to the passage that needed repairing.

The second half draws from `_STREAM_RESPLIT` off the chunk's own seed, clear of
the flow at 1, the vocoder at 2 and the retry ladder from 8 up. A half that
still overruns ships as it is; one did, and it was not cut.

Fingerprint `f95cd1f9abae916d` to `7b59fba15ad17992`; the default
`d705d7533801aa80` to `86c9f40273db3fb0`. Both moved once more before release,
`7b59fba15ad17992` to `baf8462effe3ba47` and `86c9f40273db3fb0` to
`797c55169dbc3204`, when the funnel gained the `No`/`Nl` pass (`funnel-4`) —
unrelated to the silence campaign, and recorded here because this page traces
the chain.

### The trigger is deliberately broad

`cap_resplit` fires on "the window filled", not on "the window filled and the
speaker was still speaking".

**The measurement below predates the trigger it describes, and is kept as
history rather than restated.** It was taken over `hit_token_cap`: **54 chunks
of 9920 reached a cap, 30 of them lost words, so 24 — one in four hundred —
were split for nothing** and gained a seam they did not need. Two things have
changed since, and neither has been re-measured on the roster:

- three of those 54 were the postprocess *length ceiling* rather than the
  window — a short text that ran away, with nothing to re-split — and those are
  no longer in the population at all (`engine.py`, `_windows_for_chunk`);
- the flag is read before the trim now, so filled windows that postprocess cut
  back are in the population and were not before. That direction *adds* chunks,
  and nobody has counted how many.

So the useful shape of the old number survives — a minority of cap hits are
split for nothing — and the three figures do not. Re-running the roster is what
would replace them.

Narrowing it is possible and was priced. The tail test separates the two groups
cleanly, but it needs audio, and the decision currently happens the moment
generation ends, before anything reaches the renderer. Compute is not the
obstacle: a narrowed rule costs one extra render on the 0.54% that reach the cap
and *saves* two generations and a render on the 0.24% that did not need
splitting, so the arithmetic is neutral at worst. The obstacle is that
`_stream_pipelined` runs rendering on a worker so it overlaps the next window's
generation; making the split decision wait for a render would stall the producer
and give back the overlap that exists to shorten time to first audio. Add a
fingerprint move and a fifth pass through five implementations.

Left broad on purpose. Nobody loses content either way, the benefit is 24 seams
across ten languages, and it is not established that those seams cost anything:
the measured seam-over-a-second rate moved 2.6% to 2.9% across both necessary
and unnecessary splits, and the two were not separated.

