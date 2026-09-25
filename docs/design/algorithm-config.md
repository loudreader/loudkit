# Algorithm config

Maintainer notes for `python/loudkit/config.py`: why each value is an
algorithm value, and the measurement behind it. The runtime docstrings say
what each field does.

## The two layers

Every setting belongs to one of two frozen dataclasses. `AlgorithmConfig`
holds the values that decide what the engine computes, and it is identical on
every backend. `ExecutionConfig` (see `execution-config.md`) holds how the
engine computes it: devices, precision and kernels. It can differ per
backend, and some execution choices change the numbers;
`docs/reference/IDENTITY-CONTRACT.md` states what each one guarantees.

To classify a new setting, change it on one backend only. If the output
becomes a different reading of the text, the setting is an algorithm value.
If the output is the same reading computed differently, it is an execution
value.

The engine enforces the split, because a wrong algorithm value can still
produce plausible audio. Guidance mode is an example: dual-path guidance on an
estimator distilled for single-path use gives plausible audio from a
different algorithm, and a comparison of outputs alone does not reveal it. So
each algorithm value is defined in one place, each component carries the
engine's `AlgorithmConfig`, and the engine refuses a component whose
fingerprint differs.

## The fingerprint

`AlgorithmConfig.fingerprint()` is the first sixteen hex digits of SHA-256
over `canonical_form()`. The canonical form is specified:

- floats use `repr`, the shortest string that round-trips, which is the same
  for every IEEE-754 double;
- numpy scalars are coerced to Python numbers;
- only fields the schema defines are hashed;
- keys are sorted;
- unset optional fields are omitted at every depth.

The last rule lets a new field with a default leave the fingerprint of every
unchanged algorithm as it is. Without it, every upgrade would change every
fingerprint.

`recipe_version` is in the hash because the sampling law, the Euler grid
formula, the framing recipe and the EOS arithmetic are code, not fields. Two
builds can agree on every field and still compute different things if one has
a different sampler. Bump `recipe_version` when the recipe changes.

`decode_mode` defaults to unset, not `"single"`, so its absence does not
change a fingerprint. A manifest with an explicit `"single"` hashes the same
as one without a `decode` block.

The fingerprint is an integrity check. An audible change moves it, and the
pinned values move with it (see `docs/reference/COMPATIBILITY.md`).

## Sampling

`SamplingConfig` is LR-SAMPLER-v1. It is specified so that the five
implementations agree bit for bit:

- The RNG is counter-based: a token's random number depends only on
  `(seed, stream, step, index)`. `torch.multinomial` is not used, because it
  gives different samples for the same probabilities on x86 and arm64.
- `min_p` is evaluated in logit space. It needs no softmax, no CDF scan and no
  other reduction whose order a backend could change.

`silence_token_ids` are exempt from the `min_p` filter and from nothing else.
A pause token is the only way the model pauses, and a filter that removes it
removes prosody. Without the exemption, the measured median long-form gap
goes from 2.46 s to 4.64 s. The repetition penalty applies to silence ids like
every other token. With both exemptions, a silence run has no exit: an
instrumented run found none in 1,031 trap steps, and 33% of long-form
paragraphs had a hole longer than one second.

Do not widen the list to the render census (`silence_render_ids` on the
postprocess preset). Exempting exactly the truly silent ids from `min_p` is
also measured harmful: it doubles the pause-time share. The detectors read the
census; the sampler reads this list.

`min_tokens_floor` and `min_tokens_text_ratio` form the early-EOS guard,
`max(10, int(text_tokens * 1.2))`. Speech runs at about 1.7 to 2.6 tokens per
text token, so a 1.2x floor stops early truncations without forcing overlong
reads. The dataclass defaults are 0 (off), so the bare sampling law has no
floor. The production values come from the manifest's `eos_floor` block, or
from `backends.PRODUCTION_EOS_FLOOR` and `PRODUCTION_EOS_TEXT_RATIO` for a
checkpoint whose manifest does not carry them.

## Chunking

A window holds about 10.2 s of speech, so text longer than a couple of
sentences is split, generated in pieces and joined. Where the splits fall, and
what each piece is conditioned on, change where the speech pauses. That is
audible, so they are algorithm values.

`prefix_tokens`: a chunk generated independently restarts its pitch contour
like a new sentence, which sounds like a stutter at the join. On the reference
voice, the contour restarts about 74 Hz higher at an independent join, and
about 7 Hz with a six-token prefix. Six tokens cost little extra time.

`abbreviations`: the token before a period, without the period. `Dr` and `St`
are in the list because of a second measurement. The surveyed corpus (120
passages per language) contained neither. On 24 books of English prose, the
list without them still ended a chunk on `St.` 178 times and on `Dr.` 80
times.

One union list serves every language. Over 1200 passages in ten languages, a
language-blind union re-chunks the corpus identically to ten per-language
lists: the same 4968 chunks, the same 17 remaining harmful cuts and the same 37
passages moved. The cost is one false hold in 2253 sites, at a position never
chosen as a cut. Ten lists would need a language tag, which the splitter does
not have, and a dispatch in five implementations.

This list is separate from the funnel's. `Grammar.abbreviations` expands the
unambiguous abbreviations (`e.g.` to "for example") before the splitter sees
them. What reaches the splitter is what the funnel does not expand because it
is ambiguous: `St.` is Saint or Street. The tuple is hashed as written, so keep
it sorted.

`cap_resplit`: `split_text` budgets characters against a constant measured on
one voice in three languages. A slower speaker fills the window before the
text ends. The generator then stops mid-word and the rest of the chunk is
lost, because chunk texts are fixed before any of them renders. In a census
of 9920 rendered chunks in ten languages (August 2026), 54 chunks hit a cap
and 30 were still speaking when it closed. Counting only the window cap
leaves 51 and 27. The 27 are all from the two slowest voices, and they lose
about 0.5% of those voices' words.

`"word"` splits such a chunk at the word boundary nearest its middle and
generates both halves under the same chunk index, so no later seed moves. A
second measurement drove `engine.stream` on the loudr-1 checkpoint over the
51 passages that carry a cap hit, once with `"off"` and once with `"word"`.
Windows at the cap go from 52 to 2, and windows still speaking at the cap
from 29 to 1; the remaining one is a single unbroken word. These passages
were selected for a cap hit, so the counts are not rates for the whole
roster. A half that still overruns is not split again. In the same
measurement one half still reached the cap and finished its clause on a
0.42 s tail, and a split without a bound would add a new way to fail. `"off"`
keeps the truncation, for a checkpoint measured without re-splitting.

`mid_sentence_period`: `"hold"` does not treat a period as a boundary when it
follows an abbreviation or precedes a word that starts in lower case. The
splitter then looks for an earlier break and falls back to a word boundary.
It does not take the held candidate instead: over 1200 passages, that would
save six word breaks and cost seven period breaks, five of which leave a title
dangling. The lower-case rule catches periods that the funnel creates. An
ellipsis becomes `...`, and a run of `[.,;:]` folds to one mark, so "grzeją
się... ciepłem" arrives as "grzeją się. ciepłem". That is the main cause in
Polish, which has no abbreviation cuts at all. The mechanism is a suffix test,
with one guard on the character before the match and one on the character
after the separator. It uses no regular expression, case folding or Unicode
class, so the five implementations agree character for character
(`docs/design/preprocess.md`).

`first_chunk_max_tokens`: time to first audio is the first chunk's generation
plus its render, and both grow with its length. Capping only the first chunk
starts the stream at the first clause. Measured on an M3 Pro on 2026-08-20,
before the 0.1.0 release, a 96-token first budget cuts first audio from about
1.9 s to 1.4 s, on a clause boundary. Smaller budgets save little more and cut
mid-clause. Below about 48 tokens the first chunk is no longer a phrase. The
field changes where the first split falls, so it is an algorithm value and
changes the fingerprint; when unset, it is absent from the hash. Only the
Python implementation honours it. The four ports refuse a manifest that sets
the key, `null` included, because they would split the text differently under
the same `recipe_version`.

`max_tokens` must not exceed the render window, and neither may
`sampling.max_new_tokens`. The three budgets are in three manifest blocks. A
config that guarantees an overflow mid-passage is refused at load, before any
audio plays.

## Window

`WindowConfig` is an algorithm value, not a backend detail, because padding
changes the output. In one measurement, the static window's padding was the
whole difference between two renderers: mel correlation 0.975 to 0.993,
lowest on the shortest sentence. Hardware, fp16 and the framework contributed
nothing to it. A backend that needs fixed shapes pads to them with this
recipe.

`pad_token_id`: padding with token 0, an ordinary speech unit, leaks into the
tail through the encoder's attention: +3 dB of high-band mel energy after the
last real token. The released checkpoints pad with silence unit 4254.

`static_prompt_tokens`: the reference prompt is framed at exactly 238 tokens
(longer truncates, shorter pads with the silence token), and its mel
condition occupies exactly twice that many frames.

## Edge fade

`edge_fade_seconds` is an algorithm value by the test above. A half-cosine
ramp on both edges of every rendered window changes what a listener hears at
a join. It must be the same on every backend, or the implementations disagree
at the seam. `_validate_numeric_core` bounds it to `[0.001, 0.05]`. A ramp
longer than a syllable attenuates word onsets. A ramp of only a few samples
does not cover the loud first frame of the flow decoder.

It is the one field with a conditional canonical form. `canonical_form`
writes the effective value, except 0.005, which it omits: 0.005 is the fade
that older checkpoints were fingerprinted with, so a manifest that declares it
keeps that fingerprint. The default is 0.02, applied by `window._fade_edges`.
`docs/design/postprocess.md` says what the ramp does to the waveform, and
`docs/reference/IDENTITY-CONTRACT.md` states the serialization.

## The manifest

`loudkit.manifest.algorithm_from` reads a checkpoint's manifest. An absent key
takes its default, because older packs predate several blocks. A present key
must have the right JSON type: `window: []` and `sampling_defaults: []` are
refused, not read as the defaults. A string is a `Sequence` of characters, so
`silence_token_ids: "123"` and `split_on: ". "` are refused by name, not
loaded as three tokens or two separators. The checks raise exceptions and do
not use `assert`, because `python -O` strips asserts and the manifest is
external data.

The right type means the JSON type, not what a cast accepts. Every scalar
goes through `_number`, `_int`, `_opt_int` or `_flag`, and every id list
through `_int_list`. A JSON boolean and a string of digits are refused by
name. A count given as a non-integral number is refused, not truncated, which
is the rule `postprocess_from` applies to its own counts. `int()` and
`float()` accept all three, so without these helpers `pad_token_id: true`
would load as token 1 and `max_new_tokens: "7"` as a budget of seven, each
under an unchanged fingerprint.

The count rule covers the keys that count things: the window cap and its
three lengths, the chunk budget and prefix carry, the two speech tokens, the
generation budget, the EOS floor, the step count, the sample rate, the
vocabulary size, the detector counts, and every element of the three
censuses. It does not cover `token_rate_hz`, `temperature`,
`repetition_penalty`, `min_p`, `min_tokens_text_ratio`, `guidance_rate`,
`edge_fade_seconds` or the `euler_grid` points, which take 2.7 as a value; the
rule would refuse a correct manifest there. `format_version` is outside it as
well: `checkpoint.read_manifest` reads that key with a bare `int()`, so the
reference opens a file declaring 2.7 as version 2. Go, TypeScript and Swift
truncate it the same way; Rust refuses it. The four ports carry the count rule
and both of its edges.

A refusal names the whole path, such as `manifest['window']['pad_token_id']`,
because the person who reads the error is often not the one who wrote the
manifest. `null` is a valid value where a key means it: the ragged window
lengths and the unset first-chunk budget.

The render censuses (`silence_render_ids`, `quiet_render_ids`) are properties
of the weights. They are at the manifest top level beside
`silence_token_ids`, and a manifest that puts them inside the `postprocess`
block is refused.
