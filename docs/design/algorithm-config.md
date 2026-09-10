# Algorithm config: what is computed, and why each value is where it is

Maintainer notes for `python/loudkit/config.py`. The runtime docstrings say
what each field does; this page says why it is an algorithm value and what
measurement settled it.

## The two layers

Every setting belongs to one of two frozen dataclasses. `AlgorithmConfig` is
everything that decides what comes out and is identical on every backend.
`ExecutionConfig` (see `execution-config.md`) is everything that decides how
fast and is free to differ per backend. The test for a new setting: if it
changed on one backend only, would the output be a different reading of the
text, or the same reading computed differently? A different reading is
algorithm.

The split is enforced rather than documented because a misapplied execution
setting can change the audio while both outputs stay plausible. Guidance mode
is the founding example: applying dual-path guidance to an estimator that was
distilled for single-path use yields plausible audio under a different
algorithm, and no output comparison alone can tell it from correct output. So
every algorithm value has exactly one home, every component carries the
engine's config, and the engine refuses a component whose fingerprint
disagrees.

## The fingerprint

`AlgorithmConfig.fingerprint()` is sixteen hex digits of SHA-256 over
`canonical_form()`. The canonical form is specified, not incidental: floats
use `repr` (the shortest round-tripping string, the same on every IEEE-754
double), numpy scalars are coerced, only fields the schema knows are hashed,
keys are sorted, and unset optional fields drop out at every depth. That last
rule is what lets a field be added with a default without re-fingerprinting
every algorithm that did not change; a check that cries wolf on every upgrade
is a check people learn to override.

`recipe_version` travels in the hash because the sampling law, the Euler grid
formula, the framing recipe and the EOS arithmetic are code, not fields. Two
builds could agree on every value and still compute different things because
one shipped a different sampler. Bump it when the recipe changes.

`decode_mode` is unset rather than `"single"` so that adding the field left
every existing fingerprint alone; an explicit `"single"` in a manifest hashes
the same as an absent block for the same reason.

The fingerprint is a final integrity check, not a design constraint. When an
audible change moves it, the pins move with it (see `COMPATIBILITY.md`).

## Sampling

`SamplingConfig` is LR-SAMPLER-v1, specified so five implementations agree
bit for bit: a counter-based RNG (a token's random number depends on
`(seed, stream, step, index)` alone, because `torch.multinomial` gives
different samples for the same probabilities on x86 and arm64) and `min_p`
evaluated in logit space (no softmax, no CDF scan, no reduction whose order a
backend could vary).

`silence_token_ids` are exempt from the `min_p` floor and from nothing else.
A pause token is the only way to pause, and a cutoff that removes it removes
prosody: dropping the exemption was measured catastrophic (median long-form
gap 2.46 s to 4.64 s). The ids were also exempt from the repetition penalty
until the interior-stall study showed the pair of exemptions makes a silence
run absorbing (no exit in 1,031 instrumented trap steps, a hole over one
second in 33% of long-form paragraphs). The penalty now applies to silence
like every other token. Do not widen the list to the render census
(`silence_render_ids` on the postprocess preset): exempting exactly the
truly-silent ids from `min_p` was measured harmful too, doubling pause-time
share. The detectors read the census; the sampler keeps this list.

`min_tokens_floor` and `min_tokens_text_ratio` are the shipped engine's
early-EOS guard, `max(10, textIds * 6/5)`: speech runs about 1.7 to 2.6
tokens per text token, so a 1.2x floor stops early truncations without
forcing overlong reads. The dataclass defaults are 0 (off) so the bare law
stays the textbook one; the production values come from the manifest, or
from `backends.PRODUCTION_EOS_FLOOR` for a checkpoint packed before its
manifest carried them.

## Chunking

A window carries about 10.2 s of speech, so anything longer than a couple of
sentences is split, generated in pieces and joined. Where the splits fall and
what each piece is conditioned on decide where the reader breathes, which is
audible, so they are algorithm values.

`prefix_tokens`: generating each chunk independently restarts its pitch
contour like a new sentence, heard as a stutter at the join. On the reference
voice the contour restarts about 74 Hz higher at an independent join against
about 7 Hz with a six-token prefix. Six costs little generation time on
discarded tokens and removes the restart.

`abbreviations`: the token before the period, without it. `Dr` and `St` are
there on a second measurement: the surveyed corpus (120 passages per
language) contained neither, but on 24 books of English prose the list
without them still ended a chunk on `St.` 178 times and on `Dr.` 80 times.
One union list serves every language: over 1200 passages in ten languages a
language-blind union re-chunks the corpus identically to ten per-language
lists (same 4968 chunks, same 17 remaining harmful cuts, same 37 passages
moved), at the cost of one false hold in 2253 sites at a position never
chosen as a cut. Ten lists would need a language tag the splitter does not
have and a dispatch in five implementations. This list is not the funnel's:
`Grammar.abbreviations` expands the unambiguous ones (`e.g.` to "for
example") before the splitter sees them; what reaches here is the residue the
funnel refuses to expand because `St.` is Saint or Street. The tuple is
hashed as written, so keep it sorted.

`cap_resplit`: `split_text` budgets characters against a constant measured on
one voice in three languages. A slower speaker fills the window before the
text runs out, the generator stops mid-word and the remainder is lost,
because chunk texts are fixed before any of them renders. Over 9920 rendered
chunks in ten languages, 51 hit the window cap and 27 were still speaking
when it closed, about 0.5% of the words of the two slowest voices. `"word"`
splits the chunk at the word boundary nearest its middle and generates both
halves under the same chunk index so no later seed moves; measured over the
51 passages, windows still speaking at the cap went from 29 to 1, and the
remaining one is a single unbroken word. A half that still overruns is not
split again: over the 51 passages exactly one half still reached the cap and
finished its clause on a 0.42 s tail, and an unbounded split is a new way to
fail. `"off"` ships the truncation, for a checkpoint measured under the old
law.

`mid_sentence_period`: `"hold"` treats a period after an abbreviation, or
before a word starting in lower case, as not a boundary; the splitter looks
for an earlier break and falls through to a word boundary. It does not take
the held candidate after all: over 1200 passages that saved six word breaks
and cost seven period breaks, five of which left a title dangling. The
lower-case rule catches what the funnel manufactures: an ellipsis becomes
`...`, a run of `[.,;:]` folds to one mark, so "grzeją się... ciepłem"
arrives as "grzeją się. ciepłem". That is the dominant cause in Polish, which
has no abbreviation cuts at all. The mechanism is a suffix test, one guard on
the character before the match and one on the character after the separator,
with no regular expression, case folding or Unicode class, so five
implementations agree character for character (`docs/design/preprocess.md`).

`first_chunk_max_tokens`: time to first audio is the first chunk's
generation plus its render, and both scale with its length. Capping only the
first chunk starts the stream at the first clause. On an Apple laptop a
96-token first budget cut first audio from about 1.9 s to 1.4 s on a clause
boundary; smaller budgets shave little more and cut mid-clause. Below about
48 tokens the first chunk stops being a phrase. It changes where the first
split falls, so it is an algorithm value and re-fingerprints; unset, it is
absent from the hash. Python only for now.

`max_tokens` must not exceed the render window and neither may
`sampling.max_new_tokens`: the three budgets live in three manifest blocks,
and a config that guarantees a mid-passage overflow is refused at load rather
than after audio has played.

## Window

`WindowConfig` is here rather than in a backend because the pad-and-truncate
recipe was the entire measured deviation of one renderer from another (mel
correlation 0.975 to 0.993, worst on the shortest sentence). It traced not to
the hardware, not to fp16 and not to the framework, but to the static
window's padding differing from the reference path's. A backend that needs
fixed shapes may pad to them; it may not decide what padding means.

`pad_token_id`: padding with token 0, an ordinary speech unit, bleeds into
the tail through the encoder's attention (+3 dB of high-band mel energy after
the last real token). The shipped engine pads with silence unit 4254.

`static_prompt_tokens`: the reference prompt is framed at exactly 238 tokens
(longer truncates, shorter pads with the silence token) and its mel condition
occupies exactly twice that many frames.

## Edge fade

`edge_fade_seconds` is an algorithm value by the test above: a half-cosine
ramp on both edges of every rendered window changes what a listener hears at
a join, and it changes it the same way on every backend or the ports disagree
at the seam. It is bounded to `[0.001, 0.05]` in `_validate_numeric_core`,
because a ramp longer than a syllable eats word onsets and a ramp shorter than
a few samples does not cover the flow decoder's hot first frame.

It is the one field with a conditional canonical form: `canonical_form` writes
the effective value unless it is the historical 0.005, which drops out so
manifests predating the field keep their fingerprints. The shipped default is
0.02, applied by `window._fade_edges`. `docs/design/postprocess.md` says what
the ramp does to the waveform and `docs/reference/IDENTITY-CONTRACT.md` states
the serialization.

## The manifest

`loudkit.manifest.algorithm_from` reads a checkpoint's manifest. Absent keys
default (older packs predate several blocks); present keys must be the right
shape. `manifest.get(key) or default` once treated `window: []` and
`sampling_defaults: {}` as the defaults, loading a truncated pack under a
fingerprint that said its algorithm had been chosen. A string is a `Sequence`
of characters, so `silence_token_ids: "123"` and `split_on: ". "` are refused
by name rather than loaded as three tokens or two separators. Raised, never
asserted, because `python -O` strips asserts and the manifest is external
data.

Right shape means the JSON type, not what a cast would accept. Every scalar
goes through `_number`, `_int`, `_opt_int` or `_flag`, and every id list
through `_int_list`: a JSON boolean and a string of digits are refused by
name, and a count given as a non-integral number is refused rather than
truncated, the rule `postprocess_from` already applied to its own counts.
`int()` and `float()` accept all three, so `pad_token_id: true` loaded as
token 1 and `max_new_tokens: "7"` as a budget of seven, each yielding a
working engine under a fingerprint that said the manifest had been read.

The count rule covers the keys that count things: the window cap and its
three lengths, the chunk budget and prefix carry, the two speech tokens, the
generation budget, the EOS floor, the step count, the sample rate, the
vocabulary size, the detector counts, and every element of the three
censuses. It does not cover `token_rate_hz`, `temperature`,
`repetition_penalty`, `min_p`, `min_tokens_text_ratio`, `guidance_rate`,
`edge_fade_seconds` or the `euler_grid` points, which take 2.7 as a value; a
check applied there refuses a manifest that is correct. `format_version` is
outside it as well, because `checkpoint._read_manifest` reads that key with a
bare `int()`, so a file declaring 2.7 opens in all five and what decides it
is the number. The four ports carry the rule and both of its edges.
A refusal names the whole path, `manifest['window']['pad_token_id']`, because
the reader that refuses is not the one the manifest author reads. `null` is
still a value where a key means it: the ragged window lengths and the unset
first-chunk budget.

The render censuses (`silence_render_ids`, `quiet_render_ids`) are properties
of the weights and live at the manifest top level beside `silence_token_ids`;
a manifest that puts them inside the `postprocess` block is refused so one
value has one home.
