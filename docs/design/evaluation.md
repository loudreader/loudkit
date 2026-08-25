# Evaluation methodology

Three tiers, cheapest first, each with what it can and cannot see.

## Tier 0 — runs in CI, costs nothing

The conformance fixtures: text preparation, numbers (hand-written + the CLDR
differential), postprocess verdicts, end-to-end tokens. Integer-exact across
five implementations. These catch drift and regression; they cannot hear.

## Tier 1 — the ASR round-trip, against per-language floors

`tools/eval_roundtrip.py` renders the probe corpus and scores transcripts
against each language's **own** floor (`tools/eval_floors.json`): Whisper's CER
on human speech is 2.1–3.1% in eight of our languages and **4.8% in Danish** —
one global threshold would hold Danish to an impossible bar while letting
English coast. The gate is `CER ≤ 2.0 × floor`, both numbers always printed
together.

What this tier cannot see, written down so nobody trusts it further than it
goes: French liaison (ASR normalises it away), the Portuguese variant axis
(ASR is variant-agnostic), and mispronunciations the ASR corrects from its own
language prior — a published audit found 46 of 110 high-risk cases masked.
Polish respelled spans must be excluded from scoring: they raise WER while
being correct. When the number matters, run two ASR families — a same-family
verifier recovers 2–3× more apparent headroom than a cross-family pair.

## Tier 2 — one native-speaker hour per language

The highest-value evaluation available, and an inexpensive one — an hour
per language with a native speaker. The protocol:

1. Render `tests/data/probes/probes.json` for the language
   (`tools/eval_roundtrip.py` does this).
2. Sit a native speaker down with the wavs and the texts. Binary pass/fail per
   item, no scales — a Likert score on a stød error measures the listener's
   politeness, not the audio.
3. Every item names its failure class (`stød-minimal-pair`,
   `compound-boundary`, `liaison-interdite`, `accent-2-pair`), so a fail is a
   diagnosis, not a mood.

The probes are targeted, not sampled, because that is where the evidence says
the signal lives: the one controlled study that compared input representations
found them indistinguishable on a held-out set and distinguishable at 70/30 on
targeted stimuli. A random-sample listening test at the measured 1–10% failure
base rate would measure zero.

## Tier 3 — only for a published comparative claim

30 native listeners per language, forced-choice AB on the targeted stimuli,
~$430 per language per round. Not comparable across languages or across studies
— MOS scale increments change the result — so claims stay within-language.

## What "quality evaluated" claims

Until Tier 2 has run for a language, the README's claim stays as it is: the
engine *reads* the language; only English has been quality-evaluated. Tier 0
and Tier 1 upgrade that to "conformance-tested and intelligibility-measured
against the language's own ASR floor" — which is more than most open TTS
publishes, and still not a claim that it *sounds right*. Only a native ear
buys that sentence.

## The NST lexicons — the unused gold

Språkbanken's NST pronunciation lexicons are all CC0 and all fetched by
`tools/fetch_nst_lexicons.py`: Swedish (927k entries, **accent-1/accent-2
marked** — labels no open toolchain uses, and espeak-ng has no word-accent
machinery at all), Danish (238k entries, **stød marked** — the contrast the
standard phonemizer collapses: hun/hund come out identical), Norwegian (785k,
tonelag). Manually checked, POS-tagged, compound-decomposed.

What they are for here, in order of value: scoring the Tier-2 probe classes
that no ASR can hear (a stød pass/fail can be checked against the lexicon's
marking, not just a native ear); deriving accent labels if a future voice
training wants them; and adjudicating the Danish tusind/tusinde dispute the
CLDR corpus could not settle. SAMPA, not IPA — budget the mapping before
consuming.
