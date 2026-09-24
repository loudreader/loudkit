# Evaluation methodology

Five tiers, cheapest first. Each tier states what it measures and what it
misses.

## Tier 0: conformance fixtures in CI

The conformance fixtures cover text preparation, numbers (hand-written cases
and the CLDR differential), postprocess verdicts and end-to-end tokens. They
are integer-exact across five implementations. They catch drift and
regressions. They do not measure how the audio sounds.

## Tier 1: ASR round-trip against per-language floors

`research/eval_roundtrip.py` renders the probe corpus. You transcribe the
renders with an ASR of your choice, and the script scores each language against
its **own** floor in `research/eval_floors.json`. A floor is Whisper-large-v3
CER on human FLEURS speech
([Fleurs-SLU, Appendix A.5](https://arxiv.org/html/2501.06117v1#A1.SS5)):
2.1–3.1% in eight of the languages and 4.8% in Danish. One global threshold
would be too strict for Danish. The gate is `CER ≤ 2.0 × floor`, and the script
prints both numbers together.

The factor 2.0 is a project heuristic. It is not calibrated, and a floor
applies to one ASR on one corpus: with another ASR, or on the probe corpus,
read the ratio as a rough guide. Finnish, Norwegian and Swedish have no floor
in the file, and the script reports them without a verdict.

What this tier does not see:

- French liaison. ASR normalises it away.
- The Portuguese variant axis. ASR does not separate pt-PT from pt-BR.
- Mispronunciations the ASR corrects from its own language prior. An audit of
  Chinese news TTS found 46 masked errors in 110 high-risk cases
  ([Luo and Wan, 2026](https://arxiv.org/abs/2608.10606)).

Exclude Polish respelled spans from scoring: they raise WER while being
correct. `eval_roundtrip.py` does not mask them for you. When the number
matters, score with two ASR families. In a Best-of-N study on F5-TTS and
LibriSpeech-PC, a verifier from the evaluator's own family recovered 2–3× more
apparent headroom than a cross-family pair
([Yu and Kang, 2026](https://arxiv.org/abs/2607.08256)).

## Tier 2: one native-speaker hour per language

About one hour per language with a native speaker. The protocol:

1. Render `tests/data/probes/probes.json` for the language
   (`research/eval_roundtrip.py` does this).
2. Give a native speaker the WAVs and the texts. Score each item pass or fail.
   Do not use a rating scale: on a stød error, a scale score mixes the
   listener's leniency into the result.
3. Every item names its failure class (`stød-minimal-pair`,
   `compound-boundary`, `liaison-interdite`, `accent-2-pair`), so each fail
   points to one pronunciation feature.

The probes target known contrasts instead of sampling text at random. One
controlled study that compared input representations found them
indistinguishable on a held-out set and distinguishable at 70/30 on targeted
stimuli. At the measured 1–10% failure base rate, a small random sample often
contains no failure at all: at 1%, 20 random items show none 82% of the time.

## Tier 2.5: the pairwise judge

An audio-input model hears loudkit and a comparator read the same passage and
says which read it better. Every passage is judged in both presentation
orders. The set is 400 frozen English passages, and a run costs about $13 per
comparator at August 2026 prices. No listeners are needed.

The judge gives a quick comparative number. It is not a listening panel: its
idea of good prosody is its own and is not validated against human listeners
here, and it hears voice identity whatever the rubric says. Method, confounds
and commands are in [pairwise-judge.md](pairwise-judge.md).

## Tier 3: listening panel, for a published comparative claim

30 native listeners per language, forced-choice AB on the targeted stimuli,
about $430 per language per round. The protocol is forced-choice because MOS
results depend on the scale increments. Results do not compare across
languages or across studies, so each claim stays within one language.

## What "quality-evaluated" means

Until Tier 2 has run for a language, the README claim stands: the engine
*reads* the language, and only English is quality-evaluated. Tiers 0 and 1
support "conformance-tested, and intelligibility measured against the
language's own ASR floor". That is not a claim that the language *sounds
right*. Only a native-speaker evaluation (Tier 2 or Tier 3) supports that
claim.

## NST pronunciation lexicons

Språkbanken's NST pronunciation lexicons are CC0, and
`research/fetch_nst_lexicons.py` fetches all three:

- Swedish: 927k entries, marked for accent 1 and accent 2. espeak-ng does not
  model word accent.
- Danish: 238k entries, marked for stød. The standard phonemizer drops this
  contrast: *hun* and *hund* come out identical.
- Norwegian: 785k entries, marked for tonelag.

The entries are manually checked, POS-tagged and compound-decomposed.

Uses here:

- Reference pronunciations for the Tier-2 classes an ASR cannot hear. The
  lexicon gives the expected stød for a probe word, so a listener's pass or
  fail has a written reference.
- Accent labels, if a future voice training needs them.
- The Danish *tusind*/*tusinde* dispute, which the CLDR corpus leaves open
  (see `tests/data/conformance/numbers_cldr.json`).

The transcriptions are SAMPA, not IPA. Plan the mapping before you use them.
