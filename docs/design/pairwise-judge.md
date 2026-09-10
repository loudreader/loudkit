# Tier 2.5 — the order-balanced pairwise judge

The comparative tier of [evaluation.md](evaluation.md) that needs no people.
It sits between the native-speaker hour of Tier 2 and the 30-listener panel of
Tier 3: cheap enough to run before a release, weak enough that it must never be
quoted as if a panel had spoken. Both halves of that sentence matter.

## What it measures

Two systems read the same passage. A judge model hears both and says which read
it better, on two dimensions kept apart because they dissociate:

- **Prosody** — phrasing, emphasis, rhythm, sentence melody.
- **Correctness** — whether the written words came out, word for word.

A system can read beautifully and drop a clause. One combined score hides that;
two scores show it.

Scoring is the usual convention: loudkit preferred is 1, tie 0.5, comparator
preferred 0, and the reported figure is the mean over every judgment. **50% is
parity**, not a pass.

## Why every passage is judged twice

A judge asked "recording 1 or recording 2" does not answer symmetrically. It
has a favourite slot, and the size of that preference is not small. Every
passage is therefore judged once in each presentation order, and the harness
reports two numbers that let a reader see the machinery:

- **Position bias** — the mean score of whichever recording was played first,
  across every call and both systems. 0.5 means the slot did not matter.
- **Order consistency** — the share of passages where both orders named the
  same winner. A high score with low consistency is a coin flip with a rosette
  on it.

Publish the headline number with both. A 70% preference at 45% order
consistency is not a 70% preference.

## The reading set

400 English passages of 250 to 500 characters, frozen in
[`tests/data/judge/reading-en.json`](../../tests/data/judge/reading-en.json)
with a content hash, built by
[`research/build_reading_set.py`](../../research/build_reading_set.py) from 22
public-domain Project Gutenberg books.

Paragraphs, not sentences, because the thing being measured is reading:
phrasing across clause boundaries, breath placement, and whether a system still
knows where it is four sentences in. A set of isolated sentences measures none
of that.

Dialogue, verse, headings and digits are filtered out. Dialogue and verse
reward a different skill; digits belong in a text-normalisation probe, where a
failure is a diagnosis rather than a confound on the correctness dimension.

**This set is contaminated and says so.** Gutenberg prose is in the training
data of most open TTS systems, loudkit included, and LibriVox recordings of
these same books are in many. It measures reading quality on familiar material.
It does not measure generalisation, and no claim built on it should say
otherwise.

## Confounds that are in the number

Written down rather than argued away.

**Voice identity.** Prosody is not fully separable from the voice carrying it.
A comparator on a fixed provider voice against loudkit on a cloned reference is
not a clean comparison, and the asymmetry favours whichever side sounds more
like a person. Name the voice used on each side wherever the number appears.

**Loudness.** Each clip is independently normalised to -20 dBFS average before
it is sent, on the samples, before any lossy encode. Level is the single
easiest thing to prefer for the wrong reason.

**Encoding.** Clips are sent as 128 kbit/s mp3 by default, to keep a 400-
passage run inside a sane upload budget. The same encode is applied to both
sides, so it cannot favour either. Pass `--audio-format wav` to remove it.

**The judge is not a panel.** Its notion of "better prosody" is its own, and
unvalidated against human listeners here. Tier 3 — 30 native listeners, forced
choice, ~$430 per language per round — is the tier that settles a disputed
claim. This one produces a defensible number in an afternoon, and should be
described as what it is.

## Running it

Render each system once. Rendering is resumable, so an interrupted run
continues and a single system can be re-rendered without touching the others.

```bash
python research/render_comparators.py --system loudkit --voice joe
python research/render_comparators.py --system kokoro --voice af_heart
python research/render_comparators.py --system pockettts --voice alba
python research/render_comparators.py --system kitten --voice expr-voice-2-m
python research/render_comparators.py --system piper --voice /path/to/en_US-lessac-medium.onnx
```

The comparators are the systems a reader could run on the same laptop:
[Kokoro](https://github.com/hexgrad/kokoro),
[Piper](https://github.com/OHF-Voice/piper1-gpl),
[Pocket TTS](https://github.com/kyutai-labs/pocket-tts) (Kyutai, 100M, CPU) and
[KittenTTS](https://github.com/KittenML/KittenTTS) (25M, ONNX). An
`elevenlabs` backend exists for anyone who wants a paid-API ceiling on the
chart; it stays off unless a key and a voice are both set.

Each system reads the passage by its own long-text path. Pocket TTS and Kokoro
take a paragraph directly; KittenTTS has no long-text path, so the passage is
split at sentence ends and joined. Measuring a system on an interface it never
claimed to have would be scoring the harness, not the system.

Pair two renders and judge them:

```bash
python research/render_comparators.py --manifest --system-b kokoro
python research/judge_pairwise.py \
    --manifest out/judge/loudkit-vs-kokoro.manifest.jsonl \
    --out out/judge/loudkit-vs-kokoro.jsonl --system-b kokoro
```

Aggregate every comparator into one table and one chart. `--losses` lists the
passages lost in *both* orders, with the judge's reason, which is the only
output here that says what to go and fix:

```bash
python research/judge_report.py out/judge/loudkit-vs-*.jsonl \
    --svg docs/assets/pairwise-judge.svg --json out/judge/summary.json
```

Then listen. `judge_contact_sheet.py` writes a page with every passage, every
system's render side by side, and the verdict that was given, worst first:

```bash
python research/judge_contact_sheet.py --results out/judge/loudkit-vs-*.jsonl \
    --out out/judge/listen.html
```

A number nobody has listened behind is a number nobody should publish. The
sheet exists so a person can overrule the judge, and so a reviewer can check
that the rubric measured what it claims to.

`--label` files a render under a name other than the backend's, which is how an
ablation is run: two loudkit builds, or two voices, are the same shape as two
systems. Swapping loudkit's voice and re-running is the control for the voice
confound, and it should be run before any comparative number is published.

## Qualify the judge before you trust it

A cheap model that cannot hear still answers. It answers by picking a slot, and
the answer looks exactly like a verdict. Qualify every judge model before its
numbers go anywhere, on two small sets:

- **Known-answer pairs.** Four passages where one render is objectively worse by
  a measure that needs no model, such as a silence gap over a second. The judge
  must find them, and must find them in both presentation orders.
- **Null pairs.** The same system against itself at a different seed. The judge
  should land near 50% with no strong slot preference.

Sixteen judgments, about twenty cents, and it settles the question. Read
**order consistency** and **position bias**, not the score: a slot-driven model
still scores above 50% on known-answer pairs, because half the time the slot it
prefers happens to hold the better render.

Measured on this harness in August 2026, on that protocol:

| model | $/judgment | known-answer | order consistency | position bias |
|---|---:|---:|---:|---:|
| `google/gemini-3.1-pro-preview` | 0.016 | passes | 80-95% | 51% |
| `google/gemini-3.7-flash` | 0.0017 | 68.8% | 50% | 41% |
| `google/gemini-3.1-flash-lite` | 0.0007 | 62.5% | 25% | 12% |
| `google/gemini-2.5-flash-lite` | 0.0005 | 62.5% | 25% | 75% |
| `xiaomi/mimo-v2.5` | 0.0007 | picked the worse render, both orders | | |

Free audio models were all unusable: `thinkingmachines/inkling` and
`inkling-small` return HTTP 403 unless the account opts into prompt logging, and
`nvidia/nemotron-3-nano-omni` never receives the audio and says so. No DeepSeek,
Qwen or MiniMax model on OpenRouter accepts audio input at all.

The lesson generalises past this harness: when a judgment is cheap enough to be
free, check that something was judged.

## Do not reach for a judge when a measurement exists

A preference score answers "which reads better". It is the wrong instrument for
"did this defect go away". A defect with an objective definition -- a silence run
past a threshold, a truncated render, a dropped word -- has a measurement that is
deterministic, reproducible, free, and not an opinion. Measure it.

The judge earns its cost only where the question is genuinely subjective and
there is no measurement to take.

## Use two judge families

One judge is one opinion. Run the set a second time with `--model` pointing at
a different family and compare *direction*, not magnitude. Measured here on 20
passages: `google/gemini-3.1-pro-preview` returned 80-95% order consistency and
near-neutral position bias, while `openai/gpt-audio` returned 30-45% consistency
and a strong preference for whichever recording played second. Both agreed on
who won. Only the first is worth quoting.

That asymmetry is the argument for reporting order consistency and position
bias next to every score: they are what told us which judge to believe.

## Cost and credentials

The judge is any audio-input model on an OpenAI-compatible endpoint. The
default is `google/gemini-3.1-pro-preview` over OpenRouter. Credentials come
from `OPENROUTER_API_KEY`, or from opencode's stored auth; the token is never
printed and never written to the output.

Measured on this harness: about **$0.016 per judgment**, so 400 passages in two
orders is roughly **$13 per comparator**. Pilot with `--limit 20` and a cheaper
audio model such as `google/gemini-3.7-flash` before committing to a full run.

Runs are resumable. Re-running the same command retries only what failed.

## Reporting it

State, next to the number: the judge model, the comparator's voice and
loudkit's voice, the passage count, the tie rate, the confidence interval, the
order consistency, and the position bias. The interval is a passage-cluster
bootstrap — passages are resampled with replacement and both of a passage's
judgments travel together, because they are not independent observations.
Resampling individual calls reports an interval roughly a third too narrow.
