# Tier 2.5: the order-balanced pairwise judge

The comparative tier of [evaluation.md](evaluation.md) that needs no
listeners. It sits between the native-speaker hour of Tier 2 and the
30-listener panel of Tier 3. It is cheap enough to run before a release. Never
quote its score as if a listening panel had produced it.

## What it measures

Two systems read the same passage. A judge model hears both and says which read
it better, on two separate dimensions:

- prosody: phrasing, emphasis, rhythm, sentence melody;
- correctness: whether the written words came out, word for word.

The two are scored apart because a fluent read can still drop a clause.

Each judgment scores 1 when loudkit is preferred, 0.5 for a tie and 0 when the
comparator is preferred. The reported figure is the mean over all judgments.
50% means equal aggregate preference on this set. It is not a pass mark.

## Why every passage is judged twice

A judge asked "recording 1 or recording 2" does not answer symmetrically: it
prefers one slot, often by a wide margin. Every passage is therefore judged
once in each presentation order, and `research/judge_report.py` reports two
numbers next to the score:

- Position bias: the mean score of whichever recording played first, pooled
  over every call, both systems and both dimensions. 0.5 means no net slot
  preference in that pool. Opposite preferences on the two dimensions can
  cancel, so a neutral value does not prove the judge ignores the slot.
- Order consistency: the share of passages judged in both orders where the two
  orders gave the same verdict (the same winner, or a tie both times). A high
  score with low order consistency is unreliable.

Publish the score with both numbers. A 70% preference at 45% order consistency
does not show a stable preference.

## The reading set

400 English passages of 250 to 500 characters, frozen in
[`tests/data/judge/reading-en.json`](../../tests/data/judge/reading-en.json)
with a content hash, built by
[`research/build_reading_set.py`](../../research/build_reading_set.py) from 22
public-domain Project Gutenberg books.

The passages are paragraphs, so the judge hears phrasing across clause
boundaries, breath placement and continuity over several sentences. A set of
isolated sentences does not test continuity.

The builder filters out dialogue, verse, headings and digits. Dialogue and verse
test a different skill. Number reading belongs in a text-normalisation probe:
here a misread number would confound the correctness dimension.

The set is probably contaminated. Gutenberg prose, and LibriVox recordings of
these books, are common in TTS training data, and loudkit's upstream training
data is not documented (see the [model card](../MODEL_CARD.md)). The set
measures reading quality on familiar material. Do not use it to claim
generalisation to unseen text.

## Confounds in the number

**Voice identity.** Prosody is not fully separable from the voice that carries
it. A comparator on a fixed provider voice against loudkit on a cloned
reference is a confounded comparison, and the judge may favour the side that
sounds more natural. Name the voice on each side wherever the number appears.

**Loudness.** The harness normalises each clip independently to -20 dBFS RMS,
on the samples, before any lossy encode. A level difference would otherwise
leak into the preference.

**Encoding.** Clips are sent as 128 kbit/s mp3 by default, to keep the upload
size of a 400-passage run small. Both sides get the same encoder settings,
although compression can affect two voices differently. Pass
`--audio-format wav` to send uncompressed audio.

**The judge is not a panel.** Its idea of "better prosody" is its own and is not
validated against human listeners here. Tier 3 (30 native listeners, forced
choice, about $430 per language per round) is the tier that settles a disputed
claim. Describe this tier's number as a judge-model preference.

## Running it

Render each system once:

```bash
python research/render_comparators.py --system loudkit --voice joe
python research/render_comparators.py --system kokoro --voice af_heart
python research/render_comparators.py --system pockettts --voice alba
python research/render_comparators.py --system kitten --voice expr-voice-2-m
python research/render_comparators.py --system piper --voice /path/to/en_US-lessac-medium.onnx
```

Each system renders into its own directory under `out/judge/audio/`, so one
system can be re-rendered without touching the others. A render skips every
passage whose WAV already exists, so an interrupted run continues where it
stopped. A re-run into the same directory therefore keeps the old WAVs. To
change a system's voice, seed or build, render under a new `--label`.

The comparators run locally:
[Kokoro](https://github.com/hexgrad/kokoro),
[Piper](https://github.com/OHF-Voice/piper1-gpl),
[Pocket TTS](https://github.com/kyutai-labs/pocket-tts) (Kyutai, 100M, CPU) and
[KittenTTS](https://github.com/KittenML/KittenTTS) (25M, ONNX). An
`elevenlabs` backend calls the paid API; it stays off unless a key and a voice
are both set.

Each system reads the passage by its own long-text path. Pocket TTS and Kokoro
take a paragraph directly. KittenTTS has no long-text path, so the harness
splits the passage at sentence ends and joins the audio. Record these adapter
differences with the result: segmentation can affect the score.

Pair two renders and judge them:

```bash
python research/render_comparators.py --manifest --system-b kokoro
python research/judge_pairwise.py \
    --manifest out/judge/loudkit-vs-kokoro.manifest.jsonl \
    --out out/judge/loudkit-vs-kokoro.jsonl --system-b kokoro
```

Aggregate the judgment files into one table and one chart. Name each judgment
file: a `loudkit-vs-*.jsonl` glob also matches the `*.manifest.jsonl` pair
files, and the report stops with a `KeyError` on them. `--losses` lists the
passages lost in *both* orders, with the judge's reason, as candidates for
listening:

```bash
python research/judge_report.py \
    out/judge/loudkit-vs-kokoro.jsonl \
    --svg docs/assets/pairwise-judge.svg --json out/judge/summary.json
```

Add the judgment file of every other finished comparison as another path.
Each path must exist: the report reads every file it is given.

Then listen. `judge_contact_sheet.py` writes a page with every passage, every
system's render side by side, and the verdict, worst first. It takes the same
explicit judgment files:

```bash
python research/judge_contact_sheet.py \
    --results out/judge/loudkit-vs-kokoro.jsonl \
    --out out/judge/listen.html
```

Listen to the renders before you publish a score. The sheet lets a person
overrule the judge, and lets a reviewer check that the rubric measured what it
claims to. Record any disagreement next to the automated score.

`--label` files a render under a name other than the backend's. That is how an
ablation runs: two loudkit builds, or two voices, have the same shape as two
systems. Render loudkit with a second voice under a new label, pair it, and
judge it into a new `--out` file. This checks the voice confound. Run it before
you publish a comparative number.

## Qualify the judge before you trust it

A model that cannot hear the audio still answers. It answers by picking a slot,
and the answer looks exactly like a verdict. Qualify every judge model before
its numbers go anywhere, on two small sets:

- Known-answer pairs: four passages where one render has a defect that a
  measurement confirms, such as a silence gap over one second. The judge must
  find the defective render in both presentation orders.
- Same-system pairs: the same system at two different seeds. Expect a score
  near 50% and no strong slot preference. One pair can still differ in
  quality, so judge the set as a whole.

The two sets take sixteen judgments and cost about $0.20 at August 2026
prices. They expose a judge that does not hear the audio and a strong slot
preference. They do not prove a judge valid in general.

Read **order consistency** and **position bias** first. A judge that always
picks the same slot scores exactly 50% on the order-balanced known-answer pairs,
because its slot holds the better render in half the calls. A judge that is
partly slot-driven scores above 50%, as the flash models in the table do.

Measured on this harness in August 2026, on that protocol:

| model | $/judgment | known-answer | order consistency | position bias |
|---|---:|---:|---:|---:|
| `google/gemini-3.1-pro-preview` | 0.016 | passes | 80-95% | 51% |
| `google/gemini-3.7-flash` | 0.0017 | 68.8% | 50% | 41% |
| `google/gemini-3.1-flash-lite` | 0.0007 | 62.5% | 25% | 12% |
| `google/gemini-2.5-flash-lite` | 0.0005 | 62.5% | 25% | 75% |
| `xiaomi/mimo-v2.5` | 0.0007 | picked the worse render, both orders | | |

In the same test no free audio model was usable. `thinkingmachines/inkling`
and `inkling-small` returned HTTP 403 unless the account opted into prompt
logging. `nvidia/nemotron-3-nano-omni` did not receive the audio and said so.
No DeepSeek, Qwen or MiniMax model on OpenRouter accepted audio input. Check
current endpoints before a new run.

## Use a measurement where one exists

A preference score answers "which reads better". It does not answer "did this
defect go away". A defect with an objective definition (a silence run past a
threshold, a truncated render, a dropped word) has a direct measurement. Use
the measurement, and check the detector against labelled examples.

Use the judge only where the question is subjective and no measurement exists.

## Use two judge families

Run the set a second time with a judge from a different model family. Pass a
different `--model` **and** a new `--out` file: the resume key is the passage
id and the presentation order, not the model, so a reused `--out` makes the
second judge skip every passage. Compare the direction of the two preferences;
their magnitudes differ.

Measured on 20 passages: `google/gemini-3.1-pro-preview` returned 80-95% order
consistency and near-neutral position bias. `openai/gpt-audio` returned 30-45%
consistency and a strong preference for whichever recording played second.
Both picked the same winner. Report order consistency and position bias for
each judge next to its score: they show which judge's score is stable.

## Cost and credentials

The judge is an audio-input model behind an OpenAI-compatible endpoint. The
default is `google/gemini-3.1-pro-preview` over OpenRouter. Credentials come
from `OPENROUTER_API_KEY`. The token is never printed and never written to the
output.

At August 2026 prices a judgment cost about **$0.016**, so 400 passages in two
orders cost about **$13 per comparator**. Check current pricing before a run.
Pilot with `--limit 20` and a cheaper audio model such as
`google/gemini-3.7-flash`, and write the pilot to its own `--out` file. In the
full run's file, the pilot's judgments would count as done and mix two judges
in one result.

Runs resume: a re-run skips every passage and order already in `--out` and
judges the rest. Resume only with the same manifest, audio and judge. Start a
new `--out` file for anything else.

## Reporting it

State, next to the number: the judge model, the comparator's voice and
loudkit's voice, the passage count, the tie rate, the confidence interval, the
order consistency, and the position bias. The interval is a passage-cluster
bootstrap: passages are resampled with replacement, and both of a passage's
judgments travel together because they are not independent observations.
Resampling individual calls ignores that dependence. When the two orders
agree, its interval is too narrow, by up to about 30%.
