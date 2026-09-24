# Timestamps

`Result.chunks` gives the timing of the audio in two tiers: exact chunk spans
and estimated word spans. Use the chunk spans to highlight the sentence being
spoken, to cut on a boundary or to seek. Use the word spans only where an
estimate is enough.

```python
result = engine.synthesize("One. Two. Three.", voice, seed=7)
for chunk in result.chunks:
    print(f"{chunk.start:6.3f}–{chunk.end:6.3f}  {chunk.text}")
    for word in chunk.words:
        print(f"    ~{word.start:6.3f}  {word.text}")
```

## Tier 1: chunk spans are exact

The engine renders each chunk to its own waveform and concatenates them, so it
knows every chunk's sample offset and sample length. `ChunkTiming.start` and
`.end` are those offsets divided by the sample rate. Nothing is estimated.

Two properties hold, and tests check them:

- Chunk *k*'s `end` is the same float as chunk *k+1*'s `start`, because both
  come from the same integer sample offset over the same rate. Offsets
  accumulate in samples and are converted once. A highlight that selects the
  chunk with `start <= time < end` therefore lights exactly one chunk at any
  time inside the audio.
- The first `start` is `0.0`, the last `end` is `Result.duration`, and the
  spans cover the audio with no gap and no overlap.

A single-window `synthesize()` gets one entry covering the whole result.

## Tier 2: word times are an estimate

The model does not output a word alignment. `ChunkTiming.words` splits the
chunk text on whitespace and divides the chunk's measured duration among the
words in proportion to their length in characters.

The estimate is good enough to highlight words at sentence scale. It is wrong
where the speech rate or the pauses vary: a long word said fast, a short word
held, a breath before a clause. **The error grows with the length of the
chunk**, because one early error shifts every later word. A long paragraph
rendered as one chunk drifts more than a sentence.

For measured word boundaries, use a forced aligner.

What the estimate does guarantee, and what the tests pin:

- monotonic: word *i*'s `end` is word *i+1*'s `start`;
- bounded: every word lies inside its chunk's span;
- complete: every whitespace-separated word appears, exactly once, in order.

Punctuation stays attached to its word (`"world!"`), because the split is on
whitespace. Each word is therefore a substring of `chunk.text`.

Word length is counted in **code points**, not bytes, so the same text gets the
same weights in all five implementations.

## The text is the post-funnel text

`ChunkTiming.text` is the text that was tokenised, not the text you passed in.
The speech funnel runs first: numbers become words, abbreviations expand, and
Polish respells embedded English. `"I have 3 apples."` comes back as
`"I have three apples."`. It is the model's input, not a transcript of the
audio. A highlight matched against your original string drifts at the first
digit. Highlight against `chunk.text`, or keep your own map back to the
original.

## Streaming

Each streamed `Result` is one chunk and carries one `ChunkTiming` that starts
at zero, relative to that result's own audio. Add the offsets as you play the
chunks:

```python
at = 0.0
for part in engine.stream(text, voice, seed=7):
    span = part.chunks[0].shifted(at)  # moves the words with the chunk
    schedule(span.text, span.start, span.end)
    at += part.duration
```

`synthesize()` does the same internally, and counts in samples.

## Interaction with `speed`

Timings are measured on the waveform the caller receives, **after** any
time-stretch. A result rendered with `speed=1.5` reports the shortened spans
directly. There is no `1/speed` correction to apply, and applying one would
double-count. See [speed.md](speed.md).

## The other four implementations

Go, Rust, TypeScript and Swift compute the same two tiers with the same
arithmetic: sample offsets accumulated as integers, words weighted by
code-point count. Each port's stream chunk type (`Chunk`, or `StreamChunk` in
TypeScript) carries its own timing, and long-form synthesis returns the joined
timeline.
