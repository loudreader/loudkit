# Timestamps

Highlighting the sentence being spoken, cutting on a boundary, seeking to a
word: all of it starts at `Result.chunks`. It answers in **two tiers, one of
which is a guess**. Keeping them apart is the whole design. A single "word
timings" list that quietly mixed a measurement with an estimate would be worse
than shipping neither.

```python
result = engine.synthesize_long("One. Two. Three.", voice, seed=7)
for chunk in result.chunks:
    print(f"{chunk.start:6.3f}–{chunk.end:6.3f}  {chunk.text}")
    for word in chunk.words:
        print(f"    ~{word.start:6.3f}  {word.text}")
```

## Tier 1: chunk spans are exact

The engine renders each chunk to its own waveform and concatenates them, so it
already knows every chunk's sample offset and sample length. `ChunkTiming.start`
and `.end` are those offsets divided by the sample rate. Nothing is estimated.

Two properties hold and are tested:

- **Adjacent to the last bit.** Chunk *k*'s `end` is the same float as chunk
  *k+1*'s `start`, because both are the same integer sample offset over the same
  rate. Offsets accumulate in samples and are converted once. A highlight driven
  by `time >= start` therefore cannot flicker in a gap or light two chunks at
  once, a failure mode that a comparison with a tolerance would never catch.
- **Complete.** The first `start` is `0.0`, the last `end` is `Result.duration`,
  and the spans tile the audio with nothing left over.

A single-window `synthesize()` gets one entry covering the whole result.

## Tier 2: word times are an estimate

The model emits speech tokens, not an alignment. **Nothing in this pipeline
knows where a word begins.** `ChunkTiming.words` splits the chunk on whitespace
and shares the chunk's real duration out in proportion to each word's length in
characters. That is the entire algorithm.

It is right often enough to drive a highlight at sentence scale, and wrong in
the expected ways: a long word said fast, a short word held, a breath before a
clause. **The error grows with the length of the chunk**, because one bad guess
early shifts everything after it. A sentence is usually fine. A long paragraph
rendered as a single chunk is not.

If you need real word boundaries, you need a forced aligner. This is not one.

What the estimate does guarantee, and what the tests pin:

- monotonic: word *i*'s `end` is word *i+1*'s `start`;
- bounded: every word lies inside its chunk's span;
- complete: every whitespace-separated word appears, exactly once, in order.

Punctuation stays attached to its word (`"world!"`), because the split is on
whitespace. A caller lighting up the word wants the full stop lit with it, and a
caller matching back against their own text needs the substring to be a
substring.

Word length is counted in **code points**, not bytes, so the same text weights
the same way in all five implementations.

## The text is the post-funnel text

`ChunkTiming.text` is what was tokenised, not what you passed in. The speech
funnel runs first: numbers become words, abbreviations expand, and Polish
respells embedded English. `"I have 3 apples."` comes back as `"I have three
apples."`, because that is what the engine spoke and therefore what the timings
describe. Matching a highlight against your original string will drift the moment
a digit appears. Highlight against `chunk.text`, or map back yourself.

## Streaming

Each streamed `Result` is one chunk and carries one `ChunkTiming` **starting at
zero**. A streamed chunk is its own result and cannot know what preceded it, so
reporting anything else would be a guess about the caller's playback. Stitch the
offsets as you go:

```python
at = 0.0
for part in engine.stream(text, voice, seed=7):
    span = part.chunks[0].shifted(at)  # moves the words with the chunk
    schedule(span.text, span.start, span.end)
    at += part.duration
```

`synthesize_long()` does exactly this internally, in samples rather than
seconds.

## Interaction with `speed`

Timings are measured on the waveform the caller receives, **after** any
time-stretch. A result rendered with `speed=1.5` reports the shortened spans
directly. There is no `1/speed` correction to apply, and applying one would
double-count. See [speed.md](speed.md).

## The other four implementations

Go, Rust, TypeScript and Swift compute the same two tiers with the same
arithmetic: sample offsets accumulated as integers, words weighted by code-point
count. The per-chunk type each port already had (`Chunk`) carries its own
timing, and the long-form path returns the stitched timeline.
