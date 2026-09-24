# 2. Long text and streaming

A window holds about ten seconds of speech. Longer text is split into chunks,
generated chunk by chunk, and joined. `synthesize` returns one waveform for
the whole text. `stream` runs the same synthesis and yields each chunk as it
is ready, so playback can start before the passage is finished.

## Streaming

The first chunk sets the time to first audio.

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
voice = engine.voice("joe")

passage = (
    "The train leaves the city at sunrise and follows the river north. "
    "After the first stop, the valley narrows and the mountains come into view. "
    "By noon we reach the village where the walk begins."
)

for i, result in enumerate(engine.stream(passage, voice, seed=7)):
    print(f"{result.duration:.2f}s chunk ready")  # start playing it here
    result.save(f"chunk-{i:03}.wav")  # one file per chunk
```

`stream` yields one `Result` per chunk, as each becomes ready. Give each chunk
its own filename: saving them all to one name keeps only the last. For one
waveform, use `synthesize` below.

Each chunk after the first is conditioned on the last speech tokens of the
chunk before it. This helps the pitch contour continue across the join. Each
later chunk also gets its own seed, derived from yours and from its position in
the text. A stream stopped early has played the same chunks as the start of
the full render. The first chunk uses your seed unchanged, so text that fits
one window gives the same audio from `stream` and from `synthesize`. All five
implementations use the same seed rule.

## Long-form: one waveform for the whole passage

```python
result = engine.synthesize(passage, voice, seed=7)
result.save("passage.wav")
```

The result is what `stream` yields, joined into one `Result`, with the same
splits, joins and seeds.

## What one waveform costs

`synthesize` collects every chunk, then joins them. At the peak, memory holds
the returned audio and mel plus the chunks they were built from: about twice
the passage. One minute of speech peaks at about 13 MB: 5.8 MB of float32
audio at 24 kHz and 1 MB of mel, doubled during the join. One hour stays under
1 GB. The model and the backend use memory of their own on top of this.

For longer content, write each chunk to disk as it arrives. Memory then does
not grow with the text: `stream` holds at most two chunks ahead of your loop.

```python
import soundfile as sf

with sf.SoundFile(
    "book.wav", "w", samplerate=engine.algorithm.sample_rate, channels=1
) as out:
    for chunk in engine.stream(passage, voice, seed=7):
        out.write(chunk.audio)
```

The audio is what `synthesize` returns, because both render the same chunks
with the same seeds. The file has no
[loudkit record of how the audio was made](../reference/provenance.md).
`Result.save` writes that record, and it needs the whole render.

## Where the splits fall

`AlgorithmConfig.chunking` sets where one chunk ends and the next starts.
Every backend and every port applies the same policy:

- The splitter budgets each chunk at `max_tokens` speech tokens (default 255,
  about 10 s). It estimates the tokens from the character count.
- It cuts the text at the strongest separator that fits: a sentence end, then
  a clause end, then a comma.
- A chunk's first tokens are conditioned on the previous chunk's last
  `prefix_tokens` tokens (default 6).
- A period that does not end a sentence is not a cut. A period is mid-sentence
  when the token in front of it is in `abbreviations`, or when the next word
  starts in lower case. So `"But Mr. Smith went home"` is one chunk.
- If a chunk fills its window before its text ends, the engine splits the
  chunk in two at a word boundary and generates both halves. The budget counts
  characters and the window counts tokens, so a chunk can overrun. A half that
  still fills its window is not split again.

In Go, Rust, Swift and JS, a stream chunk carries an `index`. The two halves of
a re-split chunk carry the same `index`, so the split does not move the seed of
any later chunk. The Python `Result` has no `index`: number the results in
arrival order with `enumerate`, as in the first example.

`synthesize(..., single_window=True)` renders exactly one window and raises
`WindowOverflowError` when the text does not fit it. Conformance tests use it
to test one window.

You can tune the splitter, and every backend reads the same policy. A change
to it changes the audio:

```python
from dataclasses import replace

# Copy the model's own settings: an `algorithm=` override replaces all of them.
chunkier = engine.algorithm.with_(chunking=replace(engine.algorithm.chunking, max_tokens=120))
engine2 = lk.load("loudreader/loudr-1", algorithm=chunkier)
```

## Carrying that join across two calls

Inside one call, each chunk continues from the tail of the chunk before it. A
new call starts with an empty tail, so text sent in parts (paragraph by
paragraph, or reply by reply) gets no continuation at each call boundary.
`previous_tokens` supplies it:

```python
first = engine.synthesize("Part one, which ends mid-thought", voice, seed=7)
second = engine.synthesize(
    "and part two, which continues it.",
    voice,
    seed=8,
    previous_tokens=first.tokens,
)
```

It is the same carry that joins the chunks inside one call. The rules:

- Pass the whole previous `Result.tokens`. The engine uses the last
  `chunking.prefix_tokens` of them. loudr-1-turbo aligns that tail to whole
  token pairs.
- Only the first chunk of the new call uses it. Every later chunk is
  conditioned on the chunk before it, as usual.
- `previous_tokens=None` gives the same audio as a call without it. Like the
  seed, it is an execution input and leaves the algorithm settings unchanged.
- Two calls with the same inputs and the same `previous_tokens` give the same
  audio.
- A token id outside the acoustic codebook raises `InvalidTokensError`.

Over HTTP the tail comes back as the `X-Loudkit-Continuation` header on
`/v1/synthesize` and as `continuation` on the stream's `done` event. Send it back
as `previous_tokens`. The MCP `synthesize` tool answers with the same field under
the same name. See [Server and agents](04-server-and-agents.md).

## Stopping a render

Pass `should_cancel`, a callable that returns a bool, to either call. The
engine checks it at every decode step and between render stages. A render
stage that has started runs to its end. The chunk in progress is discarded.

```python
import threading

interrupted = threading.Event()

for chunk in engine.stream(passage, voice, seed=7, should_cancel=interrupted.is_set):
    play(chunk)  # interrupted.set() from another thread ends the loop
```

All five implementations follow the same rule. `stream` yields the chunks that
finished before the cancel, then ends without an error. `synthesize` returns
the whole passage or raises `CancelledError` (Go `ErrCancelled`, Rust
`error::CANCELLED`, JS `CancelledError`, Swift `LoudKitError.cancelled`). It
never returns the chunks that finished. To stop playback at once, also discard
the audio your player has queued.

## Settings for time to first audio

- `Engine.warm(voice)` runs one short synthesis. The first synthesis on a
  device pays one-time costs: kernel autotune, graph capture, allocator pools.
  Call `warm()` once at startup in a long-running process. `loudkit serve`,
  gRPC and MCP already do.
- `ChunkConfig.first_chunk_max_tokens` caps only the first chunk, so the stream
  can start on the first clause instead of a full window of about 10 s. This
  changes where the first chunk ends, so it changes the audio. At a 96-token
  budget, first audio fell from 1.9 s to 1.4 s on an M3 Pro, and from 3.1 s to
  2.55 s on a Jetson Orin Nano Super (a 0.1.0 measurement).
  [Benchmarks](../benchmarks.md) lists the 0.1.1 first-audio times without the
  cap.
- `stream(..., latency_mode=True)` is the default. `stream` renders window *k*
  while it generates window *k+1*. When the generator and the renderer share a
  device, `stream` holds back the second window's generation until the first
  window has rendered. `synthesize` turns this off and keeps the full overlap,
  because it returns only after the last chunk.

## Next

[Cloning a voice](03-cloning-a-voice.md)
