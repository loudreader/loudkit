# 2. Long text and streaming

A window carries roughly ten seconds of speech. Anything longer is split,
generated in pieces, and joined. `synthesize` does that for you and hands back
one waveform; `stream` is the same synthesis delivered chunk by chunk, for when
the first audio matters.

## Streaming: hear the first sentence now

Time to first audio is set by the first chunk, not by the passage.

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
voice = engine.voice("joe")

passage = (
    "This is a longer passage, written to exercise more than a single window. "
    "It should run through several chunks and a couple of joins. "
    "So the streaming path and the long-form path are both measured."
)

for i, result in enumerate(engine.stream(passage, voice, seed=7)):
    print(f"{result.duration:.2f}s chunk ready")  # play it, don't wait
    result.save(f"chunk-{i:03}.wav")  # one file per chunk
```

`stream` yields one `Result` per chunk, as each becomes ready. Give each chunk
its own filename: saving them all to one name keeps only the last. For one
waveform, use `synthesize` below.

Each chunk is conditioned on the tail of the previous one, so the pitch contour
carries across a join. Every chunk after the first also gets its own derived
seed, so the passage streams identically whether or not the caller stops early.
The first chunk draws your seed itself, which is why a text that fits one window
reads the same through `stream`, through `synthesize`, and in every language.

## Long-form: one waveform for the whole passage

```python
result = engine.synthesize(passage, voice, seed=7)
result.save("passage.wav")
```

This equals concatenating `stream`, with the same splitting, joins and seeds, as
one `Result`. Use it when you want a file. Use `stream` when you want to start
playing before the passage is finished.

## What one waveform costs

The length is bounded by memory, not by the splitter. `synthesize` drains
the whole stream, then joins it: peak memory is the returned audio and mel plus
every chunk they were built from, about twice the passage at the moment it is
largest. A minute of speech is roughly 13 MB: 5.8 MB of fp32 audio at 24 kHz,
1 MB of mel, doubled while the join runs. An hour is under a gigabyte. A book
is not.

For content that long, write each chunk as it arrives. This holds one chunk at a
time, whatever the length:

```python
import soundfile as sf

with sf.SoundFile(
    "book.wav", "w", samplerate=engine.algorithm.sample_rate, channels=1
) as out:
    for chunk in engine.stream(passage, voice, seed=7):
        out.write(chunk.audio)
```

The bytes are the same bytes: `stream` and `synthesize` render the same
chunks with the same seeds. What you give up is the note on how the audio was
made, which `Result.save` writes and which needs the whole render to describe
it.

## Where the splits fall

Splitting decides where the reader breathes. It lives in
`AlgorithmConfig.chunking` and is identical on every backend and in every
language. The policy:

- a chunk is allowed up to `max_tokens` speech tokens (default 255, ~10 s),
- the text is cut at the strongest separator that fits: sentence end before
  clause end before comma,
- a chunk's first tokens condition on the previous chunk's last
  `prefix_tokens` tokens (default 6), for prosodic continuity,
- a period that does not end a sentence is not a cut. Two things make a period
  mid-sentence: the token in front of it is in `abbreviations`, or the next
  word starts in lower case. So `"But Mr. Smith went home"` is one chunk.
- a chunk the generator could not finish inside its window is split in two and
  both halves are spoken. The budget is characters and the window is tokens, so
  a chunk can overrun; when it does, the words that did not fit would otherwise
  be lost.

One consequence worth knowing when you consume `stream`: a re-split yields two
chunks carrying the **same** `index`, because moving it would move every later
chunk's seed. If you key on the index, key on arrival order instead.

`synthesize(..., single_window=True)` renders exactly one window and raises
`WindowOverflowError` on longer text. It exists for conformance tests, where
the unit under test is one window.

You can tune the splitter, and every backend reads the same policy. Changing
it changes the audio:

```python
from dataclasses import replace

# Start from the model's own settings, not from library defaults: an
# `algorithm=` override replaces all of them.
chunkier = engine.algorithm.with_(chunking=replace(engine.algorithm.chunking, max_tokens=120))
engine2 = lk.load("loudreader/loudr-1", algorithm=chunkier)
```

## Carrying that join across two calls

The prefix removes the stutter inside one call. A reader that fetches a chapter
paragraph by paragraph, or an agent that speaks one reply at a time, hits the
same join at every call boundary, because the carry started empty every time.

`previous_tokens` seeds it:

```python
first = engine.synthesize("Part one, which ends mid-thought", voice, seed=7)
second = engine.synthesize(
    "and part two, which continues it.",
    voice,
    seed=8,
    previous_tokens=first.tokens,
)
```

It is the same carry the loop above maintains, started non-empty, so there is
one conditioning path to keep correct. What follows from that:

- Only the last `chunking.prefix_tokens` are used. Pass the whole previous
  `Result.tokens` and let the engine slice.
- Only the **first** chunk of the new call takes it. Every chunk after that is
  conditioned on the one before, as always.
- `previous_tokens=None` is byte-for-byte the plain call. This is an execution
  input like the seed, not an algorithm value.
- Same inputs, same bytes. Two identical calls with identical history render
  identically.
- Ids outside the acoustic codebook are refused by name, at the boundary.

Over HTTP the tail comes back as the `X-Loudkit-Continuation` header on
`/v1/synthesize` and as `continuation` on the stream's `done` event. Send it back
as `previous_tokens`. The MCP `synthesize` tool answers with the same field under
the same name. See [Server and agents](04-server-and-agents.md).

## Stopping a render

A listener interrupts. Pass `should_cancel`, a callable returning a bool, to
either call; it is polled on every decode step, so the engine stops within one
step and the chunk in flight is discarded, never rendered.

```python
import threading

interrupted = threading.Event()

for chunk in engine.stream(passage, voice, seed=7, should_cancel=interrupted.is_set):
    play(chunk)  # interrupted.set() from another thread ends the loop
```

The contract is the same in every language. `stream` delivers the chunks that
finished before the cancel and then ends, without raising: those chunks are the
partial, and you flipped the flag. `synthesize` returns the whole passage or
raises `CancelledError` (Go `ErrCancelled`, Rust `error::CANCELLED`, JS
`CancelledError`, Swift `LoudKitError.cancelled`); it never hands back the
chunks that finished, because a short result and an interrupted one would be
the same object. Audio already delivered is yours to flush.

## The latency knobs

Three levers matter when a listener is waiting:

- **`Engine.warm(voice)`**. The first synthesis on a device is the slowest it
  will ever run: kernel autotune, graph capture, allocator pools. Call `warm()`
  once at startup in a long-running process, so the first request pays only warm
  latency. `loudkit serve`, gRPC and MCP already do.
- **`ChunkConfig.first_chunk_max_tokens`**. Cap only the first chunk, so the
  stream opens on the first clause instead of a full ~10 s window. Measured at a
  96-token budget: first audio 1.9 s to 1.4 s on an M3 Pro, 3.1 s to 2.6 s on a
  Jetson Orin. This changes where the first chunk ends, so it changes the
  audio.
- **`stream(..., latency_mode=...)`**. `stream` renders window *k* while
  generating window *k+1*. On a single shared GPU that overlap competes with the
  first window's render, so `stream` defaults to protecting first audio.
  `synthesize` drains everything before anyone hears a byte, so it turns the
  protection off and keeps the full overlap.

## Next

Now that you can speak a passage, [clone a voice](03-cloning-a-voice.md) that
is yours.
