# 2. Streaming and long-form

A window carries roughly ten seconds of speech. Anything longer is split,
generated in pieces, and joined. loudkit gives you both shapes: a stream for when
the first audio matters, and a long-form call for when you want one waveform.

## Streaming: hear the first sentence now

Time to first audio is set by the first chunk, not by the passage.

```python
import loudkit as lk

engine = lk.load("loudr-1.safetensors")
voice = lk.VoiceProfile.load("voices/joe.safetensors")

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
waveform, use `synthesize_long` below.

Each chunk is conditioned on the tail of the previous one, so the pitch contour
carries across a join. Each chunk also gets its own derived seed, so the passage
streams identically whether or not the caller stops early.

## Long-form: one waveform for the whole passage

```python
result = engine.synthesize_long(passage, voice, seed=7)
result.save("passage.wav")
```

This equals concatenating `stream`, with the same splitting, joins and seeds, as
one `Result`. Use it when you want a file. Use `stream` when you want to start
playing before the passage is finished.

## What long-form costs

The length is bounded by memory, not by the splitter. `synthesize_long` drains
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

The bytes are the same bytes: `stream` and `synthesize_long` render the same
chunks with the same seeds. What you give up is the provenance manifest, which
`Result.save` writes and which needs the whole render to describe it.

## Where the splits fall

Splitting is an algorithm decision: it determines where the reader breathes. It
lives in `AlgorithmConfig.chunking` and is identical on every backend. The
policy:

- a chunk is allowed up to `max_tokens` speech tokens (default 255, ~10 s),
- the text is cut at the strongest separator that fits: sentence end before
  clause end before comma,
- a chunk's first tokens condition on the previous chunk's last
  `prefix_tokens` tokens (default 6), for prosodic continuity.

You can tune it, and every backend reads the same policy:

```python
from dataclasses import replace
from loudkit.config import AlgorithmConfig

chunkier = AlgorithmConfig().with_(chunking=replace(AlgorithmConfig().chunking, max_tokens=120))
engine2 = lk.load("loudr-1.safetensors", algorithm=chunkier)
```

## Carrying that join across two calls

The prefix removes the stutter inside one call. A reader that fetches a chapter
paragraph by paragraph, or an agent that speaks one reply at a time, hits the
same join at every call boundary, because the carry started empty every time.

`previous_tokens` seeds it:

```python
first = engine.synthesize_long("Part one, which ends mid-thought", voice, seed=7)
second = engine.synthesize_long(
    "and part two, which continues it.",
    voice,
    seed=8,
    previous_tokens=first.tokens,
)
```

It is the same carry the loop above maintains, started non-empty, so there is
one conditioning path to keep correct. What follows from that:

- Only the last `chunking.prefix_tokens` are used. Pass the whole previous
  `Result.tokens` and let the engine slice, rather than keeping your own copy of
  an algorithm value.
- Only the **first** chunk of the new call takes it. Every chunk after that is
  conditioned on the one before, as always.
- `previous_tokens=None` is byte-for-byte the old behaviour, and the fingerprint
  does not move. This is an execution input like the seed, not an algorithm
  value.
- Same inputs, same bytes. Two identical calls with identical history render
  identically.
- Ids outside the acoustic codebook are refused by name, at the boundary, rather
  than three stages later as an index error.

Over HTTP the tail comes back as the `X-Loudkit-Continuation` header on
`/v1/synthesize` and as `continuation` on the stream's `done` event. Send it back
as `previous_tokens`. The MCP `synthesize` tool answers with the same field under
the same name. See
[04-server-and-agents.md](04-server-and-agents.md).


## The latency knobs

Three levers matter when a listener is waiting, all execution-safe:

- **`Engine.warm(voice)`**. The first synthesis on a device is the slowest it
  will ever run: kernel autotune, graph capture, allocator pools. Call `warm()`
  once at startup in a long-running process, so the first request pays only warm
  latency.
  `loudkit serve`, gRPC and MCP already do.
- **`ChunkConfig.first_chunk_max_tokens`**. Cap only the first chunk, so the
  stream opens on the first clause instead of a full ~10 s window. Measured at a
  96-token budget: first audio 1.9 → 1.4 s on an M3 Pro, 3.1 → 2.6 s on a Jetson
  Orin. This is an algorithm value, so setting it re-fingerprints.
- **`stream(..., latency_mode=...)`**. `stream` renders window *k* while
  generating window *k+1*. On a single shared GPU that overlap competes with the
  first window's render, so `stream` defaults to protecting first audio.
  `synthesize_long` drains everything before anyone hears a byte, so it turns
  the protection off and keeps the full overlap.

## Next

Now that you can speak a passage, clone a voice that is yours.
