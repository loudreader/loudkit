# Speed

`speed` changes the playback rate and keeps the pitch, like the speed control
of a video player. At 1.5x the same reading takes two thirds of the time.

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
voice = lk.voice("joe", repo="loudreader/loudr-1")

result = engine.synthesize("A long passage.", voice, seed=7, speed=1.5)
result.speed     # 1.5
result.duration  # two thirds of what 1.0 would give
```

The result carries the value it was rendered at, and
[the provenance manifest](provenance.md) records it too.

## Where it is accepted

| entry point | how |
| --- | --- |
| `Engine.synthesize`, `Engine.stream` | `speed=1.5` |
| CLI | `loudkit speak --speed 1.5` |
| HTTP, both routes | `"speed": 1.5` in the request body |
| MCP `synthesize` tool | `speed` argument |
| gRPC | `speed` field, `optional` so an omitted one means 1.0 |

Long-form synthesis and streaming stretch each chunk separately, with the same
constants. The chunks are already rendered separately, so the stretch adds no
join of its own.

## The range is 0.5 to 2.0

A value outside that range is **refused, not clamped**. A non-finite value is
refused too. Python raises `ValueError`, Go returns an error, Rust returns
`Err`, Swift throws `LoudKitError.shape` and TypeScript throws a `RangeError`.
The HTTP server answers 4xx, including on the OpenAI-compatible route. OpenAI's
specification allows 0.25 to 4.0, and the error on that route names the loudkit
range.

Each implementation exports the bounds as constants:

| implementation | the two constants |
| --- | --- |
| Python | `lk.MIN_SPEED`, `lk.MAX_SPEED` |
| TypeScript | `MIN_SPEED`, `MAX_SPEED`, from the package index |
| Go | `timestretch.MinSpeed`, `timestretch.MaxSpeed` |
| Rust | `loudkit::timestretch::MIN_SPEED`, `MAX_SPEED` |
| Swift | `TimeStretch.minSpeed`, `TimeStretch.maxSpeed` |

Go, Rust and Swift also export the stretcher itself:
`timestretch.TimeStretch`, `loudkit::timestretch::time_stretch` and
`TimeStretch.timeStretch`. Python and TypeScript do not export it. Set the
speed through the `speed` argument.

## `speed=1.0` skips the stretch

At the default speed the stretch returns its input unchanged. The edge fade
still runs on every window at every speed and tapers both ends of the window.
Tests check that the stretch returns the same array at 1.0, and that
`speed=1.0` gives the same audio as a call without `speed`.

`speed` is an execution input, like the seed and the text, and is **not part of
the algorithm fingerprint**. Two renders at different speeds have the same
fingerprint and different audio. See
[the identity contract](IDENTITY-CONTRACT.md).

## What it does to the audio

The algorithm is WSOLA (waveform similarity overlap-add). All five
implementations use the same algorithm and the same constants. It works in the
time domain: it cuts the input into 25 ms frames and overlap-adds them at a hop
scaled by `speed`. Each read position moves by up to ±10 ms, to where the
waveform best continues the frame already written. A plosive is copied with its
frame or skipped, so it is not smeared. The stretch uses no random numbers, so
the same input gives the same output.

Output length is exactly `floor(n / speed + 0.5)` samples.

At 1.25x the result is hard to tell from a natural reading. Toward the bounds
it sounds processed, because the alignment search cannot always find a match
within ±10 ms. The artefact is a faint roughness, and sometimes a doubled
consonant. 0.5x sounds the least natural, because the stretch repeats audio
that was spoken once.

`speed` changes playback only. A faster speaking style needs a change in the
model.

## Interaction with timestamps

The postprocess detectors judge the generated speech tokens before any audio
is rendered. The render then runs the vocoder, the stretch and the edge fade,
in that order.

`Result.chunks` is computed on the final waveform, so the spans it reports are
the spans of the audio you get. **There is no `1/speed` correction to apply**,
and applying one would double-count. See [timestamps.md](timestamps.md).
