# Speed

`speed` is what the control on a video player means: 1.5x is the same voice,
sooner. The pitch does not move, so it is not a resampler.

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
voice = lk.voice("joe", repo="loudreader/loudr-1")

result = engine.synthesize_long("A long passage.", voice, seed=7, speed=1.5)
result.speed     # 1.5
result.duration  # two thirds of what 1.0 would give
```

The result carries the value it was rendered at, and
[the provenance manifest](provenance.md) records it too.

## Where it is accepted

| entry point | how |
| --- | --- |
| `Engine.synthesize`, `Engine.synthesize_long`, `Engine.stream` | `speed=1.5` |
| CLI | `loudkit speak --speed 1.5` |
| HTTP, both routes | `"speed": 1.5` in the request body |
| MCP `synthesize` tool | `speed` argument |
| gRPC | `speed` field, where `0` is read as unset |

Long-form and streaming stretch each chunk on its own, with the same constants.
There is no join to hear, because a chunk is a whole utterance either way.

## The range is 0.5 to 2.0, and it is enforced

Outside that range the call is **refused, not clamped**. A caller who asked for
3x and silently got 2x has a bug that only a stopwatch finds. Python raises
`ValueError`, Go returns an error, Rust returns `Err`, Swift throws
`LoudKitError.shape` and TypeScript throws a `RangeError`. The HTTP server
answers 4xx, including on the OpenAI-compatible route,
whose own specification allows 0.25 to 4.0 and whose reply says which half of
that this engine takes. A non-finite value is refused as well.

The bounds are public, because a UI drawing a speed slider needs them and should
not retype them:

| implementation | the two constants |
| --- | --- |
| Python | `lk.MIN_SPEED`, `lk.MAX_SPEED` |
| TypeScript | `MIN_SPEED`, `MAX_SPEED`, from the package index |
| Go | `timestretch.MinSpeed`, `timestretch.MaxSpeed` |
| Rust | `loudkit::timestretch::MIN_SPEED`, `MAX_SPEED` |
| Swift | `TimeStretch.minSpeed`, `TimeStretch.maxSpeed` |

The stretcher itself is not public in any of them. It is how the engine renders,
not something a caller composes with.

## `speed=1.0` is an exact bypass

The default does not enter the DSP at all. The waveform you get is the vocoder's
own array, unmodified and, in Python, the same object. Every conformance vector,
every golden byte and every existing caller is unaffected by this feature
existing. Tests assert it rather than assume it.

`speed` is also **not part of the algorithm fingerprint**. It is an execution
input like the seed and the text: two engines that disagree about it are still
computing the same thing. See
[the identity contract](IDENTITY-CONTRACT.md).

## What it does to the audio

The algorithm is WSOLA, waveform similarity overlap-add, written from first
principles and identical in all five implementations. It stays in the time
domain: it cuts the input into ~25 ms frames and overlap-adds them at a hop
scaled by `speed`, moving each read position by up to ±10 ms to wherever the
waveform best continues the frame already written. A plosive is copied whole or
not copied, so there is nothing to smear. There is no RNG and no adaptivity: the
same input gives the same output, always.

Output length is exactly `floor(n / speed + 0.5)` samples.

At 1.25x this is hard to distinguish from a native reading. Toward the bounds it
is audibly processed: the alignment search cannot always find a match within
±10 ms, and the artefact is a faint roughness, occasionally a doubled consonant.
0.5x is the least convincing direction, because stretching invents overlap that
was never spoken.

A genuinely faster *reading*, rather than faster *playback*, is a different
feature. It would have to come from the model, not from the samples.

## Interaction with timestamps

The stretch runs last, after the postprocess detectors have inspected the render
and before the waveform is returned. Those detectors measure pacing as duration
per token, and a 2x reading stretched first would look like a dropout to them.

`Result.chunks` is computed on the stretched waveform, so the spans it reports
are the spans of the audio you were handed. **There is no `1/speed` correction
to apply**, and applying one would double-count. See
[timestamps.md](timestamps.md).
