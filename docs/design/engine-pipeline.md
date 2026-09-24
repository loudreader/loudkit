# Engine pipeline: seeds, windows, the stream, and the wedge

Maintainer notes for `python/loudkit/__init__.py`, `engine.py`, `contracts.py`,
`window.py`, `stream.py`, `timing.py` and `result.py`: why the sequencing is
the way it is, and what each piece is for. The runtime docstrings point here
and do not repeat it.

## Loading and calling

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
engine.synthesize("Hello there.", engine.voice("joe"), seed=7).save("out.wav")
```

`Engine.synthesize` takes text of any length; `Engine.stream` is the same
synthesis delivered chunk by chunk. The same text, voice, seed and execution
config give a bit-identical waveform on a given build and device. Tokens agree
across backends only where `docs/reference/IDENTITY-CONTRACT.md` says they do.
`docs/design/algorithm-config.md` says why the algorithm and the execution
layers are separate.

```python
engine = lk.load("loudreader/loudr-1")      # downloads, caches
engine = lk.load("./loudr-1.safetensors")   # exactly this file
```

A path that exists is loaded as that file; any other string is resolved as a
repo id. Two models exist, `loudreader/loudr-1` and
`loudreader/loudr-1-turbo`, and both run on every Python backend: torch, ONNX
and CoreML. `load` refuses a backend whose runtime is not installed, in one
sentence, before any download. `device` is `"cpu"`, `"cuda"`, `"mps"`,
`"onnx"`, `"coreml"` or `None` for the best available, and must agree with
`execution.device`. `execution` names the execution fields to change and
leaves the rest to the checkpoint. `algorithm` overrides the checkpoint's
algorithm, and the fingerprint reflects the override. `revision` is a branch,
tag or commit for a repo id. Pin a commit for a reproducible load, because a
branch or a tag can move.

`synthesize` renders text that fits one window in one window. It splits
longer text at sentence boundaries (at a word boundary when a sentence is too
long), conditions each chunk on the tail of the one before, and joins the
audio. Its tokens are the ones `stream` yields for the same seed. The
arguments:

- `seed`: with the same build, device and execution config, the same seed
  gives a bit-identical waveform.
- `language`: for the text frontend. `None` takes the voice's language, then
  `"en"`.
- `speed`: playback speed in `[0.5, 2.0]`, pitch preserved. At `1.0` the time
  stretch leaves the audio unchanged.
- `previous_tokens`: `Result.tokens` of the call this one continues. Only the
  tail is used.
- `single_window`: refuse text that does not fit one window, with
  `WindowOverflowError`, instead of splitting it.
- `should_cancel`: polled on every decode step; `synthesize` raises
  `CancelledError` when it returns true.

```python
mine = lk.enroll("me.wav", "loudreader/loudr-1", name="mine")
```

A path is read at its native sample rate; a sample array is taken at
`sample_rate`. The enroller resamples with the Hann-windowed-sinc resampler
that all five implementations share. `enroll` builds a new enroller per call.
For bulk enrollment, build one with
`loudkit.backends.torch_backend.build_torch_enroller` from the enrollment
checkpoint (`loudr-1-enrollment.safetensors`) and reuse it. `enroll` raises
`FileNotFoundError` when the release ships no enrollment weights or no voice
encoder, because it cannot clone without them. Obtain the speaker's consent
first: see `RESPONSIBLE_USE.md`.

## The components

```
text ──▶ TextFrontend ──▶ text tokens
                               │
voice ─▶ VoiceEnroller ──▶ VoiceProfile
                               │
                               ▼
                        TokenGenerator ──▶ speech tokens   (25 Hz, discrete)
                               │
                               ▼
                          MelDecoder ──▶ mel               (80 bins)
                               │
                               ▼
                            Vocoder ──▶ waveform           (24 kHz)
```

`Engine` composes a text frontend, a token generator, a mel decoder and a
vocoder. The algorithm values are in `AlgorithmConfig`, and the speed comes
from the backends. The engine sequences the stages, derives the seeds, and
refuses a component whose algorithm differs from its own. It checks the
vocoder too: the vocoder reads the window config for its padding geometry, and
a vocoder that frames the tail differently produces different audio. A
component with no `config` is refused, not skipped, because an unchecked
component could compute anything while every reported fingerprint agrees.

The engine is a frozen dataclass, so no code can swap a component after that
check. Its one mutable field, `_wedged`, is written through
`object.__setattr__`. It records whether the engine can still run, not what
it computes.

`TokenGenerator`, `MelDecoder` and `Vocoder` each carry the engine's
`AlgorithmConfig` and hold no algorithm state of their own: the config
decides what a stage computes, and the backend decides how. They are
protocols because an implementation may be a torch module, an ONNX session or
a CoreML package, which have no common base class. The protocols define only
the call signatures.

The stage boundaries are *value* boundaries. Tokens are integers, a mel is an
array, a waveform is an array. No component hands another a live model object,
a device handle or a cache. This allows the device split below, and it allows
two backends to be compared stage by stage when they disagree.

For that comparison, every `TokenGenerator` provides `teacher_forced_logits`,
which returns `(len(forced) + 1, vocab)`. Once two free-running generations
differ by one token, they read different histories, and every later logit is
incomparable. Teacher forcing holds the context identical, so the logits
differ only by the arithmetic.

The sample rate and the token rate are manifest fields, read by
`manifest.algorithm_from` and validated on `AlgorithmConfig`. The geometry
under them is fixed: one speech token is `TOKEN_MEL_RATIO` mel frames and one
mel frame is `UPSAMPLE_PER_FRAME` samples, in every backend. A checkpoint
could therefore declare a rate the renderer does not produce. Every duration
the engine states comes from the declared rate, and every sample comes from
the geometry: with a 12.5 Hz tokenizer, the 255-token window would be
announced as 20.4 s of speech and rendered as 10.2 s. The engine requires
`token_rate_hz == sample_rate / (TOKEN_MEL_RATIO * UPSAMPLE_PER_FRAME)` and
refuses to build otherwise. The declared rate is also in the fingerprint, so a
checkpoint that changes it is a different algorithm.

## Stages may run on different devices

The token generator is autoregressive: a few hundred small dispatches per
token at batch one. The renderer runs a few large parallel passes over the
whole window. On MPS the default places the generator on the CPU and the
renderer on the GPU. With the streaming pipeline, the two stages then run in
parallel: window k renders on the GPU while window k+1 generates on the CPU. A
single window has no overlap to gain; `execution-config.md` has the
measurements. The components pass arrays, not device tensors, so the split
needs no extra code.

## Stage seeds

Each stage draws its own sub-stream of the caller's seed through
`window._derive(seed, stream)`, a 64-bit mix. A change in how many numbers the
sampler consumes cannot shift the flow prior, so an optimisation in one stage
does not shift another stage's random numbers. The streams:

| stream | who draws it |
|---|---|
| 1 | the flow prior (mel decoder) |
| 2 | the vocoder excitation |
| 8 + n | retry attempt n of a condemned window |
| 16 + k | chunk k > 0 of a long passage |
| 4096 | the second half of a re-split chunk, derived from the chunk's own seed |

Chunk 0 draws the caller's seed itself, not `derive(seed, 16)`. So one
`synthesize` covers every length: a text that fits one window renders what the
four ports' single-window call renders, the `end_to_end` conformance vectors
hold on the default path, and the five implementations agree token for token
on them. Chunk k > 0 draws its own stream, so its seed depends only on the
caller's seed and k, not on how many chunks came before. Its audio also
depends on the prefix carried from the chunk before it. The chunks a stream
yields before the caller stops it are the same as the first chunks of a
stream drained fully.

`Engine.warm` renders "Ready." at seed 0 and discards the audio. Every stream
is a pure function of the caller's seed, and no stage draws from a global
generator, so the warm-up does not change a later render. Measured: one
passage at one seed on CPU gives one token hash and one audio hash cold, after
warming on the same voice, and after warming on a different voice.
`Engine.stream` over three chunks, which draws a seed per chunk, gives the
same result. An entry point can take the warm-up or leave it; see
[transports](transports.md) for which ones warm and how to turn it off.

The retry ladder (`_STREAM_RETRY`, streams 8 + n) is below the chunk streams
(16 + k). `postprocess.RETRY_LADDER_HEADROOM` repeats the size of that gap, so
the preset refuses a `retry_max_attempts` that would reach a chunk stream; a
test checks the copy against the originals. The re-split stream derives from
the chunk seed, because chunk streams have no upper bound. It only has to
avoid the streams already drawn from the chunk seed.

### Why the sampler is a component

`Sampler` is a component because it owns the RNG stream. The RNG stream is a
common cause of disagreement between two correct backends:
`torch.multinomial` gives different samples for the same probability vector
and the same generator on x86 and arm64.

The engine does not receive a sampler, unlike the four stage protocols beside
it. It builds its own sampler per window, because the seed ladder that
addresses the RNG belongs to the engine. The sampler is passed to
`TokenGenerator.generate`, so a backend that implements that protocol receives
one and must sample with it.

A sampler is called with three arguments. `logits` are raw, unnormalised
scores over the speech vocabulary. `step` is the index of this decode step,
and the RNG is addressed by it, so the random number drawn at a step does not
depend on how many numbers were drawn before. This lets two backends agree
while computing in different orders. `seen` marks the tokens already emitted,
for the repetition penalty, which applies to every seen token, silence
included. Silence ids are exempt only from the `min_p` filter, so a pause
stays reachable when the filter would drop it; see
`SamplingConfig.silence_token_ids`.

`eos_peak` returns `(step, probability)` for where the model came closest to
stopping. It returns `(-1, 0.0)` when the stop token was never plausible, or
when the sampler was built without one. It is part of the `Sampler` protocol
because the engine reads it from the sampler that ran and passes it to the
detectors. Three detectors read it: `silence_tail` and `terminal_echo` compare
its probability against a threshold, and the desperation band uses its
position. A sampler without it breaks the postprocess layer.

## One window: the token phase and the render phase

`window.generate_window` runs the funnel, encodes, samples, judges the row
with the postprocess detectors, retries a condemned row and trims. It returns
a `GeneratedWindow`: the speech tokens, plus the text, seed, verdict and cap
flags. `window.render_window` decodes the mel, vocodes, stretches and fades
the edges. The two are separate so the stream can render window k while it
generates window k+1. The render is deterministic given the `GeneratedWindow`
and the voice, so running it on another thread does not change the output.

On the long-form path, the funnel runs on the whole text before splitting, so
the splitter budgets the text that will be spoken: `1234` becomes "one
thousand two hundred and thirty-four". The funnel may remove everything (a
footnote marker, an emoji). That is refused after the funnel as well as
before it, on the path both `synthesize` and `stream` take.

The length ceiling is applied during generation, not after: tokens past it
cost time and are certain to be discarded. A condemned window (dropout, or a
suspect verdict) is regenerated from a derived seed, up to
`retry_max_attempts` times. The first attempt that is not condemned ships.
When every attempt is condemned, the one with the fewest true-silence tokens
ships, the earlier one on a tie, so the ladder stays a pure function of the
caller's seed. On the worst voices, 30% of condemned windows exhaust the
ladder, and shipping the last attempt instead gives worse rows than the first.

Two facts are measured on the raw row, before postprocess trims it:
`hit_token_cap` (the row reached its cap, which may be the length ceiling for
a runaway short text) and `hit_window_cap` (the row filled the window). A trim
can cut a filled window down to a few tokens. A flag read from the trimmed
length would call a filled window short, and the one-window refusal and the
re-split would let the overflow through with its tail unspoken.

The overflow refusal (`single_window=True`) runs between the two phases,
before the mel decode and the vocoder pass, which cannot change the answer.
The four ports refuse before their renderer for the same reason.

`speed` is applied after the detectors, because they judge pacing by duration
per token, and stretching first would move every number they compare. At
`speed=1.0`, `time_stretch` returns its input unchanged; the edge fade that
follows writes to a copy.

## Re-splitting a chunk that filled its window

`windows_for_chunk` regenerates a chunk whose window filled as two halves,
split at the middle word, both under the original chunk index, so no later
chunk's seed moves. Only the seeds are preserved: the next chunk is
conditioned on the tail of the second half, so the audio of every later chunk
changes. The window has to be what stopped the chunk, not the length ceiling:
halving a runaway short text gives two halves that each run away to a smaller
ceiling. A half that still overruns is not split again.

## The carry

`carry_from` takes the last `chunking.prefix_tokens` of the previous tokens,
the same slice a chunk join takes, so a request boundary and a chunk boundary
use one mechanism. The whole input is validated, not only the slice used, so
a bad id is refused whatever the text length.

The carry reaches the generator as `prefix`: speech tokens from the preceding
chunk, fed in as context and **not** included in the return value. A chunk
generated independently restarts its pitch contour like a new sentence, and
the restart is audible at every join. Conditioning on the tail of the
previous chunk reduces it (see `prefix_tokens` in `algorithm-config.md`).
`prefix` is a parameter of `TokenGenerator.generate` because adding a
parameter to a `Protocol` breaks every implementation written against it.

Under `fusion_mtp2` the generator consumes tokens two at a time, one KV slot
per pair, and pairs whatever prefix it receives from the prefix's own start. A
tail whose first token is the second half of a pair would fuse the right
tokens with the wrong partners, at every join where the previous chunk ended
on an odd count. `carry_pair_aligned` ends the slice on a pair boundary and
starts it on one, so it hands over `wanted + 1` tokens when `wanted` is odd.

## The stream

`stream.pipelined` runs a producer thread through the token phase and submits
each finished window to a single-thread render pool; the consumer takes the
futures in order. `_PIPELINE_DEPTH` (two) bounds how far generation runs
ahead of the consumer, so an abandoned consumer stops the engine instead of
letting it generate the whole passage into memory. A depth of two lets the
renderer work while the next window generates.

`latency_mode` protects time to first audio when both stages share a device,
where generating window 1 competes with window 0's render for the same GPU.
On a shared device, generation of window 1 waits until window 0's render is
done, which costs one render of overlap. `synthesize` returns nothing until
the whole passage is done, so it turns the protection off and keeps the
overlap. On a split placement the flag has no effect.

Cancellation keeps its token-level latency: the producer polls the caller's
`should_cancel` and the stream's own stop flag inside the decode loop. The
partial chunk is discarded without being rendered, which saves the mel decode
and the vocoder pass for audio that would never be delivered. Inside the
pipeline a cancel is `CancelledError`, raised by whichever phase polled the
flag and carried from the producer thread like any other exception. `stream`
catches it and ends: the chunks already yielded belong to the caller.
`synthesize` lets it propagate instead of joining the chunks that finished,
which would be silent truncation. See `barge-in.md`.

## Teardown and the wedge

When the consumer closes early, the pipeline sets its stop flag, cancels
every render that has not started, drains the queue so the producer can exit,
and joins the producer for `_PRODUCER_JOIN_TIMEOUT` (60 s). The join waits
only for the render in flight, not for the rest of the passage.
`ThreadPoolExecutor.__exit__` drains its own queue, so without the
cancellation sweep the join would wait for `depth x render`, and a slow
renderer would look like a stuck thread. A lock pairs the stop check with the
submit, so a window generated in the instant before teardown cannot enter
the pool after the sweep.

A producer that outlives the join still holds the token generator and the
renderer. The engine is then wedged: every later call raises a `RuntimeError`
that names the thread, because otherwise the next request would compete
silently with a thread that no caller waits on. It is a plain `RuntimeError`,
not a `LoudkitError`, because the transports classify every `LoudkitError` as
the caller's fault, and this fault is the server's; it reaches a client as
`server_fault`. The state is never cleared, because nothing in the process
can prove that the thread has exited; a long-running process replaces the
engine. A transport that cannot reclaim a stream it abandoned marks the engine
the same way, through `engine._wedge`.

## Result

`Result` keeps the tokens and the mel because they localise a disagreement
between two backends to one stage: tokens to the first, mel to the second,
waveform to the third. `chunks` hold exact sample offsets, built from the
parts with integer accumulation, so every chunk's end is the next one's
start; the word times inside are estimates. `provenance` holds what `save()`
writes into the provenance manifest: the fingerprint, the recipe, the voice
name and file digest, the checkpoint digest, the backend and the execution
line. Names are labels that anyone can reuse; the digests identify the bytes.
`speed` is recorded, not inferred, because a stretched reading and a
naturally faster one cannot be told apart from the audio.

`Result.save` writes a 16-bit WAV through `provenance.write_wav`, with the
provenance manifest by default. It writes a `.partial` file beside the target
and renames it onto the target with `os.replace`, so a failed write leaves the
previous file or no file, never a WAV with a partial manifest.

### Chunk times are exact, word times are estimated

`timing.py` reports two kinds of time and keeps them separate: exact chunk
boundaries and estimated word boundaries.

The engine renders each chunk to its own waveform and concatenates them, so
it knows every chunk's sample offset and sample length without estimating
anything. `ChunkTiming` reports those, converted to seconds. Chunk *k*'s `end`
is bit-identical to chunk *k+1*'s `start`: both are the same integer sample
offset divided by the same sample rate, so a highlight driven by them has no
gap and no overlap. `timeline` accumulates offsets in **samples**, not
seconds, and divides by the rate only at the end. Accumulated seconds would
make chunk *k*'s `end` and chunk *k+1*'s `start` two different sums of the
same floats, which can differ in the last bit. A test that compares with a
tolerance does not see that difference; a player that switches on
`time >= start` does.

Word times are estimated. The model emits speech tokens, not an alignment, so
nothing in this pipeline knows where a word begins. `WordTiming` distributes a
chunk's real duration across its words in proportion to their length in
characters. The estimate is wrong in predictable ways: a long word said fast,
a short word held, a pause before a clause. The error grows with the length
of the chunk, because an early error shifts every later boundary, so it suits
sentence-length chunks better than a long paragraph read as one chunk. For a
real alignment, use a forced aligner; this is not one.

`estimate_words` allocates by **character count**, not by token count or by
any acoustic measure, because a word's characters are the only thing known
here. Whitespace is not counted: the gap between two words belongs to
whichever side of the boundary the caller's player is on, and splitting it
would create a third kind of span. Boundaries come from a running character
total, not from adding per-word durations, so the spans cannot drift: the
first `start` is exactly `start`, the last `end` is exactly `end`, and every
interior boundary is shared by the two words that meet at it.

Both kinds are computed *after* any time-stretch and edge fade, on the
waveform the caller receives, so `Result.speed` needs no correction applied to
them.
