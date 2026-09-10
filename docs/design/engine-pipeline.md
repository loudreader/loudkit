# Engine pipeline: seeds, windows, the stream, and the wedge

Maintainer notes for `python/loudkit/__init__.py`, `engine.py`, `contracts.py`,
`window.py`, `stream.py`, `timing.py` and `result.py`. It says why the
sequencing is the way it is and what each piece is for; the runtime docstrings
point here rather than repeat it.

## The front door

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
engine.synthesize("Hello there.", engine.voice("joe"), seed=7).save("out.wav")
```

`Engine.synthesize` takes text of any length; `Engine.stream` is the same
synthesis delivered chunk by chunk. The same text, voice and seed give a
bit-identical waveform on a given build and device, and the same tokens on
every backend that runs the same precision. `docs/design/algorithm-config.md`
says why the algorithm and the execution layers are kept apart.

```python
engine = lk.load("loudreader/loudr-1")      # downloads, caches
engine = lk.load("./loudr-1.safetensors")   # exactly this file
```

A path that exists always wins. Two models exist and the name is the whole
choice: `loudreader/loudr-1` runs everywhere, `loudreader/loudr-1-turbo`
supports torch, ONNX and CoreML; an unavailable backend is refused in one
sentence. `device` is `"cpu"`, `"cuda"`, `"mps"`, `"onnx"`, `"coreml"` or
`None` for the best available, and must agree with `execution.device`.
`execution` names the execution fields to change and leaves the rest to the
checkpoint. `algorithm` overrides the checkpoint's algorithm, and the
fingerprint says so. `revision` is a branch, tag or commit for a repo id; pin
it for anything reproducible.

`synthesize` speaks text that fits one window in one, and splits longer text
at sentence boundaries, each chunk conditioned on the tail of the one before,
then joins them. The tokens are the ones `stream` yields for the same seed.
Its arguments are the entry points to the rest of this page. `seed`: same seed
and same build give a bit-identical waveform. `language`: for the text
frontend, and `None` takes the voice's, then `"en"`. `speed`: playback speed
in `[0.5, 2.0]`, pitch preserved, and `1.0` is exact. `previous_tokens`:
`Result.tokens` of the call this one continues, of which only the tail is
used. `single_window`: refuse text that does not fit one window, with
`WindowOverflowError`, instead of splitting. `should_cancel`: polled on every
decode step, raising `CancelledError` when it fires.

```python
mine = lk.enroll("me.wav", "loudreader/loudr-1", name="mine")
```

A path is read at its native rate; samples are taken at `sample_rate`. The
enroller resamples with the one law every port shares. It is built per call,
so enrol in bulk with `loudkit.backends.torch_backend.build_torch_enroller`.
`enroll` raises `FileNotFoundError` when the release ships no voice encoder,
because then it cannot clone. Consent is yours to obtain: see
`RESPONSIBLE_USE.md`.

## Four components, one line of data

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
vocoder, and does almost nothing else: the decisions live in
`AlgorithmConfig` and the speed in the backends. What is left is sequencing,
seeding, and refusing to run a component whose algorithm disagrees with the
engine's. The vocoder is checked too: it looks like a pure renderer, but it
reads the window config for padding geometry, and a vocoder framing the tail
differently produces a different reading. A component with no `config` is
refused rather than skipped, because an unchecked component can compute
anything while every reported fingerprint agrees.

The engine is frozen so nothing can swap a component after that check. The
one mutable fact, `_wedged`, is written through `object.__setattr__` and is
about the engine's fate, not about what it computes.

Every protocol takes an `AlgorithmConfig` and is forbidden from carrying
algorithm state of its own. A backend supplies implementations; it does not
supply behaviour. They are protocols rather than base classes because an
implementation may be a torch module, an ONNX session or a CoreML package,
and none of those want a shared ancestor. What they share is a shape of call,
which is what a protocol says and a base class only implies.

**The contract that matters most** is that these boundaries are *value*
boundaries. Tokens are integers, a mel is an array, a waveform is an array.
No component hands another a live model object, a device handle, or a cache.
That is what makes the device split below possible, and what makes it
possible to compare two backends stage by stage when they disagree.

That comparison has a protocol member of its own. Every implementation is
required to provide `teacher_forced_logits`, returning
`(len(forced) + 1, vocab)`, because it is the only comparison between
backends that is not confounded by chaos: once two free-running generations
differ by one token they are reading different histories, and every later
logit is incomparable. Teacher forcing holds the context identical and asks
only what the arithmetic did.

The sample rate and the token rate are manifest fields, read by
`manifest.algorithm_from` and validated on `AlgorithmConfig`. The geometry
under them is not: one speech token is `TOKEN_MEL_RATIO` mel frames and one
mel frame is `UPSAMPLE_PER_FRAME` samples, in every backend. So a checkpoint
can declare a rate the renderer does not produce, and a 12.5 Hz tokenizer is
the case that shows why it must not: every duration the engine states comes
from the declared rate, every sample comes from the geometry, and the
255-token window would be announced as 20.4 s of speech and rendered as
10.2 s. The engine requires the two to describe one thing and refuses to
build when they do not. The declared rate also feeds the fingerprint, so a
checkpoint that changes it is a different algorithm before it is a different
duration.

## Stages may live on different devices

The token generator is autoregressive: a few hundred tiny dispatches per
token at batch one. The renderer is one large parallel pass. On Apple
silicon the generator is faster on the CPU and the renderer on the GPU, and
with the streaming pipeline the split's value is parallelism: window k
renders on the GPU while window k+1 generates on the CPU. The components
pass arrays, not device tensors, which is what makes the split possible.

## Seeds are derived, not shared

Each stage draws its own sub-stream of the caller's seed through
`window._derive(seed, stream)`, a 64-bit mix. A change in how many numbers
the sampler consumes cannot shift the flow prior, so an optimisation in one
stage cannot silently alter another stage's output. The streams:

| stream | who draws it |
|---|---|
| 1 | the flow prior (mel decoder) |
| 2 | the vocoder excitation |
| 8 + n | retry attempt n of a condemned window |
| 16 + k | chunk k > 0 of a long passage |
| 4096 | the second half of a re-split chunk, derived from the chunk's own seed |

Chunk 0 draws the caller's seed itself rather than `derive(seed, 16)`. That
is what lets one `synthesize` cover every length: a text that fits one
window renders what the four ports' single-window call renders, the
`end_to_end` conformance vectors hold on the default path, and five
implementations agree token for token. Chunk k > 0 draws its own stream so a
chunk's audio never depends on how many chunks came before it, which is also
what makes a stream a caller stops early identical to one drained fully.

This is also what makes `Engine.warm` free of consequence. It renders
"Ready." at seed 0 and throws the audio away, and because every stream is a
pure function of the caller's seed, and no stage draws from a global
generator, that render cannot move a later one by a sample. Checked as well
as argued: one passage at one seed on CPU, cold, after warming on the same
voice, and after warming on a different one, gives one token hash and one
audio hash, and so does `Engine.stream` over three chunks, which draws a seed
per chunk. Warming is therefore an optimisation an entry point may take or
leave: see [transports](transports.md) for who takes it and how to turn it
off.

The retry ladder (`_STREAM_RETRY`) sits below the chunk streams, and
`postprocess.RETRY_LADDER_HEADROOM` restates the gap so the preset refuses a
`retry_max_attempts` that would collide with a chunk stream; a test pins the
copy to the originals. The re-split stream derives off the chunk seed because
chunk streams run upwards with no ceiling; the only values to stay clear of
are the ones already drawn from that seed.

### The sampler is a component because the RNG stream is

`Sampler` is kept a component rather than a function because it owns the RNG
stream, and the RNG stream is the single thing most likely to make two
correct backends disagree: `torch.multinomial` gives different samples for
the same probability vector and the same generator on x86 and arm64.

**Where it is injected.** Not into `Engine`, unlike the four stage protocols
beside it: the engine builds its own sampler per window, because the seed
ladder that addresses the RNG is the engine's. It is injected into
`TokenGenerator.generate`, so a backend that implements that protocol
receives one and must use it rather than sample its own way. That is the
seam, and this is the type crossing it.

What a sampler is called with says the rest. `logits` are raw scores over the
speech vocabulary, unnormalised. `step` is the index of this decode step, and
the RNG is addressed by it, so the result does not depend on how many tokens
were drawn before, which is what lets two backends agree while computing in
different orders. `seen` is which tokens have already been emitted, for the
repetition penalty, which applies to every seen token, silence included.
What silence is exempt from is the `min_p` floor, and nothing else, so a
pause stays reachable even when the filter would drop it; see
`SamplingConfig.silence_token_ids`.

`eos_peak` answers `(-1, 0.0)` when the stop token was never plausible, or
when this sampler was built without one. It is declared in the contract
because the engine reads it off whatever sampler ran and hands it to the
detectors, two of which compare it against a threshold, so it is an *audible*
value: a sampler that does not carry one breaks the postprocess layer rather
than merely losing a diagnostic. It is a requirement of the contract that the
contract did not state.

## One window: the token phase and the render phase

`window.generate_window` runs the funnel, encodes, samples, judges the row
with the postprocess detectors, retries a condemned row, and trims.
`window.render_window` decodes the mel, vocodes, and stretches. They are
split so the stream can render window k while generating window k+1; the two
phases share nothing but the tokens that cross between them, which is why the
split cannot change a byte of output.

The funnel runs on the whole text before splitting on the long-form path, so
the splitter budgets the text that will be spoken (Polish respelling expands
"download" to "daunlod"). It may remove everything (a footnote marker, an
emoji), and that is refused after the funnel as well as before it, at the one
place both `synthesize` and `stream` pass.

The length ceiling is applied during generation, not after: tokens past it
cost real time and are certain to be discarded. A condemned window (dropout,
or a suspect verdict) is regenerated from a derived seed up to
`retry_max_attempts` times; the attempt that ships is the one with the
fewest true-silence tokens, the earlier one on a tie, so the ladder stays a
pure function of the caller's seed. On the worst voices 30% of condemned
fires exhaust the ladder, and keeping the last attempt shipped rows worse
than the first.

Two facts are measured on the raw row before postprocess touches it:
`hit_token_cap` (the cap, which may be the length ceiling on a runaway short
text) and `hit_window_cap` (the window filled). A trim can cut a filled
window down to a handful of tokens; reading the trimmed length made a filled
window look short, so the one-window refusal and the re-split both let the
overflow through with the tail unspoken.

The overflow refusal (`single_window=True`) sits between the two phases:
the mel decode and the vocoder pass are the expensive half and nothing about
them changes the answer. The four ports refuse before their renderer for the
same reason.

`speed` is applied last, after the detectors, because they judge pacing by
duration per token and stretching first would move every number they
compare. `speed=1.0` returns the vocoder's array itself.

## Re-splitting a chunk that filled its window

`windows_for_chunk` regenerates a chunk whose window filled as two halves
split at the middle word, both under the original chunk index so no later
chunk's seed moves. It preserves the seeds and nothing more: the next chunk
is conditioned on the tail of the second half, so the audio of every later
chunk moves. Repairing a passage changes that passage; what the index buys is
that the change is confined to it. The window has to be what stopped it, not
the length ceiling: halving a runaway short text gives two runaways with
smaller ceilings each. A half that still overruns is not split again.

## The carry

`carry_from` takes the last `chunking.prefix_tokens` of the previous tokens,
the same slice a chunk join takes, so a request boundary and a chunk boundary
are one join with one mechanism. The whole input is validated rather than
only the slice used, so a bad id is refused whatever the text length.

The carry reaches the generator as `prefix`: speech tokens from the preceding
chunk, fed in as context and **not** included in the return value. It is why
long-form reading does not stutter. Generated independently, each chunk
restarts its pitch contour like a fresh sentence, and the restart is audible
at every join; conditioning on the tail of the previous chunk removes it. It
is in the protocol from the first release on purpose: adding a parameter to a
`Protocol` after other people have written implementations against it breaks
all of them, and this is the one extension already known to be coming.

Under `fusion_mtp2` the generator consumed tokens two at a time, one KV slot
per pair, and re-pairs whatever prefix it is handed from the prefix's own
start. A tail whose first token was the second half of a pair would fuse the
right tokens with the wrong partners, on every join where the previous chunk
ended on an odd count. `carry_pair_aligned` ends the slice on a pair boundary
and starts it on one, handing over `wanted + 1` tokens when `wanted` is odd.

## The stream

`stream.pipelined` runs a producer thread through the token chain and hands
each finished window to a single-thread render pool; the generator drains the
futures in order. `_PIPELINE_DEPTH` (two) bounds how far generation runs
ahead of the consumer, so an abandoned consumer stops the engine instead of
letting it speak the whole book into memory. Two is enough to keep the
renderer busy while the next window generates.

`latency_mode` protects time to first audio when both stages share a device:
generating window 1 contends with window 0's render for the same GPU. When
the stages share a device, generation of window 1 waits until window 0's
render is out the door, at the cost of one render's worth of overlap.
`synthesize` drains the whole passage before anyone hears a byte, so it
turns the protection off and keeps the overlap. On split placements the
stages do not contend and the flag changes nothing.

Cancellation keeps its token-level latency: the producer polls the caller's
`should_cancel` and the generator's own close inside the decode loop. The
partial chunk is discarded without being rendered, because the mel decode
and vocoder pass would be producing audio no one will hear, and on a slow
device that render is the larger half of the barge-in latency. Inside the
pipeline a cancel is `CancelledError`, raised by whichever phase polled the
flag and ferried from the producer thread like any other exception. `stream`
catches it and ends: the chunks already yielded are the caller's. `synthesize`
lets it out rather than joining the chunks that finished, which would be
silent truncation. See `barge-in.md`.

## Teardown and the wedge

When the consumer closes early, the pipeline sets its stop flag, cancels
every render that has not started, drains the queue so the producer can
exit, and joins the producer for `_PRODUCER_JOIN_TIMEOUT` (60 s). The join is
for the render in flight, not the passage; `ThreadPoolExecutor.__exit__`
drains its own queue, so without the cancellation sweep the join waited for
`depth x render` and a merely slow renderer read as a stuck thread. The lock
around the submit pairs the stop check with the submit, so a window generated
in the instant before teardown cannot slip into the pool after the sweep.

A producer that outlives the join still holds the token generator and the
renderer. The engine is then wedged: every later call raises a
`RuntimeError` naming the thread, because the alternative is the next request
contending silently with a thread nobody is waiting on. It is a plain
`RuntimeError` and not a `LoudkitError` because the boundaries classify every
`LoudkitError` as the caller's fault and this is the server's; it reaches a
client as `server_fault`. Never cleared: nothing on this side can prove the
thread has left, so a long-running process replaces the engine. A transport
that cannot reclaim a stream it abandoned reaches the same verdict by
`engine._wedge`.

## Result

`Result` keeps the tokens and the mel because they localise a disagreement
between two backends to one stage: tokens to the first, mel to the second,
waveform to the third. `chunks` are exact sample offsets rebuilt from the
parts with integer accumulation, so every chunk's end is the next one's
start; the word times inside are estimates. `provenance` holds
what `save()` writes into the provenance manifest: the fingerprint, the recipe, the
voice name and file digest, the checkpoint digest, the backend and the
execution line. A name is a label anyone can reuse; the digests name the
bytes. `speed` is recorded rather than inferred because a stretched reading
and a naturally faster one are the same numbers afterwards.

`Result.save` writes a 16-bit WAV through `provenance.write_wav`, with the
manifest by default. Everything is written beside the target and renamed onto
it, so a render that dies halfway leaves the previous file or none, never a
WAV with half a JUMBF box.

### Chunk times are exact, word times are estimated

A reading app highlights the sentence it is speaking. That needs two
different kinds of answer, and `timing.py` is careful to keep them apart,
because conflating them is how a feature like this becomes a lie.

**Chunk times are exact.** The engine renders each chunk to its own waveform
and concatenates them, so it knows every chunk's sample offset and sample
length without estimating anything. `ChunkTiming` reports those, converted to
seconds. Chunk *k*'s `end` is bit-identical to chunk *k+1*'s `start`: both
are the same integer sample offset divided by the same sample rate, so a
highlight driven by them can neither gap nor overlap. That holds because
`timeline` accumulates offsets in **samples**, not seconds, and divides by
the rate once at the end. Accumulating seconds instead would make chunk *k*'s
`end` and chunk *k+1*'s `start` two different sums of the same floats,
differing in the last bit, a gap or an overlap of a few nanoseconds,
invisible in a test that compares with a tolerance and visible as a flicker
in a highlight that switches on `time >= start`.

**Word times are estimated.** The model emits speech tokens, not an
alignment; nothing in this pipeline knows where a word begins. `WordTiming`
distributes a chunk's real duration across its words in proportion to how
long each word is in characters, and that is all it is. It is right often
enough to be useful for a highlight at sentence scale and wrong in the ways
you would expect: a long word said fast, a short word held, a pause before a
clause. The error grows with the length of the chunk, because a single bad
guess early shifts everything after it, one sentence is usually fine, a long
paragraph read as one chunk is not. If you need real alignment, you need a
forced aligner; this is not one, and pretending otherwise would be worse than
the estimate.

`estimate_words` allocates by **character count**, not by token count or by
any acoustic measure: a word's characters are the only thing known here, and
they correlate with duration well enough at sentence scale to drive a
highlight. Whitespace itself is not charged for, the gap between two words
belongs to whichever side of the boundary the caller's player is on, and
splitting it would only invent a third kind of span. Boundaries are computed
from a running character total rather than by adding per-word durations, so
the spans cannot drift: the first `start` is exactly `start`, the last `end`
is exactly `end`, and every interior boundary is shared by the two words that
meet at it.

Both are computed *after* any time-stretch, on the waveform the caller
actually receives, so `Result.speed` needs no correction applied to them.
