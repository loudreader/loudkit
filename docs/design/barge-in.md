# Barge-in: stopping a render that is already speaking

A voice agent is interrupted mid-sentence. The listener starts talking; the
agent has to stop. Three references in the code point here for "the whole
contract", so here it is: what stopping guarantees, what it costs, and the half
of it that is not the engine's to do.

## The contract

`should_cancel` is a callback taking no arguments and returning a bool. It is
accepted by `Engine.stream`, `Engine.synthesize`, `render_bytes` and
`render_stream_chunks`. Return `True` and:

1. **The decode loop stops within one forward pass.** It is polled on every
   decode step, in every decode path: the eager loop, the fused two-token
   loop, and the static-cache loop a captured CUDA graph runs. The token that
   was about to be sampled is discarded.
2. **The partial chunk is not rendered.** Those tokens are speech the listener
   has already stopped wanting, so the mel decode and the vocoder pass are
   skipped rather than run and thrown away.
3. **`stream` ends; `synthesize` raises.** `stream` delivers the chunks that
   finished before the cancel and then stops, without raising: those chunks
   are the partial, and the caller flipped the flag. `synthesize` raises
   `CancelledError`, a `LoudkitError` with code `cancelled`, and never hands
   back the chunks that finished, because half a passage with no way to tell
   it is half is the silent truncation this library refuses everywhere else.

The same contract holds in the other four engines, in their own spelling. Go's
`Synthesize` returns `ErrCancelled` (test with `errors.Is`) and `Stream`
returns `nil` after the chunks it delivered. Rust's `synthesize` returns
`Err(error::CANCELLED)` (test with `error::is_cancelled`) and `stream` returns
`Ok(())`; both read `Options.should_cancel`, and `stream` also takes the
closure as an argument. JS throws `CancelledError` from `synthesize` and ends
the `stream` async generator. Swift throws `LoudKitError.cancelled` from
`synthesize` and returns from `stream`. Inside each engine the cancel travels
as that one signal from the poll that fired; `stream` is the single place
that swallows it.

## What it costs

One forward pass, worst case: the interrupt arms between two polls, so the
budget is the time for the decode loop to come round again. That is what
`tools/bench.py` reports as `cancel_latency_s`, and it measures the worst case
deliberately: the callback flips while a forward pass is in flight, which is
where a real barge-in lands.

The alternative it replaced is the number worth remembering: a callback checked
only at chunk boundaries leaves the interrupt waiting for up to a whole chunk,
which is around ten seconds of speech nobody wants.

## What it does not do

**It does not un-deliver audio.** Chunks already yielded from a stream are the
caller's, and they will play unless the caller drops them. Over SSE they are
events already on the wire. Stopping the engine is half of a barge-in; flushing
the playback buffer is the other half, and in a real client it is usually the
larger one:

```python
def barge_in():
    interrupted.set()      # the engine stops within a decode step
    player.flush()         # everything already queued has to go too
```

**It does not interrupt a kernel already running.** Cancellation is
cooperative: the flag is read where the decode loop reads it, which is where
nearly all of the time goes, but a mel decode, a vocoder pass or a time-stretch
that has entered its backend runs to the end of that call. A cancel lands
within one such step, never within zero.

**It does not make a cancelled render reproducible as a shorter one.** The
tokens that were generated before the stop are discarded, not returned. A
cancelled `synthesize` has no output to compare against anything.

## What the transports do with it

Both HTTP routes watch for a client disconnect on the event loop while the
forward pass runs in a worker thread, and flip the flag the decode loop polls.
`POST /v1/synthesize/stream` and `POST /v1/synthesize` both do; a departed
client no longer holds the engine's single slot to the last token.
`render_bytes` lets the `CancelledError` through, and each door answers it
itself: HTTP with a 499, gRPC by returning nothing, since the client a status
would go to is gone.

gRPC uses the same mechanism through `context.add_callback`, which fires for
both of the ways an RPC ends early: the client cancelling, and the client's
deadline expiring. `_MAX_STREAM_S` sets the same flag on a timer, because a
check read *between* chunks cannot fire while the render it bounds is the thing
that has not returned.

MCP does not: a tool call is one short synthesis and there is no transport
signal to hang a cancel on.
