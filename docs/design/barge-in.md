# Barge-in: stopping a render that is already speaking

A listener interrupts a voice agent mid-sentence, and the agent has to stop.
Barge-in has two parts: the engine stops synthesis, and the client drops the
audio it has already queued.

## The contract

`should_cancel` is a callback that takes no arguments and returns a bool.
`Engine.stream`, `Engine.synthesize`, `render_bytes` and
`render_stream_chunks` accept it. When it returns `True`:

1. **The decode loop stops at its next poll.** It is polled on every decode
   step, in every decode path: the eager loop, the fused two-token loop, and
   the static-cache loop a captured CUDA graph runs. The token that was about
   to be sampled is discarded.
2. **The partial chunk is not rendered.** Its tokens are discarded, and the
   mel decode and the vocoder pass for them do not run.
3. **`stream` ends; `synthesize` raises.** `stream` delivers the chunks that
   finished before the cancel, then stops without raising. `synthesize` raises
   `CancelledError`, a `LoudkitError` with code `cancelled`, and does not
   return the chunks that finished. A partial passage returned as a result
   would look the same as a complete one.

The four ports keep the same contract in their own spelling. Go's
`Synthesize` returns `ErrCancelled` (test with `errors.Is`) and `Stream`
returns `nil` after the chunks it delivered. Rust's `synthesize` returns
`Err(error::CANCELLED)` (test with `error::is_cancelled`) and `stream` returns
`Ok(())`; both read `Options.should_cancel`, and `stream` also takes the
closure as an argument. JS throws `CancelledError` from `synthesize` and ends
the `stream` async generator. Swift throws `LoudKitError.cancelled` from
`synthesize` and returns from `stream`. Inside each implementation the
cancel propagates as that one signal from the poll that fired, and `stream`
is the one place that turns it into a normal end.

## What it costs

While the decode loop runs, the cost is at most one forward pass: the
interrupt arrives between two polls, and the loop reads it on the next one.
`tools/bench.py` reports this as `cancel_latency_s`. It arms the callback while
a forward pass is in flight, which is the worst case for the decode loop and
where a real barge-in lands. A render stage that is already running adds its
own time (see below), and `cancel_latency_s` does not measure it.

A callback checked only at chunk boundaries would wait for up to a whole
chunk, which is about ten seconds of speech.

## What it does not do

**It does not recall delivered audio.** Chunks already yielded from a stream
belong to the caller, and they play unless the caller drops them. Over SSE
they are events already on the wire. On an interruption, set the cancel flag
and flush the playback buffer:

```python
def barge_in():
    interrupted.set()      # the engine stops at its next decode step
    player.flush()         # drop the audio already queued
```

**It does not interrupt a kernel already running.** Cancellation is
cooperative. The flag is read in the decode loop and between render stages. A
mel decode, a vocoder pass or a time-stretch that has entered its backend runs
to the end of that call before the next check.

**It does not make a cancelled render reproducible as a shorter one.** The
tokens generated before the stop are discarded, not returned. A cancelled
`synthesize` has no output to compare against anything.

## What the transports do with it

Both HTTP routes, `POST /v1/synthesize` and `POST /v1/synthesize/stream`,
watch for a client disconnect on the event loop while the render runs in a
worker thread, and set the flag the decode loop polls. A client that
disconnects does not hold the engine's single slot until the last token.
`render_bytes` lets the `CancelledError` through, and each transport answers
it: HTTP with a 499, gRPC with an empty reply, because the RPC is already
closed.

gRPC sets the same flag through `context.add_callback`, which fires in both
cases where an RPC ends early: the client cancels, or the client's deadline
expires. `_MAX_STREAM_S` sets the same flag on a timer, because a check
between chunks cannot fire while the render it bounds has not returned.

MCP maps `notifications/cancelled` to the same flag. The SDK cancels the tool
coroutine, `synthesize` sets the flag as the coroutine unwinds, and the worker
thread stops at its next poll and releases the synthesis lock
(`transports.md`).
