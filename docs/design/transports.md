# Transports and the synthesis facade

Maintainer notes for `python/loudkit/synthesis.py` and
`python/loudkit/transports/`: what each module decides and why. The user page
is [Server and agents](../guides/04-server-and-agents.md).

## One synthesis path, three transports

`synthesis.render_bytes` and `synthesis.render_stream_chunks` are the two
functions in the package that turn an engine plus a voice profile into
encoded audio. The first returns the whole passage; the second streams it
chunk by chunk, and it is kept beside the first so both deliveries run one
synthesis. HTTP, gRPC and MCP are adapters over those two functions: each
resolves a voice by name and calls one of them. None of them clamps a speed,
resolves a language or slices a continuation prefix. The engine owns each of
those rules, so a transport cannot drift from it.

Each transport's suite asserts that it returns, byte for byte, what
`render_bytes` returns for the same text, voice and seed: in `wav` and `flac`
for HTTP and MCP, in `wav` for gRPC. The OpenAI-compatible route is checked
against the native one, which is checked against the library.

`tests/test_import_graph.py` enforces the layering inside the package:
`synthesis` imports no transport, and the three transports import `synthesis`
and never each other. It records `loudkit.*` edges only. What keeps a web
framework out of `synthesis` is `pyproject.toml`, which puts fastapi in the
`server` extra. The table lists the three shared modules beside the
transports, then the three transports:

| module | holds |
|---|---|
| `transports/limits.py` | the request caps, `EngineSlot` and the bounds on waiting for and holding the engine, the bearer-token rules, the ASGI guard |
| `transports/schemas.py` | the HTTP request models, the reply headers, the stream events, OpenAI's spellings and envelope, the error vocabulary |
| `transports/resolve.py` | `open_release`: a checkpoint reference and a device in, the release's checkpoint and the voice library beside it out |
| `transports/http.py` | the FastAPI app: routes, error mapping, `serve` |
| `transports/grpc.py` | the `Speech` servicer over `proto/loudkit.proto`, `serve` |
| `transports/mcp.py` | the three MCP tools over stdio, `run_stdio` |

`loudkit serve` starts HTTP; `--grpc` and `--mcp` start the other two over
the same engine load. The three transports accept different flags. A flag the
chosen transport cannot act on is refused by name with exit code 2
(`_SERVE_DROPS` in `cli.py`), not parsed and dropped. Otherwise a server could
start without the `--token` it was given, or without the
`--first-chunk-tokens` that changes its fingerprint on the other two
transports, and the operator could not tell.

## The synthesis facade

### Samples to frames, once

`_quantise` converts float samples to int16 by one rule, `floor(x * 32768)`
clipped to the int16 range, and every container receives the resulting
frames. libsndfile rounds differently per format (its WAV writer floors, its
FLAC writer rounds), so handing it floats would make two lossless formats
disagree by one LSB on half the samples. With the conversion here, `wav`,
`pcm16` and `flac` carry identical samples; `ogg`, `mp3` and `opus` are lossy
and do not round-trip. The rule is the one libsndfile's WAV writer applies, so
WAV output is the same as libsndfile would write from floats. Clipping, not
scaling by 32767, keeps the engine's `[-1, 1]` contract without moving every
sample to make room for a value the vocoder does not emit.

`_check_encodable` asks the loaded libsndfile for the codec before the engine
runs. `mp3` and `opus` depend on how that library was built, not on the
request, and a refusal after the render would waste the render.

`_encode` writes every container through `soundfile`, `pcm16` included, so
endianness and frame size are decided in one place. `pcm16` is labelled
`application/octet-stream` with an explicit rate header, not `audio/L16`,
because RFC 2586 defines L16 as big-endian and these frames are
little-endian.

### The caps are defined with the facade

`_MAX_TEXT_LEN` (10 000 characters), `_MAX_PREVIOUS_TOKENS` (4096) and
`_MAX_WAIT_S` (120 s) are defined in `synthesis.py`, so every transport
enforces the same limits. `_check_text` applies the text cap in both render
helpers, so an embedder calling either directly is bounded too. `limits.py`
re-exports the three for the transports.

### `render_bytes` and `render_stream_chunks`

Both pass `language` to the engine as given, `None` included. Both take a
`should_cancel` callback that the engine polls on every decode step, so a
caller that goes away stops using the GPU at the next decode step; a render
stage already running finishes first. `long_form=false` asks for a single
window, which refuses text that does not fit one; it is the wire name for
the engine's `single_window`. Cancelling a stream does not recall chunks
already yielded: over the wire those are sent, and stopping their playback is
the client's job (`barge-in.md`).

`_first_exception` unwraps an anyio task group's exception group to its root
by duck typing, because `except*` needs Python 3.11 and the package supports
3.10.

### The voice library

`VoiceLibrary` resolves a voice by name inside one configured directory, so a
request can name any voice in that directory and cannot read any other file.
Loading a profile is a whole-file read, a SHA-256 and a safetensors parse, so
parsed profiles are cached. The cache is bounded in bytes, not entries,
because profiles differ by an order of magnitude in size. Its bookkeeping is
under a lock because the servers call it from thread pools. The lock is not
held across the file read, so two threads that miss on the same name both
read it; the alternative, serializing every other name behind a miss, costs
more.

## What every transport enforces (`limits.py`)

### The request check

`over_cap(text, previous_tokens)` returns the reason a request is refused, or
`None`: empty text, text over the cap, or a continuation over the cap. gRPC
and MCP call it before resolving a voice. HTTP applies the same caps in
`SpeakRequest`, which pydantic validates before the route handler runs. All
three answer the refusal with the code `invalid_request`
(`tests/test_server_limits.py`).

### The engine slot

The engine is single-flight: it holds mutable decoder state, and a CUDA graph
capture is not reentrant. Each transport serializes synthesis behind one slot
and bounds the wait for it in two ways, because the two bounds fail
differently.

- A depth bound refuses at once past it, so the methods that need no engine
  keep answering while synthesis is saturated. It is `_MAX_QUEUED`, 32, on
  HTTP, where a waiter is a coroutine and the number is chosen. On gRPC and
  MCP, where a waiter holds a thread, it is `_queue_depth_for` of the pool:
  half the pool, so the other half can still answer.
- A time bound (`_MAX_WAIT_S`) frees the callers within the depth when one
  render never returns.

The render itself is not cancelled at that deadline: it runs against an engine
that is not reentrant, and abandoning it would let the next caller in while it
still writes. The deadline frees only the queue. `/health` and `Describe`
report the age of the render that holds the slot.

`EngineSlot` is the HTTP form of this: an `anyio.Semaphore(1)` that both
routes take on the event loop before any worker thread, so a waiter never
holds a worker thread. It is a semaphore, not a lock, because the stream route
takes the slot in the handler and releases it from the response task, and
`max_value=1` makes a double release raise. gRPC keeps a `threading.Lock` per
engine instance (`_lock_for`), keyed by identity, so two servers over one
engine share it and two equal engines do not. MCP holds a lock with the same
two bounds. The HTTP app and a gRPC server over one engine in one process do
not share a slot. Use one engine per transport process.

### Warming the engine

The first render of a process can cost more than later ones: kernel
autotuning, graph capture, and on Metal the pipeline compilation that the OS
then caches. On a Jetson Orin, the vocoder's first call measured 2.49 s
against a 708 ms warm steady state. Warming moves that cost to startup, before
the first request.

How large the cost is depends on the machine. Four renders of one passage in
one process on an M3 Pro: on MPS 2.44 s, then 2.40, 2.17 and 2.37 s, a first
render premium of about 0.1 s inside a noise band of 0.3 s. On CPU 18.8 s,
then 19.4, 21.3 and 24.7 s: the first render is the fastest, and other load on
the machine outweighs any saving from a warm-up. On the same M3 Pro, nine
fresh MPS processes each rendered at full speed from the first call, so the
pipeline compilation, which takes seconds the first time a shape is seen, is
cached by the OS and not by the process. The Apple path behaves the same way,
measured in [apple](../platforms/apple.md).

`build_app`, gRPC's `build_server` and MCP's `build_server` each take
`warm=`. It is off by default, because building a transport is not a decision
to spend seconds on the GPU; an embedder that wires one into their own process
decides when. `serve` and `run_stdio` turn it on, so every transport warms on
the library's first voice before it accepts requests, whether it loaded the
engine or received one. All three call `synthesis.warm_engine`, which picks
the voice, reads the environment and prints the line, so the three
transports behave the same.

`LOUDKIT_NO_WARM` turns warming off everywhere. Any non-empty value turns it
off, `0` and `false` included. An unset or empty variable leaves warming on.

A failed warm-up is logged and is not fatal, a voice that cannot be read
included: a truncated profile at the head of the library costs that one
voice, and the server still starts.

`run_stdio` warms before it begins reading stdin, so the MCP `initialize`
handshake waits for the warm-up render, twenty seconds or more on CPU. A host
whose initialization timeout is shorter than that needs `LOUDKIT_NO_WARM`
set.

The line that names the voice and the seconds goes to stderr, because stdout
carries the MCP protocol.

`loudkit speak` does not warm. A warm-up moves the first-render cost to a time
when nobody waits, and a command that renders once and exits has no such
time. Warming replaces `overhead + N renders` with `overhead + a small render
+ N renders`, so on the measured machines it only adds. Measured on a
two-window passage on the M3 Pro's CPU: 51.6 s and 52.2 s without, 73.6 s
with a warm-up that cost 22.7 s and saved nothing on the render (mel 42.3 s
against 42.4 s). On a three-chunk passage: 79.4 s without, and 24.8 s of
warm-up plus 76.3 s of render with. On MPS the same passage runs 6.7 s
without and 6.7 s with, the 1.2 s warm-up saving about what it cost. "Ready."
is also not a cheap render: the mel decoder works on a fixed window, so it
costs about what a sentence costs.

### The stream lease

A disconnect watcher detects a peer that hung up. A peer that stays connected
and stops reading is different: the write blocks, the response never ends,
and no poll runs. `_MAX_STREAM_S` (600 s) bounds how long one streamed
response may hold the engine, counted from when it takes the slot.

On HTTP, `stream_lease` enforces it by cancelling the response task, not by
setting the cancel flag, because the body generator is blocked inside
Starlette's `send`, where no flag is read. The cancellation unwinds into the
lease's `finally`, the one frame on every exit path. The lease is attached to
the response object, not to the body generator, because Starlette sends
`http.response.start` before it pulls the first item. A socket that fails
there leaves the generator unstarted, so a `finally` inside it would never
run, and the slot would stay taken for the life of the process.

On gRPC the same bound is a `threading.Timer` that sets the cancel flag
`Engine.stream` polls on every decode step. A stream renders on a producer
thread and hands chunks to the gRPC worker through a two-slot queue. The
producer's bounded `put` polls the same flag, so the cap frees the engine
whether the stream is stuck rendering or stuck delivering, once the running
call returns. Only that peer's worker stays in the blocked write.

### Reclaim before release

However a stream ends, the engine's own threads are reclaimed before the slot
is released. Closing the chunk generator runs `Engine.stream`'s teardown,
which cancels the renders that will not be read and joins the engine's
producer thread. Releasing first would let that thread run inside a
non-reentrant engine that the next caller has already entered. A reclaim that
fails marks the engine wedged instead of passing it on, because the stages
cannot be shown to be free. On HTTP the reclaim runs off the event loop,
shielded from the cancellation that usually triggers it. On gRPC it runs on
the producer thread, so reclaim and release run in that order on one thread.

### The HTTP guard

`_Guard` is raw ASGI middleware. The `@app.middleware("http")` form sees the
body only when the route reads it, so it cannot check a body's size before
the body arrives. `_Guard` checks headers and counts body bytes as they
arrive. It decides, in order:

* **Host pinning on the loopback default.** A browser page can resolve any
  hostname to 127.0.0.1 (DNS rebinding) and read answers from a server that
  trusts its bind address. `_host_is_loopback` strips a port the way RFC 3986
  says, so IPv6 literals work. A public bind is authenticated instead. The
  HTTP and gRPC `serve` use the same function to decide whether a bind is
  public, so they classify every loopback address the same way.
* **Bearer auth**, compared with `hmac.compare_digest`, so the comparison time
  does not depend on the length of the matching prefix.
* **A body bound before the read.** Pydantic's `max_length` runs after
  Starlette has buffered the whole body, so it protects the engine but not the
  process. `_MAX_BODY_BYTES` is checked against `Content-Length` first. The
  header's digit count is checked before its value (more than 20 digits is
  refused unparsed), because CPython by default refuses to parse an integer
  longer than 4300 digits. For a chunked request that declares no length, the
  bytes are counted while reading. The bound is derived from every capped
  field at its worst encoding: twelve bytes per astral character under
  `ensure_ascii`, and thirteen per continuation id (eleven for the widest
  `int32`, two for the separator `json.dumps` writes). A request inside the
  published caps is therefore never refused for its size. A request can use
  both caps at once, so both terms are in the bound. Once the bound is passed,
  the app's output is discarded and the guard sends the 413 itself. The only
  thing ASGI offers mid-stream is a synthetic disconnect, which FastAPI would
  answer with a 500.
* **The cross-site refusal on the write path.** Host pinning stops a
  rebinding page from reading an answer. It does not stop a blind write, and
  a render started blind costs as much as any other. A POST under `/v1` whose
  `Sec-Fetch-Site` is not `same-origin` or `none` is refused. So is one whose
  content type is not JSON (`application/json` or a `+json` type), which
  forces a CORS preflight that this server never answers. A client that is not
  a browser sends a JSON content type with no extra step.
* **A token bucket per client address**, only on a public bind: twelve
  requests, then one every two seconds, for the POST routes that cost a
  render. `allow_public` turns it on, because it is the one flag that lifts
  the Host pin. A token does not: on loopback a token is defence in depth that
  an embedder may want, not a statement that strangers can reach the port.
  `GET /v1/voices` and `/health` are never limited; a load balancer polls the
  second. The bucket table has a fixed size and evicts its least recently
  used entry, so a rotating source cannot grow it without bound, and eviction
  can only grant tokens. The key is the peer address ASGI reports, with no
  `X-Forwarded-For` handling. Behind the reverse proxy the user guide
  recommends, every caller has the proxy's address and shares one bucket.
  Rate limit at the proxy, which knows who the caller is.

### Tokens

`_token_fault` refuses a token that is empty or whitespace only (the guard
would then accept the literal header `Bearer ` from anyone), one with a
character an HTTP header cannot carry intact, and one shorter than
`_MIN_TOKEN_CHARS` (16 characters, which is 96 bits when the characters come
from the `secrets.token_urlsafe` alphabet). Length alone does not make a
token random: generate it with `secrets.token_urlsafe`. `_token_fault` returns
a phrase, which `build_app` wraps in a `ValueError` and `serve` in a
`SystemExit`. `build_app` checks the token as well as `serve`, because
`build_app` is exported: an embedder that wires the app into their own ASGI
stack never calls `serve`. gRPC's `build_server` is exported for the same
reason and applies the bind rule the same way. `serve` ignores a token on
loopback. On a public bind it requires one, and if none is given it
generates one and prints it to stderr, because stdout is the service log.

## HTTP (`http.py`, `schemas.py`)

The API is under `/v1`, so a later version can be served beside it.
`/health` is unversioned, because a liveness probe is infrastructure. It
answers 503 in two states, which lets a load balancer take the instance out
of rotation:

- `stuck`: a render has held the slot longer than `_SLOW_RENDER_S` (120 s).
  The answer carries a `Retry-After`, because the render can still finish.
- `wedged`: `Engine.wedged` is set. The answer carries no `Retry-After`: an
  engine whose stages could not be shown idle refuses every synthesis from
  then on, so waiting does not help and only a new engine does.

Every error answer carries a `code` from the catalog in `loudkit.errors`. A
refusal raised through `_refuse` carries the code of the exception that caused
it. Boundary conditions that exist only here (a full queue, an oversized body)
map from the status through `_CODE_BY_STATUS`. An exception from the renderer
that is not a `LoudkitError` is a defect, not a refusal: it is logged and
answered 500 `server_fault`, never echoed, because its message may hold a
path. `ValueError` is the common one and `RuntimeError` is the one a wedged
engine raises, so the handler chain ends with `Exception`, not a specific
type. The handler that adds `code` is registered for Starlette's
`HTTPException`, not FastAPI's subclass: the router raises the base class for
a 404 and a 405, and Starlette dispatches on the raised class's own MRO.
Pydantic puts the rejected input in its error list, and Starlette's
`JSONResponse` renders with `allow_nan=False`, which refuses `nan`, so
`_json_safe` rewrites non-finite values before the list is rendered.

`/v1/synthesize` loads the voice off the event loop, takes the slot, renders
in a worker thread, and meanwhile runs a disconnect watcher on the loop that
sets the cancel flag the decode loop polls. The reply carries the duration,
token count, sample rate, truncation flag and continuation tail as headers,
because the body is the audio. `X-Loudkit-Continuation` is omitted, not sent
empty, when there is nothing to carry.

`/v1/synthesize/stream` decides everything that can be a status code before
it writes a byte of the response: in the route, it validates the continuation
ids, refuses a format that cannot stream, loads the voice and takes the slot.
From then on, `_return_the_slot` releases the slot, attached to the response
through `_LeasedStream`. Each event carries one complete, playable payload in
base64, with the `sample_rate` and the `fingerprint` beside it, as gRPC's
chunks carry them, because an event passed on past the response has only what
is inside it. `wav`, `pcm16` and `flac` stream. `ogg`, `mp3` and `opus` do
not, because an Ogg bitstream's state spans the whole stream and each MP3
chunk carries its own encoder delay. The response also states both values in
headers, `X-Loudkit-Sample-Rate` and `X-Loudkit-Fingerprint`, with the
one-shot route's spelling. The terminal `done` event carries the aggregate
`truncated` and the passage's continuation tail. When synthesis failed after
the 200 status was sent, it also carries `error`, `error_kind` (`bad_request`
or `server_fault`, the status the stream could no longer send) and
`error_code`.

`/v1/audio/speech` answers OpenAI's request shape, so a client pointed at this
server's `/v1` needs only configuration, and the API key it sends is the
bearer token the guard already checks. It builds a `SpeakRequest` and calls
the one-shot route, so its bytes are that route's bytes. Compatibility is
partial, and each deviation is stated:

- `model` is accepted and ignored.
- `response_format` defaults to `wav`, which is lossless and every client can
  decode.
- `mp3` and `opus` are encoded, `opus` as Ogg Opus, which is separate from the
  Vorbis `ogg`.
- `aac`, which libsndfile cannot write, is refused by name.
- A `speed` outside this engine's range is refused, not clamped.
- `stream_format: "sse"` is ignored, so the whole utterance comes back in one
  response.

Errors are rewritten into OpenAI's envelope, because OpenAI clients read
`error.message`.

## gRPC (`grpc.py`)

A second network transport gives a typed schema that a client generator can
read, and backpressure: a slow consumer on `SynthesizeStream` stops the
producer instead of filling a buffer. It is an extra, because a single
caller on loopback gains little from either. The transport has no authentication, so a
non-loopback bind is refused outright, not made configurable. Both `serve`
and the server `build_server` returns refuse it. `build_server` is exported
and returns a server that is not bound yet, so an embedder who starts one
never calls `serve`. The returned server's own `add_insecure_port` applies the
same rule, with the same `_host_is_loopback` behind it. A `unix:` target
passes, because a filesystem socket cannot be reached over a network.

Every refusal the servicer decides carries `loudkit-error-code` in trailing
metadata, the same catalog HTTP sends as `code`. gRPC's status codes are
coarser than the catalog, which is why the metadata exists. gRPC core refuses
a request message over 256 KiB before the servicer runs, as
`RESOURCE_EXHAUSTED` with core's own message and no metadata.

`speed` and `long_form` are `optional` in the schema and read through
`HasField`. Otherwise a proto3 scalar's zero cannot be told apart from an
absent field, and both of these have a default that is not zero: an absent
`speed` is 1.0 and an absent `long_form` is true. With `HasField`, an
explicit `speed: 0.0` is refused here as it is on HTTP (422) and MCP.

`_take_engine` bounds the wait in three ways: the depth bound, `_MAX_WAIT_S`,
and the caller's own deadline. A waiter whose deadline passes while it queues
has an RPC that gRPC has already closed, so the wait is capped at
`time_remaining()`, and a successful acquisition is checked against
`is_active()` before it counts. The wait is sliced, not one
`acquire(timeout=...)`, so a cancelled RPC frees its thread within one poll,
not after the full wait.

`Synthesize` refuses a reply that a default client cannot receive. gRPC
clients default to a 4 MiB receive limit in every language, and the limit
belongs to the client, so the server cannot raise it. The preflight estimates
the reply from the text as the engine will speak it, after the normalization
funnel, because the funnel expands text (a thousand digits become about five
thousand characters of number words). It uses the chunker's conservative
token estimate, the token rate and the requested speed. It treats every
container as uncompressed and reserves `_REPLY_HEADROOM` for the WAV header,
the provenance box and protobuf framing, so some refused passages would have
fitted. The refusal names `SynthesizeStream`, which sends one message per
chunk, so only each chunk has to fit.

Cancelling a call sets one flag on both RPCs. A client cancel and an expired
deadline arrive through `context.add_callback`, the stream cap through a
timer, and all three set the event the engine polls on every decode step.
Cancellation is cooperative: a backend call already executing runs to its
end before the next check (`barge-in.md`).

Each stream chunk's `continuation` is cumulative: the passage's tail as of
that chunk, built from all chunks so far. A client that chains from the last
chunk it received therefore passes a full prefix, even when the closing
sentence produced fewer tokens than the prefix length. Every reply carries its
`sample_rate`, the only place a `pcm16` caller can read it.

## MCP (`mcp.py`)

Three tools: `list_voices`, `synthesize(text, voice, seed, language, speed,
previous_tokens, format)` and `describe`. `synthesize` returns the audio as
base64 with `format`, `media_type`, `duration`, `tokens`, `sample_rate`,
`fingerprint`, `truncated` and `continuation`: the facts the HTTP route puts
in headers. `flac` matters more here than over HTTP, because the reply lands
in a model's context. `truncated` reports a cut-off explicitly. `continuation`
is the tail, not every token id, because a few hundred integers in a tool
result cost context and the engine uses only the tail.

An expected refusal returns a normal tool result: `error` with `error_kind`
`bad_request`, plus `supported` or `available` naming the valid values. Every
refusal carries a `code` from the frozen catalog, the same code the other two
transports give for the same request. A bare
`NotImplementedError` or `ValueError` from the renderer is a defect, and it
goes to the framework's own failure path. Answering it as `bad_request` would
tell the agent to retry a request that was never wrong.

The host may send tool calls concurrently, so `synthesize` serializes behind a
lock with both shared bounds. The lock acquire runs beside the render on a
worker thread, because blocking the event loop for `_MAX_WAIT_S` would stall
the notification that cancels the call. A waiter therefore holds a thread, so
the depth bound is `_queue_depth_for` of the SDK's own thread pool: half of
it, the rule gRPC applies to its pool. Queue admission is counted on the loop
and refused there, before a thread is taken. Past the bound the answer is
`busy` at once, so `list_voices` and `describe` keep answering while synthesis
is saturated.

`synthesize` is a coroutine, and both blocking parts, the lock acquire and the
render, run on a worker thread. A host cancels a tool call with
`notifications/cancelled`, which the SDK applies by cancelling the handler's
scope, and only a coroutine has a scope to cancel. The coroutine does not
wait for the thread (`abandon_on_cancel=True`). It sets the cancel flag as it
unwinds, the render stops at its next decode-step poll, and the worker
releases the lock in its `finally`. A backend call already running is not
interrupted. This is the same `should_cancel` wiring HTTP and gRPC use.

`build_server` accepts an engine that is already loaded, which is how tests
inject a fake engine without loading a checkpoint. With an engine and a voice
directory given, it does not read the checkpoint argument. Otherwise it calls
`open_release`, so a repo id yields the checkpoint inside the snapshot, and
the default voice directory is the snapshot's own `voices/`. The other two
`serve` functions use the same `open_release` call.
