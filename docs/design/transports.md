# Transports and the synthesis facade

Maintainer notes for `python/loudkit/synthesis.py` and
`python/loudkit/transports/`: what each module decides and why. Not user
documentation; the user page is
[Server and agents](../guides/04-server-and-agents.md).

## One synthesis path, three doors

`synthesis.render_bytes` is the only place in the package that turns an
engine plus a voice profile into encoded audio. `render_stream_chunks` is its
streaming twin, kept beside it so a reader that streams and a reader that
waits for the whole passage are one synthesis with two deliveries. HTTP, gRPC
and MCP are adapters over those two functions: each resolves a voice by name
and calls one of them. None of them clamps a speed, resolves a language or
slices a continuation prefix; the engine owns each of those rules, and a
transport that re-implemented one would be a second place for it to drift.
Each of the three suites asserts that its transport returns, byte for byte, what
`render_bytes` returns for the same text, voice and seed, in both encodings. The
OpenAI-compatible route is pinned to the native one, which is pinned to the
library.

`tests/test_import_graph.py` holds the intra-package layering: `synthesis`
imports no transport, and the three transports import `synthesis` and never
each other. It records `loudkit.*` edges only, so what keeps a web framework
out of `synthesis` is `pyproject.toml`, which leaves fastapi to the `server`
extra. Three modules sit beside the transports and are not transports:

| module | holds |
|---|---|
| `transports/limits.py` | the request caps, `EngineSlot` and the bounds on waiting for and holding the engine, the bearer-token rules, the ASGI guard |
| `transports/schemas.py` | the HTTP request models, the reply headers, the stream events, OpenAI's spellings and envelope, the error vocabulary |
| `transports/resolve.py` | `open_release`: a checkpoint reference and a device in, the release's checkpoint and the voice library beside it out |
| `transports/http.py` | the FastAPI app: routes, error mapping, `serve` |
| `transports/grpc.py` | the `Speech` servicer over `proto/loudkit.proto`, `serve` |
| `transports/mcp.py` | the three MCP tools over stdio, `run_stdio` |

`loudkit serve` starts HTTP; `--grpc` and `--mcp` start the other two over
the same engine load. The flags are not the same set across the three, so a
flag the chosen transport cannot act on is refused by name with exit 2
(`_SERVE_DROPS` in `cli.py`) rather than parsed and dropped: a server that
comes up without the `--token` it was handed, or without the
`--first-chunk-tokens` that re-fingerprints it on the other two doors, is a
server the operator cannot tell apart from one that honoured them.

## The synthesis facade

### Samples to frames, once

`_quantise` converts float samples to int16 by one rule, `floor(x * 32768)`
clipped to the int16 range, and every container is handed the resulting
frames. libsndfile applies different roundings per format (its WAV writer
floors, its FLAC writer rounds), so handing it floats would make two formats
both documented as lossless disagree by one LSB on half the samples. Doing
the conversion here makes wav, pcm16 and flac carry identical samples by
construction; ogg, mp3 and opus are lossy and do not round-trip. The rule is what
libsndfile's WAV writer already does, so the WAV bytes are unchanged by it.
Clipping rather than scaling by 32767 keeps the engine's `[-1, 1]` contract
without moving every sample to make room for a value the vocoder does not
emit.

`_check_encodable` asks the loaded libsndfile for the codec before the
engine runs. `mp3` and `opus` depend on how that library was built, not on
the request, and a refusal after the render has spent the render.

`_encode` still writes every container through `soundfile`, `pcm16`
included: hand-rolling the frame layout would be a second place for
endianness and frame size to be decided. `pcm16` is labelled
`application/octet-stream` with an explicit rate header rather than
`audio/L16`, because RFC 2586 defines L16 as big-endian and these frames are
little-endian.

### The caps travel with the facade

`_MAX_TEXT_LEN` (10 000 characters), `_MAX_PREVIOUS_TOKENS` (4096) and
`_MAX_WAIT_S` (120 s) are defined in `synthesis.py` because a limit one
transport enforces and another does not is two products wearing one name.
`_check_text` applies the text cap in both render helpers, so an embedder
calling either directly is bounded too. `limits.py` re-exports the three for
the transports.

### `render_bytes` and `render_stream_chunks`

Both hand `language` to the engine as given, `None` included; both take a
`should_cancel` callback the engine polls on every decode step, so a caller
that goes away stops costing GPU time within one forward pass. `long_form`
false asks for a single window, which refuses text that does not fit one; it
is the wire's name for the engine's `single_window`. Cancelling a stream
does not recall chunks already yielded: over the wire those are sent, and
stopping their playback is the client's job (`barge-in.md`).

`_first_exception` unwraps an anyio task group's exception group to its root
by duck typing, because `except*` is 3.11 and the package supports 3.10.

### The voice library

`VoiceLibrary` resolves a voice by name inside one directory, so a request
can name any voice on disk and cannot read any other file. Loading a
profile is a whole-file read, a SHA-256 and a safetensors parse, so parsed
profiles are cached, bounded in bytes rather than entries because profiles
differ by an order of magnitude in size. The cache's bookkeeping is under a
lock because the servers call it from thread pools; the lock is not held
across the file read, so two threads racing on one cold name both read it,
which is cheaper than serialising every other name behind a miss.

## What every transport enforces (`limits.py`)

### The request check

`over_cap(text, previous_tokens)` returns the reason a request is refused at
the door, or `None`: empty text, text over the cap, a continuation over the
cap. gRPC and MCP call it before resolving a voice. HTTP applies the same
caps in `SpeakRequest`, which pydantic checks before a route exists. All
three answer the refusal with the code `invalid_request`
(`tests/test_server_limits.py`).

### The engine slot

The engine is single-flight: it holds mutable decoder state, and a CUDA graph
capture is not reentrant. Each transport serialises synthesis behind one
slot and bounds the wait for it two ways, because the two bounds fail
differently. A depth bound refuses immediately past it, which keeps the
engine-free methods answering while synthesis is saturated: `_MAX_QUEUED`,
32, on HTTP, where a waiter is a coroutine and the number is chosen; and
`_queue_depth_for` of the pool on gRPC and MCP, where a waiter costs a
thread and half the pool is what is left to answer with. A time bound
(`_MAX_WAIT_S`) frees the callers within the depth when one render never
returns. The render itself is not cancelled at that deadline: it runs
against an engine that is not reentrant, and abandoning it would let the
next caller in while it is still writing. What the deadline frees is the
queue; `/health` and `Describe` report the stuck render's age instead.

`EngineSlot` is the HTTP shape of this: an `anyio.Semaphore(1)` taken on
the event loop by both routes before any worker thread, so a waiter never
occupies a worker it cannot advance. A semaphore rather than a lock because
the stream route takes the slot in the handler and gives it back from the
response task, and `max_value=1` keeps a double release loud. gRPC keeps a
`threading.Lock` per engine instance (`_lock_for`), keyed by identity so two
servers over one engine share it and two equal engines do not. MCP holds a
lock with the same two bounds. Running the HTTP app and a gRPC server over
one engine in one process is not single-flight across the pair; one engine
per transport process is the supported shape.

### Warming the engine

The first render of a process can cost more than the ones after it: kernel
autotuning, graph capture, and on Metal the pipeline compilation the OS then
caches. On a Jetson Orin the vocoder's first call measured 2.49 s against a
708 ms warm steady state. That cost belongs to startup, not to whoever
happens to call first.

How big it is depends on the machine, and this laptop is the wrong one to
judge it by. Four renders of one passage in one process, M3 Pro: on MPS
2.44 s then 2.40, 2.17, 2.37, a premium of about 0.1 s inside a noise band of
0.3; on CPU 18.8 s then 19.4, 21.3, 24.7, where the first render is the
fastest and whatever else the laptop is doing is worth more than any warm-up
could save. What this machine does show is the other half: nine fresh MPS
processes each rendered at full speed straight away, so the pipeline
compilation, seconds the first time a shape is seen at all, is cached by the
OS and not by the process. The Apple path behaves the same way, measured in
[apple](../platforms/apple.md).

`build_app`, gRPC's `build_server` and MCP's `build_server` each take
`warm=`. It is off by default, because building a transport is not by itself
a decision to spend seconds on the GPU, and an embedder wiring one into their
own process says when. `serve` and `run_stdio` turn it on, so every door
warms before it opens, on the library's first voice, whether it loaded the
engine or was handed one. All three reach `synthesis.warm_engine`, which is
where the voice is picked, the environment is read and the line is printed,
so the three doors cannot drift apart.

`LOUDKIT_NO_WARM` turns it off everywhere. It is a switch rather than a
boolean: any value at all means "set", `0` and `false` included, and a
deployer who wants the warm-up leaves it out of the environment. Nothing the
warm-up does is fatal, the voice it reads included: a truncated profile at
the head of the library costs that one voice, and the server it was going to
speed up still starts. That is the whole point of it being an optimisation.

It is what an MCP host wants first, too. `run_stdio` warms before it begins
reading stdin, so the `initialize` handshake waits out the render, twenty-odd
seconds of it on a CPU box. A host with an initialisation timeout shorter
than that wants `LOUDKIT_NO_WARM` set.

The line naming the voice and the seconds goes to stderr, where the MCP
transport's protocol is not.

`loudkit speak` does not warm, and the reason is worth keeping. A warm-up
moves the extra cost somewhere nobody is waiting; a command that renders and
exits has no such place, because the whole run is the wait. The arithmetic is
that warming replaces `overhead + N renders` with `overhead + a small render
+ N renders`, so it can only ever add. Measured on a two-window passage on
this laptop's CPU: 51.6 s and 52.2 s without, 73.6 s with a warm-up that cost
22.7 s and took nothing off the render (mel 42.3 s against 42.4 s). Again on
a three-chunk passage: 79.4 s without, 24.8 s of warm-up plus 76.3 s of
render with. On MPS the same passage runs 6.7 s without and 6.7 s with, the
1.2 s warm-up buying back about what it cost, so the best case for warming a
one-shot command is that it changes nothing. Nor is "Ready." the cheap render
it looks like: the mel decoder works on a fixed window, so it costs about what
a sentence costs.

### The stream lease

A disconnect watcher answers about a peer that hung up. A peer that stays
connected and stops reading is a different thing: the write blocks, the
response never ends, and no poll sees it. `_MAX_STREAM_S` (600 s) bounds how
long one streamed response may hold the engine, counted from the moment it
takes the slot.

On HTTP, `stream_lease` enforces it by cancelling the response task, not by
setting the cancel flag: the body generator is parked inside Starlette's
`send`, where no flag is read again. The cancellation unwinds into the
lease's `finally`, which is the one frame on every exit. The lease is wired
to the response object rather than to the body generator, because Starlette
sends `http.response.start` before it pulls the first item, and a socket
that dies there leaves the generator never started; a `finally` inside it
would never run and the slot would be held by nobody for the life of the
process.

On gRPC the same bound is a `threading.Timer` that sets the cancel flag
`Engine.stream` polls on every decode step. A stream renders on a producer
thread and hands chunks across a two-slot queue to the gRPC worker; the
producer's bounded `put` polls the same flag, so the cap frees the engine
whether the stream is stuck rendering or stuck delivering, while only that
peer's worker stays in the blocked write.

### Reclaim before release

Whichever way a stream ends, the engine's own threads are reclaimed before
the slot is released: closing the chunk generator runs `Engine.stream`'s
teardown, which cancels the renders nobody will read and joins the engine's
producer thread. Releasing first would let that thread run inside a
non-reentrant engine the next caller has already entered. A reclaim that
fails marks the engine wedged rather than handing it on, because the stages
cannot be shown to be free. On HTTP the reclaim runs off the loop, shielded
from the cancellation that usually brings it about; on gRPC it runs on the
producer thread, so reclaim-then-release cannot interleave with anything.

### The HTTP guard

`_Guard` is raw ASGI middleware, because the `@app.middleware("http")` form
only sees a body the route below reads on demand, and nowhere in that layer
can bytes be refused before they arrive. It decides, in order:

* **Host pinning on the loopback default.** A browser page can resolve any
  hostname to 127.0.0.1 (DNS rebinding) and read answers from a server that
  trusts its bind. `_host_is_loopback` strips a port the way RFC 3986 says,
  so IPv6 literals work. A public bind is authenticated instead. The same
  function is how the HTTP and gRPC `serve` decide that a bind is public,
  so every loopback address is one address to all of them.
* **Bearer auth**, compared with `hmac.compare_digest` so a plain comparison
  cannot leak the matched prefix through timing.
* **A body bound before the read.** Pydantic's `max_length` runs after
  Starlette has buffered the whole body, so it protects the engine and not
  the process. `_MAX_BODY_BYTES` is checked against `Content-Length` first
  (its digit count before its value, because CPython refuses to parse an
  integer past 4300 digits), and counted while reading for a chunked request
  that declares none. Derived from every capped field at its worst encoding,
  twelve bytes per astral character under `ensure_ascii` and thirteen per
  continuation id (eleven for the widest `int32`, two for the separator
  `json.dumps` writes), so a request inside the published caps is not refused
  for its size. Both caps at once is the multi-call reading the API
  advertises, so both terms are in the bound. Once the bound is passed the app's
  output is swallowed and the 413 is sent by the guard, because the only
  thing ASGI offers mid-stream is a manufactured disconnect, which FastAPI
  would otherwise answer with a 500.
* **The cross-site refusal on the write path.** Host pinning stops a
  rebinding page from reading an answer; it does nothing about a blind
  write, and a render started blind costs what a real one costs. A POST
  under `/v1` from a `Sec-Fetch-Site` other than `same-origin` or `none` is
  refused, and so is one that is not `application/json`, which forces a
  preflight this server never answers. Non-browser clients send the content
  type already.
* **A token bucket per client address**, only on a public bind: twelve
  requests, then one every two seconds, for the POST routes that cost a
  render. `allow_public` is what turns it on, because it is the one flag that
  lifts the Host pin; a token does not, since a token on loopback is defence
  in depth an embedder may want and not a statement that strangers can reach
  the port. `GET /v1/voices` and `/health` are never limited; a load balancer
  polls the second. The bucket table is capped and evicts its least recently
  used entry, so a rotating source cannot mint entries and eviction can only
  grant tokens. The key is the peer address ASGI reports, with no
  `X-Forwarded-For` handling: behind the reverse proxy this guide recommends,
  every caller is one client and shares one bucket. Rate limit at the proxy,
  which is the layer that knows who the caller is.

### Tokens

`_token_fault` refuses a token that is empty or whitespace (the guard would
then accept the literal header `Bearer ` from anyone), one that carries a
character an HTTP header cannot hold intact, and one shorter than
`_MIN_TOKEN_CHARS` (16, about 96 bits of the `token_urlsafe` alphabet). It
returns a phrase, so `build_app` wraps it in a `ValueError` and `serve` in a
`SystemExit`. `build_app` checks it as well as `serve` because `build_app` is
exported: an embedder wiring the app into their own ASGI stack never runs a
line of `serve`. `build_server` on the gRPC side is exported for the same
reason and carries the bind rule the same way. On loopback a token is
ignored, as documented; on a public bind one is required, and generated and
printed to stderr if not supplied, because stdout is the service log.

## HTTP (`http.py`, `schemas.py`)

The API is under `/v1` so a later shape can be served beside it. `/health`
is unversioned: a liveness probe is infrastructure. It answers 503 in two
states, which is what a load balancer needs to take an instance out of
rotation. `stuck`, while a render has held the slot longer than
`_SLOW_RENDER_S` (120 s), with a `Retry-After`: that server will answer
again. `wedged`, once `Engine.wedged` is set, without one: an engine whose
stages could not be shown idle refuses every synthesis from there, so
waiting is not the remedy and only a new engine is.

Every error answer carries `code` from the catalog in `loudkit.errors`. A
refusal raised through `_refuse` carries the code of the exception that
caused it; boundary conditions that exist only here (full queue, oversized
body) map from the status via `_CODE_BY_STATUS`. Any exception from the
renderer that is not a `LoudkitError` is a defect, not a refusal: it is
logged and answered 500 `server_fault`, never echoed, because its message may
hold a path. `ValueError` is the common one and `RuntimeError` is the one a
wedged engine raises, so the ladder ends with `Exception` rather than with a
type. The handler that adds `code` is registered for Starlette's
`HTTPException`, not FastAPI's subclass: the router raises the base class for
a 404 and a 405, and Starlette dispatches on the raised class's own MRO.
Pydantic
embeds the rejected input in its error list, and `json.dumps` refuses `nan`,
so `_json_safe` walks the list before it is rendered.

`/v1/synthesize` loads the voice off the loop, takes the slot, renders in a
worker thread, and runs a disconnect watcher on the loop meanwhile that
flips the cancel flag the decode loop polls. The reply carries the duration,
token count, sample rate, truncation flag and continuation tail as headers,
because the body is the audio. `X-Loudkit-Continuation` is omitted when
there is nothing to carry rather than sent empty.

`/v1/synthesize/stream` decides everything that can be a status code before
a byte of the response is written: the continuation ids are validated, a
non-streamable format is refused, the voice is loaded and the slot is taken,
all in the route. From there the slot is given back by `_return_the_slot`,
wired to the response through `_LeasedStream`. Each event carries one
complete, playable payload in base64, with the `sample_rate` and the
`fingerprint` beside it as gRPC's chunks carry them, because an event handed
on past the response has only what is inside it; `wav`, `pcm16` and `flac`
stream, and ogg, mp3 and opus do not, because an Ogg bitstream's state spans
the whole stream and each MP3 chunk carries its own encoder delay. The
response also states both in headers, `X-Loudkit-Sample-Rate` and
`X-Loudkit-Fingerprint`, the one-shot route's spelling. The terminal
`done` event carries the aggregate `truncated`, the passage's continuation
tail, and, when synthesis failed after the 200 was spent, `error`,
`error_kind` (`bad_request` or `server_fault`, the status code the stream
could not send) and `error_code`.

`/v1/audio/speech` answers OpenAI's shape, so a client pointed at this
server's `/v1` needs configuration and nothing else, and the API key it
sends is the bearer token the guard already checks. It builds a
`SpeakRequest` and calls the one-shot route, so the bytes are that route's
bytes. Deviations are decided in favour of saying so: `model` is accepted
and ignored; `response_format` defaults to `wav`, lossless and decodable by
every client; `mp3` and `opus` are encoded, `opus` as Ogg Opus and
not the Vorbis `ogg` beside it; `aac`, which libsndfile cannot write, is
refused by name; a `speed` outside this engine's range is refused rather than
clamped; `stream_format: "sse"` is ignored, so the whole utterance comes
back in one response. Errors are re-dressed into OpenAI's envelope, because
their clients read `error.message`.

## gRPC (`grpc.py`)

Why a second network transport: a typed schema a client generator can read,
and backpressure, since a slow consumer on `SynthesizeStream` stops the
producer instead of filling a buffer. Neither matters on loopback with one
caller, which is why it is an extra. The transport has no authentication, so
a non-loopback bind is refused outright rather than made configurable, and
refused by the built server as well as by `serve`. `build_server` is exported
and returns a server nobody has bound yet, so an embedder who starts one
themselves never runs a line of `serve`; the server it hands back carries the
rule on its own `add_insecure_port`, which is the same rule and the same
`_host_is_loopback` behind it. A `unix:` target passes: a filesystem socket
has no network to be reached over.

Every refusal the servicer decides carries `loudkit-error-code` in trailing
metadata, the same catalog HTTP carries as `code`; gRPC's status codes are
coarser than the catalog, which is why the metadata exists. A request
message over 256 KiB is refused by gRPC core before a servicer exists, as
`RESOURCE_EXHAUSTED` with core's own message and no metadata.

`speed` and `long_form` are `optional` in the schema and read through
`HasField`. A proto3 scalar's zero is indistinguishable from an absent field
otherwise, and both of these have a default that is not their zero: an
unmentioned `speed` is 1.0 and an unmentioned `long_form` is true. Read by
truthiness, an explicit `speed: 0.0` rendered here while HTTP answered 422 and
MCP refused it by name, which is one door admitting a request the other two
reject.

`_take_engine` bounds the wait three ways: the depth bound, `_MAX_WAIT_S`,
and the caller's own deadline. A waiter whose deadline passes while it
queues has an RPC gRPC has already closed, so the wait is capped at
`time_remaining()` and a successful acquisition is re-checked against
`is_active()` before it counts. The wait is sliced rather than one
`acquire(timeout=...)`, so a cancelled RPC gives its thread back within one
poll instead of after the full wait.

`Synthesize` refuses a reply a default client cannot receive. gRPC clients
cap a message at 4 MiB in every language, and the bound belongs to the peer,
so the server cannot raise it. The preflight estimates the reply from the
text as the engine will speak it, after the normalisation funnel, because
the funnel expands (a thousand digits become about five thousand characters
of number words), using the chunker's own conservative token estimate, the
token rate and the requested speed. It treats every container as
uncompressed and reserves `_REPLY_HEADROOM` for the WAV header, the
provenance box and protobuf framing, so the passages turned away near the
line would mostly have fitted. The refusal names `SynthesizeStream`, which
sends one message per chunk and has no such ceiling.

Cancelling a call is one flag on both RPCs: a client cancel and an expired
deadline arrive on `context.add_callback`, the stream cap on a timer, and
all set the event the engine polls on every decode step. Cancellation is
cooperative: a backend kernel already executing runs to the end of its
call, so a cancel lands within one step, never within zero.

Each stream chunk's `continuation` is cumulative: the passage's tail as of
that chunk, folded from the chunks so far, so chaining from the last chunk
received hands the engine a full prefix even when the closing sentence was
shorter than one. Every reply names its `sample_rate`, which is the only
place a `pcm16` caller can read it from.

## MCP (`mcp.py`)

Three tools: `list_voices`, `synthesize(text, voice, seed, language, speed,
previous_tokens, format)` and `describe`. `synthesize` returns the audio as
base64 with `format`, `media_type`, `duration`, `tokens`, `sample_rate`,
`truncated` and `continuation`: the facts the HTTP route puts in headers.
`flac` matters more here than over HTTP because the reply lands in a
model's context. `truncated` is the cut-off signal an autonomous caller is
least able to notice on its own; `continuation` is the tail rather than
every token id, because a few hundred integers in a tool result is context
the agent pays for and cannot act on.

A refusal is an answer, not a protocol error: `error` with `error_kind`
`bad_request`, plus `supported` or `available` naming what would have
worked, and `code` from the frozen catalog on every one of them, the word
the other two doors give the same request. A
bare `NotImplementedError` or `ValueError` from the renderer is a defect and
escapes to the framework's own failure path; answering it as `bad_request`
would tell the agent to retry a request that was never wrong.

The host may dispatch tool calls concurrently, so `synthesize` serialises
behind a lock with both shared bounds. The acquire sits beside the render on
a worker thread, because blocking the loop for `_MAX_WAIT_S` would stall the
notification that would cancel the call; a waiter therefore costs a thread,
so the depth bound is `_queue_depth_for` of the SDK's own pool, half of it,
the arithmetic gRPC applies to its. It is counted on the loop and refused
there, before a thread is taken: past it the answer is `busy` at once, which
is what keeps `list_voices` and `describe` answering while synthesis is
saturated.

`synthesize` is a coroutine and both blocking halves, the lock acquire and
the render, run on a worker thread. A host cancels a tool call by sending
`notifications/cancelled`, which the SDK applies by cancelling the handler's
scope, and only a coroutine has a scope to cancel. The thread is left behind
rather than waited for (`abandon_on_cancel`), and the cancel flag the decode
loop polls is set as the coroutine unwinds, so the render stops inside a
forward pass and gives the lock back on its own way out. This is the same
`should_cancel` wiring the HTTP and gRPC doors use; the acquire is on the
thread for the same reason, since a loop blocked for `_MAX_WAIT_S` could not
deliver the cancellation notice.

`build_server` accepts an already loaded engine, which is how tests inject a
fake one without a checkpoint load; with an engine and a voice directory
given, the checkpoint argument is not read at all. Otherwise it goes to
`open_release`, so a repo id yields the checkpoint inside the snapshot and
the default voice directory is the snapshot's own `voices/` -- not the same
rule as the other two `serve` functions, the same call.
