# 4. Server, streaming API and MCP

The model is 747 MB and takes a few seconds to load, so for anything interactive
you load it once and answer requests. loudkit ships four ways to reach one warm
engine:

- an **HTTP server** (`loudkit serve`) with two API surfaces on the same
  process: loudkit's own REST routes, and OpenAI's
  `POST /v1/audio/speech`;
- a **gRPC service**
  ([`proto/loudkit.proto`](../../proto/loudkit.proto), `loudkit grpc`);
- an **MCP server** for agents (`loudkit mcp`);
- a **Speech Dispatcher module**
  ([`integrations/speech-dispatcher/`](../../integrations/speech-dispatcher/)),
  which forwards to a running `loudkit serve` and makes loudkit a voice for
  every Linux application that speaks -- Orca, Firefox, `spd-say`.

HTTP, gRPC and Speech Dispatcher are supported. MCP is a preview: `loudkit mcp`
runs but is not one of the eight commands `loudkit --help` lists.
None of them holds **a synthesis path of its own**. Each builds an `Engine` and
calls it, so a request cannot reach code the library tests do not cover.

```bash
pip install "loudkit[server]"    # REST, and the OpenAI-compatible route with it
pip install "loudkit[mcp]"       # MCP (pulls the server's engine too)
pip install "loudkit[grpc]"      # gRPC
```

## Scope

**It is a working example, not a production deployment.** It keeps a model warm
on your own machine for a script, an editor, or an agent. Read both lists below
before you put it anywhere else.

What it has:

* **Binds `127.0.0.1` by default.** `--host` changes that, and changing it is a
  decision.
* **A request body limit.** 10 000 characters of text, plus a byte ceiling
  computed from it, enforced before the body is buffered.
* **A bounded queue.** The engine is single-flight, so requests serialise. At
  most 32 may wait; past that the answer is `503`.
* **Errors that distinguish your mistake from ours.** `4xx` for the request,
  `5xx` for a defect, and `error_kind` on the streaming route, where a status
  code cannot be sent late.

What it does not have, at all, on the default loopback bind:

* **No authentication on loopback.** Any local process can synthesize. Your OS
  account is the boundary. A non-loopback bind refuses to start without
  `--allow-public` and a bearer token.
* **No per-caller rate limiting or quotas.** The queue bound protects the
  process, not fairness between callers. One client can fill it.
* **No TLS.** Text and audio cross the wire in the clear.
* **No multi-tenancy, no isolation, no accounting, no abuse controls.** One
  engine, one voice library, no notion of who is asking.

Do not expose this server to a network you do not control. Anyone who reaches the
port can spend your GPU, read every voice you loaded, and see every request in
transit. A public endpoint with no auth is also a compute faucet. Put it behind
something that authenticates and terminates TLS, or keep it on loopback.

## The REST server

```bash
loudkit serve --checkpoint loudr-1.safetensors \
    --voices voices --port 8765
```

Binds to localhost. The voice library is a directory, and a request names a voice
**by name**, not by path. Anyone who reaches the port can speak in any voice on
disk, but cannot read arbitrary files.

The API is under `/v1`: `/v1/voices`, `/v1/synthesize`, `/v1/synthesize/stream`.
A later change of shape can then be served beside this one. `GET /health` is
**not** versioned, because a load balancer should not be reconfigured when the
synthesis payload gains a field.

Three bounds are enforced rather than described:

* A **non-loopback bind needs `--allow-public` and a bearer token.** Supply one
  with `--token`, or let the server generate and print it. Every request then
  carries `Authorization: Bearer <token>`. There is no "trusted network" mode: an
  authless server on a network lets anyone who reaches the port speak in every
  voice on the machine.

  **This server speaks plain HTTP.** It takes no `--ssl-certfile`, so on a public
  bind the token and the audio both cross the network in clear. Anyone who can
  see the traffic reads the credential once and uses it thereafter. A bearer
  token over cleartext looks like security and is not. Terminate TLS in front of
  it, with Caddy, nginx, a cloud load balancer or an SSH tunnel, and keep this
  process on loopback behind that. Built-in TLS would bring its own certificate
  handling, renewal and cipher configuration.
* **Request bodies are refused before they are read.** The 10 000 character text
  cap is a pydantic check, which runs only after the whole body is buffered, so
  it protects the engine and not the process. The byte bound is derived from the
  character cap at its worst encoding, twelve bytes per character: `json.dumps`
  defaults to `ensure_ascii=True`, and an astral character becomes two `\uXXXX`
  escapes. No request that passes the documented cap can be refused for its size.
* **At most 32 requests queue for the engine**, which is single-flight. Past that
  the answer is `503` with `Retry-After`. An unbounded queue turns a slow engine
  into unbounded memory, with every client still holding a connection.

### One-shot synthesis

```bash
curl -X POST localhost:8765/v1/synthesize -H 'Content-Type: application/json' \
  -d '{"text":"Hello.","voice":"joe","seed":7}' -o out.wav
```

The headers describe the audio in the body:

```
X-Loudkit-Duration      1.96
X-Loudkit-Tokens        49
X-Loudkit-Sample-Rate   24000
X-Loudkit-Fingerprint   79f71f5821477353
X-Loudkit-Truncated     false
X-Loudkit-Continuation  312,4088,77,1901,55,640
```

Same text, voice and seed → same bytes, every time, over HTTP too.

**Check `X-Loudkit-Truncated`.** `true` means generation stopped at the token cap
instead of at a stop token, so the audio is cut off mid-sentence. The reply is
still a 200 and the WAV is still real, because truncation is not a transport
error. A client that ignores the header reads a severed utterance as a finished
one.

**`speed` is playback speed**, `0.5` to `2.0`, pitch preserved. Outside that
range the request is refused rather than clamped. `1.0` is the default and an
exact bypass: the same bytes this route has always returned. See
[speed.md](../reference/speed.md).

**`format` picks the encoding**, one of:

| `format` | `Content-Type` | notes |
|---|---|---|
| `wav` (default) | `audio/wav` | 16-bit PCM. Byte-identical to what this route has always returned. |
| `pcm16` | `application/octet-stream` | Header-less 16-bit frames, **little-endian**, at `X-Loudkit-Sample-Rate`. For feeding a device or a socket directly. |
| `flac` | `audio/flac` | Lossless, about a quarter the size of the WAV. |
| `ogg` | `audio/ogg` | Vorbis. Lossy: same frame count, not the same numbers. |

Anything else is a `422` naming the four. There is no mp3 or opus: both need an
encoder this project does not ship everywhere, and a format that works on one
machine and fails on another is worse than one that was never offered.

`pcm16` is not labelled `audio/L16;rate=24000`. RFC 2586 defines L16 as
**big**-endian and these frames are little-endian, so the label would be a lie a
conforming client would act on, and a header-less payload cannot be inspected for
byte order.

**`previous_tokens` continues a previous request.** Send back the
`X-Loudkit-Continuation` value from the reply before this one. The first chunk is
then conditioned on that tail, exactly as chunks inside one passage are, so a
chapter read paragraph by paragraph does not restart its pitch contour at every
request. It travels as a header rather than a body field because the body is a
WAV. Longer histories are accepted up to 4096 ids, and only the tail is used.
Past that, `422`. Full contract in
[02-streaming-and-long-form.md](02-streaming-and-long-form.md#carrying-that-join-across-two-calls).

**Omit `language` and the voice decides.** Left out, the request is read in the
language the voice was enrolled in, falling back to `en` for a profile that
carries none. The chain is
[the same everywhere](../reference/preprocess.md#which-language-this-layer-runs-as), and
the server does not have a copy of it. Name a `language` only for cross-lingual
synthesis, such as an English voice reading Polish text.

`400` means the request asked for something this build cannot do, most often a
`language` off the twelve-id roster the text layer is written for. The body names
both the refused id and the twelve, so a client can retry into one that works.
The same refusal reaches the MCP tool and the CLI as `unsupported: ...`.

### Streaming synthesis (Server-Sent Events)

The same synthesis, delivered chunk by chunk, so a client can start playing the
first sentence while the rest is still being rendered:

```bash
curl -N -X POST localhost:8765/v1/synthesize/stream -H 'Content-Type: application/json' \
  -d '{"text":"A longer passage with several sentences.","voice":"joe"}'
```

Each event is a JSON object with the chunk's audio in base64 plus its
`media_type`, duration, token count and `truncated`. The final event is
`{"done": true, ...}` and carries the aggregate `truncated` across every chunk,
so a client that reads only the terminal event still learns that something was
cut off. Streaming is delivery, not a second synthesis: it is the engine's
`stream()` under the hood.

`speed` and `previous_tokens` work here too. The `done` event carries
`continuation`, the tail of the whole passage rather than the tail of each piece
of it, which is what a chaining client wants.

**`format` on this route is `wav`, `pcm16` or `flac`.** Every event must be
complete and playable on its own. Ogg is a container whose seek table and
stream serial number belong to one continuous stream, not to one payload per
chunk, so asking for it is a `422` naming the three that work. Raw `pcm16`
frames concatenate with `+`, which is why they stream.

**Read until `done`, and check it for `error`.** Once the first chunk is out, the
200 is spent: a synthesis that fails halfway through cannot become a status code.
It arrives as `{"done": true, "error": "...", "error_kind": "..."}` instead. A
stream that stops looks exactly like a passage that finished, so a client that
treats "the connection closed" as success reads a truncated passage as a complete
one. Errors that land before the response, such as an unknown voice or a full
queue, are still ordinary status codes (404, 503).

`error_kind` is the status code the stream could not send:

| value | meaning | retry? |
|---|---|---|
| `bad_request` | something about this call: a language off the roster, a chunk over the window. What `/v1/synthesize` would answer `400` or `422` for. | yes, with a different request |
| `server_fault` | a defect in this build. What `/v1/synthesize` would answer `500` for. | not with the same build |

Without it, an agent cannot tell the two apart and retries the request that was
never the problem.

Closing the connection stops the synthesis, not just the sending. The server
polls for the disconnect while the forward pass runs and cancels the decode loop
within one step. That is what makes barge-in possible for a voice agent:
interrupt the speaker and the GPU is free within one decode step, not at the
end of the current ~10 s chunk. A kernel already running is not interrupted.

## The MCP server (preview)

`loudkit mcp` is registered and runnable, and it is not part of the advertised
surface. Any MCP-aware agent (Claude Code, Cursor, Cline, and so on) can speak
in a cloned voice:

```bash
loudkit mcp --checkpoint loudr-1.safetensors --voices voices
```

Three tools ship: `list_voices`,
`synthesize(text, voice, seed, language, speed, previous_tokens, format)` and
`describe()`. Everything after `voice` is optional, and an omitted `language`
means the voice's own.

`synthesize` returns the audio in base64, WAV by default, with `format`,
`media_type`, `duration`, `tokens`, `sample_rate`, `fingerprint`, `truncated` and
`continuation`. Use `format: "flac"` for the same samples at about a quarter the
size, which matters more here than over HTTP because the reply lands in a model's
context.

`truncated` is the same cut-off signal the HTTP header carries, and the one an
autonomous caller is least able to notice on its own. `continuation` is the tail
to pass back as `previous_tokens` on the next call, so an agent reading a long
text in pieces does not restart its prosody at every one. It is the tail rather
than every token id, because a few hundred integers in a tool result is context
the agent pays for and cannot act on. In a client:

```
# any MCP client
synthesize: text="Deploy complete." voice="joe" seed=7
```

`describe()` returns the resolved algorithm and execution config, the line every
run should answer about itself. If a synthesis surprises you, ask the engine
which mode was active before anything else.

## The OpenAI-compatible route

`POST /v1/audio/speech` answers OpenAI's speech API. A great deal of software
already speaks that shape, so supporting it costs no adapter, no plugin and no
code on either side. Point a client at this server's `/v1` as its base URL and it
stops caring which engine is behind it.

```bash
curl -s http://127.0.0.1:8765/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"Deploy complete.","voice":"joe","response_format":"wav"}' \
  -o out.wav
```

Their own client, unmodified:

```python
from openai import OpenAI

speech = OpenAI(api_key="…", base_url="http://127.0.0.1:8765/v1")
speech.audio.speech.create(
    model="tts-1", voice="joe", input="Deploy complete.", response_format="wav"
).write_to_file("out.wav")
```

The `api_key` is this server's bearer token when one is set, and is ignored when
one is not. The guard that checks it is middleware over the whole app, so the API
key a conforming client already sends *is* the authentication. There is no second
mechanism to configure.

An agent that supports a custom OpenAI base URL therefore needs configuration and
nothing else. OpenClaw is the worked example. It takes a `baseUrl` for its
`openai` TTS provider:

```json5
{ tts: { provider: "openai",
         providers: { openai: { apiKey: "local",
                                baseUrl: "http://127.0.0.1:8765/v1",
                                model: "tts-1",
                                responseFormat: "wav" } } } }
```

### Where the two APIs disagree

Every difference is decided in favour of saying so rather than guessing.

| their field | here |
|---|---|
| `model` | accepted and ignored. This server has one engine, and `/health` names it. Refusing a request for naming a model would break every client that must send one. |
| `response_format`, unset | **`wav`**, where OpenAI defaults to mp3. This is the one deliberate deviation: no mp3 encoder ships here (see `AudioFormat`), so honouring their default would answer every unconfigured client with an error instead of audio. |
| `response_format: mp3 \| aac \| opus` | refused, `400`, naming the formats that do work. `opus` is the trap: the `ogg` this server writes is Ogg **Vorbis**, which shares a media type with Ogg Opus and shares no bitstream, so answering with it would be a decode failure wrapped in a `200`. |
| `speed` | their range is 0.25–4.0 and this engine's is 0.5–2.0. Outside it is a `400` quoting the range, not a silent clamp. A caller that asked for 4× and received 2× has been handed audio it did not ask for, with nothing in the reply saying so. |
| `stream_format: "sse"` | not implemented. The field is ignored, so the whole utterance comes back in one response: correct audio, delivered less eagerly than asked. `/v1/synthesize/stream` is the route that streams, in this server's own envelope. |

`wav`, `flac` and `pcm` work, and this server's own `pcm16` and `ogg` are
accepted too, so a caller that knows what it is talking to need not translate.

Errors come back in OpenAI's envelope, `{"error": {"message": …}}`, rather than
FastAPI's `detail`. A conforming client reads `error.message` and would otherwise
show its user a blank HTTP failure, with the useful part (which voices exist,
which formats work) discarded.

## The gRPC service

The same engine behind a typed schema, for clients generated from
[`proto/loudkit.proto`](../../proto/loudkit.proto):

```bash
loudkit grpc --checkpoint loudr-1.safetensors --voices voices
```

Four methods: `Synthesize`, `SynthesizeStream`, `Describe`, `ListVoices`.
Loopback only. There is no auth on this transport at all, so a non-loopback
bind is refused outright rather than made configurable.

Every refusal the server decides carries `loudkit-error-code` in its trailing
metadata, the same frozen vocabulary the HTTP bodies carry as `code`. One
refusal is decided below the server: a request message over 256 KiB is refused
by gRPC core itself, as `RESOURCE_EXHAUSTED` with core's own message and no
`loudkit-error-code`. Core refuses before a servicer exists, so there is no
frame left to attach the metadata from.

The contract, in the order a request meets it:

* **Admission respects your deadline.** The engine is single-flight and
  callers queue for it, but the wait is capped at the request's own
  `time_remaining()`. A caller whose deadline expires in the queue never takes
  the engine, so the `DEADLINE_EXCEEDED` it receives is also the truth about
  what the server spent.
* **`Synthesize` refuses a reply your client cannot receive.** Default gRPC
  clients cap a message at 4 MiB. The preflight estimates the reply from the
  text as it will be spoken, after normalization, because the funnel expands:
  a thousand characters of digits become about five thousand characters of
  number words. It also reserves headroom for the WAV header, the provenance
  manifest and protobuf framing. The refusal names `SynthesizeStream`, which
  has no such ceiling.
* **Cancelling a call stops the work, on both RPCs.** A cancel and an expired
  deadline both set the flag the engine polls on every token decode step.
  Cancellation is cooperative, not preemptive: it cannot interrupt a backend
  kernel that is already executing, so a mel decode, a vocoder pass or a
  time-stretch inside its backend call runs to the end of that call first. A
  cancel lands within one such step, never within zero.
* **A stream is produced behind a bounded queue.** The render runs on its own
  thread and hands chunks across a two-slot queue to the socket. A peer that
  stays connected and stops reading stalls the queue, not the engine: at the
  server's stream cap the render stops and the engine returns to the pool,
  while only that peer's delivery stays stuck.
* **Every reply names its `sample_rate`.** wav, flac and ogg also record it in
  their headers; raw `pcm16` frames have no header, so the field is the only
  place a `pcm16` caller can read the rate from.
* **Chunk `continuation` is cumulative.** Each chunk carries the passage's
  tail as of that chunk, so chaining from the last chunk you received always
  hands the engine a full prefix, even when the closing sentence was shorter
  than one. The final chunk's value equals `Synthesize`'s `continuation` for
  the same request.

Running the HTTP server and the gRPC server over one engine in one process is
not supported: each transport holds its own single-flight lease, and nothing
arbitrates between them. One engine per transport process.

## Shared synthesis path

A second path is a second thing to keep in agreement, and this library exists
because two paths drifted once. The test suite asserts that the server's bytes
are identical to calling the engine directly, and the MCP tool, the gRPC service
and the OpenAI-compatible route all resolve through the same `render_bytes`. One
engine, four transports, one path.

## Next

Measure your hardware, stage by stage.

> Behind a reverse proxy every client shares the proxy's address, so the
> per-client rate limiter collapses into one global bucket. Give the proxy its
> own per-client limit.
