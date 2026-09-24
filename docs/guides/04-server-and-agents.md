# 4. Server, streaming API and MCP

The loudr-1 checkpoint is 747 MB and takes a few seconds to load. For
interactive use, load it once in a server process and send it requests.
loudkit has four ways to reach a loaded engine:

- an HTTP server (`loudkit serve`) with two APIs in one process: loudkit's own
  REST routes and OpenAI's `POST /v1/audio/speech`;
- a gRPC service
  ([`proto/loudkit.proto`](../../proto/loudkit.proto), `loudkit serve --grpc`);
- an MCP server for agents (`loudkit serve --mcp`);
- a Speech Dispatcher module
  ([`integrations/speech-dispatcher/`](../../integrations/speech-dispatcher/)).
  It forwards to a running `loudkit serve`, so applications that use Speech
  Dispatcher, such as Orca, Firefox and `spd-say`, can speak with loudkit.

`loudkit serve` starts the HTTP server, and `--grpc` or `--mcp` selects another
transport. HTTP, gRPC and Speech Dispatcher are supported; MCP is a preview.
The HTTP, gRPC and MCP servers call the same `Engine` through one shared
synthesis path. The test suite checks that the server returns the same bytes
as a direct engine call.

```bash
pip install "loudkit[server,hub]"    # REST, and the OpenAI-compatible route with it
pip install "loudkit[mcp,hub]"       # MCP (pulls the server's engine too)
pip install "loudkit[grpc,hub]"      # gRPC
```

## Scope

The server keeps a model loaded on your own machine for a script, an editor or
an agent. It is not hardened for a public deployment. Read both lists below
before you run it anywhere else.

What it has:

* It binds `127.0.0.1` by default. `--host` changes the address.
* A request size limit: 10 000 characters of text, and a byte limit on the body.
  A body over the byte limit is refused before it is read in full.
* A bounded queue in front of the engine, which renders one request at a time.
* Status codes that separate the two kinds of error: `4xx` for a problem in the
  request, `5xx` for a server fault. The streaming route reports late failures
  in `error_kind`, because its status code is already sent.

What it does not have on the default loopback bind:

* Authentication. Any process on this machine can synthesize, including the
  processes of other users. A non-loopback bind needs `--allow-public` and then
  requires a bearer token.
* Per-caller rate limits or quotas. The queue bound protects the process. It
  does not share capacity fairly between callers, and one client can fill it.
* TLS. Text and audio cross the connection unencrypted.
* Multi-tenancy, isolation, usage accounting or abuse controls. There is one
  engine and one voice library, and the server does not identify callers.

Do not expose this server to a network you do not control. Anyone who reaches
the port can synthesize in every loaded voice on your hardware. On plain HTTP,
anyone on the network path can also read the requests. Keep the server on
loopback, or put it behind a proxy that authenticates callers and terminates
TLS.

## The REST server

```bash
loudkit serve --checkpoint loudreader/loudr-1 --port 8765
```

The server binds to localhost and uses the voices in the release. Pass
`--voices <directory>` to use a directory of your own profiles. A request names
a voice, and the server looks the name up in that library. It does not accept
file paths.

The API is under `/v1`: `/v1/voices`, `/v1/synthesize`, `/v1/synthesize/stream`.
`GET /health` is not versioned.

The server enforces these limits:

* A non-loopback bind needs `--allow-public`. Every request must then carry
  `Authorization: Bearer <token>`. Put the token in the `LOUDKIT_TOKEN`
  environment variable. `--token` also works, but other accounts on the host
  can read a process's command line. If you give no token, the server generates
  one and prints it to stderr. A token must be at least 16 printable ASCII
  characters. There is no public mode without a token.

  On a public bind, each client address can send 12 synthesis requests in a
  burst, then one every two seconds. Past that the answer is `429`. Behind a
  reverse proxy, all clients share the proxy's address and one budget.

  The server speaks plain HTTP and has no TLS option. On a public bind the token
  and the audio cross the network unencrypted, and anyone who sees the traffic
  can read the token and reuse it. Terminate TLS in front of the server, with
  Caddy, nginx, a cloud load balancer or an SSH tunnel. Keep the server on
  loopback behind it.

  On loopback the server ignores any token and has no rate limit, so
  authenticate and rate limit at the proxy. It also accepts only a `Host`
  header that names this machine: `localhost`, `127.0.0.1` or `::1`, with any
  port. Any other `Host` gets `403` with the code `bad_host`. Configure the
  proxy to send a loopback `Host`, for example `127.0.0.1:8765`. nginx does
  this by default. Caddy and most cloud load balancers forward the client's
  `Host`. In Caddy, add `header_up Host {upstream_hostport}` to the
  `reverse_proxy` block.
* Text is capped at 10 000 characters, and a longer text gets `422`. The body
  also has a byte limit, sized so that text at the character cap fits in its
  longest standard JSON encoding. A larger body gets `413`.
  [Transports](../design/transports.md) has the arithmetic.
* The engine renders one request at a time. At most 32 requests are in flight
  or waiting, the one rendering included. Past that the answer is `503` with
  `Retry-After`.

Before it reports ready, the server renders once in the first voice of its
library and discards the audio. This moves the extra cost of the first render
to startup. The stderr line `warm: first-use costs paid on voice '<name>'`
gives the voice and the time it took. `--grpc` and `--mcp` warm up the same way.

To skip the warm-up and start sooner, set `LOUDKIT_NO_WARM` to any non-empty
value. `LOUDKIT_NO_WARM=0` also skips it. To warm up again, unset the variable
or set it to an empty value. If the warm-up fails, the server prints the reason
and starts anyway.

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
X-Loudkit-Fingerprint   <16 hex digits naming the algorithm build>
X-Loudkit-Truncated     false
X-Loudkit-Continuation  312,4088,77,1901,55,640
```

On one server, the same text, voice and seed give the same samples every time.
The bytes are also identical in every format except `ogg` and `opus`, whose Ogg
container carries a random stream serial. Another build, device or execution
setting can give different samples. See the
[identity contract](../reference/IDENTITY-CONTRACT.md).

**Check `X-Loudkit-Truncated`.** `true` means generation stopped at the token cap
instead of at a stop token, so the audio is cut off mid-sentence. The reply is
still a `200` with a valid WAV, so only the header tells you.

**`speed` is playback speed**, `0.5` to `2.0`, pitch preserved. A value outside
that range gets `422`. The default, `1.0`, leaves the audio unchanged. See
[speed.md](../reference/speed.md).

**`format` picks the encoding**, one of:

| `format` | `Content-Type` | notes |
|---|---|---|
| `wav` (default) | `audio/wav` | 16-bit PCM. |
| `pcm16` | `application/octet-stream` | Header-less 16-bit frames, **little-endian**, at `X-Loudkit-Sample-Rate`. For feeding a device or a socket directly. |
| `flac` | `audio/flac` | Lossless, about half the size of the WAV. |
| `ogg` | `audio/ogg` | Vorbis, lossy. The same number of samples, with different values. |
| `mp3` | `audio/mpeg` | Lossy and small. The default of OpenAI clients. |
| `opus` | `audio/ogg; codecs=opus` | Ogg Opus, the format of chat voice notes. The `ogg` format above is Vorbis. |

Any other name gets `422` listing the six. `mp3` and `opus` need the MPEG and
Opus codecs of the libsndfile that `soundfile` loads, and the `soundfile` wheels
include them. If the local libsndfile lacks them, the server refuses `mp3` and
`opus` by name before synthesis starts.

The `pcm16` media type is `application/octet-stream`. `audio/L16` does not fit:
RFC 2586 defines L16 as big-endian, and these frames are little-endian.

**`previous_tokens` continues a previous request.** Send the
`X-Loudkit-Continuation` value of the previous reply as a JSON array of
integers. The engine conditions the first chunk on that tail, the same way it
joins the chunks of one passage. The reply carries the tail in a header because
the body is audio. Up to 4096 ids are accepted, and only the tail is used. More
ids get `422`. The full contract is in
[02-streaming-and-long-form.md](02-streaming-and-long-form.md#carrying-that-join-across-two-calls).

**Omit `language` to use the voice's language.** The request is then read in
the language the voice was enrolled in, or in `en` if the profile records none.
The rule is
[the same everywhere](../design/preprocess.md#which-language-this-layer-runs-as).
Set `language` only for cross-lingual synthesis, such as an English voice
reading Polish text.

`400` means this build cannot do what the request asks. Most often the
`language` is not one of the twelve ids the text layer supports. The body names
the refused id and the twelve supported ids. The CLI prints the same refusal as
`unsupported: ...`. The MCP tool returns it as `error`, with
`"code": "unsupported_language"` and the `supported` list.

### Streaming synthesis (Server-Sent Events)

The streaming route delivers the same synthesis chunk by chunk, so a client can
start playing before the rest is rendered:

```bash
curl -N -X POST localhost:8765/v1/synthesize/stream -H 'Content-Type: application/json' \
  -d '{"text":"A longer passage with several sentences.","voice":"joe"}'
```

Each event is a JSON object with the chunk's audio in base64, plus its
`media_type`, duration, token count, `truncated`, `sample_rate` and
`fingerprint`. The response also carries the rate in `X-Loudkit-Sample-Rate`,
because raw `pcm16` frames do not state it. The final event is
`{"done": true, ...}`. Its `truncated` is true if any chunk was cut off. The
route uses the engine's `stream()`.

`speed` and `previous_tokens` work here too. The `done` event carries
`continuation`, the tail of the whole passage, to send as `previous_tokens` in
the next request.

**`format` on this route is `wav`, `pcm16` or `flac`.** Each event is a
complete file in that format, playable on its own. `ogg`, `opus` and `mp3` get
`422`; use `/v1/synthesize` for them. Raw `pcm16` frames from successive events
concatenate directly.

**Read until `done`, and check it for `error`.** Once the response has started,
a failure cannot change its `200` status. The failure arrives as
`{"done": true, "error": "...", "error_kind": "..."}` instead. Treat a stream
that ends without a `done` event as incomplete. Errors found before the response
starts, such as an unknown voice or a full queue, are ordinary status codes
(404, 503).

`error_kind` classifies a failure after the response has started:

| value | meaning | what to do |
|---|---|---|
| `bad_request` | A problem with this request, such as a language outside the supported ids or a chunk too long for the window. `/v1/synthesize` answers these with `400` or `422`. | Change the request before you retry. |
| `server_fault` | A defect in this build. `/v1/synthesize` answers these with `500`. | Check the server log before you retry. |

Closing the connection stops the synthesis. The server checks for a disconnect
while the render runs and cancels the decode loop within one step. A voice
agent can use this to interrupt speech: the engine stops within one decode
step, before the end of the current chunk of about 10 s. A backend call that is
already running finishes first. `/v1/synthesize` stops the same way when its
client disconnects.

A client that stays connected but stops reading is not a disconnect, and the
server cannot detect it. The write blocks and the response does not end. The
HTTP and gRPC streams are capped at ten minutes of wall-clock time from the
moment they take the engine. At the cap the render stops and the engine is
released. The cap applies to every stream, so on a slow device split a long
text across requests.

## The MCP server (preview)

`loudkit serve --mcp` serves the Model Context Protocol on stdio, for MCP
clients such as Claude Code, Cursor or Cline:

```bash
loudkit serve --mcp --checkpoint loudreader/loudr-1
```

It has three tools: `list_voices`,
`synthesize(text, voice, seed, language, speed, previous_tokens, format)` and
`describe()`. Everything after `voice` is optional, and an omitted `language`
means the voice's own.

`synthesize` returns the audio in base64, WAV by default, with `format`,
`media_type`, `duration`, `tokens`, `sample_rate`, `fingerprint`, `truncated` and
`continuation`. `format: "flac"` gives the same samples at about half the size.
This helps when the MCP host puts the reply into a model's context.

`truncated` is the same cut-off signal the HTTP header carries. `continuation`
is the tail to pass back as `previous_tokens` on the next call, so that a long
text read in several calls joins the way the chunks of one passage do. The
reply carries only the tail the engine uses. A refusal comes back as `error`,
with `error_kind` and `code`. In a client:

```
# any MCP client
synthesize: text="Deploy complete." voice="joe" seed=7
```

`describe()` returns the resolved algorithm and execution settings. When a
synthesis sounds wrong, check it first to see which mode was active.

## The OpenAI-compatible route

`POST /v1/audio/speech` implements OpenAI's speech API, with the differences
listed below. Set a client's base URL to this server's `/v1`, and use a voice
from this server's library.

```bash
curl -s http://127.0.0.1:8765/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"Deploy complete.","voice":"joe","response_format":"wav"}' \
  -o out.wav
```

With OpenAI's Python client:

```python
from openai import OpenAI

speech = OpenAI(api_key="…", base_url="http://127.0.0.1:8765/v1")
speech.audio.speech.create(
    model="tts-1", voice="joe", input="Deploy complete.", response_format="wav"
).write_to_file("out.wav")
```

On a public bind, set `api_key` to this server's bearer token. On loopback the
server ignores it, so any value works.

### Connect your agent

To connect an agent that accepts a custom OpenAI base URL:

1. Start the server with `loudkit serve`. It listens on `127.0.0.1:8765`.
2. Point the agent's OpenAI speech provider at `http://127.0.0.1:8765/v1`. The
   agent asks for an API key. On loopback any value works; on a public bind,
   use the server's token.
3. Set a voice from this server's library, which `GET /v1/voices` lists.
   OpenAI's voice names (`alloy`, `coral`) do not exist here.

The agent picks the format. Hermes Agent and OpenClaw ask for Opus when the
reply is a voice note and for mp3 otherwise. This server returns both.

OpenClaw takes a `baseUrl` for its `openai` TTS provider:

```json5
{ tts: { provider: "openai",
         providers: { openai: { apiKey: "local",
                                baseUrl: "http://127.0.0.1:8765/v1",
                                model: "tts-1",
                                speakerVoice: "joe" } } } }
```

Hermes Agent takes a `base_url` for its `openai` provider in
`~/.hermes/config.yaml`:

```yaml
tts:
  provider: openai
  openai:
    base_url: http://127.0.0.1:8765/v1
    api_key: local
    voice: joe
```

### Where the two APIs disagree

| their field | here |
|---|---|
| `model` | Accepted and ignored. The server has one engine, and `/health` names it. |
| `voice` | A name in this server's library, which `/v1/voices` lists. OpenAI's own names (`alloy`, `coral` and the others) get `404`. |
| `response_format`, unset | `wav`. OpenAI's default is mp3. |
| `response_format: aac` | `400`, naming the formats that work. libsndfile has no AAC encoder. |
| `speed` | OpenAI's range is 0.25 to 4.0, and this engine's is 0.5 to 2.0. A value outside it gets `400` quoting the range. |
| `stream_format: "sse"` | Ignored. The whole utterance comes back in one response. `/v1/synthesize/stream` streams, in this server's own event format. |

`wav`, `flac`, `pcm`, `mp3` and `opus` work. This server's own `pcm16` and
`ogg` are accepted too.

Errors from this route use OpenAI's format, `{"error": {"message": …}}`. Two
kinds of error still use loudkit's `{"detail": …, "code": …}` format:

* a body that fails validation (`422`), for example a missing `voice` or an
  `input` over 10 000 characters;
* the refusals checked before any route runs: `401`, `403`, `413`, `415` and
  `429`.

A client must handle both formats.

## The gRPC service

The same engine behind a typed schema, for clients generated from
[`proto/loudkit.proto`](../../proto/loudkit.proto):

```bash
loudkit serve --grpc --checkpoint loudreader/loudr-1
```

It listens on `127.0.0.1:50051` and has four methods: `Synthesize`,
`SynthesizeStream`, `Describe`, `ListVoices`. The gRPC transport has no
authentication, so it refuses a non-loopback bind.

Every refusal the server decides carries `loudkit-error-code` in its trailing
metadata, with the same codes the HTTP bodies carry as `code`. One refusal comes
from gRPC itself: a request message over 256 KiB gets `RESOURCE_EXHAUSTED` with
gRPC's own message and no `loudkit-error-code`.

The contract, in the order a request meets it:

* Waiting for the engine respects your deadline. Callers queue for the
  single-flight engine, but no longer than the request's own
  `time_remaining()`. A caller whose deadline expires in the queue gets
  `DEADLINE_EXCEEDED` and never takes the engine.
* `Synthesize` refuses a reply that a default client cannot receive. Default
  gRPC clients accept messages up to 4 MiB. The server estimates the reply size
  from the text after normalization, which can expand it: a thousand characters
  of digits become about five thousand characters of number words. The estimate
  reserves room for the WAV header, the
  [machine-readable note](../reference/provenance.md) and protobuf framing. The
  refusal names `SynthesizeStream`, which sends one message per chunk.
* Cancelling a call stops the work, on both RPCs. A cancel and an expired
  deadline both set the flag the engine checks on every token decode step. A
  backend call that is already running (a mel decode, a vocoder pass or a
  time-stretch) finishes first, and the stop takes effect after it returns.
* A stalled reader blocks only its own stream. At the ten-minute stream cap the
  render stops and the engine is released, even if delivery to that peer stays
  blocked.
* Every reply states its `sample_rate`. wav, flac, ogg, mp3 and opus also record
  it in their headers. Raw `pcm16` frames have no header, so read the rate from
  this field.
* Chunk `continuation` is cumulative. Each chunk carries the passage's tail up
  to that chunk. To continue after the last chunk you received, pass its value.
  The final chunk's value equals the `continuation` of `Synthesize` for the same
  request.

Running the HTTP server and the gRPC server over one engine in one process is
not supported. Each transport holds its own single-flight lease, and nothing
arbitrates between them. Run one transport per process.
[Transports](../design/transports.md) describes how the servers share the
synthesis path.

## Next

[Troubleshooting](../reference/troubleshooting.md) has the server section's
usual failures and their fixes.
