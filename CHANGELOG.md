# Changelog

Notable changes, in the shape [Keep a Changelog](https://keepachangelog.com)
suggests. Versioning and what counts as a breaking change are spelled out in
[docs/reference/COMPATIBILITY.md](docs/reference/COMPATIBILITY.md).


## [0.1.0] — 2026-08-25

First public release.

### Added

#### Speech

- A two-stage engine optimized for local inference: an autoregressive token
  generator at 25 Hz and a parallel renderer. It runs faster than real time on
  a laptop, a phone and a Jetson. Benchmarks with the command that reproduces
  each number are in
  [docs/benchmarks.md](docs/benchmarks.md).
- **20 voices across 10 languages**, every one enrolled from a recording made
  or released for speech-technology use — consented donations and CC0 / CC-BY
  corpora — with donor or source, licence and consent basis named. Quality is
  evaluated for **English**; the rest are read, not yet measured.
- **Voice cloning from about ten seconds** of audio, producing a ~150 KB profile
  that is a file, not a model.
- Long-form synthesis that splits at sentence boundaries and carries prosody
  across the joins, and streaming that yields the first sentence rather than
  waiting for the paragraph.
- Token-level cancellation, so a voice agent's barge-in goes quiet within one
  decode step instead of at the end of a chunk. It is cooperative: a kernel
  already running is not interrupted.

#### Latency

- Streaming renders window *k* while generating window *k+1* — the chunk chain
  is sequential only through the tokens, so the overlap changes no byte
  (asserted against the serial path). Measured on an M3 Pro: a six-window
  passage 1.29x faster, the Apple bench row 2.81x → 3.43x.
- `ChunkConfig.first_chunk_max_tokens` caps only the first chunk, so a stream
  opens on the first clause instead of a full window: first audio ~1.9 s →
  ~1.4 s at a 96-token budget (M3 Pro). An algorithm value — setting it
  re-fingerprints; unset it is absent from the fingerprint. Python first; the
  ports follow.
- The generator's conditioning row is memoised per voice (a content key), so
  chunks and repeated requests in one voice stop recomputing it.
- `Engine.warm()` pays the first-use costs (kernel autotune, graph capture,
  allocator pools) at startup; `serve`, gRPC and MCP call it after loading, so
  the first request pays warm latency (measured on a 3090 with graphs:
  1.09 s → 0.73 s first audio, for 1.3 s once at boot).

#### Command line

- `loudkit doctor` — what this machine can run and the one command that fixes
  each gap. `loudkit download <repo>` — the checkpoint, the graphs one backend
  needs and all twenty voices into the shared cache; `--with-cloning` adds the
  encoder and the enrollment graphs. `loudkit voices <repo>` — the menu without the gigabyte.
  `loudkit verify <path>` — a checkpoint, voice profile or rendered WAV checked
  against its own claims (payload digest, profile validation, C2PA binding).

#### Errors

- A frozen catalog of error codes, defined in `loudkit.errors` and spoken by
  every transport: `exc.code` on Python exceptions, `"code"` in HTTP error
  bodies, `"error_code"` on the SSE terminal event, `loudkit-error-code` in
  gRPC trailing metadata. Codes are never renamed or reused, only added. See
  [docs/reference/errors.md](docs/reference/errors.md).

#### Five implementations, one behaviour

- Python is the reference; **Swift, Go, Rust and TypeScript are full ports**, not
  wrappers, held to shared conformance fixtures. All five compute the same
  algorithm fingerprint independently and agree.
- The same text, voice and seed give a bit-identical waveform on a given build,
  device and backend. Waveforms are **not** guaranteed to match across backends
  or devices. The engine refuses to start if two of its components disagree
  about what to compute. See
  [docs/reference/IDENTITY-CONTRACT.md](docs/reference/IDENTITY-CONTRACT.md).

#### Text handling

- Numbers, decimals, currency, units, times and abbreviations verbalised in
  **12 languages** from first principles — no LGPL dependency — and checked
  against a 1300-row CLDR differential.
- Sentence-boundary chunking, NFC normalisation, output charset closure, and a
  Polish anglicism respeller.
- Control-tag injection closed: bracketed text in user input can no longer
  trigger model behaviours.

#### It checks its own output

- Six detectors read the tokens the generator just produced and find the failure
  no audio filter can — a hallucinated tail after a long silence, in the same
  voice. Every threshold comes from a measured trace, and was
  re-measured across the nine spoken languages the roster then carried
  (Swedish, added later, ships unmeasured on the same margin). A chunk that is
  certainly wrong but that no rule can place is *reported*, not silently
  shipped. See [docs/reference/postprocess.md](docs/reference/postprocess.md).
- Selective re-roll: a chunk that fails is re-rendered from a derived seed
  rather than returned.

#### For a reading app

- `Result.chunks` gives exact per-chunk spans from sample offsets, plus
  **estimated** word times by proportional allocation — documented as an
  estimate, because it is not a forced aligner
  ([docs/reference/timestamps.md](docs/reference/timestamps.md)).
- `speed=` is the video-player control: faster without a pitch shift, via WSOLA
  written from first principles ([docs/reference/speed.md](docs/reference/speed.md)).
- `previous_tokens=` carries prosody across two separate calls.

#### Interfaces

- A local HTTP server under `/v1` with one-shot and SSE streaming routes and
  four output encodings (wav, pcm16, flac, ogg) — a **working example, not a
  production deployment**. A public bind requires a bearer token; a loopback
  bind does not require one, and authenticates when given one (see the server
  guide).
- An MCP server for agents, and a Linux Speech Dispatcher module so screen
  readers can use it.
- Discovery without loading anything: `loudkit.languages()`, `loudkit.voices()`.

#### Provenance

- Saved WAVs and server replies carry an unsigned C2PA claim-only manifest
  binding the fingerprint, recipe and seed to a hash of the audio
  ([docs/reference/provenance.md](docs/reference/provenance.md)). The provenance
  guide documents the manifest fields, verification and signing boundary.

### Known limitations

- Quality evaluated for English only.
- Ordinals are English-only: it is the one language here that writes a letter
  suffix (`22nd`), and the other eleven need the inflection work the token layer
  is being built for.
- Yearless numeric dates (`12.3.`) are deliberately **not** read as dates in any
  language. The shape is indistinguishable from a decimal at the end of a
  sentence, and reading it wrong is worse than reading it plainly.
- There is no expressiveness control. The checkpoint architecture reserves a
  conditioning slot for an emotion scalar, but the value does not change the
  output in the released checkpoint. The field was
  removed from `VoiceProfile` and the profile format before release; the slot
  is fed the training constant, and a profile that carries a legacy `emotion`
  header key still loads (the key is ignored).
- Apache-2.0, derived from Chatterbox (MIT, Resemble AI). The enrollment
  architectures come from CosyVoice, S3Tokenizer and 3D-Speaker (Apache-2.0),
  FunASR and Real-Time-Voice-Cloning (MIT); the Polish respelling lexicon is
  derived from CMUdict (CMU, BSD-2-clause). Everything upstream is permissively
  licensed, nothing is copyleft, and each holder is named in [NOTICE](NOTICE).
