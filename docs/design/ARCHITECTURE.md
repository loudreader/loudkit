# Architecture

loudkit is one engine with five implementations: Python (the reference), Rust,
Go, TypeScript and Swift. The layers below state what each Python module may
import. The table after them names the file that holds each concept in each
language.

## The layers (Python reference)

A module imports from its own row and the rows above it, with two
exceptions. `config` imports `frontend.chunking` and `manifest`, and each of
them imports `config` back; a lazy import breaks each cycle. `contracts`
imports `config` and `voice` from lower rows. `tests/test_import_graph.py`
parses every module and fails on any edge that is missing from its allowlist.

```
errors · contracts · provenance · rng · timing   foundation
_version                                         (_version is a leaf: the
        │                                        literal provenance reads)
        ▼
checkpoint · postprocess                         release layout on disk;
release · checksums                              artifact guards; what a
        │                                        release is and its SHA256SUMS
        ▼
hub · config · execution · manifest              release resolution; algorithm
frontend/                                        knobs; execution knobs; the
        │                                        manifest reader; the text
        │                                        funnel (numbers, dates,
        ▼                                        letters, chunking, speechtext,
voice · sampler                                  text, textconfig)
        │
        ▼
models/  ·  backends/                            signal & network modules
        │                                        (flow, generator, vocoder,
        │                                        enroll, resample, noise,
        ▼                                        windowing, timestretch)
window · result                                  one window's synthesis, and
        │                                        the value a call returns
        ▼
   engine · stream                               orchestration: one synthesis
        │                                        path, whole or chunk by chunk
        ▼
    synthesis                                    render_bytes and
        │                                        render_stream_chunks: the
        │                                        transport-agnostic render calls
        ▼
transports/  http · mcp · grpc                   three adapters, peers, never
        │                                        layered on one another
        ▼
       cli
```

`config` and `frontend/` share a row because of their cycle. `backends/`
imports `models/` from its own row and lazily imports `engine` from a lower
row. `stream` imports `engine` for typing and calls the `window` functions.
The allowlist in `tests/test_import_graph.py` is the complete edge list.

The test enforces these rules:

* `frontend/*` never imports the engine, a backend or packaging.
* `models/*` never imports the frontend.
* `postprocess` does not import `config`. `config` imports the detector
  configuration from `postprocess`.
* `hub` never imports the engine or a transport, so release resolution works
  before any weights exist.
* A transport importing a peer, or the cli, fails the suite.
* Adding any edge means adding it to the allowlist in the same commit.

## Where a concept lives, per language

Python is the reference. The other four are full implementations that pass
the same conformance fixtures (`tests/data/conformance/`). File layouts follow
each language's conventions; concept names are the same in all five.

| concept | Python | Rust | Go | TypeScript | Swift |
|---|---|---|---|---|---|
| numbers grammar | `frontend/numbers.py` | `src/numbers.rs` | `speechtext/numbers.go` | `src/numbers.ts` | `LoudKitText/Numbers.swift` |
| date rules | `frontend/dates.py` | `src/dates.rs` | `speechtext/dates.go` | `src/dates.ts` | `LoudKitText/Dates.swift` |
| letter names | `frontend/letters.py` | `src/letters.rs` | `speechtext/letters.go` | `src/letters.ts` | `LoudKitText/Letters.swift` |
| sentence splitting | `frontend/chunking.py` | `src/chunking.rs` | `chunking/chunking.go` | `src/chunking.ts` | `LoudKit/Chunking.swift` |
| artifact guards | `postprocess.py` | `src/postprocess.rs` | `postprocess/postprocess.go` | `src/postprocess.ts` | `LoudKit/Postprocess.swift` |
| text funnel driver | `frontend/speechtext.py` | `src/speechtext.rs` | `speechtext/speechtext.go` | `src/speechText.ts` | `LoudKitText/SpeechText.swift` |
| tokenizer frontend | `frontend/text.py` | `src/frontend.rs` | `frontend/frontend.go` | `src/frontend.ts` | `LoudKit/TextFrontend.swift` |
| sampler (Philox) | `sampler.py` | `src/sampler.rs` | `sampler/sampler.go` | `src/sampler.ts` | `LoudKit/Sampler.swift` |
| RNG core | `rng.py` | `src/rng.rs` | `rng/rng.go` | `src/rng.ts` | `LoudKit/Philox.swift` |
| engine orchestration | `engine.py` | `src/engine.rs` | `engine/engine.go` | `src/engine.ts` | `LoudKit/Engine.swift` |
| voice profiles | `voice.py` | `src/voice.rs` | `voice/voice.go` | `src/voice.ts` | `LoudKit/VoiceProfile.swift` |
| release resolution | `hub.py` | `src/hub.rs` | `hub.go` | `src/hub.ts` | `LoudKit/Hub.swift` |
| provenance manifests | `provenance.py` | n/a | n/a | n/a | n/a |
| synthesis surface | `synthesis.py` | n/a¹ | n/a¹ | n/a¹ | n/a¹ |
| HTTP / MCP / gRPC adapters | `transports/` | n/a² | n/a² | n/a² | n/a² |

¹ Server-only surfaces. The ports ship libraries. Rust and Go add a thin CLI
(`rust/src/main.rs`, `go/cmd/loudkit/main.go`), the npm package declares no
`bin`, and the Swift package's one executable target is the `Hello` example.
² Python only: the other languages ship no server.

## Reading order for a new contributor

1. `docs/reference/IDENTITY-CONTRACT.md`: what "same input, same audio" means
   and under which conditions it holds.
2. `python/loudkit/engine.py`: the synthesis path everything shares.
3. Any file in this table, in the language you will work in. Its header names
   its Python reference.
