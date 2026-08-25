# Architecture

One engine, five implementations, one contract. This page maps what each layer
may depend on, and where a concept lives in every language.

## The layers (Python reference)

Dependencies point downwards only, with one sanctioned exception: `config` and
`frontend.chunking` are mutually dependent by design (a lazy import breaks the
cycle), and `contracts` reaches config and voice. The direction is enforced, not
conventional. `tests/test_import_graph.py` parses every module and fails on an
undeclared edge.

```
errors · contracts · provenance · rng · timing   foundation
        │
        ▼
checkpoint · postprocess                         release layout on disk;
        │                                        artifact guards
        ▼
hub · config · frontend/                         release resolution; algorithm
        │                                        knobs; the text funnel
        │                                        (numbers, dates, letters,
        ▼                                        chunking, polish, text,
voice · sampler                                  textconfig)
        │
        ▼
models/  ·  backends/                            signal & network modules
        │                                        (flow, generator, vocoder,
        │                                        enroll, resample, noise,
        ▼                                        windowing, timestretch)
      engine                                     orchestration: one synthesis path
        │
        ▼
    synthesis                                    render_bytes: the only place
        │                                        audio is made; transport-agnostic
        ▼
transports/  http · mcp · grpc                   three adapters, peers, never
        │                                        layered on one another
        ▼
       cli
```

A row is a level, not a promise about its members. `config` and `frontend/`
share one because of the sanctioned cycle, and `backends/` reaches sideways
into `models/` and lazily back up into `engine`. The allowlist is the exact
statement; this is the shape.

Rules with teeth:

* `frontend/*` never imports the engine, a backend or packaging.
* `models/*` never imports the frontend.
* `postprocess` knows nothing about `config`. The configuration reads the
  detectors, never the other way round.
* `hub` never imports the engine or a transport. Release resolution has to work
  before any weights exist.
* A transport importing a peer, or the cli, fails the suite.
* Adding any edge means adding it to the allowlist in the same commit.

## Where a concept lives, per language

Python is the reference. The other four are full implementations held to the
same conformance fixture (`tests/data/conformance/`). Layouts differ by
ecosystem idiom; names do not.

| concept | Python | Rust | Go | TypeScript | Swift |
|---|---|---|---|---|---|
| numbers grammar | `frontend/numbers.py` | `src/numbers.rs` | `speechtext/numbers.go` | `src/numbers.ts` | `LoudKitText/Numbers.swift` |
| date rules | `frontend/dates.py` | `src/dates.rs` | `speechtext/dates.go` | `src/dates.ts` | `LoudKitText/Dates.swift` |
| letter names | `frontend/letters.py` | `src/letters.rs` | `speechtext/letters.go` | `src/letters.ts` | `LoudKitText/Letters.swift` |
| sentence splitting | `frontend/chunking.py` | `src/chunking.rs` | `chunking/chunking.go` | `src/chunking.ts` | `LoudKit/Chunking.swift` |
| artifact guards | `postprocess.py` | `src/postprocess.rs` | `postprocess/postprocess.go` | `src/postprocess.ts` | `LoudKit/Postprocess.swift` |
| text funnel driver | `frontend/polish.py` | `src/speechtext.rs` | `speechtext/speechtext.go` | `src/speechText.ts` | `LoudKit/TextFrontend.swift` |
| sampler (Philox) | `sampler.py` | `src/sampler.rs` | `sampler/sampler.go` | `src/sampler.ts` | `LoudKit/Sampler.swift` |
| RNG core | `rng.py` | `src/rng.rs` | `rng/rng.go` | `src/rng.ts` | `LoudKit/Philox.swift` |
| engine orchestration | `engine.py` | `src/engine.rs` | `engine/engine.go` | `src/engine.ts` | `LoudKit/Engine.swift` |
| voice profiles | `voice.py` | `src/voice.rs` | `voice/voice.go` | `src/voice.ts` | `LoudKit/VoiceProfile.swift` |
| release resolution | `hub.py` | `src/checkpoint.rs` | `checkpoint/` | `src/checkpoint.ts` | `LoudKit/Checkpoint.swift` |
| provenance manifests | `provenance.py` | n/a | n/a | n/a | n/a |
| synthesis surface | `synthesis.py` | n/a¹ | n/a¹ | n/a¹ | n/a¹ |
| HTTP / MCP / gRPC adapters | `transports/` | n/a² | n/a² | n/a² | n/a² |

¹ Server-only surfaces. The ports ship libraries and thin CLIs, which is why
their `main` entry points are a hundred-odd lines.
² Python-only by definition: the other languages have no server to adapt.

## Reading order for a new contributor

1. `docs/reference/IDENTITY-CONTRACT.md`: what "same input, same audio" means
   here, precisely.
2. `python/loudkit/engine.py`: the synthesis path everything shares.
3. Any file in this table, in the language you will work in. Its header names
   its Python reference.
