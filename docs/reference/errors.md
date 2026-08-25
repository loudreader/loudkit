# What each implementation raises, and what a caller can do about it

The five implementations refuse the same inputs, and a shared fixture tests them
for it. They do not share an error *type* system. What each raises differs by
language, so a caller porting between them cannot assume the same catch will
work.

They do share a **catalog of error codes**: short stable strings naming each
condition. The catalog is frozen. Codes are never renamed or reused, only added.
It is defined in `loudkit.errors` and spoken by every transport, so the same
refusal has the same name whether it arrived as a Python exception, an HTTP body
or a gRPC status.

## The catalog

| code | condition | Python class |
|---|---|---|
| `invalid_tokens` | a speech token id outside the codebook, negative, fractional, or an empty sequence | `InvalidTokensError` |
| `unsupported_language` | a language this build's frontend cannot preprocess | `UnsupportedLanguageError` |
| `voice_not_found` | a voice name the library does not have | `VoiceNotFoundError` |
| `number_grammar` | a number past the largest scale the language's grammar names | `NumberGrammarError` |
| `window_overflow` | text longer than one window with chunking off | `WindowOverflowError` |
| `provenance_invalid` | a C2PA manifest present and not verifiable | `ProvenanceError` |
| `invalid_request` | the text funnel emptied the request, or any other refusal a class does not name yet | `NothingToSpeakError`, or a bare `ValueError` at a boundary |
| `server_fault` | not a refusal: a stub method, a dependency failure, a bug | anything outside `LoudkitError` |

Conditions that exist only at a transport boundary carry their own codes:
`unauthorized` (401), `bad_host` (a public bind refused, 403),
`payload_too_large` (413), `rate_limited` (429), `busy` (queue full or engine
wedged, 503), `timeout` (a stream held the engine past its bound).

## Where each transport says it

| transport | where the code travels |
|---|---|
| Python | `exc.code` on every `LoudkitError`; `loudkit.errors.error_code(exc)` reads the class and returns `invalid_request` for anything outside it. `server_fault` is the transport's word, applied to an exception the boundary did not classify as a refusal |
| HTTP | `"code"` in every JSON error body, beside `detail` |
| SSE stream | `"error_code"` on the terminal `done` event, beside `error_kind` |
| gRPC | `loudkit-error-code` trailing metadata, beside the status code |
| Go / Rust / JS / Swift | not yet; see the upgrade path below |

## The shapes, per language

| | how failure is expressed | can a caller branch on it? |
|---|---|---|
| **Python** | seven raised classes under `LoudkitError` in `loudkit.errors`, each also a stdlib type: `InvalidTokensError(LoudkitError, ValueError)`, `VoiceNotFoundError(LoudkitError, FileNotFoundError)`, and so on | **yes**, by class or by stdlib supertype. `except ValueError` catches the request errors; `except LoudkitError` catches everything this library raises and nothing else |
| **Swift** | one enum, `LoudKitError`, with five cases: `.manifest`, `.asset`, `.shape`, `.prediction`, `.cancelled` | **yes**, by case, though five cases is coarser than seven classes |
| **Go** | `fmt.Errorf` with a message; one sentinel, `voice.ErrNoPadToken` | **no**, except for that one sentinel. `errors.Is` has nothing to match |
| **Rust** | `Result<T, String>` throughout | **no**. The error *is* the message |
| **JS** | `Error` with a message | **no** |

The dual inheritance on the Python side means a caller who has not read this
library still catches the right things: "you passed a bad value" is a
`ValueError` wherever it comes from.

## The conditions, and what each port does with them

Every row is a condition all five implementations detect. Where a port's cell
reads `n/a`, the condition cannot arise on that transport, because the surface
has no such field. Detection is still identical in the engine.

| condition | Python | Swift | Go / Rust / JS |
|---|---|---|---|
| language this build cannot preprocess | `UnsupportedLanguageError` (`NotImplementedError`) | `.manifest` | message |
| text longer than one window, chunking off | `WindowOverflowError` (`ValueError`) | `.shape` | message |
| speech token id out of range | `InvalidTokensError` (`ValueError`), carrying `token` and `limit` | `.shape` | message |
| speech token that is not a whole number | `InvalidTokensError` | n/a (`Int` by type) | n/a (`i64` by type) |
| empty token sequence | `InvalidTokensError` | `.shape` | message |
| text the funnel emptied | `NothingToSpeakError` (`ValueError`) | `.shape` | message |
| voice name not in the library | `VoiceNotFoundError` (`FileNotFoundError`) | `.asset` | message |
| voice file over 8 MB | `ValueError` | `.asset` | message |
| voice profile dimensions wrong | `ValueError` | `.shape` | message |
| manifest dimensions the weights cannot fill | `ValueError` | `.manifest` | message |
| `max_new_tokens` not positive | `ValueError` | `.manifest` | message |
| sample rate not positive | `ValueError` | `.shape` | message |
| unknown postprocess mode | `ValueError` | `.manifest` | message |
| `chunking.max_tokens` not positive | `ValueError` | `.manifest` | message |
| number past the largest scale the grammar names | `NumberGrammarError` (`ValueError`) | `.shape` | message |
| C2PA manifest present and not verifiable | `ProvenanceError` (`ValueError`) | n/a | n/a |
| cancelled mid-render | returns nothing | `.cancelled` | message |

## Why errors differ by port

Two of the three "message" columns are idiomatic for their language. Changing
them would produce worse ports rather than better ones.

Go's convention is a message plus sentinels for the cases a caller branches on.
That is why `ErrNoPadToken` exists and no other sentinel does: nothing calls
for branching on the rest yet. Rust's `Result<T, String>` is the choice most open to
criticism. An enum with `thiserror` is the idiom, and `String` throws away the
distinction at the boundary where a caller would use it. JS has no error
taxonomy of its own.

One thing here *is* a defect, and this page exists to make it visible. A Go
caller embedding the library gets a string where the same condition over HTTP
gets a `400` and a `voice_not_found`. The information exists, and four of five
ports discard it before the caller sees it. The catalog above is the target
vocabulary for closing that gap. A port that grows typed errors names them with
these codes, not with new ones.

## If you are choosing a language

Python if you want to branch on failure. Swift if you want to branch coarsely.
Go, Rust and JS if failures are things you log and abort on, which is usually
sufficient for a synthesis call in a request handler.

The upgrade path, when somebody needs it: Rust to a `thiserror` enum, Go to
sentinels per condition, JS to subclasses of `Error`. Each carries the catalog
code for its condition. The conditions are enumerated above, the names are
frozen, and every one of them is already tested, so the work is wiring, not
discovery.
