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
| `window_overflow` | a single window's generation did not fit; reached from Python only through `synthesize(single_window=True)` and `synthesize_tokens` | `WindowOverflowError` |
| `audio_not_found` | an enrollment audio path that is not there | `AudioNotFoundError` |
| `provenance_invalid` | a provenance manifest present and not verifiable | `ProvenanceError` |
| `invalid_request` | the text funnel emptied the request, an audio format the loaded libsndfile cannot write, or any other refusal a class does not name yet | `NothingToSpeakError`, `UnsupportedFormatError`, or a bare `ValueError` at a boundary |
| `cancelled` | the caller's `should_cancel` returned true during `synthesize`; `stream` ends without raising | `CancelledError` |

Conditions that exist only at a transport boundary carry their own codes:
`server_fault` (not a refusal: a stub method, a dependency failure, a bug;
anything outside `LoudkitError`), `unauthorized` (401), `bad_host` (a public bind refused, 403),
`payload_too_large` (413), `rate_limited` (429), `busy` (queue full or engine
wedged, 503), `timeout` (a stream held the engine past its bound).

## Where each transport says it

| transport | where the code travels |
|---|---|
| Python | `exc.code` on every `LoudkitError`; `loudkit.errors.error_code(exc)` reads the class and returns `invalid_request` for anything outside it. `server_fault` is the transport's word, applied to an exception the boundary did not classify as a refusal |
| HTTP | `"code"` in every JSON error body, beside `detail` |
| SSE stream | `"error_code"` on the terminal `done` event, beside `error_kind` |
| gRPC | `loudkit-error-code` trailing metadata, beside the status code |
| JS | `errorCode(error)`, the same word the transports send |
| Go / Rust / Swift | not yet; see the upgrade path below |

## The shapes, per language

| | how failure is expressed | can a caller branch on it? |
|---|---|---|
| **Python** | ten raised classes under `LoudkitError` in `loudkit.errors`, nine of them also a stdlib type: `InvalidTokensError(LoudkitError, ValueError)`, `VoiceNotFoundError(LoudkitError, FileNotFoundError)`, and so on. `CancelledError` is the one that is a `LoudkitError` and nothing else | **yes**, by class or by stdlib supertype. `except ValueError` catches the request errors, and `except LoudkitError` catches those ten and nothing else. It is not a catch-all: the package also raises 208 bare `ValueError` and 56 other stdlib exceptions, so a caller that must not crash still catches `Exception` |
| **Swift** | one enum, `LoudKitError`, with six cases: `.manifest`, `.asset`, `.shape`, `.prediction`, `.cancelled`, `.windowOverflow(tokens:window:)` | **yes**, by case, though six cases is coarser than ten classes |
| **Go** | `fmt.Errorf` with a message; two sentinels, `loudkit.ErrCancelled` and `voice.ErrNoPadToken` | **no**, except for those two. `errors.Is` has nothing else to match |
| **Rust** | `Result<T, String>` throughout | **no**. The error *is* the message |
| **JS** | eight classes under `LoudkitError`, two of them also stdlib types (`LoudkitRangeError extends RangeError`, and `InvalidTokensError` under it), plus `errorCode(error)`. Most failures are still a plain `Error`: 21 throws are typed and 180 are not | **partly**. The typed conditions branch by class; everything else arrives as `Error`, and `errorCode` answers `invalid_request` for it |

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
| a single window's generation did not fit | `WindowOverflowError` (`ValueError`), carrying `n_tokens` and `window` | `.windowOverflow(tokens:window:)`, carrying the same two | message |
| speech token id out of range | `InvalidTokensError` (`ValueError`), carrying `token` and `limit` | `.shape` | message |
| speech token that is not a whole number | `InvalidTokensError` | n/a (`Int` by type) | n/a (`i64` by type) |
| empty token sequence | `InvalidTokensError` | `.shape` | message |
| text the funnel emptied | `NothingToSpeakError` (`ValueError`) | `.shape` | message |
| voice name not in the library | `VoiceNotFoundError` (`FileNotFoundError`) | `.asset` | message |
| enrollment audio path not there | `AudioNotFoundError` (`FileNotFoundError`) | n/a (enrolls from samples) | n/a (enrolls from samples) |
| voice file over 8 MB | `ValueError` | `.asset` | message |
| voice profile dimensions wrong | `ValueError` | `.shape` | message |
| manifest dimensions the weights cannot fill | `ValueError` | `.manifest` | message |
| `max_new_tokens` not positive | `ValueError` | `.manifest` | message |
| sample rate not positive | `ValueError` | `.shape` | message |
| unknown postprocess mode | `ValueError` | `.manifest` | message |
| `chunking.max_tokens` not positive | `ValueError` | `.manifest` | message |
| number past the largest scale the grammar names | `NumberGrammarError` (`ValueError`) | `.shape` | message |
| provenance manifest present and not verifiable | `ProvenanceError` (`ValueError`) | n/a | n/a |
| cancelled mid-render (`synthesize`; `stream` ends quietly) | `CancelledError` | `.cancelled` | Go `ErrCancelled` (`errors.Is`); Rust `error::CANCELLED` (`error::is_cancelled`); JS `CancelledError` |

## Why errors differ by port

Two of the three "message" columns are idiomatic for their language. Changing
them would produce worse ports rather than better ones.

Go's convention is a message plus sentinels for the cases a caller branches on.
That is why `ErrNoPadToken` and `ErrCancelled` exist and no other sentinel
does: nothing calls for branching on the rest yet. Rust's `Result<T, String>` is the choice most open to
criticism. An enum with `thiserror` is the idiom, and `String` throws away the
distinction at the boundary where a caller would use it. JS is the port that
started the upgrade: its classes mirror Python's and `errorCode` returns the
same word the HTTP and gRPC bodies carry, but only twenty-one of its throws
use them, so most conditions still reach a caller as a bare `Error`.

One thing here *is* a defect, and this page exists to make it visible. A Go
caller embedding the library gets a string where the same condition over HTTP
gets a `400` and a `voice_not_found`. The information exists, and four of five
ports discard it before the caller sees it. The catalog above is the target
vocabulary for closing that gap. A port that grows typed errors names them with
these codes, not with new ones.

## If you are choosing a language

Python if you want to branch on failure. Swift if you want to branch
coarsely, JS for the conditions it types. Go and Rust if failures are things
you log and abort on, which is usually sufficient for a synthesis call in a
request handler.

The upgrade path, when somebody needs it: Rust to a `thiserror` enum, Go to
sentinels per condition, JS to its own classes at the throws that still raise
a bare `Error`. Each carries the catalog code for its condition. The conditions are enumerated above, the names are
frozen, and every one of them is already tested, so the work is wiring, not
discovery.
