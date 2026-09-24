# Error codes and exceptions

The tables below list what each of the five implementations raises for each
condition. The error types differ by language, so a `catch` written for one
port does not carry over to another.

loudkit has one **catalog of error codes**: short, stable strings that name
each condition. Codes are never renamed or reused, only added. The catalog is
defined in `loudkit.errors`. Python, Swift, JS and every transport report
these codes, so one refusal has one code whether it arrives as an exception,
an HTTP body or a gRPC status. Go and Rust report no codes. A port that adds
typed errors uses the codes of this catalog.

## The catalog

| code | condition | Python class |
|---|---|---|
| `invalid_tokens` | a speech token id outside the codebook, negative, fractional, or an empty sequence | `InvalidTokensError` |
| `unsupported_language` | a language this build's frontend cannot preprocess | `UnsupportedLanguageError` |
| `voice_not_found` | a voice name the library does not have | `VoiceNotFoundError` |
| `number_grammar` | a number past the largest scale the language's grammar names | `NumberGrammarError` |
| `window_overflow` | a single window's generation did not fit; reached from Python only through `synthesize(single_window=True)` and `synthesize_tokens` | `WindowOverflowError` |
| `audio_not_found` | an enrollment audio path that is not there | `AudioNotFoundError` |
| `provenance_invalid` | a provenance manifest that is present but unreadable or malformed | `ProvenanceError` |
| `invalid_request` | the text funnel emptied the request, an audio format the loaded libsndfile cannot write, or another refusal with no class of its own | `NothingToSpeakError`, `UnsupportedFormatError`, or a bare `ValueError` at a boundary |
| `cancelled` | the caller's `should_cancel` returned true during `synthesize`; `stream` ends without raising | `CancelledError` |

Conditions that exist only at a transport boundary have their own codes:

- `server_fault` (500) is not a refusal: a stub method, a dependency failure
  or a bug. It covers any exception that the boundary does not classify as a
  refusal.
- `unauthorized` (401): a missing or wrong bearer token.
- `bad_host` (403): a `Host` header that is not loopback, sent to a
  loopback-only server.
- `payload_too_large` (413).
- `rate_limited` (429).
- `busy` (503): the queue is full or the engine is wedged.
- `timeout`: a stream held the engine past its time limit.

## Where each transport says it

| transport | where the code travels |
|---|---|
| Python | `exc.code` on every `LoudkitError`; `loudkit.errors.error_code(exc)` reads the class and returns `invalid_request` for anything outside it. `server_fault` is the transports' code for an exception the boundary did not classify as a refusal |
| HTTP | `"code"` beside `detail` in every native JSON error body. The OpenAI-compatible route answers in OpenAI's envelope (`error.message`, with `error.code` set to `null`). Two kinds of error on that route keep `detail` and `code`: the authentication and limit refusals in front of it, and `422` schema errors such as a missing `voice` (`code` is `invalid_request`). See [Server and agents](../guides/04-server-and-agents.md) |
| SSE stream | `"error_code"` on the terminal `done` event, beside `error_kind` |
| gRPC | `loudkit-error-code` trailing metadata, beside the status code, on every refusal the server decides. gRPC core refuses a request message over 256 KiB without it |
| JS | `errorCode(error)` |
| Swift | `LoudKitError.code`: `cancelled`, `window_overflow`, `voice_not_found`, `unsupported_language` or `invalid_tokens`, and `invalid_request` otherwise |
| Go / Rust | no error codes |

## The shapes, per language

| | how failure is expressed | can a caller branch on it? |
|---|---|---|
| **Python** | ten raised classes under `LoudkitError` in `loudkit.errors`. Nine of them are also a stdlib type: six are `ValueError`s, `VoiceNotFoundError` and `AudioNotFoundError` are `FileNotFoundError`s, and `UnsupportedLanguageError` is a `NotImplementedError`. `CancelledError` is only a `LoudkitError` | **yes**, by class or by stdlib type. `except LoudkitError` catches those ten and nothing else. In 0.1.1 the package also has 208 bare `ValueError` raise sites and 56 other stdlib ones, so a caller that must not crash still catches `Exception` |
| **Swift** | one enum, `LoudKitError`, with six cases: `.manifest`, `.asset`, `.shape`, `.prediction`, `.cancelled`, `.windowOverflow(tokens:window:)` | **yes**, by case or by `.code`. Six cases are coarser than the ten Python classes |
| **Go** | `fmt.Errorf` with a message, and two sentinels: `loudkit.ErrCancelled` and `windowing.ErrNoPadToken` | **no**, except for those two, with `errors.Is` |
| **Rust** | `Result<T, String>` throughout | **no**. The error is the message; `error::is_cancelled` detects cancellation |
| **JS** | two roots. `LoudkitError extends Error` has six subclasses: `CancelledError`, `NothingToSpeakError`, `NumberGrammarError`, `UnsupportedLanguageError`, `VoiceNotFoundError` and `WindowOverflowError`. `LoudkitRangeError extends RangeError` has one: `InvalidTokensError`. `errorCode(error)` covers both roots. In 0.1.1, 16 throw sites use these classes and 186 throw a plain `Error` or `RangeError` | **partly**. The typed conditions branch by class. Everything else arrives as `Error` or `RangeError`, and `errorCode` returns `invalid_request` for it |

With the dual inheritance, a Python stdlib handler catches the typed errors
too. For example, `except ValueError` catches `InvalidTokensError`, and
`except FileNotFoundError` catches `VoiceNotFoundError`.

## The conditions, and what each port does with them

Each row is one condition. A cell that reads `n/a` means that port has no
input or call where the condition can arise.

| condition | Python | Swift | Go / Rust | JS |
|---|---|---|---|---|
| language this build cannot preprocess | `UnsupportedLanguageError` (`NotImplementedError`) | `.asset` | message | `UnsupportedLanguageError` |
| a single window's generation did not fit | `WindowOverflowError` (`ValueError`), carrying `n_tokens` and `window` | `.windowOverflow(tokens:window:)`, carrying the same two | message | `WindowOverflowError` |
| speech token id out of range | `InvalidTokensError` (`ValueError`), carrying `token` and `limit` | `.shape` | message | `InvalidTokensError` |
| speech token that is not a whole number | `InvalidTokensError` | n/a (`Int` by type) | n/a (integer by type) | `InvalidTokensError` |
| empty token sequence | `InvalidTokensError` | `.shape` | message | message |
| text the funnel emptied | `NothingToSpeakError` (`ValueError`) | `.shape` | message | `NothingToSpeakError` |
| voice name not in the library | `VoiceNotFoundError` (`FileNotFoundError`) | `.asset` | message | `VoiceNotFoundError` |
| enrollment audio path not there | `AudioNotFoundError` (`FileNotFoundError`) | `.asset` | Go: the `os.Open` error, which matches `fs.ErrNotExist`; Rust: message | Node's `ENOENT` error |
| voice file over 8 MB | `ValueError` | `.asset` | message | message |
| voice profile dimensions wrong | `ValueError` | `.shape` | message | message |
| manifest dimensions the weights cannot fill | `ValueError` | `.manifest` | message | message |
| `max_new_tokens` not positive | `ValueError` | `.manifest` | message | message |
| sample rate not positive | `ValueError` | `.shape` | message | message |
| unknown postprocess mode | `ValueError` | `.manifest` | message | message |
| `chunking.max_tokens` not positive | `ValueError` | `.manifest` | message | message |
| number past the largest scale the grammar names | `NumberGrammarError` (`ValueError`) | `Numbers.cardinal` returns `nil` | message | `NumberGrammarError` |
| provenance manifest present but unreadable or malformed | `ProvenanceError` (`ValueError`) | n/a | n/a | n/a |
| cancelled mid-render (`synthesize`; `stream` ends quietly) | `CancelledError` | `.cancelled` | Go `ErrCancelled` (`errors.Is`); Rust `error::CANCELLED` (`error::is_cancelled`) | `CancelledError` |

## Where the ports differ

Go reports a message, plus a sentinel for each case a caller branches on:
`loudkit.ErrCancelled` and `windowing.ErrNoPadToken`. Rust returns
`Result<T, String>`, and `error::is_cancelled` detects cancellation. JS types
the conditions in the table above, and most other failures are a plain
`Error`.

A Go or Rust caller that embeds the library gets a message where HTTP returns
a status and a code for the same condition. For a missing voice, HTTP returns
`404` and `voice_not_found`.
