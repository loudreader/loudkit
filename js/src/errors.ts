/**
 * The errors this library raises on purpose, and what a caller can do with
 * them.
 *
 * Every class here carries a `code` from the same frozen catalog the Python
 * package writes out in `docs/reference/errors.md`: class-level and stable, so
 * a transport can send it and a caller in any language can branch on it.
 * Python reference: `loudkit/errors.py`.
 *
 * **Additive by construction.** Each class extends the built-in this port
 * already threw at that site and leaves `name` alone, so `err instanceof
 * Error`, `err instanceof RangeError`, `err.name` and `String(err)` are what
 * they were and no `catch` block has to learn anything. What is new is the
 * `code`, and it is a getter rather than a field so that it lives on the
 * prototype the way Python's class attribute does: `Object.keys(err)` is still
 * empty and `JSON.stringify(err)` is still `{}`, which is what a caller
 * serialising an error sees today. `errorCode` is how the code is read, in
 * both languages.
 *
 * That is also why there are two roots rather than one. Python spells the
 * mixed kinds with multiple inheritance (`NumberGrammarError` is a
 * `LoudkitError` and a `ValueError`); JS has no such thing, and between a
 * common base and the `RangeError` identity callers already hold, the identity
 * wins. `errorCode` is the branch point that covers both, exactly as Python's
 * `error_code` is.
 *
 * Two of Python's nine are absent, and deliberately: `AudioNotFoundError`,
 * because a missing recording surfaces here as Node's own ENOENT and wrapping
 * it would change the message a caller reads today, and `ProvenanceError`,
 * because this port has no provenance reader to raise it. Neither can be added
 * without moving something a caller already sees.
 *
 * The structured fields Python attaches (`n_tokens`, `window`, `token`,
 * `limit`, `ref`, `available`) are not here yet. The codes are what a transport
 * needs; the fields are a 0.1.2 item because every one of them is a new
 * constructor signature.
 */

/** The catalog code for a refusal that is simply a bad request. */
const DEFAULT_CODE = "invalid_request";

/** Base for every error loudkit raises deliberately. */
export class LoudkitError extends Error {
  get code(): string {
    return DEFAULT_CODE;
  }
}

/**
 * Base for the deliberate refusals that are also range errors.
 *
 * A separate root because JS classes have one parent: these sites threw
 * `RangeError` before this file existed, a test pins it, and taking that away
 * to give them a shared base would be the one thing this file promises not to
 * do.
 */
export class LoudkitRangeError extends RangeError {
  get code(): string {
    return DEFAULT_CODE;
  }
}

/** A number could not be said in the requested language. */
export class NumberGrammarError extends LoudkitError {
  override get code(): string {
    return "number_grammar";
  }
}

/** A language this build's text frontend cannot preprocess. */
export class UnsupportedLanguageError extends LoudkitError {
  override get code(): string {
    return "unsupported_language";
  }
}

/** No voice by that name or path. */
export class VoiceNotFoundError extends LoudkitError {
  override get code(): string {
    return "voice_not_found";
  }
}

/** A speech token sequence the caller supplied that the engine cannot use. */
export class InvalidTokensError extends LoudkitRangeError {
  override get code(): string {
    return "invalid_tokens";
  }
}

/**
 * The text funnel removed every character of the request.
 *
 * No code of its own, in Python either: emoji, bare symbols and invisible
 * marks are legal input at a transport and gone by the frontend, and what
 * reaches the caller is that the request was not one.
 */
export class NothingToSpeakError extends LoudkitError {}

/** More speech tokens than the renderer's window holds. */
export class WindowOverflowError extends LoudkitError {
  override get code(): string {
    return "window_overflow";
  }
}

/**
 * The catalog code for `error`, the one mapping every transport uses.
 *
 * Anything else is `invalid_request`, including a Node error that happens to
 * carry a `code` of its own: `ENOENT` is not in this catalog and reporting it
 * as though it were would make the catalog unclosed.
 */
export function errorCode(error: unknown): string {
  return error instanceof LoudkitError || error instanceof LoudkitRangeError
    ? error.code
    : DEFAULT_CODE;
}

/**
 * What a refusal calls the JSON value it was handed.
 *
 * The JSON type, not the host language's: `ManifestReader.jsonTypePhrase` in
 * the Swift port says the same words, and a refusal about the contents of a
 * file should name what is in the file.
 */
export function describeJson(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "an array";
  if (typeof value === "object") return "an object";
  if (typeof value === "string") return "a string";
  if (typeof value === "boolean") return "a boolean";
  return "a number";
}

/**
 * A JSON value spelled the way the reference's refusals quote it back, so the
 * same bad file reads the same in both.
 *
 * `JSON.parse` widens every number to a double, so the digits a file carried
 * are gone by the time a refusal quotes them: `5` and `5.0` both print as `5`.
 * Object keys are sorted, matching the other three ports.
 */
export function pyRepr(value: unknown): string {
  if (value === null || value === undefined) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") {
    // repr's own rule: single quotes unless the value holds one and no double
    // quote.
    if (value.includes("'") && !value.includes('"')) return `"${value}"`;
    return `'${value.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
  }
  if (Array.isArray(value)) return `[${value.map(pyRepr).join(", ")}]`;
  // Everything a JSON document can still hold here is an object, so this is
  // the last branch rather than one more guard with a stringify behind it.
  const block = value as Record<string, unknown>;
  const body = Object.keys(block)
    .sort()
    .map((k) => `${pyRepr(k)}: ${pyRepr(block[k])}`);
  return `{${body.join(", ")}}`;
}
