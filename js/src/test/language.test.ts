/**
 * The obvious call must not be the wrong one.
 *
 * Without the voice link, `engine.synthesize("Cześć", polishVoice, 7)` runs
 * Polish text through the English frontend: `language` defaults to `"en"` and a
 * profile's own `language` — recorded at enrollment — is never consulted. The
 * chain is argument, then voice, then `"en"`, and these are its links.
 *
 * Tested against the resolver rather than through `synthesize` because this
 * port has no weight-free engine seam: `Engine` has a private constructor and
 * six ONNX `Session`s, so nothing can drive the pipeline without a checkpoint
 * and a runtime. The resolver is the whole of the behaviour under test.
 */
import test from "node:test";
import assert from "node:assert";

import { resolveLanguage } from "../engine.js";

test("a Polish voice reads Polish by default", () => {
  assert.strictEqual(resolveLanguage(undefined, { language: "pl" }), "pl");
});

test("an explicit language overrides the profile", () => {
  assert.strictEqual(resolveLanguage("en", { language: "pl" }), "en");
});

test("a profile without a language falls back to English", () => {
  // A hand-built profile can carry an empty language, and an empty language id
  // is not a language — it would tag the text `[]`. A header that simply omits
  // the key loads as "en" instead, so it never reaches this branch.
  assert.strictEqual(resolveLanguage(undefined, { language: "" }), "en");
});
