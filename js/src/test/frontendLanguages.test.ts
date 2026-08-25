/**
 * A tag the tokenizer knows is not a language the kit can speak.
 *
 * The vocabulary carries tags for 31 languages; the text layer is written for
 * twelve. A blacklist of only zh/ja/he/ko/ru lets the other 26 go
 * straight through: `encode(text, "bg")` NFKD-mangles Cyrillic into ids the
 * model reads as sounds it never learned — no error, plausible-sounding audio,
 * wrong language.
 *
 * The roster is asserted against the number grammars rather than a literal
 * list: `numbers.json` is the one authority, and a port with its own copy is a
 * port that disagrees with Python the next time a grammar is added.
 */
import test from "node:test";
import assert from "node:assert";

import { supportedLanguages } from "../frontend.js";
import { supportedNumberLanguages } from "../numbers.js";

test("the roster is the twelve in numbers.json", () => {
  const roster = supportedLanguages();
  assert.deepStrictEqual(roster, supportedNumberLanguages());
  assert.strictEqual(roster.length, 12);
});

test("on-roster languages are accepted, off-roster tags are not", () => {
  const roster = supportedLanguages();
  for (const lang of ["en", "pl", "sv"]) {
    assert.ok(roster.includes(lang), `${lang} is on the roster`);
  }
  // Cyrillic and Czech are tokenizer tags, never languages this build speaks.
  for (const lang of ["bg", "cs", "zh"]) {
    assert.ok(!roster.includes(lang), `${lang} must be refused`);
  }
});
