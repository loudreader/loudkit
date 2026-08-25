/**
 * Polish lexical respelling — bit-parity checks against the Python port.
 *
 * `lexicalRespelling` (wired into `speechText` for language="pl") rewrites
 * English embedded in Polish the way a Polish reader says it. The expected
 * values are the ones the Swift/Python ear tests approved; a drift here means
 * the JS reader and the Python/Swift readers disagree on Polish.
 */

import test from "node:test";
import assert from "node:assert";

import { lexicalRespelling } from "../respell.js";
import { spellAcronyms } from "../letters.js";
import { speechText } from "../speechText.js";

test("curated lexicon respells common anglicisms", () => {
  const cases: Array<[string, string]> = [
    ["download", "dałnloud"],
    ["deadline", "dedlajn"],
    ["feedback", "fidbek"],
    ["weekend", "łikend"],
    ["workflow", "łorkfloł"],
    ["release", "rilis"],
  ];
  for (const [word, want] of cases) {
    assert.equal(lexicalRespelling(word, "pl"), want, word);
  }
});

test("case is preserved", () => {
  assert.equal(lexicalRespelling("GitHub", "pl"), "Githab");
  assert.equal(lexicalRespelling("Download", "pl"), "Dałnloud");
});

test("phrases respell as a unit", () => {
  assert.equal(lexicalRespelling("release notes", "pl"), "rilis nołc");
  assert.equal(lexicalRespelling("pull request", "pl"), "pul rekłest");
  assert.equal(lexicalRespelling("code review", "pl"), "koud riwju");
});

test("only Polish is respelled", () => {
  assert.equal(lexicalRespelling("download", "en"), "download");
});

test("numbers become cardinals", () => {
  assert.equal(lexicalRespelling("0", "pl"), "zero");
  assert.equal(lexicalRespelling("1", "pl"), "jeden");
  assert.equal(lexicalRespelling("15", "pl"), "piętnaście");
  assert.equal(lexicalRespelling("101", "pl"), "sto jeden");
  assert.equal(lexicalRespelling("1234", "pl"), "tysiąc dwieście trzydzieści cztery");
});

test("decimals read whole comma fraction", () => {
  assert.equal(lexicalRespelling("2.5", "pl"), "dwa przecinek pięć");
});

// The respeller no longer owns this decision. It saw one word at a time, so it
// could not tell an initialism from a shout and spelled "TO JEST WAŻNE" letter
// by letter. `spellAcronyms` decides for all twelve languages while the
// surrounding capitals are still visible; the respeller now sees the
// already-spelled lowercase form and leaves it alone.
test("acronyms are spelled earlier in the funnel now", () => {
  assert.equal(spellAcronyms("GPT", "pl"), "gie-pe-te");
  assert.equal(spellAcronyms("USB", "pl"), "u-es-be");
  // word-acronyms keep their word form
  assert.equal(spellAcronyms("NASA", "pl"), "nasa");
  assert.equal(spellAcronyms("PIN", "pl"), "pin");
  // and the whole funnel still produces the Polish letter names
  assert.ok(speechText("Model GPT jest dobry.", "pl").includes("gie-pe-te"));
  // a run of capitals is emphasis, and the respeller must not undo that
  assert.equal(speechText("CIA CIA", "pl"), "CIA CIA");
});

test("English word alone stays Polish, in a span transliterates", () => {
  assert.equal(lexicalRespelling("brown", "pl"), "brown");
  assert.equal(lexicalRespelling("the quick brown fox", "pl"), "da kłyk brałn faks");
});

test("inflection via stem", () => {
  assert.equal(lexicalRespelling("update", "pl"), "apdejt");
  assert.equal(lexicalRespelling("updates", "pl"), "apdejc");
  assert.equal(lexicalRespelling("deadline'u", "pl"), "dedlajnu");
  // Endings longer than one character: POLISH_ENDINGS was a set of single
  // characters (see the word-list test below), so only `deadline'u` — whose
  // ending happens to be the single char `u` — passed. `em` and `a` are the
  // two the Python suite pins.
  assert.equal(lexicalRespelling("updatem", "pl"), "apdejtem");
  assert.equal(lexicalRespelling("mailem", "pl"), "mailem");
});

test("the word lists are lists of words, not sets of characters", () => {
  // `new Set("a b " + "c d".split(" "))` binds `.split` to the second literal
  // alone: `string + Array` stringifies, and `new Set(string)` iterates
  // CHARACTERS. Three of the four lists were built that way, which is
  // invisible from any single respelling — so the sizes are asserted, and
  // then the outputs that actually moved.
  assert.equal(lexicalRespelling("host", "pl"), "host", "KEEP_POLISH is dead");
  assert.equal(lexicalRespelling("python", "pl"), "python", "KEEP_POLISH is dead");
  // With no Polish function words in the set, an English span is never broken
  // by one, so a four-word run swallows `tak`/`nie` and reads native Polish
  // as English.
  assert.equal(lexicalRespelling("tak done nie not", "pl"), "tak done nie not");
});

test("Polish words are untouched", () => {
  assert.equal(lexicalRespelling("temperatura", "pl"), "temperatura");
  assert.equal(lexicalRespelling("piątku", "pl"), "piątku");
});

test("full sentence matches the Python funnel", () => {
  assert.equal(
    speechText("Pobierz download i zrób code review.", "pl"),
    "Pobierz dałnloud i zrób koud riwju."
  );
  assert.equal(
    speechText("Rabat 15% na weekend!", "pl"),
    "Rabat piętnaście procent na łikend!"
  );
  assert.equal(
    speechText("The quick brown fox jumps over the lazy dog.", "pl"),
    "Da kłyk brałn faks dżamps ołwer da lejzi dog."
  );
});

test("lexicon loads from the packaged data", () => {
  assert.equal(lexicalRespelling("download", "pl"), "dałnloud");
  // The generated long tail, with a real expectation: an identity assertion
  // (`f("queue") === f("queue")`) would hold with the 6.5 MB lexicon deleted —
  // a test of nothing, standing where the check on the packaged data belongs.
  // Rust's tests/respell.rs:112 asserts it the same way.
  assert.equal(lexicalRespelling("queue", "pl"), "kju");
  assert.equal(lexicalRespelling("commit", "pl"), "komit");
});
