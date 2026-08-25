/**
 * The number verbalizer against both shared corpora: the hand-written fixture
 * (expectations from each language's own reference description) and the CLDR
 * differential (1300 rows Unicode wrote; disputed rows skipped with reasons).
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";

import { cardinal, expandNumbers, expandTimes } from "../numbers.js";

 
function fixture(name: string): any {
  const p = join(
    import.meta.dirname, "..", "..", "..", "tests", "data", "conformance", name
  );
  return JSON.parse(readFileSync(p, "utf-8"));
}

test("cardinal matches the hand fixture", () => {
  const fx = fixture("numbers.json");
  const languages = Object.keys(fx.cardinals);
  assert.ok(languages.length > 0, "nothing was compared");
  for (const lang of languages) {
    for (const c of fx.cardinals[lang]) {
      assert.equal(cardinal(c.value, lang), c.expect, `${lang} ${c.value}`);
    }
  }
  for (const c of fx.gendered) {
    assert.equal(
      cardinal(c.value, c.language, c.gender),
      c.expect,
      `${c.language} ${c.value} g=${c.gender}`
    );
  }
});

test("cardinal matches CLDR", () => {
  const fx = fixture("numbers_cldr.json");
  let checked = 0;
  for (const [lang, cases] of Object.entries(fx.cases)) {
    for (const c of cases as any[]) {
      if (c.disputed) continue;
      let got: string;
      try {
        got = cardinal(c.value, lang, c.gender ?? "");
      } catch {
        continue; // past our scale: the refusal is the declared behaviour
      }
      checked += 1;
      assert.equal(got, c.expect, `${lang} ${c.value} g=${c.gender}`);
    }
  }
  assert.ok(checked > 1000, `only ${checked} CLDR rows ran`);
});

test("expandNumbers in running text", () => {
  const cases: [string, string, string][] = [
    ["I have 21 apples.", "en", "I have twenty-one apples."],
    ["3.5", "en", "three point five"],
    ["1,200", "en", "one thousand two hundred"],
    ["3,5", "pl", "trzy przecinek pięć"],
    ["Es kostet 250 Euro.", "de", "Es kostet zweihundertfünfzig Euro."],
    ["21 apples", "xx", "21 apples"],
    ["no numbers here", "en", "no numbers here"],
  ];
  for (const [text, lang, want] of cases) {
    assert.equal(expandNumbers(text, lang), want, text);
  }
});

// The word a digit run is glued to is a word in any script. JS `\w` is
// `[A-Za-z0-9_]` and stays ASCII even under the `u` flag, so the guards and the
// walks that were written with it saw no letter in `é2` and read it *étwo*
// while the other four ports left it written; `zł200 000` came out
// *złdoscientos mil*. Every case below is a fuzz divergence against Python.
test("a run glued to a letter is left written in every script", () => {
  const cases: [string, string, string][] = [
    ["é2", "en", "é2"],
    ["é3", "en", "é3"],
    // The backward walk crosses the grouping space and finds the letter two
    // characters further back, which is the walk's whole reason to exist.
    ["é.1 210 5.", "da", "é.1 210 fem."],
    ["zł200 000", "es", "zł200 000"],
    ["Ω2", "en", "Ω2"],
    ["ж2", "en", "ж2"],
    ["文2", "en", "文2"],
    // Not a refusal of everything near a letter: the `2` is its own token.
    ["ł0 2", "es", "ł0 dos"],
  ];
  for (const [text, lang, want] of cases) {
    assert.equal(expandNumbers(text, lang), want, text);
  }
});

// A maximal run of digits and separators that does not reduce to a single
// readable number is left written, and a ragged group inside one does not stop
// it from being that run: `1 0023R` read as *en 0023R* here and in Python while
// Go left the token whole. Half the run spoken with the rest welded to a letter
// is the reading the right-hand guard exists to refuse.
test("a ragged run glued to a letter is left written", () => {
  const cases: [string, string, string][] = [
    ["1 0023R", "da", "1 0023R"],
    ["1 0023x", "fr", "1 0023x"],
    ["1 1000d", "de", "1 1000d"],
    ["0 0001e", "fr", "0 0001e"],
    ["1 2345E", "fi", "1 2345E"],
    // Nothing glued: the ragged run is two numbers and both are read.
    ["1 000.0 3", "nl", "duizend komma nul drie"],
    // The letter is behind a space the walk must not cross: four digits are
    // not a group, so `1000` is not a continuation of `e3`.
    ["e3 1000", "sv", "e3 ettusen"],
    ["e6 1003", "it", "e6 milletre"],
    // Nor does the forward walk cross into the exponent two tokens away.
    ["Son 1000 5.1e+3 aqui.", "es", "Son mil 5.1e+3 aqui."],
    // One or two digits after the space are not a group either.
    ["R2 5 iOS.", "de", "R2 fünf iOS."],
    // A ragged run with no letter in it stays a sequence of numbers.
    [
      "Dial 1 202 555 0199 now.",
      "en",
      "Dial one two hundred and two five hundred and fifty-five zero one nine nine now.",
    ],
  ];
  for (const [text, lang, want] of cases) {
    assert.equal(expandNumbers(text, lang), want, text);
  }
});

// German writes the time with the word the spoken form also carries: the
// reading puts the infix between hour and minutes, so the written "Uhr"
// behind the digits is that same token and is consumed, not duplicated.
test("a written infix is not said twice", () => {
  const cases: [string, string][] = [
    ["um 14:30 Uhr", "um vierzehn Uhr dreißig"],
    // A tab before the word consumes exactly like a space.
    ["um 14:30\tUhr", "um vierzehn Uhr dreißig"],
    ["um 24:00 Uhr an.", "um vierundzwanzig Uhr an."],
    // The dotted form runs through the second pattern.
    ["Termin um 14.30 Uhr.", "Termin um vierzehn Uhr dreißig."],
    // Without the word nothing changes.
    ["um 14:30", "um vierzehn Uhr dreißig"],
    // The noun on its own is not part of any time.
    ["Es ist 14:30 Uhr und die Uhr tickt.", "Es ist vierzehn Uhr dreißig und die Uhr tickt."],
    // Infix inside a longer word keeps its head.
    ["Die Uhrzeit ist 14:30.", "Die Uhrzeit ist vierzehn Uhr dreißig."],
  ];
  for (const [text, want] of cases) {
    assert.equal(expandTimes(text, "de"), want, text);
  }
  // Eleven of the twelve grammars carry an empty infix: nothing to consume.
  assert.equal(expandTimes("at 14:30 sharp", "en"), "at fourteen thirty sharp");
});
