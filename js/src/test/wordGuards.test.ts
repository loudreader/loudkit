/**
 * The funnel's word-boundary guards, against the reference's Unicode ones.
 *
 * ECMAScript `\w` and `\b` are ASCII even under the `u` flag, and Python's are
 * not. Every guard written that way expanded text the reference left written,
 * under the same `funnel-4` recipe and the same grammar digest: one string with
 * two readings and nothing to tell them apart. The expectations below are what
 * `expand_abbreviations` and `expand_dates` return for the same input.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { expandDates } from "../dates.js";
import { expandAbbreviations } from "../numbers.js";

test("an abbreviation glued to a non-ASCII letter stays written", () => {
  // `é` and `å` are word characters to Python and not to ECMAScript `\w`, so
  // these two read as *ézum Beispiel* and *åfrån och med* here alone.
  assert.equal(expandAbbreviations("éz.B. test", "de"), "éz.B. test");
  assert.equal(expandAbbreviations("åfr.o.m. test", "sv"), "åfr.o.m. test");
  // The ASCII neighbours the guard already refused, unchanged.
  assert.equal(expandAbbreviations("wz.B. test", "de"), "wz.B. test");
  assert.equal(expandAbbreviations("z.B.x test", "de"), "z.B.x test");
  // And the case the guard exists to admit.
  assert.equal(expandAbbreviations("z.B. test", "de"), "zum Beispiel test");
});

test("two abbreviations that touch are both read", () => {
  // The guards are lookarounds, so nothing is consumed between them. Consuming
  // the separator moved the scan past the second abbreviation and left it
  // written, where the reference expands both.
  assert.equal(expandAbbreviations("z.B. z.B.", "de"), "zum Beispiel zum Beispiel");
});

test("a written date glued to a non-ASCII letter stays written", () => {
  // Same guard, same failure: these read as dates here and as prose in Python,
  // Rust, Go and Swift.
  assert.equal(expandDates("ę2 marca 2026", "pl"), "ę2 marca 2026");
  assert.equal(expandDates("éMarch 12, 2026", "en"), "éMarch 12, 2026");
});

test("a non-ASCII letter behind the year drops the year, not the date", () => {
  // The right-hand guard refuses the year group and the pattern falls back to
  // the yearless match, which is what the reference does with the same input.
  assert.equal(expandDates("12 marca 2026ę", "pl"), "dwunastego marca 2026ę");
  assert.equal(expandDates("March 12, 2026é", "en"), "March twelfth 2026é");
});

test("the dates the guards admit are unchanged", () => {
  assert.equal(
    expandDates("2 marca 2026", "pl"),
    "drugiego marca dwa tysiące dwudziestego szóstego"
  );
  assert.equal(expandDates("March 12, 2026", "en"), "March twelfth twenty twenty-six");
});

test("a numeral that is not a digit is a word character too", () => {
  // Python's `\w` is `str.isalnum()` plus the underscore, which holds `\u00bd`
  // (No) and `\u2160` (Nl) as well as decimal digits. Guarding with `\p{Nd}`
  // alone left those two outside the class, so a date beside one read as a date
  // here and as prose in the reference.
  assert.equal(expandDates("12 marca 2026\u00bd", "pl"), "dwunastego marca 2026\u00bd");
  assert.equal(expandDates("\u216012 marca 2026", "pl"), "\u216012 marca 2026");
});
