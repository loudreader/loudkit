/**
 * A digit run with two or more separators is a version, an address or a date —
 * never a number.
 *
 * Reading one as a number says the segments as one value: with the comma as the
 * decimal mark the dots are treated as thousands grouping and the segments
 * concatenate, so `192.168.0.1` is spoken as "nineteen million two hundred
 * sixteen thousand eight hundred one". The Python reference additionally
 * crashes on these.
 *
 * Every literal below is one that shipped wrong; the tests pin them.
 */

import test from "node:test";
import assert from "node:assert";

import { expandNumbers, expandTimes, supportedNumberLanguages } from "../numbers.js";

const NOT_QUANTITIES = ["1.2.3", "1.2.3.4", "192.168.0.1", "12.03.2026", "10.0.0.255"];

test("digits that are not quantities are left alone", () => {
  for (const lang of supportedNumberLanguages()) {
    for (const literal of NOT_QUANTITIES) {
      assert.equal(expandNumbers(literal, lang), literal, `${lang}: ${literal}`);
    }
  }
});

test("real numbers still read", () => {
  // The guard must not buy correctness by refusing everything.
  for (const lang of supportedNumberLanguages()) {
    for (const literal of ["7", "2,5", "2.5"]) {
      assert.notEqual(expandNumbers(literal, lang), literal, `${lang}: ${literal}`);
    }
  }
});

test("grouped thousands are still a number", () => {
  // The rule is "three digits after the first separator", not "at most one".
  for (const lang of supportedNumberLanguages()) {
    if (lang === "en") continue; // English groups with commas, not dots
    assert.notEqual(expandNumbers("1.234.567", lang), "1.234.567", lang);
  }
  assert.notEqual(expandNumbers("1,234,567", "en"), "1,234,567");
});

test("a time is not part of a date", () => {
  // `12.03` matches inside `12.03.2026`, so the ordinary written date of five
  // of the twelve languages would be spoken as a clock time with the year
  // trailing.
  for (const lang of supportedNumberLanguages()) {
    for (const literal of ["12.03.2026", "am 05.11.2025 kam"]) {
      assert.equal(expandTimes(literal, lang), literal, `${lang}: ${literal}`);
    }
    // A dotted time reads only where the dot is not the decimal point: `14.30`
    // is half past two in eleven of these languages and a number in the
    // twelfth. Asserting it for all twelve made every English decimal with two
    // fraction digits a clock time.
    if (lang !== "en") {
      assert.notStrictEqual(expandTimes("14.30", lang), "14.30", `${lang}: 14.30`);
    } else {
      assert.strictEqual(expandTimes("14.30", lang), "14.30", `${lang}: 14.30`);
    }
    for (const literal of ["14:30", "at 14:30."]) {
      assert.notEqual(expandTimes(literal, lang), literal, `${lang}: ${literal}`);
    }
  }
});
