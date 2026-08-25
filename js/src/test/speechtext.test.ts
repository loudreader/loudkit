/**
 * The speech funnel, against the fixture every port is checked with.
 *
 * Hand-written cases in five languages are five tests of five different
 * things. `tests/data/conformance/speechtext.json` is one test of one thing,
 * and a disagreement names itself. That file's own note says so — "Every port
 * must reproduce these exactly; a difference is a divergence, not a dialect".
 * All the bindings read
 * the `chunking` section and hand-write their funnel expectations; hand-written
 * expectations alone are how three separate divergences (an uppercase language
 * tag, non-ASCII digits, a typographic apostrophe sliced by bytes) can stay
 * green in all three at once.
 */

import test from "node:test";
import assert from "node:assert";
import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { speechText } from "../speechText.js";

function fixturePath(name: string): string {
  if (process.env.LOUDKIT_FIXTURE_DIR)
    return join(process.env.LOUDKIT_FIXTURE_DIR, name);
  let dir = dirname(fileURLToPath(import.meta.url));
  for (;;) {
    const candidate = join(dir, "tests", "data", "conformance", name);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  throw new Error(`cannot locate ${name}: set LOUDKIT_FIXTURE_DIR or run from the loudkit repo`);
}

interface Case {
  text: string;
  language?: string | null;
  expected: string;
}

test("the funnel matches the shared fixture", () => {
  const cases: Case[] = JSON.parse(
    readFileSync(fixturePath("speechtext.json"), "utf8")
  ).cases;
  // A renamed key would leave this loop comparing nothing and reporting a
  // pass, which is the failure this whole file exists to prevent.
  assert.ok(cases.length > 0, "the fixture has no cases; nothing was compared");

  const bad: string[] = [];
  for (const c of cases) {
    const got = speechText(c.text, c.language ?? undefined);
    if (got !== c.expected) {
      bad.push(
        `  text=${JSON.stringify(c.text)} lang=${JSON.stringify(c.language ?? null)}\n` +
          `    want ${JSON.stringify(c.expected)}\n    got  ${JSON.stringify(got)}`
      );
    }
  }
  assert.equal(
    bad.length,
    0,
    `the funnel disagrees with the shared fixture in ${bad.length}/${cases.length} cases:\n` +
      bad.join("\n")
  );
});
