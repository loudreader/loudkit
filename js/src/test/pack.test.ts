/**
 * What a recipient actually gets.
 *
 * `data/` is gitignored: it is copied out of the Python tree by
 * `scripts/copy-data.mjs` at build time. That makes a missing entry in the
 * copier invisible here, because every machine that ever built the package
 * still has the file an earlier build left behind. `numerals.json` was read
 * unconditionally by the funnel and copied by nobody, and `npm test` was green
 * on this machine while a fresh `git archive` died with `ENOENT` on the first
 * non-ASCII numeral.
 *
 * So these tests ask the two questions the suite could not: does the copier
 * name every data file the source reads, and does a numeral fold in a process
 * whose only data directory is the one inside the tarball.
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import assert from "node:assert";

// This file runs compiled, from `dist/test/`, so the package root is found by
// walking up to the `package.json` rather than by counting directories.
function packageRoot(): string {
  let dir = dirname(fileURLToPath(import.meta.url));
  while (!existsSync(join(dir, "package.json"))) {
    const up = dirname(dir);
    if (up === dir) throw new Error("no package.json above " + import.meta.url);
    dir = up;
  }
  return dir;
}

const root = packageRoot();
const src = join(root, "src");

/** Every `data/<name>.json` any source file opens or imports. */
function dataFilesTheSourceReads(): Set<string> {
  const names = new Set<string>();
  for (const entry of readdirSync(src)) {
    if (!entry.endsWith(".ts")) continue;
    const text = readFileSync(join(src, entry), "utf8");
    for (const m of text.matchAll(/data\/([A-Za-z0-9_]+\.json)/g)) names.add(m[1]);
  }
  return names;
}

test("the copier and the pack check name every data file the source reads", () => {
  const needed = dataFilesTheSourceReads();
  assert.ok(needed.size >= 2, "no data files found in src/, so the scan is broken");
  // The `files` array, not the whole file: both scripts *discuss*
  // `numerals.json` in their comments, so a substring search over the source
  // passed while the array itself was two entries long.
  const copier = readFileSync(join(root, "scripts", "copy-data.mjs"), "utf8");
  const copied = /const files = \[([^\]]*)\]/.exec(copier);
  assert.ok(copied, "scripts/copy-data.mjs no longer declares a `files` array");
  const packCheck = readFileSync(join(root, "scripts", "check-pack.mjs"), "utf8");
  for (const name of needed) {
    assert.ok(copied[1].includes(`"${name}"`), `scripts/copy-data.mjs does not copy data/${name}`);
    assert.ok(packCheck.includes(name), `scripts/check-pack.mjs does not check data/${name}`);
  }
});

test("a non-ASCII numeral folds inside the packed tarball", () => {
  const work = mkdtempSync(join(tmpdir(), "loudkit-pack-"));
  try {
    // Everything this test writes goes under `work`, the npm cache included.
    // `npm pack` otherwise reaches into the user's `~/.npm/_cacache`, which is
    // shared with every other npm on the machine: on a developer box with a
    // cache written by another user, or by root, this failed with `EPERM` and
    // took the whole `npm test` run down with it. A test that touches shared
    // state is a test that fails for reasons that have nothing to do with the
    // code.
    const env = { ...process.env, npm_config_cache: join(work, "npm-cache") };
    // `--ignore-scripts`, so this packs what is on disk rather than rebuilding:
    // the suite has already built dist/, and the question here is what the
    // `files` list carries out of this directory.
    const out = execFileSync(
      "npm",
      ["pack", "--ignore-scripts", "--pack-destination", work, "--json"],
      { cwd: root, encoding: "utf8", env }
    );
    const tarball = join(work, JSON.parse(out)[0].filename as string);
    execFileSync("tar", ["-xzf", tarball, "-C", work]);
    const pkg = join(work, "package");
    assert.ok(
      existsSync(join(pkg, "data", "numerals.json")),
      "the tarball ships no numeral table"
    );

    // Run from the extracted tree, whose only data/ is the one just unpacked.
    const probe = join(work, "probe.mjs");
    writeFileSync(
      probe,
      `import { speechText } from ${JSON.stringify(join(pkg, "dist", "speechText.js"))};\n` +
        `process.stdout.write(speechText("Add \\u0663 cups", "en"));\n`
    );
    const said = execFileSync("node", [probe], { encoding: "utf8" });
    assert.equal(said, "Add three cups");
  } finally {
    // A tarball plus an unpacked copy plus a cache is about 11 MB, and the
    // suite runs on every commit.
    rmSync(work, { recursive: true, force: true });
  }
});
