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
import { fileURLToPath, pathToFileURL } from "node:url";
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
    // npm's own entry point, run by this Node, rather than the `npm` command.
    // On Windows that command is `npm.cmd`, which `execFileSync` will not find
    // by name (`spawnSync npm ENOENT`) and, since Node 20.12, will not spawn as
    // a batch file either (`spawnSync npm.cmd EINVAL`). A shell would run it
    // and would also reinterpret the arguments, one of which is a path. The
    // script's own launcher exports where npm lives, and this suite runs under
    // `npm test`.
    const cli = process.env.npm_execpath;
    const [command, head] = cli ? [process.execPath, [cli]] : ["npm", []];
    const out = execFileSync(
      command,
      [...head, "pack", "--ignore-scripts", "--pack-destination", work, "--json"],
      { cwd: root, encoding: "utf8", env }
    );
    // Named relative to `work` rather than absolutely, because GNU tar reads
    // a colon in an archive name as a `host:path` remote spec: handed
    // `C:\Users\...\loudkit-0.1.1.tgz` it tries to reach a machine called
    // `C` and reports "Cannot connect to C: resolve failed". `--force-local`
    // fixes that and is GNU-only, so bsdtar, which is what macOS ships,
    // refuses the flag instead. A bare filename has no colon in it and every
    // tar reads it the same way.
    const tarball = JSON.parse(out)[0].filename as string;
    execFileSync("tar", ["-xzf", tarball], { cwd: work });
    const pkg = join(work, "package");
    assert.ok(
      existsSync(join(pkg, "data", "numerals.json")),
      "the tarball ships no numeral table"
    );

    // Run from the extracted tree, whose only data/ is the one just unpacked.
    //
    // The specifier is a `file://` URL, not the path. ESM accepts only file,
    // data and node URLs, and on Windows an absolute path begins with a drive
    // letter, so `C:\...\speechText.js` is read as the scheme `c:` and the
    // import fails before the module is opened. `pathToFileURL` is the one
    // spelling that is a valid specifier on every platform.
    const probe = join(work, "probe.mjs");
    writeFileSync(
      probe,
      `import { speechText } from ${JSON.stringify(
        pathToFileURL(join(pkg, "dist", "speechText.js")).href
      )};\n` +
        `process.stdout.write(speechText("Add \\u0663 cups", "en"));\n`
    );
    // This Node, by path, not whichever `node` the PATH resolves to: the
    // probe imports the tree this suite just packed and has to run under the
    // interpreter the suite is running under.
    const said = execFileSync(process.execPath, [probe], { encoding: "utf8" });
    assert.equal(said, "Add three cups");
  } finally {
    // A tarball plus an unpacked copy plus a cache is about 11 MB, and the
    // suite runs on every commit.
    rmSync(work, { recursive: true, force: true });
  }
});
