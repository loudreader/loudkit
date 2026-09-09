/**
 * The README quickstart and `examples/hello.mjs` are one file.
 *
 * A quickstart nobody runs rots. This one is `node examples/hello.mjs`, so the
 * block on the page has to be the script in the tarball, character for
 * character, or the reader copies one and runs the other.
 */

import assert from "node:assert";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

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

test("the README quickstart is examples/hello.mjs verbatim", () => {
  const readme = readFileSync(join(root, "README.md"), "utf8");
  const blocks = [...readme.matchAll(/```javascript\n([\s\S]*?)```/g)].map((m) => m[1]);
  assert.ok(blocks.length > 0, "the README shows no javascript block");
  const example = readFileSync(join(root, "examples", "hello.mjs"), "utf8");
  assert.ok(
    blocks.includes(example),
    "no javascript block in README.md is examples/hello.mjs word for word"
  );
});

test("the quickstart names only exports the package has", () => {
  const example = readFileSync(join(root, "examples", "hello.mjs"), "utf8");
  const imported = /^import \{([^}]*)\} from "loudkit";$/m.exec(example);
  assert.ok(imported, "examples/hello.mjs no longer imports from the package by name");
  const index = readFileSync(join(root, "src", "index.ts"), "utf8");
  for (const name of imported[1].split(",").map((s) => s.trim())) {
    assert.ok(
      new RegExp(`\\b${name}\\b`).test(index),
      `src/index.ts does not export ${name}, which the quickstart imports`
    );
  }
});
