// Hand the built test files to the runner by name.
//
// `node --test dist/test/*.test.js` relies on the shell expanding the glob,
// and npm runs scripts through cmd.exe on Windows, which does not: the runner
// is handed the literal pattern. Naming the files here is the one form that
// does not depend on the shell, and it does not depend on which Node versions
// accept a directory either.

import { readdirSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const built = join(here, "..", "dist", "test");

const files = readdirSync(built)
  .filter((name) => name.endsWith(".test.js"))
  .sort()
  .map((name) => join(built, name));

if (files.length === 0) {
  console.error(`no test files under ${built}; run tsc first`);
  process.exit(1);
}

const { status } = spawnSync(process.execPath, ["--test", ...files], {
  stdio: "inherit",
});
process.exit(status ?? 1);
