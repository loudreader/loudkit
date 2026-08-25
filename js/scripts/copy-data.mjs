/**
 * Copy the shared data files from the Python source tree into data/.
 *
 * data/ is gitignored — the single source of truth for both files is
 * python/loudkit/models/data/ at the repository root, and this binding takes a
 * copy at build time so `npm pack` can ship it. Both copies are required:
 * numbers.ts imports data/numbers.json at compile time, so without this step
 * `tsc` fails on a fresh clone; pl_en_respell.json is loaded lazily and its
 * absence would silently degrade Polish respelling instead.
 *
 * Outside the monorepo (an unpacked tarball, a vendored copy) the source
 * tree does not exist; then existing files are left alone and missing ones
 * are an error, because nothing else can supply them.
 */
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
// `js/` sits one level under the repo root, and the Python tree has lived at
// `python/loudkit/` since the repository went one-directory-per-language. This
// used to read `"..", "..", "src", …`, which pointed two levels up — outside
// the repository entirely — at a `src/` layout that no longer exists. The
// source was therefore never reachable, so every fresh clone fell into the
// out-of-monorepo branch below and `npm test` died in `pretest`.
const sourceDir = join(root, "..", "python", "loudkit", "models", "data");
const files = ["numbers.json", "pl_en_respell.json"];

mkdirSync(join(root, "data"), { recursive: true });
for (const name of files) {
  const target = join(root, "data", name);
  const source = join(sourceDir, name);
  // Copy whenever the source is reachable, rather than only when the target is
  // missing. Skipping an existing target meant a stale copy was never
  // refreshed: `numbers.json` here sat two features behind the Python source
  // for a whole day, and nothing noticed, because the file existed.
  if (!existsSync(source)) {
    // Outside the monorepo an existing copy is all there is, and it is correct
    // — the tarball ships it. Only a missing copy with no source is fatal.
    if (existsSync(target)) continue;
    console.error(
      `data/${name} is missing and ${source} does not exist to copy from — ` +
        `run this from the loudkit monorepo, or restore data/ from the npm tarball.`,
    );
    process.exit(1);
  }
  copyFileSync(source, target);
  console.log(`copied ${name}`);
}
