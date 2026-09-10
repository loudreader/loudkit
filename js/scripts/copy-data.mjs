/**
 * Copy the shared data files from the Python source tree into data/.
 *
 * data/ is gitignored: the single source of truth for every file here is
 * python/loudkit/models/data/ at the repository root, and this binding takes a
 * copy at build time so `npm pack` can ship it. All three are required:
 * numbers.ts imports data/numbers.json at compile time, so without this step
 * `tsc` fails on a fresh clone; numerals.json is read on the first fold and its
 * absence is an `ENOENT` on any text carrying a non-ASCII numeral;
 * pl_en_respell.json is loaded lazily and its absence would silently degrade
 * Polish respelling instead.
 *
 * The list is not a place to be economical. data/ is gitignored, so a file
 * added to the funnel and not to this array goes on working from the copy an
 * earlier build left on a developer's machine, and fails only on a fresh
 * checkout. `check-pack.mjs` names each file for the same reason.
 *
 * Outside the monorepo (an unpacked tarball, a vendored copy) the source
 * tree does not exist; then existing files are left alone and missing ones
 * are an error, because nothing else can supply them.
 */
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
// `js/` sits one level under the repo root and the Python tree at
// `python/loudkit/`, so exactly one `..` separates them. A path that climbs one
// level too far leaves the repository, which is not an error here: the source is
// then unreachable, every fresh clone falls into the out-of-monorepo branch
// below, and `npm test` dies in `pretest`.
const sourceDir = join(root, "..", "python", "loudkit", "models", "data");
const files = ["numbers.json", "numerals.json", "pl_en_respell.json"];

mkdirSync(join(root, "data"), { recursive: true });
for (const name of files) {
  const target = join(root, "data", name);
  const source = join(sourceDir, name);
  // Copy whenever the source is reachable, rather than only when the target
  // is missing. Skipping an existing target leaves a stale copy in place
  // indefinitely, and nothing notices, because the file exists.
  if (!existsSync(source)) {
    // Outside the monorepo an existing copy is all there is, and it is correct
    // because the tarball ships it. Only a missing copy with no source is fatal.
    if (existsSync(target)) continue;
    console.error(
      `data/${name} is missing and ${source} does not exist to copy from. ` +
        `run this from the loudkit monorepo, or restore data/ from the npm tarball.`,
    );
    process.exit(1);
  }
  copyFileSync(source, target);
  console.log(`copied ${name}`);
}
