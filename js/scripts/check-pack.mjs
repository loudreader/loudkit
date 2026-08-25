/**
 * Refuse to publish a package that cannot be imported.
 *
 * `package.json` points `main`/`types` at `dist/` and lists `data/` in `files`, and
 * both are generated and gitignored. Without a `prepack` step, `npm pack` from
 * a clean checkout produced a tarball of **two files** — README.md and
 * package.json, 1148 bytes — which installs fine and fails at the first
 * `import`. Nothing in the test suite notices, because the tests run against
 * the working tree rather than the artefact.
 *
 * The Polish lexicon is checked by size as well as existence: `prebuild` copies
 * it from the Python package and silently does nothing when the source is
 * missing, so an empty or absent `data/` means the respelling tables were never
 * copied and the binding would degrade to unrespelled English inside Polish
 * text — the failure that is inaudible in review and obvious to a listener.
 */
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

const LEXICON_MIN_BYTES = 1_000_000; // the real table is ~6.6 MB

const problems = [];

function filesBelow(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    return entry.isDirectory() ? filesBelow(path) : [path];
  });
}

for (const entry of ["dist/index.js", "dist/index.d.ts"]) {
  if (!existsSync(join(root, entry))) {
    problems.push(`${entry} is missing — run \`npm run build\``);
  }
}

// A source map that points outside the tarball is worse than no map: editors
// advertise source-level debugging and then open a file the user never got.
// TypeScript's `inlineSources` puts the matching source beside each path in
// `sources`, so the map is self-contained without publishing the whole src/
// tree as a second API surface.
const maps = existsSync(join(root, "dist"))
  ? filesBelow(join(root, "dist")).filter((path) => path.endsWith(".js.map"))
  : [];
if (maps.length === 0) {
  problems.push("dist/ contains no JavaScript source maps");
}
for (const path of maps) {
  let map;
  try {
    map = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    problems.push(`${path}: invalid source map JSON (${error.message})`);
    continue;
  }
  if (
    !Array.isArray(map.sources) ||
    !Array.isArray(map.sourcesContent) ||
    map.sources.length !== map.sourcesContent.length ||
    map.sourcesContent.some((source) => typeof source !== "string")
  ) {
    problems.push(`${path}: sources are not embedded one-for-one`);
  }
}

const lexicon = join(root, "data", "pl_en_respell.json");
if (!existsSync(lexicon)) {
  problems.push(
    "data/pl_en_respell.json is missing — `prebuild` copies it from " +
      "../../python/loudkit/models/data/, which means this was packed outside the " +
      "loudkit repo"
  );
} else if (statSync(lexicon).size < LEXICON_MIN_BYTES) {
  problems.push(
    `data/pl_en_respell.json is only ${statSync(lexicon).size} bytes; the ` +
      `respelling table should be several megabytes`
  );
}

// The number grammars are compiled into dist/ by the JSON import, but the
// tarball also ships data/ so the file stays inspectable next to its siblings.
if (!existsSync(join(root, "data", "numbers.json"))) {
  problems.push(
    "data/numbers.json is missing — `prebuild` copies it from " +
      "../../python/loudkit/models/data/"
  );
}

// The tarball is what a recipient gets, and it is not this repository: without
// these files in it, neither the legal terms nor the required dual-use
// declaration reach the recipient. They are checked in rather than generated,
// so a missing one means someone deleted it.
for (const entry of ["LICENSE", "NOTICE", "DISCLOSURE"]) {
  if (!existsSync(join(root, entry))) {
    problems.push(`${entry} is missing — copy it from the repository root`);
  }
}

if (problems.length > 0) {
  console.error("refusing to pack an unusable package:");
  for (const p of problems) console.error(`  - ${p}`);
  process.exit(1);
}

console.log(
  "pack check OK: dist/, self-contained source maps, data/, terms and disclosure are present"
);
