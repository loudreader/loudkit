// Build the Starlight content collection out of the repository's own markdown.
//
// docs/ is the single source of truth and stays plain, GitHub-renderable
// markdown: no front matter, no MDX, no site-only files mixed in. Starlight
// needs the opposite — a `title` in front matter on every entry — so something
// has to bridge the two. That something is this script, and the bridge is
// one-directional and disposable: it regenerates src/content/docs/ from
// scratch on every `npm run build` and `npm run dev`, and that directory is
// gitignored. Nothing here is ever committed, so docs/ can never drift from a
// stale copy under site/.
//
// What it does per file, and why:
//
//   * derives the page title from the file's first H1, then removes that H1
//     from the body — Starlight renders the title as the page's <h1> itself,
//     so leaving the original in place would print it twice;
//   * drops any pre-existing front matter (docs/MODEL_CARD.md carries a
//     Hugging Face model-card header, which is meaningful on the Hub and is
//     not a Starlight schema);
//   * writes a `sourcePath` field recording where the page came from, which is
//     what the link-rewriting remark plugin uses to resolve repo-relative
//     links from the right origin.
//
// Link rewriting itself is deliberately *not* done here. A regex over raw
// markdown cannot tell a link from the same characters inside a fenced code
// block; the remark plugin works on the parsed tree and can.

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pageUrl, REPO_URL, BRANCH } from '../site.config.mjs';

const SITE_DIR = path.resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const REPO = path.resolve(SITE_DIR, '..');
const OUT = path.join(SITE_DIR, 'src', 'content', 'docs');
const HANDWRITTEN = path.join(SITE_DIR, 'src', 'handwritten');
const DOCMAP = path.join(SITE_DIR, '.docmap.json');

/** Root-level markdown the site publishes as pages of its own. */
const ROOT_PAGES = ['SUPPORTED.md', 'VOICES.md', 'RESPONSIBLE_USE.md'];

// README.md is consumed, but as the landing page's source material rather than
// as a page: src/handwritten/index.mdx distils it. Links pointing *at* the
// README from inside docs/ therefore go to GitHub, where the anchors they use
// actually exist.

/**
 * Files under docs/ that must not become pages.
 *
 * docs/coreml-execution.md is gitignored — it describes an execution path that
 * is not part of this release — so it exists on some working copies and never
 * in CI. Naming it here keeps the two builds identical instead of letting the
 * page set depend on whose machine ran it.
 */
const EXCLUDE = new Set([
  'docs/coreml-execution.md',
  // Historical candidate-voice study. None of those profiles ship; keeping
  // it searchable on the product site makes it look like current evidence.
  'docs/VOICE_QUALITY_REPORT.md',
]);

/**
 * docs/design/ stays out of the site entirely: those are engineering
 * notebooks — decision records, measurement plans, release checklists —
 * written for people changing the engine, not for people using it. They
 * stay tracked and GitHub-readable; a site link into docs/design/ falls
 * back to the GitHub blob URL like any other non-page repo file.
 */
const EXCLUDE_DIRS = ['docs/design/'];

/** docs/README.md and docs/guides/README.md are indexes, not "README" pages. */
const RENAME = new Map([
  ['docs/README.md', 'overview'],
  ['docs/guides/README.md', 'guides/index'],
]);

/**
 * Pages that get a site component spliced in above their own body.
 *
  * The roster grid lives on the Demo page only. Splicing a second copy into
  * VOICES.md duplicated the listening surface across two pages; the Voices
  * page keeps the facts — sources, licences, hashes — and links to the Demo
  * for the audio.
  */

/**
 * The voice audio the site plays, copied into the build.
 *
 * Two kinds live under this one directory and both are copied, because the
 * copy recurses: `<name>.opus` is the generated sample the roster grid plays,
 * and `refs/<name>.opus` is the enrollment reference the demo page plays
 * beside it. They stay one tree so a voice cannot get a sample here and a
 * reference somewhere else.
 */
const AUDIO_FROM = path.join(REPO, 'docs', 'voices', 'roster', 'audio');
const AUDIO_TO = path.join(SITE_DIR, 'public', 'voices');

/**
 * Route id for a repo-relative markdown path.
 *
 * One rule for the filename — lowercase, underscores to hyphens — so that
 * MODEL_CARD.md, PROVENANCE-voice-encoder.md and errors.md all land somewhere
 * predictable, and the docs/ prefix is dropped so that docs/reference/errors.md
 * is /reference/errors/ rather than /docs/reference/errors/.
 */
function routeId(rel) {
  if (RENAME.has(rel)) return RENAME.get(rel);
  const parts = rel.split('/');
  if (parts[0] === 'docs') parts.shift();
  const file = parts.pop().replace(/\.md$/, '').toLowerCase().replace(/_/g, '-');
  return [...parts, file].join('/');
}

function walk(dir, acc = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, acc);
    else if (entry.isFile() && entry.name.endsWith('.md')) acc.push(full);
  }
  return acc;
}

/** Strip a leading YAML front-matter block, if the file has one. */
function stripFrontMatter(text) {
  if (!text.startsWith('---\n')) return text;
  const end = text.indexOf('\n---', 3);
  if (end === -1) return text;
  return text.slice(text.indexOf('\n', end + 1) + 1);
}

/**
 * Pull the first H1 out of the body and return it as the title.
 *
 * Titles are plain text in Starlight's front matter, so the inline markdown a
 * heading may carry has to come off: `# Provenance: \`ve.safetensors\`` is the
 * page "Provenance: ve.safetensors". Only backticks and emphasis appear in
 * this repo's headings, so only those are handled — anything more would be
 * guessing at markdown this codebase does not write.
 *
 * The guides number their H1s ("# 3. Cloning a voice") because a repository
 * directory has no other way to state its reading order. The sidebar states
 * it by position, so the ordinal is dropped from the title and the page is
 * "Cloning a voice" in the tab, the sidebar and search. The heading in docs/
 * keeps its number: on GitHub it is the only order there is.
 */
function extractTitle(body) {
  const lines = body.split('\n');
  const i = lines.findIndex((l) => /^#\s+\S/.test(l));
  if (i === -1) return { title: null, body };
  const title = lines[i]
    .replace(/^#\s+/, '')
    .replace(/^\d+\.\s+/, '')
    .replace(/`/g, '')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .trim();
  lines.splice(i, 1);
  while (lines[i] === '') lines.splice(i, 1);
  return { title, body: lines.join('\n') };
}

/**
 * Copy a tree, returning every copied file's path relative to `to`.
 *
 * The recursion prefixes each subdirectory's results rather than letting the
 * nested call name them relative to itself, so `refs/carmen.opus` comes back
 * as `refs/carmen.opus` and not as a second `carmen.opus`. The separator is
 * forced to `/` because these names are compared against and printed beside
 * URL paths, not filesystem paths.
 */
function copyDir(from, to) {
  if (!fs.existsSync(from)) return [];
  const copied = [];
  for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
    const src = path.join(from, entry.name);
    const dst = path.join(to, entry.name);
    if (entry.isDirectory()) {
      copied.push(...copyDir(src, dst).map((rel) => `${entry.name}/${rel}`));
    } else {
      writeIfChanged(dst, fs.readFileSync(src));
      copied.push(entry.name);
    }
  }
  return copied;
}

/**
 * Write a file only when its bytes would change, and remember it.
 *
 * Rewriting an identical file still stamps a new mtime, and a dev server
 * watching that file recompiles it. When the file is one of the MDX pages,
 * the recompile lands mid-request and the route answers 500 until the server
 * restarts. A build running beside `astro dev` used to break the two
 * handwritten pages exactly that way. Comparing first means an unchanged
 * page is not touched at all.
 */
const written = new Set();
function writeIfChanged(dst, contents) {
  written.add(path.resolve(dst));
  const next = Buffer.isBuffer(contents) ? contents : Buffer.from(contents, 'utf8');
  if (fs.existsSync(dst) && fs.readFileSync(dst).equals(next)) return false;
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.writeFileSync(dst, next);
  return true;
}

/** Remove generated files this run did not write, and the dirs they leave. */
function pruneStale(dir) {
  if (!fs.existsSync(dir)) return;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      pruneStale(full);
      if (fs.readdirSync(full).length === 0) fs.rmdirSync(full);
    } else if (!written.has(path.resolve(full))) {
      fs.rmSync(full);
    }
  }
}

// ---------------------------------------------------------------------------

fs.mkdirSync(OUT, { recursive: true });

const sources = [
  ...ROOT_PAGES.map((f) => path.join(REPO, f)),
  ...walk(path.join(REPO, 'docs')),
]
  .map((abs) => path.relative(REPO, abs).split(path.sep).join('/'))
  .filter((rel) => !EXCLUDE.has(rel))
  .filter((rel) => !EXCLUDE_DIRS.some((dir) => rel.startsWith(dir)))
  .sort();

/** repo-relative source path -> { id, url } — what the remark plugin reads. */
const pages = {};
const untitled = [];

// The route set, as one digest, stamped into every page below.
//
// Astro caches a rendered page by the digest of that page's own text. Link
// rewriting reads the route map instead, which is state the digest cannot
// see, so adding or removing a page used to leave every other page holding
// links resolved against the old map. Carrying the digest in the front
// matter puts that state inside the text: the map changes, every page
// changes, and Astro invalidates them the way it invalidates anything else.
// Deleting the cache instead worked, and took any dev server running beside
// the build down with it.
const routeDigest = crypto
  .createHash('sha256')
  .update(sources.map((rel) => `${rel} ${routeId(rel)}`).join('\n'))
  .digest('hex')
  .slice(0, 12);

for (const rel of sources) {
  const id = routeId(rel);
  const raw = fs.readFileSync(path.join(REPO, rel), 'utf8');
  const { title, body } = extractTitle(stripFrontMatter(raw));

  if (!title) untitled.push(rel);

  // "Edit this page" has to reach the file a reader can actually change. The
  // generated copy under site/ is not it, and that is where Starlight's global
  // editLink base would point, so every page carries its own.
  const frontMatter = [
    '---',
    `title: ${JSON.stringify(title ?? id)}`,
    `sourcePath: ${JSON.stringify(rel)}`,
    `routeDigest: ${JSON.stringify(routeDigest)}`,
    `editUrl: ${JSON.stringify(`${REPO_URL}/edit/${BRANCH}/${rel}`)}`,
    '---',
    '',
  ].join('\n');

  const dest = path.join(OUT, `${id}.md`);
  writeIfChanged(dest, frontMatter + body.replace(/^\n+/, ''));

  pages[rel] = { id, url: pageUrl(id) };
}

// The voice audio, copied into a gitignored directory for the same reason
// the pages are: one copy of the audio in the tree.
const audio = copyDir(AUDIO_FROM, AUDIO_TO);
const refs = audio.filter((f) => f.startsWith('refs/'));
const samples = audio.filter((f) => !refs.includes(f));

const handwritten = copyDir(HANDWRITTEN, OUT);

pruneStale(OUT);
pruneStale(AUDIO_TO);

writeIfChanged(DOCMAP, JSON.stringify({ pages }, null, 2));

// No cache is deleted here. `routeDigest` above carries the route set into
// every page's front matter, so Astro invalidates the pages itself when the
// map moves. See the comment beside it.

if (untitled.length) {
  console.error(`sync-docs: no H1 found in:\n  ${untitled.join('\n  ')}`);
  process.exit(1);
}

console.log(
  `sync-docs: ${sources.length} pages from docs/ and the repo root, ` +
    `${handwritten.length} site-owned (${handwritten.join(', ')}) -> src/content/docs/, ` +
    `${samples.length} voice samples and ${refs.length} enrollment references ` +
    '-> public/voices/',
);
