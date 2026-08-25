// Make the repo's own relative links work on the site.
//
// The markdown under docs/ links the way a repository links: `../reference/
// errors.md`, `../../VOICES.md`, `proto/loudkit.proto`, `guides/`. Those are
// correct on GitHub and meaningless once the same file is a page at
// /loudkit/reference/errors/. Rewriting them in docs/ would break GitHub, and
// keeping two copies of every file is exactly what this site is built to
// avoid — so the rewrite happens here, at the last possible moment.
//
// It runs on the parsed tree rather than the raw text on purpose. Fenced code
// blocks in these docs contain plenty of text that a regex would mistake for a
// link; a remark plugin only ever sees real `link` and `definition` nodes.
//
// Three destinations, in order of preference:
//
//   1. a page on this site, when the target is markdown the site publishes;
//   2. a section of the documentation index, when the target is a docs/
//      subdirectory that has no index page of its own (docs/reference/ etc.);
//   3. the file on GitHub, for everything else — source, proto schemas, NOTICE,
//      the audio samples, and README.md, whose anchors exist there and not in
//      the landing page distilled from it.
//
// (3) is a feature, not a fallback: a link to proto/loudkit.proto should go to
// the actual schema, and a docs site cannot host it.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { visit } from 'unist-util-visit';
import { BLOB, TREE, REPO_URL, pageUrl } from '../../site.config.mjs';

const SITE_DIR = path.resolve(fileURLToPath(new URL('.', import.meta.url)), '..', '..');
const REPO = path.resolve(SITE_DIR, '..');
const DOCMAP = path.join(SITE_DIR, '.docmap.json');

/**
 * Directories that are linked to as directories and have no page of their own.
 *
 * docs/README.md becomes /overview/ and carries a heading per section, so a
 * link to docs/reference/ can land on the part of the index that lists the
 * reference pages instead of bouncing the reader out to GitHub.
 */
const DIR_SECTIONS = {
  'docs/reference': 'reference',
  'docs/design': 'design',
  'docs/platforms': 'platforms',
};

let docmap = null;
function pages() {
  // Read lazily: the sync step writes this file, and on a cold `npm run build`
  // the plugin module is loaded by the Astro config before that has happened.
  if (!docmap) docmap = JSON.parse(fs.readFileSync(DOCMAP, 'utf8')).pages;
  return docmap;
}

function isExternal(url) {
  return /^[a-z][a-z0-9+.-]*:/i.test(url) || url.startsWith('//');
}

function rewrite(url, sourcePath) {
  if (!url || isExternal(url) || url.startsWith('#')) return url;

  const hashAt = url.indexOf('#');
  const hash = hashAt === -1 ? '' : url.slice(hashAt);
  const target = hashAt === -1 ? url : url.slice(0, hashAt);
  if (!target) return url;

  const trailingSlash = target.endsWith('/');
  const rel = path.posix
    .normalize(path.posix.join(path.posix.dirname(sourcePath), target))
    .replace(/\/$/, '');

  const page = pages()[rel];
  if (page) return page.url + hash;

  // docs/ and docs/guides/ do have index pages, under their own route ids.
  if (rel === 'docs') return pageUrl('overview') + hash;
  if (rel === 'docs/guides') return pageUrl('guides/index') + hash;
  if (rel in DIR_SECTIONS) return `${pageUrl('overview')}#${DIR_SECTIONS[rel]}`;

  const abs = path.join(REPO, rel);
  const isDir =
    trailingSlash || (fs.existsSync(abs) && fs.statSync(abs).isDirectory());
  return `${isDir ? TREE : BLOB}/${rel}${isDir ? '/' : ''}${hash}`;
}

export function remarkRepoLinks() {
  return (tree, file) => {
    const sourcePath = file?.data?.astro?.frontmatter?.sourcePath;
    // Pages the site owns (the landing page) are written against the live URLs
    // already and have no repository origin to resolve against.
    if (!sourcePath) return;

    visit(tree, ['link', 'definition'], (node) => {
      node.url = rewrite(node.url, sourcePath);
      // A link that looked local in the repository and now points at GitHub
      // takes the reader off the site, and nothing in the text says so. The
      // title is the whole mechanism the markdown needs: it becomes the
      // anchor's tooltip, and the stylesheet hangs an external-link arrow off
      // the same href. A title the author wrote themselves is left alone.
      if (node.url.startsWith(`${REPO_URL}/`) && !node.title) {
        node.title = 'on GitHub';
      }
    });
  };
}

export default remarkRepoLinks;
