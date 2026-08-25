// Shared constants for the site build.
//
// Three separate things need these values — the Astro config, the docs sync
// step and the link-rewriting remark plugin — and they have to agree exactly or
// internal links land on 404s. So they live in one plain .mjs module that all
// three import, rather than being repeated in three places.

/** GitHub Pages origin. Project pages, so the repo name is the base path. */
export const SITE = 'https://loudreader.github.io';
export const BASE = '/loudkit';

/** Where a link into a non-documentation repo file has to point instead. */
export const REPO_URL = 'https://github.com/loudreader/loudkit';
export const BRANCH = 'main';
export const BLOB = `${REPO_URL}/blob/${BRANCH}`;
export const TREE = `${REPO_URL}/tree/${BRANCH}`;

/** Absolute URL of a page, given the route id the sync step assigned it. */
export function pageUrl(id) {
  const path = id === 'index' ? '' : id.replace(/(^|\/)index$/, '$1');
  return `${BASE}/${path}${path.endsWith('/') || path === '' ? '' : '/'}`;
}
