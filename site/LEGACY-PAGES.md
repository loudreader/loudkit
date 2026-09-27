# Retired GitHub Pages URLs

The canonical website is https://loudkit.loudreader.io/. Vercel builds and
publishes that site. GitHub Pages at https://loudreader.github.io/loudkit/
should publish only the migration artifact made by `scripts/legacy-redirects.mjs`.

The generator retains all 31 URLs found in the old live sitemap on 27 September
2026. Each maps to the same route on the new domain, with `/loudkit` removed.
The old home goes to the new home; `/overview/`, generated from `docs/README.md`,
stays the documentation index. Every destination must return HTTP 200 and
self-canonicalize before the workflow can publish.

GitHub Pages cannot set per-path HTTP 301/308 responses. The generated HTML uses
an immediate (zero-second) meta refresh, a canonical link, and a visible link.
JavaScript performs the same redirect while retaining query strings and section
fragments. These are client redirects, not server-side 301 responses. Unknown
paths remain 404s with a link to the current site. The migration artifact contains
no duplicated documentation and no sitemap advertising the old URLs.

## Check and build

No dependency installation or Astro build is required:

```sh
node --test site/scripts/legacy-redirects.test.mjs
node site/scripts/legacy-redirects.mjs --check-targets --out /tmp/loudkit-legacy-preview
```

The output directory must be empty. `--check-targets` checks the actual live
new-domain pages; omit it for an offline artifact build. The normal output is
`site/legacy-dist`, isolated from the Vercel/Astro `site/dist` output.

## Publish safely

The public repository is `loudreader/loudkit`, its default branch is `main`,
and GitHub Pages is configured to use GitHub Actions. The workflow deploys only
on `main`; pull requests exercise mapping, destination validation, and artifact
creation without publishing.

To migrate independently of engine development, start from the current public
`main` and include only these files in the change:

- `.github/workflows/docs.yml`
- `site/scripts/legacy-redirects.mjs`
- `site/scripts/legacy-redirects.test.mjs`
- `site/LEGACY-PAGES.md`
- the `legacy-dist/` ignore entry in `site/.gitignore`

Do not publish the unrelated engine branch or copy the website worktree as a
whole. Merge the focused change to public `main`; the `docs` workflow publishes
the redirect artifact automatically. Verify the old root and all 30 deep URLs
have the expected canonical, immediate refresh, and a live new destination.
No domain or DNS settings need to change.

The previous Pages workflow's IndexNow submission is intentionally removed:
only the canonical Vercel website submits new content URLs. Keep this redirect
artifact available for old links and for crawlers discovering the move.
