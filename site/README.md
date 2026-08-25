# The documentation site

[Astro Starlight](https://starlight.astro.build), published to GitHub Pages at
<https://loudreader.github.io/loudkit>. Built and deployed by
[`.github/workflows/docs.yml`](../.github/workflows/docs.yml) on every push to
`main` that touches `docs/`, `site/`, or the four root-level markdown files the
site publishes.

```bash
cd site
npm install
npm run dev      # http://localhost:4321/loudkit/
npm run build    # -> site/dist/
```

## It renders docs/. It does not contain docs/.

`docs/` is the single source of truth and stays plain, GitHub-renderable
markdown — no front matter, no MDX, no site-only files threaded through it. A
reader who arrives at the repository instead of the site gets the same
documentation, and nobody has to remember which of two copies to edit.

So there are no copies. `scripts/sync-docs.mjs` regenerates
`src/content/docs/` from `docs/` and the repository root on every `npm run
build` and `npm run dev`, and that directory is gitignored. It is derived
output with the lifetime of a build, not a checked-in mirror that can quietly
fall a commit behind.

Symlinks would have been less machinery, but Starlight needs something the
source files do not have and are not going to grow: a `title` in front matter
on every page. The sync step supplies it from the file's own first H1 — and
then removes that H1 from the body, because Starlight renders the title as the
page's `<h1>` itself and two of them look like a mistake. It also drops
`docs/MODEL_CARD.md`'s Hugging Face model-card header, which is meaningful on
the Hub and is not a Starlight schema.

The pages the site actually owns are in `src/handwritten/`: `index.mdx`, the
landing page, distilled from the README rather than generated from it, and
`demo.mdx`, the voice showcase. They are copied into the generated collection
alongside everything else, and they are the only prose here that has to be
kept honest against the repository by hand — in particular the landing page's
measured figures, which are quoted from `docs/benchmarks.md`.

## Links

The markdown links the way a repository links: `../reference/errors.md`,
`../../VOICES.md`, `proto/loudkit.proto`, `guides/`. Correct on GitHub, and
meaningless once the same file is a page at `/loudkit/reference/errors/`.
`src/plugins/remark-repo-links.mjs` rewrites them at render time, to one of
three destinations:

1. **a page on this site**, when the target is markdown the site publishes;
2. **a section of the documentation index**, for `docs/` subdirectories that
   have no index page of their own — a link to `docs/reference/` lands on the
   part of `/loudkit/overview/` that lists the reference pages;
3. **the file on GitHub**, for everything else.

The third is a feature rather than a fallback. A link to `proto/loudkit.proto`
should reach the actual schema, and a documentation site cannot host one. The
same applies to `NOTICE`, to the voice samples under `docs/voices/roster/`, and
to `README.md` — whose anchors exist on GitHub and not in the landing page
distilled from it.

It runs as a remark plugin, on the parsed tree, rather than as a pass over the
raw text in the sync step. These documents are full of fenced code blocks
containing text a regex would happily mistake for a link; remark only ever
sees real link nodes.

## Route names

One rule, in `routeId()`: drop the `docs/` prefix, lowercase the filename, turn
underscores into hyphens. `docs/reference/errors.md` becomes
`/loudkit/reference/errors/` and `docs/MODEL_CARD.md` becomes
`/loudkit/model-card/`. Two files are renamed by hand because they are indexes
rather than pages named "README": `docs/README.md` is `/loudkit/overview/` and
`docs/guides/README.md` is `/loudkit/guides/`.

`docs/coreml-execution.md` is excluded by name. It is gitignored — the shipped
Apple execution path is not part of this release — so it exists on some working
copies and never in CI, and the page set must not depend on whose machine ran
the build.

## The sidebar

Set out longhand in `astro.config.mjs`, ordered for a first-time reader:
Quickstart, the model, the rest of the guides, performance, platforms,
reference, and last "Project" — the documentation index, what 0.1 supports and
responsible use. `docs/README.md` keeps its own order, which is the
repository's. Adding a page under `docs/` publishes it; listing it in the
sidebar is a second, deliberate step.

## The voice grid

`src/components/VoiceGrid.astro` renders the twenty shipped voices as a
playable card grid, filtered by language. It reads
`docs/voices/roster/provenance.json` at build time, so the roster is stated
once and the site holds no copy of it. The samples are copied by the sync step
from `docs/voices/roster/audio/` into `public/voices/`, which is gitignored for
the same reason `src/content/docs/` is.

It appears twice: on `/demo/`, a site-owned page in `src/handwritten/`, and at
the top of the Voices page. The second one needs MDX, which `VOICES.md` is not,
so `EMBED` in the sync step writes that page as `.mdx` with the component
spliced in above the body. MDX reads `<` and `{` as JSX, so the step refuses to
embed a file containing either rather than producing a build error further
down.

## Tables

`src/plugins/rehype-table-scroll.mjs` wraps every markdown table in a scroll
container and marks the columns whose longest cell is short. Without it,
`table-layout: auto` gives the width to the one column holding a paragraph and
wraps the label columns one word per line; with it the labels keep their
natural width and the overflow becomes a horizontal scroll on the wrapper
rather than a squeeze.

## What else is here

- **`site.config.mjs`** — the origin and base path, in one place. The Astro
  config, the sync step and the link plugin all read them, and they have to
  agree exactly or every internal link is wrong at once.
- **`starlight-llms-txt`** — publishes `/llms.txt`, `/llms-small.txt` and
  `/llms-full.txt`, so a model can read the documentation without crawling it.
- **MDX and Starlight's components** are installed and working. The landing
  page uses `<Tabs>` for the five install commands, which is the shape the
  five-language examples will want when they are written.
