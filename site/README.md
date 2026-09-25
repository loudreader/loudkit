# The documentation site

[Astro Starlight](https://starlight.astro.build), published to GitHub Pages at
<https://loudreader.github.io/loudkit>.
[`.github/workflows/docs.yml`](../.github/workflows/docs.yml) builds the site for
every push and pull request that touches `docs/`, `site/`, `README.md`,
`SUPPORTED.md`, `VOICES.md`, `RESPONSIBLE_USE.md` or the workflow itself. It
deploys only from `main`.

```bash
cd site
npm install
npm run dev      # http://localhost:4321/loudkit/
npm run build    # -> site/dist/
```

## Content source

The pages come from `docs/` and from three root-level files: `SUPPORTED.md`,
`VOICES.md` and `RESPONSIBLE_USE.md`. These stay plain GitHub markdown, without
front matter, MDX or site-only files.

`scripts/sync-docs.mjs` generates `src/content/docs/` from them before every
`npm run build` and `npm run dev`. The directory is build output and is
gitignored.

Starlight needs a `title` in each page's front matter. The sync step takes it
from the file's first H1, drops a leading guide number such as "3.", and
removes that H1 from the body, because Starlight renders the title as the page
`<h1>`. It also removes any existing front matter, such as the Hugging Face
header of `docs/MODEL_CARD.md`, and writes `sourcePath` and `editUrl` into each
page.

`README.md` is not a page. `src/handwritten/index.mdx`, the landing page, is
written by hand from it. `src/handwritten/demo.mdx` is the voice gallery page.
The sync step copies both into the generated collection. Keep them, and the
visible copy in `src/components/`, consistent with the repository by hand. This
applies especially to the figures that `index.mdx` and `CudaCard.astro` quote
from `docs/benchmarks.md`.

## Excluded files

`scripts/sync-docs.mjs` keeps these out of the site:

- `docs/design/`, the notes for contributors. They stay readable on GitHub.
- `docs/coreml-execution.md` and `docs/VOICE_QUALITY_REPORT.md`, by name.
  They are not tracked and exist only on some working copies, so excluding
  them keeps the page set the same on every machine.

## Links

The markdown links the way a repository links: `../reference/errors.md`,
`../../VOICES.md`, `proto/loudkit.proto`, `guides/`. These links work on GitHub
but not on the site. `src/plugins/remark-repo-links.mjs` rewrites them at
render time, to one of three destinations:

1. **a page on this site**, when the target is markdown the site publishes;
2. **a section of the documentation index**, for `docs/` subdirectories
   without an index page. A link to `docs/reference/` goes to that section of
   `/loudkit/overview/`;
3. **the file on GitHub**, for everything else.

Links to files that are not pages go to GitHub: `proto/loudkit.proto`,
`NOTICE`, the voice samples under `docs/voices/roster/`, and `README.md`,
because its anchors exist only on GitHub.

The plugin works on the parsed markdown tree, so text inside fenced code blocks
is never rewritten.

## Route names

`routeId()` drops the `docs/` prefix, lowercases the filename and turns
underscores into hyphens. `docs/reference/errors.md` becomes
`/loudkit/reference/errors/` and `docs/MODEL_CARD.md` becomes
`/loudkit/model-card/`. `docs/README.md` is the documentation index and is
published as `/loudkit/overview/`.

## The sidebar

`astro.config.mjs` sets the sidebar explicitly, in this order: Start, Guides,
The model, Reference, Platforms, Performance and Project (the documentation
index, what 0.1 supports and responsible use). `docs/README.md` keeps its own
order. A new page under `docs/` is published unless the sync step excludes it,
but it appears in the sidebar only after you add it to `astro.config.mjs`.

## The voice gallery

`src/components/VoiceGrid.astro` shows the 28 included voices on the Demo page,
with language and presentation filters and a name search. It reads
`docs/voices/roster/provenance.json` at build time and joins the comparison
data from `docs/voices/preview/catalog.json` by voice name, so the site holds
no copy of either. The Voices page lists sources and licences and links to the
Demo for playback.

`catalog.json` adds matched loudr-1 and loudr-1-turbo samples for the ten
English voices, and downloadable profile files for eight of them.
`tools/build_voices_md.py` generates `VOICES.md` from the roster.

The sync step copies `docs/voices/roster/audio/` into `public/voices/` and
`docs/voices/preview/` into `public/voices/preview/`. Both are gitignored, for
the same reason as `src/content/docs/`.

## Tables

`src/plugins/rehype-table-scroll.mjs` wraps each markdown table in a
horizontally scrolling container. It also marks the columns whose longest cell
is short, so those columns keep their natural width and do not wrap.

## Other files

- `site.config.mjs`: the origin and base path. The Astro config, the sync step
  and the link plugin all read them from this one file.
- `starlight-llms-txt`: publishes `/llms.txt`, `/llms-small.txt` and
  `/llms-full.txt`.
- The landing page uses Starlight's `<Tabs>` and `<TabItem>` for the five
  install examples.
- The site header and the favicon use `assets/logo-mark-flat.png`. The sync
  step copies it to `public/loudkit.png`, which is gitignored.
