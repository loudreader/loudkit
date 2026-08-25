// @ts-check
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import starlightLlmsTxt from 'starlight-llms-txt';
import { SITE, BASE, REPO_URL, BRANCH } from './site.config.mjs';
import { remarkRepoLinks } from './src/plugins/remark-repo-links.mjs';
import { rehypeTableScroll } from './src/plugins/rehype-table-scroll.mjs';

// The sidebar is ordered for a first-time reader: the three pages that get
// something running, then the model, the rest of the guides, the measurements,
// the platforms, the reference, and last the project's own index and promises.
// docs/README.md keeps its own order, which is the repository's order.
// Labels are given explicitly where a page's own H1 is a sentence rather than
// a name; everything else inherits its title from the file.
export default defineConfig({
  site: SITE,
  base: BASE,
  markdown: {
    processor: unified({
      remarkPlugins: [remarkRepoLinks],
      rehypePlugins: [rehypeTableScroll],
    }),
  },
  integrations: [
    starlight({
      title: 'loudkit',
      description:
        'On-device text to speech. 20 voices across 10 languages, five language SDKs, ' +
        'and voice cloning from ten seconds of audio.',
      social: [
        { icon: 'github', label: 'GitHub', href: REPO_URL },
      ],
      // Every page is generated from a file elsewhere in the repository, so
      // "edit this page" cannot be derived from the page's path under site/.
      // The sync step writes an absolute `editUrl` into each page's front
      // matter instead; this base only enables the link.
      editLink: { baseUrl: `${REPO_URL}/edit/${BRANCH}/` },
      plugins: [starlightLlmsTxt()],
      // The site theme. Everything is expressed as Starlight's own --sl-*
      // custom properties plus a few `lk-` classes the landing page uses, so
      // no Starlight component is replaced and the sidebar, search and theme
      // toggle keep their stock behaviour.
      // The fonts are the same two loudreader.io uses, and they are bundled
      // rather than pulled from Google's CDN: a site for a tool whose whole
      // claim is that nothing leaves your machine should not make every
      // reader fetch a font from a third party.
      customCss: [
        '@fontsource-variable/inter',
        '@fontsource/space-grotesk/300.css',
        '@fontsource/space-grotesk/400.css',
        '@fontsource/space-grotesk/500.css',
        '@fontsource/space-grotesk/700.css',
        './src/styles/loudkit.css',
      ],
      // LoudReader's product favicons, in site/public/. Starlight applies the
      // base path to `favicon` on its own; the two head links below are
      // written by hand, so they carry BASE themselves. On Pages the site is
      // served from /loudkit, and a root-relative /favicon-32x32.png would be
      // a 404 there and correct only in local preview.
      favicon: '/favicon.ico',
      head: [
        {
          tag: 'meta',
          attrs: { name: 'theme-color', content: '#f7f5f2' },
        },
        {
          tag: 'link',
          attrs: {
            rel: 'icon',
            type: 'image/png',
            sizes: '32x32',
            href: `${BASE}/favicon-32x32.png`,
          },
        },
        {
          tag: 'link',
          attrs: {
            rel: 'apple-touch-icon',
            sizes: '180x180',
            href: `${BASE}/apple-touch-icon.png`,
          },
        },
      ],
      sidebar: [
        {
          label: 'Quickstart',
          items: [
            { slug: 'demo', label: 'Demo' },
            { slug: 'guides/01-getting-started', label: 'Quickstart' },
            { slug: 'guides/02-streaming-and-long-form' },
            { slug: 'guides/03-cloning-a-voice' },
          ],
        },
        {
          label: 'The model',
          items: [
            { slug: 'model-card', label: 'Model card' },
            { slug: 'voices', label: 'Voices' },
            { slug: 'provenance-voice-encoder', label: 'Voice encoder provenance' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { slug: 'guides', label: 'About the guides' },
            { slug: 'guides/04-server-and-agents' },
            { slug: 'guides/05-benchmarking' },
            { slug: 'guides/06-embedding' },
            { slug: 'guides/07-js-ts' },
            { slug: 'guides/08-go' },
            { slug: 'guides/09-rust' },
            { slug: 'guides/10-swift' },
          ],
        },
        {
          label: 'Performance',
          items: [
            { slug: 'benchmarks', label: 'Benchmarks' },
            { slug: 'parity-measured', label: 'Measured parity' },
          ],
        },
        {
          label: 'Platforms',
          items: [
            { slug: 'platforms/apple', label: 'Apple' },
            { slug: 'platforms/docker', label: 'Docker' },
            { slug: 'platforms/jetson', label: 'Jetson' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { slug: 'reference/troubleshooting', label: 'Troubleshooting' },
            { slug: 'reference/errors', label: 'Errors' },
            { slug: 'reference/compatibility', label: 'Compatibility' },
            { slug: 'reference/timestamps', label: 'Timestamps' },
            { slug: 'reference/speed', label: 'Speed' },
            { slug: 'reference/provenance', label: 'Provenance' },
          ],
        },
        {
          label: 'Engine internals',
          collapsed: true,
          items: [
            { slug: 'reference/architecture', label: 'Architecture map' },
            { slug: 'reference/onnx-graphs', label: 'ONNX graphs' },
            { slug: 'reference/identity-contract', label: 'Identity contract' },
            { slug: 'reference/preprocess', label: 'Text normalization' },
            { slug: 'reference/postprocess', label: 'Postprocess' },
            { slug: 'reference/typing', label: 'Typing' },
          ],
        },
        {
          label: 'Project',
          items: [
            { slug: 'overview', label: 'Documentation index' },
            { slug: 'supported', label: 'What 0.1 supports' },
            { slug: 'responsible-use', label: 'Responsible use' },
          ],
        },
      ],
    }),
  ],
});
