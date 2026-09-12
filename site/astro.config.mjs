// @ts-check
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import starlightLlmsTxt from 'starlight-llms-txt';
import { SITE, BASE, REPO_URL, BRANCH } from './site.config.mjs';
import { remarkRepoLinks } from './src/plugins/remark-repo-links.mjs';
import { rehypeTableScroll } from './src/plugins/rehype-table-scroll.mjs';

// The sidebar follows docs/README.md: the twelve pages a user needs first, in
// the order that index lists them, then the reference, the platforms, the
// measurements and the project's own promises. docs/design/ is not on the
// site at all (see scripts/sync-docs.mjs).
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
      logo: { src: '../assets/logo-mark-flat.png', alt: '', replacesTitle: false },
      description:
        'On-device text to speech. A searchable voice gallery, five language SDKs, ' +
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
      // The landing page's hero carries two playable voices beside the copy;
      // every other page has no hero and renders nothing from this override.
      components: { Hero: './src/components/Hero.astro' },
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
      favicon: '/loudkit.png',
      head: [
        { tag: 'meta', attrs: { name: 'theme-color', content: '#f7f5f2' } },
        // Ownership of the Search Console property for
        // https://loudreader.github.io/loudkit/. The site is a Pages *project*
        // site, so the host root belongs to the organisation and the file
        // method cannot reach it; the tag travels with every page instead.
        // Removing it un-verifies the property.
        {
          tag: 'meta',
          attrs: {
            name: 'google-site-verification',
            content: 'gKH1wStrtc3YRJp93zMWxLBzLHnJuDDpR2I9CYis47E',
          },
        },
      ],
      sidebar: [
        {
          label: 'Start',
          items: [
            { slug: 'demo', label: 'Voice gallery' },
            { slug: 'guides/01-getting-started', label: 'Getting started' },
            { slug: 'guides/10-swift', label: 'Swift' },
            { slug: 'guides/08-go', label: 'Go' },
            { slug: 'guides/09-rust', label: 'Rust' },
            { slug: 'guides/07-js-ts', label: 'JavaScript and TypeScript' },
            { slug: 'guides/11-choosing-a-model', label: 'Choosing a model' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { slug: 'guides/03-cloning-a-voice', label: 'Cloning a voice' },
            { slug: 'guides/02-streaming-and-long-form', label: 'Long text and streaming' },
            { slug: 'guides/04-server-and-agents', label: 'Server and agents' },
            { slug: 'reference/troubleshooting', label: 'Troubleshooting' },
          ],
        },
        {
          label: 'The model',
          items: [
            { slug: 'model-card', label: 'Model card' },
            { slug: 'model-card-turbo', label: 'Turbo model card' },
            { slug: 'voices', label: 'Voices' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { slug: 'reference/compatibility', label: 'Compatibility' },
            { slug: 'reference/errors', label: 'Errors' },
            { slug: 'reference/timestamps', label: 'Timestamps' },
            { slug: 'reference/speed', label: 'Speed' },
            { slug: 'reference/provenance', label: 'Provenance' },
            { slug: 'reference/identity-contract', label: 'Identity contract' },
            { slug: 'provenance-voice-encoder', label: 'Voice encoder licence chain' },
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
          label: 'Performance',
          items: [
            { slug: 'benchmarks', label: 'Benchmarks' },
            { slug: 'parity-measured', label: 'Measured parity' },
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
