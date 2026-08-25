import { defineCollection, z } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

export const collections = {
  docs: defineCollection({
    loader: docsLoader(),
    schema: docsSchema({
      extend: z.object({
        // Written by scripts/sync-docs.mjs: the repo-relative path this page
        // was generated from. The link-rewriting remark plugin resolves the
        // file's own relative links against it. Absent on pages the site owns.
        sourcePath: z.string().optional(),
        // Also written by the sync step: a digest of the whole route set.
        // It puts the link map inside each page's own text, so Astro's
        // content cache invalidates the pages when the map moves.
        routeDigest: z.string().optional(),
      }),
    }),
  }),
};
