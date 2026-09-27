// GitHub Pages cannot configure per-path HTTP 301/308 responses. Keep the
// retired project site as immediate HTML redirects, with explicit canonicals.
// No Astro build, package installation, or repository source pages are needed.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const DESTINATION = 'https://loudkit.loudreader.io';
export const LEGACY_ORIGIN = 'https://loudreader.github.io/loudkit';

// Snapshot of every URL in the live legacy sitemap on 2026-09-27. These stay
// stable when the current documentation adds or removes pages. /overview/
// is the published route for docs/README.md, not /readme/ or /docs/readme/.
export const LEGACY_ROUTES = [
  '/',
  '/benchmarks/',
  '/demo/',
  '/guides/01-getting-started/',
  '/guides/02-streaming-and-long-form/',
  '/guides/03-cloning-a-voice/',
  '/guides/04-server-and-agents/',
  '/guides/07-js-ts/',
  '/guides/08-go/',
  '/guides/09-rust/',
  '/guides/10-swift/',
  '/guides/11-choosing-a-model/',
  '/model-card-turbo/',
  '/model-card/',
  '/overview/',
  '/parity-measured/',
  '/platforms/apple/',
  '/platforms/docker/',
  '/platforms/jetson/',
  '/provenance-voice-encoder/',
  '/reference/cli/',
  '/reference/compatibility/',
  '/reference/errors/',
  '/reference/identity-contract/',
  '/reference/provenance/',
  '/reference/speed/',
  '/reference/timestamps/',
  '/reference/troubleshooting/',
  '/responsible-use/',
  '/supported/',
  '/voices/',
];

export function targetFor(route) {
  if (!LEGACY_ROUTES.includes(route)) throw new Error(`Unknown legacy route: ${route}`);
  return `${DESTINATION}${route}`;
}

export function redirectHtml(route) {
  const target = targetFor(route);
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Loudkit has moved</title>
  <link rel="canonical" href="${target}">
  <script>
    const destination = new URL(${JSON.stringify(target)});
    destination.search = window.location.search;
    destination.hash = window.location.hash;
    window.location.replace(destination.href);
  </script>
  <meta http-equiv="refresh" content="0; url=${target}">
</head>
<body>
  <h1>Loudkit has moved</h1>
  <p>Continue to <a href="${target}">${target}</a>.</p>
</body>
</html>
`;
}

export async function checkTargets() {
  // Small batches avoid a burst of requests to production. A failed or
  // noncanonical destination must be fixed before a migration is published.
  for (let start = 0; start < LEGACY_ROUTES.length; start += 5) {
    await Promise.all(LEGACY_ROUTES.slice(start, start + 5).map(async (route) => {
      const target = targetFor(route);
      const response = await fetch(target, { redirect: 'manual', signal: AbortSignal.timeout(20000) });
      if (response.status !== 200) throw new Error(`${target}: HTTP ${response.status}`);
      const html = await response.text();
      const canonicalTag = html.match(/<link\b(?=[^>]*\brel=["']canonical["'])[^>]*>/i)?.[0];
      const canonical = canonicalTag?.match(/\bhref=["']([^"']+)["']/i)?.[1];
      if (canonical !== target) throw new Error(`${target}: unexpected canonical ${canonical}`);
      if (/<meta\b(?=[^>]*\bname=["']robots["'])(?=[^>]*\bcontent=["'][^"']*noindex)[^>]*>/i.test(html)) {
        throw new Error(`${target}: destination is noindex`);
      }
    }));
  }
  console.log(`Verified ${LEGACY_ROUTES.length} live canonical destinations.`);
}

export async function generate(output) {
  // Refuse to mix redirect pages into an existing site build. CI starts with
  // an empty directory; local runs can use a new --out directory each time.
  await fs.mkdir(output, { recursive: true });
  if ((await fs.readdir(output)).length > 0) throw new Error(`Output directory must be empty: ${output}`);
  for (const route of LEGACY_ROUTES) {
    const filename = path.join(output, route.slice(1), 'index.html');
    await fs.mkdir(path.dirname(filename), { recursive: true });
    await fs.writeFile(filename, redirectHtml(route));
  }
  await fs.writeFile(path.join(output, '.nojekyll'), '');
  // Unknown URLs remain genuine 404s instead of redirecting unrelated paths
  // to the home page. All 31 published URLs above have their own destination.
  await fs.writeFile(path.join(output, '404.html'), `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="robots" content="noindex">
<title>Page not found — Loudkit</title></head><body><h1>Page not found</h1>
<p>Loudkit is now at <a href="${DESTINATION}/">${DESTINATION}/</a>.</p></body></html>\n`);
  await fs.writeFile(path.join(output, 'llms.txt'), `# Loudkit documentation has moved\n\nCanonical website: ${DESTINATION}/\nDocumentation index: ${DESTINATION}/overview/\nAgent-readable index: ${DESTINATION}/llms.txt\n`);
  console.log(`Generated ${LEGACY_ROUTES.length} legacy redirects in ${output}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  let output = fileURLToPath(new URL('../legacy-dist/', import.meta.url));
  let verify = false;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--check-targets') verify = true;
    else if (args[i] === '--out' && args[i + 1]) output = path.resolve(args[++i]);
    else throw new Error(`Unknown or incomplete argument: ${args[i]}`);
  }
  if (verify) await checkTargets();
  await generate(output);
}
