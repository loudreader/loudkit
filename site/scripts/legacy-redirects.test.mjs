import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';
import { DESTINATION, LEGACY_ROUTES, generate, redirectHtml, targetFor } from './legacy-redirects.mjs';

test('old home and documentation index keep distinct destinations', () => {
  assert.equal(targetFor('/'), `${DESTINATION}/`);
  assert.equal(targetFor('/overview/'), `${DESTINATION}/overview/`);
  assert.equal(LEGACY_ROUTES.length, new Set(LEGACY_ROUTES).size);
  assert.throws(() => targetFor('//example.com/'));
  assert.throws(() => targetFor('/missing/'));
});

test('redirect preserves a deep section and query without changing the destination host', () => {
  let redirected;
  const html = redirectHtml('/guides/01-getting-started/');
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  vm.runInNewContext(script, {
    URL,
    window: { location: {
      search: '?ref=docs&next=https://example.com/',
      hash: '#install',
      replace(value) { redirected = value; },
    } },
  });
  const target = new URL(redirected);
  assert.equal(target.origin, DESTINATION);
  assert.equal(target.pathname, '/guides/01-getting-started/');
  assert.equal(target.searchParams.get('ref'), 'docs');
  assert.equal(target.hash, '#install');
  assert.match(html, /http-equiv="refresh" content="0; url=https:\/\/loudkit\.loudreader\.io\/guides\/01-getting-started\/"/);
});

test('artifact covers every legacy route and refuses an existing site directory', async (t) => {
  const output = await fs.mkdtemp(path.join(os.tmpdir(), 'loudkit-legacy-test-'));
  t.after(() => fs.rm(output, { recursive: true, force: true }));
  await generate(output);
  for (const route of LEGACY_ROUTES) {
    const html = await fs.readFile(path.join(output, route.slice(1), 'index.html'), 'utf8');
    assert.ok(html.includes(`<link rel="canonical" href="${targetFor(route)}">`));
    assert.ok(html.includes(`<a href="${targetFor(route)}">`));
    assert.ok(!html.includes('noindex'));
  }
  const missing = await fs.readFile(path.join(output, '404.html'), 'utf8');
  assert.ok(missing.includes('noindex'));
  assert.ok(!missing.includes('http-equiv="refresh"'));
  await assert.rejects(() => generate(output), /must be empty/);
});
