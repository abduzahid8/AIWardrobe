import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { resolveImages, HOST_ALLOWLIST, KNOWN_BLOCKED, IMAGE_ACCESS_VERSION } from './image_access.mjs';

test('unapproved hosts are blocked with zero network traffic', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'imgacc-'));
  const out = await resolveImages({
    urls: [{ id: 'x', imageUrl: 'https://blocked-merchant.example/item.jpg' }],
    outDir: path.join(dir, 'img'),
    auditPath: path.join(dir, 'audit.jsonl'),
  });
  assert.equal(out[0].status, 'blocked');
  assert.equal(out[0].file, null);
  const audit = fs.readFileSync(path.join(dir, 'audit.jsonl'), 'utf8');
  assert.match(audit, /blocked_unapproved_host/);
});

test('known-403 hosts are labeled distinctly in the audit', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'imgacc-'));
  const out = await resolveImages({
    urls: [{ id: 'y', imageUrl: 'https://www.apc.com/item.jpg' }],
    outDir: path.join(dir, 'img'),
    auditPath: path.join(dir, 'audit.jsonl'),
  });
  assert.equal(out[0].status, 'blocked');
  assert.match(fs.readFileSync(path.join(dir, 'audit.jsonl'), 'utf8'), /blocked_known_403_host/);
});

test('allowlist is explicit and versioned', () => {
  assert.ok(HOST_ALLOWLIST.has('static.zara.net'));
  assert.ok(HOST_ALLOWLIST.has('cdn.shopify.com'));
  assert.ok(!HOST_ALLOWLIST.has('www.apc.com'));
  assert.ok(KNOWN_BLOCKED.has('www.apc.com'));
  assert.equal(IMAGE_ACCESS_VERSION, 'image-access-v1');
});

test('merchant media refuses credentials, private hosts, alternate ports and redirects', async (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'imgacc-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const calls = [];
  const original = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    calls.push({ url, redirect: options.redirect });
    return new Response(null, { status: 302, headers: { Location: 'https://127.0.0.1/private' } });
  };
  try {
    const urls = [
      'http://static.zara.net/x.jpg', 'https://static.zara.net:8443/x.jpg',
      'https://user@static.zara.net/x.jpg', 'https://[::ffff:127.0.0.1]/private',
      'https://static.zara.net/x.jpg',
    ].map((imageUrl, id) => ({ id, imageUrl }));
    const results = await resolveImages({ urls, outDir: dir, auditPath: null });
    assert.deepEqual(results.map((r) => r.status), ['blocked', 'blocked', 'blocked', 'blocked', 'failed']);
    assert.deepEqual(calls, [
      { url: urls[4].imageUrl, redirect: 'error' }, { url: urls[4].imageUrl, redirect: 'error' },
    ]);
  } finally {
    globalThis.fetch = original;
  }
});
