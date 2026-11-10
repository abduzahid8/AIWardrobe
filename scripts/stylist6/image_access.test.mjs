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
