import test from 'node:test';
import assert from 'node:assert/strict';
import { POST, GET } from '../api/stylist6-generate-outfits.ts';
import { STYLIST6_CLOUD_VERSION } from '../../stylist6/cloud_handler.mjs';

test('route and handler report the identical engine version', async () => {
  const get = GET();
  assert.equal(get.status, 405);
  const payload = await get.json();
  assert.equal(payload.generationVersion, STYLIST6_CLOUD_VERSION);
  assert.match(payload.generationVersion, /^stylist-6-/);
});

test('stylist6 route fails closed without a transport secret', async () => {
  const saved = process.env.STYLIST6_NODE_SHARED_SECRET;
  delete process.env.STYLIST6_NODE_SHARED_SECRET;
  try {
    const response = await POST(new Request('https://node.example/api/stylist6-generate-outfits', {
      method: 'POST', headers: { authorization: 'Bearer x' }, body: '{}',
    }));
    assert.equal(response.status, 503);
    const payload = await response.json();
    assert.equal(payload.code, 'stylist_compute_unconfigured');
    assert.equal(payload.generationVersion, STYLIST6_CLOUD_VERSION);
  } finally {
    if (saved === undefined) delete process.env.STYLIST6_NODE_SHARED_SECRET;
    else process.env.STYLIST6_NODE_SHARED_SECRET = saved;
  }
});

test('stylist6 route rejects unsigned callers and wrong methods', async () => {
  process.env.STYLIST6_NODE_SHARED_SECRET = 'route-test-shared-secret-0123456789abcdef';
  const unsigned = await POST(new Request('https://node.example/api/stylist6-generate-outfits', {
    method: 'POST', headers: { authorization: 'Bearer test-jwt-value' }, body: '{}',
  }));
  assert.equal(unsigned.status, 401);
  const get = GET();
  assert.equal(get.status, 405);
  const oversize = await POST(new Request('https://node.example/api/stylist6-generate-outfits', {
    method: 'POST',
    headers: { authorization: 'Bearer test-jwt-value', 'content-length': String(1024 * 1024 + 1) },
    body: '{}',
  }));
  assert.equal(oversize.status, 413);
});
