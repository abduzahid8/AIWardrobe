import test from 'node:test';
import assert from 'node:assert/strict';
import endpoint from '../api/transport-health.ts';
import { signStylistTransport } from '../transport/signature.mjs';

const key = 'transport-only-test-secret-with-32-or-more-chars';
const body = '{"probe":"aiw-stylist-transport-v1"}';
const authorization = 'Bearer aiw-transport-probe-no-user';
test('unsigned transport probe is rejected before any generation', async () => {
  const saved = process.env.STYLIST_NODE_SHARED_SECRET;
  process.env.STYLIST_NODE_SHARED_SECRET = key;
  try {
    const r = await endpoint.fetch(new Request('https://compute.local/api/transport-health', {
      method: 'POST',headers:{authorization},body
    }));
    assert.equal(r.status,401);
  } finally {
    saved === undefined ? delete process.env.STYLIST_NODE_SHARED_SECRET :
      process.env.STYLIST_NODE_SHARED_SECRET = saved;
  }
});
test('valid HMAC probe returns 200 without invoking the Stylist handler', async () => {
  const saved = process.env.STYLIST_NODE_SHARED_SECRET;
  process.env.STYLIST_NODE_SHARED_SECRET = key;
  try {
    const signed = await signStylistTransport({secret:key,body,authorization});
    const r=await endpoint.fetch(new Request('https://compute.local/api/transport-health',{
      method:'POST',headers:{authorization,...signed},body
    }));
    assert.equal(r.status,200);
    assert.deepEqual(await r.json(),{ok:true,transport:'signed'});
  } finally {
    saved === undefined ? delete process.env.STYLIST_NODE_SHARED_SECRET :
      process.env.STYLIST_NODE_SHARED_SECRET=saved;
  }
});
