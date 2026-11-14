import test from 'node:test';
import assert from 'node:assert/strict';
import endpoint, { POST } from '../api/generate-outfits.ts';
import { signStylistTransport } from '../transport/signature.mjs';

const secret = 'test-hmac-secret-for-node-handler-not-production-12345';
const authorization = 'Bearer placeholder-not-a-user-jwt';
const body = '{"prompt":"Old Money","stylePreferences":"old_money","occasion":"Work"}';

test('unsigned callers cannot reach generation', async () => {
  process.env.STYLIST_NODE_SHARED_SECRET = secret;
  const response = await POST(new Request('https://node.example/api/generate-outfits', {
    method: 'POST', headers: { authorization }, body,
  }));
  assert.equal(response.status, 401);
  assert.equal((await response.json()).code, 'unauthorized');
});

test('signed requests run the original, quota-enforcing Stylist handler', async () => {
  process.env.STYLIST_NODE_SHARED_SECRET = secret;
  const signed = await signStylistTransport({ secret, authorization, body });
  // No Supabase credentials in this CI test. Once signature verification
  // succeeds, the ORIGINAL rate limiter must fail closed with its own
  // rate_limit_unavailable response, not our Node adapter 503.
  const oldUrl = process.env.SUPABASE_URL;
  const oldKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  delete process.env.SUPABASE_URL;
  delete process.env.SUPABASE_SERVICE_ROLE_KEY;
  try {
    const response = await POST(new Request('https://node.example/api/generate-outfits', {
      method: 'POST',
      headers: {
        authorization,
        apikey: 'test-publishable',
        'content-type': 'application/json',
        ...signed,
      },
      body,
    }));
    assert.equal(response.status, 503);
    assert.equal(response.status, 503);
    assert.match(response.headers.get('server-timing') || '', /^aiw_node_cpu;dur=\d+(?:\.\d+)?, aiw_node_wall;dur=\d+(?:\.\d+)?$/);
    assert.equal((await response.json()).error, 'rate_limit_unavailable');
    assert.equal(typeof globalThis.__AIW_STYLIST_HANDLER__, 'function');
  } finally {
    if (oldUrl === undefined) delete process.env.SUPABASE_URL;
    else process.env.SUPABASE_URL = oldUrl;
    if (oldKey === undefined) delete process.env.SUPABASE_SERVICE_ROLE_KEY;
    else process.env.SUPABASE_SERVICE_ROLE_KEY = oldKey;
  }
});

test('Vercel default fetch entrypoint is exported and fails closed', async () => {
  assert.equal(typeof endpoint.fetch, 'function');
  process.env.STYLIST_NODE_SHARED_SECRET = secret;
  const unsigned = await endpoint.fetch(new Request('https://node.example/api/generate-outfits', {
    method: 'POST', headers: { authorization }, body,
  }));
  assert.equal(unsigned.status, 401);
  const forbiddenMethod = await endpoint.fetch(new Request('https://node.example/api/generate-outfits', {
    method: 'GET',
  }));
  assert.equal(forbiddenMethod.status, 405);
});
