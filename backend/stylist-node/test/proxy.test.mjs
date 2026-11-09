import test from 'node:test';
import assert from 'node:assert/strict';
import { verifyStylistTransport } from '../transport/signature.mjs';

const secret = 'test-secret-signature-path-enough-characters-12345';
const bypass = 'abcd1234ABCD5678efgh9012EFGH3456';
const endpoint = 'https://compute-test.vercel.app/api/generate-outfits';

let proxyHandler;
globalThis.Deno = {
  serve(fn) { proxyHandler = fn; },
  env: { get(name) {
    return {
      STYLIST_NODE_URL: endpoint,
      STYLIST_NODE_SHARED_SECRET: secret,
      STYLIST_VERCEL_BYPASS_SECRET: bypass,
    }[name];
  } },
};
await import('../../stylist-proxy/index.ts');

test('proxy rejects missing authorization before contacting Node', async () => {
  assert.equal(typeof proxyHandler, 'function');
  const reply = await proxyHandler(new Request('https://supabase.example/functions/v1/generate-outfits', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: '{}',
  }));
  assert.equal(reply.status, 401);
  assert.equal((await reply.json()).code, 'unauthorized');
});

test('proxy HMAC-signs exact body and token, preserves status, and forbids redirects', async () => {
  const oldFetch = globalThis.fetch;
  const body = JSON.stringify({ prompt: 'Old Money', occasion: 'Work' });
  let called = 0;
  globalThis.fetch = async (url, init) => {
    called++;
    assert.equal(String(url), endpoint);
    assert.equal(init.method, 'POST');
    assert.equal(init.redirect, 'error');
    assert.equal(init.body, body);
    assert.equal(init.headers.apikey, 'public-anon-key');
    assert.equal(init.headers['x-vercel-protection-bypass'], bypass);
    assert.equal(await verifyStylistTransport({
      secret, body,
      authorization: 'Bearer user-token',
      timestamp: init.headers['x-aiw-timestamp'],
      nonce: init.headers['x-aiw-nonce'],
      signature: init.headers['x-aiw-signature'],
    }), true);
    return Response.json({ success: true, outfits: [{ id: 'sample' }] }, {
      status: 200,
      headers: { 'server-timing': 'aiw_node_cpu;dur=42.1, aiw_node_wall;dur=411.5' },
    });
  };
  try {
    const result = await proxyHandler(new Request('https://supabase.example/functions/v1/generate-outfits', {
      method: 'POST', body,
      headers: {
        authorization: 'Bearer user-token',
        apikey: 'public-anon-key',
        'content-type': 'application/json',
      },
    }));
    assert.equal(result.status, 200);
    assert.equal(result.headers.get('server-timing'), 'aiw_node_cpu;dur=42.1, aiw_node_wall;dur=411.5');
    assert.equal(result.headers.get('access-control-expose-headers'), 'server-timing');
    assert.equal((await result.json()).outfits[0].id, 'sample');
    assert.equal(called, 1);
  } finally {
    globalThis.fetch = oldFetch;
  }
});

test('proxy rejects oversized requests and does not invoke compute', async () => {
  const result = await proxyHandler(new Request('https://supabase.example/functions/v1/generate-outfits', {
    method: 'POST', body: '{}',
    headers: {
      authorization: 'Bearer user-token',
      'content-type': 'application/json',
      'content-length': '2097152',
    },
  }));
  assert.equal(result.status, 413);
});
