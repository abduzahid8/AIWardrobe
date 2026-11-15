import test from 'node:test';
import assert from 'node:assert/strict';
import { verifyStylistTransport } from '../transport/signature.mjs';

const secret = 'test-secret-signature-path-enough-characters-12345';
const s6Secret = 'stylist6-secret-signature-path-enough-chars-67890';
const bypass = 'abcd1234ABCD5678efgh9012EFGH3456';
const endpoint = 'https://compute-test.vercel.app/api/generate-outfits';
const s6Endpoint = 'https://compute-test.vercel.app/api/stylist6-generate-outfits';
const envOverrides = {};

let proxyHandler;
globalThis.Deno = {
  serve(fn) { proxyHandler = fn; },
  env: { get(name) {
    if (name in envOverrides) return envOverrides[name];
    return {
      STYLIST_NODE_URL: endpoint,
      STYLIST_NODE_SHARED_SECRET: secret,
      STYLIST6_NODE_URL: s6Endpoint,
      STYLIST6_NODE_SHARED_SECRET: s6Secret,
      STYLIST_VERCEL_BYPASS_SECRET: bypass,
      SUPABASE_URL: 'https://supabase.example',
      SUPABASE_ANON_KEY: 'public-anon-key',
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

test('proxy supports server-controlled Stylist 6 rollout and cohort routing without trusting caller flags', async () => {
  const oldFetch = globalThis.fetch;
  const body = JSON.stringify({ stylePreferences: ['old_money'], occasion: 'work', engine: 'stylist6' });
  const visitedUrls = [];

  globalThis.fetch = async (url, init) => {
    const urlStr = String(url);
    visitedUrls.push(urlStr);
    if (urlStr === 'https://supabase.example/auth/v1/user') {
      const token = init.headers.authorization;
      if (token === 'Bearer cohort-user-token') {
        return Response.json({ id: '11111111-2222-3333-4444-555555555555' }, { status: 200 });
      }
      return Response.json({ id: '99999999-8888-7777-6666-555555555555' }, { status: 200 });
    }
    const expectedSecret = urlStr === s6Endpoint ? s6Secret : secret;
    assert.equal(await verifyStylistTransport({
      secret: expectedSecret,
      body,
      authorization: init.headers.authorization,
      timestamp: init.headers['x-aiw-timestamp'],
      nonce: init.headers['x-aiw-nonce'],
      signature: init.headers['x-aiw-signature'],
    }), true);
    return Response.json({
      success: true,
      generationVersion: urlStr === s6Endpoint ? 'stylist-6-cloud-v1' : 'stylist-5-eval-v6',
      outfits: [{ id: urlStr === s6Endpoint ? 's6-look' : 's5-look' }],
    }, { status: 200 });
  };

  try {
    // 1. Default rollout ('stylist5') ignores caller engine='stylist6' and routes to Stylist 5
    delete envOverrides.STYLIST_ENGINE_ROLLOUT;
    let res = await proxyHandler(new Request('https://supabase.example/functions/v1/generate-outfits', {
      method: 'POST', body,
      headers: { authorization: 'Bearer cohort-user-token', apikey: 'public-anon-key', 'content-type': 'application/json' },
    }));
    assert.equal((await res.json()).generationVersion, 'stylist-5-eval-v6');
    assert.equal(visitedUrls.at(-1), endpoint);

    // 2. Cohort mode routes verified allowlisted user to Stylist 6 and non-cohort user to Stylist 5
    envOverrides.STYLIST_ENGINE_ROLLOUT = 'cohort';
    envOverrides.STYLIST6_ROLLOUT_USER_IDS = '11111111-2222-3333-4444-555555555555';
    res = await proxyHandler(new Request('https://supabase.example/functions/v1/generate-outfits', {
      method: 'POST', body,
      headers: { authorization: 'Bearer cohort-user-token', apikey: 'public-anon-key', 'content-type': 'application/json' },
    }));
    assert.equal((await res.json()).generationVersion, 'stylist-6-cloud-v1');
    assert.equal(visitedUrls.at(-1), s6Endpoint);

    res = await proxyHandler(new Request('https://supabase.example/functions/v1/generate-outfits', {
      method: 'POST', body,
      headers: { authorization: 'Bearer non-cohort-user-token', apikey: 'public-anon-key', 'content-type': 'application/json' },
    }));
    assert.equal((await res.json()).generationVersion, 'stylist-5-eval-v6');
    assert.equal(visitedUrls.at(-1), endpoint);

    // 3. Full cutover ('stylist6') routes all requests to Stylist 6
    envOverrides.STYLIST_ENGINE_ROLLOUT = 'stylist6';
    res = await proxyHandler(new Request('https://supabase.example/functions/v1/generate-outfits', {
      method: 'POST', body,
      headers: { authorization: 'Bearer non-cohort-user-token', apikey: 'public-anon-key', 'content-type': 'application/json' },
    }));
    assert.equal((await res.json()).generationVersion, 'stylist-6-cloud-v1');
    assert.equal(visitedUrls.at(-1), s6Endpoint);
  } finally {
    delete envOverrides.STYLIST_ENGINE_ROLLOUT;
    delete envOverrides.STYLIST6_ROLLOUT_USER_IDS;
    globalThis.fetch = oldFetch;
  }
});
